import argparse
from distutils.util import strtobool
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, reduce
from torchinfo import summary
from thop import profile, clever_format


def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(p=dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = math.ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out = out + self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out

class AttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads, head_dim, num_landmarks=64):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            head_dim - Dimensionality of each attention head
            num_landmarks - Number of landmarks for Nystrom Attention
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = NystromAttention(
            dim = embed_dim,
            dim_head = head_dim,
            heads = num_heads,
            num_landmarks = num_landmarks,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = False,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.25, # dropout in the attention block
        )
        self.shortcut1 = nn.Identity()

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(p=0.25)
        )
        self.shortcut2 = nn.Identity()
         
    def forward(self, x):
        # x -> (batch_size, seq_len, channels)
        
        mask = torch.ones(size=(x.size(0), x.size(1)), device=x.device).bool()
        
        # Pre-Layer Normalization
        dx = self.ln1(x)

        # Attention Block
        attn_output = self.attn(dx, mask=mask) 
        shortcut1 = self.shortcut1(x)
        x = shortcut1 + attn_output

        # Pre-Layer Normalization
        dx = self.ln2(x)

        # MLP Block
        shortcut2 = self.shortcut2(x)
        x = shortcut2 + self.ffn(dx) 

        return x


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1) # Stride 1 for second conv

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels)
        )
        
    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += identity 
        
        return out
    
class BottleneckBlock1D(nn.Module):
    def __init__(self, channels, stride=1):
        super(BottleneckBlock1D, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, stride=1)

        self.bn2 = nn.BatchNorm1d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, stride=1)
        
    def forward(self, x):
        
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        return out

class AttentionGate1D(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super(AttentionGate1D, self).__init__()
        # Linear layers to compute queries, keys, and values
        self.query_conv = nn.Conv1d(g_channels, inter_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(x_channels, inter_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(x_channels, x_channels, kernel_size=1)

        # Output convolution to refine the attended feature map
        self.output_conv = nn.Conv1d(x_channels, x_channels, kernel_size=1)

        # Activation function
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from the decoder path (batch_size, g_channels, length)
            x: Skip connection from the encoder path (batch_size, x_channels, length)
        Returns:
            Attended feature map (batch_size, x_channels, length)
        """
        # Compute queries, keys, and values
        query = self.query_conv(g)  # Shape: (batch_size, inter_channels, length)
        key = self.key_conv(x)      # Shape: (batch_size, inter_channels, length)
        value = self.value_conv(x)  # Shape: (batch_size, x_channels, length)

        # Compute attention weights (scaled dot-product attention)
        attention = torch.bmm(query.permute(0, 2, 1), key)  # Shape: (batch_size, length, length)
        attention = attention / (query.size(1) ** 0.5)      # Scale by sqrt(inter_channels)
        attention = self.softmax(attention)                # Normalize along the last dimension

        # Apply attention weights to the value
        attended_features = torch.bmm(value, attention.permute(0, 2, 1))  # Shape: (batch_size, x_channels, length)

        # Refine the attended feature map
        output = self.output_conv(attended_features)  # Shape: (batch_size, x_channels, length)

        return output
    

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim=768, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor([10000.0])) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x -> (batch_size, seq_len, channels)

        seq_len = x.size(1)  # Get sequence length from input

        pe = self.pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)  # Expand for batch

        x = x + pe

        return self.dropout(x)
    

class SelfAttentionBlock1D(nn.Module):
    def __init__(self, seq_len, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(SelfAttentionBlock1D, self).__init__()
        self.pos_encoder = PositionalEncoding(embed_dim=d_model, max_len=seq_len) # Positional encoding
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True) # Set batch_first=True
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src):
        # src shape: (batch_size, channels, length) -> transform to (batch_size, length, channels) for MultiheadAttention
        src = src.permute(0, 2, 1) # (batch_size, length, channels)

        # Add positional encoding
        src_pe = self.pos_encoder(src) # Add PE to original shape then permute back
        
        # Self-attention
        attn_output, _ = self.self_attn(src_pe, src_pe, src_pe)
        src = src + self.dropout1(attn_output) # Add & Norm
        src = self.norm1(src)

        # Feedforward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output) # Add & Norm
        src = self.norm2(src)

        src = src.permute(0, 2, 1) # (batch_size, channels, length)
        return src
    

class EUNet(nn.Module):
    def __init__(self,  
                 ecg=False,
                 resp=False,
                 sig2sig=False,
                 fs=125,
                 input_seq_len_s=5,
                 channels='16, 32, 64, 128',
                 kernel_size=3,
                 num_heads_attention=1,
                 dim_feedforward_attention=128,
                 attention_type='self_attention'):
        super(EUNet, self).__init__()
        
        # Input Data Setup
        self.input_seq_len = input_seq_len_s * fs
        self.ecg = ecg
        self.resp = resp
        self.sig2sig = sig2sig
        
        # Architecture Setup
        self.kernel_size = kernel_size
        self.num_heads_attention = num_heads_attention
        self.dim_feedforward_attention = dim_feedforward_attention
        self.attention_type = attention_type # Store the choice

        # Encoder filter configurations
        self.encoder_filters = [int(ch) for ch in channels.split(',')]
        
        # Get input channels for each modality
        self.ppg_in_channels, self.ecg_in_channels, self.resp_in_channels = self.get_input_channels()
        self.in_channels = self.ppg_in_channels + self.ecg_in_channels + self.resp_in_channels
        
        # --- Modality-specific Encoders ---
        # PPG Encoder
        self.ppg_encoder_blocks = nn.ModuleList()
        if self.ppg_in_channels > 0:
            self.ppg_encoder_blocks.append(ResidualBlock1D(self.ppg_in_channels, self.encoder_filters[0], stride=1))
            for i in range(len(self.encoder_filters) - 1):
                self.ppg_encoder_blocks.append(ResidualBlock1D(self.encoder_filters[i], self.encoder_filters[i+1], stride=2))

        # ECG Encoder
        self.ecg_encoder_blocks = nn.ModuleList()
        if self.ecg_in_channels > 0:
            self.ecg_encoder_blocks.append(ResidualBlock1D(self.ecg_in_channels, self.encoder_filters[0], stride=1))
            for i in range(len(self.encoder_filters) - 1):
                self.ecg_encoder_blocks.append(ResidualBlock1D(self.encoder_filters[i], self.encoder_filters[i+1], stride=2))

        # RESP Encoder
        self.resp_encoder_blocks = nn.ModuleList()
        if self.resp_in_channels > 0:
            self.resp_encoder_blocks.append(ResidualBlock1D(self.resp_in_channels, self.encoder_filters[0], stride=1))
            for i in range(len(self.encoder_filters) - 1):
                self.resp_encoder_blocks.append(ResidualBlock1D(self.encoder_filters[i], self.encoder_filters[i+1], stride=2))
        
        # Determine the total channels at the bottleneck level for the shared bottleneck
        self.bottleneck_in_channels = 0
        if self.ppg_in_channels > 0:
            self.bottleneck_in_channels += self.encoder_filters[-1]
        if self.ecg_in_channels > 0:
            self.bottleneck_in_channels += self.encoder_filters[-1]
        if self.resp_in_channels > 0:
            self.bottleneck_in_channels += self.encoder_filters[-1]

        self.bottleneck = BottleneckBlock1D(self.bottleneck_in_channels, stride=1)

        # --- Shared Decoder Path ---
        self.decoder_blocks = nn.ModuleList()
        self.att_gates = nn.ModuleList()

        decoder_input_base_channels = self.bottleneck_in_channels
        
        for i in range(len(self.encoder_filters)):
            current_encoder_level = len(self.encoder_filters) - 1 - i 
            
            skip_channels = 0
            if self.ppg_in_channels > 0:
                skip_channels += self.encoder_filters[current_encoder_level]
            if self.ecg_in_channels > 0:
                skip_channels += self.encoder_filters[current_encoder_level]
            if self.resp_in_channels > 0:
                skip_channels += self.encoder_filters[current_encoder_level]
            
            g_channels = decoder_input_base_channels if i == 0 else self.encoder_filters[current_encoder_level + 1] 
            
            att_inter_channels = self.encoder_filters[current_encoder_level]
            
            self.att_gates.append(AttentionGate1D(g_channels, skip_channels, att_inter_channels))
            
            self.decoder_blocks.append(ResidualBlock1D(g_channels + skip_channels, self.encoder_filters[current_encoder_level], stride=1))
            
            decoder_input_base_channels = self.encoder_filters[current_encoder_level]

        # Conditional final processing layer instantiation
        if self.attention_type == 'gru':
            self.gru_layer = nn.GRU(
                input_size=self.encoder_filters[0], 
                hidden_size=self.encoder_filters[0], # Hidden dim matches embedding dim
                num_layers=2, 
                batch_first=True, 
                bidirectional=True
            )
            # If bidirectional, output will be 2 * hidden_size. Need to project back.
            self.gru_proj = nn.Linear(self.encoder_filters[0] * 2, self.encoder_filters[0])
        elif self.attention_type == 'nystrom_attention':
            self.nystrom_attention_block = AttentionBlock(
                embed_dim=self.encoder_filters[0], 
                hidden_dim=dim_feedforward_attention, 
                num_heads=num_heads_attention,
                # Ensure dim_head * heads = embed_dim. If num_heads_attention is 0, handle to prevent division by zero.
                head_dim=self.encoder_filters[0] // num_heads_attention if num_heads_attention > 0 else self.encoder_filters[0], 
                num_landmarks=min(64, self.input_seq_len) # Adjusted to be at most sequence length
            )
        elif self.attention_type == 'self_attention':
            self.self_attention_stack1 = SelfAttentionBlock1D(
                seq_len=self.input_seq_len,
                d_model=self.encoder_filters[0],
                num_heads=num_heads_attention,
                dim_feedforward=dim_feedforward_attention
            )
            self.self_attention_stack2 = SelfAttentionBlock1D(
                seq_len=self.input_seq_len,
                d_model=self.encoder_filters[0],
                num_heads=num_heads_attention,
                dim_feedforward=dim_feedforward_attention
            )
        else:
            raise ValueError(f"Unknown attention_type: {self.attention_type}. Choose from 'self_attention', 'nystrom_attention', 'gru'.")

        # Output Layer
        self.final_conv = nn.Conv1d(self.encoder_filters[0], 1, kernel_size=1, stride=1, padding=0)
 
        # Kaiming initialization
        self.init_params() 

    def forward(self, x):
        
        x = x.unsqueeze(-1) if len(x.shape) == 2 else x 
        x = x.permute(0, 2, 1) 

        # Separate inputs for each modality
        current_channel_idx = 0
        ppg_input = None
        ecg_input = None
        resp_input = None

        if self.ppg_in_channels > 0:
            ppg_input = x[:, current_channel_idx : current_channel_idx + self.ppg_in_channels, :]
            current_channel_idx += self.ppg_in_channels
        if self.ecg_in_channels > 0:
            ecg_input = x[:, current_channel_idx : current_channel_idx + self.ecg_in_channels, :]
            current_channel_idx += self.ecg_in_channels
        if self.resp_in_channels > 0:
            resp_input = x[:, current_channel_idx : current_channel_idx + self.resp_in_channels, :]
            current_channel_idx += self.resp_in_channels

        # --- Modality-specific Encoders ---
        ppg_enc_features = []
        if ppg_input is not None:
            e = ppg_input
            for block in self.ppg_encoder_blocks:
                e = block(e)
                ppg_enc_features.append(e) 
        
        ecg_enc_features = []
        if ecg_input is not None:
            e = ecg_input
            for block in self.ecg_encoder_blocks:
                e = block(e)
                ecg_enc_features.append(e) 

        resp_enc_features = []
        if resp_input is not None:
            e = resp_input
            for block in self.resp_encoder_blocks:
                e = block(e)
                resp_enc_features.append(e) 

        # Concatenate features at the bottleneck level
        bottleneck_inputs = []
        if ppg_enc_features:
            bottleneck_inputs.append(ppg_enc_features[-1])
        if ecg_enc_features:
            bottleneck_inputs.append(ecg_enc_features[-1])
        if resp_enc_features:
            bottleneck_inputs.append(resp_enc_features[-1])

        b = self.bottleneck(torch.cat(bottleneck_inputs, dim=1))

        # --- Shared Decoder Path with Attention Gates ---
        d = b 
        
        for i in range(len(self.encoder_filters)):
            current_encoder_level = len(self.encoder_filters) - 1 - i 
            
            level_skip_features = []
            if ppg_enc_features:
                level_skip_features.append(ppg_enc_features[current_encoder_level])
            if ecg_enc_features:
                level_skip_features.append(ecg_enc_features[current_encoder_level])
            if resp_enc_features:
                level_skip_features.append(resp_enc_features[current_encoder_level])
            
            concatenated_skip = torch.cat(level_skip_features, dim=1) if level_skip_features else None

            d_upsampled = F.interpolate(d, size=concatenated_skip.size(2), mode='linear', align_corners=True) if concatenated_skip is not None else d
            
            att_skip = self.att_gates[i](d_upsampled, concatenated_skip) if concatenated_skip is not None else None
            
            decoder_input_for_block = torch.cat([d_upsampled, att_skip], dim=1) if att_skip is not None else d_upsampled
            
            d = self.decoder_blocks[i](decoder_input_for_block)

        # Final processing layer based on attention_type
        # The output of 'd' is (batch, channels, length)
        sa_output = d # Initialize with decoder output

        if self.attention_type == 'gru':
            # GRU expects input as (batch, seq_len, features)
            # Current d is (batch, features, seq_len)
            gru_input = d.permute(0, 2, 1) # (batch, length, channels)
            gru_output, _ = self.gru_layer(gru_input)
            # If bidirectional, output is 2 * hidden_size, project back
            gru_output = self.gru_proj(gru_output)
            sa_output = gru_output.permute(0, 2, 1) # Back to (batch, channels, length)
        elif self.attention_type == 'nystrom_attention':
            # Nystrom Attention expects (batch, seq_len, dim)
            nystrom_input = d.permute(0, 2, 1) # (batch, length, channels)
            nystrom_output = self.nystrom_attention_block(nystrom_input)
            sa_output = nystrom_output.permute(0, 2, 1) # Back to (batch, channels, length)
        elif self.attention_type == 'self_attention':
            # SelfAttentionBlock1D already handles internal permutations
            sa_output = self.self_attention_stack1(d)
            sa_output = self.self_attention_stack2(sa_output)

        # Output Layer
        output = self.final_conv(sa_output)

        output = output.permute(0, 2, 1)

        return output.squeeze(-1)
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def get_input_channels(self):
        
        ppg_in_channels = 0
        ecg_in_channels = 0
        resp_in_channels = 0

        # PPG must always  be present
        ppg_in_channels = 1

        if self.ecg:
            ecg_in_channels = 1
        if self.resp:
            resp_in_channels = 1
            
        return ppg_in_channels, ecg_in_channels, resp_in_channels
    
    def print_summary(self, batch_size=256):
        ppg_in_channels, ecg_in_channels, resp_in_channels = self.get_input_channels()
        total_input_channels = ppg_in_channels + ecg_in_channels + resp_in_channels
        input_data = torch.rand((batch_size, self.input_seq_len, total_input_channels))

        print("\n--- Model Summary (Full Forward/Backward) ---")
        summary(self, input_data=[input_data], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))

        print("\n--- MACs and Parameters (Full Forward/Backward) ---")
        macs, num_params = profile(self, inputs=(input_data,))
        macs, num_params = clever_format([macs, num_params], "%.7f")
        print(f'EUNet has {num_params} params and {macs} macs.')
        
        
def parseargs():
    parser = argparse.ArgumentParser(description="PhysioFormer summary, # params and MACS")

    parser.add_argument('--batch_size', default=128, type=int, help='batch size for the dummy input')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load only ecg or not')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to load also resp with ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
    parser.add_argument('--channels', default='16, 32, 64, 128', type=str, help='channels for the individual encoders') 
    parser.add_argument('--num_heads_attention', default=1, type=int, help='heads number of the final self-attention layer') 
    parser.add_argument('--dim_feedforward_attention', default=128, type=int, help='dimension of the final self-attention layer') 
    parser.add_argument('--kernel_size', default=3, type=int, help='convolutional layer kernel size')
    parser.add_argument('--attention_type', default='self_attention', type=str, 
                        choices=['self_attention', 'nystrom_attention', 'gru'], 
                        help='Type of final processing layer: "self_attention", "nystrom_attention", or "gru"')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    global args
    args = parseargs()    

    net = EUNet(
        ecg=args.ecg,
        resp=args.resp,
        sig2sig=args.sig2sig,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        channels=args.channels, 
        kernel_size=args.kernel_size,
        num_heads_attention=args.num_heads_attention,
        dim_feedforward_attention=args.dim_feedforward_attention,
        attention_type=args.attention_type 
    )        
    net.print_summary(batch_size=args.batch_size)