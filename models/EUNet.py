import argparse
from distutils.util import strtobool
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum

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

        l = ceil(n / m)
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

        #  Key Change: No initial unsqueeze and transpose
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
        #self.self_attn = NystromAttention(
        #   dim=d_model,
        #    dim_head=(d_model // num_heads),
        #    heads=num_heads,
        #    num_landmarks=64,    # number of landmarks
        #    pinv_iterations=6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
        #    residual=False,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
        #    dropout=0.25, # dropout in the attention block
        #)
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

        #mask = torch.ones(size=(src_pe.size(0), src_pe.size(1)), device=src_pe.device).bool()
        
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
                 ppg_derivatives=False,
                 ppg_emd=False,
                 ppg_freqs=False,
                 fs=125,
                 input_seq_len_s=5,
                 channels='32, 64, 128, 256, 512',
                 kernel_size=3,
                 num_heads_attention=1,
                 dim_feedforward_attention=128,
                 set_tunable_params='all',
                 return_embedding=False):
        super(EUNet, self).__init__()
        
        # Input Data Setup
        self.input_seq_len = input_seq_len_s * fs
        self.ecg = ecg
        self.resp = resp
        self.sig2sig = sig2sig
        self.ppg_derivatives = ppg_derivatives
        self.ppg_emd = ppg_emd
        self.ppg_freqs = ppg_freqs
        self.return_embedding = return_embedding
        
        # Architecture Setup
        self.kernel_size = kernel_size
        self.num_heads_attention = num_heads_attention
        self.dim_feedforward_attention = dim_feedforward_attention
        
        ppg_in_channels, ecg_in_channels, resp_in_channels = self.get_input_channels()
        self.in_channels = ppg_in_channels + ecg_in_channels + resp_in_channels

        filters = [int(ch) for ch in channels.split(',')]

        # Encoder Path
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock1D(self.in_channels, filters[0], stride=1), # First block stride 1, no downsampling
            ResidualBlock1D(filters[0], filters[1], stride=2),
            ResidualBlock1D(filters[1], filters[2], stride=2),
            ResidualBlock1D(filters[2], filters[3], stride=2),
            ResidualBlock1D(filters[3], filters[4], stride=2)
        ])

        # Bottleneck
        self.bottleneck = BottleneckBlock1D(filters[4], stride=1) # Keep same resolution at bottleneck

        # Decoder Path
        self.decoder_blocks = nn.ModuleList([
            # Upsample filters[4] to filters[3], then combine with skip filters[3] to output filters[3]
            ResidualBlock1D(filters[4] + filters[3], filters[3], stride=1),
            ResidualBlock1D(filters[3] + filters[2], filters[2], stride=1),
            ResidualBlock1D(filters[2] + filters[1], filters[1], stride=1),
            ResidualBlock1D(filters[1] + filters[0], filters[0], stride=1)
        ])

        # Attention Gates
        # The attention gate needs to know the channels of the gating signal (from decoder)
        # and the channels of the skip connection (from encoder)
        self.att_gate4 = AttentionGate1D(filters[4], filters[3], filters[3]) # Gating: Bottleneck, Skip: Enc4
        self.att_gate3 = AttentionGate1D(filters[3], filters[2], filters[2]) # Gating: Dec1 output, Skip: Enc3
        self.att_gate2 = AttentionGate1D(filters[2], filters[1], filters[1]) # Gating: Dec2 output, Skip: Enc2
        self.att_gate1 = AttentionGate1D(filters[1], filters[0], filters[0]) # Gating: Dec3 output, Skip: Enc1

        # Final Self-Attention Module
        # This will operate on the output of the last decoder block before final convolution
        # d_model should be the channels of the last decoder block output (filters[0])
        self.self_attention_stack1 = SelfAttentionBlock1D(
            seq_len=self.input_seq_len,
            d_model=filters[0],
            num_heads=num_heads_attention,
            dim_feedforward=dim_feedforward_attention
        )
        self.self_attention_stack2 = SelfAttentionBlock1D(
            seq_len=self.input_seq_len,
            d_model=filters[0],
            num_heads=num_heads_attention,
            dim_feedforward=dim_feedforward_attention
        )

        # Output Layer
        # The final output is an ABP signal, which is a 1D signal of the same length as input.
        # So, it's a 1-channel output.
        self.final_conv = nn.Conv1d(filters[0], 1, kernel_size=1, stride=1, padding=0)
 
        # Kaiming initialization that should work fine with the z-score preprocessing
        self.init_params() 
        
        # Freeze/Tune model parameters
        #self.set_tunable_layers(set_tunable_params)

    def forward(self, x):
        
        x = x.unsqueeze(-1) if len(x.shape) == 2 else x # x -> (batch_size, seq_len, channels)
        
        # x shape: [batch_size, 625, num_modalities] (if multi-modal)
        # For Conv1d, input needs to be [batch_size, channels, length]
        # Our input is [batch_size, length, channels]
        x = x.permute(0, 2, 1) # -> [batch_size, num_modalities, 625]

        # Encoder
        e1 = self.encoder_blocks[0](x) # No downsampling, length 625, channels filters[0] (32)
        e2 = self.encoder_blocks[1](e1) # Downsample x2, length 313, channels filters[1] (64)
        e3 = self.encoder_blocks[2](e2) # Downsample x2, length 157, channels filters[2] (128)
        e4 = self.encoder_blocks[3](e3) # Downsample x2, length 79, channels filters[3] (256)
        e5 = self.encoder_blocks[4](e4) # Downsample x2, length 40, channels filters[4] (512)
        
        # Bottleneck
        b = self.bottleneck(e5) # Length 40, channels filters[4] (512)

        # Decoder with Attention Gates
        # Upsampling with ConvTranspose1d / F.interpolate + Conv
        # We need to calculate target_size for upsampling.
        # Decoder Block 1 (from b to d1)
        # Target size is e4.size(2)
        # g: b (filters[4]), x: e4 (filters[3])
        # Decoder Block 1
        g = F.interpolate(b, size=e4.size(2), mode='linear', align_corners=True) # g is the upsampled bottleneck.
        att_e4 = self.att_gate4(g, e4) # att_e4 is the attended encoder skip feature (e4 * psi)
        d1 = self.decoder_blocks[0](torch.cat([g, att_e4], dim=1)) # Concatenates upsampled bottleneck (g) with attended encoder skip (att_e4)
        
        # Decoder Block 2 (from d1 to d2)
        # Target size is e3.size(2)
        # g: d1 (filters[3]), x: e3 (filters[2])
        g = F.interpolate(d1, size=e3.size(2), mode='linear', align_corners=True)
        att_e3 = self.att_gate3(g, e3)
        d2 = self.decoder_blocks[1](torch.cat([g, att_e3], dim=1))

        # Decoder Block 3 (from d2 to d3)
        # Target size is e2.size(2)
        # g: d2 (filters[2]), x: e2 (filters[1])
        g = F.interpolate(d2, size=e2.size(2), mode='linear', align_corners=True)
        att_e2 = self.att_gate2(g, e2)
        d3 = self.decoder_blocks[2](torch.cat([g, att_e2], dim=1))

        # Decoder Block 4 (from d3 to d4)
        # Target size is e1.size(2) (initial length 625)
        # g: d3 (filters[1]), x: e1 (filters[0])
        g = F.interpolate(d3, size=e1.size(2), mode='linear', align_corners=True)
        att_e1 = self.att_gate1(g, e1)
        d4 = self.decoder_blocks[3](torch.cat([g, att_e1], dim=1)) # Output d4: filters[0] channels, original length

        # Final Self-Attention Module
        sa_output = self.self_attention_stack1(d4)
        sa_output = self.self_attention_stack2(sa_output)

        # Output Layer
        # For regression, often no final activation or a linear one implicitly.
        # The output shape should match the input shape's length for ABP signals.
        output = self.final_conv(sa_output)

        # Permute back to [batch_size, length, channels] for consistency with input
        output = output.permute(0, 2, 1)

        return output.squeeze(-1)
    
    def init_params(self):
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_input_channels(self):
        if self.ecg:
            ecg_in_channels = 1
        else:
            ecg_in_channels = 0
        
        if self.resp:
            resp_in_channels = 1
        else:
            resp_in_channels = 0
        
        if self.ppg_derivatives:
            ppg_in_channels = 3  # PPG + PPG' + PPG''
        elif self.ppg_emd:
            ppg_in_channels = 4  # PPG_IMF0 + PPG_IMF1 + PPG_IMF2 + PPG_IMF3
        elif self.ppg_freqs:
            ppg_in_channels = 16  # PPG_IMF0 + PPG_IMF1 + PPG_IMF2 + PPG_IMF3  
        else:
            ppg_in_channels = 1  # PPG
            
        return ppg_in_channels, ecg_in_channels, resp_in_channels
    
    def print_summary(self, batch_size=256):
        in_channels = self.get_input_channels()
        input = torch.rand((batch_size, self.input_seq_len, in_channels[0] + in_channels[1] + in_channels[2]))

        summary(self, input_data=[input], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))

        macs, num_params = profile(self, inputs=(input,))
        macs, num_params = clever_format([macs, num_params], "%.7f")
        print(f'UNet has {num_params} params and {macs} macs.')
    
    def set_tunable_layers(self, tune):
        
        # First set all parameters to be trainable
        for p in self.parameters():
            p.requires_grad = True
        
        # Update the whole model
        if tune == "all":
            return 
        # Update only the regressor last linear
        elif tune == "last_linear":
            for p in self.ppg_feature_gen.parameters():
                p.requires_grad = False
            
            if self.ecg:
                for p in self.ecg_feature_gen.parameters():
                    p.requires_grad = False
            
                if self.resp: 
                    for p in self.resp_feature_gen.parameters():
                        p.requires_grad = False
            
            for p in self.t_gru.parameters():
                p.requires_grad = False
                
            for p in self.projection_head.parameters():
                p.requires_grad = False
        else:
            raise Exception("undefined tune")
        
        
def parseargs():
    parser = argparse.ArgumentParser(description="PhysioFormer summary, # params and MACS")

    parser.add_argument('--batch_size', default=128, type=int, help='batch size for the dummy input')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load only ecg or not')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to load also resp with ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--ppg_derivatives', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg derivatives or not')
    parser.add_argument('--ppg_emd', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg imfs or not')
    parser.add_argument('--ppg_freqs', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg freqs or not')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
    parser.add_argument('--channels', default='32, 64, 128, 256, 512', type=str, help='channels produced by the convolutional blocks')
    parser.add_argument('--num_heads_attention', default=1, type=int, help='heads number of the final self-attention layer') 
    parser.add_argument('--dim_feedforward_attention', default=128, type=int, help='dimension of the final self-attention layer') 
    parser.add_argument('--kernel_size', default=3, type=int, help='convolutional layer kernel size')
    parser.add_argument('--set_tunable_params', default='all', type=str, help='which model parameters to tune (all, only regressor, only encoder, etc.)')
    parser.add_argument('--return_embedding', default='False', type=lambda x: bool(strtobool(x)), help='whether to return the model embedding before the regressor or not')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    global args
    args = parseargs()    
    
    net = EUNet(
        ecg=args.ecg,
        resp=args.resp,
        sig2sig=args.sig2sig,
        ppg_derivatives=args.ppg_derivatives,
        ppg_emd=args.ppg_emd,
        ppg_freqs=args.ppg_freqs,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        channels=args.channels,
        kernel_size=args.kernel_size,
        num_heads_attention=args.num_heads_attention,
        dim_feedforward_attention=args.dim_feedforward_attention,
        set_tunable_params=args.set_tunable_params,
        return_embedding=args.return_embedding
        )        
    net.print_summary(batch_size=args.batch_size)