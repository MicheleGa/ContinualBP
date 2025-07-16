import argparse
from distutils.util import strtobool
import itertools
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
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_groups=8):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        
        groups_gn1 = num_groups if out_channels >= num_groups else 1
        self.gn1 = nn.GroupNorm(groups_gn1, out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1) # Stride 1 for second conv

        groups_shortcut = num_groups if out_channels >= num_groups else 1
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.GroupNorm(groups_shortcut, out_channels)
        )
        
    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out += identity
        
        return out
    
class BottleneckBlock1D(nn.Module):
    def __init__(self, channels, stride=1, num_groups=8):
        super(BottleneckBlock1D, self).__init__()
        
        groups_gn1 = num_groups if channels >= num_groups else 1
        self.gn1 = nn.GroupNorm(groups_gn1, channels)
        
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, stride=1)

        groups_gn2 = num_groups if channels >= num_groups else 1
        self.gn2 = nn.GroupNorm(groups_gn2, channels)
        
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, stride=1)
        
    def forward(self, x):
        
        out = self.gn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.gn2(out)
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
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)
        x = x + pe
        return self.dropout(x)
    

class SelfAttentionBlock1D(nn.Module):
    def __init__(self, seq_len, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(SelfAttentionBlock1D, self).__init__()
        self.pos_encoder = PositionalEncoding(embed_dim=d_model, max_len=seq_len)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
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
        src = src.permute(0, 2, 1)

        # Add positional encoding
        src_pe = self.pos_encoder(src)

        attn_output, _ = self.self_attn(src_pe, src_pe, src_pe)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        src = src.permute(0, 2, 1) # Permute back to (batch_size, channels, length)
        return src
    

class SSLUNet(nn.Module):
    def __init__(self,
                 ecg=False,
                 resp=False,
                 sig2sig=False,
                 fs=125,
                 input_seq_len_s=5,
                 channels='32, 64, 128, 256, 512',
                 kernel_size=3,
                 num_heads_attention=1,
                 dim_feedforward_attention=128,
                 num_groups_gn=8 # Added num_groups_gn argument for GroupNorm
                ):
        super(SSLUNet, self).__init__()
        
        # Input Data Setup
        self.input_seq_len = input_seq_len_s * fs
        self.ecg = ecg
        self.resp = resp
        self.sig2sig = sig2sig
        
        # Architecture Setup
        self.kernel_size = kernel_size
        self.num_heads_attention = num_heads_attention
        self.dim_feedforward_attention = dim_feedforward_attention
        self.num_groups_gn = num_groups_gn # Store the number of groups for GroupNorm
        
        # For supervised tasks, use the full channel calculation
        ppg_in_channels, ecg_in_channels, resp_in_channels = self.get_input_channels()
        self.in_channels = ppg_in_channels + ecg_in_channels + resp_in_channels
            
        filters = [int(ch) for ch in channels.split(',')]
        self.encoder_filters = filters # Store for use in other heads
        
        # --- Shared Encoder Path ---
        self.encoder_blocks = nn.ModuleList([
            ResidualBlock1D(self.in_channels, filters[0], stride=1, num_groups=self.num_groups_gn),
            ResidualBlock1D(filters[0], filters[1], stride=2, num_groups=self.num_groups_gn),
            ResidualBlock1D(filters[1], filters[2], stride=2, num_groups=self.num_groups_gn),
            ResidualBlock1D(filters[2], filters[3], stride=2, num_groups=self.num_groups_gn),
            ResidualBlock1D(filters[3], filters[4], stride=2, num_groups=self.num_groups_gn)
        ])
        self.bottleneck = BottleneckBlock1D(filters[4], stride=1, num_groups=self.num_groups_gn)

        # --- Shared Decoder Path (U-Net style, leading to `sa_output`) ---
        self.decoder_blocks = nn.ModuleList([
            ResidualBlock1D(filters[4] + filters[3], filters[3], stride=1, num_groups=self.num_groups_gn),
            ResidualBlock1D(filters[3] + filters[2], filters[2], stride=1, num_groups=self.num_groups_gn),
            ResidualBlock1D(filters[2] + filters[1], filters[1], stride=1, num_groups=self.num_groups_gn),
            ResidualBlock1D(filters[1] + filters[0], filters[0], stride=1, num_groups=self.num_groups_gn)
        ])

        self.att_gate4 = AttentionGate1D(filters[4], filters[3], filters[3])
        self.att_gate3 = AttentionGate1D(filters[3], filters[2], filters[2])
        self.att_gate2 = AttentionGate1D(filters[2], filters[1], filters[1])
        self.att_gate1 = AttentionGate1D(filters[1], filters[0], filters[0])

        self.self_attention_stack1 = SelfAttentionBlock1D(
            seq_len=self.input_seq_len,
            d_model=filters[0], # d_model is channels of output from last decoder block
            num_heads=num_heads_attention,
            dim_feedforward=dim_feedforward_attention
        )
        self.self_attention_stack2 = SelfAttentionBlock1D(
            seq_len=self.input_seq_len,
            d_model=filters[0], # d_model is channels of output from last decoder block
            num_heads=num_heads_attention,
            dim_feedforward=dim_feedforward_attention
        )
        
        # Output embedding dimension for the SSL heads (output of last SA block)
        # This is [Batch, filters[0], self.input_seq_len]
        # For projection/permutation heads, we need to flatten this to a fixed-size vector
        self.ssl_embedding_dim = filters[0] * self.input_seq_len # e.g., 32 * 625 = 20000

        # --- NEW SSL HEADS (attached to `sa_output`) ---
        # 1.  Reconstruction Head (for MSR)
        # This head takes the `sa_output` (which is already [Batch, Filters[0], Length])
        # and directly maps it to [Batch, 2, Length] for PPG/ECG reconstruction.
        self.reconstruction_head = nn.Conv1d(filters[0], 2, kernel_size=1) # Output 2 channels (PPG, ECG)

        # --- Final Output Layer (for supervised training's direct output) ---
        # This is the original final_conv, which maps to 1 channel (ABP)
        self.final_conv_supervised = nn.Conv1d(filters[0], 1, kernel_size=1, stride=1, padding=0)
 
        # Kaiming initialization that should work fine with the z-score preprocessing
        self.init_params() 
        
    def _run_full_unet_path(self, x):
        """
        Helper to run the full U-Net encoder-decoder-SA path.
        This will be the shared feature extractor for all tasks.
        Args:
            x (torch.Tensor): Input signal [Batch, Length, N_Channels]
        Returns:
            torch.Tensor: Output of the last self-attention block [Batch, filters[0], Length]
            list: Skip connections from encoder (e1, e2, e3, e4)
        """
        # Ensure input is 3D for processing
        x_conv = x.unsqueeze(-1) if len(x.shape) == 2 else x # [Batch, Length, N_channels]
        x_conv = x_conv.permute(0, 2, 1) # -> [batch_size, N_channels, Length]

        # Encoder
        e1 = self.encoder_blocks[0](x_conv)
        e2 = self.encoder_blocks[1](e1)
        e3 = self.encoder_blocks[2](e2)
        e4 = self.encoder_blocks[3](e3)
        e5 = self.encoder_blocks[4](e4)
        
        # Bottleneck
        b = self.bottleneck(e5)

        # Decoder with Attention Gates
        g = F.interpolate(b, size=e4.size(2), mode='linear', align_corners=True)
        att_e4 = self.att_gate4(g, e4)
        d1 = self.decoder_blocks[0](torch.cat([g, att_e4], dim=1))
        
        g = F.interpolate(d1, size=e3.size(2), mode='linear', align_corners=True)
        att_e3 = self.att_gate3(g, e3)
        d2 = self.decoder_blocks[1](torch.cat([g, att_e3], dim=1))

        g = F.interpolate(d2, size=e2.size(2), mode='linear', align_corners=True)
        att_e2 = self.att_gate2(g, e2)
        d3 = self.decoder_blocks[2](torch.cat([g, att_e2], dim=1))

        g = F.interpolate(d3, size=e1.size(2), mode='linear', align_corners=True)
        att_e1 = self.att_gate1(g, e1)
        d4 = self.decoder_blocks[3](torch.cat([g, att_e1], dim=1))

        # Final Self-Attention Module
        sa_output = self.self_attention_stack1(d4)
        sa_output = self.self_attention_stack2(sa_output)
        
        # TODO: can return other features from deeper layers to increase performances
        
        return sa_output, [e1, e2, e3, e4, e5, b] # sa_output: [Batch, filters[0], Length], also return encoder intermediates if needed by other parts of the forward (unlikely in this setup)

    def encode(self, signal):
        """
        Shared feature extractor for SSL tasks.
        It runs the full U-Net encoder-decoder-SA path.
        Args:
            signal (torch.Tensor): Input signal [Batch, Length, 2] (PPG, ECG)
        Returns:
            torch.Tensor: Feature map from last SA block [Batch, filters[0], Length]
        """
        # Ensure the input to encode is always [Batch, Length, 2]
        if signal.shape[-1] != 2:
            raise ValueError(f"Expected signal with 2 modalities for encode, got {signal.shape[-1]}.")
        
        sa_output, _ = self._run_full_unet_path(signal)
        return sa_output

    def reconstruct(self, sa_output_embedding):
        """
        Reconstruction head for MSR.
        Args:
            sa_output_embedding (torch.Tensor): Output from last SA block [Batch, filters[0], Length]
        Returns:
            torch.Tensor: Reconstructed signal [Batch, Length, 2] (PPG, ECG)
        """
        reconstructed_signal_conv = self.reconstruction_head(sa_output_embedding)
        return reconstructed_signal_conv.permute(0, 2, 1) # Permute back to [Batch, Length, Modalities]

    def forward(self, x):
        """
        Main forward pass for supervised training.
        This runs the full U-Net path and then the final supervised convolution.
        Args:
            x (torch.Tensor): Input signal [Batch, Length, N_Channels]. N_Channels depends on args.
        Returns:
            torch.Tensor: Predicted output for supervised task [Batch, Length] (for sig2sig) or [Batch, 1] (for SBP/DBP if regressor is changed).
        """
        
        # Ensure input is 3D for processing
        x = x.unsqueeze(-1) if len(x.shape) == 2 else x # x -> (batch_size, seq_len, N_channels)
        
        # Run the full U-Net path to get the final SA output feature map
        sa_output, _ = self._run_full_unet_path(x) # sa_output: [Batch, filters[0], Length]

        # Output Layer for supervised task
        output = self.final_conv_supervised(sa_output)

        # Permute back to [batch_size, length, 1] then squeeze to [batch_size, length]
        return output.permute(0, 2, 1).squeeze(-1)
    
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm): # Changed from BatchNorm1d to GroupNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_input_channels(self):
        
        ppg_in_channels = 0
        ecg_in_channels = 0
        resp_in_channels = 0

        # PPG must always be present
        ppg_in_channels = 1

        # Determine ECG and RESP channels
        if self.ecg:
            ecg_in_channels = 1
        if self.resp:
            resp_in_channels = 1
            
        return ppg_in_channels, ecg_in_channels, resp_in_channels
    
    def print_summary(self, batch_size=256):
        # The input to summary should match how `forward` is called.
        
        ppg_in_channels, ecg_in_channels, resp_in_channels = self.get_input_channels()
        total_input_channels = ppg_in_channels + ecg_in_channels + resp_in_channels
        
        print("\n--- Model Summary (Full Forward/Backward) ---")
        dummy_input = torch.rand((batch_size, self.input_seq_len, total_input_channels))
        summary(self, input_data=dummy_input, col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))

        # MACs calculation (for the full forward pass for consistency)
        print("\n--- MACs and Parameters (Full Forward/Backward) ---")
        # Use the input_main_channels for MACs calculation as it represents the typical input to the first encoder block
        macs, num_params = profile(self, inputs=(dummy_input,), verbose=False)
        macs, num_params = clever_format([macs, num_params], "%.7f")
        print(f'SSLUNet has {num_params} params and {macs} MACs.')

        print("\n--- SSL Head Summaries (Attached to SA output) ---")
        
        # Summarize the entire model, assuming it runs the full UNet path for `encode`
        print("Summarizing full U-Net for SSL `encode` path:")
        sa_output_dummy, _ = self._run_full_unet_path(dummy_input)
        print(f"  Encoder/U-Net output (SA block) shape: {sa_output_dummy.shape}")

        print("\n--- Reconstruction Head Summary (MSR) ---")
        dummy_sa_output_recon = torch.rand((batch_size, self.encoder_filters[0], self.input_seq_len))
        summary(self.reconstruction_head, input_data=dummy_sa_output_recon)

    
    def set_tunable_layers(self, tune):
        # First set all parameters to be trainable
        for p in self.parameters():
            p.requires_grad = True
        
        if tune == "all":
            return
        elif tune == "last_linear":
            # Freeze all layers except the supervised final_conv_supervised
            for name, param in self.named_parameters():
                if "final_conv_supervised" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif tune == "encoder_only":
            # Example: Freeze all heads, train only encoder blocks and bottleneck
            for name, param in self.named_parameters():
                if any(head_name in name for head_name in ["reconstruction_head", "final_conv_supervised", "decoder_blocks", "att_gate", "self_attention_stack"]):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            raise ValueError(f"Undefined tune option: {tune}")
        
        
def parseargs():
    parser = argparse.ArgumentParser(description="SSLUNet summary, # params and MACS")

    parser.add_argument('--batch_size', default=128, type=int, help='batch size for the dummy input')
    parser.add_argument('--ecg', default='True', type=lambda x: bool(strtobool(x)), help='whether to load ecg or not. True for SSL')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to load also resp with ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
    parser.add_argument('--channels', default='32, 64, 128, 256, 512', type=str, help='channels produced by the convolutional blocks')
    parser.add_argument('--num_heads_attention', default=1, type=int, help='heads number of the final self-attention layer')
    parser.add_argument('--dim_feedforward_attention', default=128, type=int, help='dimension of the final self-attention layer')
    parser.add_argument('--kernel_size', default=3, type=int, help='convolutional layer kernel size')
    parser.add_argument('--num_groups_gn', default=8, type=int, help='number of groups for GroupNorm layers') # Added num_groups_gn argument
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()

    net = SSLUNet(
        ecg=args.ecg,
        resp=args.resp,
        sig2sig=args.sig2sig,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        channels=args.channels,
        kernel_size=args.kernel_size,
        num_heads_attention=args.num_heads_attention,
        dim_feedforward_attention=args.dim_feedforward_attention,
        num_groups_gn=args.num_groups_gn # Pass the new argument
    )
    net.print_summary(batch_size=args.batch_size)