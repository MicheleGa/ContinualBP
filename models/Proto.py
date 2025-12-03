import argparse
import torch
from torch import nn
from torchinfo import summary
from thop import profile, clever_format
from component_factory import BPRegressor

"""
Simplified Proto encoder using:
- Dilated ConvNeXt-style 1D blocks
- Clean channel progression: 1 → 16 → 32 → 64 → 128
- No conditional projection logic
- Transformer encoder on top
"""

# ------------------------------------------------------
# ConvNeXt-like 1D Block with dilation (simplified)
# ------------------------------------------------------
class ConvNeXtBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1, expansion=4, downsample=True):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation

        # Depthwise conv
        self.dwconv = nn.Conv1d(
            in_ch, in_ch, kernel_size=kernel_size,
            padding=padding, groups=in_ch, dilation=dilation
        )

        # Residual channel mapping (always needed since channels always change)
        self.res_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

        # Normalization + MLP operate on (B, L, C)
        self.norm = nn.LayerNorm(in_ch)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, expansion * in_ch),
            nn.GELU(),
            nn.Linear(expansion * in_ch, in_ch),
        )

        # Final 1×1 to match output channels
        self.pw_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

        self.down = nn.AvgPool1d(kernel_size=2, stride=2) if downsample else nn.Identity()

    def forward(self, x):
        # x: (B, C, L)
        res = self.res_conv(x)

        y = self.dwconv(x)
        y = y.permute(0, 2, 1)      # (B, L, C)
        y = self.norm(y)
        y = self.mlp(y)
        y = y.permute(0, 2, 1)      # (B, C, L)
        y = self.pw_conv(y)

        y = y + res
        y = self.down(y)
        return y


# ------------------------------------------------------
# Temporal Stack (channel progression fixed)
# ------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.b0 = ConvNeXtBlock1D(1, 32, kernel_size=7, dilation=1)
        self.b1 = ConvNeXtBlock1D(32, 64, kernel_size=7, dilation=2)
        self.b2 = ConvNeXtBlock1D(64, 128, kernel_size=7, dilation=4)
        self.b3 = ConvNeXtBlock1D(128, embed_dim, kernel_size=7, dilation=8)

    def forward(self, x):
        # Accepts (B,T) or (B,1,T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x[:, :1, :]

        y = self.b0(x)
        y = self.b1(y)
        y = self.b2(y)
        y = self.b3(y)

        # Output: (B,128,L) → (B,L,128)
        return y.permute(0, 2, 1)


# ------------------------------------------------------
# Transformer Encoder
# ------------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, ff_mult=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout),
                'norm1': nn.LayerNorm(embed_dim),
                'ff': nn.Sequential(
                    nn.Linear(embed_dim, ff_mult * embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_mult * embed_dim, embed_dim),
                ),
                'norm2': nn.LayerNorm(embed_dim)
            })
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for block in self.layers:
            attn_out, _ = block['attn'](x, x, x)
            x = block['norm1'](x + attn_out)
            ff_out = block['ff'](x)
            x = block['norm2'](x + ff_out)
        return x


# ------------------------------------------------------
# Full Feature Generator (ConvNeXt → Transformer)
# ------------------------------------------------------
class GenSignalFeatures(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.temporal = TemporalBlock(embed_dim=embed_dim)     # outputs 128-dim features

        self.transformer = TransformerEncoder(embed_dim=embed_dim, num_heads=4, num_layers=2)

    def forward(self, signal):
        x = self.temporal(signal)          # (B,L,128)
        x = self.transformer(x)            # (B,L,E)
        return x


# ------------------------------------------------------
# Proto model
# ------------------------------------------------------
class Proto(nn.Module):
    def __init__(self, ecg=False, fs=125, input_seq_len_s=10, embed_dim=128):
        super().__init__()
        self.input_seq_len = fs * input_seq_len_s
        self.ecg = ecg
        self.embed_dim = 2 * embed_dim if ecg else embed_dim
        self.in_channels = 2 if ecg else 1

        self.ppg_f = GenSignalFeatures(embed_dim)
        if ecg:
            self.ecg_f = GenSignalFeatures(embed_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() != 3:
            raise ValueError("Input must be [B,T,C] or [B,C,T]")

        # Ensure (B,C,T)
        if x.shape[1] == self.input_seq_len and x.shape[2] == self.in_channels:
            x = x.permute(0, 2, 1)
        if x.shape[1] != self.in_channels:
            raise ValueError("Channel mismatch")

        ppg = self.ppg_f(x[:, 0, :])
        ppg = torch.mean(ppg, dim=1)

        if self.ecg:
            ecg = self.ecg_f(x[:, 1, :])
            ecg = torch.mean(ecg, dim=1)
            return torch.cat((ppg, ecg), dim=-1)
        return ppg


# ------------------------------------------------------
# CLI + model profiling
# ------------------------------------------------------
def parseargs():
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--ecg', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--fs', default=125, type=int)
    p.add_argument('--input_seq_len_s', default=10, type=int)
    p.add_argument('--embed_dim', default=128, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parseargs()
    encoder = Proto(args.ecg, args.fs, args.input_seq_len_s, args.embed_dim)

    x = torch.rand((args.batch_size, args.input_seq_len_s * args.fs, 2 if args.ecg else 1))

    print("\n--- Encoder Summary ---")
    summary(encoder, input_data=[x], col_names=("input_size","output_size","num_params","mult_adds"))

    macs, params = profile(encoder, inputs=(x,))
    print("MACs, Params:", clever_format([macs, params], "%.3f"))

    head = BPRegressor(encoder.embed_dim, 3)
    y = torch.rand((args.batch_size, encoder.embed_dim))

    print("\n--- Head Summary ---")
    summary(head, input_data=[y])