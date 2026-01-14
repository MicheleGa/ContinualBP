import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer
from torchinfo import summary
from thop import profile, clever_format
from component_factory import BPRegressor, AttentionPool


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BIOTEncoder(nn.Module):
    def __init__(
        self,
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=16,
        n_fft=200,
        hop_length=100,
        **kwargs
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )

        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )

        self.positional_encoding = PositionalEncoding(emb_size)

        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def stft(self, sample):
        spectral = torch.stft(
            input=sample.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0, perturb=False):
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape

            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        emb = torch.cat(emb_seq, dim=1)
        emb = self.transformer(emb).mean(dim=1)
        return emb
    

class BIOT(nn.Module):
    r"""
    Source: https://github.com/ycq091044/BIOT/blob/main/model/biot.py
    """
    def __init__(self,
                 ecg=False,
                 fs=200,
                 input_seq_len_s=10,
                 embed_dim=256,
                 num_heads=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 n_fft=200,
                 hop_length=100,
                 pretrained_path=None):
        super().__init__()

        self.input_seq_len = input_seq_len_s * fs
        self.ecg = ecg
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.in_channels = 2 if self.ecg else 1

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        self.encoder = BIOTEncoder(
            emb_size=embed_dim,
            heads=num_heads,
            depth=num_encoder_layers,
            n_channels=self.in_channels, # Note that we exclude part of the ckpt that mismatch the number of channels
            n_fft=n_fft,
            hop_length=hop_length
        )

        if pretrained_path != '':
            ckpt = torch.load(pretrained_path, map_location='cpu')
            
            # If checkpoint has a nested encoder key
            if 'encoder' in ckpt:
                ckpt = ckpt['encoder']

            # Keys to exclude
            exclude_keys = {
                'index',
                'channel_tokens.weight',
                'channel_tokens.bias'
            }

            # Filter the checkpoint dict
            ckpt = {k: v for k, v in ckpt.items() if k not in exclude_keys}

            # Load weights
            missing, unexpected = self.encoder.load_state_dict(ckpt, strict=False)
            print(f"Loaded pretrained encoder from {pretrained_path}")
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")

    def forward(self, x, n_channel_offset=0):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        if len(x.shape) != 3:
            raise ValueError("Input tensor must have three dimensions [B, T, C] or [B, C, T]")
        if (x.shape[1] == self.input_seq_len and x.shape[2] == self.in_channels):
            x = x.permute(0, 2, 1)
        if (x.shape[1] != self.in_channels or x.shape[2] != self.input_seq_len):
            raise ValueError("The provided input tensor is not [B, C, T] and does not have the right T,C provided during initialization")
        
        return self.encoder(x)
  

def parseargs():
    parser = argparse.ArgumentParser(description="BIOT summary")
    
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for training/inference (default: 128)')
    parser.add_argument('--ecg', action=argparse.BooleanOptionalAction, default=False, help='enable ECG signal processing mode')
    parser.add_argument('--fs', default=125, type=int, help='sampling frequency in Hz (default: 125)')
    parser.add_argument('--input_seq_len_s', default=10, type=int, help='input sequence length in seconds (default: 10)')
    parser.add_argument('--embed_dim', default=256, type=int, help='embedding dimension (default: 256)')
    parser.add_argument('--num_heads', default=8, type=int, help='number of attention heads (default: 8)')
    parser.add_argument('--num_encoder_layers', default=4, type=int, help='number of transformer encoder layers (default: 4)')
    parser.add_argument('--num_decoder_layers', default=4, type=int, help='number of transformer decoder layers (default: 4)')
    parser.add_argument('--n_fft', default=200, type=int, help='FFT size for spectrogram computation (default: 200)')
    parser.add_argument('--hop_length', default=100, type=int, help='hop length for spectrogram window (default: 100)')
    parser.add_argument('--pretrained_encoder_ckpt_path', default='', type=str, help='path to pretrained encoder checkpoint (optional)')
    parser.add_argument('--use_lora', action=argparse.BooleanOptionalAction, default=False, help='enable LoRA (Low-Rank Adaptation) fine-tuning')
    parser.add_argument('--lora_r', default=8, type=int, help='LoRA rank parameter (default: 8)')
    parser.add_argument('--lora_alpha', default=16, type=float, help='LoRA alpha scaling parameter (default: 16)')
    parser.add_argument('--lora_dropout', default=0.1, type=float, help='LoRA dropout rate (default: 0.1)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()

    encoder = BIOT(
        ecg=args.ecg, 
        fs=args.fs, 
        input_seq_len_s=args.input_seq_len_s, 
        embed_dim=args.embed_dim, 
        num_heads=args.num_heads, 
        num_encoder_layers=args.num_encoder_layers, 
        num_decoder_layers=args.num_decoder_layers, 
        n_fft=args.n_fft, 
        hop_length=args.hop_length,
        pretrained_path=args.pretrained_encoder_ckpt_path
        )
    input_tensor = torch.rand((args.batch_size, args.input_seq_len_s * args.fs, 2 if args.ecg else 1))
    print("\n--- Model Encoder Summary ---")
    summary(encoder, input_data=[input_tensor], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))
    macs, num_params = profile(encoder, inputs=(input_tensor,))
    macs, num_params = clever_format([macs, num_params], "%.7f")
    print(f'BIOT Encoder has {num_params} params and {macs} MACs.')
    
    prediction_head = BPRegressor(args.embed_dim, 3)
    input_tensor = torch.rand((args.batch_size, args.embed_dim))
    print("\n--- Model Prediction Head Summary ---")
    summary(prediction_head, input_data=[input_tensor], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))
    macs, num_params = profile(prediction_head, inputs=(input_tensor,))
    macs, num_params = clever_format([macs, num_params], "%.7f")
    print(f'BIOT Prediction Head has {num_params} params and {macs} MACs.')
