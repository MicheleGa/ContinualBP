import argparse
from distutils.util import strtobool
import math
import numpy as np
import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer
from torchinfo import summary
from thop import profile, clever_format


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
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
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
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
            attn_layer_dropout=0.2,  # dropout right after self-attention layer
            attn_dropout=0.2,  # dropout post-attention
        )
        self.positional_encoding = PositionalEncoding(emb_size)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, 256)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            # (batch_size, ts, emb)
            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            # (batch_size, ts, emb)
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

            # perturb
            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        # (batch_size, 16 * ts, emb)
        emb = torch.cat(emb_seq, dim=1)
        # (batch_size, emb)
        emb = self.transformer(emb).mean(dim=1)
        return emb


class BPWaveformDecoder(nn.Module):
    """Decoder that maps sequence embeddings directly to waveform samples."""
    def __init__(self, input_dim, output_dim, depth=4, heads=8):
        super().__init__()
        self.transformer = LinearAttentionTransformer(
            dim=input_dim,
            depth=depth,
            heads=heads,
            max_seq_len=1024,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )
        self.linear_out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"BPRegressor expected input dims 3, got {list(x.shape)}")
        
        # x: (batch, seq_len, embed_dim)
        x = self.transformer(x)          # (batch, seq_len, embed_dim)
        x = x.mean(dim=1)                 # pool sequence → (batch, embed_dim)
        x = self.linear_out(x)            # (batch, output_len)
        return x


class BPRegressor(torch.nn.Module):
    """
    Blood pressure regressor for SBP/DBP/MAP prediction during Reptile stage.
    """
    def __init__(self, input_dim, output_dim=3):  # 3 for SBP/DBP/MAP
        super().__init__()
        
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(input_dim, input_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(input_dim // 2, input_dim // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(input_dim // 4, output_dim)
        )
    
    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"BPRegressor expected input dims 3, got {list(x.shape)}")

        return self.regressor(x.mean(dim=1)) # pool sequence → (batch, embed_dim)


class BIOTPredictionHead(nn.Module):
    """
    Wrapper head that selects the appropriate prediction head.
    - If output_dim == 3 → BPRegressor
    - Else → BPWaveformDecoder
    """
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        
        if output_dim == 3:
            # BPRegressor expects input_dim → embed_dim
            self.head = BPRegressor(input_dim=embed_dim, output_dim=output_dim)
        else:
            # BPWaveformDecoder requires output_len
            self.head = BPWaveformDecoder(input_dim=embed_dim, output_dim=output_dim)

    def forward(self, x):
        return self.head(x)


class BIOT(nn.Module):
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
        
        # Project from 2 → pretrained's 18 channels
        self.pretrained_channels = 18 if pretrained_path else self.in_channels
        self.channel_proj = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.pretrained_channels,
            kernel_size=1
        )

        self.encoder = BIOTEncoder(
            embed_dim=embed_dim,
            heads=num_heads,
            depth=num_encoder_layers,
            n_channels=self.pretrained_channels,
            n_fft=n_fft,
            hop_length=hop_length
        )

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            if 'encoder' in ckpt:
                ckpt = ckpt['encoder']

            missing, unexpected = self.encoder.load_state_dict(ckpt, strict=False)
            print(f"Loaded pretrained encoder from {pretrained_path}")
            print(f"Missing keys: {missing}, Unexpected: {unexpected}")
            
    def forward(self, x, n_channel_offset=0):
        
        # Dataloader may squeeze the channel dimension when there is only PPG        
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        if len(x.shape) != 3:
            raise ValueError("Input tensor must have three dimensions [B, T, C] or [B, C, T]")
        
        # Permute to [B, C, T]
        if (x.shape[1] == self.input_seq_len and x.shape[2] == self.in_channels):
            x = x.permute(0, 2, 1)
        
        # Input sequence length and number of channels must correspond to those of initialization
        if (x.shape[1] != self.in_channels or x.shape[2] != self.input_seq_len):
            raise ValueError("The provided input tensor is not [B, C, T] and does not have the right T,C provided during initialization")
        
        # Project input channels to match pretrained model's expected channels
        if self.pretrained_channels != self.in_channels:
            x = self.channel_proj(x)  # (B, 18, T)
        
        # Build token sequence for each channel
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.encoder.stft(x[:, i:i+1, :])
            channel_spec_emb = self.encoder.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape

            channel_token_emb = (
                self.encoder.channel_tokens(self.encoder.index[i + n_channel_offset])
                .unsqueeze(0).unsqueeze(0).repeat(batch_size, ts, 1)
            )
            channel_emb = self.encoder.positional_encoding(channel_spec_emb + channel_token_emb)
            emb_seq.append(channel_emb)
        
        emb = torch.cat(emb_seq, dim=1)        # (B, total_ts, emb_dim)

        return self.encoder.transformer(emb)    # (B, total_ts, emb_dim)


def parseargs():
    parser = argparse.ArgumentParser(description="BIOT summary")
    
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for model input')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to use ECG input (True/False)')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='enable signal-to-signal prediction (True/False)')
    parser.add_argument('--fs', default=125, type=int, help='sampling frequency (Hz)')
    parser.add_argument('--input_seq_len_s', default=10, type=int, help='input sequence length in seconds')
    parser.add_argument('--embed_dim', default=256, type=int, help='embedding dimension for model layers')
    parser.add_argument('--num_heads', default=8, type=int, help='number of attention heads')
    parser.add_argument('--num_encoder_layers', default=4, type=int, help='number of encoder layers')
    parser.add_argument('--num_decoder_layers', default=4, type=int, help='number of decoder layers')
    parser.add_argument('--n_fft', default=256, type=int, help='fft window size for spectral embedding')
    parser.add_argument('--hop_length', default=32, type=int, help='hop length for STFT')
    parser.add_argument('--pretrained_encoder_ckpt_path', default='', type=str, help='path to pretrained encoder weights')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    
    model = BIOT(
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

    total_input_channels = model.in_channels
    input_data = torch.rand((args.batch_size, args.input_seq_len_s * args.fs, 2 if args.ecg else 1))
    print("\n--- Model Encoder Summary ---")
    summary(model, input_data=[input_data], 
            col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))
    macs, num_params = profile(model, inputs=(input_data,))
    macs, num_params = clever_format([macs, num_params], "%.7f")
    print(f'BIOT Encoder has {num_params} params and {macs} MACs.')
    feat_dim = model.embed_dim
    output_dim = args.input_seq_len_s * args.fs if args.sig2sig else 3  # Full waveform or SBP/DBP/MAP
    bp_prediction_head = BIOTPredictionHead(feat_dim, output_dim)
    feats = model(input_data)
    print("\n--- Model Prediction Head Summary ---")
    summary(bp_prediction_head, input_data=[feats], 
            col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))
    macs, num_params = profile(bp_prediction_head, inputs=(feats,))
    macs, num_params = clever_format([macs, num_params], "%.7f")
    print(f'BIOT Prediction Head has {num_params} params and {macs} MACs.')
    