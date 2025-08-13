import argparse
from distutils.util import strtobool
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer
from torchinfo import summary
from thop import profile, clever_format


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, embed_dim=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, embed_dim)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, embed_dim)
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
    

class EncoderToDecoderProjector(nn.Module):
    def __init__(self, input_len, output_len, embed_dim):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.embed_dim = embed_dim

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=self.output_len, mode='linear', align_corners=True)
        x = x.permute(0, 2, 1)
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.transformer = LinearAttentionTransformer(
            dim=embed_dim,
            heads=num_heads,
            depth=num_layers,
            max_seq_len=max_seq_len,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )

    def forward(self, x):
        return self.transformer(x)


class BIOTDecoder(nn.Module):
    def __init__(self, 
                 embed_dim=256,
                 num_heads=8,
                 num_decoder_layers=4,
                 input_seq_len=625,
                 output_seq_len=625,
                 dropout=0.1,
                 output_channels=1):
        super().__init__()
        self.decoder = nn.Sequential(
            EncoderToDecoderProjector(input_seq_len, output_seq_len, embed_dim),
            PositionalEncoding(embed_dim, dropout=dropout, max_len=output_seq_len),
            DecoderTransformer(embed_dim, num_heads, num_decoder_layers, output_seq_len),
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 4, output_channels)
            )
        )

    def forward(self, encoder_sequence):
        return self.decoder(encoder_sequence)


class SSLBIOT(nn.Module):
    def __init__(self,
                 ecg=False,
                 resp=False,
                 sig2sig=False,
                 fs=125,
                 input_seq_len_s=5,
                 embed_dim=256,
                 num_heads=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 n_fft=256,
                 hop_length=128,
                 output_seq_len=625,
                 use_stft=True):
        super().__init__()

        self.input_seq_len = input_seq_len_s * fs
        self.output_seq_len = self.input_seq_len if sig2sig else output_seq_len
        self.use_stft = use_stft
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.register_buffer("hann_window", torch.hann_window(self.n_fft), persistent=False)

        self.ecg = ecg
        self.resp = resp
        self.sig2sig = sig2sig

        self.ppg_in_channels, self.ecg_in_channels, self.resp_in_channels = self.get_input_channels()
        self.in_channels = self.ppg_in_channels + self.ecg_in_channels + self.resp_in_channels

        self.embed_dim = embed_dim
        self.encoder_transformer = LinearAttentionTransformer(
            dim=embed_dim,
            heads=num_heads,
            depth=num_encoder_layers,
            max_seq_len=self.input_seq_len,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )

        if use_stft:
            self.patch_embedding = PatchFrequencyEmbedding(embed_dim=embed_dim, n_freq=n_fft // 2 + 1)
        else:
            self.time_domain_projection = nn.Linear(1, embed_dim)

        self.positional_encoding = PositionalEncoding(embed_dim)
        self.channel_tokens = nn.Embedding(self.in_channels, embed_dim)
        self.index = nn.Parameter(torch.LongTensor(range(self.in_channels)), requires_grad=False)

        self.supervised_decoder_head = BIOTDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            input_seq_len=self.input_seq_len * self.in_channels,  # Approx upper bound
            output_seq_len=self.output_seq_len,
            output_channels=1
        )

        self.reconstruction_decoder_head = BIOTDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            input_seq_len=self.input_seq_len * self.in_channels,
            output_seq_len=self.input_seq_len,
            output_channels=self.in_channels
        )

    def stft(self, sample):
        spectral = torch.stft(
            input=sample.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.hann_window,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)

    def encode(self, x, n_channel_offset=0, perturb=False):
        x = x.unsqueeze(-1) if len(x.shape) == 2 else x
        x = x.permute(0, 2, 1)

        emb_seq = []
        for i in range(x.shape[1]):
            channel_input = x[:, i : i + 1, :]

            if self.use_stft:
                channel_emb = self.stft(channel_input)
                channel_emb = self.patch_embedding(channel_emb)
            else:
                channel_input = channel_input.squeeze(1).unsqueeze(-1)
                channel_emb = self.time_domain_projection(channel_input)

            batch_size, ts, _ = channel_emb.shape

            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0).unsqueeze(0).repeat(batch_size, ts, 1)
            )
            channel_emb = self.positional_encoding(channel_emb + channel_token_emb)

            if perturb:
                ts_orig = channel_emb.shape[1]
                ts_new = np.random.randint(ts_orig // 2, ts_orig + 1)
                selected_ts = np.random.choice(range(ts_orig), ts_new, replace=False)
                selected_ts.sort()
                channel_emb = channel_emb[:, selected_ts]

            emb_seq.append(channel_emb)

        emb = torch.cat(emb_seq, dim=1)
        emb = self.encoder_transformer(emb)
        return emb

    def forward(self, x, n_channel_offset=0, perturb=False):
        encoder_seq = self.encode(x, n_channel_offset, perturb)
        output = self.supervised_decoder_head(encoder_seq)
        return output.squeeze(-1)

    def reconstruct(self, encoder_seq):
        reconstructed = self.reconstruction_decoder_head(encoder_seq)
        return reconstructed.permute(0, 2, 1)

    def get_input_channels(self):
        ppg_in_channels = 1
        ecg_in_channels = 1 if self.ecg else 0
        resp_in_channels = 1 if self.resp else 0
        return ppg_in_channels, ecg_in_channels, resp_in_channels
    
    
    def print_summary(self, batch_size=128):
        ppg_in_channels, ecg_in_channels, resp_in_channels = self.get_input_channels()
        total_input_channels = ppg_in_channels + ecg_in_channels + resp_in_channels
        input_data = torch.rand((batch_size, self.input_seq_len, total_input_channels))

        print("\n--- Model Summary (Full Supervised Forward/Backward) ---")
        summary(self, input_data=[input_data], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))

        print("\n--- MACs and Parameters (Full Supervised Forward/Backward) ---")
        macs, num_params = profile(self, inputs=(input_data,))
        macs, num_params = clever_format([macs, num_params], "%.7f")
        print(f'SSLBIOT has {num_params} params and {macs} MACs.')

        print("\n--- SSL Head Summaries ---")
        print("\n--- Encoder (Shared Feature Extractor) Summary ---")
        dummy_encoder_input = torch.rand((batch_size, self.input_seq_len, total_input_channels))
        dummy_encoder_output = self.encode(dummy_encoder_input)
        print(f"  Encoder output shape: {dummy_encoder_output.shape}") # Should be (batch_size, embed_dim)

        # Summarize the 'reconstruction_decoder_head'
        print("\n--- Reconstruction Decoder Head Summary (for MSR) ---")
        summary(self.reconstruction_decoder_head, input_data=dummy_encoder_output)
        
        # Test reconstruction functionality
        print("\n--- Testing Reconstruction Function ---")
        reconstructed = self.reconstruct(dummy_encoder_output)
        print(f"  Reconstructed signal shape: {reconstructed.shape}")  # Should be (batch_size, input_seq_len, in_channels)
        
        print("\n--- Supervised Decoder Head Summary (for BP prediction) ---")
        summary(self.supervised_decoder_head, input_data=dummy_encoder_output)
        

def parseargs():
    parser = argparse.ArgumentParser(description="SSLBIOT summary, # params and MACs")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for dummy input')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='Include ECG')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='Include respiration')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='Signal-to-signal mode')
    parser.add_argument('--fs', default=125, type=int, help='Sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='Input sequence length in seconds')
    parser.add_argument('--embed_dim', default=256, type=int, help='Transformer embedding dimension')
    parser.add_argument('--num_heads', default=8, type=int, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', default=4, type=int, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', default=4, type=int, help='Number of decoder layers')
    parser.add_argument('--n_fft', default=256, type=int, help='STFT n_fft parameter')
    parser.add_argument('--hop_length', default=128, type=int, help='STFT hop length parameter')
    parser.add_argument('--use_stft', default='False', type=lambda x: bool(strtobool(x)), help='Use STFT or raw time-domain input')
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()

    model = SSLBIOT(
        ecg=args.ecg,
        resp=args.resp,
        sig2sig=args.sig2sig,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        use_stft=args.use_stft
    )

    model.print_summary(batch_size=args.batch_size)
