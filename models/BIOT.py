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


class BIOTDecoder(nn.Module):
    def __init__(self, 
                 embed_dim=256,
                 num_heads=8,
                 num_decoder_layers=4,
                 output_seq_len=625,
                 dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.output_seq_len = output_seq_len
        self.num_decoder_layers = num_decoder_layers
        
        # Positional encoding for decoder
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout, max_len=output_seq_len)
        
        # Learnable queries for output reconstruction
        self.bp_queries = nn.Parameter(torch.randn(1, output_seq_len, embed_dim))
        
        # Memory projection to expand encoder features
        self.memory_projection = nn.Linear(embed_dim, embed_dim)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_decoder_layers)
        ])
        
        # Self-attention layers
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_decoder_layers)
        ])
        
        # Layer normalization layers
        self.cross_attn_layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_decoder_layers)
        ])
        
        self.self_attn_layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_decoder_layers)
        ])
        
        self.ffn_layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_decoder_layers)
        ])
        
        # Feed-forward networks
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_decoder_layers)
        ])
        
        # Final projection to blood pressure values
        self.bp_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1)
        )
        
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: (batch_size, embed_dim) - encoded features from BIOT encoder
        Returns:
            bp_signal: (batch_size, output_seq_len, 1) - reconstructed blood pressure
        """
        batch_size = encoder_features.size(0)
        
        # Project and expand encoder features to create memory
        memory = self.memory_projection(encoder_features)  # (batch_size, embed_dim)
        memory = memory.unsqueeze(1).expand(-1, self.output_seq_len, -1)  # (batch_size, output_seq_len, embed_dim)
        
        # Initialize decoder input with learnable queries
        decoder_input = self.bp_queries.expand(batch_size, -1, -1)  # (batch_size, output_seq_len, embed_dim)
        decoder_input = self.positional_encoding(decoder_input)
        
        # Pass through decoder layers
        for i in range(self.num_decoder_layers):
            # Self-attention
            self_attn_output, _ = self.self_attention_layers[i](
                query=decoder_input,
                key=decoder_input,
                value=decoder_input
            )
            decoder_input = self.self_attn_layer_norms[i](decoder_input + self_attn_output)
            
            # Cross-attention with encoder memory
            cross_attn_output, _ = self.cross_attention_layers[i](
                query=decoder_input,
                key=memory,
                value=memory
            )
            decoder_input = self.cross_attn_layer_norms[i](decoder_input + cross_attn_output)
            
            # Feed-forward network
            ffn_output = self.feed_forwards[i](decoder_input)
            decoder_input = self.ffn_layer_norms[i](decoder_input + ffn_output)
        
        # Final projection to blood pressure values
        bp_signal = self.bp_projection(decoder_input)  # (batch_size, output_seq_len, 1)
        
        return bp_signal


class BIOT(nn.Module):
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
                 output_seq_len=625):
        super().__init__()
        
        # Input Data Setup
        self.input_seq_len = input_seq_len_s * fs
        self.output_seq_len = output_seq_len
        self.ecg = ecg
        self.resp = resp
        self.sig2sig = sig2sig
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("hann_window", torch.hann_window(self.n_fft), persistent=False)
        
        # Get input channels for each modality
        self.ppg_in_channels, self.ecg_in_channels, self.resp_in_channels = self.get_input_channels()
        self.in_channels = self.ppg_in_channels + self.ecg_in_channels + self.resp_in_channels

        # Architecture Setup
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        # Encoder components (from original BIOT)
        self.patch_embedding = PatchFrequencyEmbedding(
            embed_dim=self.embed_dim, n_freq=self.n_fft // 2 + 1
        )
        
        self.encoder_transformer = LinearAttentionTransformer(
            dim=self.embed_dim,
            heads=self.num_heads,
            depth=self.num_encoder_layers,
            max_seq_len=self.input_seq_len,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )
        
        self.positional_encoding = PositionalEncoding(self.embed_dim)

        # Channel token embedding
        self.channel_tokens = nn.Embedding(self.in_channels, self.embed_dim)
        self.index = nn.Parameter(
            torch.LongTensor(range(self.in_channels)), requires_grad=False
        )
        
        # Output decoder to reconstruct BP
        self.decode = BIOTDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            output_seq_len=output_seq_len,
            dropout=0.1
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
        """Encoder forward pass - extracts features from input signals"""
        x = x.unsqueeze(-1) if len(x.shape) == 2 else x 
        
        # x -> (batch_size, channel, ts)
        x = x.permute(0, 2, 1) 
        
        emb_seq = []
        for i in range(x.shape[1]):
            # Get spectral representation
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            
            # Add channel token embedding
            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            
            # Add positional encoding
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

            # Optional perturbation for data augmentation
            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]
            
            emb_seq.append(channel_emb)

        # Concatenate all channel embeddings
        emb = torch.cat(emb_seq, dim=1)
        
        # Pass through encoder transformer and get global representation
        emb = self.encoder_transformer(emb).mean(dim=1)  # (batch_size, embed_dim)
        
        return emb

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        Full forward pass: encode input signals and decode blood pressure
        
        Args:
            x: Input tensor (batch_size, seq_len, channels)
        Returns:
            output: Reconstructed blood pressure (batch_size, output_seq_len, 1)
        """
        # Encode input signals to get global representation
        encoder_features = self.encode(x, n_channel_offset, perturb)
        
        # Decode blood pressure in time domain
        output = self.decode(encoder_features)
        
        return output

    def get_input_channels(self):
        ppg_in_channels = 1  # PPG must always be present
        ecg_in_channels = 1 if self.ecg else 0
        resp_in_channels = 1 if self.resp else 0
        return ppg_in_channels, ecg_in_channels, resp_in_channels
    
    def print_summary(self, batch_size=128):
        ppg_in_channels, ecg_in_channels, resp_in_channels = self.get_input_channels()
        total_input_channels = ppg_in_channels + ecg_in_channels + resp_in_channels
        input_data = torch.rand((batch_size, self.input_seq_len, total_input_channels))

        print("\n--- Model Summary (Full Forward/Backward) ---")
        summary(self, input_data=[input_data], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))

        print("\n--- MACs and Parameters (Full Forward/Backward) ---")
        macs, num_params = profile(self, inputs=(input_data,))
        macs, num_params = clever_format([macs, num_params], "%.7f")
        print(f'BIOT has {num_params} params and {macs} MACs.')


def parseargs():
    parser = argparse.ArgumentParser(description="BIOT summary, # params and MACS")

    parser.add_argument('--batch_size', default=128, type=int, help='batch size for the dummy input')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to include ECG')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to include respiratory signal')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='signal-to-signal mode')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
    parser.add_argument('--output_seq_len', default=625, type=int, help='output sequence length (BP samples)')
    parser.add_argument('--embed_dim', default=256, type=int, help='transformer embedding dimension')
    parser.add_argument('--num_heads', default=8, type=int, help='number of attention heads')
    parser.add_argument('--num_encoder_layers', default=4, type=int, help='number of encoder layers')
    parser.add_argument('--num_decoder_layers', default=4, type=int, help='number of decoder layers')
    parser.add_argument('--n_fft', default=256, type=int, help='STFT n_fft parameter')
    parser.add_argument('--hop_length', default=128, type=int, help='STFT hop length parameter')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()    

    # Create time-domain blood pressure model
    net = BIOT(
        ecg=args.ecg,
        resp=args.resp,
        sig2sig=args.sig2sig,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        output_seq_len=args.output_seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )
    
    net.print_summary(batch_size=args.batch_size)
    