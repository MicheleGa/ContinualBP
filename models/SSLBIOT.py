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


class DecoderExpansion(nn.Module):
    """Expands global representation to sequence length"""
    def __init__(self, embed_dim, output_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_seq_len = output_seq_len
        
        # Learnable expansion using a linear layer
        self.expansion = nn.Linear(embed_dim, embed_dim * output_seq_len)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, embed_dim)
        Returns:
            x: (batch_size, output_seq_len, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Expand to sequence length
        x = self.expansion(x)  # (batch_size, embed_dim * output_seq_len)
        x = x.view(batch_size, self.output_seq_len, self.embed_dim)
        
        return x


class DecoderTransformer(nn.Module):
    """Decoder transformer using LinearAttentionTransformer"""
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
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            x: (batch_size, seq_len, embed_dim)
        """
        return self.transformer(x)


class BIOTDecoder(nn.Module):
    """
    Improved decoder architecture using LinearAttentionTransformer
    for both supervised (BP prediction) and self-supervised (signal reconstruction) tasks
    """
    def __init__(self, 
                 embed_dim=256,
                 num_heads=8,
                 num_decoder_layers=4,
                 output_seq_len=625,
                 dropout=0.1,
                 output_channels=1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.output_seq_len = output_seq_len
        self.num_decoder_layers = num_decoder_layers
        self.output_channels = output_channels
        
        # Build decoder components
        self.decoder = self._build_decoder()
        
    def _build_decoder(self):
        """Build the decoder architecture using LinearAttentionTransformer"""
        return nn.Sequential(
            # First expand the global representation to sequence length
            DecoderExpansion(self.embed_dim, self.output_seq_len),
            
            # Apply positional encoding to the expanded sequence
            PositionalEncoding(self.embed_dim, dropout=0.1, max_len=self.output_seq_len),
            
            # Decoder transformer layers
            DecoderTransformer(
                embed_dim=self.embed_dim,
                num_heads=8,  # Fixed to 8 heads to match encoder
                num_layers=self.num_decoder_layers,
                max_seq_len=self.output_seq_len
            ),
            
            # Final projection to desired output channels
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim // 4, self.output_channels)
            )
        )
        
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: (batch_size, embed_dim) - encoded features from BIOT encoder
        Returns:
            output_signal: (batch_size, output_seq_len, output_channels) - reconstructed signal(s)
        """
        return self.decoder(encoder_features)


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
                 output_seq_len=625):
        super().__init__()
        
        # Input Data Setup
        self.input_seq_len = input_seq_len_s * fs
        self.ecg = ecg
        self.resp = resp
        self.sig2sig = sig2sig 
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("hann_window", torch.hann_window(self.n_fft), persistent=False)
        if sig2sig:
            self.output_seq_len = self.input_seq_len
        else:
            self.output_seq_len = output_seq_len
        
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
            max_seq_len=self.input_seq_len, # Max sequence length for transformer
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )
        
        self.positional_encoding = PositionalEncoding(self.embed_dim)

        # Channel token embedding
        self.channel_tokens = nn.Embedding(self.in_channels, self.embed_dim)
        self.index = nn.Parameter(
            torch.LongTensor(range(self.in_channels)), requires_grad=False
        )
        
        # --- Supervised Decoder Head ---
        # This is the decoder for BP reconstruction (supervised task)
        # Outputs single channel (blood pressure)
        self.supervised_decoder_head = BIOTDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            output_seq_len=self.output_seq_len,
            dropout=0.1,
            output_channels=1  # BP reconstruction, always 1 channel
        )

        # --- Self-Supervision: Reconstruction Head for MSR ---
        # Reconstructs all input signals (PPG, ECG, Resp)
        # Outputs same number of channels as input
        self.reconstruction_decoder_head = BIOTDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_decoder_layers=num_decoder_layers,
            output_seq_len=self.input_seq_len, # Reconstruct to original input sequence length
            dropout=0.1,
            output_channels=self.in_channels # Reconstruct all input channels
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
        """
        Encoder forward pass - extracts features from input signals.
        This serves as the shared feature extractor for SSL tasks.
        
        Args:
            x: Input tensor (batch_size, seq_len, channels)
        Returns:
            emb: Encoded features (batch_size, embed_dim)
        """
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
                ts_orig = channel_emb.shape[1]
                ts_new = np.random.randint(ts_orig // 2, ts_orig + 1) # Ensure ts_new is at least ts_orig // 2
                # Ensure selected_ts does not exceed ts_orig
                selected_ts_indices = np.random.choice(range(ts_orig), ts_new, replace=False)
                # Sort indices to maintain temporal order, which might be important for positional encoding
                selected_ts_indices.sort()
                channel_emb = channel_emb[:, selected_ts_indices]
            
            emb_seq.append(channel_emb)

        # Concatenate all channel embeddings
        emb = torch.cat(emb_seq, dim=1)
        
        # Pass through encoder transformer and get global representation
        emb = self.encoder_transformer(emb).mean(dim=1)  # (batch_size, embed_dim)
        
        return emb

    def reconstruct(self, encoder_features):
        """
        Reconstruction head for MSR (Multi-Signal Reconstruction).
        This method will reconstruct the original input signals from the encoder features.
        
        Args:
            encoder_features (torch.Tensor): Output from the encoder (batch_size, embed_dim)
        Returns:
            reconstructed_signal (torch.Tensor): Reconstructed input signals (batch_size, input_seq_len, in_channels)
        """
        reconstructed_signal = self.reconstruction_decoder_head(encoder_features)
        return reconstructed_signal.permute(0, 2, 1) # Permute back to [Batch, Length, Modalities]

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        Full forward pass for supervised training: encode input signals and decode blood pressure.
        
        Args:
            x: Input tensor (batch_size, seq_len, channels)
        Returns:
            output: Reconstructed blood pressure (batch_size, output_seq_len)
        """
        # Encode input signals to get global representation
        encoder_features = self.encode(x, n_channel_offset, perturb)
        
        # Decode blood pressure in time domain using the supervised head
        output = self.supervised_decoder_head(encoder_features)
        
        # Output shape: (batch_size, output_seq_len, 1) -> (batch_size, output_seq_len)
        return output.squeeze(-1)

    def get_input_channels(self):
        # PPG must always be present
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
    parser = argparse.ArgumentParser(description="SSLBIOT summary, # params and MACS")

    parser.add_argument('--batch_size', default=128, type=int, help='batch size for the dummy input')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to include ECG')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to include respiratory signal')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='signal-to-signal mode (supervised output shape)')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
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

    # Create SSL BIOT model
    net = SSLBIOT(
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
        hop_length=args.hop_length
    )
    
    net.print_summary(batch_size=args.batch_size)