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
        self.channel_tokens = nn.Embedding(n_channels, 256)
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


class BPWaveformDecoder(nn.Module):
    """Decoder that maps sequence embeddings directly to waveform samples."""
    def __init__(self, emb_size, output_len, depth=4, heads=8):
        super().__init__()
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            depth=depth,
            heads=heads,
            max_seq_len=1024,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )
        self.linear_out = nn.Linear(emb_size, output_len)

    def forward(self, x):
        # x: (batch, seq_len, emb_size)
        x = self.transformer(x)          # (batch, seq_len, emb_size)
        x = x.mean(dim=1)                 # pool sequence → (batch, emb_size)
        x = self.linear_out(x)            # (batch, output_len)
        return x


class BPRegressor(torch.nn.Module):
    """
    Blood pressure regressor for SBP/DBP/MAP prediction during Reptile stage.
    """
    def __init__(self, input_dim, output_dim=2):  # 2 for SBP/DBP, can be 3 for SBP/DBP/MAP
        super(BPRegressor, self).__init__()
        
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.regressor(x)


class BIOTPredictionHead(nn.Module):
    """
    Wrapper head that selects the appropriate prediction head.
    - If output_dim == 3 → BPRegressor
    - Else → BPWaveformDecoder
    """
    def __init__(self, emb_size, output_dim, depth=4, heads=8):
        super().__init__()
        
        if output_dim == 3:
            # BPRegressor expects input_dim → emb_size
            self.head = BPRegressor(input_dim=emb_size, output_dim=output_dim)
        else:
            # BPWaveformDecoder requires output_len
            self.head = BPWaveformDecoder(emb_size, output_dim, depth=depth, heads=heads)

    def forward(self, x):
        return self.head(x)


class MAMLLearner(nn.Module):
    """
    Learner wrapper for MAML-style functional forward using torch.func.functional_call.
    Supports both the BIOT encoder and a flexible prediction head (BPRegressor or BPWaveformDecoder).
    """
    def __init__(self, model, head: nn.Module):
        super().__init__()
        self.model = model
        self.head = head  # can be BPRegressor or BPWaveformDecoder

        # Capture parameter names in deterministic order
        self.model_named_params = list(self.model.named_parameters())
        self.head_named_params = list(self.head.named_parameters())

        # Flattened list of parameters (order must match .parameters())
        self._flattened_params = [p for _, p in self.model_named_params if p.is_floating_point() or p.is_complex()]
        self._flattened_params += [p for _, p in self.head_named_params if p.is_floating_point() or p.is_complex()]
        
        # Save names for reconstruction of param dicts
        self._model_names_in_order = [n for n, _ in self.model_named_params]
        self._head_names_in_order = [n for n, _ in self.head_named_params]

        # Guardrail: expected embedding size from encoder
        self.expected_feat_dim = getattr(self.model, "embed_dim", None)

        # Track which type of head we’re using
        self.is_regressor = isinstance(self.head, BPRegressor)
        self.is_decoder   = isinstance(self.head, BPWaveformDecoder)

    def _split_vars_to_dicts(self, vars_list):
        """
        Split a flat list of tensors into two param dicts matching model and head.
        """
        n_model = len(self._model_names_in_order)
        model_vars = vars_list[:n_model]
        head_vars = vars_list[n_model:]

        model_param_dict = {name: tensor for name, tensor in zip(self._model_names_in_order, model_vars)}
        head_param_dict = {name: tensor for name, tensor in zip(self._head_names_in_order, head_vars)}
        return model_param_dict, head_param_dict

    def forward(self, x, vars=None):
        """
        If vars is None -> standard forward.
        If vars is not None -> functional forward with provided fast weights.
        """
        if vars is None:
            feats = self.model(x)

            if self.is_regressor:
                # Expect [B, D]
                assert feats.dim() == 2, f"Regressor head expects [B, D], got {list(feats.shape)}"
                if self.expected_feat_dim is not None:
                    assert feats.shape[-1] == self.expected_feat_dim, (
                        f"Encoder features mismatch: expected {self.expected_feat_dim}, got {feats.shape[-1]}"
                    )
            elif self.is_decoder:
                # Expect [B, T, D]
                assert feats.dim() == 3, f"Decoder head expects [B, T, D], got {list(feats.shape)}"

            return self.head(feats)

        # ---- Functional forward (MAML inner loop) ----
        model_param_dict, head_param_dict = self._split_vars_to_dicts(vars)

        feats = torch.func.functional_call(self.model, model_param_dict, (x,))

        if self.is_regressor:
            # Guard against accidental sequence output
            if feats.dim() == 3:
                feats = feats.mean(dim=1)
            assert feats.dim() == 2, f"Regressor head expects [B, D], got {list(feats.shape)}"
            if self.expected_feat_dim is not None:
                assert feats.shape[-1] == self.expected_feat_dim, (
                    f"Encoder features mismatch: expected {self.expected_feat_dim}, got {feats.shape[-1]}"
                )

        elif self.is_decoder:
            assert feats.dim() == 3, f"Decoder head expects [B, T, D], got {list(feats.shape)}"

        out = torch.func.functional_call(self.head, head_param_dict, (feats,))
        return out


class BIOT(nn.Module):
    def __init__(self,
                 ecg=False,
                 resp=False,
                 sig2sig=False,
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
        self.sig2sig = sig2sig
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.ppg_in_channels, self.ecg_in_channels = self.get_input_channels()
        self.in_channels = self.ppg_in_channels + self.ecg_in_channels

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        # Project 2 → pretrained's 18 channels
        self.pretrained_channels = 18 if pretrained_path else self.in_channels
        self.channel_proj = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.pretrained_channels,
            kernel_size=1
        )

        self.encoder = BIOTEncoder(
            emb_size=embed_dim,
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
            self.encoder.load_state_dict(ckpt, strict=False)
            print(f"Loaded pretrained encoder from {pretrained_path}")

        self.decoder = BPWaveformDecoder(
            emb_size=embed_dim,
            output_len=self.input_seq_len,
            depth=num_decoder_layers,
            heads=num_heads
        )

    def forward(self, x, n_channel_offset=0):
        # Accept both (B, T, C) and (B, C, T)
        if x.dim() == 3 and x.shape[1] == self.input_seq_len and x.shape[2] == self.in_channels:
            x = x.permute(0, 2, 1)
        elif x.dim() == 3 and x.shape[1] == self.in_channels:
            pass
        else:
            x = x.permute(0, 2, 1)
        
        if self.pretrained_channels != self.in_channels:
            # Project input channels to match pretrained model's expected channels
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

        return self.encoder.transformer(emb).mean(axis=1)    # (B, emb_dim)

    def get_input_channels(self):
        ppg_in_channels = 1
        ecg_in_channels = 1 if self.ecg else 0
        return ppg_in_channels, ecg_in_channels

    def print_summary(self, batch_size=128):
        total_input_channels = self.in_channels
        input_data = torch.rand((batch_size, self.input_seq_len, total_input_channels))
        print("\n--- Model Summary ---")
        summary(self, input_data=[input_data], 
                col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))
        macs, num_params = profile(self, inputs=(input_data,))
        macs, num_params = clever_format([macs, num_params], "%.7f")
        print(f'BIOT has {num_params} params and {macs} MACs.')


def parseargs():
    parser = argparse.ArgumentParser(description="BIOT summary")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)))
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)))
    parser.add_argument('--fs', default=125, type=int)
    parser.add_argument('--input_seq_len_s', default=5, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_encoder_layers', default=4, type=int)
    parser.add_argument('--num_decoder_layers', default=4, type=int)
    parser.add_argument('--n_fft', default=200, type=int)
    parser.add_argument('--hop_length', default=100, type=int)
    parser.add_argument('--pretrained_path', default='', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    net = BIOT(
        ecg=args.ecg,
        sig2sig=args.sig2sig,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        pretrained_path=args.pretrained_path
    )
    net.print_summary(batch_size=args.batch_size)
