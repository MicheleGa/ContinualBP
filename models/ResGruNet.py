import argparse
from functools import partial
from distutils.util import strtobool

import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from thop import profile, clever_format


class TemporalResBlock(nn.Module):
    def __init__(self, inCh, outCh, kernel_size=7, act='relu', pooling='max', num_groups=8):
        super().__init__()
        
        padding = (kernel_size - 1) // 2 
        self.conv1 = nn.Conv1d(in_channels=inCh, out_channels=inCh, kernel_size=kernel_size, padding=padding, groups=inCh)
        
        # GroupNorm for conv1
        groups = num_groups if inCh >= num_groups else 1
        self.gn1 = nn.GroupNorm(groups, inCh)
        
        self.relu1 = nn.ReLU() if act == 'relu' else nn.LeakyReLU()
        self.l1 = nn.Linear(inCh, outCh)
        
        self.relu2 = nn.ReLU() if act == 'relu' else nn.LeakyReLU()
        self.l2 = nn.Linear(outCh, outCh)
        
        # GroupNorm for after l2
        groups = num_groups if outCh >= num_groups else 1
        self.gn3 = nn.GroupNorm(groups, outCh)
        
        self.skip = nn.Conv1d(in_channels=inCh, out_channels=outCh, kernel_size=1)
        
        self.relu_out = nn.ReLU() if act == 'relu' else nn.LeakyReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if pooling == 'max' else nn.AvgPool1d(kernel_size=2, stride=2)
            
    def forward(self, x):
        # x -> [batch_size, channels, seq_len]
        
        y = self.conv1(x)
        y = self.gn1(y)
        y = self.relu1(y)

        y = y.permute(0, 2, 1)
        y = self.l1(y)
        y = self.relu2(y)
        y = self.l2(y)
        y = y.permute(0, 2, 1)
        y = self.gn3(y)

        skip = self.skip(x)
        
        out = self.relu_out(y + skip)
        out = self.pool(out)

        return out
    

class TemporalBlock(nn.Module):
    def __init__(self, channels=[1, 64, 128, 256], kernel_size=7, act='relu', pooling='avg', num_groups=8):
        super().__init__()

        self.resblocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.resblocks.append(TemporalResBlock(inCh=channels[i], outCh=channels[i+1], kernel_size=kernel_size, act=act, pooling=pooling, num_groups=num_groups))
            self.resblocks.append(TemporalResBlock(inCh=channels[i+1], outCh=channels[i+1], kernel_size=kernel_size, act=act, pooling=pooling, num_groups=num_groups))
        
    def forward(self, x):
        # Slicing over input signals modailities may remove the channel dimension
        y = x.unsqueeze(1) if len(x.shape) == 2 else x
        
        for resblock in self.resblocks:
            y = resblock(y)
        
        return y.permute(0, 2, 1) # x -> [batch_size, seq_len, channels]


class DecoderResBlock(nn.Module):
    """
    Residual block for decoding with ConvTranspose1d upsampling.
    x -> ConvTranspose1d -> GroupNorm -> ReLU -> Conv1d -> GroupNorm
    skip: 1x1 ConvTranspose1d to match channels and upsample
    """
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=4, num_groups=8):
        super().__init__()
        padding = (kernel_size - 1) // 2
        groups_out = num_groups if out_ch >= num_groups else 1
        
        self.upconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.gn1 = nn.GroupNorm(groups_out, out_ch)
        self.relu1 = nn.ReLU()

        self.conv = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.gn2 = nn.GroupNorm(groups_out, out_ch)

        self.skip = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.relu_out = nn.ReLU()

    def forward(self, x):
        
        y = self.upconv(x)
        y = self.gn1(y)
        y = self.relu1(y)

        y = self.conv(y)
        y = self.gn2(y)

        skip = self.skip(x)

        out = self.relu_out(y + skip)
        return out


class BPWaveformDecoder(nn.Module):
    """
    Decoder that upsamples sequence embeddings (B, T, D) back to waveform (B, output_len)
    using residual ConvTranspose1d blocks.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64], kernel_size=7, stride=4, num_groups=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        in_ch = input_dim
        for h in hidden_dims:
            layers.append(
                DecoderResBlock(in_ch, h, kernel_size=kernel_size, stride=stride, num_groups=num_groups)
            )
            in_ch = h

        self.res_blocks = nn.Sequential(*layers)
        # Final projection: from last hidden dim -> 1 channel waveform
        self.out_proj = nn.Conv1d(in_ch, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: (B, T, D) embedding sequence
        returns: (B, output_len) waveform
        """
        if x.dim() != 3:
            raise ValueError(f"BPWaveformDecoder expected input dims 3, got {list(x.shape)}")

        # reshape to (B, D, T)
        x = x.permute(0, 2, 1)
        y = self.res_blocks(x)
        y = self.out_proj(y)  # (B, 1, L)
        
        # adjust to exact output length
        if y.shape[-1] != self.output_dim:
            y = F.interpolate(y, size=self.output_dim, mode="linear", align_corners=False)

        return y.squeeze(1)  # (B, output_len)
    

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


class ResGruNetPredictionHead(nn.Module):
    """
    Wrapper head that selects the appropriate prediction head.
    - If output_dim == 3 → BPRegressor
    - Else → BPWaveformDecoder
    """
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        
        if output_dim == 3:
            # BPRegressor [batch_size, time_sequence, embed_dim] → [batch_size, 3]
            self.head = BPRegressor(input_dim=embed_dim, output_dim=output_dim)
        else:
            # BPWaveformDecoder [batch_size, time_sequence, embed_dim] → [batch_size, output_dim, 1]
            self.head = BPWaveformDecoder(input_dim=embed_dim, output_dim=output_dim)

    def forward(self, x):
        return self.head(x)
    

class ResGruNet(nn.Module):

    def __init__(self, 
                 ecg=False, 
                 channels='1, 64, 128, 256', 
                 kernel_size=7, 
                 act='leaky_relu', 
                 pooling='avg', 
                 embed_dim=256,
                 fs=125,
                 input_seq_len_s=10,
                 num_groups=8):
        super().__init__()
        
        self.input_seq_len = input_seq_len_s * fs
        self.ecg = ecg
        self.embed_dim = embed_dim
        self.in_channels = 2 if self.ecg else 1
        self.embed_dim = embed_dim
                
        # Feature Extractor        
        # CNN for each modality
        channels = [int(ch) for ch in channels.split(',')]
        
        self.ppg_feature_gen = TemporalBlock(channels=channels, kernel_size=kernel_size, act=act, pooling=pooling, num_groups=num_groups)
        
        if self.ecg:
            self.ecg_feature_gen = TemporalBlock(channels=channels, kernel_size=kernel_size, act=act, pooling=pooling, num_groups=num_groups)
            
        # GRU combining the modalities
        # PPG is always present while ECG/RESP may not int(ecg) == 0 when false, same for resp of course
        self.gru = nn.GRU(input_size=channels[-1] * (1 + int(self.ecg)), hidden_size=embed_dim, batch_first=True)
        # GroupNorm for GRU output
        groups = num_groups if embed_dim >= num_groups else 1
        self.gn = nn.GroupNorm(groups, embed_dim)
        
    def forward(self, x):        
        
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

        # CNN for each modality
        # x -> [batch_size, seq_len, channels]
        
        if self.ecg:
            ppg_feats = self.ppg_feature_gen(x[:,0,:])
            ecg_feats = self.ecg_feature_gen(x[:,1,:])
            
            features = torch.cat((
                ppg_feats,
                ecg_feats
                ), 
                dim=-1)
        else:
            ppg_feats = self.ppg_feature_gen(x)
            features = ppg_feats
        
        # GRU combining the modalities
        # x -> [batch_size, seq_len, channels]
        (o0, _) = self.gru(features)   
        
        return self.gn(o0.permute(0, 2, 1)).permute(0, 2, 1) # x-> [batch_size, channels, seq_len], x-> [batch_size, seq_len, channels]
        
    
def parseargs():
    parser = argparse.ArgumentParser(description="ResGruNet summary")

    parser.add_argument('--batch_size', default=128, type=int, help='batch size for model input')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load only ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)))
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=10, type=int, help='input sequence length in seconds')
    parser.add_argument('--channels', default='1, 64, 128, 256', type=str, help='channels produced by the convolutional blocks')
    parser.add_argument('--kernel_size', default=7, type=int, help='convolutional layer kernel size')
    parser.add_argument('--act', default='leaky_relu', type=str, help='which activation to use (ReLU or LeakyReLU)')
    parser.add_argument('--pooling', default='avg', type=str, help='which poolng to use (average or max)')
    parser.add_argument('--embed_dim', default=256, type=int, help='embedding dimension return by the GRU')
    parser.add_argument('--num_groups', default=8, type=int, help='number of groups for group normalization')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()  
    
    model = ResGruNet(
        ecg=args.ecg,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        channels=args.channels,
        kernel_size=args.kernel_size,
        act=args.act,
        pooling=args.pooling,
        embed_dim=args.embed_dim,
        num_groups=args.num_groups
    )        
    
    total_input_channels = model.in_channels
    input_data = torch.rand((args.batch_size, args.input_seq_len_s * args.fs, 2 if args.ecg else 1))
    print("\n--- Model Encoder Summary ---")
    summary(model, input_data=[input_data], 
            col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))
    macs, num_params = profile(model, inputs=(input_data,))
    macs, num_params = clever_format([macs, num_params], "%.7f")
    print(f'ResGruNet Encoder has {num_params} params and {macs} MACs.')
    feat_dim = model.embed_dim
    output_dim = args.input_seq_len_s * args.fs if args.sig2sig else 3  # Full waveform or SBP/DBP/MAP
    bp_prediction_head = ResGruNetPredictionHead(feat_dim, output_dim)
    feats = model(input_data)
    print("\n--- Model Prediction Head Summary ---")
    summary(bp_prediction_head, input_data=[feats], 
            col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))
    macs, num_params = profile(bp_prediction_head, inputs=(feats,))
    macs, num_params = clever_format([macs, num_params], "%.7f")
    print(f'ResGruNet Prediction Head has {num_params} params and {macs} MACs.')