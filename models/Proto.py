import argparse
import torch
from torch import nn
from torchinfo import summary
from thop import profile, clever_format
from component_factory import BPRegressor


class TemporalResBlock(nn.Module):
    def __init__(self, inCh, outCh):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=inCh, out_channels=inCh, kernel_size=7, padding=3, groups=inCh)
        self.gn1 = nn.GroupNorm(num_groups=min(8, inCh), num_channels=inCh)
        self.relu1 = nn.ReLU()

        self.l1 = nn.Linear(inCh, outCh)
        self.relu2 = nn.ReLU()

        self.l2 = nn.Linear(outCh, outCh)
        self.gn3 = nn.GroupNorm(num_groups=min(8, outCh), num_channels=outCh)

        self.convres = nn.Conv1d(inCh, outCh, kernel_size=1)
        self.gnres = nn.GroupNorm(num_groups=min(8, outCh), num_channels=outCh)
    
        self.reluout = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.gn1(y)
        y = self.relu1(y)

        y = y.permute(0, 2, 1)
        y = self.l1(y)
        y = self.relu2(y)
        y = self.l2(y)
        y = y.permute(0, 2, 1)
        y = self.gn3(y)

        res = self.convres(x)
        res = self.gnres(res)

        out = self.reluout(y + res)
        out = self.pool(out)

        return out


class TemporalBlock(nn.Module):
    def __init__(self):
        super(TemporalBlock, self).__init__()

        self.tresblock0 = TemporalResBlock(1, 32)
        self.tresblock1 = TemporalResBlock(32, 64)
        self.tresblock2 = TemporalResBlock(64, 128)
        self.tresblock3 = TemporalResBlock(128, 128)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=-1).permute(0, 2, 1)
        y = self.tresblock0(x)
        y = self.tresblock1(y)
        y = self.tresblock2(y)
        y = self.tresblock3(y)
        y = y.permute(0, 2, 1)
        return y
   
class GenSignalFeatures(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        self.temporalblock = TemporalBlock()

        self.t_gru = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)
        self.t_ln = nn.LayerNorm(embed_dim)

    def forward(self, signal):

        feature_t = self.temporalblock(signal)          # shape [B, T, C]

        o0, h0 = self.t_gru(feature_t)                  # shape [B, T, C]
        y = self.t_ln(o0)                               # LayerNorm on features

        return y

class Proto(nn.Module):
    r"""
    ResGruNet with Group and Layer Normalization for streaming CL
    """
    def __init__(self, 
                 ecg=False,  
                 fs=125, 
                 input_seq_len_s=10,
                 embed_dim=256):
        super(Proto, self).__init__()
        self.input_seq_len = input_seq_len_s * fs
        self.ecg = ecg
        self.embed_dim = 2 * embed_dim if self.ecg else embed_dim
        self.in_channels = 2 if self.ecg else 1
                        
        self.ppg_feature_gen = GenSignalFeatures(embed_dim=embed_dim)
        
        if self.ecg:
            self.ecg_feature_gen = GenSignalFeatures(embed_dim=embed_dim)
            
    def forward(self, x):        
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        if len(x.shape) != 3:
            raise ValueError("Input tensor must have shape [B, T, C] or [B, C, T]")
        
        if (x.shape[1] == self.input_seq_len and x.shape[2] == self.in_channels):
            x = x.permute(0, 2, 1)
        
        if (x.shape[1] != self.in_channels or x.shape[2] != self.input_seq_len):
            raise ValueError("Input tensor dimension mismatch")

        ppg_feats = self.ppg_feature_gen(x[:,0,:])
        if self.ecg:
            ecg_feats = self.ecg_feature_gen(x[:,1,:])
            features = torch.cat((
                torch.mean(ppg_feats, dim=1), 
                torch.mean(ecg_feats, dim=1)
                ), 
                dim=-1)
        else:
            features = torch.mean(ppg_feats, dim=1)
        
        return features # features shape: [B, feat_dim]
    

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
    
    # Instantiate encoder and profile its MACs/bytes on a dummy input
    encoder = Proto(args.ecg, args.fs, args.input_seq_len_s, args.embed_dim)
    x = torch.rand((args.batch_size, args.input_seq_len_s * args.fs, 2 if args.ecg else 1))
    print("\n--- Encoder Summary ---")
    summary(encoder, input_data=[x], col_names=("input_size","output_size","num_params","mult_adds"))
    macs, params = profile(encoder, inputs=(x,))
    macs, params = clever_format([macs, params], "%.7f")
    print(f'Proto Encoder has {params} params and {macs} MACs.')
    
    # Instantiate prediction head and profile its MACs/bytes on a dummy input
    head = BPRegressor(encoder.embed_dim, 3)
    y = torch.rand((args.batch_size, encoder.embed_dim))
    print("\n--- Head Summary ---")
    summary(head, input_data=[y], col_names=("input_size","output_size","num_params","mult_adds"))
    macs, params = profile(head, inputs=(y,))
    macs, params = clever_format([macs, params], "%.7f")
    print(f'Proto Head has {params} params and {macs} MACs.')