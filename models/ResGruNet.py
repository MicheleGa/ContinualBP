import argparse
import torch
from torch import nn
from torchinfo import summary
from thop import profile, clever_format
from component_factory import BPRegressor
           
           
class TemporalResBlock(nn.Module):
    def __init__(self, inCh, outCh):
        super(TemporalResBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=inCh, out_channels=inCh, kernel_size=7, padding=3, groups=inCh)
        self.bn1 = nn.BatchNorm1d(num_features=inCh)
        self.relu1 = nn.ReLU()

        self.l1 = nn.Linear(inCh, outCh)
        self.relu2 = nn.ReLU()

        self.l2 = nn.Linear(outCh, outCh)
        self.bn3 = nn.BatchNorm1d(num_features=outCh)

        self.convres = nn.Conv1d(in_channels=inCh, out_channels=outCh, kernel_size=1)
        self.bnres = nn.BatchNorm1d(num_features=outCh)
    
        self.reluout = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)

        y = y.permute(0, 2, 1)
        y = self.l1(y)
        y = self.relu2(y)
        y = self.l2(y)
        y = y.permute(0, 2, 1)
        y = self.bn3(y)

        res = self.convres(x)
        res = self.bnres(res)

        out = self.reluout(y + res)
        out = self.pool(out)

        return out


class TemporalBlock(nn.Module):
    def __init__(self):
        super(TemporalBlock, self).__init__()

        self.tresblock0 = TemporalResBlock(inCh=1, outCh=32)
        self.tresblock1 = TemporalResBlock(inCh=32, outCh=64)
        self.tresblock2 = TemporalResBlock(inCh=64, outCh=128)
        self.tresblock3 = TemporalResBlock(inCh=128, outCh=128)

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
        super(GenSignalFeatures, self).__init__()

        self.temporalblock = TemporalBlock()

        self.t_gru = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)
        self.t_bn1 = nn.BatchNorm1d(num_features=78) # Doubled size to handle the 10s window with 125Hz fs

    def forward(self, signal):

        feature_t = self.temporalblock(signal)

        (o0, h0) = self.t_gru(feature_t)
        y = torch.squeeze(o0)
        y = self.t_bn1(y)

        return y
    

class ResGruNet(nn.Module):
    r"""
    Source: https://github.com/easyfan327/FewShotBP/blob/main/models/PPGECGNet_V0e2x1b.py
    """
    def __init__(self, 
                 ecg=False,  
                 fs=125, 
                 input_seq_len_s=10,
                 embed_dim=256):
        super(ResGruNet, self).__init__()
        
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
    parser = argparse.ArgumentParser(description="ResGruNet summary")

    parser.add_argument('--batch_size', default=16, type=int, help='batch size for training/inference (default: 32)')
    parser.add_argument('--ecg', action=argparse.BooleanOptionalAction, default=False, help='enable ECG signal processing mode')
    parser.add_argument('--fs', default=125, type=int, help='sampling frequency in Hz (default: 125)')
    parser.add_argument('--input_seq_len_s', default=10, type=int, help='input sequence length in seconds (default: 10)')
    parser.add_argument('--embed_dim', default=128, type=int, help='embedding dimension (default: 128)')
    parser.add_argument('--use_lora', action=argparse.BooleanOptionalAction, default=False, help='enable LoRA (Low-Rank Adaptation) fine-tuning')
    parser.add_argument('--lora_r', default=8, type=int, help='LoRA rank parameter (default: 8)')
    parser.add_argument('--lora_alpha', default=16, type=float, help='LoRA alpha scaling parameter (default: 16)')
    parser.add_argument('--lora_dropout', default=0.1, type=float, help='LoRA dropout rate (default: 0.1)')
    
    return parser.parse_args()
        
        
if __name__ == "__main__":
    args = parseargs()  

    encoder = ResGruNet(
        ecg=args.ecg, 
        fs=args.fs, 
        input_seq_len_s=args.input_seq_len_s, 
        embed_dim=args.embed_dim
        )
    input_tensor = torch.rand((args.batch_size, args.input_seq_len_s * args.fs, 2 if args.ecg else 1))
    print("\n--- Model Encoder Summary ---")
    summary(encoder, input_data=[input_tensor], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))
    macs, num_params = profile(encoder, inputs=(input_tensor,))
    macs, num_params = clever_format([macs, num_params], "%.7f")
    print(f'ResGruNet Encoder has {num_params} params and {macs} MACs.')
    
    prediction_head = BPRegressor(2 * args.embed_dim if args.ecg else args.embed_dim, 3)
    input_tensor = torch.rand((args.batch_size, 2 * args.embed_dim if args.ecg else args.embed_dim))
    print("\n--- Model Prediction Head Summary ---")
    summary(prediction_head, input_data=[input_tensor], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))
    macs, num_params = profile(prediction_head, inputs=(input_tensor,))
    macs, num_params = clever_format([macs, num_params], "%.7f")
    print(f'ResGruNet Prediction Head has {num_params} params and {macs} MACs.')
    