import argparse
from functools import partial
from distutils.util import strtobool

import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from thop import profile, clever_format

#! V0 -> V0a
#* adjusted kernelsize and padding in TemporalResBlock
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
    def __init__(self, gru=False):
        super(GenSignalFeatures, self).__init__()

        self.gru = gru
        self.temporalblock = TemporalBlock()

        if self.gru:
            self.t_gru = nn.GRU(input_size=128, hidden_size=128, batch_first=True)
            self.t_bn1 = nn.BatchNorm1d(num_features=39)

    def forward(self, signal):

        feature_t = self.temporalblock(signal)

        if self.gru:
            (o0, h0) = self.t_gru(feature_t)
            y = torch.squeeze(o0)
            y = self.t_bn1(y)
        else:
            y = feature_t

        return y
    

class PPGECGNet_V0e2x1bEncoder(nn.Module):

    def __init__(self, only_ppg=False, gru=True):
        super(PPGECGNet_V0e2x1bEncoder, self).__init__()
        
        self.only_ppg = only_ppg
        
        self.ppg_feature_gen = GenSignalFeatures(gru=gru)
        if not self.only_ppg:
            self.ecg_feature_gen = GenSignalFeatures(gru=gru)

    def forward(self, x):
        x = x['sig']
        
        if not self.only_ppg:
            ppg_t = self.ppg_feature_gen(x[:,:,0])
            ecg_t = self.ecg_feature_gen(x[:,:,1])
        else:
            ppg_t= self.ppg_feature_gen(x)

        if not self.only_ppg:
            features = torch.cat((
                torch.mean(ppg_t, 1),
                torch.mean(ecg_t, 1)
                ), 
                dim=-1)
        else:
            features = torch.mean(ppg_t, 1)
        
        return features
    

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super(ProjectionHead, self).__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)


class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super(BarlowTwinsLoss, self).__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag


class BarlowTwins(nn.Module):
    def __init__(self, only_ppg, gru, encoder_out_dim, batch_size, lambda_coeff=5e-3, z_dim=128):
        super(BarlowTwins, self).__init__()

        self.only_ppg = only_ppg
        self.gru = gru
        self.lambda_coeff = lambda_coeff
        self.z_dim = z_dim
        self.encoder = PPGECGNet_V0e2x1bEncoder(only_ppg=only_ppg, gru=gru)
        
        self.encoder_out_dim = encoder_out_dim
        self.projection_head = ProjectionHead(input_dim=encoder_out_dim, hidden_dim=2 * encoder_out_dim, output_dim=z_dim)
        self.loss_fn = BarlowTwinsLoss(batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim)

        self.batch_size = batch_size

    def forward(self, x):
        return self.encoder(x)

    def generate_features(self, y1, y2):
        z1 = self.projection_head(self.encoder(y1))
        z2 = self.projection_head(self.encoder(y2))
        return z1, z2
        
    def calculate_loss(self, z1, z2):
        return self.loss_fn(z1, z2)
    
    def requested_input_columns(self) -> list:
        return ['sig']

    def print_summmary(self):
        if not self.only_ppg:
            input = { 
                'sig': torch.rand((self.batch_size, 625, 2))
                }
        else:
            input = {
                'sig': torch.rand((self.batch_size, 625))
                }
        summary(self.encoder, input_data=[input], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))

        macs, num_params = profile(self.encoder, inputs=(input,))
        macs, num_params = clever_format([macs, num_params], "%.7f")
        print(f'BarlowTwins encoder has {num_params} params and {macs} macs.')

        enc_output = self.encoder(input)
        summary(self.projection_head, input_data=(enc_output), col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))

        macs, num_params = profile(self.projection_head, inputs=(enc_output,))
        macs, num_params = clever_format([macs, num_params], "%.7f")
        print(f'BarlowTwins projection head has {num_params} params and {macs} macs.')


def parseargs():
    parser = argparse.ArgumentParser(description="BarlowTwins summary, # params and MACS")

    parser.add_argument('--op', default='False', type=lambda x: bool(strtobool(x)), help='whether to load both ecg and ppg or only ppg')
    parser.add_argument('--gru', default='True', type=lambda x: bool(strtobool(x)), help='whether to use RNN or not')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    global args
    args = parseargs()    

    out_dim = 256 if not args.op else 128
    z_dim = 4 * out_dim
    net = BarlowTwins(only_ppg=args.op, gru=args.gru, encoder_out_dim=out_dim, batch_size=256, z_dim=out_dim)
    net.print_summmary()
        
    





        





