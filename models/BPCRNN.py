import argparse
from functools import partial
from distutils.util import strtobool

import torch
from torch import nn
import torchaudio.transforms as T
import torch.nn.functional as F
from torchinfo import summary
from thop import profile, clever_format


class RegressionHead(nn.Module):
    def __init__(self, feature_extractor_out_size, units=[64, 2], act='relu', dropout_prob=0.25):
        super(RegressionHead, self).__init__()
        self.flatten = nn.Flatten()
        
        layers = []
        input_size = feature_extractor_out_size
        
        for i, output_size in enumerate(units):
            layers.append(nn.Linear(input_size, output_size))
            
            # Apply batch_norm - act - dropout only if it's not the last layer
            if dropout_prob > 0 and i < len(units) - 1:
                layers.append(nn.BatchNorm1d(output_size))
            
                if act == 'relu':
                    layers.append(nn.ReLU())
                elif act == 'leaky_relu':
                    layers.append(nn.LeakyReLU())
                else:
                    raise ValueError(f"Unsupported activation: {act}")
        
                layers.append(nn.Dropout(dropout_prob))
                
            input_size = output_size
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

class BPCRNNFeatureExtractor(nn.Module):
    def __init__(self, ppg_in_channels, ecg_in_channels, channels, kernel_size, gru_hidden_size):
        super(BPCRNNFeatureExtractor, self).__init__()
        padding = (kernel_size - 1) // 2 
        self.conv1 = nn.Conv1d(ppg_in_channels + ecg_in_channels, channels, kernel_size=kernel_size, padding=padding)  # Padding to maintain length
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.relu3 = nn.ReLU()
        
        self.gru = nn.GRU(channels * 2, gru_hidden_size, batch_first=True)  # Input: concatenation of conv1 and conv3 outputs

    def forward(self, x):
        x = x.unsqueeze(-1) if len(x.shape) == 2 else x # x -> [batch, seq, channels]
        x = x.permute(0, 2, 1) # x -> [batch, channels, seq]

        # Convolutional layers
        conv1_out = self.relu1(self.conv1(x))
        conv2_out = self.relu2(self.conv2(conv1_out))
        conv3_out = self.relu3(self.conv3(conv2_out))
        
        # Concatenate conv1 and conv3 outputs
        gru_input = torch.cat((conv1_out, conv3_out), dim=1)  # Concatenate along the channel dimension
        
        # Transpose for GRU input (batch, seq_len, input_size)
        gru_input = gru_input.transpose(1, 2)
        
        # GRU layer
        gru_out, _ = self.gru(gru_input)
        
        return gru_out


class BPCRNNDecoder(nn.Module):
    def __init__(self, gru_hidden_size, channels, kernel_size, ppg_in_channels, ecg_in_channels):
        super(BPCRNNDecoder, self).__init__()
        
        padding = (kernel_size - 1) // 2 
        self.deconv1 = nn.ConvTranspose1d(gru_hidden_size, channels, kernel_size=kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        
        self.deconv2 = nn.ConvTranspose1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.relu2 = nn.ReLU()
        
        self.deconv3 = nn.ConvTranspose1d(channels, ppg_in_channels + ecg_in_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        # Transpose back for deconvolutional layers (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # Deconvolutional layers for reconstruction
        deconv1_out = self.relu1(self.deconv1(x))
        deconv2_out = self.relu2(self.deconv2(deconv1_out))
        reconstruction = self.deconv3(deconv2_out)
        
        return reconstruction
    

class BPCRNN(nn.Module):
    def __init__(self, only_ppg=False, ppg_derivatives=False, ppg_emd=False, gru_hidden_size=25, channels=50, kernel_size=7, input_seq_len=125, supervised=True):    
        super(BPCRNN, self).__init__()
        # From https://ieeexplore.ieee.org/document/9445687
        
        self.only_ppg = only_ppg
        self.ppg_derivatives = ppg_derivatives
        self.ppg_emd = ppg_emd
        self.supervised = supervised
        self.input_seq_len = input_seq_len
        ppg_in_channels, ecg_in_channels = self.get_input_channels()
        
        self.feature_extractor = BPCRNNFeatureExtractor(ppg_in_channels, ecg_in_channels, channels, kernel_size, gru_hidden_size)
        
        if not self.supervised:
            self.decoder = BPCRNNDecoder(gru_hidden_size, channels, kernel_size, ppg_in_channels, ecg_in_channels)
        else:
            self.projection_head = RegressionHead(gru_hidden_size * int(input_seq_len)) 
        
        self._init_params()

    def forward(self, x):
        # Encoder
        gru_out = self.feature_extractor(x)
        
        if not self.supervised:
            # Decoder for unsupervised learning
            output = self.decoder(gru_out)
        else:
            # Regressor for supervised learning
            output = self.projection_head(gru_out)
            
        return output
    
    def _init_params(self):
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_input_channels(self):
        if self.only_ppg:
            if self.ppg_derivatives:
                ppg_inchannels = 3  # PPG + PPG' + PPG''
            elif self.ppg_emd:
                ppg_inchannels = 4  # PPG_IMF0 + PPG_IMF1 + PPG_IMF2 + PPG_IMF3
            else:
                ppg_inchannels = 1  # PPG
            ecg_inchannels = 0 #No ECG when only_ppg is True
        else:  # Not only_ppg (i.e., PPG + ECG)
            if self.ppg_derivatives:
                ppg_inchannels = 3  # PPG + PPG' + PPG'' + ECG
            elif self.ppg_emd:
                ppg_inchannels = 4  # PPG_IMF0 + PPG_IMF1 + PPG_IMF2 + PPG_IMF3 + ECG
            else:
                ppg_inchannels = 1  # PPG + ECG
            ecg_inchannels = 1
        
        if ppg_inchannels == 0 and ecg_inchannels == 0: # Check for invalid cases
            raise ValueError('Invalid input signals combination ...')

        return ppg_inchannels, ecg_inchannels
    
    def print_summary(self, batch_size=256):
        in_channels = self.get_input_channels()
        input = torch.rand((batch_size, self.input_seq_len, in_channels[0] + in_channels[1]))
        
        summary(self, input_data=[input], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))

        macs, num_params = profile(self, inputs=(input,))
        macs, num_params = clever_format([macs, num_params], "%.7f")
        print(f'BPCRNN has {num_params} params and {macs} macs.')
    
    def requested_input_columns(self) -> list:
        return ['sig']
    
    def set_tunable_layers(self, tune):
        for p in self.parameters():
            p.requires_grad = True
            
        if tune == "all":
            return 
        elif tune == "fully_connected":
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            
        else:
            raise Exception("undefined tune")
    
def parseargs():
    parser = argparse.ArgumentParser(description="PPGECGNet_V0e2x1b summary, # params and MACS")

    parser.add_argument('--only_ppg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load both ecg and ppg or only ppg')
    parser.add_argument('--ppg_derivatives', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg derivatives')
    parser.add_argument('--ppg_emd', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg derivatives')
    parser.add_argument('--supervised', default='True', type=lambda x: bool(strtobool(x)), help='whether to pretrain the model or not')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
    parser.add_argument('--gru_hidden_size', default=25, type=int, help='GRU size')
    parser.add_argument('--kernel_size', default=7, type=int, help='convolutional layer kernel size')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    global args
    args = parseargs()    
    
    net = BPCRNN(
        only_ppg=args.only_ppg, 
        ppg_derivatives=args.ppg_derivatives, 
        ppg_emd=args.ppg_emd,
        gru_hidden_size=args.gru_hidden_size, 
        kernel_size=args.kernel_size,
        input_seq_len=int(args.input_seq_len_s * args.fs),
        supervised=args.supervised
        )        
    net.print_summary()













