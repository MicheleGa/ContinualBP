import argparse
from distutils.util import strtobool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from thop import profile, clever_format


class GRU(nn.Module):
    def __init__(self,  
                 ecg=False,
                 resp=False,
                 sig2sig=False,
                 ppg_derivatives=False,
                 ppg_emd=False,
                 ppg_freqs=False,
                 fs=125,
                 input_seq_len_s=5,
                 hidden_dim=128,
                 num_layers=2,
                 bidirectional=True,
                 set_tunable_params='all',
                 return_embedding=False):
        super(GRU, self).__init__()
        
        # Input Data Setup
        self.input_seq_len = input_seq_len_s * fs
        self.ecg = ecg
        self.resp = resp
        self.sig2sig = sig2sig
        self.ppg_derivatives = ppg_derivatives
        self.ppg_emd = ppg_emd
        self.ppg_freqs = ppg_freqs
        self.return_embedding = return_embedding
        
        # Architecture Setup
        ppg_in_channels, ecg_in_channels, resp_in_channels = self.get_input_channels()
        self.in_channels = ppg_in_channels + ecg_in_channels + resp_in_channels
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # GRU layer
        self.gru = nn.GRU(
            input_size=self.in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Fully connected layer to map hidden states to output
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)
 
        # Kaiming initialization that should work fine with the z-score preprocessing
        self.init_params() 
        
        # Freeze/Tune model parameters
        #self.set_tunable_layers(set_tunable_params)

    def forward(self, x):
        
        x = x.unsqueeze(-1) if len(x.shape) == 2 else x # x -> (batch_size, seq_len, channels)
        # Initialize hidden state
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            x.size(0),
            self.hidden_dim,
            device=x.device
        )

        # Pass through GRU
        gru_out, _ = self.gru(x, h0)  # gru_out: [batch, sequence length, hidden_dim * num_directions]

        # Pass through the fully connected layer
        output = self.fc(gru_out)  # output: [batch, sequence length, 1]

        return output.squeeze(-1)
    
    def init_params(self):
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_input_channels(self):
        if self.ecg:
            ecg_in_channels = 1
        else:
            ecg_in_channels = 0
        
        if self.resp:
            resp_in_channels = 1
        else:
            resp_in_channels = 0
        
        if self.ppg_derivatives:
            ppg_in_channels = 3  # PPG + PPG' + PPG''
        elif self.ppg_emd:
            ppg_in_channels = 4  # PPG_IMF0 + PPG_IMF1 + PPG_IMF2 + PPG_IMF3
        elif self.ppg_freqs:
            ppg_in_channels = 16  # PPG_IMF0 + PPG_IMF1 + PPG_IMF2 + PPG_IMF3  
        else:
            ppg_in_channels = 1  # PPG
            
        return ppg_in_channels, ecg_in_channels, resp_in_channels
    
    def print_summary(self, batch_size=256):
        in_channels = self.get_input_channels()
        input = torch.rand((batch_size, self.input_seq_len, in_channels[0] + in_channels[1] + in_channels[2]))

        summary(self, input_data=[input], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))

        macs, num_params = profile(self, inputs=(input,))
        macs, num_params = clever_format([macs, num_params], "%.7f")
        print(f'UNet has {num_params} params and {macs} macs.')
    
    def set_tunable_layers(self, tune):
        
        # First set all parameters to be trainable
        for p in self.parameters():
            p.requires_grad = True
        
        # Update the whole model
        if tune == "all":
            return 
        # Update only the regressor last linear
        elif tune == "last_linear":
            for p in self.ppg_feature_gen.parameters():
                p.requires_grad = False
            
            if self.ecg:
                for p in self.ecg_feature_gen.parameters():
                    p.requires_grad = False
            
                if self.resp: 
                    for p in self.resp_feature_gen.parameters():
                        p.requires_grad = False
            
            for p in self.t_gru.parameters():
                p.requires_grad = False
                
            for p in self.projection_head.parameters():
                p.requires_grad = False
        else:
            raise Exception("undefined tune")
        
        
def parseargs():
    parser = argparse.ArgumentParser(description="PhysioFormer summary, # params and MACS")

    parser.add_argument('--batch_size', default=128, type=int, help='batch size for the dummy input')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load only ecg or not')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to load also resp with ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--ppg_derivatives', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg derivatives or not')
    parser.add_argument('--ppg_emd', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg imfs or not')
    parser.add_argument('--ppg_freqs', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg freqs or not')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
    parser.add_argument('--hidden_dim', default=128, type=int, help='GRU hidden dimension size')
    parser.add_argument('--num_layers', default=2, type=int, help='number of GRU layers')
    parser.add_argument('--bidirectional', default=True, type=lambda x: bool(strtobool(x)), help='whether to use bidirectional GRU or not')
    parser.add_argument('--set_tunable_params', default='all', type=str, help='which model parameters to tune (all, only regressor, only encoder, etc.)')
    parser.add_argument('--return_embedding', default='False', type=lambda x: bool(strtobool(x)), help='whether to return the model embedding before the regressor or not')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    global args
    args = parseargs()    
    
    net = GRU(
        ecg=args.ecg,
        resp=args.resp,
        sig2sig=args.sig2sig,
        ppg_derivatives=args.ppg_derivatives,
        ppg_emd=args.ppg_emd,
        ppg_freqs=args.ppg_freqs,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        set_tunable_params=args.set_tunable_params,
        return_embedding=args.return_embedding
        )        
    net.print_summary(batch_size=args.batch_size)