import argparse
from functools import partial
from distutils.util import strtobool

import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from thop import profile, clever_format


class TemporalResBlock(nn.Module):
    def __init__(self, inCh, outCh, kernel_size=7, act='relu', pooling='avg'):
        super(TemporalResBlock, self).__init__()
        
        padding = (kernel_size - 1) // 2 
        self.conv1 = nn.Conv1d(in_channels=inCh, out_channels=inCh, kernel_size=kernel_size, padding=padding, groups=inCh)
        self.bn1 = nn.BatchNorm1d(num_features=inCh)
        
        if act == 'relu':
            self.relu1 = nn.ReLU()
        elif act == 'leaky_relu':
            self.relu1 = nn.LeakyReLU()
        else:
            raise ValueError('Invalid pooling type')    
        
        self.l1 = nn.Linear(inCh, outCh)
        
        if act == 'relu':
            self.relu2 = nn.ReLU()
        elif act == 'leaky_relu':
            self.relu2 = nn.LeakyReLU()
        else:
            raise ValueError('Invalid activation type')
        
        self.l2 = nn.Linear(outCh, outCh)
        self.bn3 = nn.BatchNorm1d(num_features=outCh)

        self.convres = nn.Conv1d(in_channels=inCh, out_channels=outCh, kernel_size=1)
        self.bnres = nn.BatchNorm1d(num_features=outCh)

        if act == 'relu':
            self.reluout = nn.ReLU()
        elif act == 'leaky_relu':
            self.reluout = nn.LeakyReLU()
        else:
            raise ValueError('Invalid activation type')
        
        if pooling == 'max':
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        elif pooling == 'avg':
            self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        else:
            raise ValueError('Invalid pooling type')
            
    def forward(self, x):
        # x -> [batch_size, channels, seq_len]
        
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
    def __init__(self, channels=[1, 32, 64, 128], kernel_size=7, act='relu', pooling='avg'):
        super(TemporalBlock, self).__init__()

        self.resblocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.resblocks.append(TemporalResBlock(inCh=channels[i], outCh=channels[i+1], kernel_size=kernel_size, act=act, pooling=pooling))
            self.resblocks.append(TemporalResBlock(inCh=channels[i+1], outCh=channels[i+1], kernel_size=kernel_size, act=act, pooling=pooling))
        
    def forward(self, x):
        # x -> [batch_size, seq_len, channels]
        
        x = x.unsqueeze(-1) if len(x.shape) == 2 else x
        y = x.permute(0, 2, 1) # x -> [batch_size, channels, seq_len]
        
        for resblock in self.resblocks:
            y = resblock(y)
        
        return y.permute(0, 2, 1) # x -> [batch_size, seq_len, channels]
    

class ResGRUNet(nn.Module):

    def __init__(self, 
                 ecg=False, 
                 resp=False, 
                 ppg_derivatives=False,
                 ppg_emd=False, 
                 ppg_freqs=False, 
                 channels='1, 32, 64, 128', 
                 kernel_size=7, 
                 act='leaky_relu', 
                 pooling='max', 
                 proj_head_dim=128,
                 input_seq_len=625, 
                 return_embedding=False,
                 set_tunable_params='all'):
        super(ResGRUNet, self).__init__()
        
        # Input Data Setup
        self.input_seq_len = input_seq_len
        self.ecg = ecg
        self.resp = resp
        self.ppg_derivatives = ppg_derivatives
        self.ppg_emd = ppg_emd
        self.ppg_freqs = ppg_freqs
        self.return_embedding = return_embedding
                
        # Feature Extractor        
        # CNN for each modality
        ppg_in_channels, ecg_in_channels, resp_in_channels = self.get_input_channels()
        channels = [int(ch) for ch in channels.split(',')]
        
        channels[0] = ppg_in_channels
        self.ppg_feature_gen = TemporalBlock(channels=channels, kernel_size=kernel_size, act=act, pooling=pooling)
        
        if self.ecg:
            channels[0] = ecg_in_channels
            self.ecg_feature_gen = TemporalBlock(channels=channels, kernel_size=kernel_size, act=act, pooling=pooling)
            
            if self.resp:
                channels[0] = resp_in_channels
                self.resp_feature_gen = TemporalBlock(channels=channels, kernel_size=kernel_size, act=act, pooling=pooling)
        
        # GRU combining the modalities
        # PPG is always present while ECG/RESP may not int(ecg) == 0 when false, same for resp of course
        self.t_gru = nn.GRU(input_size=channels[-1] * (1 + int(self.ecg) + int(self.resp)), hidden_size=channels[-1], batch_first=True)
        self.t_bn = nn.BatchNorm1d(num_features=channels[-1])
        
        # Combine Features
        self.flatten = nn.Flatten()        
        
        # Projection Head
        seq_len_after_conv = input_seq_len // 2**(2*(len(channels) - 1)) # -1 because there two blocks that does not change the number of channels but downsample the input
        self.projection_head = nn.Linear(seq_len_after_conv * channels[-1], proj_head_dim)
        
        # Regressor
        self.bn_out = nn.BatchNorm1d(proj_head_dim)
        self.act_out = nn.ReLU()
        self.dropout_out = nn.Dropout(p=0.25)
        self.fc_out = nn.Linear(proj_head_dim, 2) # SBP/DBP linear regression                
        
        # Kaiming initialization that should work fine with the z-score preprocessing
        self.init_params() 
        
        # Freeze/Tune model parameters
        self.set_tunable_layers(set_tunable_params)

    def forward(self, x):

        # Add channel dimension if missing (can happen with only PPG without ECG or derivatives or EMD)
        x = x.unsqueeze(-1) if len(x.shape) == 2 else x # x -> (batch_size, seq_len, channels)
        
        # CNN for each modality
        # x -> [batch_size, seq_len, channels]
        if self.ecg and not self.resp:
            ppg_t = self.ppg_feature_gen(x[:,:,:-1])
            ecg_t = self.ecg_feature_gen(x[:,:,-1])
            
            feature = torch.cat((
                ppg_t,
                ecg_t
                ), 
                dim=-1)
        elif self.ecg and self.resp:
            ppg_t = self.ppg_feature_gen(x[:,:,:-2])
            ecg_t = self.ecg_feature_gen(x[:,:,-2])
            resp_t = self.resp_feature_gen(x[:,:,-1])        

            feature = torch.cat((
                ppg_t,
                ecg_t,
                resp_t
                ), 
                dim=-1)
        else:
            ppg_t = self.ppg_feature_gen(x)
            feature = ppg_t
        
        # GRU combining the modalities
        # x -> [batch_size, seq_len, channels]
        (o0, _) = self.t_gru(feature)   
        feature = self.t_bn(o0.permute(0, 2, 1)) .permute(0, 2, 1) # x-> [batch_size, channels, seq_len], x-> [batch_size, seq_len, channels]
        
        # Flatten features
        flattened_features = self.flatten(feature)
        
        # Projection head
        embedding = self.projection_head(flattened_features)
        
        # Final regression
        out = self.fc_out(self.dropout_out(self.act_out(self.bn_out(embedding))))

        if self.return_embedding:
            return out, embedding
        else:
            return out
    
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
        print(f'ResNet has {num_params} params and {macs} macs.')
    
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
    parser = argparse.ArgumentParser(description="PPGECGNet_V0e2x1b summary, # params and MACS")

    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load only ecg or not')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to load also resp with ecg or not')
    parser.add_argument('--ppg_derivatives', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg derivatives or not')
    parser.add_argument('--ppg_emd', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg imfs or not')
    parser.add_argument('--ppg_freqs', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg freqs or not')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
    parser.add_argument('--channels', default='1, 32, 64, 128', type=str, help='channels produced by the convolutional blocks')
    parser.add_argument('--proj_head_dim', default=128, type=int, help='dimension of the projection head after the feture extractor') 
    parser.add_argument('--kernel_size', default=7, type=int, help='convolutional layer kernel size')
    parser.add_argument('--act', default='leaky_relu', type=str, help='which activation to use (ReLU or LeakyReLU)')
    parser.add_argument('--pooling', default='avg', type=str, help='which poolng to use (average or max)')
    parser.add_argument('--set_tunable_params', default='all', type=str, help='which model parameters to tune (all, only regressor, only encoder, etc.)')
    parser.add_argument('--return_embedding', default='False', type=lambda x: bool(strtobool(x)), help='whether to return the model embedding before the regressor or not')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    global args
    args = parseargs()  
    
    # RESP is loaded only if ECG is also loaded
    if args.resp and not args.ecg:
        raise ValueError('RESP can be loaded only along with ECG')  
    
    # PPG derivatives/PPG EMD/PPG freqs are loaded only if ecg (and optionally resp) are not present
    if (args.ppg_derivatives or args.ppg_emd or args.ppg_freqs) and args.ecg: 
        raise ValueError('PPG derivatives/emd/scalogram can be loaded only without ECG (and optionally RESP)')  
    
    net = ResGRUNet(
        ecg=args.ecg,
        resp=args.resp,
        ppg_derivatives=args.ppg_derivatives,
        ppg_emd=args.ppg_emd,
        ppg_freqs=args.ppg_freqs,
        channels=args.channels,
        kernel_size=args.kernel_size,
        act=args.act,
        pooling=args.pooling,
        proj_head_dim=args.proj_head_dim,
        input_seq_len=int(args.input_seq_len_s * args.fs),
        return_embedding=args.return_embedding,
        set_tunable_params=args.set_tunable_params
        )        
    net.print_summary()
    





        





