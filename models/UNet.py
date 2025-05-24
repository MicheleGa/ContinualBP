import argparse
from distutils.util import strtobool
import torch
import torch.nn as nn
from torchinfo import summary
from thop import profile, clever_format


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    

class UNet(nn.Module):
    def __init__(self,  only_ppg=False, ppg_derivatives=False, ppg_emd=False, channels='1, 32, 64, 128', kernel_size=7, input_seq_len=125):
        super(UNet, self).__init__()
        # TO-DO: what to do with ECG?
        
        self.only_ppg = only_ppg
        self.ppg_derivatives = ppg_derivatives
        self.ppg_emd = ppg_emd
        self.input_seq_len = input_seq_len
        ppg_in_channels, ecg_in_channels = self.get_input_channels()
        channels = [int(ch) for ch in channels.split(',')]
        gru_hidden_size = channels[3]
        
        padding = (kernel_size - 1) // 2 
        self.conv1 = nn.Conv1d(ppg_in_channels + ecg_in_channels, channels[1], kernel_size=kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(channels[1], channels[2], kernel_size=kernel_size, padding=padding)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(channels[2], channels[3], kernel_size=kernel_size, padding=padding)
        self.relu3 = nn.ReLU()
        
        self.gru = nn.GRU(channels[3], gru_hidden_size, batch_first=True)
        
        self.att1 = AttentionBlock(F_g=gru_hidden_size, F_l=channels[3], F_int=channels[2])
        self.deconv1 = nn.ConvTranspose1d(gru_hidden_size + channels[3], channels[2], kernel_size=kernel_size, padding=padding)
        self.relu4 = nn.ReLU()
        
        self.att2 = AttentionBlock(F_g=channels[2], F_l=channels[2], F_int=channels[1])
        self.deconv2 = nn.ConvTranspose1d(channels[2] + channels[2], channels[1], kernel_size=kernel_size, padding=padding)
        self.relu5 = nn.ReLU()
        
        self.att3 = AttentionBlock(F_g=channels[1], F_l=channels[1], F_int=channels[0])
        self.deconv3 = nn.ConvTranspose1d(channels[1] + channels[1], ppg_in_channels + ecg_in_channels, kernel_size=kernel_size, padding=padding)
                
        self._init_params()

    def forward(self, x):
        x = x.unsqueeze(-1) if len(x.shape) == 2 else x
        x = x.permute(0, 2, 1)
        
        # Encoder
        conv1_out = self.relu1(self.conv1(x))
        conv2_out = self.relu2(self.conv2(conv1_out))
        conv3_out = self.relu3(self.conv3(conv2_out))
        
        gru_input = conv3_out.transpose(1, 2)
        gru_out, _ = self.gru(gru_input)
        gru_out = gru_out.transpose(1, 2)
        
        # Decoder with attention and skip connections
        att1_out = self.att1(g=gru_out, x=conv3_out)
        deconv1_out = self.relu4(self.deconv1(torch.cat((gru_out, att1_out), dim=1)))
        
        att2_out = self.att2(g=deconv1_out, x=conv2_out)
        deconv2_out = self.relu5(self.deconv2(torch.cat((deconv1_out, att2_out), dim=1)))
        
        att3_out = self.att3(g=deconv2_out, x=conv1_out)
        reconstruction = self.deconv3(torch.cat((deconv2_out, att3_out), dim=1))
        
        return reconstruction
    
    def _init_params(self):
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
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
        assert ecg_inchannels == 0, 'ECG is not supported yet'
        if ppg_inchannels == 0 and ecg_inchannels == 0: # Check for invalid cases
            raise ValueError('Invalid input signals combination ...')

        return ppg_inchannels, ecg_inchannels
    
    def print_summary(self, batch_size=256):
        in_channels = self.get_input_channels()
        input = torch.rand((batch_size, self.input_seq_len, in_channels[0] + in_channels[1]))
        
        summary(self, input_data=[input], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))

        macs, num_params = profile(self, inputs=(input,))
        macs, num_params = clever_format([macs, num_params], "%.7f")
        print(f'UNet has {num_params} params and {macs} macs.')
        
        
def parseargs():
    parser = argparse.ArgumentParser(description="PhysioFormer summary, # params and MACS")

    parser.add_argument('--only_ppg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load both ecg and ppg or only ppg')
    parser.add_argument('--ppg_derivatives', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg derivatives')
    parser.add_argument('--ppg_emd', default='False', type=lambda x: bool(strtobool(x)), help='whether to load ppg derivatives')
    parser.add_argument('--input_seq_len', default=625, type=int, help='length of an input sample')
    parser.add_argument('--channels', default='1, 16, 32, 48', type=str, help='channels produced by the convolutional blocks')
    parser.add_argument('--kernel_size', default=7, type=int, help='convolutional layer kernel size')
    parser.add_argument('--regressor_seq_len', default=0, type=int, help='if positive, the CNN output features will be flattened rather thann averaged ovevr the sequence dimension')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    global args
    args = parseargs()    
    
    net = UNet(
        only_ppg=args.only_ppg, 
        ppg_derivatives=args.ppg_derivatives, 
        ppg_emd=args.ppg_emd,
        channels=args.channels,
        kernel_size=args.kernel_size,
        input_seq_len=args.input_seq_len
        )        
    net.print_summary()