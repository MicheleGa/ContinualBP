import argparse
from distutils.util import strtobool
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from thop import profile, clever_format


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_groups=8): # Added num_groups
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        
        # Changed from BatchNorm1d to GroupNorm with conditional groups
        groups_gn1 = num_groups if out_channels >= num_groups else 1
        self.gn1 = nn.GroupNorm(groups_gn1, out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1) # Stride 1 for second conv

        # Changed from BatchNorm1d to GroupNorm in shortcut with conditional groups
        groups_shortcut = num_groups if out_channels >= num_groups else 1
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.GroupNorm(groups_shortcut, out_channels)
        )
        
    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out) # Changed from bn1 to gn1
        out = self.relu(out)
        out = self.conv2(out)
        
        out += identity 
        
        return out
    
class BottleneckBlock1D(nn.Module):
    def __init__(self, channels, stride=1, num_groups=8): # Added num_groups
        super(BottleneckBlock1D, self).__init__()
        
        # Changed from BatchNorm1d to GroupNorm with conditional groups
        groups_gn1 = num_groups if channels >= num_groups else 1
        self.gn1 = nn.GroupNorm(groups_gn1, channels)
        
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, stride=1)

        # Changed from BatchNorm1d to GroupNorm with conditional groups
        groups_gn2 = num_groups if channels >= num_groups else 1
        self.gn2 = nn.GroupNorm(groups_gn2, channels)
        
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, stride=1)
        
    def forward(self, x):
        
        out = self.gn1(x) # Changed from bn1 to gn1
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.gn2(out) # Changed from bn2 to gn2
        out = self.relu2(out)
        out = self.conv2(out)
        
        return out

class AttentionGate1D(nn.Module):
    def __init__(self, g_channels, x_channels, inter_channels):
        super(AttentionGate1D, self).__init__()
        # Linear layers to compute queries, keys, and values
        self.query_conv = nn.Conv1d(g_channels, inter_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(x_channels, inter_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(x_channels, x_channels, kernel_size=1)

        # Output convolution to refine the attended feature map
        self.output_conv = nn.Conv1d(x_channels, x_channels, kernel_size=1)

        # Activation function
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from the decoder path (batch_size, g_channels, length)
            x: Skip connection from the encoder path (batch_size, x_channels, length)
        Returns:
            Attended feature map (batch_size, x_channels, length)
        """
        # Compute queries, keys, and values
        query = self.query_conv(g)  # Shape: (batch_size, inter_channels, length)
        key = self.key_conv(x)      # Shape: (batch_size, inter_channels, length)
        value = self.value_conv(x)  # Shape: (batch_size, x_channels, length)

        # Compute attention weights (scaled dot-product attention)
        attention = torch.bmm(query.permute(0, 2, 1), key)  # Shape: (batch_size, length, length)
        attention = attention / (query.size(1) ** 0.5)      # Scale by sqrt(inter_channels)
        attention = self.softmax(attention)                # Normalize along the last dimension

        # Apply attention weights to the value
        attended_features = torch.bmm(value, attention.permute(0, 2, 1))  # Shape: (batch_size, x_channels, length)

        # Refine the attended feature map
        output = self.output_conv(attended_features)  # Shape: (batch_size, x_channels, length)

        return output
    

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim=768, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor([10000.0])) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #  Key Change: No initial unsqueeze and transpose
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x -> (batch_size, seq_len, channels)

        seq_len = x.size(1)  # Get sequence length from input

        pe = self.pe[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)  # Expand for batch

        x = x + pe

        return self.dropout(x)
    

class SelfAttentionBlock1D(nn.Module):
    def __init__(self, seq_len, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(SelfAttentionBlock1D, self).__init__()
        self.pos_encoder = PositionalEncoding(embed_dim=d_model, max_len=seq_len) # Positional encoding
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True) # Set batch_first=True
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src):
        # src shape: (batch_size, channels, length) -> transform to (batch_size, length, channels) for MultiheadAttention
        src = src.permute(0, 2, 1) # (batch_size, length, channels)

        # Add positional encoding
        src_pe = self.pos_encoder(src) # Add PE to original shape then permute back
        
        # Self-attention
        attn_output, _ = self.self_attn(src_pe, src_pe, src_pe)
        src = src + self.dropout1(attn_output) # Add & Norm
        src = self.norm1(src)

        # Feedforward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output) # Add & Norm
        src = self.norm2(src)

        src = src.permute(0, 2, 1) # (batch_size, channels, length)
        return src
    

class UNet(nn.Module):
    def __init__(self,  
                 ecg=False,
                 resp=False,
                 sig2sig=False,
                 fs=125,
                 input_seq_len_s=5,
                 channels='32, 64, 128, 256, 512',
                 kernel_size=3,
                 num_heads_attention=1,
                 dim_feedforward_attention=128
                 ):
        super(UNet, self).__init__()
        
        # Input Data Setup
        self.input_seq_len = input_seq_len_s * fs
        self.ecg = ecg
        self.resp = resp
        self.sig2sig = sig2sig
        
        # Architecture Setup
        self.kernel_size = kernel_size
        self.num_heads_attention = num_heads_attention
        self.dim_feedforward_attention = dim_feedforward_attention
        
        ppg_in_channels, ecg_in_channels, resp_in_channels = self.get_input_channels()
        self.in_channels = ppg_in_channels + ecg_in_channels + resp_in_channels

        self.filters = [int(ch) for ch in channels.split(',')]
        num_layers = len(self.filters)

        # Encoder Path
        self.encoder_blocks = nn.ModuleList()
        # First encoder block with input_channels
        self.encoder_blocks.append(ResidualBlock1D(self.in_channels, self.filters[0], stride=1))
        # Subsequent encoder blocks
        for i in range(num_layers - 1):
            self.encoder_blocks.append(ResidualBlock1D(self.filters[i], self.filters[i+1], stride=2))

        # Bottleneck
        self.bottleneck = BottleneckBlock1D(self.filters[-1], stride=1)

        # Decoder Path
        self.decoder_blocks = nn.ModuleList()
        # Attention Gates
        self.att_gates = nn.ModuleList()

        # Decoder blocks and attention gates
        for i in range(num_layers - 1, 0, -1):
            g_channels = self.filters[i]
            x_channels = self.filters[i-1]
            inter_channels = self.filters[i-1] # Typically inter_channels is smaller, can be x_channels
            
            self.att_gates.append(AttentionGate1D(g_channels, x_channels, inter_channels))
            self.decoder_blocks.append(ResidualBlock1D(g_channels + x_channels, x_channels, stride=1))

        # Final Self-Attention Module
        self.self_attention_stack1 = SelfAttentionBlock1D(
            seq_len=self.input_seq_len,
            d_model=self.filters[0],
            num_heads=num_heads_attention,
            dim_feedforward=dim_feedforward_attention
        )
        self.self_attention_stack2 = SelfAttentionBlock1D(
            seq_len=self.input_seq_len,
            d_model=self.filters[0],
            num_heads=num_heads_attention,
            dim_feedforward=dim_feedforward_attention
        )

        # Output Layer
        self.final_conv = nn.Conv1d(self.filters[0], 1, kernel_size=1, stride=1, padding=0)
 
        # Kaiming initialization that should work fine with the z-score preprocessing
        self.init_params() 
        
    def forward(self, x):
        
        x = x.unsqueeze(-1) if len(x.shape) == 2 else x # x -> (batch_size, seq_len, channels)
        
        # x shape: [batch_size, 625, num_modalities] (if multi-modal)
        # For Conv1d, input needs to be [batch_size, channels, length]
        # Our input is [batch_size, length, channels]
        x = x.permute(0, 2, 1) # -> [batch_size, num_modalities, 625]

        # Encoder
        encoder_features = []
        current_x = x
        for i, encoder_block in enumerate(self.encoder_blocks):
            current_x = encoder_block(current_x)
            encoder_features.append(current_x)
        
        # Bottleneck
        b = self.bottleneck(encoder_features[-1])

        # Decoder with Attention Gates
        decoder_output = b
        num_layers = len(self.filters)
        for i in range(num_layers - 1):
            # i = 0 corresponds to the last decoder block (upsampling from filters[-1] to filters[-2])
            # i = 1 corresponds to upsampling from filters[-2] to filters[-3], and so on.
            # encoder_features[num_layers - 2 - i] gives the correct skip connection.
            
            skip_connection_idx = num_layers - 2 - i
            skip_connection = encoder_features[skip_connection_idx]

            g = F.interpolate(decoder_output, size=skip_connection.size(2), mode='linear', align_corners=True)
            att_skip = self.att_gates[i](g, skip_connection)
            decoder_output = self.decoder_blocks[i](torch.cat([g, att_skip], dim=1))

        # Final Self-Attention Module
        sa_output = self.self_attention_stack1(decoder_output)
        sa_output = self.self_attention_stack2(sa_output)

        # Output Layer
        output = self.final_conv(sa_output)

        # Permute back to [batch_size, length, channels] for consistency with input
        return output.permute(0, 2, 1).squeeze(-1)
    
    def init_params(self):
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.GroupNorm): # Changed from BatchNorm1d to GroupNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_input_channels(self):
        # PPG must always be present
        ppg_in_channels = 1  
        ecg_in_channels = 1 if self.ecg else 0
        resp_in_channels = 1 if self.resp else 0
        return ppg_in_channels, ecg_in_channels, resp_in_channels
    
    def print_summary(self, batch_size=256):
        in_channels = self.get_input_channels()
        input = torch.rand((batch_size, self.input_seq_len, in_channels[0] + in_channels[1] + in_channels[2]))

        summary(self, input_data=[input], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))

        macs, num_params = profile(self, inputs=(input,))
        macs, num_params = clever_format([macs, num_params], "%.7f")
        print(f'UNet has {num_params} params and {macs} macs.')
        
        
def parseargs():
    parser = argparse.ArgumentParser(description="UNet summary, # params and MACS")

    parser.add_argument('--batch_size', default=128, type=int, help='batch size for the dummy input')
    parser.add_argument('--ecg', default='False', type=lambda x: bool(strtobool(x)), help='whether to load only ecg or not')
    parser.add_argument('--resp', default='False', type=lambda x: bool(strtobool(x)), help='whether to load also resp with ecg or not')
    parser.add_argument('--sig2sig', default='False', type=lambda x: bool(strtobool(x)), help='whether to aggregate the annotation over the whole analysis window or not')
    parser.add_argument('--fs', default=125, type=int, help='signal sampling frequency')
    parser.add_argument('--input_seq_len_s', default=5, type=int, help='input sequence length in seconds')
    parser.add_argument('--channels', default='32, 64, 128, 256, 512', type=str, help='comma separated list of channels produced by the convolutional blocks')
    parser.add_argument('--num_heads_attention', default=1, type=int, help='heads number of the final self-attention layer') 
    parser.add_argument('--dim_feedforward_attention', default=128, type=int, help='dimension of the final self-attention layer') 
    parser.add_argument('--kernel_size', default=3, type=int, help='convolutional layer kernel size')
    parser.add_argument('--num_groups_gn', default=8, type=int, help='number of groups for GroupNorm layers') # Added num_groups_gn
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()    
    
    net = UNet(
        ecg=args.ecg,
        resp=args.resp,
        sig2sig=args.sig2sig,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        channels=args.channels,
        kernel_size=args.kernel_size,
        num_heads_attention=args.num_heads_attention,
        dim_feedforward_attention=args.dim_feedforward_attention
    )        
    net.print_summary(batch_size=args.batch_size)