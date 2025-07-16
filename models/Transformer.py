import argparse
from distutils.util import strtobool

from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce
from torchinfo import summary
from thop import profile, clever_format


def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class

class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(p=dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out = out + self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out
    

class AttentionBlock(nn.Module):
    
    def __init__(self, embed_dim=768, hidden_dim=3072, num_heads=12, head_dim=64):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = NystromAttention(
            dim = embed_dim,
            dim_head = head_dim,
            heads = num_heads,
            num_landmarks = 64,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = False,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.25, # dropout in the attention block
        )
        self.shortcut1 = nn.Identity()

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(p=0.25)
        )
        self.shortcut2 = nn.Identity()
         
    def forward(self, x):
        # x -> (batch_size, seq_len, channels)
        
        mask = torch.ones(size=(x.size(0), x.size(1)), device=x.device).bool()
        
        # Pre-Layer Normalization
        dx = self.ln1(x)

        # Attention Block
        attn_output = self.attn(dx, mask=mask) 
        shortcut1 = self.shortcut1(x)
        x = shortcut1 + attn_output

        # Pre-Layer Normalization
        dx = self.ln2(x)

        # MLP Block
        shortcut2 = self.shortcut2(x)
        x = shortcut2 + self.ffn(dx) 

        return x
    

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
    

class TransformerEncoder(nn.Module):
    def __init__(self, num_modalities=1, embed_dim=768, n_head=12, head_dim=64, hidden_dim=3072, num_layers=3, input_seq_len=625):
        super(TransformerEncoder, self).__init__()
        
        self.num_modalities = num_modalities

        self.pos_encoder = PositionalEncoding(embed_dim=embed_dim, max_len=(input_seq_len + 1)) # +1 for the cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.transformer_encoder = nn.Sequential(*[AttentionBlock(embed_dim=embed_dim, hidden_dim=hidden_dim, num_heads=n_head, head_dim=head_dim) for _ in range(num_layers)])
        
    def forward(self, x):
        
        # Add Classification Token along the sequence length
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x) 
        
        return x
    

class Transformer(nn.Module):
    def __init__(self,  
                 ecg=False,
                 resp=False,
                 sig2sig=False,
                 fs=125,
                 input_seq_len_s=5,
                 embed_dim=128,
                 num_heads=8,
                 dim_feedforward=512,
                 num_encoder_layers=2):
        super(Transformer, self).__init__()
        
        # Input Data Setup
        self.input_seq_len = input_seq_len_s * fs
        self.ecg = ecg
        self.resp = resp
        self.sig2sig = sig2sig
        
        # Architecture Setup
        ppg_in_channels, ecg_in_channels, resp_in_channels = self.get_input_channels()
        self.in_channels = ppg_in_channels + ecg_in_channels + resp_in_channels
        
        self.embed_dim = embed_dim

        # Linear projection to get input into model_dim (embedding)
        # Input: [batch_size, seq_len, input_dim] -> Output: [batch_size, seq_len, model_dim]
        self.input_projection = nn.Linear(self.in_channels, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim=embed_dim, max_len=self.input_seq_len) # Operates on (batch, model_dim, seq_len)

        # Transformer Encoder
        # PPG is always present while ECG/RESP may not int(ecg) == 0 when false, same for resp of course
        self.transformer_encoder = TransformerEncoder(
            num_modalities=1 + int(self.ecg) + int(self.resp), 
            embed_dim=embed_dim, 
            n_head=num_heads, 
            head_dim=(embed_dim // num_heads), 
            hidden_dim=dim_feedforward, 
            num_layers=num_encoder_layers, 
            input_seq_len=self.input_seq_len
            )

        # Output layer maps transformer output to desired output dimension
        # Output: [batch_size, seq_len, model_dim] -> [batch_size, seq_len, output_dim=1], as output is the ABP waveform
        self.fc = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        
        x = x.unsqueeze(-1) if len(x.shape) == 2 else x # x -> (batch_size, seq_len, channels)
        
        src = self.input_projection(x) # -> [batch_sisze, seq_len, channels=embed_dim]
        
        # Add positional encoding. PositionalEncoding1D expects (batch_size, embed_dim, seq_len)
        src_pe = self.pos_encoder(src)
        
        # Transformer Encoder expects input: [batch_size, seq_len, embed_dim]
        output = self.transformer_encoder(src_pe) # -> [batch_size, seq_len, embed_dim]

        # Remove class token for this time
        output = output[:, 1:, :]

        # Apply linear layer for regression output
        output = self.fc(output) # -> [batch_size, seq_len, output_dim=1]
        
        return output.squeeze(-1)
    
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
        print(f'Transformer has {num_params} params and {macs} macs.')

        
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
    parser.add_argument('--embed_dim', default=32, type=int, help='transformer embedding dimension size')
    parser.add_argument('--num_heads', default=8, type=int, help='number of heads for the self-attention mechanism')
    parser.add_argument('--num_encoder_layers', default=2, type=int, help='number of trasnformer layers')
    parser.add_argument('--dim_feedforward', default=128, type=int, help='feedforward dimension size in the transformer encoder')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()    
    
    net = Transformer(
        ecg=args.ecg,
        resp=args.resp,
        sig2sig=args.sig2sig,
        fs=args.fs,
        input_seq_len_s=args.input_seq_len_s,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.num_encoder_layers
    )        
    net.print_summary(batch_size=args.batch_size)