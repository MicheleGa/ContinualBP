import argparse
from typing import Callable, List, Tuple, Union, Final, Optional
import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torchinfo import summary
from thop import profile, clever_format
from component_factory import BPRegressor


class Chomp1d(nn.Module):

    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor):
        if self.chomp_size == 0:
            return x

        # Cut off elements from the end of the sequence that extend beyond the input sequence length
        return x[:, :, :-self.chomp_size].contiguous()
    

def conditional_apply(func: Callable, whether: bool):
    if whether:
        return func

    return lambda x: x


class GRELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input,) = ctx.saved_tensors
        # GELU derivative: φ(x) + x * Φ(x)
        # where φ = normal pdf, Φ = normal cdf
        cdf = 0.5 * (1.0 + torch.erf(input / torch.sqrt(torch.tensor(2.0, device=input.device))))
        pdf = torch.exp(-0.5 * input**2) / torch.sqrt(torch.tensor(2.0 * torch.pi, device=input.device))
        gelu_grad = cdf + input * pdf
        return grad_output * gelu_grad


class GRELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return GRELUFunction.apply(input)


def select_act_fn(act: Optional[str]) -> nn.Module:
    if act is None:
        return nn.Identity()

    if act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'silu':
        return nn.SiLU(inplace=True)
    elif act == 'grelu':
        return GRELU()
    else:
        raise ValueError(f"Unsupported activation function: {act}. Supported values are 'relu', 'gelu', 'silu', 'grelu', and None but got {act}.")
    

def init_batch_norm(module: nn.BatchNorm1d):
    nn.init.constant_(module.weight, 1)
    nn.init.constant_(module.bias, 0)


def init_tcn_conv_weight(conv: nn.Conv1d):
    nn.init.kaiming_uniform_(conv.weight,
                             mode='fan_in',
                             nonlinearity='relu')

class PointwiseLayer(nn.Sequential):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 dropout=0.2,
                 batch_norm: bool = False,
                 dropout_mode='standard',
                 use_weight_norm=False,
                 act_fn: Optional[str] = 'relu',
                 with_bias=False):
        conv = conditional_apply(weight_norm, use_weight_norm)(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                # Following: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm
                bias=with_bias and not batch_norm))
        normalize = nn.BatchNorm1d(
            out_channels) if batch_norm else nn.Identity()
        act_fn = select_act_fn(act_fn)
        dropout = (nn.Dropout1d if dropout_mode == '1d' else nn.Dropout)(dropout) if dropout > 0 else nn.Identity()

        super(PointwiseLayer, self).__init__(conv, normalize, act_fn, dropout)
        

class TemporalBottleneck(nn.Module):

    def __init__(self,
                 n_inputs: int,
                 n_channels: Tuple[int, int],
                 kernel_size: int,
                 stride: int,
                 dilation: int,
                 padding: int,
                 dropout=0.2,
                 dropout_mode='standard',
                 batch_norm=False,
                 weight_norm=False,
                 act_fn: str = 'relu',
                 groups=1,
                 residual=True,
                 force_downsample=False):
        """Basically the same idea as a bottleneck layer in a ResNet, but now
        for a 1D convolutional network.

        Usually n_channels is a tuple of (channels / expansion=4, channels).
        By setting groups > 1, we effectively use a ResNeXt bottleneck layer. Setting groups = -1,
        creates a depthwise separable convolution layer.
        """

        super(TemporalBottleneck, self).__init__()

        n_intermediates, n_outputs = n_channels
        requires_downsample = (n_inputs != n_outputs or force_downsample) and residual

        self.residual = residual

        self.temp_layer1 = PointwiseLayer(n_inputs, n_intermediates, 1, dropout, dropout_mode,
                                          batch_norm, weight_norm, act_fn, True)
        self.temp_layer2 = TemporalLayer(n_intermediates, n_intermediates,
                                         kernel_size, stride, dilation,
                                         padding, dropout, dropout_mode, batch_norm,
                                         weight_norm, act_fn, n_intermediates if groups == -1 else groups)
        self.temp_layer3 = PointwiseLayer(n_intermediates, n_outputs, 1, dropout, dropout_mode,
                                          batch_norm, weight_norm, None, True)

        self.downsample = PointwiseLayer(
            n_inputs, n_outputs, 1, dropout, dropout_mode, batch_norm, weight_norm,
            False, False) if requires_downsample else None

        self.act_fn = select_act_fn(act_fn)

    def forward(self, x: torch.Tensor):
        res = x

        out = self.temp_layer1(x)
        out = self.temp_layer2(out)
        out = self.temp_layer3(out)

        if self.residual:
            if self.downsample is not None:
                res = self.downsample(res)

            out += res

        out = self.act_fn(out)

        return out
    

class TemporalLayer(nn.Sequential):

    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int,
                 padding: int,
                 dropout=0.2,
                 dropout_mode='standard',
                 batch_norm=False,
                 use_weight_norm=False,
                 act_fn: Optional[str] = 'relu',
                 groups=1):
        padding = padding if kernel_size != 1 else 0

        # We apply padding on both sides and remove the extra values on the right side
        # We could have also only applied padding on the left side like here:
        # https://github.com/locuslab/TCN/issues/8#issuecomment-384345206
        conv = conditional_apply(weight_norm, use_weight_norm)(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation if kernel_size != 1 else 1,
                groups=groups,
                # Following:
                # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm
                bias=not batch_norm))
        chomp = Chomp1d(padding)
        normalize = nn.BatchNorm1d(n_outputs) if batch_norm else nn.Identity()
        act_fn = select_act_fn(act_fn)
        dropout = (nn.Dropout1d if dropout_mode == '1d' else nn.Dropout)(dropout) if dropout > 0 else nn.Identity()

        super(TemporalLayer, self).__init__(conv, normalize, chomp, act_fn,
                                            dropout)


class TemporalBlock(nn.Module):

    def __init__(self,
                 n_inputs: int,
                 n_channels: Tuple[int, int],
                 kernel_size: int,
                 stride: Union[int, Tuple[int, int]],
                 dilation: int,
                 padding: int,
                 dropout: float = 0.2,
                 dropout_mode: str = 'standard',
                 batch_norm: bool = False,
                 weight_norm: bool = False,
                 act_fn: str = 'relu',
                 groups: int = 1,
                 residual: bool = True,
                 force_downsample: bool = False):
        super(TemporalBlock, self).__init__()

        n_intermediates, n_outputs = n_channels
        requires_downsample = (n_inputs != n_outputs or force_downsample) and residual

        self.residual = residual

        if isinstance(stride, int):
            stride = (stride, stride)

        self.temp_layer1 = TemporalLayer(n_inputs, n_intermediates,
                                         kernel_size, stride[0], dilation,
                                         padding, dropout, dropout_mode, batch_norm,
                                         weight_norm, act_fn, n_inputs if groups == -1 else groups)
        self.temp_layer2 = TemporalLayer(n_intermediates, n_outputs,
                                         kernel_size, stride[1], dilation, padding,
                                         dropout, dropout_mode, batch_norm, weight_norm,
                                         None, n_intermediates if groups == -1 else groups)

        if requires_downsample:
            # No bias needed in this layer as the bias of temp_layer2 will have the same effect
            self.downsample = PointwiseLayer(
                n_inputs, n_outputs, stride[1],
                dropout, batch_norm, weight_norm,
                act_fn=None, with_bias=False)
        else:
            self.downsample = None

        self.act_fn = select_act_fn(act_fn)

    def forward(self, x: torch.Tensor):
        out = self.temp_layer1(x)
        out = self.temp_layer2(out)

        if self.residual:
            res = x

            if self.downsample is not None:
                k = self.temp_layer1[0].kernel_size[0]
                stride = self.temp_layer2[0].stride[0]

                if stride != 1:
                    res = res[:, :, 2*k-2:]

                res = self.downsample(res)
            else:
                stride = self.temp_layer2[0].stride[0]
                k = self.temp_layer1[0].kernel_size[0]
                if stride != 1:
                    res = res[:, :, 2*k-2::2]
        
            out += res

        out = self.act_fn(out)

        return out

class LastElement1d(nn.Module):

    def __init__(self):
        super(LastElement1d, self).__init__()

    def forward(self, x):
        return x[:, :, -1]
    

class TemporalConvNet(nn.Sequential):

    def __init__(self,
                 num_inputs: int,
                 num_channels: List[Tuple[int, int]],
                 kernel_size: Union[int, List[int]],
                 dropout=0.2,
                 dropout_mode: str = 'standard',
                 batch_norm: bool = False,
                 weight_norm: bool = False,
                 act_fn: str = 'relu',
                 bottleneck: bool = False,
                 groups: int = 1,
                 residual: bool = True,
                 force_downsample: bool = False,
                 zero_init_residual: bool = False,
                 crop_hidden_states: bool =False):
        if zero_init_residual == True and not batch_norm:
            raise ValueError(
                "To use zero_init_residual, batch_norm has to be set to True."
            )
    
        layers = []

        Block = TemporalBottleneck if bottleneck else TemporalBlock

        if crop_hidden_states == True:
            assert not bottleneck, "Bottleneck layers are not supported when crop_hidden_states is set to True."

        for i in range(len(num_channels)):
            in_channels = num_inputs if i == 0 else num_channels[i - 1][1]
            block_kernel_size = kernel_size[i] if isinstance(kernel_size, list) else kernel_size
            
            if crop_hidden_states:
                dilation_size = 1
                padding = 0
                stride = (1, 2)
            else:
                dilation_size = 2**i
                padding = (block_kernel_size - 1) * dilation_size
                stride = (1, 1)

            layers += [
                Block(in_channels,
                      num_channels[i],
                      block_kernel_size,
                      stride=stride,
                      dilation=dilation_size,
                      padding=padding,
                      dropout=dropout,
                      dropout_mode=dropout_mode,
                      batch_norm=batch_norm,
                      weight_norm=weight_norm,
                      act_fn=act_fn,
                      groups=groups,
                      residual=residual,
                      force_downsample=force_downsample)
            ]

        super(TemporalConvNet, self).__init__(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Initialize weights following Kaiming He's scheme
                init_tcn_conv_weight(m)
            elif isinstance(m, nn.BatchNorm1d):
                # Initialize BatchNorm following PyTorch's ResNet
                # implementation: https://github.com/pytorch/vision/blob/0d68c7df8640abff43355afd57c494cf5d74f4a9/torchvision/models/resnet.py#L211
                init_batch_norm(m)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, TemporalBlock):
                    nn.init.constant_(m.temp_layer2[1].weight, 0)
                elif isinstance(m, TemporalBottleneck):
                    nn.init.constant_(m.temp_layer3[1].weight, 0)
                    

def get_receptive_field_size(kernel_size: Union[int, List[int]],
                             num_layers: int,
                             dilation_exponential_base: int = 2):
    """Calculate the receptive field size of a TCN. We assume the TCN structure of the paper
    from Bai et al.

    Due to: https://github.com/locuslab/TCN/issues/44#issuecomment-677949937

    Args:
        kernel_size (Union[int, List[int]]): Size of the kernel(s).
        num_layers (int): Number of layers in the TCN.
        dilation_exponential_base (int, optional): Dilation exponential size. Defaults to 2.

    Returns:
        int: Receptive field size.
    """

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * num_layers

    return sum([
        2 * dilation_exponential_base**(l - 1) * (kernel_size[l-1] - 1)
        for l in range(1,num_layers+1)
    ]) + 1


def get_kernel_size_and_layers(required_receptive_field_size: int, kernel_sizes: List[int] = [3,5,7,9], dilation_exponential_base: int = 2):
    """Get the configuration of kernel size and number of layers that has a receptive field size closest
    (but always larger) than the required receptive field size.

    Args:
        required_receptive_field_size (int): Required receptive field size.
        kernel_sizes (List[int], optional): List of kernel sizes to choose from. Defaults to [3,5,7,9].
        dilation_exponential_base (int, optional): Dilation exponential size. Defaults to 2.

    Returns:
        Tuple[int, int]: Tuple of kernel size and number of layers.
    """

    configurations = []

    # Find for each kernel size the number of layers that has a receptive field size closest to the required receptive field size
    for kernel_size in kernel_sizes:
        num_layers = 1

        while get_receptive_field_size(kernel_size, num_layers, dilation_exponential_base) < required_receptive_field_size:
            num_layers += 1

        configurations.append((kernel_size, num_layers))

    # Find the configuration with the smallest receptive field size that is larger than the required receptive field size
    min_receptive_field_size = float('inf')
    configuration = None

    for kernel_size, num_layers in configurations:
        receptive_field_size = get_receptive_field_size(kernel_size, num_layers, dilation_exponential_base)

        if receptive_field_size < min_receptive_field_size:
            min_receptive_field_size = receptive_field_size

            configuration = (kernel_size, num_layers)

    return configuration


class TCN(nn.Module):
    
    r"""
    Source: https://github.com/V0XNIHILI/TCN-library/blob/main/src/tcn_lib/tcn.py
    """

    has_linear_layer: Final[bool]
    transpose_input: Final[bool]
    take_last_element: Final[bool]

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 channel_sizes: Union[List[Union[int, Tuple[int, int]]], int, Tuple[int, int]],
                 kernel_size: Optional[Union[int, List[int]]] = None,
                 dropout: float = 0.0,
                 dropout_mode: str = 'standard',
                 batch_norm: bool = False,
                 weight_norm: bool = False,
                 act_fn: str = 'relu',
                 bottleneck: bool = False,
                 groups: int = 1,
                 residual: bool = True,
                 force_downsample: bool = False,
                 zero_init_residual: bool = False,
                 take_last_element: bool = True,
                 input_length: Optional[int] = None,
                 crop_hidden_states: bool = False,
                 transpose_input: bool = False):
        """Temporal Convolutional Network. Implementation based off of: 
        https://github.com/locuslab/TCN/blob/master/TCN/mnist_pixel/model.py.

        Args:
            input_size (int): Dimensionality or number of channels of each input time step.
            output_size (int): Final output size. Set to -1 to omit the linear layer.
            channel_sizes (Union[List[int], List[Tuple[int, int]]]): Number of channels in each layer.
            kernel_size (Union[int, List[int]]): Kernel size. Can be specified for the whole network as a single int, per layer as a list of ints.
            dropout (float, optional): Dropout probability for the temporal convolutional layers. Defaults to 0.0.
            dropout_mode (str, optional): Dropout mode. Can be 'standard' or '1d'. Defaults to 'standard'.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            weight_norm (bool, optional): Whether to use weight normalization. Defaults to False.
            bottleneck (bool, optional): Whether to use bottleneck layers. Note that when bottleneck = True and groups = -1, a depthwise separable convolution is created. Defaults to False.
            groups (int, optional): Number of groups for the temporal convolutional layers. Set to -1 for depthwise convolutions. Defaults to 1.
            residual (bool, optional): Whether to use residual connections. Defaults to True.
            force_downsample (bool, optional): Whether to force downsample in every layer instead of doing an identity shortcut when the number of input channels is equal to the number of output channels. Defaults to False.
            zero_init_residual (bool, optional): Whether to zero initialize the residual connections (per: https://arxiv.org/abs/1706.0267). Defaults to False.
            take_last_element (bool, optional): Whether to take the last element of the output. Defaults to True.
            input_length (Optional[int], optional): Length of the input; only used to check compatibility with the receptive field size. Defaults to None.
            crop_hidden_states (bool, optional): Whether to crop the hidden states to the minimum required length; requires input_length to be specified. Defaults to False.
            transpose_input (bool, optional): Whether to transpose the input from (N, L_in, C_in) to (N, C_in, L_in). Defaults to False.
        """

        super(TCN, self).__init__()

        if kernel_size is None:
            assert input_length is not None, "If kernel_size is not specified, input_length must be specified."
            assert type(channel_sizes) is int or type(channel_sizes) is tuple, "If kernel_size is not specified but input_length is, channel_sizes must be an int or tuple of ints."

            k, l = get_kernel_size_and_layers(input_length)

            kernel_size = [k] * l

        if type(channel_sizes) is int:
            channel_sizes = [(channel_sizes, channel_sizes)] * len(kernel_size)
        elif type(channel_sizes) is tuple:
            channel_sizes = [channel_sizes] * len(kernel_size)
        else:
            # Make sure that also specifying one channel size per temporal layer works
            channel_sizes = [channel_size if type(channel_size) is not int else (channel_size, channel_size) for channel_size in channel_sizes]

        self.input_seq_len = input_length
        self.in_channels = input_size
        self.pad_inputs = None

        if crop_hidden_states:
            assert input_length is not None, "If crop_hidden_states is set to True, input_length must be specified."
            assert take_last_element is True, "Cropping the hidden states only works when the last element is taken. Set take_last_element to True."

        if input_length is not None:
            receptive_field_size = get_receptive_field_size(kernel_size, len(channel_sizes))

            if input_length > receptive_field_size:
                warnings.warn(f"Input length ({input_length}) is larger than the receptive field size ({receptive_field_size}). Use get_kernel_size_and_layers({input_length}) to find the kernel size and number of layers that have a receptive field size closest to the input length.")

            if crop_hidden_states:
                padding = receptive_field_size - input_length

                self.pad_inputs = nn.ZeroPad1d((padding, 0))

        tcn = TemporalConvNet(input_size,
                              channel_sizes,
                              kernel_size=kernel_size,
                              dropout=dropout,
                              dropout_mode=dropout_mode,
                              batch_norm=batch_norm,
                              weight_norm=weight_norm,
                              act_fn=act_fn,
                              bottleneck=bottleneck,
                              groups=groups,
                              residual=residual,
                              force_downsample=force_downsample,
                              zero_init_residual=zero_init_residual,
                              crop_hidden_states=crop_hidden_states)

        if take_last_element:
            self.embedder = nn.Sequential(tcn, LastElement1d())
        else:
            self.embedder = tcn

        self.has_linear_layer = output_size != -1

        if self.has_linear_layer:
            self.fc = nn.Linear(channel_sizes[-1][1], output_size)

        self.transpose_input = transpose_input
        self.take_last_element = take_last_element

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TCN.

        Args:
            inputs (torch.Tensor): Inputs into the TCN. Tensor should be of shape (N = batch size, C_in = input channels, L_in = input length) or (N, L_in, C_in) if transpose_input is True.

        Returns:
            torch.Tensor: Output of the TCN. Tensor will be of shape (N, C_out).
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        if len(x.shape) != 3:
            raise ValueError("Input tensor must have shape [B, T, C] or [B, C, T]")
        
        if (x.shape[1] == self.input_seq_len and x.shape[2] == self.in_channels):
            x = x.permute(0, 2, 1)
        
        if (x.shape[1] != self.in_channels or x.shape[2] != self.input_seq_len):
            raise ValueError("Input tensor dimension mismatch")

        if self.transpose_input:
            x = x.transpose(1, 2)

        if self.pad_inputs:
            x = self.pad_inputs(x)

        out = self.embedder(x)

        if self.has_linear_layer:
            if not self.take_last_element:
                # out shape = (N, C_out, L_out) -> (N * L_out, C_out), since nn.Linear expects the last dimension to be the channel dimension, currently that is the sequence length dimension
                out = out.view(-1, out.shape[1])
    
            out = self.fc(out)

        return out

def parseargs():
    parser = argparse.ArgumentParser(description="ResGruNet summary")

    parser.add_argument('--batch_size', default=16, type=int, help='batch size for training/inference (default: 32)')
    parser.add_argument('--ecg', action=argparse.BooleanOptionalAction, default=False, help='enable ECG signal processing mode')
    parser.add_argument('--fs', default=125, type=int, help='sampling frequency in Hz (default: 125)')
    parser.add_argument('--input_seq_len_s', default=10, type=int, help='input sequence length in seconds (default: 10)')
    parser.add_argument('--embed_dim', default=128, type=int, help='embedding dimension (default: 256)')
    parser.add_argument('--use_lora', action=argparse.BooleanOptionalAction, default=False, help='enable LoRA (Low-Rank Adaptation) fine-tuning')
    parser.add_argument('--lora_r', default=8, type=int, help='LoRA rank parameter (default: 8)')
    parser.add_argument('--lora_alpha', default=16, type=float, help='LoRA alpha scaling parameter (default: 16)')
    parser.add_argument('--lora_dropout', default=0.1, type=float, help='LoRA dropout rate (default: 0.1)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()  

    # TCN input parameters definition from those of argparse
    input_size = 2 if args.ecg else 1
    input_length = args.input_seq_len_s * args.fs
    output_size = args.embed_dim
    channel_sizes = [input_size, 16, 32, 48, 64, 96, output_size]
    kernel_size = [7] * 7                           
    
    # Instantiate encoder and profile its MACs/bytes on a dummy input
    encoder = TCN(
        input_size=input_size,
        output_size=output_size,
        channel_sizes=channel_sizes,
        kernel_size=kernel_size,
        input_length=input_length
    )
    input_tensor = torch.rand((args.batch_size, args.input_seq_len_s * args.fs, 2 if args.ecg else 1))
    print("\n--- Model Encoder Summary ---")
    summary(encoder, input_data=[input_tensor], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))
    macs, num_params = profile(encoder, inputs=(input_tensor,))
    macs, num_params = clever_format([macs, num_params], "%.7f")
    print(f'TCN Encoder has {num_params} params and {macs} MACs.')
    
    # Instantiate prediction head and profile its MACs/bytes on a dummy input
    prediction_head = BPRegressor(args.embed_dim, 3)
    input_tensor = torch.rand((args.batch_size, args.embed_dim))
    print("\n--- Model Prediction Head Summary ---")
    summary(prediction_head, input_data=[input_tensor], col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"))
    macs, num_params = profile(prediction_head, inputs=(input_tensor,))
    macs, num_params = clever_format([macs, num_params], "%.7f")
    print(f'TCN Prediction Head has {num_params} params and {macs} MACs.')
    