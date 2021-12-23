from torch import nn
import numpy as np
from torch.nn.utils import spectral_norm
from collections import OrderedDict

from .spatial_softmax import SpatialSoftmax


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=True)  # assume not using batchnorm so use bias


def conv2d_size_out(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation *
                                      (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation *
                                      (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class ConvNet(nn.Module):
    def __init__(
        self,
        input_n_channel=1,  # not counting z_conv
        append_dim=0,  # not counting z_mlp
        cnn_kernel_size=[5, 3],
        cnn_stride=[2, 1],
        output_n_channel=[16, 32],
        img_size=128,
        verbose=True,
        use_sm=True,
        use_bn=True,
        use_spec=False,
        use_residual=False,
    ):

        super(ConvNet, self).__init__()

        self.append_dim = append_dim
        assert len(cnn_kernel_size) == len(output_n_channel), (
            "The length of the kernel_size list does not match with the " +
            "#channel list!")
        self.n_conv_layers = len(cnn_kernel_size)

        if np.isscalar(img_size):
            height = img_size
            width = img_size
        else:
            height, width = img_size

        # Use ModuleList to store [] conv layers, 1 spatial softmax and [] MLP
        # layers.
        self.moduleList = nn.ModuleList()

        #= CNN: W' = (W - kernel_size + 2*padding) / stride + 1
        # Nx1xHxW -> Nx16xHxW -> Nx32xHxW
        for i, (kernel_size, stride, out_channels) in enumerate(
                zip(cnn_kernel_size, cnn_stride, output_n_channel)):

            # Add conv
            padding = 0
            if i == 0:
                in_channels = input_n_channel
            else:
                in_channels = output_n_channel[i - 1]
            module = nn.Sequential()
            conv_layer = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding)
            if use_spec:
                conv_layer = spectral_norm(conv_layer)
            module.add_module("conv_1", conv_layer)

            # Add batchnorm
            if use_bn:
                module.add_module('bn_1',
                                  nn.BatchNorm2d(num_features=out_channels))

            # Always ReLU
            module.add_module('act_1', nn.ReLU())

            # Add module
            self.moduleList.append(module)

            # Add residual block, does not change shape
            if use_residual:
                self.moduleList.append(
                    ResidualBlock(out_channels, out_channels))

            # Update height and width of images after modules
            height, width = conv2d_size_out([height, width], kernel_size,
                                            stride, padding)

        #= Spatial softmax, output 64 (32 features x 2d pos) or Flatten
        self.use_sm = use_sm
        if use_sm:
            module = nn.Sequential(
                OrderedDict([('softmax',
                              SpatialSoftmax(height=height,
                                             width=width,
                                             channel=output_n_channel[-1]))]))
            cnn_output_dim = int(output_n_channel[-1] * 2)
        else:
            module = nn.Sequential(OrderedDict([('flatten', nn.Flatten())]))
            cnn_output_dim = int(output_n_channel[-1] * height * width)
        self.moduleList.append(module)
        self.cnn_output_dim = cnn_output_dim

        if verbose:
            print(self.moduleList)

    def get_output_dim(self):
        return self.cnn_output_dim

    def forward(self, x):

        if x.dim() == 3:
            x = x.unsqueeze(1)  # Nx1xHxW
        for module in self.moduleList:
            x = module(x)
        return x