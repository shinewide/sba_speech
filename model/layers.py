import torch
from torch import nn

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=None, dilation=1,
                 bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

    def forward(self, input):
        output = self.conv(input)
        return output

if __name__ == '__main__':
    L_in = 14
    kernel_size = 5
    dilation = 1
    stride = 1

    # L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    L_out = (L_in + -1 + 1)











