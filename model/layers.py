import torch
from torch import nn


class LinearNorm(nn.Module):
    def __init__(self, in_dim, output_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()

        self.linear_layer = nn.Linear(in_dim, output_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, inputs):
        return self.linear_layer(inputs)


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

        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, input):
        output = self.conv(input)
        return output


if __name__ == '__main__':
    random_tensor = torch.randn((2, 5, 5))

    input_dim = 5
    output_dim = 3

    linear = LinearNorm(input_dim, output_dim)

    print(random_tensor.size())
    linear_vector = linear(random_tensor)
    print(linear_vector.size())







