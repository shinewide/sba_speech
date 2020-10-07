import torch
from torch import nn
from torch.nn import functional as F
from model.layers import ConvNorm, LinearNorm
from hparams import hparams as hps


class Prenet(nn.Module):
    def __init__(self):
        super(Prenet, self).__init__()
        out_sizes = hps.prenet_out_sizes
        in_sizes = [hps.prenet_input_dim] + out_sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False, w_init_gain='relu') for (in_size, out_size) in
             zip(in_sizes, out_sizes)]
        )

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), hps.prenet_dropout_p, self.training)
            # print('prenet linear size : ', x.size())

        return x


class PostNet(nn.Module):
    def __init__(self):
        super(PostNet, self).__init__()
        kernel_size = 5
        padding = int((kernel_size - 1) / 2)

        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=kernel_size, stride=1,
                         padding=padding, dilation=1,
                         w_init_gain='tanh'),
                nn.BatchNorm1d(512)
            )
        )

        for i in range(3):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512, 512,
                             kernel_size=kernel_size,
                             padding=padding, stride=1, dilation=1,
                             w_init_gain='tanh'),
                    nn.BatchNorm1d(512)
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=kernel_size,
                         padding=padding, stride=1, dilation=1),
                nn.BatchNorm1d(80)
            )
        )

    def forward(self, mel_outputs):
        x = mel_outputs
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)

        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x


if __name__ == '__main__':
    postNet = PostNet()
    print(postNet)