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
            [LinearNorm(in_size, out_size, bias=False) for (in_size, out_size) in zip(in_sizes, out_sizes)]
        )

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), hps.prenet_dropout_p, self.training)
            # print('prenet linear size : ', x.size())

        return x







if __name__ == '__main__':
    prenet = Prenet()