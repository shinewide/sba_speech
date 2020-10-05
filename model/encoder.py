from torch import nn
from torch.nn import functional as F
from model.layers import ConvNorm
from hparams import hparams as hps


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hps.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hps.character_embedding_dim,
                         hps.encoder_embedding_dim,
                         kernel_size=hps.encoder_kernel_size,
                         stride=1, padding=int((hps.encoder_kernel_size - 1) / 2),
                         dilation=1),
                nn.BatchNorm1d(hps.encoder_embedding_dim))
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, inputs):
        x = inputs
        # input : char embedding
        # conv1d -> batchnorm1d -> conv1d -> batchnorm1d -> conv1d -> batchnorm1d
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
            print('encoder conv : ', x.size())

        # x = conv_1(inputs)
        # x = conv_2(x)
        # x = conv_3(x)
