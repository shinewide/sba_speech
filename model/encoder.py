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
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hps.encoder_embedding_dim))
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hps.encoder_embedding_dim,
                            int(hps.encoder_embedding_dim / 2), num_layers=1,
                            batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths):
        x = inputs
        # input : char embedding
        # (conv1d -> batchnorm1d -> relu -> drop out) * 3
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), hps.encoder_dropout_p, self.training)
            # print('encoder conv : ', x.size())

        x = x.transpose(1, 2)
        # print('encoder lstm input : ', x.size())

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # print('lstm output : ', x.size())
        # hidden : (vec : hidden state, vec)
        # hidden state : (2, B, LSTM_DIM : 256)
        # hidden = hidden[0]
        #
        # encoder_context = hidden.view(hidden.size(1), -1)

        # x : (B, Seq_len, forward_dim + backward_dim)
        return x
        # return encoder_context

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs











