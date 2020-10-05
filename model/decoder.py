import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
# from model.attention import Attention
from model.layers import ConvNorm, LinearNorm
from model.modules import Prenet
from hparams import hparams as hps
from utils import get_mask_from_lengths


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.n_frames_per_step = hps.n_frames_per_step
        self.n_mel_channels = hps.n_mel_channels
        self.encoder_embedding_dim = hps.encoder_embedding_dim
        self.attention_rnn_dim = hps.attention_rnn_dim
        self.decoder_rnn_dim = hps.decoder_rnn_dim

        self.prenet = Prenet()

        # attention rnn input dim : 256 + 512
        # self.attention_rnn = nn.LSTMCell(hps.prenet_output_dim + hps.encoder_embedding_dim,
        #                                  hps.attention_rnn_dim)

        # decoder rnn input : 256 + 512 = 768
        # decoder rnn output : 1024
        self.decoder_rnn = nn.LSTMCell(hps.prenet_output_dim + hps.encoder_embedding_dim,
                                        hps.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(hps.decoder_rnn_dim,
                                            hps.n_mel_channels * hps.n_frames_per_step)


    def get_go_frame(self, memory):
        batch_size = memory.size(0)
        go_frames = Variable(
            memory.data.new(batch_size,
                            self.n_frames_per_step * self.n_mel_channels).zero_())
        return go_frames

    def parse_decoder_inputs(self, decoder_inputs):
        batch_size = decoder_inputs.size(0)
        frame_size = decoder_inputs.size(2)

        decoder_inputs = decoder_inputs.transpose(1, 2).contiguous()
        # print('decoder input transpose : ', decoder_inputs.size())
        decoder_inputs = decoder_inputs.view(batch_size,
                                            int(frame_size / self.n_frames_per_step), -1)
        # print('decoder input view : ', decoder_inputs.size())
        decoder_inputs = decoder_inputs.transpose(0, 1)
        # print('decoder input transpose : ', decoder_inputs.size())
        return decoder_inputs

    def initailze_decoder_states(self, memory, mask):
        batch_size = memory.size(0)
        max_time = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(batch_size, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(batch_size, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(batch_size, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(batch_size, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(batch_size, max_time).zero_())
        self.attention_weights_cum = Variable(memory.data.new(batch_size, max_time).zero_())
        self.attention_context = Variable(memory.data.new(batch_size, self.encoder_embedding_dim).zero_())

        self.memory = memory
        # self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def decode(self, decoder_input):
        print('decoder input : ', decoder_input.size())

        decoder_cell_input = torch.cat((decoder_input, self.memory), -1)
        print('cell input size : ', decoder_cell_input.size())

        self.decoder_cell, self.decoder_hidden = self.decoder_rnn(decoder_cell_input,
                                                                  (self.decoder_hidden, self.decoder_cell))

        decoder_output = self.linear_projection(self.decoder_cell)
        print('decoder output : ', decoder_output.size())

        return decoder_output, None


    def forward(self, memory, decoder_inputs, memory_lengths):
        # memory : (B, Seq_len, 512) --> encoder outputs
        # decoder_inputs : (B, Mel_Channels : 80, frames)
        # memory lengths : (B)

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        print('go frames : ', decoder_input.size())
        print('decoder inputs : ', decoder_inputs.size())
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        print('decoder inputs : ', decoder_inputs.size())
        decoder_inputs = self.prenet(decoder_inputs)

        self.initailze_decoder_states(memory,
                                      mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, alignments = [], []

        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, attention_weights = self.decode(decoder_input)
            # mel_output : (1, B, 240)
            mel_outputs.append(mel_output)
            break

        print('decoder prediction : ', len(mel_outputs))














if __name__ == '__main__':
    input_lengths = torch.LongTensor([10, 5, 8])

    mask = get_mask_from_lengths(input_lengths)






