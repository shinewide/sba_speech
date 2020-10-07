import torch
from torch import nn
from torch.nn import functional as F
from model.layers import LinearNorm, ConvNorm
from hparams import hparams as hps


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(1024, 128,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(512, 128,
                                       bias=False, w_init_gain='tanh')
        self.v = LinearNorm(128, 1, bias=False)
        self.location_layer = LocationLayer()
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        # (B, 1024) -> (B, 128)
        processed_query = self.query_layer(query.unsqueeze(1))
        # print('processed_query : ', processed_query.size())
        # print('processed_memory : ', processed_memory.size())
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_memory + processed_attention_weights))
        # print('energies : ', energies.size())
        return energies.squeeze(2)

    def forward(self, attention_hidden_state, memory, proccessed_memory,
                attention_weights_cat, mask):
        alignment = self.get_alignment_energies(attention_hidden_state,
                                                proccessed_memory,
                                                attention_weights_cat)
        # alignment : (B, Seq_Len : 90)
        # print('alignment : ', alignment.size())

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        # attention weights : (B, 151) [0.9, 0.1 ......]
        # memory : (B, 151, 512) 512 * 0.9
        # print('memory : ', memory.size())
        # att_W * memory => (1, 151) * (151, 512) -> (1, 512)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        # print('attention_context : ', attention_context.size())
        attention_context = attention_context.squeeze(1)
        # print('attention_context : ', attention_context.size())

        return attention_context, attention_weights


class LocationLayer(nn.Module):
    def __init__(self):
        super(LocationLayer, self).__init__()
        kernel_size = 31
        padding = int(((kernel_size - 1) / 2))
        self.location_conv = ConvNorm(2, 32,
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      bias=False, stride=1, dilation=1)
        self.location_dense = LinearNorm(32, 128,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        # input : (B, 2, Seq_Len)
        processed_attention = self.location_conv(attention_weights_cat)
        print(processed_attention.size())
        # processed attention : (B, 32, Seq_Len)
        processed_attention = processed_attention.transpose(1, 2)
        # (B, Seq_len, 32)
        processed_attention = self.location_dense(processed_attention)
        # (B, Seq_Len, 128)
        print(processed_attention.size())
        return processed_attention






















