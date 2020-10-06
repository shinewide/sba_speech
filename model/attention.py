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

        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        # (B, 1024) -> (B, 128)
        processed_query = self.query_layer(query.unsqueeze(1))
        print('processed_query : ', processed_query.size())
        print('processed_memory : ', processed_memory.size())
        energies = self.v(torch.tanh(processed_query + processed_memory))
        print('energies : ', energies.size())
        return energies.squeeze(2)

    def forward(self, attention_hidden_state, memory, proccessed_memory,
                attention_weights_cat, mask):
        alignment = self.get_alignment_energies(attention_hidden_state,
                                                proccessed_memory,
                                                attention_weights_cat)
        print('alignment : ', alignment.size())

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        print('attention_context : ', attention_context.size())

        return attention_context, attention_weights










