import torch
from torch import nn
from model.encoder import Encoder
# from model.decoder import Decoder
# from model.modules import Postnet
from hparams import hparams as hps
from math import sqrt

class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()

        self.embedding = nn.Embedding(hps.n_symbols, hps.character_embedding_dim)

        self.encoder = Encoder()

    def forward(self, inputs):
        text_inputs = inputs

        print('input text size : ', text_inputs.size())
        character_embedding = self.embedding(text_inputs)
        # (B, Seq_len, 512)
        # (B, 512, seq_len)
        print('character embedding size : ', character_embedding.size())
        character_embedding = character_embedding.transpose(1, 2)
        print('character embedding size : ', character_embedding.size())

        encoder_outputs = self.encoder(character_embedding)


if __name__ == '__main__':
    text = 'hello tacotron'

    from text import text_to_sequence
    t2s = text_to_sequence(text, ['english_cleaners'])
    t2s = torch.IntTensor(t2s)
    t2s = t2s.unsqueeze(0)

    model = Tacotron2()
    model.eval()
    print(model.training)
    model.train()
    print(model.training)
    # model(t2s.long())



























