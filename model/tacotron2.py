import torch
from torch import nn
from model.encoder import Encoder
from model.decoder import Decoder
from model.modules import PostNet
from hparams import hparams as hps
from math import sqrt
from utils import get_mask_from_lengths


class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()

        self.embedding = nn.Embedding(hps.n_symbols, hps.character_embedding_dim)
        std = sqrt(2.0 / (hps.n_symbols + hps.character_embedding_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = PostNet()

    def parse_outputs(self, mel_outputs, mel_outputs_postnet, gate_outputs, output_lengths):
        mask = ~get_mask_from_lengths(output_lengths, pad=True)
        mask = mask.expand(80, mask.size(0), mask.size(1))
        mask = mask.permute(1, 0, 2)
        # mask : (B, 80, Frames)

        mel_outputs.data.masked_fill_(mask, 0.0)
        mel_outputs_postnet.data.masked_fill_(mask, 0.0)

        # gate outputs : (B, Frames // 3)
        slice_mask = torch.arange(0, mask.size(2), 1)
        gate_outputs.data.masked_fill_(mask[:, 0, slice_mask], 1e3)
        return mel_outputs, mel_outputs_postnet, gate_outputs

    def forward(self, inputs):
        text_inputs, input_lengths, mel_targets, output_lengths = inputs

        # print('input text size : ', text_inputs.size())
        character_embedding = self.embedding(text_inputs)
        # (B, Seq_len, 512)
        # (B, 512, seq_len)
        # print('character embedding size : ', character_embedding.size())
        character_embedding = character_embedding.transpose(1, 2)
        # print('character embedding size : ', character_embedding.size())

        encoder_outputs = self.encoder(character_embedding, input_lengths)
        # print('encoder output size : ', encoder_outputs.size())

        mel_outputs, alignments, gate_outputs = self.decoder(encoder_outputs,
                                               mel_targets,
                                               input_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        mel_outputs, mel_outputs_postnet, gate_outputs = self.parse_outputs(mel_outputs, mel_outputs_postnet, gate_outputs, output_lengths)
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, alignments, gate_outputs = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return mel_outputs, mel_outputs_postnet, alignments

if __name__ == '__main__':
    from feeder.speech_dataset import SpeechDataset, SpeechCollate

    dataset = SpeechDataset('../data/lj')
    collate_fn = SpeechCollate()

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, num_workers=0, shuffle=True,
                            batch_size=5, drop_last=True,
                            collate_fn=collate_fn)

    model = Tacotron2()
    for batch in dataloader:
        mel_padded, output_lengths, text_padded, input_lengths = batch
        model((text_padded.long(), input_lengths.long(), mel_padded.float(), output_lengths.long()))
        break
