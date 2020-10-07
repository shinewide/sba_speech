import os
import torch
import numpy as np
import matplotlib.pylab as plt
from model.tacotron2 import Tacotron2
from audio.stft import TacotronSTFT
from scipy.io.wavfile import write
from text import text_to_sequence
from hparams import hparams as hps
from utils import save_png


if __name__ == '__main__':
    model = Tacotron2()

    ckpt_dict = torch.load('./pretrained/tacotron2_statedict.pt',
                           map_location=torch.device('cpu'))

    model.load_state_dict(ckpt_dict['state_dict'])
    print('success load model')

    input_text = 'hello tacotron thank you nvidia!'

    input_sequence = np.array(
        text_to_sequence(input_text, hps.cleaner_names))[None, :]

    input_sequence = torch.autograd.Variable(
        torch.from_numpy(input_sequence)).long()

    mel_outputs, mel_outputs_postnet, alignments = model.inference(input_sequence)

    mel_outputs = mel_outputs.float().detach().numpy()[0]
    mel_outputs_postnet = mel_outputs_postnet.float().detach().numpy()[0]
    alignments = alignments.float().detach().numpy()[0].T

    save_png((mel_outputs, mel_outputs_postnet, alignments), './inference_test.png')












