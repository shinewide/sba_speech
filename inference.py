import os
import torch
import numpy as np
import matplotlib.pylab as plt
from model.tacotron2 import Tacotron2
from audio.stft import TacotronSTFT
from scipy.io.wavfile import write
from text import text_to_sequence


if __name__ == '__main__':
    model = Tacotron2()

    ckpt_dict = torch.load('./pretrained/tacotron2_statedict.pt',
                           map_location=torch.device('cpu'))

    model.load_state_dict(ckpt_dict['state_dict'])
    print('success load model')













