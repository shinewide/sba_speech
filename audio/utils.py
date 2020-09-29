import torch
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util


def dynamic_range_compression(mel_spectrogram, C=1, min_clip_val=1e-5):
    print('input', mel_spectrogram.size(), torch.min(mel_spectrogram))
    clamp = torch.clamp(mel_spectrogram, min=min_clip_val)
    print('clamp', clamp.size(), torch.min(clamp))
    clamp_factor = clamp * C
    print('clamp_factor', clamp_factor.size(), torch.min(clamp_factor))
    log_mel_spectrogram = torch.log(clamp_factor)
    print('log_mel_spectrogram', log_mel_spectrogram.size(), torch.min(log_mel_spectrogram))
    return log_mel_spectrogram