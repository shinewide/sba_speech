import torch
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util
import librosa
from hparams import hparams as hps

def load_wav_torch(wav_path):
    wav, sr = librosa.core.load(wav_path, sr=hps.sampling_rate)
    wav = wav.astype(np.float32)
    # norm wav
    wav = wav / np.max(np.abs(wav))
    wav = torch.from_numpy(wav).float()
    return wav

def dynamic_range_compression(mel_spectrogram, C=1, min_clip_val=1e-5):
    clamp = torch.clamp(mel_spectrogram, min=min_clip_val)
    clamp_factor = clamp * C
    log_mel_spectrogram = torch.log(clamp_factor)
    return log_mel_spectrogram