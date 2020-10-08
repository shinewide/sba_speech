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


def dynamic_range_decompression(mel_spectrogram, C=1):
    mel_spectrogram = torch.exp(mel_spectrogram) / C
    return mel_spectrogram


def window_sumsquare(window, n_frames, hop_length, win_length, n_fft,
                     dtype = np.float32, norm=None):
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]

    return x

def griffin_lim(mel_spectrogram, stft_fn, n_iters=30):

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*mel_spectrogram.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(mel_spectrogram, angles).squeeze(1)

    for i in range(n_iters):
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(mel_spectrogram, angles).squeeze(1)

    return signal





