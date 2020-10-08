import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from librosa.filters import mel as librosa_mel_fn
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from audio.utils import dynamic_range_compression, dynamic_range_decompression, window_sumsquare
from hparams import hparams as hps


class TacotronSTFT(nn.Module):
    def __init__(self):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = hps.n_mel_channels
        self.sampling_rate = hps.sampling_rate
        self.filter_length = hps.filter_length
        self.hop_length = hps.hop_length
        self.win_length = hps.win_length
        self.mel_fmin = hps.mel_fmin
        self.mel_fmax = hps.mel_fmax
        self.stft_fn = STFT(filter_length=self.filter_length, hop_length=self.hop_length,
                            win_length=self.win_length)
        # numpy
        mel_basis = librosa_mel_fn(self.sampling_rate, self.filter_length,
                                   self.n_mel_channels, self.mel_fmin, self.mel_fmax)
        # np -> torch
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def mel_spectrogram(self, y):
        assert (torch.min(y.data) >= -1)
        assert (torch.max(y.data) <= 1)

        # y : (B, N)
        magnitude, _ = self.stft_fn.transform(y)
        # mag : (B, D, F)
        magnitude = magnitude.data
        # mel_basis : (80, D)
        # (80, D) * (D, F) -> (B, 80, F)
        mel_spectrogram = torch.matmul(self.mel_basis, magnitude)
        # mel spectrogram -> log mel spectrogram
        mel_spectrogram = self.spectral_normalize(mel_spectrogram)
        return mel_spectrogram

    def spectral_normalize(self, mel_spectrogram):
        log_mel_spectrogram = dynamic_range_compression(mel_spectrogram)
        return log_mel_spectrogram

    def spectral_de_normalize(self, mel_spectrogram):
        mel_spectrogram = dynamic_range_decompression(mel_spectrogram)
        return mel_spectrogram


class STFT(nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert (filter_length >= win_length)

            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        # (4, 10000)
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # (4, 1, 10000)
        input_data = input_data.view(num_batches, 1, num_samples)
        # (4, 1, 10000)
        # (4, 1, 1, 10000)
        input_data = F.pad(input_data.unsqueeze(1),
                           (int(self.filter_length / 2), int(self.filter_length / 2),
                            0, 0),
                           mode='reflect')

        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(input_data,
                                     Variable(self.forward_basis, requires_grad=False),
                                     stride=self.hop_length, padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length / 2):]

        return inverse_transform

    def forward(self, input_data):
        # (B, waveform)
        self.magnitude, self.phase = self.transform(input_data)
        # (B, SpecDim, Frames), (B, SpecDim, Frames)
        reconstruction = self.inverse(self.magnitude, self.phase)
        # (B, waveform)
        return reconstruction


if __name__ == '__main__':
    taco_stft = TacotronSTFT()
    audio_path = '../data/lj/wavs/LJ001-0002.wav'
    import librosa

    wav, sr = librosa.core.load(audio_path, sr=22050)
    assert (sr == 22050)
    wav = torch.from_numpy(wav).float()
    # (N) -> (B, N)
    wav = wav.unsqueeze(0)

    mel_spec = taco_stft.mel_spectrogram(wav)
    # print(mel_spec.size())

    # # stft = STFT()
    #
    # audio_path = '../data/lj/wavs/LJ001-0002.wav'
    #
    # import librosa
    # wav, sr = librosa.core.load(audio_path, sr=22050)
    # wav = torch.from_numpy(wav).float()
    #
    # wav = wav.unsqueeze(0)
    #
    # stft = STFT()
    #
    # magnitude, phase = stft.transform(wav)
    # print(magnitude.size())
    # print(phase.size())
    #
    # # (80, 513) * (513, 164)
    # mel_spectrogram = torch.mm(mel_basis, magnitude.squeeze(0))
    #
    # print(mel_spectrogram.size())

if __name__ == '__main__':
    stft = STFT()
