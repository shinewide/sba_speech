import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from librosa.filters import mel as librosa_mel_fn
from scipy.signal import get_window
from librosa.util import pad_center, tiny


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
        print('num_batches', num_batches)
        print('num_samples', num_samples)

        self.num_samples = num_samples

        # (4, 1, 10000)
        input_data = input_data.view(num_batches, 1, num_samples)
        print('input_data', input_data.size())
        # (4, 1, 10000)
        # (4, 1, 1, 10000)
        input_data = F.pad(input_data.unsqueeze(1),
                           (int(self.filter_length / 2), int(self.filter_length / 2),
                            0, 0),
                            mode='reflect')

        print('input_data', input_data.size())

        input_data = input_data.squeeze(1)
        print('input_data', input_data.size())

        forward_transform = F.conv1d(input_data,
                                     Variable(self.forward_basis, requires_grad=False),
                                     stride=self.hop_length, padding=0)
        print('forward_transform', forward_transform.size())

        cutoff = int((self.filter_length / 2) + 1)
        print('cutoff', cutoff)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase


if __name__ == '__main__':
    # stft = STFT()

    audio_path = '../data/lj/wavs/LJ001-0002.wav'

    import librosa
    wav, sr = librosa.core.load(audio_path, sr=22050)
    wav = torch.from_numpy(wav).float()

    wav = wav.unsqueeze(0)

    stft = STFT()

    magnitude, phase = stft.transform(wav)
    print(magnitude.size())
    print(phase.size())

    mel_basis = torch.randn((1, 80, 513))

    mel_spec = torch.matmul(mel_basis, magnitude)

    print('mel_spec',  mel_spec.size())




















if __name__ == '__main__':
    stft = STFT()



























