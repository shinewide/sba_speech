import os
import torch
import random
from text import text_to_sequence
from hparams import hparams as hps
from torch.utils.data import Dataset

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def load_metadata(fdir):
    traindata_dir = os.path.join(fdir, 'traindata_%d' % hps.sampling_rate)
    mel_dir = os.path.join(traindata_dir, 'mels')
    metadata_path = os.path.join(traindata_dir, 'train_metadata.csv')

    metadata = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            items = line.split('|')
            mel_filename = items[0]
            mel_filepath = os.path.join(mel_dir, mel_filename)
            trainscript = items[1]
            metadata.append([mel_filepath, trainscript])

    return metadata


class SpeechDataset(Dataset):
    def __init__(self, fdir):
        self.metadata = load_metadata(fdir)
        random.shuffle(self.metadata)

    def get_item(self, items):
        mel_path = items[0]
        transcript = items[1]
        mel_spectrogram = self.get_mel(mel_path)
        text = self.get_text(transcript)
        return mel_spectrogram, text

    def get_mel(self, path):
        mel_spectrogram = torch.load(path)
        return mel_spectrogram

    def get_text(self, transcript):
        text = text_to_sequence(transcript, cleaner_names=hps.cleaner_names)
        text = torch.IntTensor(text)
        return text

    def __getitem__(self, index):
        return self.get_item(self.metadata[index])

    def __len__(self):
        return len(self.metadata)


class SpeechCollate:
    def __init__(self):
        self.n_frames_per_step = hps.n_frames_per_step

    def __call__(self, batch):
        # batch : (mel, text)

        # longTensor = [98, 100]
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[1]) for x in batch]),
            dim=0, descending=True)

        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][1]
            text_padded[i, :text.size(0)] = text

        num_mels = hps.n_mel_channels
        max_target_len = max([x[0].size(1) for x in batch])

        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step

        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        output_lengths.zero_()

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][0]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

        print(gate_padded)
        return mel_padded, output_lengths, text_padded, input_lengths


if __name__ == '__main__':
    dataset = SpeechDataset('../data/lj')
    collate_fn = SpeechCollate()

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, num_workers=0, shuffle=True,
                            batch_size=2, drop_last=True,
                            collate_fn=collate_fn)
    from tqdm import tqdm

    for batch in tqdm(dataloader):
        for data in batch:
            print(data[0].size(), data[1].size())
            break

        print()
