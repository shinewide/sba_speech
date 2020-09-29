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
            mel_filepath = os.path.join(mel_dir, '%s.pt' % mel_filename)
            trainscript = items[1]
            metadata.append([mel_filepath, trainscript])

    return metadata








class SpeechDataset(Dataset):
    def __init__(self, fdir):
        self.metadata = load_metadata(fdir)
        random.shuffle(self.metadata)

    def __getitem__(self, index):
        return self.metadata[index]

    def __len__(self):
        return len(self.metadata)