import torch
from torch.utils.data import DataLoader
from feeder.speech_dataset import SpeechDataset, SpeechCollate
from model.tacotron2 import Tacotron2

def train(dataset_dir):
    # init Tacotron2
    model = Tacotron2()

    # prepare data loader
    dataset = SpeechDataset(dataset_dir)
    collate_fn = SpeechCollate()

    batch_size = 2
    dataloader = DataLoader(dataset, num_workers=0, shuffle=True,
                            batch_size=batch_size, drop_last=True,
                            collate_fn=collate_fn)

    # change train mode
    model.train()

    epoch = 0
    max_epoch = 1
    iteration = 1

    while epoch < max_epoch:
        for batch in dataloader:
            mel_padded, output_lengths, text_padded, input_lengths = batch
            mel_predict = model((text_padded.long(), input_lengths.long(), mel_padded.float(), output_lengths.long()))

            print(mel_predict.size(), mel_padded.size())
            iteration += 1
        epoch += 1


if __name__ == '__main__':
    dataset_dir = './data/lj'

    train(dataset_dir)







