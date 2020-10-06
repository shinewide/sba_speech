import torch
from torch.utils.data import DataLoader
from feeder.speech_dataset import SpeechDataset, SpeechCollate
from model.tacotron2 import Tacotron2
from model.loss import Tacotron2Loss
from time import time

def train(dataset_dir):
    # init Tacotron2
    model = Tacotron2()

    # init loss fn
    criterion = Tacotron2Loss()

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

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
    max_epoch = 100
    iteration = 1

    while epoch < max_epoch:
        for batch in dataloader:
            stime = time()
            mel_padded, output_lengths, text_padded, input_lengths = batch
            mel_predict = model((text_padded.long(), input_lengths.long(), mel_padded.float(), output_lengths.long()))

            loss, loss_item = criterion(mel_predict, mel_padded)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            dur_time = time() - stime
            print('epoch : {}, iteration : {}, loss : {:.8f}, time : {:.1f}s/it'.format(epoch + 1, iteration, loss_item, dur_time))

            iteration += 1
        epoch += 1


if __name__ == '__main__':
    dataset_dir = './data/lj'

    train(dataset_dir)







