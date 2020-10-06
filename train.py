import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from feeder.speech_dataset import SpeechDataset, SpeechCollate
from model.tacotron2 import Tacotron2
from model.loss import Tacotron2Loss
from time import time
from utils import save_png


def save_model(save_dir, model, optimizer, iteration):
    save_filename = 'tacotron2_ckpt_{}'.format(iteration)
    save_path = os.path.join(save_dir, save_filename)

    ckpt_dict = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'iteration': iteration}

    torch.save(ckpt_dict, save_path)


def load_model(ckpt_path, model, optimizer):
    ckpt_dict = torch.load(ckpt_path)
    # model, optimizer, iteration
    model.load_state_dict(ckpt_dict['model'])
    optimizer.load_state_dict(ckpt_dict['optimizer'])
    iteration = ckpt_dict['iteration'] + 1
    return model, optimizer, iteration


def train(dataset_dir, log_dir, load_path=None):
    # init Tacotron2
    model = Tacotron2()

    # init loss fn
    criterion = Tacotron2Loss()

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    epoch = 0
    max_epoch = 100
    iteration = 1
    save_iters = 2

    if load_path is not None:
        model, optimizer, iteration = load_model(load_path, model, optimizer)

    # prepare data loader
    dataset = SpeechDataset(dataset_dir)
    collate_fn = SpeechCollate()

    batch_size = 2
    dataloader = DataLoader(dataset, num_workers=0, shuffle=True,
                            batch_size=batch_size, drop_last=True,
                            collate_fn=collate_fn)

    # change train mode
    model.train()

    while epoch < max_epoch:
        total_loss = 0
        for batch in dataloader:
            stime = time()
            mel_padded, output_lengths, text_padded, input_lengths = batch
            mel_predict = model((text_padded.long(), input_lengths.long(), mel_padded.float(), output_lengths.long()))

            loss, loss_item = criterion(mel_predict, mel_padded)
            total_loss += loss_item
            model.zero_grad()
            loss.backward()
            optimizer.step()

            dur_time = time() - stime
            print('epoch : {}, iteration : {}, loss : {:.8f}, time : {:.1f}s/it'.format(epoch + 1, iteration, loss_item,
                                                                                        dur_time))

            if iteration % save_iters == 0:
                save_model(log_dir, model, optimizer, iteration)
                mel_output = mel_predict[0].detach().numpy().astype(np.float32)
                mel_target = mel_padded[0].detach().numpy().astype(np.float32)
                png_path = os.path.join(log_dir, 'mel_{}.png'.format(iteration))
                save_png((mel_output, mel_target), png_path)

            iteration += 1
        epoch += 1


if __name__ == '__main__':
    dataset_dir = './data/lj'
    log_dir = './logs'
    load_path = './logs/tacotron2_ckpt_5'

    train(dataset_dir, log_dir, None)
