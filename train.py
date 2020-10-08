import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from feeder.speech_dataset import SpeechDataset, SpeechCollate
from model.tacotron2 import Tacotron2
from model.loss import Tacotron2Loss
from time import time
from utils import save_png, mode


def parse_batch(batch, use_gpu):
    mel_padded, output_lengths, text_padded, input_lengths, gate_padded = batch
    mel_padded = mode(mel_padded, use_gpu=use_gpu)
    output_lengths = mode(output_lengths, use_gpu=use_gpu)
    text_padded = mode(text_padded, use_gpu=use_gpu)
    input_lengths = mode(input_lengths, use_gpu=use_gpu)
    gate_padded = mode(gate_padded, use_gpu=use_gpu)
    return (mel_padded, output_lengths, text_padded,
            input_lengths, gate_padded)

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


def train(dataset_dir, log_dir, load_path=None, device='cpu'):
    use_gpu = True if device == 'gpu' else False

    # init Tacotron2
    model = Tacotron2()
    mode(model, use_gpu=use_gpu)

    # nvidia tacotron weight append
    # nvidia_ckpt_dict = torch.load(load_path,
    #                               map_location=torch.device('cpu'))
    # model.load_state_dict(nvidia_ckpt_dict['state_dict'])
    # load_path = None

    # init loss fn
    criterion = Tacotron2Loss()

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    epoch = 0
    max_epoch = 1
    iteration = 1
    save_iters = 1

    # if load_path is not None:
    #     model, optimizer, iteration = load_model(load_path, model, optimizer)

    # init lr scheduler
    lr_lambda = lambda step: 4000 ** 0.5 * min((step + 1) * 4000 ** -1.5, (step + 1) ** -0.5)
    if load_path is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lr_lambda,
                                                      last_epoch=iteration)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lr_lambda)

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

        for batch in dataloader:
            stime = time()
            batch = parse_batch(batch, use_gpu)
            mel_padded, output_lengths, text_padded, input_lengths, gate_padded = batch
            mel_predict, mel_post_predict, gate_predict, alignments = model((text_padded.long(), input_lengths.long(), mel_padded.float(), output_lengths.long()))

            loss, mel_loss, mel_post_loss, gate_loss = criterion(mel_predict, mel_post_predict, gate_predict, mel_padded, gate_padded)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            dur_time = time() - stime
            lr = optimizer.param_groups[0]['lr']
            print('epoch : {}, iteration : {}, mel_loss : {:.8f}, mel_post_loss : {:.8f}, gate_loss : {:.8f}, time : {:.1f}s/it (lr : {})'.format(
                epoch + 1, iteration, mel_loss, mel_post_loss, gate_loss, dur_time, lr))

            if iteration % save_iters == 0:
                save_model(log_dir, model, optimizer, iteration)
                mel_output = mel_predict[0].cpu().detach().numpy().astype(np.float32)
                mel_target = mel_padded[0].cpu().detach().numpy().astype(np.float32)
                alignment = alignments[0].cpu().detach().numpy().astype(np.float32).T
                png_path = os.path.join(log_dir, 'mel_{}.png'.format(iteration))
                save_png((mel_output, mel_target, alignment), png_path)

            iteration += 1
        epoch += 1


if __name__ == '__main__':
    dataset_dir = './data/lj'
    log_dir = './logs'
    load_path = None
    device = 'gpu'  # gpu, cpu

    train(dataset_dir, log_dir, load_path, device)
