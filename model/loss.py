import torch
from torch import nn

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, mel_predict, mel_post_predict, mel_target):
        mel_target.required_grad = False
        p = 5
        mel_loss = nn.MSELoss()(p * mel_predict, p * mel_target)
        mel_post_loss = nn.MSELoss()(p * mel_post_predict, p * mel_target)

        loss = mel_loss + mel_post_loss
        mel_loss_item = (mel_loss / (p ** 2)).item()
        mel_post_loss_item = (mel_post_loss / (p ** 2)).item()
        return loss, mel_loss_item, mel_post_loss_item