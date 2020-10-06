import torch
from torch import nn

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, mel_predict, mel_target):
        mel_target.required_grad = False

        mel_loss = nn.MSELoss()(mel_predict, mel_target)
        return mel_loss, mel_loss.item()