import torch
from torch import nn

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, mel_predict, mel_post_predict, gate_predict, mel_target, gate_target):
        mel_target.required_grad = False
        gate_target.required_grad = False
        slice_arange = torch.arange(0, gate_target.size(1), 1)
        gate_target = gate_target[:, slice_arange]

        gate_predict = gate_predict.view(-1, 1)
        gate_target = gate_target.view(-1, 1)
        p = 5
        mel_loss = nn.MSELoss()(p * mel_predict, p * mel_target)
        mel_post_loss = nn.MSELoss()(p * mel_post_predict, p * mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_predict, gate_target)

        loss = mel_loss + mel_post_loss + gate_loss
        mel_loss_item = (mel_loss / (p ** 2)).item()
        mel_post_loss_item = (mel_post_loss / (p ** 2)).item()
        gate_loss_item = gate_loss.item()
        return loss, mel_loss_item, mel_post_loss_item, gate_loss_item