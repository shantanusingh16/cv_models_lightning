import torch
import torch.nn as nn

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        less_than_beta = diff < self.beta
        loss = torch.where(less_than_beta, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)
        return loss.mean()