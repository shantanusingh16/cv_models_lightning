import torch
import torch.nn as nn

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, pred, target, valid_lengths=None):
        if valid_lengths is not None:
            # Create a mask for valid detections
            mask = torch.zeros_like(pred[:, :, 0], dtype=torch.bool)
            for i, length in enumerate(valid_lengths):
                mask[i, :length] = True
            
            # Apply mask to pred and target
            pred = pred[mask]
            target = target[mask]

        diff = torch.abs(pred - target)
        less_than_beta = diff < self.beta
        loss = torch.where(less_than_beta, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)
        
        return loss.mean()