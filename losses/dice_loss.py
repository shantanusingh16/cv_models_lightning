import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, predictions, targets):
        predictions = F.softmax(predictions, dim=1)
        
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth + self.eps)
        return 1 - dice_score

# Example usage:
# criterion = DiceLoss()
# loss = criterion(predictions, targets)