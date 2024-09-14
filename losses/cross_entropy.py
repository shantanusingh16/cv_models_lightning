import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target, valid_lengths=None):
        if valid_lengths is not None:
            # Mask for valid detections
            mask = torch.zeros_like(target, dtype=torch.bool)
            for i, length in enumerate(valid_lengths):
                mask[i, :length] = True
            
            # Apply mask to input and target
            input = input[mask]
            target = target[mask]
        
        return F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index)