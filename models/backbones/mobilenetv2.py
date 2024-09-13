import torch
import torch.nn as nn
import timm

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, pretrained=False):
        super(MobileNetV2, self).__init__()
        self.model = timm.create_model('mobilenetv2_100', pretrained=pretrained, num_classes=num_classes)
        
        # Adjust the last layer if width_mult is different from 1.0
        if width_mult != 1.0:
            last_channel = int(1280 * width_mult)
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.model.classifier[1].in_features, last_channel),
                nn.Linear(last_channel, num_classes)
            )

    def forward(self, x):
        features = []
        for name, module in self.model.named_children():
            x = module(x)
            if name in ['blocks', 'conv_stem', 'bn1']:
                features.append(x)
        return x, features

def mobilenetv2(num_classes=1000, width_mult=1.0, pretrained=False):
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult, pretrained=pretrained)