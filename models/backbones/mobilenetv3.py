import torch
import torch.nn as nn
import timm

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, mode='large', width_mult=1.0, pretrained=False):
        super(MobileNetV3, self).__init__()
        assert mode in ['large', 'small'], "Mode must be either 'large' or 'small'"
        
        model_name = f'mobilenetv3_{mode}_100' if width_mult == 1.0 else f'mobilenetv3_{mode}_075'
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # Adjust the last layer if width_mult is different from 1.0 or 0.75
        if width_mult not in [1.0, 0.75]:
            last_channel = int(1280 * width_mult) if mode == 'large' else int(1024 * width_mult)
            self.model.classifier = nn.Sequential(
                nn.Linear(self.model.classifier[0].in_features, last_channel),
                nn.Hardswish(inplace=True),
                nn.Dropout(0.2, inplace=True),
                nn.Linear(last_channel, num_classes)
            )

    def forward(self, x):
        features = []
        for name, module in self.model.named_children():
            x = module(x)
            if name in ['blocks', 'conv_stem', 'bn1']:
                features.append(x)
        return x, features

def mobilenetv3(num_classes=1000, mode='large', width_mult=1.0, pretrained=False):
    return MobileNetV3(num_classes=num_classes, mode=mode, width_mult=width_mult, pretrained=pretrained)