import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.mobilenetv2 import mobilenetv2
from ..backbones.mobilenetv3 import mobilenetv3

class SSD(nn.Module):
    def __init__(self, num_classes, backbone):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        
        # Extra feature layers
        self.extras = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1280, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
        ])

        # Localization and confidence layers
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(1280, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        ])

        self.conf_layers = nn.ModuleList([
            nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(1280, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)
        ])

    def forward(self, x):
        detection_feed = []
        confidence, location = [], []
        
        # Get features from backbone
        _, features = self.backbone(x)
        
        # Use relevant features from backbone
        x = features[-1]
        detection_feed.append(features[-2])
        
        # Apply extra layers
        for layer in self.extras:
            x = layer(x)
            detection_feed.append(x)
        
        # Apply localization and confidence layers
        for i, feat in enumerate(detection_feed):
            location.append(self.loc_layers[i](feat).permute(0, 2, 3, 1).contiguous())
            confidence.append(self.conf_layers[i](feat).permute(0, 2, 3, 1).contiguous())
        
        location = torch.cat([o.view(o.size(0), -1) for o in location], 1)
        confidence = torch.cat([o.view(o.size(0), -1) for o in confidence], 1)
        
        return location, confidence

def create_ssd(num_classes, backbone='mobilenetv2', pretrained=False):
    if backbone == 'mobilenetv2':
        base_net = mobilenetv2(pretrained=pretrained)
    elif backbone == 'mobilenetv3':
        base_net = mobilenetv3(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    return SSD(num_classes, base_net)