import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.mobilenetv2 import mobilenetv2
from ..backbones.mobilenetv3 import mobilenetv3

def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Depthwise Separable 2D Convolution."""
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class SSDLite(nn.Module):
    def __init__(self, num_classes, backbone):
        super(SSDLite, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        
        # Extra feature layers
        self.extras = nn.ModuleList([
            SeperableConv2d(1280, 512, kernel_size=3, stride=2, padding=1),
            SeperableConv2d(512, 256, kernel_size=3, stride=2, padding=1),
            SeperableConv2d(256, 256, kernel_size=3, stride=2, padding=1),
            SeperableConv2d(256, 128, kernel_size=3, stride=2, padding=1)
        ])

        # Localization and confidence layers
        self.loc_layers = nn.ModuleList([
            SeperableConv2d(320, 4 * 4, kernel_size=3, padding=1),
            SeperableConv2d(1280, 6 * 4, kernel_size=3, padding=1),
            SeperableConv2d(512, 6 * 4, kernel_size=3, padding=1),
            SeperableConv2d(256, 6 * 4, kernel_size=3, padding=1),
            SeperableConv2d(256, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(128, 4 * 4, kernel_size=1)
        ])

        self.conf_layers = nn.ModuleList([
            SeperableConv2d(320, 4 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(1280, 6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(512, 6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(256, 6 * num_classes, kernel_size=3, padding=1),
            SeperableConv2d(256, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(128, 4 * num_classes, kernel_size=1)
        ])

    def forward(self, x):
        detection_feed = []
        confidence, location = [], []
        
        # Get features from backbone
        _, features = self.backbone(x)
        
        # Use relevant features from backbone
        x = features[-1]
        detection_feed.extend(features[-2:])
        
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

def create_ssdlite(num_classes, backbone='mobilenetv2', pretrained=False):
    if backbone == 'mobilenetv2':
        base_net = mobilenetv2(pretrained=pretrained)
    elif backbone == 'mobilenetv3':
        base_net = mobilenetv3(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    return SSDLite(num_classes, base_net)