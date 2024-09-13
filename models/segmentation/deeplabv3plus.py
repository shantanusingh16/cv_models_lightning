import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.mobilenetv2 import mobilenetv2
from ..backbones.mobilenetv3 import mobilenetv3
from .deeplabv3 import ASPP

class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(DeepLabV3PlusDecoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        
        self.output = nn.Sequential(
            nn.Conv2d(304, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1, stride=1)
        )

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.output(x)
        return x

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, backbone):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = backbone
        
        if isinstance(backbone, mobilenetv2):
            self.low_level_features = backbone.model.features[0:4]
            self.high_level_features = backbone.model.features[4:]
            aspp_in_channels = 1280
            low_level_channels = 24
        elif isinstance(backbone, mobilenetv3):
            if backbone.model.architecture == 'large':
                self.low_level_features = backbone.model.features[0:6]
                self.high_level_features = backbone.model.features[6:]
                aspp_in_channels = 960
                low_level_channels = 40
            else:  # small
                self.low_level_features = backbone.model.features[0:4]
                self.high_level_features = backbone.model.features[4:]
                aspp_in_channels = 576
                low_level_channels = 24
        else:
            raise ValueError("Unsupported backbone")

        self.aspp = ASPP(aspp_in_channels, [12, 24, 36])
        self.decoder = DeepLabV3PlusDecoder(low_level_channels, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Backbone
        low_level_feat = self.low_level_features(x)
        x = self.high_level_features(low_level_feat)

        # ASPP
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x, low_level_feat)

        # Upsample
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return x

def create_deeplabv3plus(num_classes, backbone='mobilenetv2', pretrained=False):
    if backbone == 'mobilenetv2':
        base_net = mobilenetv2(pretrained=pretrained)
    elif backbone == 'mobilenetv3':
        base_net = mobilenetv3(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    return DeepLabV3Plus(num_classes, base_net)