from functools import partial

import torch
import torch.nn as nn


cfg = {
    'VGG9tinier': [16, 'M', 24, 'M', 32, 32, 'M', 40, 40, 'M'],
    'VGG9tiny': [16, 'M', 32, 'M', 48, 48, 'M', 64, 64, 'M'],
    'VGG9light': [32, 'M', 48, 'M', 64, 64, 'M', 128, 128, 'M'],
    'VGG11light': [32, 'M', 48, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16Narrow': [64, 64, 'M', 128, 128, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M'],
    'VGG16BnDropV2': ['64d3', 64, 'M', '128d4', 128, 'M', '256d4', '256d4', 256, 'M', '512d4', '512d4', 512, 'M', '512d4', '512d4', 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGFullyConv(nn.Module):
    def __init__(self, vgg_name, num_classes=10, dropout=False):
        super(VGGFullyConv, self).__init__()
        self.vgg_name = vgg_name
        self.num_classes = num_classes

        out_filters = None
        i = 1
        while out_filters is None:
            candidate_value = cfg[vgg_name][-i]
            if isinstance(candidate_value, int):
                out_filters = candidate_value
            i+=1

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = None
        # Support old structure when classifier is not used:
        if vgg_name =='VGG16BnDropV2':
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, self.num_classes),
            )
            self.fc = None
            self.dropout = None
        else:
            self.fc = nn.Linear(out_filters, num_classes)
            self.dropout = None
            if dropout:
                self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.features(x)
        # global pooling
        out = out.mean(-1).mean(-1)
        # Support old structure when classifier is not used:
        if self.classifier is None:
            if self.dropout is not None:
                out = self.dropout(out)
            out = self.fc(out)
        else:
            out = self.classifier(out)
        return out

    @staticmethod
    def parse_conv_channels(out_channels):
        dropout = None
        if not isinstance(out_channels, int):
            if 'd' in out_channels:
                dropout = float('0.' + out_channels.split('d')[1])
                out_channels = int(out_channels.split('d')[0])
            else:
                raise ValueError('Unrecognized format for out channels ' + out_channels)
        return out_channels, dropout

    def _make_layers(self, cfg):
        out_channels, dropout = self.parse_conv_channels(cfg[0])
        layers = [ConvBNRelu(3, out_channels, dropout=dropout)]
        downsample_rate = 1
        in_channels = out_channels
        for out_channels in cfg[1:]:
            if out_channels == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                downsample_rate *= 2
            else:
                out_channels, dropout  = self.parse_conv_channels(out_channels)
                layers += [ConvBNRelu(in_channels, out_channels, dropout=dropout)]
                in_channels = out_channels
        return nn.Sequential(*layers)


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=None):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        if self.dropout is not None:
            out = self.dropout(out)
        return out
