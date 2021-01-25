import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck, _resnet, conv1x1, conv3x3


class CifarResnet(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dropout=None):
        super(CifarResnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.dropout = None if dropout is None else nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)

        return x

    def compute_flops_memory(self, include_fc=True):
        cost = get_conv_cost(self.conv1)
        flops_cost = cost
        downsample = 1
        memory_cost = cost

        # check if BasicBlock or Bottleneck
        num_convs = 3 if hasattr(self.layer1[0], 'conv3') else 2

        for i in range(1, 4):
            if i > 1 :
                downsample *= 2
            layer = getattr(self, 'layer'+str(i))
            if layer[0].downsample is not None:
                cost = get_conv_cost(layer[0].downsample[0])
                flops_cost += cost / (downsample ** 2)
                memory_cost += cost
            block_num = 0
            while hasattr(layer, str(block_num)):
                for j in range(1, num_convs + 1):
                    conv = getattr(layer[block_num], 'conv' + str(j))
                    cost = get_conv_cost(conv)
                    flops_cost += cost / (conv.groups * downsample ** 2)
                    memory_cost += cost
                block_num += 1

        if include_fc:
            fc_cost = self.fc.in_features * self.fc.out_features
            flops_cost += fc_cost / (32**2)
            memory_cost += fc_cost

        return flops_cost, memory_cost


def get_conv_cost(m):
    assert isinstance(m, nn.Conv2d)
    res = m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] / (m.dilation[0] * m.dilation[1])
    return res


def resnet56(num_classes):
    return CifarResnet(BasicBlock, [9,9,9], num_classes)

def resnet56_dropout_20(num_classes):
    return CifarResnet(BasicBlock, [9,9,9], num_classes, dropout=0.2)

def resnet56_dropout_10(num_classes):
    return CifarResnet(BasicBlock, [9,9,9], num_classes, dropout=0.1)

def resnet56_dropout_50(num_classes):
    return CifarResnet(BasicBlock, [9,9,9], num_classes, dropout=0.5)

def resnet56_dropout_80(num_classes):
    return CifarResnet(BasicBlock, [9,9,9], num_classes, dropout=0.8)
