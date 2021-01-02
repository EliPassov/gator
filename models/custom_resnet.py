from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, conv1x1, conv3x3, resnet18, resnet34, resnet50

from models.cifar_resnet import CifarResnet


class CustomBasicBlock(BasicBlock):
    def __init__(self, in_channels, conv1_out, conv2_out, stride=1, downsample=None, groups=1, dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(in_channels, conv1_out, stride)
        self.bn1 = norm_layer(conv1_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(conv1_out, conv2_out)
        self.bn2 = norm_layer(conv2_out)
        self.downsample = downsample
        self.stride = stride


class CustomBottleNeck(Bottleneck):
    def __init__(self, in_channels, conv1_out, conv2_out, conv3_out, stride=1, downsample=None, groups=1, dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, conv1_out)
        self.bn1 = norm_layer(conv1_out)
        self.conv2 = conv3x3(conv1_out, conv2_out, stride, groups, dilation)
        self.bn2 = norm_layer(conv2_out)
        self.conv3 = conv1x1(conv2_out, conv3_out)
        self.bn3 = norm_layer(conv3_out)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


class CustomResNet(ResNet):
    def __init__(self, block, channels_config, num_classes=1000, zero_init_residual=False, groups=1,
                 replace_stride_with_dilation=None, norm_layer=None, cifar_resnet=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.cifar_resnet=cifar_resnet
        self.num_layers = 3 if cifar_resnet else 4

        self.groups = groups
        self.channels_config = channels_config
        first_channels = channels_config['conv1']
        if cifar_resnet:
            self.conv1 = nn.Conv2d(3, first_channels, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, first_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(first_channels)
        self.relu = nn.ReLU(inplace=True)
        if cifar_resnet:
            # a bit misleading but makes it generic
            self.maxpool = nn.Sequential()
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.prev_block_out = first_channels
        self.current_channel = 0

        # the 64, 128, 256, 512 input is ignored and only kept for similarity for original code
        self.layer1 = self._make_layer_custom(block, channels_config['layer1'])
        self.layer2 = self._make_layer_custom(block, channels_config['layer2'], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer_custom(block, channels_config['layer3'], stride=2, dilate=replace_stride_with_dilation[1])
        if self.cifar_resnet:
            self.layer4 = nn.Sequential()
        else:
            self.layer4 = self._make_layer_custom(block, channels_config['layer4'], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.prev_block_out, num_classes)

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

    def get_and_check_channels(self, num_channels, block_config):
        channels = [self.prev_block_out]
        for i in range(1, num_channels + 1):
            channels.append(block_config['conv' + str(i)])
        self.prev_block_out = channels[-1]
        return channels

    def _make_layer_custom(self, block, layer_config, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        num_convs = 3 if block == CustomBottleNeck else 2
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or block == CustomBottleNeck:
            downsample_out_channels =  layer_config['0']['conv'+str(num_convs)]
            downsample = nn.Sequential(
                conv1x1(self.prev_block_out, downsample_out_channels, stride),
                norm_layer(downsample_out_channels),
            )

        layers = []
        channels = self.get_and_check_channels(num_convs, layer_config['0'])
        layers.append(block(*channels, stride, downsample, self.groups, previous_dilation, norm_layer))
        self.prev_block_out = channels[-1]
        block_index = 1
        while str(block_index) in layer_config:
            channels = self.get_and_check_channels(num_convs, layer_config[str(block_index)])
            layers.append(block(*channels, groups=self.groups, dilation=self.dilation, norm_layer=norm_layer))
            block_index += 1

        return nn.Sequential(*layers)

    def compute_flops_memory(self):
        cost = get_conv_cost(self.conv1)
        if self.cifar_resnet:
            flops_cost = cost
            downsample = 1
        else:
            flops_cost = cost / 4
            downsample = 4
        memory_cost = cost

        # check if BasicBlock or Bottleneck
        num_convs = 3 if hasattr(self.layer1[0], 'conv3') else 2

        for i in range(1, self.num_layers + 1):
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
                    cost = get_conv_cost(getattr(layer[block_num], 'conv' + str(j)))
                    flops_cost += cost / (downsample ** 2)
                    memory_cost += cost
                block_num += 1
        return flops_cost, memory_cost


def get_conv_cost(m):
    assert isinstance(m, nn.Conv2d)
    res = m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] / (m.dilation[0] * m.dilation[1])
    return res


def filter_mapping_from_default_resnet(net):
    assert isinstance(net, ResNet) or isinstance(net, CifarResnet)
    num_layer = 3 if isinstance(net, CifarResnet) else 4
    channels_config = { 'conv1': net.conv1.out_channels}
    for l in range(1, num_layer + 1):
        layer_name = 'layer' + str(l)
        layer = getattr(net, layer_name)
        layer_config = {}
        for j, block in enumerate(layer.children()):
            num_convs = 2 if isinstance(block, BasicBlock) else 3
            layer_config[str(j)] = {}
            for c in range(1, num_convs + 1):
                conv_name = 'conv'+str(c)
                layer_config[str(j)][conv_name] = getattr(getattr(block, conv_name), 'out_channels')
        channels_config[layer_name] = layer_config
    return channels_config


def custom_resnet_18(channels_config, num_classes=1000):
    return CustomResNet(CustomBasicBlock, channels_config, num_classes)


def custom_resnet_34(channels_config, num_classes=1000):
    return CustomResNet(CustomBasicBlock, channels_config, num_classes)


def custom_resnet_50(channels_config, num_classes=1000):
    return CustomResNet(CustomBottleNeck, channels_config, num_classes)


def custom_resnet_56(channels_config, num_classes=10):
    return CustomResNet(CustomBasicBlock, channels_config, num_classes, cifar_resnet=True)


if __name__ == '__main__':
    # import torch
    # channels_config = torch.load('/home/eli/Eli/Training/Imagenet/resnet50/resnet50_pre_0_995_w_0_25_gm_0_2_w_0_5/net_e_80_custom_resnet')['channels_config']
    # custom_net = custom_resnet_50(channels_config, num_classes=1000).cuda()

    # from models.cifar_resnet import resnet56
    # net = resnet56(10)
    net = resnet50(False)
    channels_config = filter_mapping_from_default_resnet(net)
    custom_net = custom_resnet_50(channels_config)
    # custom_net = custom_resnet_56(channels_config)
    custom_net = custom_net.cuda()
    import torch
    custom_net.eval()
    sample = torch.rand((1,3,224,224)).cuda()
    res = custom_net(sample)

    print(custom_net.compute_flops_memory())
    print(custom_net)

    # import torch
    # from models.gates_mapper import ResNetGatesModulesMapper
    # from models.wrapped_gated_models import ResNet50_gating
    # from models.gate_wrapped_module import create_conv_channels_dict
    #
    # net_name = 'ResNet50_gating'
    # weight_path = '/home/eli/Eli/Training/Imagenet/resnet50/resnet50_pre_0_995_w_0_25_gm_0_2_w_0_5/net_e_80'
    #
    # net, aaa = ResNet50_gating(1000)
    # state_dict = torch.load(weight_path)['state_dict']
    # state_dict = {k[7:]: v for k,v in state_dict.items()}
    # net.load_state_dict(state_dict)
    # mapper = ResNetGatesModulesMapper(net.net, False, map_for_replacement=True)
    #
    # channels_config = create_conv_channels_dict(mapper, net.forward_hooks)
    # custom_net = custom_resnet_50(channels_config)
    #
    # custom_net = custom_net.cuda()
    #
    # custom_net.eval()
    # sample = torch.rand((1,3,224,224)).cuda()
    # res = custom_net(sample)
    #
    # print(res)

