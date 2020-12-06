from functools import partial
from collections import defaultdict

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck


class ConvInfo:
    def __init__(self, conv, downsample, multiplier=1):
        self.factor = conv.kernel_size[0] * conv.kernel_size[1]
        self.downsample = downsample
        self.multiplier = multiplier

    @property
    def flop_factor(self):
        return self.multiplier * self.factor / (self.downsample ** 2)

    @property
    def memory_factor(self):
        return self.multiplier * self.factor


class ChannelsHyperEdge:
    def __init__(self, channels, conv_is_in_tuples=None, conv_names=None):
        self.channels = channels
        if isinstance(conv_is_in_tuples, list):
            for a in conv_is_in_tuples:
                assert isinstance(a, tuple)
        else:
            assert conv_is_in_tuples is None

        self.convs_and_sides = [] if conv_is_in_tuples is None else conv_is_in_tuples
        if conv_names is None:
            self.conv_names = None
        else:
            assert isinstance(conv_names, list)
            self.conv_names = conv_names

    def add(self, conv, is_in, conv_name=None):
        self.convs_and_sides.append((conv, is_in))
        if conv_name is not None:
            if self.conv_names is None and len(self.convs_and_sides) > 1:
                raise ValueError('Cannot add a convolution with name when previous convolutions had no name')
            else:
                self.conv_names.append(conv_name)
        else:
            if self.conv_names is not None:
                raise ValueError('Attempting to add a convolution with no name when previous convolutions had a name')

    @property
    def convs(self):
        return [t[0] for t in self.convs_and_sides]

    def __hash__(self):
        return id(self)


class ReferenceToSetter:
    """ A wrapper for the parent object reference and attribute name to enable to store a replacement call in a lambda"""
    def __init__(self, parent, attribute_name):
        self.parent = parent
        self.attribute_name = attribute_name

    def __call__(self, value):
        setattr(self.parent, self.attribute_name, value)


class GatesModulesMapper:
    def __init__(self, net, no_last_conv=False, map_for_replacement=False):
        self.net = net
        self.no_last_conv = no_last_conv
        # conv to info
        self.conv_info = {}
        # map of associated modules (batch norm) to conv id
        self.alias_map = {}
        self.reverse_alias_map = {}
        # list of hyper edges which contain conv modules and in/out tuple
        self.hyper_edges = []
        # dict of (conv, in/out) to hyper edge
        self.hyper_edges_map = {}

        self.map_modules(no_last_conv)
        self.post_map()

        self.replacement_map = None
        if map_for_replacement:
            self.map_modules_for_replacement()

    def create_hyper_edge(self, channels, conv_is_in_tuples, conv_names):
        hyper_edge = ChannelsHyperEdge(channels, conv_is_in_tuples, conv_names)
        for conv, is_in in conv_is_in_tuples:
            self.hyper_edges_map[(conv, is_in)] = hyper_edge
        return hyper_edge

    def add_to_hyper_edge(self, hyper_edge, conv, is_in, conv_name=None):
        hyper_edge.add(conv, is_in, conv_name)
        self.hyper_edges_map[(conv, is_in)] = hyper_edge

    def map_modules(self, no_last_conv=True):
        self.conv_index = 0
        self.map_modules_inner(no_last_conv)

    def map_modules_inner(self, no_last_conv=True):
        raise NotImplementedError('base method')

    def map_modules_for_replacement(self):
        raise NotImplementedError('base method')

    def apply_replacement(self, module, new_module):
        raise NotImplementedError('base method')

    def post_map(self):
        self.reverse_alias_map = {(m,s):c for (c,s),m in self.alias_map.items()}

    @property
    def flop_cost(self):
        cost = 0
        for c,i in self.conv_info.items():
            cost += c.in_channels * c.out_channels * i.factor / (i.downsample ** 2)
        return cost

    @property
    def memory_cost(self):
        cost = 0
        for c,i in self.conv_info.items():
            cost += c.in_channels * c.out_channels * i.factor
        return cost


class NaiveSequentialGatesModulesMapper(GatesModulesMapper):
    def map_modules_inner(self, no_last_conv=True):
        downsample = 1
        last_conv = None
        for m in self.net.modules():
            if hasattr(m, 'stride'):
                stride = m.stride[0] if isinstance(m.stride, tuple) else m.stride
                if stride > 1:
                    downsample *= stride
            if isinstance(m, nn.Conv2d):
                self.conv_info[m] = ConvInfo(m, downsample)
                if last_conv is not None:
                    t1 = (last_conv, False)
                    t2 = (m, True)
                    hyper_edge = self.create_hyper_edge(m.in_channels, [t1, t2])
                    self.hyper_edges.append(hyper_edge)
                last_conv = m
            #optional
            if isinstance(m, nn.BatchNorm2d):
                self.alias_map[(last_conv, False)] = m
        if not no_last_conv:
            t1 = (last_conv, False)
            last_hyper_edge = self.create_hyper_edge(last_conv.out_channels, [t1])
            self.hyper_edges.append(last_hyper_edge)


class ResNetGatesModulesMapper(GatesModulesMapper):
    def map_modules_inner(self, no_last_conv=True):
        self.conv_info[self.net.conv1] = ConvInfo(self.net.conv1, 2)
        self.alias_map[(self.net.conv1, False)] = self.net.bn1
        inter_block_hyper_edge = self.create_hyper_edge(self.net.conv1.out_channels, [(self.net.conv1, False)], ['conv1'])
        downsample = 4 # stride 2 in conv + maxpool

        for i in range(1, 5):
            layer_name = 'layer' + str(i)
            layer = getattr(self.net, layer_name)
            for j, block in enumerate(layer.children()):
                # add input to residual block to inter block edge
                conv_prefix = layer_name + '.' + str(j) + '.'
                self.add_to_hyper_edge(inter_block_hyper_edge, block.conv1, True, conv_prefix + 'conv1')
                if j == 0 and (i > 1 or isinstance(block, Bottleneck)):
                    down_conv = block.downsample[0]
                    if i > 1: # exclude the first "downsample" bottleneck which is not really downsampling
                        downsample *= 2
                    self.conv_info[down_conv] = ConvInfo(down_conv, downsample)
                    self.alias_map[(down_conv, False)] = block.downsample[1]

                    # add downsmaple input to inter block edge
                    self.add_to_hyper_edge(inter_block_hyper_edge, down_conv, True, conv_prefix + 'downsample')
                    self.hyper_edges.append(inter_block_hyper_edge)

                    # create new inter block edge
                    inter_block_hyper_edge = self.create_hyper_edge(down_conv.out_channels,
                                                                    [(down_conv, False)],
                                                                    [conv_prefix + 'downsample'])

                self.conv_info[block.conv1] = ConvInfo(block.conv1, downsample)
                self.conv_info[block.conv2] = ConvInfo(block.conv2, downsample)
                self.alias_map[(block.conv1, False)] = block.bn1
                self.alias_map[(block.conv2, False)] = block.bn2

                # create small inner edge between two blocks
                inner_hyper_edge = self.create_hyper_edge(block.conv1.out_channels,
                                                          [(block.conv1, False), (block.conv2, True)],
                                                          [conv_prefix + 'conv1', conv_prefix + 'conv2'])
                self.hyper_edges.append(inner_hyper_edge)

                if isinstance(block, BasicBlock):
                    # add second block output
                    self.add_to_hyper_edge(inter_block_hyper_edge, block.conv2, False, conv_prefix + 'conv2')
                else:
                    assert isinstance(block, Bottleneck)
                    self.conv_info[block.conv3] = ConvInfo(block.conv3, downsample)
                    self.alias_map[(block.conv3, False)] = block.bn3

                    # create second small inner edge between two blocks
                    inner_hyper_edge = self.create_hyper_edge(block.conv2.out_channels,
                                                              [(block.conv2, False), (block.conv3, True)],
                                                              [conv_prefix + 'conv2', conv_prefix + 'conv3'])
                    self.hyper_edges.append(inner_hyper_edge)

                    # add third block output
                    self.add_to_hyper_edge(inter_block_hyper_edge, block.conv3, False, conv_prefix + 'conv3')

        if not no_last_conv:
            self.hyper_edges.append(inter_block_hyper_edge)
        else:
            # remove mapping to the last hyper edge, as it is not being added
            for conv_and_side in inter_block_hyper_edge.convs_and_sides:
                del self.hyper_edges_map[conv_and_side]

    def map_modules_for_replacement(self):
        self.replacement_map = {}
        self.replacement_map[self.net.conv1] = ReferenceToSetter(self.net, 'conv1')
        self.replacement_map[self.net.bn1] = ReferenceToSetter(self.net, 'bn1')
        for i in range(1, 5):
            layer = getattr(self.net, 'layer' + str(i))
            for j, block in enumerate(layer.children()):
                if j == 0 and (i > 1 or isinstance(block, Bottleneck)):
                    self.replacement_map[block.downsample[0]] = ReferenceToSetter(block.downsample, '0')
                    self.replacement_map[block.downsample[1]] = ReferenceToSetter(block.downsample, '1')
                self.replacement_map[block.conv1] = ReferenceToSetter(block, 'conv1')
                self.replacement_map[block.bn1] = ReferenceToSetter(block, 'bn1')
                self.replacement_map[block.conv2] = ReferenceToSetter(block, 'conv2')
                self.replacement_map[block.bn2] = ReferenceToSetter(block, 'bn2')
                if isinstance(block, Bottleneck):
                    self.replacement_map[block.conv3] = ReferenceToSetter(block, 'conv3')
                    self.replacement_map[block.bn3] = ReferenceToSetter(block, 'bn3')
        self.replacement_map[self.net.fc] = ReferenceToSetter(self.net, 'fc')

    def apply_replacement(self, module, new_module, print_replacements=False):
        assert self.replacement_map is not None, "Mapper needs to be instantiated with map_for_replacement=True"
        replace_key(self.conv_info, module, new_module, not isinstance(module, nn.Conv2d))
        replace_binary_second_tuple_key(self.alias_map, module, new_module, True)
        replace_binary_second_tuple_key(self.reverse_alias_map, module, new_module, True)
        replace_binary_second_tuple_key(self.hyper_edges_map, module, new_module, True)
        ref_to_set = replace_key(self.replacement_map, module, new_module, not isinstance(module, nn.Conv2d))
        ref_to_set(new_module)
        if print_replacements:
            print('replaced reference {} with reference {}'.format(id(module), id(new_module)))
        assert module not in self.replacement_map
        assert new_module in self.replacement_map


def replace_key(d, key, new_key, optional_key=False):
    if optional_key and key not in d:
        # this should be fine as the error is not raised
        return KeyError()
    val = d[key]
    d[new_key] = val
    del d[key]
    return val


def replace_tuple_key(d, options, position, key, new_key, optional_key=False):
    results = []
    for option in options:
        list_key = list(option) if isinstance(option, tuple) else [option]
        new_list_key = list_key.copy()
        new_list_key.insert(position, new_key)
        new_full_key = tuple(new_list_key)
        list_key.insert(position, key)
        full_key = tuple(list_key)
        res = replace_key(d, full_key, new_full_key, True)
        if not isinstance(res, KeyError):
            results.append(res)
    if len(results) == 0 and not optional_key:
        raise KeyError('unable to find any key combination of {} with {} at position {}'.format(key, options, position))
    return results


def replace_binary_second_tuple_key(d, key, new_key, optional_key=False):
    return replace_tuple_key(d, [True, False], 0, key, new_key, optional_key)
