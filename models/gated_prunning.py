import torch
from torch import nn


def prune_conv(conv, keep_mask, is_input, bn=None):
    if is_input:
        assert bn is None
        new_conv = nn.Conv2d(int(keep_mask.sum().item()), conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                             conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
        # reshape required as slicing adds a dimension for some weird reason
        print(new_conv, '\n','-'*20,'\n',keep_mask)
        new_conv.weight = nn.Parameter(conv.weight.data[:, keep_mask.nonzero(), ...].reshape(new_conv.weight.shape))
        if conv.bias is not None:
            new_conv.bias = nn.Parameter(conv.bias.data)
    else:
        new_conv = nn.Conv2d(conv.in_channels, int(keep_mask.sum().item()), conv.kernel_size, conv.stride, conv.padding,
                             conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode)
        new_conv.weight = nn.Parameter(conv.weight[keep_mask.nonzero(), ...].reshape(new_conv.weight.shape))
        if conv.bias is not None:
            new_conv.bias = nn.Parameter(conv.bias.data[keep_mask.nonzero()].reshape(new_conv.bias.shape))
        elif bn is not None:
            new_bn = nn.BatchNorm2d(int(keep_mask.sum().item()))
            new_bn.bias = nn.Parameter(bn.bias.data[keep_mask.nonzero()].reshape(new_bn.bias.shape))
            new_bn.weight = nn.Parameter(bn.weight.data[keep_mask.nonzero()].reshape(new_bn.bias.shape))
            new_bn.running_mean = nn.Parameter(bn.running_mean.data[keep_mask.nonzero()].reshape(new_bn.bias.shape))
            new_bn.running_var = nn.Parameter(bn.running_var.data[keep_mask.nonzero()].reshape(new_bn.bias.shape))

    if bn is not None:
        return new_conv, new_bn

    return new_conv

def prune_fc(fc, keep_list, is_input):
    reduced_features_len = int(keep_list.sum().item())
    if is_input:
        new_fc = nn.Linear(reduced_features_len, fc.out_features, fc.bias is not None)
        new_fc.weight = nn.Parameter(fc.weight.data[:, keep_list.nonzero()].reshape(new_fc.weight.shape))
    else:
        new_fc = nn.Linear(fc.in_features, reduced_features_len, fc.bias is not None)
        new_fc.weight = nn.Parameter(fc.weight.data[keep_list.nonzero(), :].reshape(new_fc.weight.shape))
        if fc.bias is not None:
            new_fc.bias = nn.Parameter(fc.bias.data[keep_list.nonzero()].reshape(new_fc.bias.shape))
    return new_fc


def prune_for_single_hook_individual_sides(h, channels_to_keep, mapper, print_replacements=False):
    for m, s in h.modules_and_sides:
        bn = None
        if not s:
            conv = mapper.reverse_alias_map.get((m, s), m)
            if conv != m:
                bn = m
        if bn is None:
            if print_replacements:
                print('Replacing conv {} side {} shape {}'.format(id(conv), 'in' if s else 'out', conv.weight.shape))

            new_conv = prune_conv(conv, channels_to_keep, s)
            mapper.apply_replacement(conv, new_conv)
            if print_replacements:
                print('Replaced with conv {} shape {}'.format(id(conv), conv.weight.shape))
        else:
            assert not s
            if print_replacements:
                print('Replacing conv {} side {} shape {}'.format(id(conv), 'in' if s else 'out', conv.weight.shape))
                print('Replacing bn {} shape {}'.format(id(bn), bn.weight.shape))
            new_conv, new_bn = prune_conv(conv, channels_to_keep, False, bn)
            mapper.apply_replacement(conv, new_conv)
            mapper.apply_replacement(bn, new_bn)
            if print_replacements:
                print('Replaced with conv {} shape {}'.format(id(new_conv), new_conv.weight.shape))
                print('Replaced with bn {} shape {}'.format(id(new_bn), new_bn.weight.shape))
    if print_replacements:
        print('-'*40)


def prune_net_with_hooks_individual_sides(net_with_hooks, mapper, print_replacements=False):
    hooks = net_with_hooks.forward_hooks

    for h in hooks:
        channels_to_keep  = h.gating_module.active_channels_mask
        prune_for_single_hook_individual_sides(h, channels_to_keep, mapper, print_replacements)


def collect_prunes_for_hook(h, channels_to_keep, mapper, print_replacements=False):
    for m, s in h.modules_and_sides:
        bn = None
        if not s:
            conv = mapper.reverse_alias_map.get((m, s), m)
            if conv != m:
                bn = m
        if bn is None:
            if print_replacements:
                print(
                    'Replacing conv {} side {} shape {}'.format(id(conv), 'in' if s else 'out', conv.weight.shape))

            new_conv = prune_conv(conv, channels_to_keep, s)
            mapper.apply_replacement(conv, new_conv)
            if print_replacements:
                print('Replaced with conv {} shape {}'.format(id(conv), conv.weight.shape))
        else:
            assert not s
            if print_replacements:
                print(
                    'Replacing conv {} side {} shape {}'.format(id(conv), 'in' if s else 'out', conv.weight.shape))
                print('Replacing bn {} shape {}'.format(id(bn), bn.weight.shape))
            new_conv, new_bn = prune_conv(conv, channels_to_keep, False, bn)
            mapper.apply_replacement(conv, new_conv)
            mapper.apply_replacement(bn, new_bn)
            if print_replacements:
                print('Replaced with conv {} shape {}'.format(id(new_conv), new_conv.weight.shape))
                print('Replaced with bn {} shape {}'.format(id(new_bn), new_bn.weight.shape))
    if print_replacements:
        print('-' * 40)


class ConvToPrune:
    def __init__(self, conv, bn=None):
        self.conv = conv
        self.bn = bn
        self.in_channels_to_keep = None
        self.out_channels_to_keep = None

    def add_in_prune(self, in_channels_to_keep):
        if self.in_channels_to_keep is not None:
            raise ValueError('In channels already set for purnning')
        self.in_channels_to_keep = in_channels_to_keep

    def add_out_prune(self, out_channels_to_keep):
        if self.out_channels_to_keep is not None:
            raise ValueError('Out channels already set for purnning')
        self.out_channels_to_keep = out_channels_to_keep

    def get_pruned_conv(self):
        if self.in_channels_to_keep is None and self.out_channels_to_keep is None:
            raise ValueError('Both in and out channels are not set for pruning')

        new_in_channels = self.in_channels_to_keep if self.in_channels_to_keep is not None else torch.ones(self.conv.in_channels).long()
        new_out_channels = self.out_channels_to_keep if self.out_channels_to_keep is not None else torch.ones(self.conv.out_channels).long()
        new_in_channels_count = int(new_in_channels.sum().item())
        new_out_channels_count = int(new_out_channels.sum().item())

        new_conv = nn.Conv2d(new_in_channels_count, new_out_channels_count, self.conv.kernel_size, self.conv.stride,
                             self.conv.padding, self.conv.dilation, self.conv.groups, self.conv.bias is not None,
                             self.conv.padding_mode)

        # cannot index on two dimensions at the same time, hence two steps
        new_weights = self.conv.weight.data[new_out_channels.nonzero(), ...].reshape(new_out_channels_count,
                                                                                     self.conv.weight.size(1),
                                                                                     self.conv.weight.size(2),
                                                                                     self.conv.weight.size(3))
        new_weights = new_weights[:, new_in_channels.nonzero(), ...].reshape(new_conv.weight.shape)
        new_conv.weight = nn.Parameter(new_weights)
        # new_conv.weight = nn.Parameter(self.conv.weight[new_in_channels.nonzero(), new_out_channels.nonzero(), ...].reshape(new_conv.weight.shape))

        new_bn = None
        if self.conv.bias is not None:
            new_conv.bias = nn.Parameter(self.conv.bias.data[new_out_channels.nonzero()].reshape(new_conv.bias.shape))
        elif self.bn is not None:
            new_bn = nn.BatchNorm2d(new_out_channels_count)
            new_bn.bias = nn.Parameter(self.bn.bias.data[new_out_channels.nonzero()].reshape(new_bn.bias.shape))
            new_bn.weight = nn.Parameter(self.bn.weight.data[new_out_channels.nonzero()].reshape(new_bn.bias.shape))
            # new_bn.running_mean = nn.Parameter(self.bn.running_mean.data[new_out_channels.nonzero()].reshape(new_bn.bias.shape))
            # new_bn.running_var = nn.Parameter(self.bn.running_var.data[new_out_channels.nonzero()].reshape(new_bn.bias.shape))
            new_bn.running_mean = self.bn.running_mean.data[new_out_channels.nonzero()].reshape(new_bn.bias.shape)
            new_bn.running_var = self.bn.running_var.data[new_out_channels.nonzero()].reshape(new_bn.bias.shape)

        return new_conv, new_bn


def prune_net_with_hooks(net_with_hooks, mapper, apply_replacements=False, print_replacements=False):
    hooks = net_with_hooks.forward_hooks
    prune_map ={}

    for h in hooks:
        channels_to_keep = h.gating_module.active_channels_mask
        for m, s in h.modules_and_sides:
            conv = mapper.reverse_alias_map.get((m, s), m)
            if print_replacements:
                print('Looking at conv {} side {}'.format(id(conv), 'in' if s else 'out'))
            if conv not in prune_map:
                if print_replacements:
                    print('conv not in map, adding')
                bn = mapper.alias_map.get((conv, False), None)
                conv_to_prune = ConvToPrune(conv, bn)
                prune_map[conv] = conv_to_prune
            else:
                conv_to_prune = prune_map[conv]
            # setting in/out twice will trigger and error
            if s:
                if print_replacements:
                    print('Adding in channels')
                conv_to_prune.add_in_prune(channels_to_keep)
            else:
                if print_replacements:
                    print('Adding out channels')
                conv_to_prune.add_out_prune(channels_to_keep)
    conv_to_new_modules = {}
    for conv, conv_to_prune in prune_map.items():
        new_conv, new_bn = conv_to_prune.get_pruned_conv()
        conv_to_new_modules[conv] = (new_conv, new_bn)
        if apply_replacements:
            mapper.apply_replacement(conv_to_prune.conv, new_conv)
        if new_bn is not None and apply_replacements:
            mapper.apply_replacement(conv_to_prune.bn, new_bn)
    new_fc = prune_fc(net_with_hooks.net.fc, channels_to_keep, True)
    conv_to_new_modules[net_with_hooks.net.fc] = (new_fc, None)
    if apply_replacements:
        mapper.apply_replacement(net_with_hooks.net.fc, new_fc)
    return conv_to_new_modules
