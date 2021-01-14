from functools import partial
import numpy as np
import torch
from torch import nn

from models.channel_gates import ModuleChannelsLogisticGating, ModuleChannelsLogisticGatingMasked, WeightedL1Criterion
from utils.funtional import AdjustedMultiplier, AdjustedDivisor, GetterArgsFunctional, SumMany
from models.gated_prunning import prune_net_with_hooks


def create_gating_modules(mapper, gating_class, gate_init_prob, random_init, gate_weights=None):
    # create all gate modules
    if isinstance(gate_init_prob, float):
        gate_init_prob = [gate_init_prob for _ in range(len(mapper.hyper_edges))]
    hyper_edges_to_hooks = {}
    for i, h in enumerate(mapper.hyper_edges):
        convs = []
        sides = []
        for j in range(len(h.convs_and_sides)):
            conv, side = h.convs_and_sides[j]
            sides.append(side)
            # get alias module if exists
            module = mapper.alias_map.get((conv, side), conv)
            convs.append(module)
        # inject gate_weights[i] in call
        gating_call = partial(gating_class, gate_weights=gate_weights[i]) if gate_weights is not None else gating_class

        hyper_edges_to_hooks[h] = gating_call(convs, h.channels, sides, None, gate_init_prob[i], random_init)
    return hyper_edges_to_hooks


def create_wrapped_net(net, mapper, gradient_multiplier=1.0, adaptive=True,
                       gating_class=ModuleChannelsLogisticGatingMasked, gate_init_prob=0.99, random_init=False,
                       factor_type="flop_factor", edge_multipliers=None, gradient_secondary_multipliers=None,
                       create_multiple_optimizers=False, gate_weights=None, static_total_cost=None):
    if edge_multipliers is not None:
        assert isinstance(edge_multipliers, list)
        assert len(edge_multipliers) == len(mapper.hyper_edges)
    if gradient_secondary_multipliers is not None:
        assert isinstance(gradient_secondary_multipliers, list)
        assert len(gradient_secondary_multipliers) == len(mapper.hyper_edges)
    hooks = []
    auxiliary_criteria = []

    # get static total cost if it wasn't predefined (to be used by custom resnet which was pruned
    if static_total_cost is None:
        static_total_cost = getattr(mapper, factor_type.replace('factor', 'cost'))
    # create all gate modules
    hyper_edges_to_hooks = create_gating_modules(mapper, gating_class, gate_init_prob, random_init, gate_weights)

    if create_multiple_optimizers:
        param_groups, lr_adjustment_map = [{'params': net.parameters()}], {}

    # link costs and create losses
    for i, h in enumerate(mapper.hyper_edges):
        hook = hyper_edges_to_hooks[h]
        costs_funcs = []
        constant_costs = []
        for conv, is_in in h.convs_and_sides:
            factor = getattr(mapper.conv_info[conv], factor_type)
            if edge_multipliers is not None:
                factor = factor * edge_multipliers[i]
            # for input the cost depends on output channels and vice versa
            dependency_key = (conv, not is_in)
            # create a dependency function if the channel number can change as a result of another gate
            if adaptive and dependency_key in mapper.hyper_edges_map:
                dependency_hook = hyper_edges_to_hooks[mapper.hyper_edges_map[dependency_key]]
                channels_getter = dependency_hook.gating_module.active_channels
                costs_funcs.append(AdjustedMultiplier(channels_getter, factor))
            # if not, create a constant function
            else:
                const_channels = conv.out_channels if is_in else conv.in_channels
                constant_costs.append(const_channels * factor)
        sum_cost = SumMany(costs_funcs, *constant_costs)
        weight_func = AdjustedMultiplier(sum_cost, 1 / (static_total_cost))
        auxiliary_criteria.append(WeightedL1Criterion(weight=weight_func))
        gradient_total_multiplier = gradient_multiplier
        if gradient_secondary_multipliers is not None:
            gradient_total_multiplier *= gradient_secondary_multipliers[i]
        gradient_adjustment = AdjustedDivisor(weight_func, gradient_total_multiplier)
        # print(len(h.convs_and_sides), gradient_adjustment(), weight_func())
        if create_multiple_optimizers:
            # param groups has already main module params when inserting the first adjustment which gets the index 1
            lr_adjustment_map[len(param_groups)] = gradient_adjustment
            param_groups.append({'params': hook.parameters()})
        else:
            hook.set_gradient_adjustment(gradient_adjustment)
        hooks.append(hook)

    return hooks, auxiliary_criteria, (param_groups, lr_adjustment_map) if create_multiple_optimizers else None


def compute_flop_cost_change(net_with_hooks, mapper, factor_type="flop_factor"):
    hooks = net_with_hooks.forward_hooks

    conv_to_filters = {}

    for h in hooks:
        for m, s in h.modules_and_sides:
            # map back any non convolutional module
            conv = mapper.reverse_alias_map.get((m, s), m)
            if conv not in conv_to_filters:
                conv_to_filters[conv] = {}
            if s in conv_to_filters[conv]:
                raise ValueError('more then one hook referring to the same module and side')
            conv_to_filters[conv][s] = h.gating_module.active_channels()
    # add missing convolutions
    for m in net_with_hooks.modules():
        if isinstance(m, nn.Conv2d) and m not in conv_to_filters:
            conv_to_filters[m] = {}
    # add missing (static) sides
    for m, d in conv_to_filters.items():
        for b in [True, False]:
            if b not in d:
                conv_to_filters[m][b] = m.in_channels if b else m.out_channels

    original_cost = 0
    new_cost = 0

    for m in net_with_hooks.modules():
        if isinstance(m, nn.Conv2d):
            factor = getattr(mapper.conv_info[m], factor_type)
            original_cost += factor * m.in_channels * m.out_channels
            new_cost += factor * conv_to_filters[m][True] * conv_to_filters[m][False]

    return original_cost, new_cost


def compute_flop_cost(net, mapper, factor_type="flop_factor"):
    original_cost = 0

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            factor = getattr(mapper.conv_info[m], factor_type)
            original_cost += factor * m.in_channels * m.out_channels

    return original_cost


def dot_string_to_tree_dict(d, name, val):
    dot_pos = name.find('.')
    if dot_pos > -1:
        key = name[: dot_pos]
        sub_key = name[dot_pos + 1: ]
        if key not in d:
            d[key] = {}
        dot_string_to_tree_dict(d[key], sub_key, val)
    else:
        assert name not in d
        d[name] = val


def create_edge_to_channels_map(mapper, hooks):
    hyper_edge_to_active_channels = {}
    for h in hooks:
        m, s = h.modules_and_sides[0]
        conv = mapper.reverse_alias_map.get((m, s), m)
        hyper_edge = mapper.hyper_edges_map[(conv, s)]
        # hyper_edge_to_active_channels[hyper_edge] = int((h.gating_module.active_channels_mask).sum().item())
        hyper_edge_to_active_channels[hyper_edge] = h.gating_module.active_channels_mask
    return hyper_edge_to_active_channels


def create_conv_channels_dict(net_with_hooks, mapper, new_modules_mapping=False):
    # Map each hook association to hyper edge and assigns hook's active channels to that edge
    hyper_edge_to_active_channels = create_edge_to_channels_map(mapper, net_with_hooks.forward_hooks)
    conv_to_new_modules = prune_net_with_hooks(net_with_hooks, mapper)
    # create a dictionary representing the tree structure of the network with convolution output filter values
    conv_channels_dict = {}
    state_dict = {}
    for hyper_edge in mapper.hyper_edges:
        assert hyper_edge.conv_names is not None
        assert len(hyper_edge.conv_names) == len(hyper_edge.convs_and_sides)
        active_channels_mask = hyper_edge_to_active_channels[hyper_edge]
        num_channels = int(active_channels_mask.sum().item())
        # take only out channels of convolutions
        for i, (c, s) in enumerate(hyper_edge.convs_and_sides):
            new_modules = conv_to_new_modules[mapper.reverse_alias_map.get((c, s), c)]
            # skip edge if entirely pruned
            if new_modules[0] is None:
                continue
            if not s:
                dot_string_to_tree_dict(conv_channels_dict, hyper_edge.conv_names[i], num_channels)

            # resnet specific code, generalize later

            conv_name = hyper_edge.conv_names[i]
            for k, v in new_modules[0].state_dict().items():
                if 'downsample' in conv_name:
                    state_dict[conv_name + '.0.' + k ] = v
                else:
                    state_dict[conv_name + '.' + k] = v
            for k, v in new_modules[1].state_dict().items():
                if 'downsample' in conv_name:
                    state_dict[conv_name + '.1.' + k ] = v
                else:
                    state_dict[conv_name.replace('conv', 'bn') + '.' + k] = v
        for k, v in conv_to_new_modules[net_with_hooks.net.fc][0].state_dict().items():
            state_dict['fc.' + k] = v

    return conv_channels_dict, state_dict
