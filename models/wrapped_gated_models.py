from copy import deepcopy
import warnings

import pandas as pd
import torch
from torch import nn
from torchvision.models.resnet import ResNet, resnet18, resnet34, resnet50

from models.channel_gates import ModuleChannelsLogisticGating, ModuleChannelsLogisticGatingMasked, \
    logistic_quantile_funciton
from models.gate_wrapped_module import create_wrapped_net, create_conv_channels_dict
from models.gates_mapper import NaiveSequentialGatesModulesMapper, ResNetGatesModulesMapper
from models.net_auxiliary_extension import NetWithAuxiliaryOutputs, CriterionWithAuxiliaryLosses, ClassificationLayerHook
from models.custom_resnet import custom_resnet_18, custom_resnet_34, custom_resnet_50
from models.gated_prunning import get_pruned_hooks_weights
from utils.save_warpper import save_version_aware


def channel_gating_reporting(layer, input, target):
    res = {'layer_{}_ch_use_mean'.format(layer): input.mean().item(),
           'layer_{}_ch_use_count'.format(layer): (input.mean(0) > 0.5).sum().item()/input.size(1)}
    return res


def aux_classifier_report_function(layer, input, target):
    res = {'layer_{}_top1 accuracy'.format(layer): (input.argmax(1)==target).sum().item() / len(target),
           'layer_{}_top5 accuracy'.format(layer):
               (input.argsort(1)[:,:5] == target.view(-1,1).repeat(1,5)).sum().item() / len(target)}
    return res


def get_wrapped_gating_net_and_criteria(net, main_criterion, criteria_weight=0.0, gradient_multiplier=1.0,
                                        adaptive=True, gating_class=ModuleChannelsLogisticGatingMasked,
                                        gate_init_prob=0.99, random_init=False, factor_type='flop_factor',
                                        no_last_conv=False, edge_multipliers=None, gradient_secondary_multipliers=None,
                                        aux_classification_losses_modules=None, aux_classification_losses_weights=None,
                                        aux_classification_lr_multiplier=None, gate_weights=None):
    mapper_class = ResNetGatesModulesMapper if isinstance(net, ResNet) else NaiveSequentialGatesModulesMapper

    if criteria_weight == 0.0:
        warnings.warn("created gating module with 0 weight, make sure you are not training it")

    hooks, auxiliary_criteria, param_groups_lr_adjustment_map = create_wrapped_net(net, mapper_class(net, no_last_conv),
        gradient_multiplier, adaptive, gating_class, gate_init_prob, random_init, factor_type, edge_multipliers,
        gradient_secondary_multipliers, create_multiple_optimizers=True, gate_weights=gate_weights)

    report_func = channel_gating_reporting

    if aux_classification_losses_modules is not None:
        if isinstance(criteria_weight, float):
            criteria_weight = [criteria_weight for _ in range(len(hooks))]
        report_func = [report_func for _ in range(len(hooks))]

        aux_hooks, aux_losses, aux_classification_losses_weights, aux_param_groups_lr_adjustment_map= add_aux_hooks(
            net, 1000, aux_classification_losses_modules, aux_classification_losses_weights,
            aux_classification_lr_multiplier)

        criteria_weight = criteria_weight + aux_classification_losses_weights

        hooks.extend(aux_hooks)
        auxiliary_criteria.extend(aux_losses)
        report_func = report_func + [aux_classifier_report_function for _ in range(len(aux_hooks))]
        if aux_param_groups_lr_adjustment_map is not None:
            if param_groups_lr_adjustment_map is None:
                raise ValueError()
            (param_groups, lr_adjustment_map) = param_groups_lr_adjustment_map
            param_groups.extend(aux_param_groups_lr_adjustment_map[0])
            lr_adjustment_map.update(aux_param_groups_lr_adjustment_map[1])
            param_groups_lr_adjustment_map= (param_groups, lr_adjustment_map)

    criterion = CriterionWithAuxiliaryLosses(main_criterion, auxiliary_criteria, criteria_weight, False, report_func)
    net_with_aux = NetWithAuxiliaryOutputs(net, hooks)
    return net_with_aux, criterion, param_groups_lr_adjustment_map


def get_wrapped_aux_net(net, main_criterion, aux_classification_losses_modules, aux_classification_losses_weights,
                        aux_classification_lr_multiplier=None):
    aux_hooks, aux_losses, aux_classification_losses_weights, aux_param_groups_lr_adjustment_map = add_aux_hooks(
        net, 1000, aux_classification_losses_modules, aux_classification_losses_weights,
        aux_classification_lr_multiplier)
    report_func = aux_classifier_report_function

    criterion = CriterionWithAuxiliaryLosses(main_criterion, aux_losses, aux_classification_losses_weights, False,
                                             report_func)
    net_with_aux = NetWithAuxiliaryOutputs(net, aux_hooks)
    return net_with_aux, criterion, aux_param_groups_lr_adjustment_map


def add_aux_hooks(net, num_classes, aux_classification_losses_modules, aux_classification_losses_weights,
                  aux_classification_lr_multiplier=None):
    # combine weights to one list
    if isinstance(aux_classification_losses_weights, float):
        aux_classification_losses_weights = \
            [aux_classification_losses_weights for _ in range(len(aux_classification_losses_modules))]
    else:
        assert isinstance(aux_classification_losses_weights, list) and \
               len(aux_classification_losses_weights) == len(aux_classification_losses_modules)

    if aux_classification_lr_multiplier is not None:
        if isinstance(aux_classification_lr_multiplier, float):
            aux_classification_lr_multiplier = \
                [aux_classification_lr_multiplier for _ in range(len(aux_classification_losses_modules))]
        assert len(aux_classification_lr_multiplier) == len(aux_classification_losses_weights)

        param_groups, lr_adjustment_map = [ {'params': net.parameters()}], {}

    # create additional hooks
    net_children = {k: v for k, v in net.named_modules()}
    aux_hooks = []
    for i, d in enumerate(aux_classification_losses_modules):
        module_name, c = d["module"], d["channels"]
        module = net_children[module_name]
        aux_hook = ClassificationLayerHook(module, c, num_classes)
        aux_hooks.append(aux_hook)
        lr_adjustment_map[len(param_groups)] = lambda : aux_classification_lr_multiplier[i]
        param_groups.append({'params': aux_hook.parameters()})

    aux_losses = [nn.CrossEntropyLoss() for _ in range(len(aux_hooks))]

    return aux_hooks, aux_losses, aux_classification_losses_weights, \
           (param_groups, lr_adjustment_map) if aux_classification_lr_multiplier is not None  else None


def read_net(net, criterion_name, net_weight_path):
    wrapped_net, _, _ = globals()[criterion_name](net)
    full_state_dict = torch.load(net_weight_path)
    weights_state_dict = {k[7:]: v for k, v in full_state_dict['state_dict'].items()}
    wrapped_net.load_state_dict(weights_state_dict)
    return wrapped_net, full_state_dict


def custom_resnet_from_gated_net(net, criterion_name, net_weight_path, new_file_path=None, no_last_conv=False,
                                 old_format=True):
    wrapped_net, full_state_dict = read_net(net, criterion_name, net_weight_path)
    mapper = ResNetGatesModulesMapper(net, no_last_conv)
    channels_config, new_weights_state_dict = create_conv_channels_dict(wrapped_net, mapper)

    new_state_dict = deepcopy(full_state_dict)
    del new_state_dict['optimizer']
    new_state_dict['channels_config'] = channels_config
    new_state_dict['state_dict'] = {'module.' + k:v for k,v in new_weights_state_dict.items()}
    if new_file_path is not None:
        save_version_aware(new_state_dict, new_file_path, old_format)
    else:
        custom_net_func = None
        for ind in ['18', '34', '50']:
            if ind in net_name:
                custom_net_func = globals()['custom_resnet_' + ind]
        return custom_net_func(channels_config)


def clamp_gate_weights(gate_weights, gate_max_probs):
    if isinstance(gate_max_probs, float):
        gate_max_probs = [gate_max_probs for _ in range(len(gate_weights))]

    for i in range(len(gate_weights)):
        gate_max_weight = logistic_quantile_funciton(gate_max_probs[i])
        gate_weights[i] = gate_weights[i].clamp(max=gate_max_weight)


def prune_custom_resnet(net, no_last_conv=False, clamp_init_prob=False, net_config_kwargs={}):
    """
    Apply pruning on standard or custom resnet with gates
    :param net: Original net
    :param no_last_conv:
    :param clamp_init_prob: limit the max gating value by the initialized value. This option should be used when
    initializing new training for higher pruning to reduce the gating weights which were previously pushed towards
    high value and now might be eligible for pruning
    :param net_config_kwargs: gated network configuration
    :return:
    """
    mapper = ResNetGatesModulesMapper(net.net, no_last_conv)
    channels_config, new_weights_state_dict = create_conv_channels_dict(net, mapper)

    custom_net_func = custom_resnet_50
    for ind in ['18', '34', '50']:
        if ind in type(net).__name__:
            custom_net_func = globals()['custom_resnet_' + ind]
    new_sub_net = custom_net_func(channels_config)
    gate_weights = get_pruned_hooks_weights(net)

    if clamp_init_prob:
        init_probs = net_config_kwargs['gate_init_prob']
        clamp_gate_weights(gate_weights, init_probs)

    return get_wrapped_gating_net_and_criteria(new_sub_net, nn.CrossEntropyLoss(), gate_weights=gate_weights,
                                               **net_config_kwargs)


def pruned_custom_net_from_gated_net(net, criterion_name, net_weight_path, new_file_path, gate_max_probs,
                                     no_last_conv=False, old_format=True):
    wrapped_net, full_state_dict = read_net(net, criterion_name, net_weight_path)

    mapper = ResNetGatesModulesMapper(net, no_last_conv)
    channels_config, new_weights_state_dict = create_conv_channels_dict(wrapped_net, mapper)

    new_state_dict = deepcopy(full_state_dict)
    new_state_dict['state_dict'] = wrapped_net.state_dict()
    del new_state_dict['optimizer']
    new_state_dict['channels_config'] = channels_config

    custom_net_func = custom_resnet_50
    for ind in ['18', '34', '50']:
        if ind in type(net).__name__:
            custom_net_func = globals()['custom_resnet_' + ind]
    new_sub_net = custom_net_func(channels_config)
    gate_weights = get_pruned_hooks_weights(wrapped_net)

    clamp_gate_weights(gate_weights, gate_max_probs)

    new_net, _, _ = get_wrapped_gating_net_and_criteria(new_sub_net, nn.CrossEntropyLoss(), gate_weights=gate_weights)

    new_net.net.load_state_dict(new_weights_state_dict)
    new_state_dict['state_dict'] = {'module.' + k:v for k,v in new_net.state_dict().items()}

    save_version_aware(new_state_dict, new_file_path, old_format)


# ResNet18_gating = lambda num_classes=1000, kwargs={}: get_wrapped_gating_net_and_criteria(
#     resnet18(True, num_classes=num_classes), nn.CrossEntropyLoss(), **kwargs)
#
# ResNet34_gating = lambda num_classes=1000, kwargs={}: get_wrapped_gating_net_and_criteria(
#     resnet34(True, num_classes=num_classes), nn.CrossEntropyLoss(), **kwargs)

ResNet50_gating = lambda net, kwargs={}: get_wrapped_gating_net_and_criteria(net, nn.CrossEntropyLoss(), **kwargs)

Resnet50_aux_losses = lambda num_classes=1000, kwargs={}: get_wrapped_aux_net(net, nn.CrossEntropyLoss(), **kwargs)


if __name__ == '__main__':
    net_name = 'ResNet50_gating'
    net_weight_path='/home/eli/Eli/Training/Imagenet/resnet50/resnet50_pre_0_995_w_0_25_gm_0_2_w_0_5/net_e_80'
    net, _, _ = globals()[net_name](1000)
    full_state_dict = torch.load(net_weight_path)
    weights_state_dict = {k[7:]: v for k, v in full_state_dict['state_dict'].items()}
    net.load_state_dict(weights_state_dict)
    custom_net, _, _ = prune_custom_resnet(net)

    custom_net = custom_net.cuda()

    custom_net.eval()
    sample = torch.rand((1,3,224,224)).cuda()
    res = custom_net(sample)

    print(res)

