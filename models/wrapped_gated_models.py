from copy import deepcopy

import pandas as pd
import torch
from torch import nn
from torchvision.models.resnet import ResNet, resnet18, resnet34, resnet50

from models.channel_gates import ModuleChannelsLogisticGating, ModuleChannelsLogisticGatingMasked
from models.gate_wrapped_module import create_wrapped_net, create_conv_channels_dict
from models.gates_mapper import NaiveSequentialGatesModulesMapper, ResNetGatesModulesMapper
from models.net_auxiliary_extension import NetWithAuxiliaryOutputs, CriterionWithAuxiliaryLosses, ClassificationLayerHook
from models.custom_resnet import custom_resnet_18, custom_resnet_34, custom_resnet_50

def channel_gating_reporting(layer, input, target):
    res = {'layer_{}_ch_use_mean'.format(layer): input.mean().item(),
           'layer_{}_ch_use_count'.format(layer): (input.mean(0) > 0.5).sum().item()/input.size(1)}
    return res


def aux_classifier_report_function(layer, input, target):
    res = {'layer_{}_top1 accuracy'.format(layer): (input.argmax(1)==target).sum().item() / len(target),
           'layer_{}_top5 accuracy'.format(layer):
               (input.argsort(1)[:,:5] == target.view(-1,1).repeat(1,5)).sum().item() / len(target)}
    return res


def get_wrapped_gating_net_and_criteria(net, main_criterion, criteria_weight, gradient_multiplier=1.0, adaptive=True,
                                        gating_class=ModuleChannelsLogisticGatingMasked, gate_init_prob=0.99,
                                        random_init=False, factor_type='flop_factor', no_last_conv=False,
                                        edge_multipliers=None, gradient_secondary_multipliers=None,
                                        aux_classification_losses_modules=None, aux_classification_losses_weights=None):
    mapper_class = ResNetGatesModulesMapper if isinstance(net, ResNet) else NaiveSequentialGatesModulesMapper

    hooks, auxiliary_criteria, param_groups_lr_adjustment_map = create_wrapped_net(net, mapper_class(net, no_last_conv),
        gradient_multiplier, adaptive, gating_class, gate_init_prob, random_init, factor_type, edge_multipliers,
        gradient_secondary_multipliers, create_multiple_optimizers=True)

    report_func = channel_gating_reporting

    if aux_classification_losses_modules is not None:
        if isinstance(criteria_weight, float):
            criteria_weight = [criteria_weight for _ in range(len(hooks))]
        report_func = [report_func for _ in range(len(hooks))]

        aux_hooks, aux_losses, aux_classification_losses_weights = add_aux_hooks(
            net, 1000, aux_classification_losses_modules, aux_classification_losses_weights)

        criteria_weight = criteria_weight + aux_classification_losses_weights

        hooks.extend(aux_hooks)
        auxiliary_criteria.extend(aux_losses)
        report_func = report_func + [aux_classifier_report_function for _ in range(len(aux_hooks))]

    criterion = CriterionWithAuxiliaryLosses(main_criterion, auxiliary_criteria, criteria_weight, False, report_func)
    net_with_aux = NetWithAuxiliaryOutputs(net, hooks)
    return net_with_aux, criterion, param_groups_lr_adjustment_map


def get_wrapped_aux_net(net, main_criterion, aux_classification_losses_modules, aux_classification_losses_weights):
    aux_hooks, aux_losses, aux_classification_losses_weights = add_aux_hooks(
        net, 1000, aux_classification_losses_modules, aux_classification_losses_weights)
    report_func = aux_classifier_report_function

    criterion = CriterionWithAuxiliaryLosses(main_criterion, aux_losses, aux_classification_losses_weights, False,
                                             report_func)
    net_with_aux = NetWithAuxiliaryOutputs(net, aux_hooks)
    return net_with_aux, criterion, None


def add_aux_hooks(net, num_classes, aux_classification_losses_modules, aux_classification_losses_weights):
    # combine weights to one list
    if isinstance(aux_classification_losses_weights, float):
        aux_classification_losses_weights = \
            [aux_classification_losses_weights for _ in range(len(aux_classification_losses_modules))]
    else:
        assert isinstance(aux_classification_losses_weights, list) and \
               len(aux_classification_losses_weights) == len(aux_classification_losses_modules)

    # create additional hooks
    net_children = {k: v for k, v in net.named_modules()}
    aux_hooks = []
    for i, d in enumerate(aux_classification_losses_modules):
        module_name, c = d["module"], d["channels"]
        module = net_children[module_name]
        aux_hooks.append(ClassificationLayerHook(module, c, num_classes))

    aux_losses = [nn.CrossEntropyLoss() for _ in range(len(aux_hooks))]

    return aux_hooks, aux_losses, aux_classification_losses_weights


def custom_resnet_from_gated_net(net_name, net_weight_path, new_file_path=None, no_last_conv=False):
    net, _, _ = globals()[net_name](1000)
    full_state_dict = torch.load(net_weight_path)
    weights_state_dict = {k[7:]: v for k, v in full_state_dict['state_dict'].items()}
    net.load_state_dict(weights_state_dict)
    mapper = ResNetGatesModulesMapper(net.net, no_last_conv, map_for_replacement=True)
    channels_config, new_weights_state_dict = create_conv_channels_dict(net, mapper)

    new_state_dict = deepcopy(full_state_dict)
    del new_state_dict['optimizer']
    new_state_dict['channels_config'] = channels_config
    new_state_dict['state_dict'] = {'module.' + k:v for k,v in new_weights_state_dict.items()}
    if new_file_path is not None:
        torch.save(new_state_dict, new_file_path)
    else:
        custom_net_func = None
        for ind in ['18', '34', '50']:
            if ind in net_name:
                custom_net_func = globals()['custom_resnet_' + ind]
        return custom_net_func(channels_config)


ResNet18_gating = lambda num_classes, kwargs :get_wrapped_gating_net_and_criteria(
    resnet18(True, num_classes=num_classes), nn.CrossEntropyLoss(), **kwargs)

ResNet34_gating = lambda num_classes, kwargs:get_wrapped_gating_net_and_criteria(
    resnet34(True, num_classes=num_classes), nn.CrossEntropyLoss(), **kwargs)

ResNet50_gating = lambda num_classes, kwargs:get_wrapped_gating_net_and_criteria(
    resnet50(True, num_classes=num_classes), nn.CrossEntropyLoss(), **kwargs)

Resnet50_aux_losses = lambda num_classes, kwargs:get_wrapped_aux_net(
    resnet50(True, num_classes=num_classes), nn.CrossEntropyLoss(), **kwargs)


if __name__ == '__main__':
    net, criterion, _ = ResNet50_gating(1000, {})
    net = net.cuda()
    sample = torch.randn(1,3,224,224).cuda()
    output = net(sample)
    target = torch.ones(1).long().cuda()
    loss = criterion(output, target)
    loss['loss'].backward()
    for h in net.forward_hooks:
        grads = h.gating_module.gating_weights.grad
        print(grads.mean().item(), grads.std().item())
    aaa = 3