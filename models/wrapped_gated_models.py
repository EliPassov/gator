import pandas as pd

import torch
from torch import nn
from torchvision.models.resnet import ResNet, resnet18, resnet34, resnet50

from models.channel_gates import ModuleChannelsLogisticGating, ModuleChannelsLogisticGatingMasked
from models.gate_wrapped_module import create_wrapped_net
from models.gates_mapper import NaiveSequentialGatesModulesMapper, ResNetGatesModulesMapper
from models.net_auxiliary_extension import NetWithAuxiliaryOutputs, CriterionWithAuxiliaryLosses


def channel_gating_reporting(layer, input):
    res = {'layer_{}_ch_use_mean'.format(layer): input.mean().item(),
           'layer_{}_ch_use_count'.format(layer): (input.mean(0) > 0.5).sum().item()/input.size(1)}
    return res


def get_wrapped_gating_net_and_criteria(net, main_criterion, criteria_weight, gradient_multiplier=1.0, adaptive=True,
                                        gating_class=ModuleChannelsLogisticGatingMasked, gate_init_prob=0.99,
                                        random_init=False, factor_type='flop_factor', no_last_conv=False,
                                        edge_multipliers_csv_path=None, gradient_multipliers_csv_path=None):
    edge_multipliers = None
    if edge_multipliers_csv_path is not None:
        edge_multipliers = pd.read_csv(edge_multipliers_csv_path, index_col=None, header=None).values
        # normalize such that the sum is the number of them (i.e. neutral weight is 1)
        edge_multipliers = edge_multipliers * len(edge_multipliers)/ edge_multipliers.sum(0)
        edge_multipliers = edge_multipliers.reshape(-1).tolist()

    gradient_secondary_multipliers = None
    if gradient_multipliers_csv_path is not None:
        gradient_secondary_multipliers = pd.read_csv(gradient_multipliers_csv_path, index_col=None, header=None)\
            .values.reshape(-1).tolist()


    mapper_class = ResNetGatesModulesMapper if isinstance(net, ResNet) else NaiveSequentialGatesModulesMapper

    hooks, auxiliary_criteria = create_wrapped_net(mapper_class(net, no_last_conv), gradient_multiplier, adaptive,
                                                   gating_class, gate_init_prob, random_init, factor_type,
                                                   edge_multipliers, gradient_secondary_multipliers)

    report_func = channel_gating_reporting

    criterion = CriterionWithAuxiliaryLosses(main_criterion, auxiliary_criteria, criteria_weight, False, report_func)
    net_with_aux = NetWithAuxiliaryOutputs(net, hooks)
    return net_with_aux, criterion


ResNet18_gating = lambda classes:get_wrapped_gating_net_and_criteria(
    resnet18(True), nn.CrossEntropyLoss(), 0.2)

ResNet34_gating = lambda classes:get_wrapped_gating_net_and_criteria(
    resnet34(True), nn.CrossEntropyLoss(), 0.2)

ResNet50_gating = lambda classes:get_wrapped_gating_net_and_criteria(
    resnet50(True), nn.CrossEntropyLoss(), 0.2, gradient_multiplier=0.2, gate_init_prob=0.995)

ResNet50_gating_custom = lambda classes:get_wrapped_gating_net_and_criteria(
    resnet50(True), nn.CrossEntropyLoss(), 0.5, gradient_multiplier=0.2, gate_init_prob=0.995
    ,edge_multipliers_csv_path='./models/data/resnet_50_hyper_edge_rel_factor.csv')

ResNet50_gating_memory = lambda classes:get_wrapped_gating_net_and_criteria(
    resnet50(True), nn.CrossEntropyLoss(), 0.5, gradient_multiplier=0.2, gate_init_prob=0.995,
    factor_type='memory_factor', gradient_multipliers_csv_path='./models/data/resnet_50_hyper_edge_memory_factor.csv')

ResNet50_gating_memory_test = lambda classes:get_wrapped_gating_net_and_criteria(
    resnet50(True), None, 0.5, gradient_multiplier=0.2, gate_init_prob=0.995,
    factor_type='flop_factor')#, gradient_multipliers_csv_path='./models/data/resnet_50_hyper_edge_memory_factor.csv')


if __name__ == '__main__':
    net, criterion = ResNet50_gating_memory_test(1000)
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