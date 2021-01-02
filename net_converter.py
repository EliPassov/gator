import argparse
import json

import torch
from torchvision.models.resnet import resnet50
from models.cifar_resnet import resnet56
from models.custom_resnet import custom_resnet_50, custom_resnet_56

from models.wrapped_gated_models import custom_resnet_from_gated_net, pruned_custom_net_from_gated_net


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--net_with_criterion', type=str,
                    help='criterion_name')
parser.add_argument('--net_name', type=str,
                    help='resnet name')
parser.add_argument('--gated_weights_path', type=str,
                    help='path to gated net weights')
parser.add_argument('--new_weights_path', type=str,
                    help='path to new file which will be created')
parser.add_argument('--include_gates', action='store_true', default=False,
                    help='prune with gates and store their weights')
parser.add_argument('--gating_config_path_for_gate_max_probs', type=str, default=None,
                    help = 'config to pick gate_init_prob from for max probability for gate to clamp the weight in case it is too high')
parser.add_argument('--new_format', action='store_true', default=False,
                    help='Use new pytorch save format if version is relevant')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.net_name == 'CustomResNet50':
        assert args.resume is not None
        channels_config = torch.load(args.gated_weights_path)['channels_config']
        net = custom_resnet_50(channels_config, 1000)
        custom_net_func = custom_resnet_50
    elif args.net_name == 'resnet56':
        net = resnet56(10)
        custom_net_func = custom_resnet_56
    elif args.net_name == 'resnet50':
        net = resnet50(False, num_classes=1000)
        custom_net_func = custom_resnet_50
    else:
        raise ValueError('Unsupported net type ' + args.net_name)

    if args.include_gates:
        gate_max_probs = None
        if args.gating_config_path_for_gate_max_probs is not None:
            with open(args.gating_config_path_for_gate_max_probs) as f:
                gate_max_probs = json.load(f)['gate_init_prob']
        pruned_custom_net_from_gated_net(net, args.net_with_criterion, args.gated_weights_path, args.new_weights_path,
                                         custom_net_func, gate_max_probs, old_format=not args.new_format)
    else:
        custom_resnet_from_gated_net(net, args.net_with_criterion, args.gated_weights_path, args.new_weights_path,
                                     custom_net_func, old_format=not args.new_format)
