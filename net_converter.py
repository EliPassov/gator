import argparse

from models.wrapped_gated_models import custom_resnet_from_gated_net


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--net_name', type=str,
                    help='resnet name')
parser.add_argument('--gated_weights_path', type=str,
                    help='path to gated net weights')
parser.add_argument('--new_weights_path', type=str,
                    help='path to new file which will be created')


if __name__ == '__main__':
    args = parser.parse_args()
    custom_resnet_from_gated_net(args.net_name, args.gated_weights_path, args.new_weights_path)
