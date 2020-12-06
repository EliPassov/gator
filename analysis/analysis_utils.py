import numpy as np
import torch

from models.wrapped_models import NetWithAuxiliaryOutputs
from utils.correlation import pairwise_distances
from models.gated_grouped_conv import create_target_mask

from external_models.cifar10_models.resnet import BasicBlock as EBB


def activations_distnaces(net, data_loader, num_distances, greater_then=0.1):
    #     fig, sblts = plt.subplots(round_num , fig_in_row, figsize=(fig_width, fig_height))
    assert isinstance(net, NetWithAuxiliaryOutputs)

    distance_averages = None
    distance_greater = None
    similar_averages = None
    layer_sizes = []
    different_distances_count = 0
    similar_distances_count = 0

    for b_id, (_, data, label) in enumerate(data_loader):
        label = label.numpy().tolist()
        _, hooks_out = net(data)
        if distance_averages is None:
            print('batch size:', len(label), ',number of layers:', len(hooks_out), ',first batch labels:', label)
            distance_averages = np.zeros(len(hooks_out))
            similar_averages = np.zeros(len(hooks_out))
            distance_greater = np.zeros(len(hooks_out))
            similar_greater = np.zeros(len(hooks_out))

        target_mask = create_target_mask(torch.LongTensor(label))
        n_d = int(target_mask.sum().item())
        different_distances_count += n_d // 2
        n_d_s = (len(target_mask) * (len(target_mask) - 1) - n_d)
        similar_distances_count += n_d_s // 2

        #         n_d = max(n_d,1)

        for i in range(len(hooks_out)):
            d = hooks_out[i].shape
            distances = pairwise_distances(hooks_out[i].view(d[0], d[1]))
            s = hooks_out[i].size(1)
            if len(layer_sizes) < len(hooks_out):
                layer_sizes.append(s)
            b = distances.size(0)
            distances = distances / s
            distances = torch.pow(distances + 1e-10, 0.5)
            different_distances = distances * target_mask
            #             sns.heatmap(distances.detach(), ax=sblts[i // fig_in_row, i % fig_in_row], xticklabels=label, yticklabels=label)
            distance_averages[i] += different_distances.sum().item() / n_d
            distance_greater[i] += ((different_distances > greater_then).sum().item()) / n_d

            similar_mask = (1 - (target_mask + torch.eye(distances.shape[0])))
            similar_distances = distances * similar_mask
            similar_averages[i] += similar_distances.sum().item() / similar_mask.sum().item()
            similar_greater[i] += ((similar_distances > greater_then).sum().item()) / n_d_s

        if similar_distances_count + different_distances_count > num_distances:
            print('ran for {} batches measuring {} different distances and {} similar distances'.format(b_id + 1, int(
                different_distances_count), int(similar_distances_count)))
            break
    distance_averages = distance_averages / (b_id + 1)
    similar_averages = similar_averages / (b_id + 1)
    distance_greater = distance_greater / (b_id + 1)
    similar_greater = similar_greater / (b_id + 1)
    for i in range(distance_averages.shape[0]):
        print(
            'layer {} ({})\t diff,simlar average {:.5f}, {:.5f} \t %(d > {:.2f}) diff, similar : {:6.2f}%, {:6.2f}%'.
                format(i, layer_sizes[i], distance_averages[i], similar_averages[i], greater_then,
                       100 * distance_greater[i], 100 * similar_greater[i]))
