import math
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.distributions import Gumbel, Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform

"""
For this code to run it requires at least pytorch 1.2.0 for mutation of values in forward hooks
Earlier version can be hacked by applying the changes from:
https://github.com/pytorch/pytorch/commit/577c04c490761279e34922da7d4553be51874e06
"""


# All distributions (Gumbel, Logistic) assume location of 0 and scale of 1 (mu = 0, betta = 1)


LogisticDistribution = lambda size: TransformedDistribution(Uniform(torch.zeros(size), torch.ones(size)), [SigmoidTransform().inv])


def gumbel_quantile_function(p):
    assert (0 <= p) and (p <= 1)
    return -math.log(-np.log(p))


def logistic_quantile_funciton(p):
    assert (0 <= p) and (p <= 1)
    return -math.log(1/p - 1)


class ModulesChannelsGatingHooks(nn.Module):
    def __init__(self, modules, pre_modules, gating_module):
        """
        A pair or two of lists of modules and pre/post module gating settings, and the associated gating weights module
        Note: all modules in their respected gating locations must match channel number with the gating_module
        :param modules: Single or list of modules on which to apply the gate on input/output
        :param pre_modules: Boolean, if true apply gate before calling module (input), otherwise after(output)
        :param gating_module: The gating module which produces the gating weights applied on all the modules
        """
        super(ModulesChannelsGatingHooks, self).__init__()
        assert isinstance(gating_module, ChannelsLogisticGating) or isinstance(gating_module, ChannelsLogisticGatingMasked)
        self.gating_module = gating_module

        if not isinstance(modules, list):
            assert not isinstance(pre_modules, list)
            modules = [modules]
            pre_modules = [pre_modules]
        else:
            assert isinstance(pre_modules, list)
            assert len(modules) == len(pre_modules)

        self.modules_and_sides = []
        self.hooks = []
        for i in range(len(modules)):
            m, p = modules[i], pre_modules[i]
            assert isinstance(m, nn.Module)
            assert isinstance(p, bool)

            self.modules_and_sides.append((m, p))

            if p:
                self.hooks.append(m.register_forward_pre_hook(self.gating_module.gate_output_pre))
            else:
                self.hooks.append(m.register_forward_hook(self.gating_module.gate_output))

    def set_gradient_adjustment(self, gradient_adjustment):
        self.gating_module.set_gradient_adjustment(gradient_adjustment)

    @property
    def output(self):
        return self.gating_module.output

    @property
    def hook(self):
        if len(self.hooks) == 1:
            return self.hooks[0]
        else:
            raise ValueError("There is more the one hook")

    def remove(self):
        """ Remove all pytorch hooks created in this module and in the associated gating module """
        for h in self.hooks:
            h.remove()
        self.gating_module.remove()


class WeightedL1Criterion(nn.Module):
    def __init__(self, weight):
        super(WeightedL1Criterion, self).__init__()
        if callable(weight):
            self.weight_fn = weight
        else:
            self.weight_fn = lambda : weight

    def forward(self, x, target):
        return self.weight_fn() * x.sum(1).mean()


class ChannelsLogisticGating(nn.Module):
    def __init__(self, channels, gradient_adjustment=None, gate_init_prob=0.99, random_init=False, hard_gating=True,
                 temperature=1):
        """
        Implementation of soft/hard gating using gumbel softmax (logistic sigmoid for binary) via pytorch hooks.
        The module expect an input module to apply the gating before or after passing through the input module
        :param channels: Number of channels in the input of the gate
        :param gradient_adjustment: A number or function for the multiplier of the gradients on the gating weights
        :param gate_init_prob: The initial probability to pass the gate
        :param random_init: randomize initialization probability around gate_init_prob
        :param hard_gating: Apply soft or hard gating
        :param temperature: temperature of the sigmoid
        """
        super(ChannelsLogisticGating, self).__init__()
        quantile_res = logistic_quantile_funciton(gate_init_prob)
        init_weights = torch.ones(channels) * quantile_res
        if random_init:
            # Add uniform +/- 10% random noise
            init_weights = init_weights + torch.rand(channels) * quantile_res / 5 - quantile_res / 10
        self.gating_weights = nn.Parameter(init_weights, requires_grad=True)
        # self.gumble = Gumbel(torch.zeros(channels),torch.ones(channels))
        self.logistic = LogisticDistribution(channels)
        self.sigmoid = nn.Sigmoid()
        self.hard_gating = hard_gating

        # placeholder to store gating result for loss
        self.output = None
        self.set_gradient_adjustment(gradient_adjustment)

        self.temperature = temperature

    def set_gradient_adjustment(self, gradient_adjustment):
        # hook for adjusting gradient in backprop
        # The function is separate from initialization to enable setting post init to resolve circular dependency
        if gradient_adjustment is not None:
            # assert not hasattr(self, 'gradient_adjustment_fn'), "cannot set gradient adjustment more then once"
            if callable(gradient_adjustment):
                self.gradient_adjustment_fn = gradient_adjustment
            else:
                self.gradient_adjustment_fn = lambda : gradient_adjustment
            self.backward_hook = self.sigmoid.register_backward_hook(self.gradient_adjustment_callback)

    def gate_output(self, module, input, output):
        """ after module hook function """
        return self(output)

    def gate_output_pre(self, module, input):
        """ before module hook function """
        return self(input[0])

    def gradient_adjustment_callback(self, module, grad_input, grad_output):
        """ gradient manipulation hook function """
        return (self.gradient_adjustment_fn() * grad_input[0],)

    @property
    def gating_probabilities(self):
        return self.gating_weights.sigmoid()

    def get_gating_mask(self, batch_size, device):
        """ compute the gating mask from local parameters given a batch size"""

        sample_tensor = torch.LongTensor([batch_size])

        # replicate gating weights for batch size and add gumbel sampled random variables
        # + for addition to the variable and - for the comparison against the other variable turned into comparing to 0

        # gating_weights = self.gating_weights.unsqueeze(0).repeat(batch_size, 1) + \
        #                  (self.gumble.sample(sample_tensor) - self.gumble.sample(sample_tensor)).to(device)

        # X ~ Gumbel(0, 1) and Y ~ Gumbel(0, 1)) ==> X - Y ~ Logistic(0, 1)
        gating_weights = self.gating_weights.unsqueeze(0).repeat(batch_size, 1) + \
                         self.logistic.sample(sample_tensor).to(device)
        gating_weights = gating_weights / self.temperature
        out = self.sigmoid(gating_weights)
        if self.hard_gating:
            # create 0.5 constants to compare sigmoid to
            half_padding = 0.5 * torch.ones_like(out, requires_grad=True)
            # combine together on the last axis
            padded = torch.cat([half_padding.unsqueeze(-1), out.unsqueeze(-1)], -1)
            # get a binary result which one is bigger
            max_ind = padded.argmax(-1).float()
            # trick to enable gradient
            gating = (max_ind - out).detach() + out
        else:
            gating = out

        return gating

    def forward(self, x):
        gating = self.get_gating_mask(x.size(0), x.device)

        # store gating result for loss and analysis
        self.output = gating
        # adjust gating to 4D tensor
        gating = gating.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.size(2), x.size(3))
        return gating * x

    def remove(self):
        if hasattr(self, 'backward_hook'):
            self.backward_hook.remove()


class ChannelsLogisticGatingMasked(ChannelsLogisticGating):
    def __init__(self, channels, gradient_adjustment=None, gate_init_prob=0.99, random_init=False, hard_gating=True,
                 temperature=1, deactivate_prob_threshold=0.5, auto_mask_recalculation=True):
        """
        A masked version which applies permanent gating once a weight has crossed a certain threshold probability
        (See base class for all other params)
        :param deactivate_prob_threshold: gate passing probability threshold under which a channel is permanently deactivated by the mask
        :param auto_mask_recalculation: if true, recalculate the mask before each forward application, otherwise, mask recalculation should be called externally
        """
        super(ChannelsLogisticGatingMasked, self).__init__(channels, gradient_adjustment, gate_init_prob, random_init,
                                                           hard_gating, temperature)
        self.active_channels_mask = nn.Parameter(data=torch.ones(channels), requires_grad=False)
        self.original_channels = channels
        self.deactivate_threshold = logistic_quantile_funciton(deactivate_prob_threshold)
        self.auto_mask_recalculation = auto_mask_recalculation

    def deactivate_channels(self):
        self.active_channels_mask.masked_fill_(self.gating_weights < self.deactivate_threshold, 0)

    def active_channels(self):
        return self.active_channels_mask.sum().item()

    def get_active_channels_mask(self, batch_size, device):
        if self.auto_mask_recalculation:
            self.deactivate_channels()
        return self.active_channels_mask.unsqueeze(0).repeat(batch_size,1).to(device)

    @property
    def gating_probabilities(self):
        return self.gating_weights.sigmoid() * self.active_channels_mask

    def forward(self, x):
        gating = self.get_gating_mask(x.size(0), x.device)

        # apply deactivation mask
        gating = gating * self.get_active_channels_mask(x.size(0), x.device)

        # store gating result for loss and analysis
        self.output = gating
        # adjust gating to 4D tensor
        gating = gating.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, x.size(2), x.size(3))
        return gating * x


class ModuleChannelsLogisticGating(ModulesChannelsGatingHooks):
    def __init__(self, module, channels, pre_module, gradient_adjustment=None, gate_init_prob=0.99, random_init=False,
                 hard_gating=True, temperature=1):
        super(ModuleChannelsLogisticGating, self).__init__(module, pre_module, ChannelsLogisticGating(
            channels, gradient_adjustment, gate_init_prob, hard_gating, temperature))


class ModuleChannelsLogisticGatingMasked(ModulesChannelsGatingHooks):
    def __init__(self, module, channels, pre_module, gradient_adjustment=None, gate_init_prob=0.99, random_init=False,
                 hard_gating=True, temperature=1, deactivate_prob_threshold = 0.5, auto_mask_recalculation=True):
        super(ModuleChannelsLogisticGatingMasked, self).__init__(module, pre_module, ChannelsLogisticGatingMasked(
            channels, gradient_adjustment, gate_init_prob, random_init, hard_gating, temperature,
            deactivate_prob_threshold, auto_mask_recalculation))
#
#
# if __name__ == '__main__':
#     torch.manual_seed(42)
#     c1 = nn.Conv2d(10,10,3)
#     c2 = nn.Conv2d(10,10,3)
#     hook = ModuleChannelsLogisticGating(c2, 10, True, 0.99).gating_module
#     loss_fn = nn.MSELoss()
#     optimizer_params = list(c1.parameters())+list(c2.parameters())+list(hook.parameters())
#     optimizer = torch.optim.SGD(optimizer_params, 0.01, momentum=0.9)
#
#     for i in range(3):
#         input = torch.randn((4,10,6,6))
#         res = c1(input)
#         res = c2(res)
#         # loss = loss_fn(res, torch.zeros_like(res))
#         loss = - hook.output.sum()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

    # import math
    # size = 1000
    # batch = 10
    # compare = 4
    # g1 = Gumbel(torch.zeros(size), torch.ones(size))
    # g2 = Gumbel(torch.zeros(size), torch.ones(size))
    # l = LogisticDistribution(size)
    #
    # x1 = g1.sample(torch.LongTensor([batch]))
    # x2 = g2.sample(torch.LongTensor([batch]))
    # x3 = l.sample(torch.LongTensor([batch]))
    #
    # print(((x1 - x2) > compare).sum().item()/size, (x3 > compare).sum().item()/size, 1 - 1/(1+math.exp(-compare)))
