import unittest
import copy

import torch
from torch import nn

from models.channel_gates import ModuleChannelsLogisticGating, ModuleChannelsLogisticGatingMasked


#some test only pass at high probability which checks out for the seed (was a bit lazy to make them exact)
torch.manual_seed(42)


class SimpleNet(nn.Module):
    def __init__(self, num_features=10):
        super(SimpleNet, self).__init__()
        self.c1 = nn.Conv2d(num_features, num_features, 3)
        self.c2 = nn.Conv2d(num_features, num_features, 3)

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        return out


class TestChannelsLogisticGating(unittest.TestCase):
    def test_hook(self):
        net = SimpleNet()
        hook = ModuleChannelsLogisticGating(net.c1, net.c1.out_channels, False).gating_module
        input = torch.randn((4, 10, 6, 6))

        output = net(input)
        loss = nn.MSELoss()(output, torch.zeros_like(output))
        loss.backward()

        # have grads
        self.assertIsNotNone(hook.gating_weights.grad)
        # grads are nonzero
        self.assertTrue((hook.gating_weights.grad!=0).float().mean() > 0.99)

    def test_pre_hook(self):
        net = SimpleNet()
        hook = ModuleChannelsLogisticGating(net.c2, net.c2.in_channels, True).gating_module
        input = torch.randn((4, 10, 6, 6))

        output = net(input)
        loss = nn.MSELoss()(output, torch.zeros_like(output))
        loss.backward()

        self.assertIsNotNone(hook.gating_weights.grad)
        self.assertTrue((hook.gating_weights.grad!=0).float().mean() > 0.99)

    def test_loss_on_gates(self):
        net = SimpleNet()
        hook = ModuleChannelsLogisticGating(net.c1, net.c1.out_channels, False).gating_module
        input = torch.randn((4, 10, 6, 6))

        output = net(input)
        loss = - hook.output.sum()
        loss.backward()

        self.assertIsNotNone(hook.gating_weights.grad)
        self.assertTrue((hook.gating_weights.grad!=0).float().mean() > 0.99)
        # make sure all gradients are negative
        self.assertTrue((hook.gating_weights.grad > 0).sum() == 0)

    def test_gradient_weight_adjustment(self):
        num_features = 100
        net1 = SimpleNet(num_features=num_features)
        net2 = copy.deepcopy(net1)
        grad_factor = 10
        gate_init_prob = 1-1e-5
        batch_size = 4
        hook1 = ModuleChannelsLogisticGating(net1.c1, net1.c1.out_channels, False, gradient_adjustment=1, gate_init_prob=gate_init_prob).gating_module
        hook2 = ModuleChannelsLogisticGating(net2.c1, net2.c1.out_channels, False, gradient_adjustment=grad_factor, gate_init_prob=gate_init_prob).gating_module
        input = torch.randn((batch_size, num_features, 6, 6))
        # optimizer_params = list(net.c1.parameters())+list(net.c2.parameters())+list(hook.parameters())
        # optimizer = torch.optim.SGD(optimizer_params, 0.01, momentum=0.9)

        output1 = net1(input)
        output2 = net2(input)
        loss1 = hook1.output.sum()
        loss2 = hook2.output.sum()
        loss = loss1 + loss2
        loss.backward()

        # stochasticity and imbalance go gumbel sampled positive and negative contribution, make it hard for an exact value match up
        # grads1 = hook1.gating_weights.grad
        # grads2 = hook2.gating_weights.grad
        # self.assertAlmostEqual(grad_factor * grads1.sum()/ grads2.sum(), 1, delta = 0.3)

        grads1_log_mean = (hook1.gating_weights.grad).log().mean()
        grads2_log_mean = (hook2.gating_weights.grad / grad_factor).log().mean()
        self.assertAlmostEqual((grads1_log_mean/ grads2_log_mean).item(), 1 , delta = 0.05)

    def test_masking(self):
        net = SimpleNet()
        hook = ModuleChannelsLogisticGatingMasked(net.c1, net.c1.out_channels, False, gate_init_prob= 0.5 + 1e-10, deactivate_prob_threshold=0.5).gating_module
        input = torch.randn((4, 10, 6, 6))

        optimizer_params = list(net.c1.parameters())+list(net.c2.parameters())+list(hook.parameters())
        optimizer = torch.optim.SGD(optimizer_params, 0.01, momentum=0.9)

        output = net(input)

        self.assertEqual((hook.active_channels_mask==0).sum().item(), 0)

        loss = hook.output.sum()
        loss.backward()
        optimizer.step()

        output = net(input)
        self.assertEqual((hook.active_channels_mask==1).sum().item(), 0)

    def test_masking_no_grad_first_gate(self):
        net = SimpleNet()
        hook1 = ModuleChannelsLogisticGatingMasked(net.c1, net.c1.out_channels, False, gate_init_prob= 0.5 + 1e-10, deactivate_prob_threshold=0.5).gating_module
        hook2 = ModuleChannelsLogisticGatingMasked(net.c2, net.c2.in_channels, True, gate_init_prob= 1- 1e-2, deactivate_prob_threshold=0.5).gating_module
        input = torch.randn((4, 10, 6, 6))

        optimizer_params = list(net.c1.parameters())+list(net.c2.parameters())+list(hook1.parameters())+list(hook2.parameters())
        optimizer = torch.optim.SGD(optimizer_params, 0.01, momentum=0.0)

        output = net(input)

        # all channels still active
        self.assertEqual((hook1.active_channels_mask==0).sum().item(), 0)
        self.assertEqual((hook2.active_channels_mask==0).sum().item(), 0)

        loss = hook1.output.sum() + hook2.output.sum()
        loss.backward()

        # all grads > 0
        self.assertEqual(((hook1.gating_weights.grad > 0 ) == 0).sum(), 0)
        self.assertEqual(((hook2.gating_weights.grad > 0 ) == 0).sum(), 0)

        optimizer.step()

        optimizer.zero_grad()

        output = net(input)
        # all channels deactivated for hook1, none for hook2
        self.assertEqual((hook1.active_channels_mask==1).sum().item(), 0)
        self.assertEqual((hook2.active_channels_mask==0).sum().item(), 0)

        loss = hook1.output.sum() + hook2.output.sum()
        loss.backward()

        # all grads are zero for hook1, all > 0 for hook2
        self.assertEqual((hook1.gating_weights.grad > 0 ).sum(), 0)
        self.assertEqual(((hook2.gating_weights.grad > 0 ) == 0).sum(), 0)

    def test_masking_no_grad_second_gate(self):
        net = SimpleNet()
        hook1 = ModuleChannelsLogisticGatingMasked(net.c1, net.c1.out_channels, False, gate_init_prob=1- 1e-2 , deactivate_prob_threshold=0.5).gating_module
        hook2 = ModuleChannelsLogisticGatingMasked(net.c2, net.c2.in_channels, True, gate_init_prob=0.5 + 1e-10, deactivate_prob_threshold=0.5).gating_module
        input = torch.randn((4, 10, 6, 6))

        optimizer_params = list(net.c1.parameters())+list(net.c2.parameters())+list(hook1.parameters())+list(hook2.parameters())
        optimizer = torch.optim.SGD(optimizer_params, 0.01, momentum=0.0)

        output = net(input)

        # all channels still active
        self.assertEqual((hook1.active_channels_mask==0).sum().item(), 0)
        self.assertEqual((hook2.active_channels_mask==0).sum().item(), 0)

        loss = hook1.output.sum() + hook2.output.sum()
        loss.backward()

        # all grads > 0
        self.assertEqual(((hook1.gating_weights.grad > 0 ) == 0).sum(), 0)
        self.assertEqual(((hook2.gating_weights.grad > 0 ) == 0).sum(), 0)

        optimizer.step()

        optimizer.zero_grad()

        output = net(input)
        # all channels deactivated for hook2, none for hook1
        self.assertEqual((hook1.active_channels_mask==0).sum().item(), 0)
        self.assertEqual((hook2.active_channels_mask==1).sum().item(), 0)

        loss = hook1.output.sum() + hook2.output.sum()
        loss.backward()

        # all grads are zero for hook2, all > 0 for hook1
        self.assertEqual(((hook1.gating_weights.grad > 0 ) == 0).sum(), 0)
        self.assertEqual((hook2.gating_weights.grad > 0 ).sum(), 0)

    def test_no_grad_from_main_loss_after_masking_first_gate(self):
        net = SimpleNet(100)
        # fill positive weights
        net.c1.weight.data.fill_(0.1)
        net.c2.weight.data.fill_(0.1)
        hook1 = ModuleChannelsLogisticGatingMasked(net.c1, net.c1.out_channels, False, gate_init_prob= 0.5 + 1e-10, deactivate_prob_threshold=0.5).gating_module
        hook2 = ModuleChannelsLogisticGatingMasked(net.c2, net.c2.in_channels, True, gate_init_prob= 1- 1e-4, deactivate_prob_threshold=0.5).gating_module
        input = torch.randn((1, 100, 6, 6))

        optimizer_params = list(hook1.parameters())+list(hook2.parameters())
        optimizer = torch.optim.SGD(optimizer_params, 1, momentum=0.0)

        output = net(input)

        # all channels still active
        self.assertEqual((hook1.active_channels_mask==0).sum().item(), 0)
        self.assertEqual((hook2.active_channels_mask==0).sum().item(), 0)

        loss = nn.MSELoss()(output, torch.zeros_like(output))
        loss.backward()

        # some grads > 0
        self.assertGreater((hook1.gating_weights.grad > 0 ).sum(), 0)
        self.assertGreater((hook2.gating_weights.grad > 0 ).sum(), 0)

        optimizer.step()


        optimizer.zero_grad()

        output = net(input)
        # all channels deactivated for hook1, none for hook2
        self.assertEqual((hook1.active_channels_mask==1).sum().item(), 0)
        self.assertEqual((hook2.active_channels_mask==0).sum().item(), 0)

        loss = nn.MSELoss()(output, torch.zeros_like(output))
        loss.backward()


        # all grads are zero for hook1 and hook2
        self.assertEqual((hook1.gating_weights.grad > 0 ).sum(), 0)
        self.assertEqual((hook2.gating_weights.grad > 0 ).sum(), 0)

    def test_no_grad_from_main_loss_after_masking_second_gate(self):
        net = SimpleNet(100)
        # fill positive weights
        net.c1.weight.data.fill_(0.1)
        net.c2.weight.data.fill_(0.1)
        hook1 = ModuleChannelsLogisticGatingMasked(net.c1, net.c1.out_channels, False, gate_init_prob=1- 1e-4 , deactivate_prob_threshold=0.5).gating_module
        hook2 = ModuleChannelsLogisticGatingMasked(net.c2, net.c2.in_channels, True, gate_init_prob=0.5 + 1e-10, deactivate_prob_threshold=0.5).gating_module
        input = torch.randn((1, 100, 6, 6))

        optimizer_params = list(hook1.parameters())+list(hook2.parameters())
        optimizer = torch.optim.SGD(optimizer_params, 1, momentum=0.0)

        output = net(input)

        # all channels still active
        self.assertEqual((hook1.active_channels_mask==0).sum().item(), 0)
        self.assertEqual((hook2.active_channels_mask==0).sum().item(), 0)

        loss = nn.MSELoss()(output, torch.zeros_like(output))
        loss.backward()

        # some grads > 0
        self.assertGreater((hook1.gating_weights.grad > 0 ).sum(), 0)
        self.assertGreater((hook2.gating_weights.grad > 0 ).sum(), 0)

        optimizer.step()

        optimizer.zero_grad()

        output = net(input)
        # all channels deactivated for hook2, none for hook1
        self.assertEqual((hook1.active_channels_mask==0).sum().item(), 0)
        self.assertEqual((hook2.active_channels_mask==1).sum().item(), 0)

        loss = nn.MSELoss()(output, torch.zeros_like(output))
        loss.backward()

        # all grads are zero for hook1 and hook2
        self.assertEqual((hook1.gating_weights.grad > 0 ).sum(), 0)
        self.assertEqual((hook2.gating_weights.grad > 0 ).sum(), 0)
