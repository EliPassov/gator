import torch
from torch.optim.optimizer import Optimizer


class MultiGroupDynamicLROptimizer:
    def __init__(self, optimizer, lr_update_map):
        """
        A class mimicking Optimizer class which will apply the given adjustment functions on param_groups of optimizer.
        The multiplied rates are overwritten on each call of step while the original learning rate is kept
        in the first group to comply with standard practices for learning rate updates.
        The class mimics but not inherits to enable using any of the inheriting optimizer methods.
        :param optimizer: the actual optimizer containing all the group of parameters
        :param lr_update_map: a map of group ids for param_groups to lambda functions which multiply the learning rate
        """
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer
        assert isinstance(lr_update_map, dict)
        for i, _ in lr_update_map.items():
            assert i < len(optimizer.param_groups)
        self.lr_update_map = lr_update_map

    def __getstate__(self):
        return self.optimizer.__getstate__()

    def __setstate__(self, state):
        self.optimizer.__setstate__(state)

    def __repr__(self):
        return self.optimizer.__repr__()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        current_lr = self.optimizer.param_groups[0]['lr']
        for group_id, lr_multiplier in self.lr_update_map.items():
             self.optimizer.param_groups[group_id]['lr'] = current_lr * lr_multiplier()
        self.optimizer.step()
        # retort learning rate to first group in case it was modified
        self.optimizer.param_groups[0]['lr'] = current_lr

    def add_param_group(self, param_group):
        raise ValueError('add_param_group must be applied on the optimizer object of this class explicitly!')

    @property
    def param_groups(self):
        return self.optimizer.param_groups
