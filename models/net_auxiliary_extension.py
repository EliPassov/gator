import torch
from torch import nn

class BaseHook(nn.Module):
    def close(self):
        self.hook.remove()


class OutputHook(BaseHook):
    def __init__(self, module):
        super(BaseHook, self).__init__()
        self.hook = module.register_forward_hook(self.hook_output)

    def hook_output(self, module, input, output):
        self.inner_output = output

    @property
    def output(self):
        return self.inner_output


class ClassificationLayerHook(OutputHook):
    def __init__(self, module, channels, num_classes, confidence=False):
        super(BaseHook, self).__init__()
        self.hook = module.register_forward_hook(self.hook_output)
        self.confidence = confidence
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.average_pool = lambda x:x.mean(-1).mean(-1)
        self.fc = nn.Linear(channels, num_classes + (1 if confidence else 0))
        self.softmax = nn.Softmax(1)
        if confidence:
            self.sigmoid = nn.Sigmoid()

    def hook_output(self, module, input, output):
        module_output = self.fc(self.average_pool(self.relu(self.bn(output))))
        if self.confidence:
            # sigmoid is performed in place which messes up the backprop, hence the creation of a new variable
            out = torch.zeros_like(module_output, device=module_output.device)
            out[:, :-1] = self.softmax(module_output[:, :-1])
            out[:,-1] = self.sigmoid(module_output[:,-1])
            self.inner_output = out
        else:
            module_output = self.softmax(module_output)
            self.inner_output = module_output


class NetWithAuxiliaryOutputs(nn.Module):
    """ A module holding a main module and any related hooks which have an output for loss/reporting purposes"""
    def __init__(self, net, forward_hooks):
        super(NetWithAuxiliaryOutputs, self).__init__()
        self.net = net
        # for forward_hook in forward_hooks:
        #     assert hasattr(forward_hook, 'output')
            # assert isinstance(forward_hook, OutputHook)
        # Assures learning for parameters inside hook
        self.forward_hooks = nn.ModuleList(forward_hooks)

    def forward(self, x):
        return (self.net(x), [forward_hook.output for forward_hook in self.forward_hooks])
        # """ Returns net and hook outputs in training, only net in evaluation"""
        # if self.training:
        #     # forwarding through net triggers hooks
        #     return (self.net(x), [forward_hook.output for forward_hook in self.forward_hooks])
        # return self.net(x)


class CriterionWithAuxiliaryLosses(nn.Module):
    def __init__(self, main_criterion=nn.CrossEntropyLoss(), auxiliary_criteria=None, auxiliary_multipliers=None,
                 get_aux_loss_aggregated=False, report_func=None, process_mapping=None):
        """
        A module for processing multiple losses and/or reporting by-products
        :param main_criterion: main loss function
        :param auxiliary_criteria: list of auxiliary loss function which should match the loss output number
        :param auxiliary_multipliers: list of multipliers for auxiliary loss
        :param get_aux_loss_aggregated: return all auxiliary loss as one number
        :param report_func: function or list of functions used for reporting outputs (not part of loss)
        :param process_mapping: mapping of the auxiliary outputs whether to use for loss, reporting or both (l/r/lr)
        In case of reporting the input should not be processed by auxiliary_criteria, in case of both it should and
        the expected output should be a tuple of the loss element and the reporting element
        """
        super(CriterionWithAuxiliaryLosses, self).__init__()
        self.main_criterion = main_criterion
        assert auxiliary_criteria is not None and isinstance(auxiliary_criteria, list)
        self.auxiliary_criteria = auxiliary_criteria
        if isinstance(auxiliary_multipliers, float) or isinstance(auxiliary_multipliers, int):
            auxiliary_multipliers = [auxiliary_multipliers] * len(auxiliary_criteria)
        if not isinstance(auxiliary_multipliers, list) or len(auxiliary_multipliers) != len(auxiliary_criteria):
            raise ValueError('auxiliary multipliers must be either a number of a list of numbers the same length as auxiliary_criteria')
        self.auxiliary_multipliers = auxiliary_multipliers
        self.get_aux_loss_aggregated = get_aux_loss_aggregated
        self.report_func = report_func
        self.process_mapping = process_mapping

    def forward(self, input, target):
        assert isinstance(input, tuple)
        assert len(input) == 2
        assert isinstance(input[1], list)

        result = {}

        if self.main_criterion is not None:
            main_loss = self.main_criterion(input[0], target)
        else:
            main_loss = torch.zeros((0), device=input[1][0].device).sum()

        if self.process_mapping is None:
            loss_inputs = input[1]
        else:
            loss_inputs = [input[1][i] for i in range(len(input[1])) if 'l' in self.process_mapping[i]]

        assert len(loss_inputs) == len(self.auxiliary_criteria)

        auxiliary_loss_outputs = [self.auxiliary_criteria[i](loss_inputs[i], target) for i in range(len(loss_inputs))]

        if self.report_func is not None:
            if self.process_mapping is None:
                report_inputs = input[1]
            else:
                report_inputs = []
                # mapping the outputs: r for initial input lr/rl for loss output,
                # for the latter we need to count the loss outputs hence the use of l_ind
                l_ind = 0
                for i in range(len(self.process_mapping)):
                    if self.process_mapping[i] == 'r':
                        report_inputs.append(input[1][i])
                    elif 'r' in self.process_mapping[i]:
                        report_inputs.append(auxiliary_loss_outputs[l_ind][1])
                        # After using the report part of the output's tuple,
                        # keep just the single loss value for the loss processing
                        auxiliary_loss_outputs[l_ind] = auxiliary_loss_outputs[l_ind][0]
                    if 'l' in self.process_mapping[i]:
                        l_ind += 1

            if isinstance(self.report_func, list):
                report_function = self.report_func
                assert len(report_function) == len(report_inputs)
            else:
                report_function = [self.report_func] * len(report_inputs)

            result['report'] = {}
            for i in range(len(report_inputs)):
                report_dic = report_function[i](i, report_inputs[i], target)
                result['report'].update(report_dic)

        auxiliary_losses = [self.auxiliary_multipliers[i] * auxiliary_loss_outputs[i] for i in range(len(auxiliary_loss_outputs))]

        total_loss = main_loss
        for aux_loss in auxiliary_losses:
            total_loss = total_loss + aux_loss

        result['loss'] = total_loss

        if self.get_aux_loss_aggregated:
            result['breakdown'] = [main_loss.item(), sum(auxiliary_losses).item()]
        else:
            result['breakdown'] = [main_loss.item()] + [a.item() for a in auxiliary_losses]

        return result
