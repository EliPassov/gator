import argparse
from datetime import datetime as dtm
from functools import partial
import json
import os
import sys
from time import time

from torchvision.models.resnet import resnet18

from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data.dataloader import default_collate

from data.dataset import SoftTargetDatasetWrapper
from data.dataset_factory import get_train_test_datasets, get_same_image_random_transform
from eval import EvaluatorWithEvalHandler, GenericEvaluator, get_ten_crop_dataloader
from external_models.cifar10_models.resnext import ResNeXt
from models.gated_grouped_conv import SUB_GROUP_LOSS_CLASSES_REQUIRING_TARGET
from models.squeeze_excitation import SEClassificationLossHook, SEWeightsLayer
from utils.distillation import KLD_loss, TemperatureFC, AverageSoftLoss, nll_cross_entropy
from utils.ensemble_net import get_ensemble_class
from utils.hook_utils import handle_reporting_hooks
from utils.net_utils import NetWithResult, load_net
from utils.probability_calculator import calc_net_prediction
from utils.run_arg_parser import parse_net_args, NET_LOAD_TYPE, parse_net_args_inner
from utils.schedule import EPOCHSnLRSchedules


LOSS_CLASSES_REQUIRING_TARGET = SUB_GROUP_LOSS_CLASSES_REQUIRING_TARGET + [SEClassificationLossHook]


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup_folder", type=str, action="store", default=None)
    parser.add_argument("--batch_size", type=int, action="store", default=16)
    parser.add_argument("--pretrained_net", type=str, action="store", default=None)
    parser.add_argument("--cifar10_net", type=str, action="store", default=None)
    parser.add_argument("--net_with_criterion", type=str, action="store", default=None)
    parser.add_argument("--num_classes", type=int, action="store", default=None)
    parser.add_argument("--trained_net_path", type=str, action="store", default=None)
    parser.add_argument("--auxiliary_net_path", type=str, action="store", default=None)
    parser.add_argument("--dataset_name", type=str, action="store", default="cifar10")
    parser.add_argument("--train_data_path", type=str, action="store", default=None)
    parser.add_argument("--auxiliary_data_path", type=str, action="store", default=None)
    parser.add_argument("--ensemble", action="store", default=None)
    parser.add_argument("--val_data_path", type=str, action="store", default=None)
    parser.add_argument("--pretrained_teacher_net", action="store", default=None)
    parser.add_argument("--teacher_net", type=str, action="store", default=None)
    parser.add_argument("--teacher_net_path", action="store", default=None)
    parser.add_argument("--teacher_loss_ratio", action="store", default=0.5)
    parser.add_argument("--correct_teacher", action='store_true', default=False)
    parser.add_argument("-T", type=float, default=1.0)
    parser.add_argument("--soft_targets", action='store_true', default=False)
    parser.add_argument("-e", action='store_true', default=False, dest='run_eval')
    parser.add_argument("--aux_classifiers_confidence", action='store', default=0.0)
    parser.add_argument("-train_aux_only", action='store_true', default=False, dest='train_aux_only')
    parser.add_argument("--load_type", type=str, default='normal')
    parser.add_argument("-class_breakdown", action='store_true', default=False, dest='class_breakdown')
    parser.add_argument("--collate_repeat", type=float, action="store", default=None)

    return parser.parse_args()


def parse_args(args):
    assert args.backup_folder is not None
    backup_folder = args.backup_folder
    if not os.path.exists(backup_folder):
        os.mkdir(backup_folder)
    with open(os.path.join(backup_folder, 'args_dictionary_with_defaults.json'), 'w+') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(backup_folder, 'args_line.txt'), 'w+') as f:
        f.write('\n'.join(sys.argv))

    batch_size = args.batch_size

    net, criterion = parse_net_args(args, True)

    epochs_n_lr_schedules = 'cifar10_short'
    if args.net_with_criterion is not None:
        if args.net_with_criterion[-6:] == '_class' or args.net_with_criterion[-11:] == '_class_conf':
            epochs_n_lr_schedules = 'cifar10_se_classification'
        elif args.trained_net_path is not None:
            epochs_n_lr_schedules = 'cifar10_retrain'
    epochs_to_run, lr_schedule = EPOCHSnLRSchedules[epochs_n_lr_schedules]

    aux_classifiers_confidence = float(args.aux_classifiers_confidence)
    if aux_classifiers_confidence > 0:
        # net.loss_hooks = nn.ModuleList([SEClassificationLossHook(layer.se_layer_weights, 1, 10, layer.se_layer_weights.out_planes,
        #                                            float(aux_classifiers_confidence)) for layer in
        #                   list(net.modules()) if isinstance(layer, BottleneckX) and hasattr(layer, 'se_layer_weights')])

        net.loss_hooks = nn.ModuleList([SEClassificationLossHook(layer.fc1, 1, 10, layer.hidden_features,
                                                   float(aux_classifiers_confidence)) for layer in
                          list(net.modules()) if isinstance(layer, SEWeightsLayer)])

    if args.trained_net_path is not None:
        net = load_net(args.trained_net_path, net, args.load_type)

    if args.train_aux_only:
        for param in net.parameters():
            param.requires_grad = False
        for hook in net.loss_hooks:
            if isinstance(hook, SEClassificationLossHook):
                for param in hook.parameters():
                    param.requires_grad = True

    train_batch_size = batch_size if args.auxiliary_net_path is None else batch_size // 2

    train_dataset, test_dataset = get_train_test_datasets(args.dataset_name, args)

    collate_fn = default_collate
    if args.collate_repeat is not None:
        collate_fn = get_same_image_random_transform(args.dataset_name, args.collate_repeat)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, num_workers=16, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=16)

    multiple_predictions = False
    aux_confidence_threshold = None
    # if hasattr(net, 'handle_batch_eval') or hasattr(net, 'handle_post_run_eval'):
    #     evaluator_class = EvaluatorWithEvalHandler
    if True:
        single_result = not(args.net_with_criterion is not None and ('_class' in args.net_with_criterion or '_pooling' in args.net_with_criterion))
        multiple_predictions = args.net_with_criterion is not None and '_class' in args.net_with_criterion
        aux_confidence = (not single_result) and '_conf' in args.net_with_criterion
        if aux_confidence:
            aux_confidence_threshold = 0.5
        evaluator_class = partial(GenericEvaluator, class_breakdown=args.class_breakdown, return_breakdown=False,
                                  single_result=single_result, multiple_predictions=multiple_predictions,
                                  aux_confidence=aux_confidence)
    evaluator = evaluator_class(NetWithResult(net, multiple_predictions, aux_confidence_threshold), test_loader)

    auxiliary_net, auxiliary_loader = None, None
    if args.auxiliary_net_path is not None:
        assert args.auxiliary_data_path is not None
        assert args.ensemble is not None
        raise NotImplementedError('Fix net loading to support state_dict')
        auxiliary_net = torch.load(args.auxiliary_net_path)
        auxiliary_net = get_ensemble_class(args.ensemble)(auxiliary_net)
        auxiliary_loader = get_ten_crop_dataloader(args.auxiliary_data_path, shuffle=True, batch_size=train_batch_size)

    teacher_net = None
    if args.teacher_net is not None:
        assert args.teacher_net_path is not None
        teacher_net = parse_net_args_inner(NET_LOAD_TYPE.Cifar10, args.teacher_net, args.num_classes, False)
        teacher_net = load_net(args.teacher_net_path, teacher_net)
    elif args.pretrained_teacher_net is not None:
        teacher_net = parse_net_args(NET_LOAD_TYPE.Pretrained, args.pretrained_teacher_net, args.num_classes, False)

    teacher_loss_ratio = float(args.teacher_loss_ratio)

    if args.soft_targets:
        print("Calculating teacher's confusion matrix with T={:5.2f} teacher correction is {}"
              .format(args.T, "on" if args.correct_teacher else "off"))
        avg, nums = calc_net_prediction(teacher_net, train_loader, T=args.T, correct_net=args.correct_teacher)
        for i in range(avg.size(0)):
            print("{} : {} \n".format(int(nums[i].item()), ", ".join(["{:.4f}".format(a) for a in avg[i, :].numpy()])))

        criterion = lambda x,t: {'loss': nll_cross_entropy(x, t, args.T)}

        train_dataset = SoftTargetDatasetWrapper(train_dataset, avg)
        test_dataset = SoftTargetDatasetWrapper(test_dataset, avg)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, num_workers=16, drop_last=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=16)

    run_eval = args.run_eval

    return net, criterion, batch_size, backup_folder, train_loader, test_loader, evaluator, epochs_to_run, lr_schedule,\
           auxiliary_net, auxiliary_loader, teacher_net, teacher_loss_ratio, run_eval


def train(net, backup_folder, train_loader, epochs_to_run, evaluator, loss_func, optimizer, lr_schedule,
          auxiliary_net=None, auxiliary_loader=None, teacher_net=None, teacher_loss_func=None, teacher_loss_ratio=None,
          avg_net_pred=None, T=None):
    net = net.cuda()

    writer = SummaryWriter(os.path.join(backup_folder, 'log', 'run_{}'.format(dtm.now().strftime("%Y%m%d_%H%M%S"))))

    avg_loss, avg_teacher_loss, avg_loss_breakdown = None, None, None

    if auxiliary_net is not None:
        auxiliary_net = auxiliary_net.cuda()
        it = iter(auxiliary_loader)

    T_std = 0
    # T_l = 1 + torch.randn(10, requires_grad=True)*0.2

    auxiliary_classifier_loss_hooks = [hook for hook in net.loss_hooks if isinstance(hook, SEClassificationLossHook)] if hasattr(net, 'loss_hooks') else []

    for epoch in range(0, epochs_to_run):
        # print adjusted weights for adaptive pruning when in/out channels of convolution being pruned effect the flop cost of pruning the other side of the same convolution
        # print(','.join(["{:8.2f}".format(i.gradient_adjustment_fn()) for i in net.forward_hooks]), "Aux weights",','.join(["{:8.2f}".format(i.weight_fn()) for i in criterion.auxiliary_criteria]))

        corrects = 0
        auxiliary_corrects = [0 for _ in range(len(auxiliary_classifier_loss_hooks))]
        auxiliary_confident_corrects = [0 for _ in range(len(auxiliary_classifier_loss_hooks))]
        auxiliary_confident_count = [0 for _ in range(len(auxiliary_classifier_loss_hooks))]

        for batch_id, (img_path, data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            w_id = batch_id + epoch * train_loader.__len__()

            lr=lr_schedule.get_value(epoch)

            if isinstance(optimizer, list):
                for op,w in optimizer:
                    for param_group in op.param_groups:
                        param_group['lr'] = lr*w
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if isinstance(optimizer, list):
                for op,_ in optimizer:
                    op.zero_grad()
            else:
                optimizer.zero_grad()

            #region Auxiliary Net (for partial unsupervised data)

            if auxiliary_net is not None:
                try:
                    img_path2, data2 = next(it)
                except StopIteration:
                    it = iter(auxiliary_loader)
                    img_path2, data2 = next(it)

                data2 = Variable(data2.cuda())
                target2 = torch.LongTensor(auxiliary_net(data2)).cuda()
                # assuming the first image in the augmented data loader is a non-augmented image
                data=torch.cat([data,data2[:, 0, ...]])
                target=torch.cat([target, target2])

            #endregion

            t0 = time()

            # Ugly code for gluing target for module which requires them\
            if hasattr(net, 'loss_hooks'):
                for hook in net.loss_hooks:
                    for loss_type in LOSS_CLASSES_REQUIRING_TARGET:
                        if isinstance(hook, loss_type):
                            hook.target = target

            output = net(data)

            if hasattr(net, 'loss_hooks'):
            # And more ugly code for removing glued target
                for hook in net.loss_hooks:
                    for loss_type in LOSS_CLASSES_REQUIRING_TARGET:
                        if isinstance(hook, loss_type):
                            del hook.target

            t1 = time()

            loss_result = loss_func(output, target)
            loss = loss_result['loss']
            if isinstance(output, tuple):
                output = output[0]

            t2 = time()

            loss_breakdown = loss_result.get('breakdown')

            if len(target.shape) == 1:
                corrects = corrects + (output.argmax(1) == target).sum().item()
            else:
                corrects = corrects + (output.argmax(1) == target.argmax(1)).sum().item()

            if hasattr(net, 'loss_hooks'):
                for i, hook in enumerate(auxiliary_classifier_loss_hooks):
                    auxiliary_corrects[i] += (hook.auxiliary_output[0] == target).sum().item()
                    if hook.auxiliary_output[1] is not None:
                        confident_responses = (hook.auxiliary_output[1] > 0.5)
                        auxiliary_confident_corrects[i] += (hook.auxiliary_output[0] == target)[confident_responses].sum().item()
                        auxiliary_confident_count[i] += confident_responses.sum().item()

            total_loss = None

            #region Distillation

            if teacher_net is not None:
                teacher_output = teacher_net(data)

                # T = 1.2
                # T2 = 1.5

                # temp = get_temperature(F.softmax(output,1), F.softmax(teacher_output,1), target, T)

                if net.T_l is None and net.T_fc is None:
                    teacher_loss = KLD_loss(output, teacher_output, T)
                elif net.T_l is not None:
                    T_specific = torch.zeros(target.size(0), 10).cuda()
                    for i in range(target.size(0)):
                        T_specific[i,:] = net.T_l[target[i]]
                    teacher_loss = KLD_loss(output, teacher_output, T_specific)
                elif net.T_fc is not None:
                    T_fc = net.T_fc(target, teacher_output, output)
                    teacher_loss = KLD_loss(output, teacher_output, T_fc.repeat(1, 10))

                avg_loss_v = torch.zeros(target.size(0), 10).cuda()
                for i in range(target.size(0)):
                    avg_loss_v[i,:] = avg_net_pred[target[i]]

                teacher_avg_loss = (nn.KLDivLoss(reduction='none')(F.log_softmax(output / T, dim=1), avg_loss_v) * (T * T)).sum()

                alpha = 1.0

                teacher_loss = (1-alpha)*teacher_loss + alpha* teacher_avg_loss

                # teacher_loss = KLD_where_better_loss(outpcorrectsut, teacher_output, target, T, 0.7)

                teacher_softmax = F.softmax(teacher_output, 1)

                # total_loss, loss, teacher_loss, custom_weights = teacher_weighted_loss(output, teacher_output, target, T)

                teacher_maxes, teacher_max_indices=teacher_softmax.max(1)
                average_max_teacher = teacher_maxes.mean().item()
                first_index = torch.range(0, teacher_output.size(0)-1).type(torch.LongTensor)
                teacher_softmax[first_index, teacher_max_indices] = 0
                average_second_teacher = teacher_softmax.max(1)[0].mean().item()
                max_second_teacher = teacher_softmax.max().item()

                # teacher_loss, temp = KLD_adaptive_loss(output, teacher_output, target, T)
                # teacher_loss = KLD_where_better_loss(output, teacher_output, target, T)

                # normal loss
                # teacher_loss = teacher_loss_func(output, teacher_output)
                # teacher_loss, temp = KLD_student_adaptive_temp(output, teacher_output, target, T, T2, ratio=0.2, correct_teacher=False)
                avg_teacher_loss = 0.9 * avg_teacher_loss + 0.1 * teacher_loss.item() if avg_teacher_loss is not None else teacher_loss.item()

                # loss, portion = combined_loss_student(output, teacher_output, target, T)


                if batch_id % 10 == 1:
                    writer.add_scalar('train/teacher_loss', avg_teacher_loss, w_id)
                    writer.add_scalar('train/average_max_teacher', average_max_teacher, w_id)
                    writer.add_scalar('train/average_second_teacher', average_second_teacher, w_id)
                    writer.add_scalar('train/max_second_teacher', max_second_teacher, w_id)
                    if net.T_l is not None:
                        writer.add_scalar('train/temp', net.T_l.mean().item(), w_id)
                        writer.add_scalar('train/temp_std', net.T_l.std().item(), w_id)
                    elif net.T_fc is not None:
                        writer.add_scalar('train/temp', T_fc.mean().item(), w_id)
                        writer.add_scalar('train/temp_std', T_fc.std().item(), w_id)

                    # writer.add_scalar('train/temp_avg', temp.mean().item(), w_id)
                    # writer.add_scalar('train/temp_std', temp.std().item(), w_id)
                    # writer.add_scalar('train/temp_min', temp.min().item(), w_id)
                    # writer.add_scalar('train/temp_max', temp.max().item(), w_id)
                    # writer.add_scalar('train/kld_portion', portion.item(), w_id)

            #endregion

            # print(temp.mean().item(),temp.std().item(),temp.min().item(),temp.max().item())

            avg_loss = 0.9*avg_loss + 0.1*loss.item() if avg_loss is not None else loss.item()
            if loss_breakdown is not None:
                if avg_loss_breakdown is None:
                    avg_loss_breakdown = [0] * len(loss_breakdown)
                    # negate 0.1 multiplication of first value
                    loss_breakdown = [10*l for l in loss_breakdown]
                for i in range(len(loss_breakdown)):
                    avg_loss_breakdown[i] = 0.9 * avg_loss_breakdown[i] + 0.1 * loss_breakdown[i]

            if batch_id % 10 == 1:
                writer.add_scalar('train/loss',avg_loss, w_id)
                if avg_loss_breakdown is not None:
                    for i, l in enumerate(avg_loss_breakdown):
                        writer.add_scalar('train/loss_{}'.format(i), avg_loss_breakdown[i], w_id)
                currennt_op = optimizer[0][0] if isinstance(optimizer, list) else optimizer
                writer.add_scalar('train/lr',currennt_op.param_groups[0]['lr'], w_id)

                if 'report' in loss_result:
                    for key, value in loss_result['report'].items():
                        writer.add_scalar('z_report/' + key, value, w_id)

                # TODO: Remove this later
                if hasattr(net, 'reporting_hooks'):
                    handle_reporting_hooks(net, writer, w_id)

            if teacher_net is not None:
                if total_loss is None:
                    loss = teacher_loss_ratio * teacher_loss + (1 - teacher_loss_ratio) * loss
                else:
                    loss = total_loss

            t3=time()

            loss.backward()
            if isinstance(optimizer, list):
                for op,_ in optimizer:
                    op.step()
            else:
                optimizer.step()

            t4=time()

            times_string = ""
            if True:
                times_string = ' [timing (ms) f/l/e/b {:.2f} {:.2f} {:.2f} {:.2f}]'.format((t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t4-t3)*1000)

            print('{}{}: e,b,l,avg-l: {}, {}: {:.4f}, {:.4f}{}'.format(
                dtm.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], times_string, epoch, batch_id, loss.item(), avg_loss,
                avg_teacher_loss if teacher_net is not None else ''))

        eval_score = evaluator.eval()
        if isinstance(net, ResNeXt) and False:
            net.set_gumble_mode(hard=True)
            eval_score_hard = evaluator.eval()
            writer.add_scalar('eval/score_gumble_hard', eval_score_hard, epoch)
            net.set_gumble_mode(hard=False)
        # path_to_save = os.path.join(backup_folder, 'net_e_{}_score_{}.pt'.format(epoch, '{:.4f}'.format(eval_score).replace('.','_')))
        path_to_save = os.path.join(backup_folder, 'net_backup.pt')
        print('\nEvaluation score:', eval_score, ' Saving to path: ', path_to_save,'\n')
        torch.save(net.state_dict(), path_to_save)
        writer.add_scalar('eval/score', eval_score, epoch)
        writer.add_scalar('train/score', corrects/(train_loader.__len__() * batch_size), epoch)
        for i, auxiliary_correct in enumerate(auxiliary_corrects):
            writer.add_scalar('z_train/correct_{}'.format(i), auxiliary_correct/(train_loader.__len__() * batch_size), epoch)
            writer.add_scalar('z_train/confident_correct_{}'.format(i), auxiliary_confident_corrects[i]/max(auxiliary_confident_count[i], 1), epoch)


if __name__=='__main__':
    # raise ValueError('Change the square of student correctness but did not run it yet')
    args = setup()
    net, criterion, batch_size, backup_folder, train_loader, test_loader, evaluator, epochs_to_run, lr_schedule, \
    auxiliary_net, auxiliary_loader, teacher_net, teacher_loss_ratio, run_eval = parse_args(args)

    net.T_l = None
    if False:
        net.T_l = (1 + torch.randn(10)*0.2).requires_grad_()

    net.T_fc=None
    if False:
        net.T_fc = TemperatureFC(2, include_target=True, include_teacher=True, include_student=True).cuda()

    # learning_rate = 1e-2
    learning_rate = 0.01
    momentum = 0.9
    # decay = 0.0005
    decay = 0.0

    if net.T_l is not None:
        optimizer_parameters = list(net.parameters()) + [net.T_l]
    elif net.T_fc is not None:
        optimizer_parameters = list(net.parameters()) + list(net.T_fc.parameters())
    else:
        optimizer_parameters = filter(lambda p: p.requires_grad, net.parameters())

    # optimizer = optim.SGD(optimizer_parameters, lr=learning_rate / batch_size, momentum=momentum, dampening=0,
    #                       weight_decay=decay * batch_size)
    optimizer = optim.SGD(optimizer_parameters, lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)

    if hasattr(net, 'net') and hasattr(net.net, 'optimizers_mapping'):
        optimizer = []
        for weight, optimizer_parameters in net.net.optimizers_mapping:
            optimizer.append((optim.SGD(optimizer_parameters, lr=learning_rate * weight, momentum=momentum, dampening=0, weight_decay=decay), weight))

    # step_size = 100 * train_loader.dataset.__len__()//batch_size
    # print(step_size)
    # step_size = 2
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # T = 5
    teacher_loss_func = None
    # teacher_loss_func = partial(KLD_loss, T=0.5)
    # teacher_loss_func = lambda student_logits, teacher_logits, gt : (KLD_loss(student_logits, teacher_logits, get_temperature(student_logits, teacher_logits,gt, T)),get_temperature(student_logits, teacher_logits,gt, T))

    T=0.2

    avg=None
    avg_c=None
    if False:
        avg, nums = calc_net_prediction(teacher_net, train_loader, T=T, correct_net=False)
        print('Without filtering when teacher is wrong')
        for i in range(avg.size(0)):
            print("{} : {} \n".format(nums[i].item(), ",".join(["{:.4f}".format(a) for a in avg[i, :].numpy()])))

        criterion = AverageSoftLoss(avg, T)

        # avg_c, nums_c = calc_net_prediction(teacher_net, train_loader, T=1.2, correct_net=True)
        # print('\n\nWith filtering when teacher is wrong')
        # for i in range(avg_c.size(0)):
        #     print("{} : {} \n".format(nums_c[i].item(), ",".join(["{:.4f}".format(a) for a in  avg_c[i, :].numpy()])))

        # exit(0)

    teacher_net = None

    if run_eval:
        if isinstance(net, ResNeXt):
            net.set_gumble_mode(hard=True)
        print('Eval score for net: {}'.format(evaluator.eval()))
        exit(0)

    train(net, backup_folder, train_loader, epochs_to_run=epochs_to_run, evaluator=evaluator, loss_func=criterion,
          optimizer=optimizer, lr_schedule=lr_schedule, auxiliary_net=auxiliary_net, auxiliary_loader=auxiliary_loader,
          teacher_net=teacher_net, teacher_loss_func=teacher_loss_func, teacher_loss_ratio=1.0, avg_net_pred=avg, T=T)
