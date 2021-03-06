import argparse
from collections import OrderedDict
from datetime import datetime as dtm
from functools import partial
import json
import os
import random
import shutil
import time
import warnings


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tensorboardX import SummaryWriter

from data.dataset_factory import get_train_test_datasets
from external_models.dcp.pruned_resnet import PrunedResnet30, PrunedResnet50, PrunedResnet70
from models.cifar_resnet import *
from models.custom_resnet import custom_resnet_50, custom_resnet_56
from models.vgg_fully_convolutional import *
from models.wrapped_gated_models import *
from utils.multi_optimizer import MultiGroupDynamicLROptimizer
from utils.save_warpper import save_version_aware


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset_name', type=str, default='imagenet',
                    help='name of dataset')
parser.add_argument("--train_data_path", type=str, action="store", default=None,
                    help='separate train path (for non imagenet)')
parser.add_argument("--val_data_path", type=str, action="store", default=None,
                    help='separate test path (for non imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-drop-rate', default=None, type=float, help='learning rate drop multiplier',
                    dest='lr_drop_rate')
parser.add_argument('--epoch-lr-step', default=None, type=float, help='num of epochs before learning rate drop',
                    dest='epoch_lr_step')
parser.add_argument('--lr_steps', type=str,
                    help='epochs at which to drop the learning rate')
parser.add_argument('--lr_rates', type=str,
                    help='rates to use before and after each step')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--backup_folder', default='', type=str,
                    help='Folder where checkpoints and tensorboard log are stored')
parser.add_argument('--custom_model', default=None, type=str,
                    help='Custom model to load instead of standard torchvision models')
parser.add_argument("--net_with_criterion", default=None, type=str,
                    help='Custom model with custom criterion (for auxiliary losses)')
parser.add_argument('--subdivision', default=1, type=int,
                    help='When > 1 enables running larger batches via gradient accumulation (i.e. how many batches to to '
                         'accumulate gradient and update the optimizer')
parser.add_argument('--create_old_format', action='store_true', default=False,
                    help='convert net in new save format to old format')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help = 'use nesterov in SGD')
parser.add_argument('--gating_config_path', type=str, default=None,
                    help = 'path to configuration file containing training parameters')
parser.add_argument('--save_interval', type=int, default=1,
                    help = 'number of epochs after which to save')


best_acc1 = 0


DATASET_NUM_CLASSES = {'imagenet':1000, 'cifar10':10, 'cifar100':100}


def get_actual_model(model):
    """
    Returns the inner model in case of parallel wrapping
    :param model:
    :return:
    """
    actual_model = model.module if (isinstance(model, torch.nn.DataParallel) or
                                  isinstance(model, torch.nn.parallel.DistributedDataParallel)) else model
    return actual_model


def main():
    args = parser.parse_args()
    if args.gating_config_path is not None:
        with open(args.gating_config_path) as f:
            setattr(args, "gating_config", json.load(f))

    if args.backup_folder != '' and not os.path.exists(args.backup_folder):
        os.mkdir(args.backup_folder)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    writer_folder = os.path.join(args.backup_folder, 'log')
    if not os.path.exists(writer_folder):
        os.mkdir(writer_folder)
    writer = SummaryWriter(writer_folder)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    criterion, param_groups_lr_adjustment_map = None, None
    num_classes = DATASET_NUM_CLASSES[args.dataset_name]
    if args.custom_model:
        print("=> creating custom model '{}'".format(args.custom_model))
        if 'CustomResNet' in args.custom_model:
            assert args.resume is not None
            channels_config = torch.load(args.resume)['channels_config']
            model_name = 'custom_resnet_' + args.custom_model[-2:]
            model = globals()[model_name](channels_config, num_classes)
        elif 'vgg' in args.custom_model.lower():
            model = VGGFullyConv(args.custom_model, num_classes)
        elif 'resnet56' in args.custom_model or 'PrunedResnet' in args.custom_model:
            model = globals()[args.custom_model](num_classes)
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True, num_classes=num_classes)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch](num_classes=num_classes)
    if args.net_with_criterion:
        print("=> creating custom model with criterion'{}'".format(args.net_with_criterion))
        model, criterion, param_groups_lr_adjustment_map = \
            globals()[args.net_with_criterion](model, getattr(args, 'gating_config', {}))

    if args.subdivision > 1:
        assert args.batch_size % args.subdivision == 0
        args.batch_size = args.batch_size // args.subdivision

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if args.lr_steps is not None:
        assert args.lr_rates is not None
        args.lr_steps = [int(a) for a in args.lr_steps.split(',')]
        args.lr_rates = [float(a) for a in args.lr_rates.split(',')]
        assert len(args.lr_steps) + 1 == len(args.lr_rates)

    # define loss function (criterion) and optimizer
    if criterion is None:  # if it wan't already defined using the net_with_criterion option
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(args.gpu)

    if param_groups_lr_adjustment_map is not None:
        # In case we have different learning rates for different parameters, create the optimizer for the first group
        # which is most likely the main parameters, and add the other groups according to the configuration.
        inner_optimizer = torch.optim.SGD(param_groups_lr_adjustment_map[0], args.lr, momentum=args.momentum,
                                          weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer = MultiGroupDynamicLROptimizer(inner_optimizer, param_groups_lr_adjustment_map[1])
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                    nesterov=args.nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch']
            if best_acc1 in checkpoint:
                best_acc1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(args.gpu)
            if isinstance(checkpoint, OrderedDict):
                checkpoint = {'module.' + k: v for k,v in checkpoint.items()}
                model.load_state_dict(checkpoint)
                warnings.warn('Loading al stated dict, no other metadata in checkpoint !!!')
            else:
                state_dict = checkpoint['state_dict']
                if args.gpu is not None and len(state_dict) == len([k for k in state_dict.keys() if k[:7] == 'module.']):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                # state_dict = {k.replace('module.net', 'module'): v for k, v in state_dict.items()}
                if args.net_with_criterion is not None and type(get_actual_model(model).net) in [CifarResnet, ResNet]:
                    state_dict = {k.replace('module', 'module.net'): v for k, v in state_dict.items()}
                    problematic_keys = model.load_state_dict(state_dict, strict=False)
                    # sanity checks because strict was disabled
                    assert len(problematic_keys.unexpected_keys) == 0
                    assert all('forward_hooks' in p for p in problematic_keys.missing_keys)
                else:
                    model.load_state_dict(state_dict)


            if 'optimizer' in checkpoint and not (args.net_with_criterion is not None
                                                  and type(get_actual_model(model).net) in [CifarResnet, ResNet]):
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch'] if 'epoch' in checkpoint else 0))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    if args.create_old_format:
        save_version_aware(checkpoint, filename=args.resume + 'old', old_format=True)
        exit(0)

    cudnn.benchmark = True

    is_train = not args.evaluate

    train_dataset, val_dataset, train_sampler = get_train_test_datasets(args, is_train=is_train)

    if is_train:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None), num_workers=args.workers,
                                                   pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    if not is_train:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.lr_steps is None:
            lr_batch_adjuster = adjust_learning_rate_and_get_batch_adjuster(optimizer, epoch, args)
        else:
            lr_batch_adjuster = adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer, lr_batch_adjuster)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, writer, epoch)
        if isinstance(acc1, torch.Tensor):
            acc1=acc1.item()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        architecture = args.custom_model if args.custom_model is not None else args.arch

        if is_best:
            best_file_name = os.path.join(args.backup_folder, 'net_best')
            save_checkpoint(model, epoch, architecture, best_acc1, optimizer, best_file_name)

        if (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0)) and \
                ((((epoch + 1) % args.save_interval) == 0) or ((epoch + 1) == args.epochs)):
            file_name = os.path.join(args.backup_folder, 'net_e_{}'.format(epoch + 1))
            save_checkpoint(model, epoch, architecture, best_acc1, optimizer, file_name)


def train(train_loader, model, criterion, optimizer, epoch, args, writer=None, lr_batch_adjuster=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if lr_batch_adjuster is not None:
            lr_batch_adjuster(optimizer, i/len(train_loader))
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss_result = criterion(output, target)

        if isinstance(output, tuple):  # in case of a net with auxiliary outputs
            output = output[0]
        loss = loss_result['loss'] if isinstance(loss_result, dict) else loss_result
        loss = loss / args.subdivision

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        if i % args.subdivision == 0:
            optimizer.zero_grad()
        loss.backward()
        if (i - 1) % args.subdivision == 0:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if writer is not None and i % 100 == 0:
            file_name = os.path.join(args.backup_folder, 'net_backup')
            architecture = args.custom_model if args.custom_model is not None else args.arch
            save_checkpoint(model, epoch, architecture, best_acc1, optimizer, file_name)

            batch_num = i + epoch * len(train_loader)
            writer.add_scalar('train/loss', losses.val, batch_num)
            writer.add_scalar('train/avg_loss', losses.avg, batch_num)
            writer.add_scalar('train/top1_avg', top1.avg, batch_num)
            writer.add_scalar('train/top5_avg', top5.avg, batch_num)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('train/lr', current_lr, batch_num)

            if isinstance(loss_result, dict) and 'report' in loss_result:
                for key, value in loss_result['report'].items():
                    writer.add_scalar('z_report/' + key, value, batch_num)


def validate(val_loader, model, criterion, args, writer=None, epoch=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            if isinstance(output, tuple):  # in case of a net with auxiliary outputs
                output = output[0]
            if isinstance(loss, dict):
                loss = loss['loss']

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        if writer is not None:
            writer.add_scalar('eval/top1', top1.avg, epoch + 1)
            writer.add_scalar('eval/top5', top5.avg, epoch + 1)

    return top1.avg


def save_checkpoint(model, epoch, architecture, best_acc1=0.0, optimizer=None,
                    filename='checkpoint.pth.tar', old_format=True):
    state_dict = {
        'epoch': epoch + 1,
        'arch': architecture,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1}
    if optimizer is not None:
        state_dict['optimizer']= optimizer.state_dict()
    if 'customresnet' in architecture.lower():
        # get parallel/non parallel model
        actual_model = get_actual_model(model)
        # get channels config when training custom resnet directly or wrapped
        state_dict['channels_config'] = actual_model.net.channels_config if hasattr(actual_model, 'net') \
            else actual_model.channels_config

    # check if version is 1.6.0 or above
    save_version_aware(state_dict, filename, old_format)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [dtm.now().strftime("%Y%m%d %H:%M:%S.%f ") + self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_lr_in_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch, args):
    assert len(args.lr_steps) + 1 == len(args.lr_rates)
    i = 0
    while i < len(args.lr_steps) and epoch > args.lr_steps[i]:
        i += 1
    adjust_lr_in_optimizer(optimizer, args.lr_rates[i])


def adjust_learning_rate_and_get_batch_adjuster(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    if args.lr_drop_rate is not None:
        assert args.epoch_lr_step is not None
        lr = args.lr * (args.lr_drop_rate ** (epoch // args.epoch_lr_step))
    adjust_lr_in_optimizer(optimizer, lr)
    lr_batch_adjuster = None
    # create batch adjuster if the step is not round
    if args.lr_drop_rate is not None and  args.epoch_lr_step - int(args.epoch_lr_step) > 1e-10:
        leftover_epoch_residual = (epoch + 1)  % args.epoch_lr_step
        if 1e-10 < leftover_epoch_residual < (1-1e-10):
            lr_batch_adjuster = lambda inner_optimizer, batch_percentage: \
                adjust_lr_in_optimizer(inner_optimizer, lr * args.lr_drop_rate) \
                    if batch_percentage > (1 - leftover_epoch_residual) else None
    # lr = args.lr * (0.1 ** (epoch // 30))
    # lr = 0.01 if epoch==0 else 0.001
    # warnings.warn('Manual Fiddling with learning rate')
    return lr_batch_adjuster


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
