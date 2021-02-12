from functools import partial
import os

import torch
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms.functional import hflip

from data.image import Rescale, normalize
from .dataset import CIFARDataset, PetsDataset, FolderPerClassFilePathDataset

base_data_transform = transforms.Compose([
    Rescale((224, 224)),
    transforms.ToTensor(),
    normalize
])

train_data_transform = transforms.Compose([
    Rescale((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_data_transform = transforms.Compose([
    Rescale((224, 224)),
    transforms.ToTensor()
])

stl_train_data_transform = transforms.Compose([
    transforms.RandomCrop(88),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

stl_test_data_transform= transforms.Compose([
    transforms.ToTensor(),
    normalize
])

cifar_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

imagenet_transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

imagenet_transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

def get_train_test_datasets(args, is_train=True):
    train_dataset = None
    if args.dataset_name == 'imagenet':
        if is_train:
            train_dataset = ImageFolder(args.train_data_path, imagenet_transforms_train)
        test_dataset = ImageFolder(args.val_data_path, imagenet_transforms_test)
    elif args.dataset_name[:5] =='cifar':
        cifar_type = int(args.dataset_name[5:])
        if is_train:
            train_dataset = CIFARDataset(args.train_data_path, cifar_transform_train, True, cifar_type)
        test_dataset = CIFARDataset(args.val_data_path, cifar_transform_test, False, cifar_type)
    elif args.dataset_name =='pets':
        if is_train:
            train_dataset = PetsDataset(args.train_data_path, train_data_transform, True)
        test_dataset = PetsDataset(args.val_data_path, test_data_transform, False)
    elif args.dataset_name == 'food':
        if is_train:
            train_dataset = FolderPerClassFilePathDataset(args.train_data_path, train_data_transform, True)
        test_dataset = FolderPerClassFilePathDataset(args.val_data_path, test_data_transform, False)
    elif args.dataset_name == 'stl10':
        if is_train:
            train_dataset = FolderPerClassFilePathDataset(args.train_data_path, stl_train_data_transform, True)
        test_dataset = FolderPerClassFilePathDataset(args.val_data_path, stl_test_data_transform, False)
    else:
        raise ValueError('Unknown dataset_name')

    if is_train and args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    return train_dataset, test_dataset, train_sampler


TEN_CROP_SHAPE_MAP = {
    'cifar10': ((30, 30), None),
    'cifar100': ((30, 30), None),
    'pets': NotImplementedError(),
    'stl10': NotImplementedError(),
    'food': NotImplementedError(),
}


def ten_crop_transform(dataset_transformst, dataset_name):
    ten_crop_shape, rescale_shape = TEN_CROP_SHAPE_MAP[dataset_name]
    if rescale_shape is not None:
        trans = transforms.Compose([
            Rescale(rescale_shape),
            transforms.TenCrop(ten_crop_shape),
            lambda crops: torch.stack([dataset_transformst(crop) for crop in crops])
        ])
    else:
        trans = transforms.Compose([
            transforms.TenCrop(ten_crop_shape),
            lambda crops: torch.stack([dataset_transformst(crop) for crop in crops])
        ])
    return trans


def flip_no_flip_transform(dataset_transformst, dataset_name):
    return transforms.Compose([
        lambda image: (image, hflip(image)),
        lambda crops: torch.stack([dataset_transformst(crop) for crop in crops])
    ])


ENSEMBLE_TRANSFORMS = {
    'ten_crop': ten_crop_transform,
    'flip': flip_no_flip_transform
}


SAME_IMAGE_TRANSFORMS = {
    'ten_crop': NotImplementedError('ten crop needs to be implemented in torch'),
    'flip': partial(torch.flip, dims=(-1,))
}
