import os
from random import randint

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset

from utils.pickle_tools import unpickle


def create_paths_from_folder(folder):
    return [os.path.join(folder, file_name) for file_name in os.listdir(folder)]


class FolderDataset(Dataset):
    def __init__(self, folder, transform, return_third_dummy=False):
        self.folder = folder
        self.images_path = create_paths_from_folder(folder)
        # self.images_path = self.images_path[:500] + self.images_path[-500:]
        self.transform = transform
        self.return_third_dummy = return_third_dummy

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        image_path = self.images_path[item]
        img = Image.open(image_path)
        img = self.transform(img)
        if self.return_third_dummy:
            return image_path, img, 0
        return image_path, img


class FolderAugmentedDataset(FolderDataset):
    def __init__(self, folder, transforms_and_augmentations, num_results):
        super(FolderAugmentedDataset, self).__init__(folder, transforms_and_augmentations)
        self.transforms_and_augmentations = transforms_and_augmentations
        self.num_results = num_results

    def __getitem__(self, item):
        image_path = self.images_path[item]
        img = Image.open(image_path)
        result_images = []
        for i in range(self.num_results):
            result_images.append(self.transforms_and_augmentations(img).unsqueeze(0))
        result_images = torch.cat(result_images)
        return image_path, result_images


class FolderAugmentedTenCropDataset(FolderDataset):
    def __init__(self, folder, transforms_and_augmentations):
        super(FolderAugmentedTenCropDataset, self).__init__(folder, transforms_and_augmentations)
        self.transforms_and_augmentations = transforms_and_augmentations

    def __getitem__(self, item):
        image_path = self.images_path[item]
        img = Image.open(image_path)
        result_images=self.transforms_and_augmentations(img)
        return image_path, result_images


# Too lazy to create an enum for cifar10/100
CIFAR_FILE_PREFIXES = {10: ('data_batch', 'test_batch'), 100: ('train', 'test')}
CIFAR_LABEL_NAMES = {10: b'labels', 100: b'fine_labels'}


class CIFARDataset(Dataset):
    def __init__(self, folder, transform, is_train, cifar_type=10):
        assert cifar_type in [10, 100]
        self.is_train = is_train
        file_prefix = CIFAR_FILE_PREFIXES[cifar_type][0 if is_train else 1]
        files_path =  [os.path.join(folder, file_name) for file_name in os.listdir(folder) if file_prefix in file_name]
        images = []
        labels = []
        for file_path in files_path:
            d = unpickle(file_path)
            images.append(d[b'data'])
            labels.append(d[CIFAR_LABEL_NAMES[cifar_type]])
        self.images = np.concatenate(images).reshape(-1,3,32,32)
        self.labels = np.concatenate(labels)

        # self.images_path = self.images_path[:500] + self.images_path[-500:]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = self.transform(Image.fromarray(self.images[item].transpose(1,2,0),'RGB'))
        return img, self.labels[item]


class PetsDataset(Dataset):
    def __init__(self, folder, transform, is_train):
        self.transform = transform
        self.images_folder = os.path.join(folder, 'images')
        annotations_file_name = 'big_train.txt' if is_train else 'small_eval.txt'
        annotations_file = os.path.join(folder,'annotations',annotations_file_name)
        annotations = pd.read_csv(annotations_file, delimiter=' ', header=None, names = ['image','class','sub_class','sub_class_index'])
        annotations['class'] = annotations['class']-1
        self.image_names = annotations['image'].tolist()
        self.classes = annotations['class'].tolist()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        image_path = os.path.join(self.images_folder, self.image_names[item] + '.jpg')
        img = Image.open(image_path)
        img = self.transform(img)
        target = self.classes[item]
        return img, target


class FolderPerClassFilePathDataset(Dataset):
    def __init__(self, folder, transform, is_train):
        self.transform = transform
        self.images_folder = os.path.join(folder, 'images')
        annotations_file_name = 'train.txt' if is_train else 'test.txt'
        annotations_file = os.path.join(folder, annotations_file_name)
        annotations = pd.read_csv(annotations_file, delimiter=',', header=None, names = ['class','image'])
        annotations['class'] = annotations['class']
        self.image_names = annotations['image'].tolist()
        self.classes = annotations['class'].tolist()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        image_path = os.path.join(self.images_folder, str(self.classes[item]), self.image_names[item])
        img = Image.open(image_path)
        img = self.transform(img)
        target = self.classes[item] - 1
        return img, target


class SoftTargetDatasetWrapper(Dataset):
    def __init__(self, inner_dataset, soft_targets):
        assert isinstance(inner_dataset, Dataset)
        self.inner_dataset = inner_dataset
        self.soft_targets = soft_targets

    def __len__(self):
        return self.inner_dataset.__len__()

    def __getitem__(self, item):
        a, x, y = self.inner_dataset.__getitem__(item)
        new_y = self.soft_targets[y]
        return a, x, new_y


class TenCropRandomTransform:
    def __init__(self, w, h, crop_w, crop_h):
        assert isinstance(w, int) and isinstance(h, int) and isinstance(crop_w, int) and isinstance(crop_h, int)
        self.w = w
        self.h = h
        self.crop_w = crop_w
        self.crop_h = crop_h

        five_fn = [
            lambda img: img.crop((0, 0, crop_w, crop_h)),
            lambda img: img.crop((w - crop_w, 0, w, crop_h)),
            lambda img: img.crop((0, h - crop_h, crop_w, h)),
            lambda img: img.crop((w - crop_w, h - crop_h, w, h)),
            lambda img: img.crop(((w - crop_w) // 2, (h - crop_h) // 2, w // 2, h // 2))
        ]

        self.function_list = five_fn + [lambda img: f(img.transpose(Image.FLIP_LEFT_RIGHT)) for f in five_fn]

    def __call__(self):
        return self.function_list[randint(0,9)]
