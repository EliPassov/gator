import os
import tarfile
import pandas as pd
from shutil import copy


def extract_train():
    train_folder = '/home/victoria/Data/ILSVRC2012/ILSVRC2012_img_train'
    new_train_folder = '/home/victoria/Data/ILSVRC2012/train'

    for i, file_name in enumerate(os.listdir(train_folder)):
        file_path = os.path.join(train_folder, file_name)
        new_folder = os.path.join(new_train_folder, file_name[:-4])
        os.mkdir(new_folder)
        tr = tarfile.open(file_path)
        tr.extractall(new_folder)
        tr.close()
        # os.remove(file_path)
        print('done with: ', i, file_path)


def prepare_val():
    val_folder = '/media/victoria/d/data/ILSVRC2012/ILSVRC2012_img_val'
    new_val_folder = '/media/victoria/d/data/ILSVRC2012/val'
    mapping_file = '/media/victoria/d/data/ILSVRC2012/meta.csv'

    mapping = ['invalid_placeholder']
    with open(mapping_file, 'r') as f:
        mapping_list = f.readlines()
    for line in mapping_list:
        mapping.append(line.split(',')[1][1:-1])
    # mapping = ['invalid_placeholder'] + pd.read_csv(mapping_file, header=0)[1].to_list()
    gt_file = '/media/victoria/d/data/ILSVRC2012/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
    with open(gt_file, 'r') as f:
        gt = f.readlines()
    gt = [int(a) for a in gt]

    sorted_files = os.listdir(val_folder)
    sorted_files.sort()

    for i, file_name in enumerate(sorted_files):
        file_path = os.path.join(val_folder, file_name)
        category_index = gt[i]
        category_name = mapping[category_index]
        dest_folder = os.path.join(new_val_folder, category_name)
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)
        new_file_path = os.path.join(dest_folder, file_name)
        copy(file_path, new_file_path)
        if i%100 == 0 and i>0:
            print('Done with ', i, '/', len(sorted_files))


def test_tensorboard():
    from tensorboardX import SummaryWriter
    writer_path = '/media/victoria/d/models/Imagenet/test_tensorboard/log'
    writer = SummaryWriter(writer_path)
    for i in range(100):
        writer.add_scalar('train/square', i**2, i)
        writer.add_scalar('eval/sqrt', i**0.5, i)


if __name__ == '__main__':
    test_tensorboard()

