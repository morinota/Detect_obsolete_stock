
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2
from config import Config
from glob import glob
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
import albumentations as albu  # Data Augmentation用
from PIL import Image
import re


class Dataset_augmentation(Dataset):
    def __init__(self, root: str, transform=None) -> None:
        self.root = root

        self.classes, self.class_to_idx = self._find_classes(root)
        self.samples = self._make_dataset()
        self.targets = [s[1] for s in self.samples]
        self.transform = transform

    def _find_classes(self, img_dir: str) -> Tuple[List, Dict]:
        """Finds the class folders in a dataset.

        Parameters
        ----------
        img_dir : str
            Root directory path.

        Returns
        -------
        _type_
            _description_
        """
        classes = [d.name for d in os.scandir(
            img_dir) if d.is_dir()]  # 各クラスのdirのパスを取得
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得するinnor関数
        """
        images = []
        for class_name in sorted(self.class_to_idx.keys()):
            class_dir = os.path.join(self.root, class_name)
            for root, _, fnames in sorted(os.walk(class_dir, followlinks=True)):
                for fname in sorted(fnames):

                    # オリジナルの画像だけ取ってくる。
                    if re.compile(pattern='[0-9]+\.(jpg|JPG)').search(fname):
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_idx[class_name])
                        images.append(item)

        return images

    def __getitem__(self, index):
        """index のサンプルが要求されたときに返す処理を実装

        Parameters
        ----------
        index : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        path, target = self.samples[index]
        # 入力側の画像データを配列で読み込み
        image: Image.Image = Image.open(path)
        image = image.convert(mode='RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, target  # 返値

    def __len__(self):
        return len(self.samples)


def save_augumentated_images(dataset_augmentated: Dataset_augmentation, augumentated_type: str = 'horizontal_frip'):
    idx_to_class = {idx: label for label,
                    idx in dataset_augmentated.class_to_idx.items()}

    # 保存
    for i in range(dataset_augmentated.__len__()):
        # filepath生成
        base_path = dataset_augmentated.samples[i][0]
        base_file_name = os.path.basename(base_path)
        base_dir_name = os.path.dirname(base_path)
        augumented_file_name = base_file_name.split(
            '.')[0] + f'_{augumentated_type}.jpg'

        file_path_augumented = os.path.join(
            base_dir_name, augumented_file_name)

        # augumentated した画像を保存
        image_object: Image.Image = dataset_augmentated[i][0]
        image_object.save(fp=file_path_augumented)


def get_transform_for_data_augmentation(augumentated_type='horizontal_frip'):
    transforms_data_aug = []
    if augumentated_type == 'horizontal_frip':
        transforms_data_aug.append(
            transforms.RandomHorizontalFlip(p=1.0)
        )
    if augumentated_type == 'random_rotation':
        transforms_data_aug.append(
            transforms.RandomRotation(degrees=[-15, 15])
        )
    if augumentated_type == 'random_erasing':
        transforms_data_aug.append(
            transforms.RandomErasing(p=1)
        )
    if augumentated_type == 'random_perspective':
        transforms_data_aug.append(
            transforms.RandomPerspective(distortion_scale=0.5,
                                         p=1.0, interpolation=3)
        )
    if augumentated_type == 'random_resized_crop':
        transforms_data_aug.append(
            transforms.RandomResizedCrop(
                size=150, scale=(0.08, 1.0), ratio=(3/4, 4/3)
            )
        )
    if augumentated_type == 'color_jitter':
        transforms_data_aug.append(
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5
            )
        )

    if augumentated_type == 'random_affine':
        transforms_data_aug.append(
            transforms.RandomAffine(
                degrees=[-10, 10], translate=(0.1, 0.1), scale=(0.5, 1.5)
            )
        )

    return transforms.Compose(transforms=transforms_data_aug)


def delete_data_augmentated_files(dataset_augmentated: Dataset_augmentation):

    def _delete_file(path: str):
        os.remove(path)
    # pathを一つずつ見ていって、オリジナル以外を削除
    for class_name in sorted(dataset_augmentated.class_to_idx.keys()):
        class_dir = os.path.join(dataset_augmentated.root, class_name)
        for root, _, fnames in sorted(os.walk(class_dir, followlinks=True)):
            for fname in sorted(fnames):
                # オリジナルの画像だけは残す
                if re.compile(pattern='[0-9]+\.(jpg|JPG)').search(fname):
                    pass
                else:  # data augmentated した画像は削除
                    file_path = os.path.join(root, fname)
                    _delete_file(path=file_path)


def conduct_offline_data_augmentation(N=100):
    image_dir = Config.image_dir_inuse_obsolete

    # 元の画像に対して、Data Augumentationを実施していく

    augumentated_type_list = [
        'horizontal_frip',
        'random_rotation',  # ランダムに回転を行う
        # 'random_erasing',
        'random_affine',  # ランダムにアフィン変換を行う。
        'random_perspective',  # ランダムに射影変換を行う.
        'color_jitter',  # ランダムに明るさ、コントラスト、彩度、色相を変化させる.
        'random_resized_crop',  # ランダムに切り抜いた後にリサイズを行う.
    ]
    for data_augumentated_type in augumentated_type_list:
        for i in range(N):
            data_transform = get_transform_for_data_augmentation(
                augumentated_type=data_augumentated_type
            )
            dataset_augmentated = Dataset_augmentation(
                root=image_dir, transform=data_transform)
            print(dataset_augmentated.__len__())
            # plt.imshow(dataset_augmentated[1][0])
            # plt.show()

            save_augumentated_images(
                dataset_augmentated, augumentated_type=(str(i) + data_augumentated_type))
        print(f'finish {data_augumentated_type}')
        print('='*20)

    # delete_data_augmentated_files(dataset_augmentated=dataset_augmentated)


if __name__ == '__main__':
    conduct_offline_data_augmentation()
