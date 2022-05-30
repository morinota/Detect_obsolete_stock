import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
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


class Dataset_image_recognition(Dataset):
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


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    image_dir = Config.image_dir_inuse_obsolete
    mean = (0.5,)
    std = (0.5,)
    # Transform を作成する。
    transform = transforms.Compose(
        [transforms.Resize(256), transforms.ToTensor()])
    dataset = ImageFolder(
        root=image_dir,
        transform=transform,
        target_transform=None,
    )
    print(dataset.class_to_idx)

    # ImageFolderで取り込んだイメージ画像のデータを使用して、データローダーを作成
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    for X_batch, y_batch in dataloader:
        print(X_batch.shape, y_batch.shape)
        # 画像の表示
        imshow(torchvision.utils.make_grid(X_batch))
        print(y_batch)
