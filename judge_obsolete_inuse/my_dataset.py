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


class MyDataset(Dataset):
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(self, img_dir: str, transform=None) -> None:
        self.img_paths = self._get_img_paths(img_dir)
        print(self.img_paths)
        self.transform = transform

    def __getitem__(self, index):
        # index のサンプルが要求されたときに返す処理を実装

        path = self.img_paths[index]
        # 入力側の画像データを配列で読み込み
        image = cv2.imread(filename=path, flags=cv2.IMREAD_COLOR)
        # BGRからRGBに配列の順序を変換?
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 正規化?
        image /= 255.0
        if self.transform is not None:
            image = self.transform(image)

        # クラスラベルの指定
        label = path.split('/')[-2]

        # 返値
        return image, label

    def __len__(self):
        return len(self.img_paths)

    def _get_img_paths(self, img_dir):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得するinnor関数
        """
        class_dirs = glob(pathname=img_dir+'/*')  # 各クラスのdirのパスを取得
        img_paths = []
        for class_dir in class_dirs:
            for img_path in glob(os.path.join(class_dir, '*.jpg')):
                img_paths.append(img_path)

        return img_paths


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
