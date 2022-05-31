from my_dataset import Dataset_image_recognition
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from config import Config
from data_augumentation import conduct_offline_data_augmentation, delete_data_augmentated_files
from model import create_model
from train import train_model
from valid import conduct_visualize_validation
import matplotlib.pyplot as plt
import numpy as np


def imshow(img: torch.Tensor):
    img = img / 2 + 0.5
    nping = img.numpy()
    plt.imshow(np.transpose(nping, (1, 2, 0)))
    plt.show()


def main():
    image_dir = Config.image_dir_inuse_obsolete

    # data augmentation
    conduct_offline_data_augmentation()

    # Transform を作成する。
    transform_train = transforms.Compose(
        [transforms.Resize(size=(256, 256)), transforms.ToTensor()])
    # create dataset for learning
    dataset = Dataset_image_recognition(
        root=image_dir, transform=transform_train
    )

    # create dataloader for learning
    dataloader = DataLoader(dataset=dataset, batch_size=2)

    # 訓練データをランダムに取得し、画像として表示
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % dataset.classes[labels[j]]
          for j in range(4)))  # ラベルの表示

    # # create model object
    # model = create_model()

    # # train model
    # model = train_model(model=model, train_dataloader=dataloader,
    #                     train_dataset=dataset)

    # # visualize validation estimation
    # conduct_visualize_validation(model)

    # delete_data_augmentated_files(image_dir)  # 最後にまた、オリジナル画像のみにしておく


if __name__ == '__main__':
    main()
