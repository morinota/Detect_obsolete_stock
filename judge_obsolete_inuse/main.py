from judge_obsolete_inuse.data_augumentation import delete_data_augmentated_files
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


def main():
    image_dir = Config.image_dir_inuse_obsolete

    # data augmentation
    delete_data_augmentated_files(image_dir)  # 最初にオリジナル画像のみにしておく
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
    # create model object
    model = create_model()

    # train model
    model = train_model(model=model, train_dataloader=dataloader,
                        train_dataset=dataset)

    # visualize validation estimation
    conduct_visualize_validation(model)


if __name__ == '__main__':
    main()
