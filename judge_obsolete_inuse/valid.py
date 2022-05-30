import matplotlib.pyplot as plt
from typing import List
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from config import Config
import time
import copy
from tqdm import tqdm
import numpy as np
from data_augumentation import Dataset_augmentation
from torchvision import models, transforms


def tensor_to_np(inp):
    "imshow for Tensor"
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_model(model, valid_dataloader: DataLoader, valid_dataset: Dataset_augmentation, num_images=6):

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    class_names = {idx: label for label,
                   idx in valid_dataset.class_to_idx.items()}

    for inputs, labels in valid_dataloader:
        print(labels)
        outputs = model(inputs) # 訓練後のNNに画像を入力
        _, predicted = torch.max(outputs, 1) # #入力した画像の行列の最大値（もっとも確率の高いもの）を返す
        print(outputs)
        print('='*20)

        


def conduct_visualize_validation(model:models.ResNet):
    image_dir = Config.image_dir_inuse_obsolete
    transform_valid = transforms.Compose(
        [transforms.Resize(size=(256, 256)), transforms.ToTensor()])
    dataset_valid = Dataset_augmentation(
        root=image_dir, transform=transform_valid)
    dataloader_valid = DataLoader(dataset=dataset_valid, batch_size=1)
    visualize_model(model, dataloader_valid, dataset_valid, num_images=4)

    