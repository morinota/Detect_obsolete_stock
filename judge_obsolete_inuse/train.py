
from email.mime import image
from turtle import back
from typing import List
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import models
from config import Config
import time
import copy
from tqdm import tqdm


def train_model(model: models.ResNet, train_dataloader: DataLoader):
    """modelオブジェクトとDataLoaderオブジェクトを渡して、モデルの学習を行う.

    Parameters
    ----------
    model : models.ResNet
        _description_
    train_dataloader : DataLoader
        _description_
    """
    # 演算を行うデバイスを設定
    # GPUを使えるかどうかに応じてtorch.deviceオブジェクトを生成
    # torch.deviceはテンソルをどのデバイスに割り当てるかを表すクラス。
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    # modelをGPUに渡す
    model.to(device)

    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()
    # 最適化手法の設定
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_ft = optim.SGD(params=params,
                             lr=Config.learning_rate_resnet,
                             momentum=Config.momentum_resnet
                             )
    # 7エポックごとにLRを0.1倍ずつ減衰させる。
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer_ft, step_size=7, gamma=0.1
    )
    # GPUのキャッシュクリア
    torch.cuda.empty_cache()

    since = time.time()
    best_model_wt = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 25
    # 学習
    for epoch in range(num_epochs):
        print(f'Epoch {epoch} / {num_epochs-1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model.train()  # 学習モードに移行
            else:
                model.eval()  # 推論モードに移行

            running_loss = 0.0  # 損失関数の値
            running_corrects = 0  # 正解率?

            # 1バッチずつ学習させていく.
            for i, batch in enumerate(tqdm(train_dataloader)):

                # batchにはそのミニバッチのimages, labelsが入っている。
                images: List[Tensor]
                labels: List[Tensor]
                images, labels = batch

                # 指定のdevice(=GPU)にTensorを転送する(ListやDictにTensorが入ってるから)
                images = list(image.to(device) for image in images)
                labels = list(label.to(device) for label in labels)

                # zero the parameter gradients
                optimizer_ft.zero_grad()  # 前のバッチで計算されたgradをリセット
                # 誤差関数の値を取得?
                loss_dict = model(images, labels)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    # tensor(max, max_indices)なのでpredは0,1のラベル
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()  # 誤差逆伝搬で、各パラメータの勾配gradの値を計算(実はgradは累算してる!だからzero_gradを使ってる)
                        optimizer_ft.step()  # - grad＊学習率を使って、パラメータの値を更新

                # statistics (1バッチ毎の指標の計算)
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
