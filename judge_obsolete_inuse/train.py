
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


def train_model(model: models.ResNet, train_dataloader: DataLoader, train_dataset: Dataset):
    """modelオブジェクトとDataLoaderオブジェクトを渡して、モデルの学習を行う.

    Parameters
    ----------
    model : models.ResNet
        _description_
    train_dataloader : DataLoader
        _description_
    """
    print('len of train data is {}'.format(train_dataset.__len__()))
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

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 25

    since = time.time()

    # 学習
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in train_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # tensor(max, max_indices)なのでpredは0,1のラベル
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()  # 誤差逆伝搬で、各パラメータの勾配gradの値を計算(実は累算してる!だからzero_gradを使ってる)
                        optimizer_ft.step()  # - grad＊学習率を使って、パラメータを更新

        print(f'epoch {epoch}/{num_epochs} : loss {loss}')

    time_elapsed = time.time() - since

    # save weight
    model_path = 'model.pth'
    # model.state_dict()として保存した方が無駄な情報を削れてファイルサイズを小さくできるらしい.
    torch.save(model.state_dict(), model_path)

    return model
