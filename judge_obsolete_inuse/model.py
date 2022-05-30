
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from config import Config


def create_model() -> models.ResNet:
    # 学習済みモデルを読み込み(転移学習)
    model_ft: models.ResNet
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    num_classes = 2
    # 出力次元数を書き換え
    model_ft.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)

    return model_ft


if __name__ == '__main__':
    model_ft = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(params=model_ft.parameters(),
                             lr=Config.learning_rate_resnet,
                             momentum=Config.momentum_resnet
                             )
    # 7エポックごとにLRを0.1倍ずつ減衰させる。
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer_ft, step_size=7, gamma=0.1
    )

    for x in list(model_ft.children()):
        print(x, '\n')
