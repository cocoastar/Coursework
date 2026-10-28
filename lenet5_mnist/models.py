"""LeNet-5 网络结构定义。

包含三个任务用到的全部模型变体：
  - LeNet5          : 任务1，忠实复刻 LeCun(1998) 原始 LeNet-5
  - LeNet5Half      : 任务2，所有卷积层卷积核数量减半
  - LeNet5Dropout   : 任务3，在原版基础上增加 dropout 层

原版 LeNet-5 结构（输入 1x32x32）：
  C1: 卷积 6@5x5   -> 6@28x28
  S2: 平均池化 2x2 -> 6@14x14
  C3: 卷积 16@5x5  -> 16@10x10
  S4: 平均池化 2x2 -> 16@5x5
  C5: 卷积 120@5x5 -> 120@1x1   (等价于全连接)
  F6: 全连接 84
  输出: 全连接 10
激活函数使用 tanh（与原论文一致）。
"""
import torch.nn as nn


class LeNet5(nn.Module):
    """任务1：原版 LeNet-5。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),   # C1
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # S2
            nn.Conv2d(6, 16, kernel_size=5),  # C3
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # S4
            nn.Conv2d(16, 120, kernel_size=5),  # C5
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),  # F6
            nn.Tanh(),
            nn.Linear(84, num_classes),  # 输出层
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LeNet5Half(nn.Module):
    """任务2：所有卷积层卷积核数量减半。

    卷积核数量: 6->3, 16->8, 120->60。
    全连接层 F6 的 84 维输出保持不变（非卷积层，不属于“卷积核减半”范畴），
    仅其输入维度随 C5 通道数由 120 变为 60。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5),   # C1: 6->3
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3, 8, kernel_size=5),   # C3: 16->8
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 60, kernel_size=5),  # C5: 120->60
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(60, 84),  # F6 输入维度随 C5 变化
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LeNet5Dropout(nn.Module):
    """任务3：在原版 LeNet-5 基础上增加 dropout 层。

    在池化后的特征层以及全连接层 F6 之后插入 Dropout。
    dropout 概率 p 由外部传入（本任务分别测试 0.01 / 0.05 / 0.10）。
    """

    def __init__(self, num_classes=10, p=0.05):
        super().__init__()
        self.p = p
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(p),  # S2 之后
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(p),  # S4 之后
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Dropout(p),  # F6 之后
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    """统计模型可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
