"""MNIST 数据加载模块。

关于数据来源：LeCun 官网 (yann.lecun.com/exdb/mnist) 的原始链接已失效，
本项目改用 torchvision 自带的 MNIST 下载器，它从稳定的镜像
(https://ossci-datasets.s3.amazonaws.com/mnist) 获取数据，无需手动下载。

LeNet-5 原论文使用 32x32 的输入，而 MNIST 原始图像为 28x28，
因此这里对图像做 padding=2 补边到 32x32，以忠实复刻原始结构。
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# MNIST 全数据集的均值和标准差（标准常用值），用于归一化
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def build_transform():
    """构建图像预处理：补边到 32x32 -> 转张量 -> 归一化。"""
    return transforms.Compose([
        transforms.Pad(2),  # 28x28 -> 32x32，复刻 LeNet-5 原始输入尺寸
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


def get_dataloaders(root="data", batch_size=128, num_workers=2):
    """返回训练集和测试集的 DataLoader。

    首次运行会自动下载数据集（约 11MB）。
    """
    tf = build_transform()
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def get_dataloaders_subset(root="data", batch_size=128, num_workers=2,
                           subset_size=1000, seed=42):
    """返回“小训练集 + 完整测试集”的 DataLoader，用于制造过拟合场景。

    反事实实验：训练集砍到 subset_size 张（默认 1000），模型容易过拟合，
    从而可以观察 dropout 的正则化收益（预期 dropout 反超原版）。
    测试集仍用完整 10000 张，保证评估可靠。
    """
    tf = build_transform()
    full_train = datasets.MNIST(root=root, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tf)

    # 固定种子选取子集，保证可复现
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_train), generator=g)[:subset_size]
    train_ds = Subset(full_train, perm.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
