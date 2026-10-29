"""数据合成模块：把 MNIST 数字贴到真实背景 patch 上，生成分割训练样本。

合成流程（每个样本）：
  1. 从 MNIST 随机取一个数字图（28x28 灰度），用灰度阈值抠出前景 mask；
  2. 从 20 张背景图中随机选一张，随机裁一个 patch 作为背景画布（CANVAS x CANVAS）；
  3. 对数字做随机缩放、随机平移，贴到背景上 -> 合成图；
  4. 同步生成分割 GT：
       - 2 分类 : 前景=1, 背景=0
       - N 分类 : 前景像素=数字类别+1 (1~10), 背景=0

为什么前景标签是 "数字类别+1" 而不是 "数字类别"？
  因为 0 要留给背景。于是 10 分类分割实际有 11 个输出通道：
  通道 0=背景，通道 1~10 分别对应数字 0~9。
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image

CANVAS = 64            # 合成图边长
FG_THRESH = 0.3        # 前景灰度阈值（归一化到 0~1 后，>阈值算数字笔画）


def load_background_patches(bg_dir, canvas=CANVAS, seed=0):
    """把 20 张背景图读进来，统一转成灰度 numpy 数组（0~1），供随机裁 patch。

    返回 list[np.ndarray]，每张是 (H, W) 的灰度图。
    """
    paths = sorted(glob.glob(os.path.join(bg_dir, "*.jpg")) +
                   glob.glob(os.path.join(bg_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"背景目录 {bg_dir} 下没有找到图片")
    bgs = []
    for p in paths:
        img = Image.open(p).convert("L")  # 转灰度
        arr = np.asarray(img, dtype=np.float32) / 255.0
        bgs.append(arr)
    return bgs


class MNISTSegDataset(Dataset):
    """合成的 MNIST 分割数据集。

    参数:
        mnist_root : MNIST 数据目录（复用作业1下载的或自动下载）
        bg_dir     : 20 张背景图所在目录
        num_classes: 2 (前景/背景) 或 11 (背景+数字0~9)
        length     : 本数据集“虚拟长度”，即一个 epoch 合成多少张（随机生成）
        train      : True 用 MNIST 训练集，False 用测试集
        seed       : 随机种子基准（保证可复现）
    """

    def __init__(self, mnist_root, bg_dir, num_classes=2, length=2000,
                 train=True, seed=42):
        self.mnist = datasets.MNIST(root=mnist_root, train=train, download=True)
        self.bgs = load_background_patches(bg_dir)
        self.num_classes = num_classes
        self.length = length
        self.train = train
        self.seed = seed

    def __len__(self):
        return self.length

    def _rng(self, idx):
        # 每个样本用确定性的随机种子 -> 整个数据集可复现
        return np.random.default_rng(self.seed + idx * 7919 + (0 if self.train else 10**6))

    def _crop_bg(self, rng):
        """从随机一张背景图随机裁出 CANVAS x CANVAS 的 patch。"""
        bg = self.bgs[rng.integers(len(self.bgs))]
        H, W = bg.shape
        if H < CANVAS or W < CANVAS:  # 背景比画布小则放大（极少见）
            bg = np.asarray(Image.fromarray((bg * 255).astype(np.uint8))
                            .resize((max(W, CANVAS), max(H, CANVAS))), np.float32) / 255.0
            H, W = bg.shape
        top = rng.integers(0, H - CANVAS + 1)
        left = rng.integers(0, W - CANVAS + 1)
        return bg[top:top + CANVAS, left:left + CANVAS].copy()

    def _get_digit(self, rng):
        """随机取一个 MNIST 数字，返回 (灰度数组 0~1, 标签)。"""
        i = int(rng.integers(len(self.mnist)))
        img, label = self.mnist[i]
        arr = np.asarray(img, dtype=np.float32) / 255.0  # 28x28
        return arr, label

    def __getitem__(self, idx):
        rng = self._rng(idx)

        # 1) 背景画布
        canvas = self._crop_bg(rng)  # (64,64) 0~1

        # 2) 取数字并随机缩放
        digit, label = self._get_digit(rng)
        scale = rng.uniform(0.8, 1.6)
        new_sz = max(10, min(CANVAS, int(28 * scale)))
        digit_img = Image.fromarray((digit * 255).astype(np.uint8)).resize((new_sz, new_sz))
        digit = np.asarray(digit_img, dtype=np.float32) / 255.0

        # 3) 随机位置粘贴
        max_top = CANVAS - new_sz
        top = int(rng.integers(0, max_top + 1))
        left = int(rng.integers(0, max_top + 1))

        # 前景 mask：灰度阈值
        fg_mask = digit > FG_THRESH  # (new_sz,new_sz) bool

        # 合成图：前景处用数字亮度覆盖背景（数字为白字，直接贴亮度）
        comp = canvas.copy()
        region = comp[top:top + new_sz, left:left + new_sz]
        region[fg_mask] = digit[fg_mask]  # 把数字笔画贴上去
        comp[top:top + new_sz, left:left + new_sz] = region

        # 4) 生成全图 GT
        gt = np.zeros((CANVAS, CANVAS), dtype=np.int64)
        gt_region = gt[top:top + new_sz, left:left + new_sz]
        if self.num_classes == 2:
            gt_region[fg_mask] = 1                 # 前景=1
        else:
            gt_region[fg_mask] = label + 1         # 数字类别+1（0留给背景）
        gt[top:top + new_sz, left:left + new_sz] = gt_region

        # 转张量：图像 (1,H,W)，GT (H,W)
        img_t = torch.from_numpy(comp).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt).long()
        return img_t, gt_t
