"""FCN 风格的 LeNet-5 分割网络（带跳跃连接 skip connection）。

核心思想（仿照 Fully Convolutional Network, FCN）：
  - 原版 LeNet-5 最后是全连接层，只能输出“整张图属于哪个数字”的 1 个分类结果；
  - 分割任务要“逐像素分类”，于是把全连接层换成 1x1 卷积（全卷积化），
    使网络输出一张与输入同分辨率的类别图；
  - 编码器下采样会丢失空间细节，导致分割边界粗糙。FCN 的关键贡献是
    “跳跃连接”(skip connection)：把编码器浅层的高分辨率特征融合进解码器，
    从而恢复边界细节。本实现仿照 FCN-8s 的思路做两级跳跃融合。

结构（输入 1x64x64）：
  编码器:
    enc1: Conv(1->6)  + Tanh              -> 6  x 64 x 64   (高分辨率，供 skip)
          Pool/2                          -> 6  x 32 x 32
    enc2: Conv(6->16) + Tanh              -> 16 x 32 x 32   (中分辨率，供 skip)
          Pool/2                          -> 16 x 16 x 16
    enc3: Conv(16->120)+Tanh              -> 120x 16 x 16
  分类头(全卷积化的 F6 + 输出层):
    Conv1x1(120->84)+Tanh, Conv1x1(84->C) -> C x 16 x 16  (逐像素打分)
  解码器(上采样 + 跳跃融合):
    上采样 16->32，与 enc2 的 1x1 投影(16->C)相加
    上采样 32->64，与 enc1 的 1x1 投影(6->C) 相加
  上采样方式 up_mode:
    'upsample' : 双线性插值（无参数）
    'deconv'   : 转置卷积 ConvTranspose2d（可学习参数）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNLeNet5(nn.Module):
    def __init__(self, num_classes=2, up_mode="upsample"):
        super().__init__()
        assert up_mode in ("upsample", "deconv")
        self.num_classes = num_classes
        self.up_mode = up_mode
        C = num_classes

        # ---- 编码器（拆成三段，便于取出中间特征做 skip）----
        self.enc1 = nn.Sequential(nn.Conv2d(1, 6, 5, padding=2), nn.Tanh())   # 64x64
        self.pool1 = nn.AvgPool2d(2)                                          # ->32
        self.enc2 = nn.Sequential(nn.Conv2d(6, 16, 5, padding=2), nn.Tanh())  # 32x32
        self.pool2 = nn.AvgPool2d(2)                                          # ->16
        self.enc3 = nn.Sequential(nn.Conv2d(16, 120, 5, padding=2), nn.Tanh())# 16x16

        # ---- 全卷积分类头：1x1 卷积替代全连接 ----
        self.head = nn.Sequential(
            nn.Conv2d(120, 84, 1), nn.Tanh(),
            nn.Conv2d(84, C, 1),
        )

        # ---- skip 分支：把浅层特征投影到 C 通道，供融合 ----
        self.skip2 = nn.Conv2d(16, C, 1)  # enc2 (32x32)
        self.skip1 = nn.Conv2d(6, C, 1)   # enc1 (64x64)

        # ---- 解码器上采样算子 ----
        if up_mode == "deconv":
            self.up1 = nn.ConvTranspose2d(C, C, 4, stride=2, padding=1)  # 16->32
            self.up2 = nn.ConvTranspose2d(C, C, 4, stride=2, padding=1)  # 32->64
            # 用双线性插值核初始化转置卷积（FCN 论文标准做法）。
            # 默认初始化在多类分割上易学不稳定，双线性核给出良好起点。
            self._init_bilinear(self.up1)
            self._init_bilinear(self.up2)

    @staticmethod
    def _init_bilinear(deconv):
        """把 ConvTranspose2d 的权重初始化为双线性上采样核。"""
        C_out, C_in, kh, kw = deconv.weight.shape
        factor = (kh + 1) // 2
        center = factor - 1 if kh % 2 == 1 else factor - 0.5
        og = torch.arange(kh).float()
        filt_1d = 1 - torch.abs(og - center) / factor
        filt = filt_1d[:, None] * filt_1d[None, :]  # (kh,kw) 双线性核
        with torch.no_grad():
            deconv.weight.zero_()
            for c in range(min(C_out, C_in)):
                deconv.weight[c, c] = filt
            if deconv.bias is not None:
                deconv.bias.zero_()

    def _upsample(self, x, size, op=None):
        if self.up_mode == "deconv":
            y = op(x)
            if y.shape[-2:] != size:
                y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            return y
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        in_size = x.shape[-2:]
        f1 = self.enc1(x)            # 6 x 64 x 64
        f2 = self.enc2(self.pool1(f1))  # 16 x 32 x 32
        f3 = self.enc3(self.pool2(f2))  # 120 x 16 x 16

        score = self.head(f3)        # C x 16 x 16

        # 第一级上采样 16->32，融合 enc2
        up1 = self._upsample(score, f2.shape[-2:],
                             op=self.up1 if self.up_mode == "deconv" else None)
        up1 = up1 + self.skip2(f2)

        # 第二级上采样 32->64，融合 enc1
        up2 = self._upsample(up1, f1.shape[-2:],
                             op=self.up2 if self.up_mode == "deconv" else None)
        up2 = up2 + self.skip1(f1)

        # 若与输入尺寸仍有偏差，最后对齐
        if up2.shape[-2:] != in_size:
            up2 = F.interpolate(up2, size=in_size, mode="bilinear", align_corners=False)
        return up2  # (N, C, 64, 64)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
