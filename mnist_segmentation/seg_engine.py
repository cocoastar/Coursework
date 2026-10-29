"""分割任务的训练、评估与可视化引擎。

分割 = 逐像素分类，因此：
  - 损失：对每个像素做交叉熵（CrossEntropyLoss 直接支持 (N,C,H,W) vs (N,H,W)）
  - 评估指标：
      * 像素准确率 (Pixel Accuracy): 预测对的像素 / 总像素
      * mIoU (mean Intersection over Union): 分割任务的标准指标，
        对每个类别算 交集/并集，再取平均。比像素准确率更能反映分割质量
        （尤其前景占比小时，像素准确率会虚高）。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class DiceCELoss(nn.Module):
    """交叉熵 + Dice 的组合损失，专门对付分割中的类别不均衡。

    - 交叉熵(带前景加权)保证逐像素分类的基本监督；
    - Dice Loss 直接优化“预测前景与真值前景的重叠度”，对“前景占比极小”
      的情形远比纯交叉熵有效（不会因背景太多而被淹没）。
    总损失 = ce_weight * CE + dice_weight * Dice。
    """

    def __init__(self, num_classes, class_weights=None,
                 ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, target):
        # logits: (N,C,H,W)  target: (N,H,W)
        ce = self.ce(logits, target)

        probs = torch.softmax(logits, dim=1)                 # (N,C,H,W)
        target_1h = nn.functional.one_hot(target, self.num_classes)  # (N,H,W,C)
        target_1h = target_1h.permute(0, 3, 1, 2).float()    # (N,C,H,W)
        dims = (0, 2, 3)
        inter = (probs * target_1h).sum(dims)
        card = probs.sum(dims) + target_1h.sum(dims)
        dice = (2 * inter + 1.0) / (card + 1.0)              # 每类 Dice，+1 平滑
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce + self.dice_weight * dice_loss


@torch.no_grad()
def compute_miou_present(preds, labels, num_classes):
    """只对“该图真值中出现过的类别”计算 IoU 后平均（更贴合单数字场景）。

    标准 mIoU 会把每张图里缺席的 9 个数字类别也计入（IoU=0），拉低均值。
    本指标逐图统计：只对 GT 中真实出现的类别算 IoU，再对所有图求平均。
    """
    N = preds.shape[0]
    ious = []
    for i in range(N):
        p = preds[i].flatten()
        l = labels[i].flatten()
        present = torch.unique(l).tolist()
        for c in present:
            pred_c = p == c
            label_c = l == c
            inter = (pred_c & label_c).sum().item()
            union = (pred_c | label_c).sum().item()
            if union > 0:
                ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def compute_miou(preds, labels, num_classes, return_per_class=False):
    """计算 mIoU 与像素准确率。preds/labels: (N,H,W) 的类别索引张量。"""
    preds = preds.flatten()
    labels = labels.flatten()
    pixel_acc = (preds == labels).float().mean().item()

    ious = []
    per_class = {}
    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c
        inter = (pred_c & label_c).sum().item()
        union = (pred_c | label_c).sum().item()
        if union == 0:
            per_class[c] = None  # 该类别在本批中不存在
            continue
        iou = inter / union
        per_class[c] = iou
        ious.append(iou)
    miou = float(np.mean(ious)) if ious else 0.0
    if return_per_class:
        return miou, pixel_acc, per_class
    return miou, pixel_acc


@torch.no_grad()
def evaluate(model, loader, device, num_classes, criterion, return_per_class=False):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    for imgs, gts in loader:
        imgs, gts = imgs.to(device), gts.to(device)
        out = model(imgs)
        loss = criterion(out, gts)
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        all_preds.append(out.argmax(1).cpu())
        all_labels.append(gts.cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    miou, pacc = compute_miou(preds, labels, num_classes)
    miou_present = compute_miou_present(preds, labels, num_classes)
    if return_per_class:
        _, _, per_class = compute_miou(preds, labels, num_classes,
                                       return_per_class=True)
        return total_loss / n, miou, pacc, per_class, miou_present
    return total_loss / n, miou, pacc, miou_present


def train_model(model, train_loader, test_loader, device, num_classes,
                epochs=10, lr=1e-3, log_fn=print, class_weights=None,
                loss_type="dice_ce", use_scheduler=True):
    """训练分割模型。

    loss_type:
        'ce'      : 纯交叉熵（带前景加权）——旧版基线
        'dice_ce' : 交叉熵 + Dice 组合损失——优化版（对类别不均衡更鲁棒）
    use_scheduler: 是否启用 cosine 学习率衰减。
    """
    model.to(device)
    # 前景加权：背景像素占比高(~94%)，背景类权重调低，迫使网络认真分割前景。
    if class_weights is None:
        w = torch.ones(num_classes)
        w[0] = 0.3
        class_weights = w
    class_weights = class_weights.to(device)

    if loss_type == "dice_ce":
        criterion = DiceCELoss(num_classes, class_weights=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if use_scheduler else None)

    history = {"train_loss": [], "test_loss": [], "test_miou": [],
               "test_pacc": [], "test_miou_present": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for imgs, gts in train_loader:
            imgs, gts = imgs.to(device), gts.to(device)
            optimizer.zero_grad()
            out = model(imgs)               # (N,C,H,W)
            loss = criterion(out, gts)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            n += imgs.size(0)
        if scheduler is not None:
            scheduler.step()

        train_loss = running / n
        test_loss, miou, pacc, miou_p = evaluate(model, test_loader, device,
                                                 num_classes, criterion)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_miou"].append(miou)
        history["test_pacc"].append(pacc)
        history["test_miou_present"].append(miou_p)
        log_fn(f"  Epoch {epoch:2d}/{epochs} | train_loss {train_loss:.4f} | "
               f"test_loss {test_loss:.4f} | mIoU {miou:.4f} | "
               f"mIoU(present) {miou_p:.4f} | pixAcc {pacc:.4f}")
    return history


@torch.no_grad()
def visualize_predictions(model, loader, device, num_classes, save_path,
                          n_samples=6):
    """画 输入图 / GT / 预测 三联对比图，直观看分割效果。"""
    model.eval()
    imgs, gts = next(iter(loader))
    imgs, gts = imgs[:n_samples].to(device), gts[:n_samples]
    out = model(imgs).argmax(1).cpu()

    cmap = "gray" if num_classes == 2 else "tab10"
    vmax = 1 if num_classes == 2 else num_classes - 1
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 2, 6))
    for i in range(n_samples):
        axes[0, i].imshow(imgs[i, 0].cpu().numpy(), cmap="gray")
        axes[1, i].imshow(gts[i].numpy(), cmap=cmap, vmin=0, vmax=vmax)
        axes[2, i].imshow(out[i].numpy(), cmap=cmap, vmin=0, vmax=vmax)
        for r in range(3):
            axes[r, i].axis("off")
    axes[0, 0].set_ylabel("input")
    for ax, name in zip(axes[:, 0], ["Input", "GT", "Pred"]):
        ax.set_title("")
    fig.suptitle(f"Segmentation: Input / GT / Prediction  (classes={num_classes})")
    # 行标签
    for r, name in enumerate(["Input", "GT", "Pred"]):
        axes[r, 0].text(-0.3, 0.5, name, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=10, rotation=90)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_curves(history, title, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], "o-", label="train")
    ax1.plot(epochs, history["test_loss"], "s-", label="test")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["test_miou"], "o-", label="mIoU")
    ax2.plot(epochs, history["test_pacc"], "s-", label="pixel acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score"); ax2.set_title("Metrics")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(save_path, dpi=120); plt.close(fig)
