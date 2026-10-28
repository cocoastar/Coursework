"""训练与评估引擎，供三个任务复用。

提供：
  - train_model : 完整训练循环，记录每个 epoch 的 train/test 指标
  - evaluate    : 在数据集上评估 loss / accuracy，可返回混淆矩阵所需的预测
  - set_seed    : 固定随机种子，保证可复现
  - plot_curves : 绘制训练曲线
  - plot_confusion_matrix : 绘制混淆矩阵
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # 无显示环境，保存为文件
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """固定随机种子以保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, criterion, collect_preds=False):
    """在给定数据集上评估，返回 (平均loss, 准确率, [可选]所有预测, 所有标签)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        if collect_preds:
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    avg_loss = total_loss / total
    acc = correct / total
    if collect_preds:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        return avg_loss, acc, all_preds, all_labels
    return avg_loss, acc


def train_model(model, train_loader, test_loader, device,
                epochs=10, lr=1e-3, log_fn=print):
    """训练模型并返回每个 epoch 的指标历史。

    使用 Adam 优化器与交叉熵损失。每个 epoch 结束后在测试集上评估。
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        log_fn(f"  Epoch {epoch:2d}/{epochs} | "
               f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
               f"test_loss {test_loss:.4f} acc {test_acc:.4f}")

    return history


def plot_curves(history, title, save_path):
    """绘制 loss 与 accuracy 随 epoch 变化的曲线。"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], "o-", label="train")
    ax1.plot(epochs, history["test_loss"], "s-", label="test")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="train")
    ax2.plot(epochs, history["test_acc"], "s-", label="test")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(preds, labels, title, save_path, num_classes=10):
    """绘制混淆矩阵热力图。"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    # 在格子中标注数字
    thresh = cm.max() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def plot_misclassified(model, loader, device, title, save_path,
                       mean=0.1307, std=0.3081, max_samples=25):
    """挑出被模型识别错的样本，拼成网格图，标注 真实值->预测值。

    用于直观观察模型“错在哪”——哪些数字容易混淆（如 4/9、3/5）。
    """
    model.eval()
    imgs, trues, preds = [], [], []
    for images, labels in loader:
        out = model(images.to(device))
        p = out.argmax(dim=1).cpu()
        wrong = (p != labels).nonzero(as_tuple=True)[0]
        for idx in wrong:
            imgs.append(images[idx])
            trues.append(labels[idx].item())
            preds.append(p[idx].item())
            if len(imgs) >= max_samples:
                break
        if len(imgs) >= max_samples:
            break

    n = len(imgs)
    if n == 0:  # 理论上几乎不会发生
        return 0
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n:
            img = imgs[i].squeeze().numpy() * std + mean  # 反归一化回 [0,1]
            ax.imshow(img, cmap="gray")
            ax.set_title(f"T:{trues[i]} -> P:{preds[i]}", fontsize=8, color="red")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return n


def plot_param_accuracy_scatter(records, save_path):
    """绘制“参数量 vs 测试准确率”权衡散点图。

    records: [{'name', 'params', 'acc'(或 mean_acc)}...]
    横轴参数量（对数刻度），纵轴准确率，直观展示轻量化模型的性价比。
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in records:
        acc = r.get("mean_acc", r.get("best_test_acc"))
        ax.scatter(r["params"], acc, s=80)
        ax.annotate(r["name"], (r["params"], acc),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters (log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Params vs Accuracy trade-off")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
