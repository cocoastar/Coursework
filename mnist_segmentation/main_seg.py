"""主入口：训练并对比 4 个分割模型组合。

4 个组合 = 2 种分割类别 × 2 种上采样方式：
    (2 分类 / 11 分类) × (Upsampling / Deconvolution)

用法:
    python main_seg.py                 # 跑全部 4 个组合（默认 25 epoch）
    python main_seg.py --epochs 30
    python main_seg.py --only 2cls_upsample   # 只跑某个组合

产出（results/ 下）：
    <组合名>_curves.png   训练曲线（loss / mIoU / pixel acc）
    <组合名>_pred.png     分割可视化（输入 / GT / 预测 三联）
    <组合名>.pt           模型权重
    summary.md / csv      4 个组合的 mIoU / 像素准确率对比表
"""
import os
import csv
import json
import argparse
import torch
from torch.utils.data import DataLoader

from dataset import MNISTSegDataset
from seg_models import FCNLeNet5, count_parameters
from seg_engine import (set_seed, train_model, evaluate,
                        visualize_predictions, plot_curves)

MNIST_ROOT = "../lenet5_mnist/data"   # 复用作业1下载的 MNIST
BG_DIR = "backgrounds"
RESULTS_DIR = "results"

# 组合定义：(组合名, 分割类别数, 上采样方式)
COMBOS = [
    ("2cls_upsample", 2, "upsample"),
    ("2cls_deconv", 2, "deconv"),
    ("10cls_upsample", 11, "upsample"),   # 11 = 背景 + 数字0~9
    ("10cls_deconv", 11, "deconv"),
]


def run_combo(name, num_classes, up_mode, args, device, log):
    log(f"\n=== {name} | classes={num_classes} | up={up_mode} ===")
    set_seed(args.seed)

    train_ds = MNISTSegDataset(MNIST_ROOT, BG_DIR, num_classes=num_classes,
                               length=args.train_size, train=True, seed=args.seed)
    test_ds = MNISTSegDataset(MNIST_ROOT, BG_DIR, num_classes=num_classes,
                              length=args.test_size, train=False, seed=args.seed)
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True)
    vl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)

    model = FCNLeNet5(num_classes=num_classes, up_mode=up_mode)
    n_params = count_parameters(model)
    log(f"  参数量 {n_params:,}")

    history = train_model(model, tl, vl, device, num_classes,
                          epochs=args.epochs, log_fn=log,
                          loss_type=args.loss, use_scheduler=not args.no_sched)

    # 最终评估（含每类 IoU 与“仅出现类”mIoU）
    import torch.nn as nn
    w = torch.ones(num_classes); w[0] = 0.3
    criterion = nn.CrossEntropyLoss(weight=w.to(device))
    test_loss, miou, pacc, per_class, miou_present = evaluate(
        model, vl, device, num_classes, criterion, return_per_class=True)
    fg_ious = [v for c, v in per_class.items() if c > 0 and v is not None]
    fg_miou = sum(fg_ious) / len(fg_ious) if fg_ious else 0.0
    log(f"  >> 最终 mIoU {miou:.4f} | 前景mIoU {fg_miou:.4f} | "
        f"mIoU(出现类) {miou_present:.4f} | pixAcc {pacc:.4f}")

    # 保存产物
    plot_curves(history, f"{name} (mIoU={miou:.3f})",
                os.path.join(RESULTS_DIR, f"{name}_curves.png"))
    visualize_predictions(model, vl, device, num_classes,
                          os.path.join(RESULTS_DIR, f"{name}_pred.png"))
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{name}.pt"))
    with open(os.path.join(RESULTS_DIR, f"{name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return {
        "name": name, "num_classes": num_classes, "up_mode": up_mode,
        "params": n_params, "miou": round(miou, 4),
        "fg_miou": round(fg_miou, 4), "pixel_acc": round(pacc, 4),
        "best_miou": round(max(history["test_miou"]), 4),
        "miou_present": round(miou_present, 4),
        "best_miou_present": round(max(history["test_miou_present"]), 4),
    }


def write_summary(records, suffix=""):
    fields = ["name", "num_classes", "up_mode", "params", "miou", "fg_miou",
              "pixel_acc", "best_miou", "miou_present", "best_miou_present"]
    with open(os.path.join(RESULTS_DIR, f"summary{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(records)

    lines = ["| 组合 | 分割类别 | 上采样 | 参数量 | mIoU(全类) | mIoU(出现类) | 前景mIoU | 像素准确率 |",
             "|---|---|---|---|---|---|---|---|"]
    label = {2: "2分类", 11: "10分类(+背景)"}
    for r in records:
        lines.append(f"| {r['name']} | {label[r['num_classes']]} | {r['up_mode']} | "
                     f"{r['params']:,} | {r['best_miou']} | {r['best_miou_present']} | "
                     f"{r['fg_miou']} | {r['pixel_acc']} |")
    with open(os.path.join(RESULTS_DIR, f"summary{suffix}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--train_size", type=int, default=6000)
    ap.add_argument("--test_size", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", type=str, default=None,
                    help="只跑指定组合名，如 2cls_upsample")
    ap.add_argument("--loss", type=str, default="dice_ce",
                    choices=["ce", "dice_ce"], help="损失类型")
    ap.add_argument("--no_sched", action="store_true", help="关闭学习率衰减")
    ap.add_argument("--suffix", type=str, default="", help="汇总文件名后缀")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_file = open(os.path.join(RESULTS_DIR, "train.log"), "a")
    def log(msg):
        print(msg); log_file.write(msg + "\n"); log_file.flush()

    log(f"设备 {device} | epochs {args.epochs} | train_size {args.train_size}")

    combos = COMBOS if args.only is None else [c for c in COMBOS if c[0] == args.only]
    records = [run_combo(n, nc, up, args, device, log) for n, nc, up in combos]

    if records:
        write_summary(records, suffix=args.suffix)
        log("\n=== 汇总 ===")
        for r in records:
            log(f"  {r['name']:16s} best_mIoU={r['best_miou']} "
                f"best_mIoU(出现类)={r['best_miou_present']} "
                f"fg_mIoU={r['fg_miou']} pixAcc={r['pixel_acc']}")
    log_file.close()


if __name__ == "__main__":
    main()
