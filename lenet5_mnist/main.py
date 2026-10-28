"""主入口：运行三个任务（多种子重复实验）+ 反事实实验，并汇总对比结果。

用法:
    python main.py                      # 多种子跑全部任务（默认 3 个种子, 10 epoch）
    python main.py --seeds 1,2,3        # 指定种子
    python main.py --epochs 20          # 自定义训练轮数
    python main.py --task 1             # 只运行指定任务 (1/2/3)
    python main.py --counterfactual     # 额外运行“小训练集反事实实验”
    python main.py --only_counterfactual  # 只跑反事实实验

产出目录结构（results/ 下，按用途分类）：
    figures/      展示用图：训练曲线、混淆矩阵、对比图(带误差棒)、错分样例、权衡散点
    reports/      汇总表：summary.md / summary.csv（含 mean±std）
    checkpoints/  模型权重 .pt（每个模型存最后一个种子的权重，用于复现/推理）
    logs/         原始训练历史 json、训练日志（存档用）
"""
import os
import csv
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_dataloaders, get_dataloaders_subset
from models import LeNet5, LeNet5Half, LeNet5Dropout, count_parameters
from engine import (set_seed, train_model, evaluate,
                    plot_curves, plot_confusion_matrix,
                    plot_misclassified, plot_param_accuracy_scatter)

RESULTS_DIR = "results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
REPORT_DIR = os.path.join(RESULTS_DIR, "reports")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

# 任务清单：(名称, 构造函数)。用构造函数而非实例，保证每个种子都是全新初始化。
MODEL_SPECS = [
    ("task1_lenet5", lambda: LeNet5()),
    ("task2_lenet5_half", lambda: LeNet5Half()),
    ("task3_dropout_01pct", lambda: LeNet5Dropout(p=0.01)),
    ("task3_dropout_05pct", lambda: LeNet5Dropout(p=0.05)),
    ("task3_dropout_10pct", lambda: LeNet5Dropout(p=0.10)),
]


def select_specs(task):
    if task == 0:
        return MODEL_SPECS
    if task == 1:
        return MODEL_SPECS[:1]
    if task == 2:
        return MODEL_SPECS[1:2]
    if task == 3:
        return MODEL_SPECS[2:]
    return MODEL_SPECS


def train_one_seed(name, build_fn, train_loader, test_loader, device,
                   epochs, seed, log, save_artifacts):
    """用指定种子训练一个模型，返回该次运行的指标与(可选)预测。"""
    set_seed(seed)
    model = build_fn().to(device)
    history = train_model(model, train_loader, test_loader, device,
                          epochs=epochs, log_fn=log)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, device, criterion, collect_preds=True)
    best_acc = max(history["test_acc"])
    result = {
        "seed": seed,
        "final_test_acc": test_acc,
        "best_test_acc": best_acc,
        "final_test_loss": test_loss,
        "final_train_acc": history["train_acc"][-1],
    }

    if save_artifacts:
        # 只为“代表性种子”保存展示图/权重，避免产物爆炸
        plot_curves(history, f"{name} (test acc={test_acc:.4f}, seed={seed})",
                    os.path.join(FIG_DIR, f"{name}_curves.png"))
        plot_confusion_matrix(preds, labels, f"{name} Confusion Matrix",
                              os.path.join(FIG_DIR, f"{name}_confusion.png"))
        plot_misclassified(model, test_loader, device,
                           f"{name} Misclassified samples (T:true -> P:pred)",
                           os.path.join(FIG_DIR, f"{name}_misclassified.png"))
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{name}.pt"))
        with open(os.path.join(LOG_DIR, f"{name}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    return result


def run_multiseed(specs, train_loader, test_loader, device, epochs, seeds, log):
    """对每个模型跑多个种子，汇总 mean±std。"""
    records = []
    for name, build_fn in specs:
        n_params = count_parameters(build_fn())
        log(f"\n=== {name} | 参数量 {n_params:,} | 种子 {seeds} ===")
        seed_results = []
        for i, seed in enumerate(seeds):
            # 只用第一个种子保存展示产物作为“代表”
            res = train_one_seed(name, build_fn, train_loader, test_loader,
                                 device, epochs, seed, log,
                                 save_artifacts=(i == 0))
            log(f"  [seed {seed}] best_test_acc={res['best_test_acc']:.4f}")
            seed_results.append(res)

        best_accs = np.array([r["best_test_acc"] for r in seed_results])
        final_accs = np.array([r["final_test_acc"] for r in seed_results])
        train_accs = np.array([r["final_train_acc"] for r in seed_results])
        rec = {
            "name": name,
            "params": n_params,
            "mean_acc": float(best_accs.mean()),
            "std_acc": float(best_accs.std()),
            "mean_final_acc": float(final_accs.mean()),
            "mean_train_acc": float(train_accs.mean()),
            "n_seeds": len(seeds),
            "seed_accs": best_accs.tolist(),
        }
        log(f"  >> mean best_test_acc = {rec['mean_acc']:.4f} ± {rec['std_acc']:.4f}")
        records.append(rec)
    return records


def write_summary(records):
    """写 csv / markdown 汇总表，并绘制带误差棒的对比图 + 权衡散点图。"""
    fields = ["name", "params", "mean_acc", "std_acc",
              "mean_final_acc", "mean_train_acc", "n_seeds"]
    with open(os.path.join(REPORT_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({k: round(r[k], 6) if isinstance(r[k], float) else r[k]
                             for k in fields})

    lines = [f"> 每个模型重复 {records[0]['n_seeds']} 个随机种子，报告 均值 ± 标准差。\n",
             "| 模型 | 参数量 | 最佳测试准确率 (mean±std) | 最终训练准确率 | 过拟合差距 |",
             "|---|---|---|---|---|"]
    for r in records:
        gap = r["mean_train_acc"] - r["mean_final_acc"]
        lines.append(f"| {r['name']} | {r['params']:,} | "
                     f"{r['mean_acc']*100:.2f}% ± {r['std_acc']*100:.2f}% | "
                     f"{r['mean_train_acc']*100:.2f}% | {gap*100:.2f}% |")
    with open(os.path.join(REPORT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # 带误差棒的对比图
    names = [r["name"] for r in records]
    means = [r["mean_acc"] for r in records]
    stds = [r["std_acc"] for r in records]
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.4), 4.8))
    ax.bar(names, means, yerr=stds, capsize=5, color="#4C78A8")
    ax.set_ylabel("Best Test Accuracy (mean +/- std)")
    ax.set_ylim(min(means) - max(0.01, max(stds) * 2), 1.0)
    ax.set_title("Best Test Accuracy across seeds")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s, f"{m:.4f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "comparison.png"), dpi=120)
    plt.close(fig)

    # 参数量-精度权衡散点图
    plot_param_accuracy_scatter(records, os.path.join(FIG_DIR, "params_vs_accuracy.png"))


def run_counterfactual(device, epochs, seeds, subset_size, log):
    """反事实实验：小训练集 -> 制造过拟合 -> 验证 dropout 能否反超原版。"""
    log(f"\n########## 反事实实验：训练集仅 {subset_size} 张 ##########")
    train_loader, test_loader = get_dataloaders_subset(subset_size=subset_size)

    specs = [
        ("cf_lenet5_original", lambda: LeNet5()),
        ("cf_dropout_10pct", lambda: LeNet5Dropout(p=0.10)),
        ("cf_dropout_20pct", lambda: LeNet5Dropout(p=0.20)),
    ]
    records = []
    for name, build_fn in specs:
        log(f"\n=== [反事实] {name} ===")
        accs, trains = [], []
        for seed in seeds:
            res = train_one_seed(name, build_fn, train_loader, test_loader,
                                 device, epochs, seed, log, save_artifacts=False)
            accs.append(res["best_test_acc"])
            trains.append(res["final_train_acc"])
        accs, trains = np.array(accs), np.array(trains)
        rec = {"name": name, "mean_acc": float(accs.mean()),
               "std_acc": float(accs.std()), "mean_train_acc": float(trains.mean())}
        log(f"  >> mean best_test_acc = {rec['mean_acc']:.4f} ± {rec['std_acc']:.4f} "
            f"(train={rec['mean_train_acc']:.4f})")
        records.append(rec)

    # 反事实对比图
    names = [r["name"] for r in records]
    means = [r["mean_acc"] for r in records]
    stds = [r["std_acc"] for r in records]
    trains = [r["mean_train_acc"] for r in records]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, trains, 0.4, label="train acc", color="#F58518")
    ax.bar(x + 0.2, means, 0.4, yerr=stds, capsize=5, label="test acc", color="#4C78A8")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Counterfactual: small train set ({subset_size} imgs)\n"
                 "train>>test = overfit; dropout narrows the gap & lifts test acc")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "counterfactual.png"), dpi=120)
    plt.close(fig)

    # 报告
    lines = [f"# 反事实实验：训练集缩小到 {subset_size} 张\n",
             f"> 每个模型重复 {len(seeds)} 个种子。制造过拟合场景以检验 dropout 的正则化收益。\n",
             "| 模型 | 测试准确率 (mean±std) | 训练准确率 | 过拟合差距 |",
             "|---|---|---|---|"]
    base = records[0]["mean_acc"]
    for r in records:
        gap = r["mean_train_acc"] - r["mean_acc"]
        delta = (r["mean_acc"] - base) * 100
        tag = "" if r["name"] == "cf_lenet5_original" else f" (相对原版 {delta:+.2f}%)"
        lines.append(f"| {r['name']} | {r['mean_acc']*100:.2f}% ± {r['std_acc']*100:.2f}%{tag} | "
                     f"{r['mean_train_acc']*100:.2f}% | {gap*100:.2f}% |")
    with open(os.path.join(REPORT_DIR, "counterfactual.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--task", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=str, default="42,123,2024",
                        help="逗号分隔的随机种子列表")
    parser.add_argument("--counterfactual", action="store_true",
                        help="在主实验后额外运行小训练集反事实实验")
    parser.add_argument("--only_counterfactual", action="store_true")
    parser.add_argument("--subset_size", type=int, default=1000)
    args = parser.parse_args()

    for d in (FIG_DIR, REPORT_DIR, CKPT_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]

    log_file = open(os.path.join(LOG_DIR, "train.log"), "a")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"设备: {device} | epochs: {args.epochs} | seeds: {seeds}")

    if not args.only_counterfactual:
        train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
        specs = select_specs(args.task)
        records = run_multiseed(specs, train_loader, test_loader, device,
                                args.epochs, seeds, log)
        if records:
            write_summary(records)
            log("\n=== 主实验汇总 (mean±std) ===")
            for r in records:
                log(f"  {r['name']:28s} params={r['params']:>8,} "
                    f"acc={r['mean_acc']*100:.2f}%±{r['std_acc']*100:.2f}%")

    if args.counterfactual or args.only_counterfactual:
        run_counterfactual(device, args.epochs, seeds, args.subset_size, log)

    log(f"\n结果已保存至 {RESULTS_DIR}/ (figures/ reports/ checkpoints/ logs/)")
    log_file.close()


if __name__ == "__main__":
    main()
