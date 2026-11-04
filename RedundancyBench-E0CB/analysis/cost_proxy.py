#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""代理成本指标 —— 从完整轨迹里量化成本并关联冗余标注。

本数据集（RedundancyBench）的 messages 里没有真实 token、没有时间戳、没有耗时。
因此成本以三类"代理指标"近似，局限见下：

  调用成本   n_tool_calls          工具调用次数
  上下文成本 tool_return_chars     tool 返回内容字符数（≈ token × 4，粗略）
  交互成本   n_turns               对话轮数（user/assistant 文本回合）

关联 annotation 后，额外给出：
  n_redundant        被标注的冗余步数
  redundant_chars    冗余步（assistant调用+tool返回）占用的字符 = 可节省的代理成本
  <类型>             四类冗余各自计数

用法:
  python3 cost_proxy.py                      # 打印全部三域人工子集表格
  python3 cost_proxy.py --csv out.csv        # 同时写 CSV
  python3 cost_proxy.py --top 15             # 按冗余+失败+重复排序取前 N 行
  python3 cost_proxy.py --subset automated   # 换成模型共识子集
"""
import argparse
import csv
import json
import os
from collections import Counter

# 原始数据只读，相对本脚本定位 data/
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..", "data")
DOMAINS = ["airline", "retail", "telecom"]
TYPES = ["Duplicated Step", "Abnormal Step", "Incorrect Step", "Exploratory Step"]
CHARS_PER_TOKEN = 4  # 粗略换算：英文约 4 字符/token


def load_domain(subset, domain):
    base = os.path.join(DATA_ROOT, subset, domain)
    traces_path = os.path.join(base, "final_traces.json")
    ann_path = os.path.join(base, "annotation.json")
    if not os.path.exists(traces_path):
        return [], {}
    traces = json.load(open(traces_path, encoding="utf-8"))
    sims = traces["simulations"]
    ann = json.load(open(ann_path, encoding="utf-8"))
    # 人工子集按 task_id，模型子集按 trajectory_id
    key = "task_id" if subset == "human_annotation" else "trajectory_id"
    ann_map = {a.get(key, a.get("task_id")): a for a in ann}
    return sims, ann_map


def analyze(sim, ann):
    msgs = sim["messages"]
    n_messages = len(msgs)
    n_tool_calls = sum(len(m.get("tool_calls") or []) for m in msgs)
    # 一个 turn ≈ 一次 user 或 assistant 文本发言（排除纯工具回合）
    n_turns = sum(
        1 for m in msgs
        if m["role"] in ("user", "assistant") and (m.get("content") or "").strip()
    )
    tool_return_chars = sum(
        len(m.get("content") or "") for m in msgs if m["role"] == "tool"
    )
    n_failed_calls = sum(
        1 for m in msgs if m["role"] == "tool" and m.get("error")
    )
    names = [tc["name"] for m in msgs for tc in (m.get("tool_calls") or [])]
    name_count = Counter(names)
    n_repeat_calls = sum(c - 1 for c in name_count.values() if c > 1)

    row = {
        "task_id": sim["task_id"],
        "n_messages": n_messages,
        "n_turns": n_turns,
        "n_tool_calls": n_tool_calls,
        "tool_return_chars": tool_return_chars,
        "approx_tokens": tool_return_chars // CHARS_PER_TOKEN,
        "n_failed_calls": n_failed_calls,
        "n_repeat_calls": n_repeat_calls,
        "n_redundant": 0,
        "redundant_chars": 0,
    }
    for t in TYPES:
        row[t] = 0

    if ann:
        idxs = ann.get("redundant_step_idx", [])
        row["n_redundant"] = len(idxs)
        for t in ann.get("redundant_step_type", []):
            if t in row:
                row[t] += 1
        # 冗余步占用的字符（可节省的代理成本）：成对下标内所有 message 的 content
        rchars = 0
        for pair in idxs:
            for i in pair:
                if 0 <= i < n_messages:
                    rchars += len(msgs[i].get("content") or "")
        row["redundant_chars"] = rchars
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="human_annotation",
                    choices=["human_annotation", "automated_annotation"])
    ap.add_argument("--domain", choices=DOMAINS)
    ap.add_argument("--csv")
    ap.add_argument("--top", type=int, default=0,
                    help="按 冗余+失败+重复 排序取前 N 行；0=全部按域序")
    args = ap.parse_args()

    domains = [args.domain] if args.domain else DOMAINS
    rows = []
    for dom in domains:
        sims, ann_map = load_domain(args.subset, dom)
        for s in sims:
            r = analyze(s, ann_map.get(s["task_id"]))
            r = {"domain": dom, **r}
            rows.append(r)

    if args.top:
        rows.sort(
            key=lambda r: (r["n_redundant"], r["n_failed_calls"], r["n_repeat_calls"]),
            reverse=True,
        )
        rows = rows[: args.top]

    # 打印精简表
    hdr = ["domain", "task_id", "n_messages", "n_tool_calls", "approx_tokens",
           "n_failed_calls", "n_repeat_calls", "n_redundant", "redundant_chars"]
    print("  ".join(f"{h:>14}" if h != "task_id" else f"{h:<40}" for h in hdr))
    for r in rows:
        line = []
        for h in hdr:
            v = str(r[h])
            line.append(f"{v:<40}" if h == "task_id" else f"{v:>14}")
        print("  ".join(line))

    # 汇总
    tot = len(rows)
    print(f"\n共 {tot} 条轨迹。"
          f"总调用 {sum(r['n_tool_calls'] for r in rows)}，"
          f"总≈token {sum(r['approx_tokens'] for r in rows)}，"
          f"冗余步占用≈token {sum(r['redundant_chars'] for r in rows)//CHARS_PER_TOKEN}（"
          f"占 {100*sum(r['redundant_chars'] for r in rows)/max(1,sum(r['tool_return_chars'] for r in rows)):.1f}%）")

    if args.csv:
        full_hdr = ["domain", "task_id", "n_messages", "n_turns", "n_tool_calls",
                    "tool_return_chars", "approx_tokens", "n_failed_calls",
                    "n_repeat_calls", "n_redundant", "redundant_chars"] + TYPES
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=full_hdr)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in full_hdr})
        print(f"\n已写出 CSV: {args.csv}")


if __name__ == "__main__":
    main()
