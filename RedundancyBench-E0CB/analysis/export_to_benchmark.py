#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 trace_method_v3.py 的输出转成 benchmark evaluation.py 可读的 StepDet payload。

evaluation.py 的 detect 任务对每条轨迹要求：
  gold:        [{redundant_step_idx:[call,ret], redundant_step_type, evidence_step_idx}]
  predictions: [{redundant_step_idx:[call,ret], redundant_step_type, evidence_step_idx, confidence}]
命中需 (step, type, evidence) 三元组完全相等；step 必须是二元对，否则不计。

本脚本把 v3 里 label∈四类冗余 的事件转成一条 prediction：
  redundant_step_idx = [event.start, event第一个return下标]（与 benchmark 的 call/return 对齐）
  redundant_step_type = LABEL_TO_BENCH[label]
  evidence_step_idx   = 由 judgment.evidence_indices 归一成对（取最早一对；无则 [-1]）
  confidence          = judgment.confidence（规则存疑可微调，默认直用）
gold 从同域 annotation.json 按 task_id 读取（仅用于评分，不回流到标注）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_method_v3 import LABEL_TO_BENCH, REDUNDANT_LABELS


def pair_evidence(indices):
    """把证据下标归一成 benchmark 认的形式：单个 [a]、一对 [a,b] a<b，或 [-1]。
    benchmark 的 evidence 可以是单下标（如 [1]）或成对（如 [26,27]）。
    优先取最靠后的相邻调用/返回对 (a, a+1)（工具交互组）；否则单下标保留；再否则最早一对。"""
    ints = sorted({i for i in (indices or []) if isinstance(i, int) and i >= 0})
    adjacent = [(a, b) for a, b in zip(ints, ints[1:]) if b == a + 1]
    if adjacent:
        return list(adjacent[-1])
    if len(ints) == 1:
        return [ints[0]]
    for a, b in zip(ints, ints[1:]):
        if a < b:
            return [a, b]
    return [-1]


def evidence_for(judgment):
    """评分口径的 evidence：优先模型显式给的 evidence_pair，否则从 evidence_indices 归一。"""
    ep = judgment.get("evidence_pair")
    if isinstance(ep, list) and ep and all(isinstance(i, int) and i >= 0 for i in ep):
        if len(ep) == 1:
            return [ep[0]]
        if len(ep) == 2 and ep[0] < ep[1]:
            return [ep[0], ep[1]]
    return pair_evidence(judgment.get("evidence_indices"))


def event_step_pair(event):
    """benchmark 的 redundant_step_idx 是 [调用下标, 返回下标]。"""
    rets = event.get("return_indices") or []
    if event.get("kind") == "tool_call" and rets:
        return [event["start"], rets[0]]
    return None  # 无返回的事件无法构成合法 step，跳过


def load_gold(data_root, domain, task_id):
    ann_path = Path(data_root) / "human_annotation" / domain / "annotation.json"
    ann = json.load(open(ann_path, encoding="utf-8"))
    rec = next((a for a in ann if str(a.get("task_id")) == str(task_id)), None)
    if not rec:
        return []
    gold = []
    idxs = rec.get("redundant_step_idx", [])
    typs = rec.get("redundant_step_type", [])
    evs = rec.get("evidence_step_idx", [])
    for k in range(len(idxs)):
        gold.append({"redundant_step_idx": idxs[k],
                     "redundant_step_type": typs[k] if k < len(typs) else None,
                     "evidence_step_idx": evs[k] if k < len(evs) else [-1]})
    return gold


def to_predictions(v3_rec):
    preds = []
    for e in v3_rec.get("events", []):
        j = e.get("judgment")
        if not j or j.get("label") not in REDUNDANT_LABELS:
            continue
        step = event_step_pair(e)
        if step is None:
            continue
        preds.append({
            "redundant_step_idx": step,
            "redundant_step_type": LABEL_TO_BENCH[j["label"]],
            "evidence_step_idx": evidence_for(j),
            "confidence": float(j.get("confidence", 0.5)),
            "event_id": e["id"],
            "reason": j.get("reason", "")})
    preds.sort(key=lambda p: p["confidence"], reverse=True)
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3-output", required=True, type=Path, help="trace_method_v3.py 的 jsonl 输出")
    ap.add_argument("--domain", required=True, choices=["airline", "retail", "telecom"])
    ap.add_argument("--data-root", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--model", default="v3", help="写入 metadata 的模型名/标识")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.output.exists() and not args.overwrite:
        ap.error("输出已存在：换名字或加 --overwrite")

    results = []
    for line in open(args.v3_output, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        tid = rec["task_id"]
        gold = load_gold(args.data_root, args.domain, tid)
        if not gold:
            print(f"跳过 task {tid}：annotation 无 gold（StepDet 仅评有冗余的轨迹）", file=sys.stderr)
            continue
        results.append({
            "identity": f"{args.domain}:{tid}",
            "trajectory_id": tid,
            "domain": args.domain,
            "dataset_id": "human_annotation",
            "gold": gold,
            "predictions": to_predictions(rec),
            "error": bool(rec.get("status") not in ("ok", "partial", "screened") and not rec.get("events"))})

    payload = {"metadata": {"task": "detect", "model": args.model,
                            "requested_units": len(results), "dataset_id": "human_annotation"},
               "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"完成：{args.output}（{len(results)} 条轨迹）。"
          f"评分：python3 -m redundancybench.evaluation {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
