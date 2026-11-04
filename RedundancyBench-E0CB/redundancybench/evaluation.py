"""Offline metrics for the three RedundancyBench judge tasks.

The runner saves Gold and parsed predictions together, so scoring needs no API
access or private experiment directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .data_loader import normalize_ref


DEFAULT_KS = (1, 5, 10)


def _ks_arg(value: str) -> tuple[int, ...]:
    try:
        ks = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--ks requires comma-separated positive integers") from exc
    if not ks or len(set(ks)) != len(ks) or any(k <= 0 for k in ks):
        raise argparse.ArgumentTypeError("--ks requires distinct positive integers")
    return ks


def _triple(item: dict[str, Any]) -> tuple[tuple[int, ...], str, tuple[int, ...]] | None:
    step = normalize_ref(item.get("redundant_step_idx"), evidence=False)
    evidence = normalize_ref(item.get("evidence_step_idx"), evidence=True)
    kind = item.get("redundant_step_type")
    if step is None or evidence is None or not isinstance(kind, str):
        return None
    return step, kind, evidence


def _confidence(item: dict[str, Any]) -> float:
    value = item.get("confidence")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return math.inf


def _detection_row(row: dict[str, Any], ks: Sequence[int]) -> dict[str, Any]:
    gold = {_triple(item) for item in row["gold"]}
    gold.discard(None)
    if not gold:
        raise ValueError(f"Detection trajectory has no Gold triples: {row.get('identity')}")
    returned = [] if row.get("error") or row.get("unjudgeable") else row.get("predictions")
    predictions = sorted(returned, key=_confidence, reverse=True) if isinstance(returned, list) else []
    scores: dict[str, float] = {}
    hits: dict[str, int] = {}
    for k in ks:
        top = predictions[: min(k, len(predictions))]
        matched: set[tuple[tuple[int, ...], str, tuple[int, ...]]] = set()
        seen_steps: set[tuple[int, ...]] = set()
        for item in top:
            if not isinstance(item, dict) or item.get("confidence") is None:
                continue
            step = normalize_ref(item.get("redundant_step_idx"), evidence=False)
            if step is None or item.get("duplicate") or item.get("duplicate_step") or step in seen_steps:
                continue
            seen_steps.add(step)
            if item.get("valid") is False:
                continue
            triple = _triple(item)
            if triple in gold:
                matched.add(triple)
        hits[str(k)] = len(matched)
        scores[str(k)] = len(matched) / len(top) if top else 0.0
    return {
        "identity": row.get("identity"),
        "trajectory_id": row.get("trajectory_id"),
        "domain": row.get("domain"),
        "dataset_id": row.get("dataset_id"),
        "gold_count": len(gold),
        "prediction_count": len(predictions),
        "hits": hits,
        "p_at_k": scores,
        "avg_p_at_k": sum(scores.values()) / len(ks),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate(task: str, rows: Sequence[dict[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    if task == "detect":
        p_at_k = {str(k): _mean([row["p_at_k"][str(k)] for row in rows]) for k in ks}
        return {
            "trajectory_count": len(rows),
            "gold_count": sum(row["gold_count"] for row in rows),
            "p_at_k": p_at_k,
            "avg_p_at_k": _mean(list(p_at_k.values())),
        }
    field = "joint_correct" if task == "retrieve" else "correct"
    correct = sum(bool(row[field]) for row in rows)
    return {
        "gold_count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
    }


def evaluate_payload(payload: dict[str, Any], ks: Sequence[int] = DEFAULT_KS) -> dict[str, Any]:
    """Score one complete runner result; failed/unparseable replies score zero."""
    metadata = payload["metadata"]
    task = metadata["task"]
    if task not in ("classify", "retrieve", "detect"):
        raise ValueError(f"Unknown task: {task}")
    results = payload["results"]
    expected = metadata.get("requested_units")
    identities = [row.get("identity") for row in results]
    if expected != len(results) or len(set(identities)) != len(identities):
        raise ValueError(f"Incomplete or duplicate result rows: {len(results)}/{expected}")
    scored: list[dict[str, Any]] = []
    for row in results:
        if task == "detect":
            scored.append(_detection_row(row, ks))
            continue
        gold_type = row["redundant_step_type"]
        type_correct = row.get("predicted_type") == gold_type and row.get("error") is None and not row.get("unjudgeable")
        item = {
            "identity": row.get("identity"),
            "trajectory_id": row.get("trajectory_id"),
            "domain": row.get("domain"),
            "dataset_id": row.get("dataset_id"),
            "type_correct": type_correct,
        }
        if task == "classify":
            item["correct"] = type_correct
        else:
            gold_evidence = normalize_ref(row.get("evidence_step_idx"), evidence=True)
            predicted_evidence = normalize_ref(row.get("predicted_evidence_idx"), evidence=True)
            item["evidence_correct"] = gold_evidence is not None and predicted_evidence == gold_evidence
            item["joint_correct"] = type_correct and item["evidence_correct"]
        scored.append(item)
    domains = sorted({row["domain"] for row in scored})
    datasets = sorted({row["dataset_id"] for row in scored})
    return {
        "task": task,
        "metric": {"classify": "StepCls Acc", "retrieve": "EvidenceRet Acc", "detect": "StepDet P@K"}[task],
        "ks": list(ks) if task == "detect" else None,
        "overall": _aggregate(task, scored, ks),
        "by_domain": {domain: _aggregate(task, [row for row in scored if row["domain"] == domain], ks) for domain in domains},
        "by_dataset": {dataset: _aggregate(task, [row for row in scored if row["dataset_id"] == dataset], ks) for dataset in datasets},
        "failed_or_unjudgeable": sum(bool(row.get("error") or row.get("unjudgeable") or row.get("format_failed")) for row in results),
        "per_unit": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="Runner JSON result files")
    parser.add_argument("--ks", type=_ks_arg, default=DEFAULT_KS)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    reports = {}
    for path in args.results:
        reports[str(path)] = evaluate_payload(json.loads(path.read_text(encoding="utf-8")), args.ks)
    report = {"evaluations": reports}
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.resolve() in {path.resolve() for path in args.results}:
            parser.error("--output must differ from the input result file")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
