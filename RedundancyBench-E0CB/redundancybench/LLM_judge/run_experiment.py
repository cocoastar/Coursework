"""Run one or all three RedundancyBench main tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data_loader import load_dataset
from .runner import RunConfig, run_task


TASKS = ("classify", "retrieve", "detect")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--dataset",
        choices=("all", "human_annotation", "automated_annotation"),
        default="all",
        help="Use one released subset or both subsets.",
    )
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--domain", choices=("all", "airline", "retail", "telecom"), default="all")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-tokens-param", choices=("max_tokens", "max_completion_tokens"), default="max_tokens")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--reasoning-effort")
    parser.add_argument(
        "--extra-body",
        type=json.loads,
        default={},
        help="JSON object merged into the API request body for provider-specific options.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument("--max-predictions", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_predictions is not None and args.max_predictions <= 0:
        parser.error("--max-predictions must be positive")
    if not isinstance(args.extra_body, dict):
        parser.error("--extra-body must be a JSON object")
    if set(args.extra_body) & {"model", "messages", "stream", "temperature", "max_tokens", "max_completion_tokens", "reasoning_effort"}:
        parser.error("--extra-body cannot override core request fields")
    dataset_names = (
        ("human_annotation", "automated_annotation")
        if args.dataset == "all"
        else (args.dataset,)
    )
    dataset = load_dataset(args.data_root, dataset_names)
    print(f"Loaded {len(dataset.trajectories)} trajectories and {len(dataset.cases)} Gold cases")
    config = RunConfig(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_tokens_param=args.max_tokens_param,
        timeout=args.timeout,
        reasoning_effort=args.reasoning_effort,
        extra_body=args.extra_body,
    )
    tasks = TASKS if args.task == "all" else (args.task,)
    for task in tasks:
        path = run_task(
            dataset,
            task,
            config,
            args.output_dir,
            domain=args.domain,
            limit=args.limit,
            include_policy=not args.no_policy,
            max_predictions=args.max_predictions,
            dry_run=args.dry_run,
            resume=args.resume,
        )
        print(f"{task}: {path}")


if __name__ == "__main__":
    main()
