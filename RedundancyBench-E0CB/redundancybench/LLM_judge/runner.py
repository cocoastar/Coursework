"""OpenAI-compatible runner for the three main tasks.

This module only calls the model and writes raw/parsed predictions.  Metric
calculation is intentionally left to the offline ``evaluation.py`` module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib import error, request

from ..data_loader import Dataset, GoldCase, Trajectory
from .prompts import PROMPT_VERSION, build_prompt
from .response_parser import PARSERS


@dataclass(frozen=True)
class RunConfig:
    model: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 32768
    max_tokens_param: str = "max_tokens"
    timeout: float = 600.0
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RunConfig":
        return cls(
            model=str(value["model"]),
            base_url=str(value["base_url"]).rstrip("/"),
            api_key_env=str(value.get("api_key_env", "OPENAI_API_KEY")),
            temperature=float(value.get("temperature", 0)),
            max_tokens=int(value.get("max_tokens", 32768)),
            max_tokens_param=str(value.get("max_tokens_param", "max_tokens")),
            timeout=float(value.get("timeout", 600)),
            reasoning_effort=value.get("reasoning_effort"),
            extra_body=dict(value.get("extra_body") or {}),
        )


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_filename(value: str) -> str:
    """Keep model names usable as filenames on all supported platforms."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return name or "model"


def _content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _request(config: RunConfig, system_prompt: str, user_prompt: str, api_key: str) -> tuple[str, dict[str, Any]]:
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "temperature": config.temperature,
        config.max_tokens_param: config.max_tokens,
    }
    if config.reasoning_effort is not None:
        body["reasoning_effort"] = config.reasoning_effort
    body.update(config.extra_body or {})
    # Prevent extra_body from enabling streaming; parsers need a complete reply.
    body["stream"] = False
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    endpoint = config.base_url + "/chat/completions"
    http_request = request.Request(
        endpoint,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with request.urlopen(http_request, timeout=config.timeout) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc
    response_json = json.loads(raw)
    choices = response_json.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise RuntimeError("API response has no choices[0]")
    message = choices[0].get("message") or {}
    metadata = {
        "http_status": status,
        "model": response_json.get("model"),
        "finish_reason": choices[0].get("finish_reason"),
        "usage": response_json.get("usage"),
        "elapsed_seconds": round(time.time() - started, 3),
        "response_message": message,
    }
    return _content(message), metadata


def _units(dataset: Dataset, task: str, domain: str | None = None, limit: int | None = None) -> list[tuple[str, Trajectory, GoldCase | None]]:
    selected = [item for item in dataset.trajectories if domain in (None, "all", item.domain)]
    units: list[tuple[str, Trajectory, GoldCase | None]] = []
    for trajectory in selected:
        if task == "detect":
            units.append((trajectory.trajectory_id, trajectory, None))
        else:
            units.extend((case.trajectory_id + "::" + str(index), trajectory, case) for index, case in enumerate(trajectory.gold))
    return units if limit is None else units[:limit]


def _result_identity(task: str, trajectory: Trajectory, case: GoldCase | None) -> str:
    if task == "detect":
        return trajectory.trajectory_id
    return f"{trajectory.trajectory_id}::{list(case.redundant_step_idx) if case else ''}"


def run_task(
    dataset: Dataset,
    task: str,
    config: RunConfig,
    output_dir: str | Path,
    *,
    domain: str = "all",
    limit: int | None = None,
    include_policy: bool = True,
    max_predictions: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> Path:
    if task not in PARSERS:
        raise ValueError(f"Unknown task: {task}")
    units = _units(dataset, task, domain, limit)
    if not units:
        raise ValueError("No matching experiment units")
    output_dir = Path(output_dir)
    file_model = _safe_filename(config.model)
    model_dir = output_dir / file_model
    output_path = model_dir / f"{file_model}_{task}.json"
    if output_path.exists() and not resume and not dry_run:
        raise FileExistsError(f"Result already exists: {output_path}; use --resume or another --output-dir")
    dataset_counts = {}
    for trajectory in dataset.trajectories:
        dataset_counts[trajectory.dataset_id] = dataset_counts.get(trajectory.dataset_id, 0) + 1
    metadata = {
        "task": task,
        "model": config.model,
        "prompt_version": PROMPT_VERSION,
        "dataset_scope": {
            "datasets": sorted(dataset_counts),
            "trajectories": dataset_counts,
            "gold_cases": len(dataset.cases),
            "skipped_empty_trajectories": len(dataset.skipped_empty),
        },
        "domain": domain,
        "include_policy": include_policy,
        "max_predictions": max_predictions,
        "requested_units": len(units),
        "run_config": {
            "base_url": config.base_url,
            "api_key_env": config.api_key_env,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "max_tokens_param": config.max_tokens_param,
            "timeout": config.timeout,
            "reasoning_effort": config.reasoning_effort,
            "extra_body": config.extra_body or {},
        },
    }
    payload = {"metadata": metadata, "results": []}
    if resume and output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        previous = payload.get("metadata", {})
        protected_keys = (
            "task",
            "model",
            "prompt_version",
            "domain",
            "include_policy",
            "max_predictions",
            "run_config",
            "dataset_scope",
            "requested_units",
        )
        mismatches = [key for key in protected_keys if previous.get(key) != metadata.get(key)]
        if mismatches:
            raise ValueError(
                "Cannot resume with changed settings: " + ", ".join(mismatches)
            )
    if dry_run:
        preview = []
        for _, trajectory, case in units[:1]:
            system, user = build_prompt(task, trajectory, case, include_policy=include_policy, max_predictions=max_predictions)
            preview.append({"identity": _result_identity(task, trajectory, case), "system": system, "user": user, "prompt_sha256": _sha256(system + "\n\n" + user)})
        dry_path = model_dir / f"{file_model}_{task}.dry_run.json"
        _atomic_write(dry_path, {"metadata": payload["metadata"], "preview": preview})
        return dry_path

    import os
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {config.api_key_env} is not set")
    model_dir.mkdir(parents=True, exist_ok=True)
    response_log = model_dir / f"{file_model}_{task}.responses.json"
    response_records: list[dict[str, Any]] = []
    if response_log.exists():
        response_records = json.loads(response_log.read_text(encoding="utf-8"))
    elif payload.get("results"):
        response_records = [
            {
                "identity": item.get("identity"),
                "prompt_sha256": item.get("prompt_sha256"),
                "raw_response": item.get("raw_response"),
                "response_metadata": item.get("response_metadata"),
                "error": item.get("error"),
            }
            for item in payload["results"]
        ]
    # Resume retries failed requests and keeps successful responses.
    existing = {
        item.get("identity")
        for item in payload.get("results", [])
        if item.get("error") is None
    }
    parser: Callable[[str], dict[str, Any]] = PARSERS[task]
    for _, trajectory, case in units:
        identity = _result_identity(task, trajectory, case)
        if identity in existing:
            continue
        system, user = build_prompt(task, trajectory, case, include_policy=include_policy, max_predictions=max_predictions)
        result: dict[str, Any] = {
            "identity": identity,
            "trajectory_id": trajectory.trajectory_id,
            "dataset_id": trajectory.dataset_id,
            "source_id": trajectory.source_id,
            "sim_index": trajectory.sim_index,
            "domain": trajectory.domain,
            "task_id": trajectory.task_id,
            "prompt_sha256": _sha256(system + "\n\n" + user),
            "raw_response": None,
            "response_metadata": None,
            "error": None,
            "parse_error": None,
        }
        if task == "classify":
            result.update({"predicted_type": None, "format_failed": False})
        elif task == "retrieve":
            result.update({"predicted_type": None, "predicted_evidence_idx": None, "format_failed": False})
        else:
            result.update({"predictions": [], "parse_failed": False, "format_failed": False})
        if task == "detect":
            result["gold"] = [item.as_dict() for item in trajectory.gold]
        elif case is not None:
            result.update(case.as_dict())
            result["annotated_type"] = case.redundant_step_type
        try:
            raw_response, metadata = _request(config, system, user, api_key)
            result["raw_response"] = raw_response
            result["response_metadata"] = metadata
            if metadata.get("finish_reason") in ("length", "max_tokens"):
                result["unjudgeable"] = True
                result["unjudgeable_reason"] = "output_token_limit"
            else:
                parsed = parser(raw_response)
                result.update(parsed)
                if task == "detect" and parsed.get("predictions") is None:
                    result["parse_failed"] = True
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        payload["results"] = [
            item for item in payload.get("results", []) if item.get("identity") != identity
        ]
        response_records = [
            item for item in response_records if item.get("identity") != identity
        ]
        response_records.append({
            "identity": identity,
            "prompt_sha256": result["prompt_sha256"],
            "raw_response": result["raw_response"],
            "response_metadata": result["response_metadata"],
            "error": result["error"],
        })
        payload["results"].append(result)
        _atomic_write(output_path, payload)
        _atomic_write(response_log, response_records)
        existing.add(identity)
    return output_path
