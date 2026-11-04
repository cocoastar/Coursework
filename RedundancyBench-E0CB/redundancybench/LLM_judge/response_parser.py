"""Task-specific parsing without scoring."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from ..data_loader import VALID_TYPES, normalize_ref


def _decode(text: str, expected: type) -> Any | None:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, expected) else None


def parse_classify(text: str) -> dict[str, Any]:
    value = _decode(text, dict)
    predicted = value.get("redundant_step_type") if value else None
    return {"predicted_type": predicted if predicted in VALID_TYPES else None, "format_failed": predicted not in VALID_TYPES}


def parse_retrieve(text: str) -> dict[str, Any]:
    value = _decode(text, dict)
    predicted_type = value.get("redundant_step_type") if value else None
    predicted_evidence = normalize_ref(value.get("evidence_step_idx"), evidence=True) if value else None
    return {
        "predicted_type": predicted_type if predicted_type in VALID_TYPES else None,
        "predicted_evidence_idx": list(predicted_evidence) if predicted_evidence is not None else None,
        "format_failed": value is None or predicted_type not in VALID_TYPES or predicted_evidence is None,
    }


def parse_detect(text: str) -> dict[str, Any]:
    value = _decode(text, list)
    if value is None:
        wrapper = _decode(text, dict)
        value = wrapper.get("predictions") if wrapper else None
    if not isinstance(value, list):
        return {"predictions": None, "format_failed": True}
    predictions: list[dict[str, Any]] = []
    for source_rank, item in enumerate(value, 1):
        if not isinstance(item, dict):
            predictions.append({"source_rank": source_rank, "valid": False, "confidence": None, "parse_error": "Prediction is not an object", "raw_item": item})
            continue
        step = normalize_ref(item.get("redundant_step_idx"), evidence=False)
        evidence = normalize_ref(item.get("evidence_step_idx"), evidence=True)
        kind = item.get("redundant_step_type")
        confidence = item.get("confidence")
        valid_confidence = isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and math.isfinite(confidence) and 0 <= confidence <= 1
        valid = step is not None and evidence is not None and kind in VALID_TYPES and valid_confidence
        predictions.append({
            "source_rank": source_rank,
            "redundant_step_idx": list(step) if step is not None else None,
            "evidence_step_idx": list(evidence) if evidence is not None else None,
            "redundant_step_type": kind,
            "confidence": float(confidence) if valid_confidence else None,
            "valid": valid,
            "parse_error": None if valid else "Invalid tuple fields or confidence",
            **({"raw_item": item} if not valid else {}),
        })
    # Unrankable entries take the first, non-scoring Top-K slots.
    predictions.sort(key=lambda item: item["confidence"] if item["confidence"] is not None else math.inf, reverse=True)
    seen_steps: set[tuple[int, ...]] = set()
    for rank, item in enumerate(predictions, 1):
        step = tuple(item["redundant_step_idx"]) if item.get("redundant_step_idx") is not None and item.get("confidence") is not None else None
        if step is not None:
            if step in seen_steps:
                item["duplicate"] = True
                item["valid"] = False
                item["parse_error"] = "Duplicate redundant step"
            seen_steps.add(step)
        item["rank"] = rank
    return {"predictions": predictions, "format_failed": any(not item["valid"] for item in predictions)}


PARSERS = {"classify": parse_classify, "retrieve": parse_retrieve, "detect": parse_detect}
