"""Load the two released data layouts into one experiment-facing structure.

The human subset is joined by ``task_id``.  The model-generated subset is
joined by ``trajectory_id``; its ``task_id`` is metadata only because one task
can have several trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


DOMAINS = ("airline", "retail", "telecom")
VALID_TYPES = (
    "Duplicated Step",
    "Abnormal Step",
    "Incorrect Step",
    "Exploratory Step",
)


class DataValidationError(ValueError):
    """Raised when a released trajectory or annotation is inconsistent."""


@dataclass(frozen=True)
class GoldCase:
    trajectory_id: str
    domain: str
    task_id: str
    redundant_step_idx: tuple[int, ...]
    redundant_step_type: str
    evidence_step_idx: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "domain": self.domain,
            "task_id": self.task_id,
            "redundant_step_idx": list(self.redundant_step_idx),
            "redundant_step_type": self.redundant_step_type,
            "evidence_step_idx": list(self.evidence_step_idx),
        }


@dataclass(frozen=True)
class Trajectory:
    dataset_id: str
    trajectory_id: str
    source_id: str | None
    sim_index: int | None
    domain: str
    task_id: str
    messages: tuple[dict[str, Any], ...]
    policy: str
    gold: tuple[GoldCase, ...]

    def as_prompt_simulation(self) -> dict[str, Any]:
        return {"messages": list(self.messages), "policy": self.policy}


@dataclass(frozen=True)
class Dataset:
    trajectories: tuple[Trajectory, ...]
    skipped_empty: tuple[str, ...]

    @property
    def cases(self) -> tuple[GoldCase, ...]:
        return tuple(case for trajectory in self.trajectories for case in trajectory.gold)

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(sorted({item.domain for item in self.trajectories}))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataValidationError(f"Missing data file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"Invalid JSON in {path}: {exc}") from exc


def _index(value: Any) -> bool:
    return type(value) is int and value >= 0


def normalize_ref(value: Any, *, evidence: bool = False) -> tuple[int, ...] | None:
    """Normalize one step/evidence reference using the frozen benchmark contract."""
    if not isinstance(value, list):
        return None
    if evidence and value == [-1]:
        return (-1,)
    if len(value) == 1 and _index(value[0]):
        return (value[0],) if evidence else None
    if len(value) == 2 and all(_index(item) for item in value):
        return tuple(value)
    if evidence and len(value) > 2 and len(value) % 2 == 0 and all(_index(item) for item in value):
        pairs = [tuple(value[pos : pos + 2]) for pos in range(0, len(value), 2)]
        if len(set(pairs)) == len(pairs) and all(left < right for left, right in pairs):
            return tuple(index for pair in sorted(pairs) for index in pair)
    return None


def _tool_interactions(messages: list[dict[str, Any]], actors: set[str]) -> set[tuple[int, int]]:
    results: dict[str, list[int]] = {}
    for index, message in enumerate(messages):
        if message.get("role") == "tool" and message.get("id") is not None:
            results.setdefault(str(message["id"]), []).append(index)
    used: set[int] = set()
    pairs: set[tuple[int, int]] = set()
    for call_index, message in enumerate(messages):
        if message.get("role") not in actors:
            continue
        for call in message.get("tool_calls") or []:
            call_id = str(call.get("id"))
            candidates = [
                index for index in results.get(call_id, [])
                if index > call_index and index not in used
            ]
            if not candidates:
                raise DataValidationError(
                    f"Unpaired tool call {call_id!r} at message {call_index}"
                )
            result_index = min(candidates)
            used.add(result_index)
            pairs.add((call_index, result_index))
    return pairs


def _validate_reference(
    messages: list[dict[str, Any]],
    reference: tuple[int, ...],
    *,
    field: str,
    assistant_only: bool,
) -> None:
    if reference == (-1,):
        if field != "evidence_step_idx":
            raise DataValidationError(f"{field} cannot be [-1]")
        return
    if any(index >= len(messages) for index in reference):
        raise DataValidationError(f"{field} points outside the message list")
    if len(reference) == 1:
        message = messages[reference[0]]
        # Singleton evidence may be a text message or a standalone tool result.
        if field != "evidence_step_idx" or not message.get("content"):
            raise DataValidationError(f"Invalid singleton {field} reference")
        return
    if len(reference) > 2:
        for pos in range(0, len(reference), 2):
            _validate_reference(
                messages,
                reference[pos : pos + 2],
                field=field,
                assistant_only=assistant_only,
            )
        return
    if assistant_only:
        pairs = _tool_interactions(messages, {"assistant"})
        if reference not in pairs:
            raise DataValidationError(f"Invalid {field} pair {list(reference)}")
        return
    # Some source evidence pairs contain two tool results; preserve their indices.
    if reference[0] >= reference[1]:
        raise DataValidationError(f"Invalid evidence order {list(reference)}")


def _parse_cases(
    *,
    trajectory_id: str,
    domain: str,
    task_id: str,
    annotation: dict[str, Any],
    messages: list[dict[str, Any]],
) -> tuple[GoldCase, ...]:
    fields = (
        annotation.get("redundant_step_idx"),
        annotation.get("redundant_step_type"),
        annotation.get("evidence_step_idx"),
    )
    if not all(isinstance(value, list) for value in fields):
        raise DataValidationError(f"Invalid annotation arrays for {trajectory_id}")
    if len({len(value) for value in fields}) != 1:
        raise DataValidationError(f"Annotation arrays are not aligned for {trajectory_id}")
    cases: list[GoldCase] = []
    seen_steps: set[tuple[int, ...]] = set()
    for position in range(len(fields[0])):
        step = normalize_ref(fields[0][position], evidence=False)
        evidence = normalize_ref(fields[2][position], evidence=True)
        kind = fields[1][position]
        if step is None or evidence is None or kind not in VALID_TYPES:
            raise DataValidationError(f"Invalid Gold entry {position} for {trajectory_id}")
        if step in seen_steps:
            raise DataValidationError(f"Duplicate Gold step for {trajectory_id}")
        _validate_reference(messages, step, field="redundant_step_idx", assistant_only=True)
        _validate_reference(messages, evidence, field="evidence_step_idx", assistant_only=False)
        seen_steps.add(step)
        cases.append(GoldCase(trajectory_id, domain, task_id, step, kind, evidence))
    return tuple(cases)


def _load_domain(root: Path, dataset_id: str, domain: str) -> tuple[list[Trajectory], list[str]]:
    domain_root = root / dataset_id / domain
    traces = _read_json(domain_root / "final_traces.json")
    annotations = _read_json(domain_root / "annotation.json")
    simulations = traces.get("simulations") if isinstance(traces, dict) else None
    if not isinstance(simulations, list) or not isinstance(annotations, list):
        raise DataValidationError(f"Invalid domain files under {domain_root}")

    automated = dataset_id == "automated_annotation"
    simulation_by_id: dict[str, dict[str, Any]] = {}
    for position, simulation in enumerate(simulations):
        if not isinstance(simulation, dict) or not isinstance(simulation.get("messages"), list):
            raise DataValidationError(f"Invalid simulation at {domain_root} position {position}")
        if automated:
            identifier = simulation.get("trajectory_id")
            if not isinstance(identifier, str):
                raise DataValidationError(f"Missing trajectory_id in {domain_root}")
        else:
            task_id = simulation.get("task_id")
            if task_id is None:
                raise DataValidationError(f"Missing task_id in {domain_root}")
            identifier = str(task_id)
        if identifier in simulation_by_id:
            raise DataValidationError(f"Duplicate simulation identity {identifier}")
        simulation_by_id[identifier] = {"position": position, **simulation}

    annotation_by_id: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise DataValidationError(f"Invalid annotation in {domain_root}")
        identifier = annotation.get("trajectory_id") if automated else annotation.get("task_id")
        if identifier is None:
            raise DataValidationError(f"Annotation without identity in {domain_root}")
        identifier = str(identifier)
        if identifier in annotation_by_id:
            raise DataValidationError(f"Duplicate annotation identity {identifier}")
        annotation_by_id[identifier] = annotation

    if set(simulation_by_id) != set(annotation_by_id):
        raise DataValidationError(f"Simulation/annotation identities differ in {domain_root}")

    loaded: list[Trajectory] = []
    skipped_empty: list[str] = []
    for identifier, simulation in simulation_by_id.items():
        annotation = annotation_by_id[identifier]
        task_id = str(simulation.get("task_id", annotation.get("task_id")))
        if str(annotation.get("task_id", task_id)) != task_id:
            raise DataValidationError(f"Task identity mismatch for {identifier}")
        policy = simulation.get("policy")
        if not isinstance(policy, str) or not policy.strip():
            raise DataValidationError(f"Missing policy for {identifier}")
        trajectory_id = identifier if automated else f"human_annotation:{domain}:{task_id}"
        source_id = str(simulation.get("source_id")) if simulation.get("source_id") is not None else None
        sim_index = simulation.get("sim_index", simulation.get("position"))
        if sim_index is not None and type(sim_index) is not int:
            raise DataValidationError(f"Invalid sim_index for {trajectory_id}")
        messages = simulation["messages"]
        cases = _parse_cases(
            trajectory_id=trajectory_id,
            domain=domain,
            task_id=task_id,
            annotation=annotation,
            messages=messages,
        )
        if not cases:
            skipped_empty.append(trajectory_id)
            continue
        loaded.append(
            Trajectory(
                dataset_id=dataset_id,
                trajectory_id=trajectory_id,
                source_id=source_id,
                sim_index=sim_index,
                domain=domain,
                task_id=task_id,
                messages=tuple(messages),
                policy=policy,
                gold=cases,
            )
        )
    return loaded, skipped_empty


def load_dataset(root: str | Path, datasets: Iterable[str] = ("human_annotation", "automated_annotation")) -> Dataset:
    """Load positive trajectories from the two released subsets.

    Empty Gold annotations are skipped deliberately because the released main
    experiment is the 189 + 1,731 positive-trajectory setting.
    """
    root = Path(root).expanduser().resolve()
    trajectories: list[Trajectory] = []
    skipped: list[str] = []
    for dataset_id in datasets:
        if dataset_id not in {"human_annotation", "automated_annotation"}:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        for domain in DOMAINS:
            loaded, empty = _load_domain(root, dataset_id, domain)
            trajectories.extend(loaded)
            skipped.extend(empty)
    identities = [item.trajectory_id for item in trajectories]
    if len(set(identities)) != len(identities):
        raise DataValidationError("Duplicate trajectory_id across datasets")
    return Dataset(tuple(trajectories), tuple(skipped))
