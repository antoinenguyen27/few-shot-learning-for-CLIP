"""Persisted result records for standardized experiment logging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping


def result_jsonl_path(run_name: str, results_root: str | Path = "results") -> Path:
    """Return the canonical JSONL result path for a named run."""

    safe_name = run_name.strip().replace("/", "_")
    if not safe_name:
        raise ValueError("run_name must not be empty")
    return Path(results_root) / f"{safe_name}.jsonl"


@dataclass(frozen=True)
class RunResult:
    """One method/dataset/shot/seed evaluation record.

    Store both validation and test metrics. The first protocol should use
    validation metrics for model selection and test metrics for final reporting.
    """

    method: str
    dataset: str
    protocol: str
    model_name: str
    pretrained: str
    shots: int
    seed: int
    metrics: Mapping[str, float]
    split_path: str
    artifact_path: str | None = None
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunResult":
        return cls(
            method=str(payload["method"]),
            dataset=str(payload["dataset"]),
            protocol=str(payload["protocol"]),
            model_name=str(payload["model_name"]),
            pretrained=str(payload["pretrained"]),
            shots=int(payload["shots"]),
            seed=int(payload["seed"]),
            metrics={str(key): float(value) for key, value in dict(payload["metrics"]).items()},
            split_path=str(payload["split_path"]),
            artifact_path=str(payload["artifact_path"]) if payload.get("artifact_path") is not None else None,
            notes=str(payload.get("notes", "")),
            extra=dict(payload.get("extra", {})),
            created_at=str(payload.get("created_at", datetime.now(timezone.utc).isoformat())),
        )


def append_result(result: RunResult, path: str | Path) -> Path:
    """Append one JSONL result record."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_dict(), sort_keys=True))
        handle.write("\n")
    return output


def read_results(path: str | Path) -> list[RunResult]:
    records: list[RunResult] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            records.append(RunResult.from_dict(payload))
    return records


def summarize_results(
    results: Iterable[RunResult],
    metric_name: str = "test/top1_accuracy",
) -> list[dict[str, Any]]:
    """Aggregate a metric across seeds for each method/dataset/shot group."""

    grouped: dict[tuple[str, str, str, str, str, int], list[float]] = {}
    for result in results:
        if metric_name not in result.metrics:
            continue
        key = (
            result.method,
            result.dataset,
            result.protocol,
            result.model_name,
            result.pretrained,
            result.shots,
        )
        grouped.setdefault(key, []).append(result.metrics[metric_name])

    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        method, dataset, protocol, model_name, pretrained, shots = key
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "protocol": protocol,
                "model_name": model_name,
                "pretrained": pretrained,
                "shots": shots,
                "metric": metric_name,
                "num_seeds": len(values),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return rows
