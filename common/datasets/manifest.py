"""Read, write, and summarize normalized dataset manifests."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .types import ImageRecord


def write_manifest(records: Iterable[ImageRecord], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: record.sample_id)
    with output.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
    return output


def read_manifest(path: str | Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            records.append(ImageRecord.from_dict(payload))
    return records


def summarize_records(records: Iterable[ImageRecord]) -> dict[str, object]:
    records = list(records)
    by_split = Counter(record.source_split for record in records)
    by_label: dict[int, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_label[record.label_id][record.source_split] += 1
    class_rows = [
        {
            "label_id": label_id,
            "class_name": next(record.class_name for record in records if record.label_id == label_id),
            "counts": dict(counter),
            "total": sum(counter.values()),
        }
        for label_id, counter in sorted(by_label.items())
    ]
    return {
        "num_records": len(records),
        "num_classes": len(by_label),
        "source_split_counts": dict(by_split),
        "classes": class_rows,
    }


def class_names_from_records(records: Iterable[ImageRecord]) -> list[str]:
    by_label: dict[int, str] = {}
    for record in records:
        existing = by_label.get(record.label_id)
        if existing is not None and existing != record.class_name:
            raise ValueError(
                f"Label {record.label_id} has inconsistent class names: {existing!r} and {record.class_name!r}"
            )
        by_label[record.label_id] = record.class_name
    if not by_label:
        return []
    expected = list(range(max(by_label) + 1))
    missing = [label for label in expected if label not in by_label]
    if missing:
        raise ValueError(f"Manifest labels are not contiguous; missing labels: {missing}")
    return [by_label[label] for label in expected]
