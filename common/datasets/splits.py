"""Deterministic few-shot split generation."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .types import ImageRecord, SplitSpec


class FewShotSplitError(ValueError):
    """Raised when a requested split is scientifically unsafe or impossible."""


def _group_by_label(records: Iterable[ImageRecord]) -> dict[int, list[ImageRecord]]:
    grouped: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label_id].append(record)
    return {label: sorted(items, key=lambda record: record.sample_id) for label, items in grouped.items()}


def _by_source_split(records: Iterable[ImageRecord]) -> dict[str, list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source_split.lower()].append(record)
    return {split: sorted(items, key=lambda record: record.sample_id) for split, items in grouped.items()}


def _take_ratio_by_class(
    records: list[ImageRecord],
    ratio: float,
    seed: int,
    min_per_class: int = 1,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    rng = random.Random(seed)
    selected: list[ImageRecord] = []
    remaining: list[ImageRecord] = []
    for label, class_records in _group_by_label(records).items():
        shuffled = list(class_records)
        rng.shuffle(shuffled)
        count = max(min_per_class, round(len(shuffled) * ratio))
        if count >= len(shuffled):
            raise FewShotSplitError(
                f"Cannot reserve {count} examples from class {label}; only {len(shuffled)} available."
            )
        selected.extend(sorted(shuffled[:count], key=lambda record: record.sample_id))
        remaining.extend(sorted(shuffled[count:], key=lambda record: record.sample_id))
    return sorted(selected, key=lambda record: record.sample_id), sorted(remaining, key=lambda record: record.sample_id)


def canonical_train_val_test(
    records: list[ImageRecord],
    base_split_seed: int = 0,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord], dict[str, object]]:
    """Return train pool, validation, and test records.

    If the raw dataset already provides train/val/test, use it. If it provides
    train/test only, carve validation from train. If it has no split, create a
    deterministic stratified train/val/test split.
    """

    by_split = _by_source_split(records)
    train = by_split.get("train", []) + by_split.get("training", [])
    val = by_split.get("val", []) + by_split.get("valid", []) + by_split.get("validation", [])
    test = by_split.get("test", [])

    if train and val and test:
        return train, val, test, {"base_split": "source_train_val_test"}

    if train and test and not val:
        val, train = _take_ratio_by_class(train, val_ratio, seed=base_split_seed)
        return train, val, test, {"base_split": "source_train_test_val_from_train", "val_ratio": val_ratio}

    if train and val and not test:
        test, train = _take_ratio_by_class(train, test_ratio, seed=base_split_seed + 10_000)
        return train, val, test, {"base_split": "source_train_val_test_from_train", "test_ratio": test_ratio}

    if train and not val and not test:
        test, train_and_val = _take_ratio_by_class(train, test_ratio, seed=base_split_seed + 10_000)
        adjusted_val_ratio = val_ratio / max(1.0 - test_ratio, 1e-8)
        val, train = _take_ratio_by_class(train_and_val, adjusted_val_ratio, seed=base_split_seed)
        return (
            train,
            val,
            test,
            {
                "base_split": "source_train_only_val_test_from_train",
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "adjusted_val_ratio_after_test_split": adjusted_val_ratio,
            },
        )

    if by_split.get("all"):
        all_records = by_split["all"]
        test, train_and_val = _take_ratio_by_class(all_records, test_ratio, seed=base_split_seed + 10_000)
        adjusted_val_ratio = val_ratio / max(1.0 - test_ratio, 1e-8)
        val, train = _take_ratio_by_class(train_and_val, adjusted_val_ratio, seed=base_split_seed)
        return (
            train,
            val,
            test,
            {
                "base_split": "stratified_from_all",
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "adjusted_val_ratio_after_test_split": adjusted_val_ratio,
            },
        )

    raise FewShotSplitError(
        "Could not derive train/val/test. Expected source_split values train/val/test, train/test, or all."
    )


def _sample_shots(
    train_pool: list[ImageRecord],
    shots: int,
    seed: int,
    allow_fewer: bool,
) -> list[ImageRecord]:
    if shots < 1:
        raise FewShotSplitError("shots must be >= 1")
    rng = random.Random(seed)
    selected: list[ImageRecord] = []
    for label, class_records in _group_by_label(train_pool).items():
        if len(class_records) < shots and not allow_fewer:
            raise FewShotSplitError(
                f"Class {label} has only {len(class_records)} train-pool examples; requested {shots} shots."
            )
        shuffled = list(class_records)
        rng.shuffle(shuffled)
        selected.extend(sorted(shuffled[: min(shots, len(shuffled))], key=lambda record: record.sample_id))
    return sorted(selected, key=lambda record: record.sample_id)


def make_few_shot_split(
    records: list[ImageRecord],
    dataset: str,
    shots: int,
    seed: int,
    protocol: str = "few_shot_all_classes",
    base_split_seed: int = 0,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    allow_fewer: bool = False,
) -> SplitSpec:
    if protocol != "few_shot_all_classes":
        raise FewShotSplitError(
            f"Unsupported protocol '{protocol}'. The implemented protocol is few_shot_all_classes."
        )
    train_pool, val, test, split_metadata = canonical_train_val_test(
        records,
        base_split_seed=base_split_seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train = _sample_shots(train_pool, shots=shots, seed=seed, allow_fewer=allow_fewer)
    class_names = {
        record.label_id: record.class_name for record in sorted(records, key=lambda item: (item.label_id, item.class_name))
    }
    return SplitSpec(
        dataset=dataset,
        protocol=protocol,
        shots=shots,
        seed=seed,
        train_ids=tuple(record.sample_id for record in train),
        val_ids=tuple(record.sample_id for record in val),
        test_ids=tuple(record.sample_id for record in test),
        metadata={
            **split_metadata,
            "base_split_seed": base_split_seed,
            "allow_fewer": allow_fewer,
            "num_classes": len(class_names),
            "class_names": {str(label): name for label, name in sorted(class_names.items())},
            "train_pool_size": len(train_pool),
        },
    )


def write_split(split: SplitSpec, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_split(path: str | Path) -> SplitSpec:
    with Path(path).open("r", encoding="utf-8") as handle:
        return SplitSpec.from_dict(json.load(handle))
