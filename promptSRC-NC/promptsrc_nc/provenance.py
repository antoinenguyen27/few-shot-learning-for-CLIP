"""Provenance hashing and fail-closed artifact validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from .config import PromptSRCNCConfig, canonical_backbone, canonical_dataset
from .data import SplitSpec


UNLABELED_POLICY_TRAIN_REMAIN = "full_training_split_minus_fewshot_labeled_train"


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_ids_hash(ids: Iterable[str]) -> str:
    return stable_json_hash([str(item) for item in ids])


def split_hash(split: SplitSpec) -> str:
    return stable_json_hash(
        {
            "dataset": canonical_dataset(split.dataset),
            "protocol": split.protocol,
            "shots": int(split.shots),
            "seed": int(split.seed),
            "train_ids": list(split.train_ids),
            "val_ids": list(split.val_ids),
            "test_ids": list(split.test_ids),
            "unlabeled_ids": list(split.unlabeled_ids),
        }
    )


def effective_unlabeled_ids(split: SplitSpec, config: PromptSRCNCConfig) -> tuple[str, ...]:
    ids = tuple(str(uid) for uid in split.unlabeled_ids)
    if config.max_unlabeled_images is None:
        return ids
    return ids[: config.max_unlabeled_images]


def checkpoint_provenance(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "method",
        "stage",
        "checkpoint_role",
        "dataset",
        "shots",
        "seed",
        "protocol",
        "backbone",
        "pretrained",
    )
    return {key: checkpoint.get(key) for key in keys}


def _expect_equal(actual: Any, expected: Any, field: str, artifact: str) -> None:
    if actual != expected:
        raise ValueError(f"{artifact} has {field}={actual!r}; expected {expected!r}")


def validate_checkpoint_for_config(
    checkpoint: Mapping[str, Any],
    config: PromptSRCNCConfig,
    *,
    artifact: str = "checkpoint",
    expected_method: str | None = None,
    expected_stage: str | None = None,
    expected_role: str | None = None,
    split: SplitSpec | None = None,
) -> None:
    if expected_method is not None:
        _expect_equal(checkpoint.get("method"), expected_method, "method", artifact)
    if expected_stage is not None:
        _expect_equal(checkpoint.get("stage"), expected_stage, "stage", artifact)
    if expected_role is not None:
        _expect_equal(checkpoint.get("checkpoint_role"), expected_role, "checkpoint_role", artifact)

    expected_fields = {
        "dataset": config.dataset,
        "shots": config.shots,
        "seed": config.seed,
        "protocol": config.protocol,
        "backbone": config.backbone,
        "pretrained": config.pretrained,
    }
    for field, expected in expected_fields.items():
        _expect_equal(checkpoint.get(field), expected, field, artifact)

    if split is None:
        return

    metadata = dict(checkpoint.get("metadata") or {})
    expected_split_hash = split_hash(split)
    actual_split_hash = metadata.get("split_hash")
    if actual_split_hash is None and "split" in metadata:
        actual_split_hash = split_hash(SplitSpec.from_dict(metadata["split"]))
    if actual_split_hash is None:
        raise ValueError(f"{artifact} is missing split_hash provenance")
    _expect_equal(actual_split_hash, expected_split_hash, "split_hash", artifact)


def validate_neighbor_metadata(
    metadata: Mapping[str, Any],
    config: PromptSRCNCConfig,
    split: SplitSpec,
    *,
    artifact: str = "neighbor metadata",
) -> None:
    expected_unlabeled_ids = effective_unlabeled_ids(split, config)
    expected_fields = {
        "dataset": config.dataset,
        "protocol": config.protocol,
        "shots": config.shots,
        "seed": config.seed,
        "clip_backbone": config.backbone,
        "pretrained": config.pretrained,
        "unlabeled_policy": UNLABELED_POLICY_TRAIN_REMAIN,
        "uses_test_images_for_unlabeled": False,
        "num_unlabeled": len(expected_unlabeled_ids),
        "unlabeled_ids_hash": ordered_ids_hash(expected_unlabeled_ids),
        "split_hash": split_hash(split),
    }
    for field, expected in expected_fields.items():
        actual = metadata.get(field)
        if field == "clip_backbone" and actual is not None:
            actual = canonical_backbone(str(actual))
        _expect_equal(actual, expected, field, artifact)
    if config.max_unlabeled_images is not None:
        _expect_equal(metadata.get("max_unlabeled_images"), config.max_unlabeled_images, "max_unlabeled_images", artifact)
