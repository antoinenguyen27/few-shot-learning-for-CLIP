"""Shared dataset data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class DatasetSource:
    """External source and layout notes for a dataset."""

    key: str
    display_name: str
    kaggle_handle: str
    expected_layout: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class ImageRecord:
    """One image example in a normalized dataset manifest.

    image_path is stored relative to the resolved raw dataset root whenever
    possible. This keeps generated manifests portable across machines.
    """

    dataset: str
    sample_id: str
    image_path: str
    label_id: int
    class_name: str
    source_split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ImageRecord":
        return cls(
            dataset=str(payload["dataset"]),
            sample_id=str(payload["sample_id"]),
            image_path=str(payload["image_path"]),
            label_id=int(payload["label_id"]),
            class_name=str(payload["class_name"]),
            source_split=str(payload["source_split"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class SplitSpec:
    """A deterministic few-shot split over a manifest."""

    dataset: str
    protocol: str
    shots: int
    seed: int
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "protocol": self.protocol,
            "shots": self.shots,
            "seed": self.seed,
            "train_ids": list(self.train_ids),
            "val_ids": list(self.val_ids),
            "test_ids": list(self.test_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitSpec":
        return cls(
            dataset=str(payload["dataset"]),
            protocol=str(payload["protocol"]),
            shots=int(payload["shots"]),
            seed=int(payload["seed"]),
            train_ids=tuple(str(item) for item in payload["train_ids"]),
            val_ids=tuple(str(item) for item in payload["val_ids"]),
            test_ids=tuple(str(item) for item in payload["test_ids"]),
            metadata=dict(payload.get("metadata", {})),
        )

