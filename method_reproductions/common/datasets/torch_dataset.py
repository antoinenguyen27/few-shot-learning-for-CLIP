"""Torch Dataset wrapper over normalized manifests and split specs."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable

from .manifest import class_names_from_records, read_manifest
from .paths import manifest_path, resolve_raw_root, split_path
from .splits import read_split
from .types import ImageRecord


class ManifestImageDataset:
    """Lazy image dataset backed by ImageRecord entries.

    output_format controls how each sample is returned:
    - "dict": {"image", "label", "sample_id", "class_name", "image_path"}
    - "tuple": (image, label)
    - "dassl": {"img", "label", "impath", "index"}

    The dict format is the repo default. The tuple and dassl options are only
    adapters for paper-code ports that expect those shapes.
    """

    def __init__(
        self,
        records: Iterable[ImageRecord],
        raw_root: str | Path,
        transform: Callable | None = None,
        output_format: str = "dict",
    ) -> None:
        self.records = list(records)
        self.raw_root = Path(raw_root).expanduser().resolve()
        self.transform = transform
        if output_format not in {"dict", "tuple", "dassl"}:
            raise ValueError("output_format must be one of: dict, tuple, dassl")
        self.output_format = output_format

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to load images.") from exc

        record = self.records[index]
        image_path = Path(record.image_path)
        if not image_path.is_absolute():
            image_path = self.raw_root / image_path
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.output_format == "tuple":
            return image, record.label_id
        if self.output_format == "dassl":
            return {
                "img": image,
                "label": record.label_id,
                "impath": str(image_path),
                "index": index,
            }
        return {
            "image": image,
            "label": record.label_id,
            "sample_id": record.sample_id,
            "class_name": record.class_name,
            "image_path": str(image_path),
        }


def records_by_ids(records: Iterable[ImageRecord], sample_ids: Iterable[str]) -> list[ImageRecord]:
    by_id = {record.sample_id: record for record in records}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"{len(missing)} split IDs were not found in the manifest: {preview}")
    return [by_id[sample_id] for sample_id in sample_ids]


@dataclass(frozen=True)
class SplitDatasets:
    """Train/validation/test datasets plus shared metadata for a split."""

    train: ManifestImageDataset
    val: ManifestImageDataset
    test: ManifestImageDataset
    classnames: list[str]
    raw_root: Path
    split_file: Path


def build_split_datasets(
    dataset: str,
    protocol: str,
    shots: int,
    seed: int,
    train_transform: Callable | None = None,
    eval_transform: Callable | None = None,
    data_root: str | Path | None = None,
    output_format: str = "dict",
) -> SplitDatasets:
    """Build train/val/test ManifestImageDataset objects for one split.

    This helper is the ergonomic path method owners should start with. It
    intentionally loads an existing manifest and split; it never samples data.
    """

    records = read_manifest(manifest_path(dataset, data_root=data_root))
    split_file = split_path(dataset, protocol, shots, seed, data_root=data_root)
    split = read_split(split_file)
    raw_root = resolve_raw_root(dataset, data_root=data_root)
    return SplitDatasets(
        train=ManifestImageDataset(
            records_by_ids(records, split.train_ids),
            raw_root,
            transform=train_transform,
            output_format=output_format,
        ),
        val=ManifestImageDataset(
            records_by_ids(records, split.val_ids),
            raw_root,
            transform=eval_transform,
            output_format=output_format,
        ),
        test=ManifestImageDataset(
            records_by_ids(records, split.test_ids),
            raw_root,
            transform=eval_transform,
            output_format=output_format,
        ),
        classnames=class_names_from_records(records),
        raw_root=raw_root,
        split_file=split_file,
    )


def build_data_loaders(
    split_datasets: SplitDatasets,
    batch_size: int,
    num_workers: int = 0,
    train_shuffle: bool = True,
):
    """Create torch DataLoaders for a SplitDatasets object."""

    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("torch is required to build data loaders.") from exc

    return {
        "train": DataLoader(
            split_datasets.train,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            split_datasets.val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            split_datasets.test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
