"""Dataset adapter registry."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import eurosat, flowers102, stanford_cars
from .sources import DATASET_SOURCES, get_dataset_source, normalize_dataset_key
from .types import ImageRecord


BUILDERS: dict[str, Callable[[str | Path], list[ImageRecord]]] = {
    "eurosat": eurosat.build_manifest,
    "flowers102": flowers102.build_manifest,
    "stanford_cars": stanford_cars.build_manifest,
}


def dataset_keys() -> list[str]:
    return sorted(DATASET_SOURCES)


def build_manifest(dataset_key: str, raw_root: str | Path) -> list[ImageRecord]:
    source = get_dataset_source(dataset_key)
    key = normalize_dataset_key(source.key)
    try:
        builder = BUILDERS[key]
    except KeyError as exc:
        raise KeyError(f"No manifest builder registered for dataset '{dataset_key}'") from exc
    return builder(raw_root)

