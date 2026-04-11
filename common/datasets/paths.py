"""Path helpers for data stored inside or outside the repository."""

from __future__ import annotations

import os
import re
from pathlib import Path

from .sources import get_dataset_source, normalize_dataset_key


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"


def env_key(dataset_key: str) -> str:
    normalized = normalize_dataset_key(dataset_key).upper()
    return re.sub(r"[^A-Z0-9]+", "_", normalized)


def get_data_root(data_root: str | Path | None = None) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    return Path(os.environ.get("FSL_CLIP_DATA", DEFAULT_DATA_ROOT)).expanduser().resolve()


def raw_marker_path(dataset_key: str, data_root: str | Path | None = None) -> Path:
    source = get_dataset_source(dataset_key)
    return get_data_root(data_root) / "raw" / source.key / "SOURCE_PATH.txt"


def resolve_raw_root(dataset_key: str, data_root: str | Path | None = None) -> Path:
    """Resolve the raw dataset path for a dataset.

    Priority:
    1. FSL_CLIP_<DATASET>_ROOT
    2. data/raw/<dataset>/SOURCE_PATH.txt written by download_data.py
    3. data/raw/<dataset>
    """

    source = get_dataset_source(dataset_key)
    env_name = f"FSL_CLIP_{env_key(source.key)}_ROOT"
    if os.environ.get(env_name):
        return Path(os.environ[env_name]).expanduser().resolve()

    marker = raw_marker_path(source.key, data_root)
    if marker.exists():
        stored = marker.read_text(encoding="utf-8").strip()
        if stored:
            return Path(stored).expanduser().resolve()

    return (get_data_root(data_root) / "raw" / source.key).resolve()


def manifest_path(dataset_key: str, data_root: str | Path | None = None) -> Path:
    source = get_dataset_source(dataset_key)
    return get_data_root(data_root) / "manifests" / f"{source.key}.jsonl"


def split_path(
    dataset_key: str,
    protocol: str,
    shots: int,
    seed: int,
    data_root: str | Path | None = None,
) -> Path:
    source = get_dataset_source(dataset_key)
    return (
        get_data_root(data_root)
        / "splits"
        / source.key
        / protocol
        / f"shots_{shots}"
        / f"seed_{seed}.json"
    )

