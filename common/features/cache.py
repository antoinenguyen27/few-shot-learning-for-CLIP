"""Stable paths and metadata for image/text feature caches."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from common.datasets.paths import get_data_root


def slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def stable_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def feature_cache_dir(
    dataset: str,
    model_name: str,
    pretrained: str,
    protocol: str,
    shots: int,
    seed: int,
    data_root: str | Path | None = None,
) -> Path:
    return (
        get_data_root(data_root)
        / "cache"
        / "features"
        / slug(dataset)
        / slug(f"{model_name}-{pretrained}")
        / slug(protocol)
        / f"shots_{shots}"
        / f"seed_{seed}"
    )


def write_cache_metadata(cache_dir: str | Path, metadata: Mapping[str, Any]) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / "metadata.json"
    output.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output

