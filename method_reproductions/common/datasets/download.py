"""KaggleHub download helpers."""

from __future__ import annotations

from pathlib import Path

from .paths import get_data_root, raw_marker_path
from .sources import DATASET_SOURCES, get_dataset_source, normalize_dataset_key


def download_dataset(dataset_key: str, data_root: str | Path | None = None) -> Path:
    """Download one dataset through kagglehub and record its cache path.

    kagglehub stores files in its own cache. To avoid copying large datasets
    into the repo, this function writes data/raw/<dataset>/SOURCE_PATH.txt.
    """

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Install project dependencies with "
            "`pip install -e .` before downloading datasets."
        ) from exc

    source = get_dataset_source(dataset_key)
    downloaded = Path(kagglehub.dataset_download(source.kaggle_handle)).expanduser().resolve()

    marker = raw_marker_path(source.key, data_root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{downloaded}\n", encoding="utf-8")
    return downloaded


def download_many(dataset_keys: list[str] | None = None, data_root: str | Path | None = None) -> dict[str, Path]:
    keys = [normalize_dataset_key(key) for key in dataset_keys] if dataset_keys else sorted(DATASET_SOURCES)
    return {key: download_dataset(key, data_root=data_root) for key in keys}
