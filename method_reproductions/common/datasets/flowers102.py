"""Oxford Flowers 102 manifest builder for the selected Kaggle dataset."""

from __future__ import annotations

import json
from pathlib import Path

from .file_utils import child_dirs, direct_image_files, find_first_existing, relative_posix
from .types import ImageRecord


SPLIT_NAMES = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "validation": "val",
    "test": "test",
}


def _find_split_root(raw_root: Path) -> Path:
    candidates = [raw_root / "dataset", raw_root]
    for candidate in candidates:
        if (candidate / "train").is_dir() and ((candidate / "valid").is_dir() or (candidate / "val").is_dir()):
            return candidate

    for candidate in sorted(path for path in raw_root.rglob("*") if path.is_dir()):
        if (candidate / "train").is_dir() and ((candidate / "valid").is_dir() or (candidate / "val").is_dir()):
            return candidate

    raise FileNotFoundError(
        f"Could not find Flowers102 split root under {raw_root}. Expected dataset/train, "
        "dataset/valid, and dataset/test with numeric class folders."
    )


def _load_class_names(raw_root: Path) -> dict[str, str]:
    cat_file = find_first_existing(raw_root, ("cat_to_name.json",))
    if cat_file is None:
        return {}
    with cat_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): str(value) for key, value in payload.items()}


def _class_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return 10_000, path.name


def build_manifest(raw_root: str | Path) -> list[ImageRecord]:
    raw_root = Path(raw_root).expanduser().resolve()
    split_root = _find_split_root(raw_root)
    class_names = _load_class_names(raw_root)

    all_class_ids = sorted(
        {class_dir.name for split in SPLIT_NAMES for class_dir in child_dirs(split_root / split)},
        key=lambda item: (int(item) if item.isdigit() else 10_000, item),
    )
    label_by_class_id = {class_id: idx for idx, class_id in enumerate(all_class_ids)}
    if not label_by_class_id:
        raise FileNotFoundError(f"No Flowers102 class folders found under {split_root}")

    records: list[ImageRecord] = []
    for split_dir_name, source_split in SPLIT_NAMES.items():
        split_dir = split_root / split_dir_name
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(child_dirs(split_dir), key=_class_sort_key):
            class_id = class_dir.name
            class_name = class_names.get(class_id, class_id)
            for image_path in direct_image_files(class_dir):
                rel_path = relative_posix(image_path, raw_root)
                records.append(
                    ImageRecord(
                        dataset="flowers102",
                        sample_id=f"flowers102/{Path(rel_path).with_suffix('').as_posix()}",
                        image_path=rel_path,
                        label_id=label_by_class_id[class_id],
                        class_name=class_name,
                        source_split=source_split,
                        metadata={"folder_class": class_id},
                    )
                )
    return sorted(records, key=lambda record: record.sample_id)

