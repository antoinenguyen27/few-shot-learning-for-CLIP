"""Stanford Cars manifest builder.

The selected Kaggle mirror should resemble the official Stanford Cars release.
This adapter supports the official .mat annotations and common CSV fallbacks.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .file_utils import find_first_existing, relative_posix
from .types import ImageRecord


def _scalar(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None and isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.size == 1:
            return _scalar(value.item())
        return [_scalar(item) for item in value.tolist()]
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _loadmat(path: Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("scipy is required to parse Stanford Cars .mat annotations.") from exc
    return loadmat(path, squeeze_me=True)


def _read_class_names_from_meta(meta_path: Path | None) -> list[str]:
    if meta_path is None or not meta_path.exists():
        return []
    payload = _loadmat(meta_path)
    raw_names = payload.get("class_names")
    if raw_names is None:
        return []
    if not isinstance(raw_names, (list, tuple)):
        try:
            raw_names = raw_names.tolist()
        except AttributeError:
            raw_names = [raw_names]
    return [str(_scalar(item)) for item in raw_names]


def _annotation_items(payload: dict[str, Any]) -> list[Any]:
    annotations = payload.get("annotations")
    if annotations is None:
        return []
    try:
        return list(annotations.flatten())
    except AttributeError:
        if isinstance(annotations, list):
            return annotations
        return [annotations]


def _annotation_field(annotation: Any, names: tuple[str, ...]) -> Any:
    dtype_names = getattr(getattr(annotation, "dtype", None), "names", None) or ()
    for name in names:
        if name in dtype_names:
            return annotation[name]
    if isinstance(annotation, dict):
        for name in names:
            if name in annotation:
                return annotation[name]
    for name in names:
        if hasattr(annotation, name):
            return getattr(annotation, name)
    raise KeyError(f"Annotation does not contain any of these fields: {names}")


def _resolve_image(raw_root: Path, fname: str, split: str) -> Path:
    candidates = [
        raw_root / fname,
        raw_root / f"cars_{split}" / fname,
        raw_root / f"cars_{split}" / f"cars_{split}" / fname,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(raw_root.rglob(Path(fname).name))
    if matches:
        return matches[0]
    return candidates[1]


def _class_name(class_names: list[str], class_index_one_based: int) -> str:
    index = class_index_one_based - 1
    if 0 <= index < len(class_names):
        return class_names[index]
    return str(class_index_one_based)


def _record_from_annotation(
    raw_root: Path,
    annotation: Any,
    split: str,
    class_names: list[str],
) -> ImageRecord:
    fname = str(_scalar(_annotation_field(annotation, ("fname", "relative_im_path", "image"))))
    class_index = int(_scalar(_annotation_field(annotation, ("class", "class_id", "label"))))
    image_path = _resolve_image(raw_root, fname, split)
    rel_path = relative_posix(image_path, raw_root)
    return ImageRecord(
        dataset="stanford_cars",
        sample_id=f"stanford_cars/{Path(rel_path).with_suffix('').as_posix()}",
        image_path=rel_path,
        label_id=class_index - 1,
        class_name=_class_name(class_names, class_index),
        source_split=split,
        metadata={"annotation_file_name": fname},
    )


def _parse_official_mat_layout(raw_root: Path) -> list[ImageRecord]:
    meta = find_first_existing(raw_root, ("cars_meta.mat",))
    class_names = _read_class_names_from_meta(meta)
    records: list[ImageRecord] = []

    all_annos = find_first_existing(raw_root, ("cars_annos.mat",))
    if all_annos is not None:
        payload = _loadmat(all_annos)
        if "class_names" in payload and not class_names:
            raw_names = payload["class_names"]
            try:
                raw_names = raw_names.tolist()
            except AttributeError:
                raw_names = [raw_names]
            class_names = [str(_scalar(item)) for item in raw_names]
        for annotation in _annotation_items(payload):
            test_flag = int(_scalar(_annotation_field(annotation, ("test",))))
            split = "test" if test_flag else "train"
            records.append(_record_from_annotation(raw_root, annotation, split, class_names))
        return records

    train_annos = find_first_existing(raw_root, ("cars_train_annos.mat",))
    test_annos = find_first_existing(raw_root, ("cars_test_annos_withlabels.mat",))
    for annos_path, split in ((train_annos, "train"), (test_annos, "test")):
        if annos_path is None:
            continue
        payload = _loadmat(annos_path)
        for annotation in _annotation_items(payload):
            records.append(_record_from_annotation(raw_root, annotation, split, class_names))
    return records


def _parse_csv_layout(raw_root: Path) -> list[ImageRecord]:
    csv_files = sorted(raw_root.rglob("*.csv"))
    annotation_files = [path for path in csv_files if "anno" in path.name.lower() or "label" in path.name.lower()]
    records: list[ImageRecord] = []
    class_names: dict[int, str] = {}

    for csv_path in annotation_files:
        split = "test" if "test" in csv_path.name.lower() else "train"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            rows = list(reader)
            lower_fields = {field.lower(): field for field in reader.fieldnames}
            path_field = next(
                (lower_fields[name] for name in ("fname", "filename", "image", "image_path", "path") if name in lower_fields),
                None,
            )
            label_field = next(
                (lower_fields[name] for name in ("class", "class_id", "target", "label") if name in lower_fields),
                None,
            )
            name_field = next(
                (lower_fields[name] for name in ("class_name", "name", "model") if name in lower_fields),
                None,
            )
            if path_field is None or label_field is None:
                continue
            raw_labels = [int(float(row[label_field])) for row in rows]
            labels_are_zero_based = 0 in raw_labels
            for row, raw_label in zip(rows, raw_labels):
                fname = row[path_field]
                class_index = raw_label + 1 if labels_are_zero_based else raw_label
                if name_field and row.get(name_field):
                    class_names[class_index] = row[name_field]
                image_path = _resolve_image(raw_root, fname, split)
                rel_path = relative_posix(image_path, raw_root)
                records.append(
                    ImageRecord(
                        dataset="stanford_cars",
                        sample_id=f"stanford_cars/{Path(rel_path).with_suffix('').as_posix()}",
                        image_path=rel_path,
                        label_id=class_index - 1,
                        class_name=class_names.get(class_index, str(class_index)),
                        source_split=split,
                        metadata={"annotation_file": relative_posix(csv_path, raw_root)},
                    )
                )
    return records


def build_manifest(raw_root: str | Path) -> list[ImageRecord]:
    raw_root = Path(raw_root).expanduser().resolve()
    records = _parse_official_mat_layout(raw_root)
    if not records:
        records = _parse_csv_layout(raw_root)
    if not records:
        raise FileNotFoundError(
            f"Could not parse Stanford Cars annotations under {raw_root}. Expected official "
            ".mat files such as devkit/cars_train_annos.mat and cars_test_annos_withlabels.mat, "
            "or CSV annotation files with filename and class columns."
        )
    return sorted(records, key=lambda record: record.sample_id)
