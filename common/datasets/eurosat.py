"""EuroSAT manifest builder."""

from __future__ import annotations

from pathlib import Path

from .file_utils import child_dirs, direct_image_files, relative_posix
from .types import ImageRecord


EUROSAT_CLASS_NAMES = {
    "AnnualCrop": "annual crop land",
    "Forest": "forest",
    "HerbaceousVegetation": "herbaceous vegetation",
    "Highway": "highway",
    "Industrial": "industrial buildings",
    "Pasture": "pasture land",
    "PermanentCrop": "permanent crop land",
    "Residential": "residential buildings",
    "River": "river",
    "SeaLake": "sea or lake",
}


def _find_rgb_root(raw_root: Path) -> Path:
    candidates = [raw_root / "EuroSAT", raw_root / "2750", raw_root]
    for candidate in candidates:
        names = {path.name for path in child_dirs(candidate)}
        if {"AnnualCrop", "Forest", "SeaLake"}.issubset(names):
            return candidate

    for candidate in sorted(path for path in raw_root.rglob("*") if path.is_dir()):
        names = {path.name for path in child_dirs(candidate)}
        if {"AnnualCrop", "Forest", "SeaLake"}.issubset(names):
            return candidate

    raise FileNotFoundError(
        f"Could not find EuroSAT RGB directory under {raw_root}. Expected EuroSAT/ or 2750/ "
        "with class folders such as AnnualCrop, Forest, and SeaLake."
    )


def build_manifest(raw_root: str | Path) -> list[ImageRecord]:
    raw_root = Path(raw_root).expanduser().resolve()
    rgb_root = _find_rgb_root(raw_root)
    class_dirs = [path for path in child_dirs(rgb_root) if direct_image_files(path)]
    if not class_dirs:
        raise FileNotFoundError(f"No EuroSAT class image folders found under {rgb_root}")

    records: list[ImageRecord] = []
    for label_id, class_dir in enumerate(sorted(class_dirs, key=lambda path: path.name)):
        class_name = EUROSAT_CLASS_NAMES.get(class_dir.name, class_dir.name)
        for image_path in direct_image_files(class_dir):
            rel_path = relative_posix(image_path, raw_root)
            records.append(
                ImageRecord(
                    dataset="eurosat",
                    sample_id=f"eurosat/{Path(rel_path).with_suffix('').as_posix()}",
                    image_path=rel_path,
                    label_id=label_id,
                    class_name=class_name,
                    source_split="all",
                    metadata={"folder_class": class_dir.name},
                )
            )
    return sorted(records, key=lambda record: record.sample_id)

