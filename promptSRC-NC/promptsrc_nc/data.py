"""Standalone dataset preprocessing and split handling for PromptSRC-NC."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import KAGGLE_HANDLES, OFFICIAL_DATASET_SOURCES, canonical_dataset
from .structured_logging import append_jsonl, read_json, write_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

EUROSAT_CLASSNAMES = {
    "AnnualCrop": "Annual Crop Land",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings",
    "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land",
    "Residential": "Residential Buildings",
    "River": "River",
    "SeaLake": "Sea or Lake",
}

FLOWERS102_DOWNLOADS = {
    "images": (
        "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
        "52808999861908f626f3c1f4e79d11fa",
    ),
    "labels": (
        "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
        "e0620be6f572b9609742df49c70aed4d",
    ),
}

STANFORD_CARS_TEST_ANNOS_WITH_LABELS_DOWNLOADS = (
    (
        "http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
        "b0a2b23655a3edd16d84508592a98d10",
    ),
    (
        "https://raw.githubusercontent.com/jhpohovey/StanfordCars-Dataset/main/stanford_cars/cars_test_annos_withlabels.mat",
        "b0a2b23655a3edd16d84508592a98d10",
    ),
)


@dataclass(frozen=True)
class ManifestRecord:
    uid: str
    dataset: str
    image_path: str
    label_id: int
    class_name: str
    source_split: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ManifestRecord":
        return cls(
            uid=str(payload.get("uid", payload.get("sample_id"))),
            dataset=canonical_dataset(str(payload["dataset"])),
            image_path=str(payload["image_path"]),
            label_id=int(payload["label_id"]),
            class_name=str(payload["class_name"]),
            source_split=str(payload["source_split"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class SplitSpec:
    dataset: str
    protocol: str
    shots: int
    seed: int
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    unlabeled_ids: tuple[str, ...]
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
            "unlabeled_ids": list(self.unlabeled_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitSpec":
        return cls(
            dataset=canonical_dataset(str(payload["dataset"])),
            protocol=str(payload["protocol"]),
            shots=int(payload["shots"]),
            seed=int(payload["seed"]),
            train_ids=tuple(str(item) for item in payload["train_ids"]),
            val_ids=tuple(str(item) for item in payload["val_ids"]),
            test_ids=tuple(str(item) for item in payload["test_ids"]),
            unlabeled_ids=tuple(str(item) for item in payload.get("unlabeled_ids", ())),
            metadata=dict(payload.get("metadata", {})),
        )


def processed_dataset_dir(data_root: str | Path, dataset: str) -> Path:
    return Path(data_root) / "processed" / canonical_dataset(dataset)


def extracted_dataset_dir(data_root: str | Path, dataset: str) -> Path:
    return Path(data_root) / "extracted" / canonical_dataset(dataset)


def manifest_path(data_root: str | Path, dataset: str) -> Path:
    return processed_dataset_dir(data_root, dataset) / "manifest.jsonl"


def split_path(data_root: str | Path, dataset: str, protocol: str, shots: int, seed: int) -> Path:
    return processed_dataset_dir(data_root, dataset) / "splits" / protocol / f"shots_{shots}" / f"seed_{seed}.json"


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item.uid):
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
    return output


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(ManifestRecord.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid manifest line {line_number} in {path}") from exc
    return records


def write_split(split: SplitSpec, path: str | Path) -> Path:
    return write_json(path, split.to_dict())


def read_split(path: str | Path) -> SplitSpec:
    return SplitSpec.from_dict(read_json(path))


def validate_split_integrity(split: SplitSpec) -> None:
    split_lists = {
        "train": tuple(split.train_ids),
        "val": tuple(split.val_ids),
        "test": tuple(split.test_ids),
        "unlabeled": tuple(split.unlabeled_ids),
    }
    for split_name, ids in split_lists.items():
        if len(set(ids)) != len(ids):
            raise ValueError(f"Split {split.dataset}/{split.shots}/seed{split.seed} has duplicate IDs in {split_name}")
    split_sets = {
        name: set(ids) for name, ids in split_lists.items()
    }
    split_items = list(split_sets.items())
    for left_index, (left_name, left_ids) in enumerate(split_items):
        for right_name, right_ids in split_items[left_index + 1 :]:
            overlap = left_ids & right_ids
            if overlap:
                first = sorted(overlap)[0]
                raise ValueError(f"Split {split.dataset}/{split.shots}/seed{split.seed} has {left_name}-{right_name} overlap at {first}")
    if split.metadata.get("uses_test_images_for_unlabeled") is True:
        raise ValueError("Primary PromptSRC-NC split forbids test images in the unlabeled pool")


def validate_split_against_manifest(records: Sequence[ManifestRecord], split: SplitSpec) -> None:
    by_id = {record.uid: record for record in records}
    missing = [
        uid
        for uid in (*split.train_ids, *split.val_ids, *split.test_ids, *split.unlabeled_ids)
        if uid not in by_id
    ]
    if missing:
        raise KeyError(f"{len(missing)} split IDs are missing from manifest; first missing id: {missing[0]}")

    val_sources = {"val", "valid", "validation"}
    for uid in split.train_ids:
        if by_id[uid].source_split != "train":
            raise ValueError(f"Few-shot train ID {uid} has source_split={by_id[uid].source_split!r}; expected 'train'")
    for uid in split.val_ids:
        if by_id[uid].source_split not in val_sources:
            raise ValueError(f"Validation ID {uid} has source_split={by_id[uid].source_split!r}; expected validation source")
    for uid in split.test_ids:
        if by_id[uid].source_split != "test":
            raise ValueError(f"Test ID {uid} has source_split={by_id[uid].source_split!r}; expected 'test'")
    for uid in split.unlabeled_ids:
        if by_id[uid].source_split != "train":
            raise ValueError(
                f"unlabeled pool ID {uid} has source_split={by_id[uid].source_split!r}; expected 'train'"
            )

    if split.protocol == "few_shot_all_classes":
        train_ids = set(split.train_ids)
        expected_unlabeled = tuple(
            record.uid
            for record in sorted((record for record in records if record.source_split == "train"), key=lambda item: item.uid)
            if record.uid not in train_ids
        )
        if tuple(split.unlabeled_ids) != expected_unlabeled:
            raise ValueError(
                "Split unlabeled pool must equal full training split minus few-shot labeled train IDs"
            )


def class_names_from_records(records: Iterable[ManifestRecord]) -> list[str]:
    by_label: dict[int, str] = {}
    for record in records:
        existing = by_label.get(record.label_id)
        if existing is not None and existing != record.class_name:
            raise ValueError(f"Label {record.label_id} has multiple class names: {existing!r}, {record.class_name!r}")
        by_label[record.label_id] = record.class_name
    if not by_label:
        raise ValueError("Manifest is empty")
    labels = sorted(by_label)
    expected = list(range(labels[-1] + 1))
    if labels != expected:
        raise ValueError(f"Labels must be contiguous from 0; found {labels[:5]} ... {labels[-5:]}")
    return [by_label[label] for label in expected]


def records_by_ids(records: Sequence[ManifestRecord], ids: Iterable[str]) -> list[ManifestRecord]:
    by_id = {record.uid: record for record in records}
    missing = [uid for uid in ids if uid not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} split ids are missing from manifest; first missing id: {missing[0]}")
    return [by_id[uid] for uid in ids]


def summarize_manifest(records: Sequence[ManifestRecord]) -> dict[str, Any]:
    split_counts: dict[str, int] = defaultdict(int)
    class_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    class_names: dict[int, str] = {}
    for record in records:
        split_counts[record.source_split] += 1
        class_counts[record.label_id][record.source_split] += 1
        class_names[record.label_id] = record.class_name
    return {
        "num_records": len(records),
        "num_classes": len(class_names),
        "source_split_counts": dict(sorted(split_counts.items())),
        "classes": [
            {
                "label_id": label,
                "class_name": class_names[label],
                "counts": dict(class_counts[label]),
                "total": sum(class_counts[label].values()),
            }
            for label in sorted(class_names)
        ],
    }


class PromptSRCNCDataset:
    """Lazy PIL image dataset over PromptSRC-NC manifest records."""

    def __init__(self, records: Iterable[ManifestRecord], transform: Callable | None = None) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for data loading") from exc

        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "img": image,
            "label": record.label_id,
            "uid": record.uid,
            "impath": record.image_path,
            "classname": record.class_name,
            "index": index,
        }


def build_data_loaders(
    train_records: Sequence[ManifestRecord],
    val_records: Sequence[ManifestRecord],
    test_records: Sequence[ManifestRecord],
    unlabeled_records: Sequence[ManifestRecord],
    train_transform: Callable,
    eval_transform: Callable,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
):
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("torch is required to build data loaders") from exc

    return {
        "train": DataLoader(
            PromptSRCNCDataset(train_records, train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            PromptSRCNCDataset(val_records, eval_transform),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            PromptSRCNCDataset(test_records, eval_transform),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "unlabeled": DataLoader(
            PromptSRCNCDataset(unlabeled_records, eval_transform),
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }


def _images_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _uid(dataset: str, root: Path, image_path: Path) -> str:
    try:
        rel = image_path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = image_path.name
    return f"{dataset}/{Path(rel).as_posix()}"


def _normalize_class_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip()


def _flowers_class_name(one_based_label: int, cat_to_name: Mapping[str, Any]) -> str:
    name = cat_to_name.get(str(one_based_label))
    if name is not None:
        return str(name)
    try:
        from torchvision.datasets import Flowers102

        return str(Flowers102.classes[one_based_label - 1])
    except Exception:
        return f"flower {one_based_label}"


def _mat_scalar(value: Any) -> Any:
    import numpy as np

    while isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected scalar MAT value, found shape {value.shape}")
        value = value.reshape(-1)[0]
    return value


def _has_direct_images(path: Path) -> bool:
    if not path.exists():
        return False
    return any(child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS for child in path.iterdir())


def _first_existing(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _image_dir_with_images(root: Path, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if _has_direct_images(root / candidate):
            return candidate
    for candidate in candidates:
        if _images_under(root / candidate):
            return candidate
    return candidates[0]


def validate_manifest_image_paths(records: Sequence[ManifestRecord], dataset: str) -> None:
    missing = [record for record in records if not Path(record.image_path).is_file()]
    if missing:
        first = missing[0]
        raise FileNotFoundError(
            f"Manifest for {canonical_dataset(dataset)} references missing image files; "
            f"found {len(missing)} missing paths, first uid={first.uid}, path={first.image_path}. "
            "Re-run prepare_data after fixing the dataset layout."
        )


def _load_zhou_split(dataset: str, root: Path, split_file: Path, path_prefix: Path) -> list[ManifestRecord]:
    if not split_file.exists():
        return []
    payload = read_json(split_file)
    records: list[ManifestRecord] = []
    for split_name in ("train", "val", "test"):
        for rel_path, label, class_name in payload[split_name]:
            image_path = (path_prefix / rel_path).resolve()
            records.append(
                ManifestRecord(
                    uid=_uid(dataset, root, image_path),
                    dataset=dataset,
                    image_path=str(image_path),
                    label_id=int(label),
                    class_name=str(class_name),
                    source_split=split_name,
                    metadata={"split_source": str(split_file), "source_layout": "zhou_promptsrc"},
                )
            )
    return sorted(records, key=lambda item: item.uid)


def _split_records_by_class(
    records: Sequence[ManifestRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    source: str,
) -> list[ManifestRecord]:
    grouped: dict[int, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label_id].append(record)
    rng = random.Random(seed)
    output: list[ManifestRecord] = []
    for label, items in sorted(grouped.items()):
        shuffled = sorted(items, key=lambda item: item.uid)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = round(n_total * train_ratio)
        n_val = round(n_total * val_ratio)
        if min(n_train, n_val, n_total - n_train - n_val) <= 0:
            raise ValueError(f"Class {label} has too few examples to split safely")
        splits = (
            ("train", shuffled[:n_train]),
            ("val", shuffled[n_train : n_train + n_val]),
            ("test", shuffled[n_train + n_val :]),
        )
        for split_name, split_items in splits:
            for record in split_items:
                output.append(
                    ManifestRecord(
                        uid=record.uid,
                        dataset=record.dataset,
                        image_path=record.image_path,
                        label_id=record.label_id,
                        class_name=record.class_name,
                        source_split=split_name,
                        metadata={**dict(record.metadata), "split_source": source},
                    )
                )
    return sorted(output, key=lambda item: item.uid)


def _trainval_test_records(
    trainval: Sequence[ManifestRecord],
    test: Sequence[ManifestRecord],
    val_ratio: float,
    seed: int,
    source: str,
) -> list[ManifestRecord]:
    grouped: dict[int, list[ManifestRecord]] = defaultdict(list)
    for record in trainval:
        grouped[record.label_id].append(record)
    rng = random.Random(seed)
    output: list[ManifestRecord] = []
    for label, items in sorted(grouped.items()):
        shuffled = sorted(items, key=lambda item: item.uid)
        rng.shuffle(shuffled)
        n_val = round(len(shuffled) * val_ratio)
        if n_val <= 0 or n_val >= len(shuffled):
            raise ValueError(f"Class {label} has too few trainval examples to split")
        for split_name, split_items in (("val", shuffled[:n_val]), ("train", shuffled[n_val:])):
            for record in split_items:
                output.append(
                    ManifestRecord(
                        uid=record.uid,
                        dataset=record.dataset,
                        image_path=record.image_path,
                        label_id=record.label_id,
                        class_name=record.class_name,
                        source_split=split_name,
                        metadata={**dict(record.metadata), "split_source": source},
                    )
                )
    for record in test:
        output.append(
            ManifestRecord(
                uid=record.uid,
                dataset=record.dataset,
                image_path=record.image_path,
                label_id=record.label_id,
                class_name=record.class_name,
                source_split="test",
                metadata={**dict(record.metadata), "split_source": source},
            )
        )
    return sorted(output, key=lambda item: item.uid)


def build_eurosat_manifest(root: Path) -> list[ManifestRecord]:
    dataset = "eurosat"
    zhou = _load_zhou_split(dataset, root, root / "split_zhou_EuroSAT.json", root / "2750")
    if zhou:
        return zhou
    rgb_root = next((candidate for candidate in (root / "2750", root / "EuroSAT", root) if _class_dirs(candidate)), None)
    if rgb_root is None:
        raise FileNotFoundError(f"Could not find EuroSAT RGB JPG folders under {root}")
    categories = sorted(path for path in rgb_root.iterdir() if path.is_dir() and path.name != "EuroSATallBands")
    records: list[ManifestRecord] = []
    for label, category_dir in enumerate(categories):
        class_name = EUROSAT_CLASSNAMES.get(category_dir.name, _normalize_class_name(category_dir.name))
        for image_path in _images_under(category_dir):
            records.append(
                ManifestRecord(
                    uid=_uid(dataset, root, image_path),
                    dataset=dataset,
                    image_path=str(image_path.resolve()),
                    label_id=label,
                    class_name=class_name,
                    source_split="all",
                    metadata={"source_layout": "eurosat_rgb_folder"},
                )
            )
    return _split_records_by_class(records, 0.5, 0.2, seed=1, source="promptsrc_style_50_20_30_from_rgb")


def _class_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [child for child in path.iterdir() if child.is_dir() and _images_under(child)]


def build_flowers_manifest(root: Path) -> list[ManifestRecord]:
    dataset = "oxford_flowers"
    zhou = _load_zhou_split(dataset, root, root / "split_zhou_OxfordFlowers.json", root / "jpg")
    if zhou:
        return zhou

    if (root / "jpg").exists() and (root / "imagelabels.mat").exists():
        from scipy.io import loadmat

        labels = loadmat(root / "imagelabels.mat")["labels"][0]
        cat_to_name_path = root / "cat_to_name.json"
        cat_to_name = read_json(cat_to_name_path) if cat_to_name_path.exists() else {}
        records = []
        for index, raw_label in enumerate(labels):
            one_based = int(raw_label)
            image_path = root / "jpg" / f"image_{index + 1:05d}.jpg"
            class_name = cat_to_name.get(str(one_based), f"flower {one_based}")
            records.append(
                ManifestRecord(
                    uid=_uid(dataset, root, image_path),
                    dataset=dataset,
                    image_path=str(image_path.resolve()),
                    label_id=one_based - 1,
                    class_name=class_name,
                    source_split="all",
                    metadata={"source_layout": "official_flowers102"},
                )
            )
        return _split_records_by_class(records, 0.5, 0.2, seed=1, source="promptsrc_style_50_20_30_from_labels")

    for test_root in (root / "dataset" / "test", root / "test"):
        if _has_direct_images(test_root) and not _class_dirs(test_root):
            raise ValueError(
                "Flowers102 source has an unlabeled Kaggle test directory; "
                "use the official VGG Flowers102 source or a class-labeled test layout"
            )

    split_roots = [
        (root / "dataset" / "train", "train"),
        (root / "dataset" / "valid", "val"),
        (root / "dataset" / "val", "val"),
        (root / "dataset" / "test", "test"),
        (root / "train", "train"),
        (root / "valid", "val"),
        (root / "val", "val"),
        (root / "test", "test"),
    ]
    cat_to_name_path = next((p for p in (root / "cat_to_name.json", root / "dataset" / "cat_to_name.json") if p.exists()), None)
    cat_to_name = read_json(cat_to_name_path) if cat_to_name_path else {}
    class_ids = sorted(
        {class_dir.name for split_root, _ in split_roots if split_root.exists() for class_dir in _class_dirs(split_root)},
        key=lambda item: int(item) if item.isdigit() else item,
    )
    if not class_ids:
        raise FileNotFoundError(f"Could not find Flowers102 official or Kaggle layout under {root}")
    label_map = {class_id: idx for idx, class_id in enumerate(class_ids)}
    records: list[ManifestRecord] = []
    for split_root, split_name in split_roots:
        if not split_root.exists():
            continue
        for class_dir in _class_dirs(split_root):
            label_id = label_map[class_dir.name]
            class_name = cat_to_name.get(class_dir.name, _normalize_class_name(class_dir.name))
            for image_path in _images_under(class_dir):
                records.append(
                    ManifestRecord(
                        uid=_uid(dataset, root, image_path),
                        dataset=dataset,
                        image_path=str(image_path.resolve()),
                        label_id=label_id,
                        class_name=class_name,
                        source_split=split_name,
                        metadata={"source_layout": "kaggle_flowers_train_valid_test"},
                    )
                )
    return sorted(records, key=lambda item: item.uid)


def build_stanford_cars_manifest(root: Path) -> list[ManifestRecord]:
    dataset = "stanford_cars"
    zhou = _load_zhou_split(dataset, root, root / "split_zhou_StanfordCars.json", root)
    if zhou:
        return zhou

    train_file = _first_existing(
        (
            root / "devkit" / "cars_train_annos.mat",
            root / "car_devkit" / "devkit" / "cars_train_annos.mat",
        )
    )
    test_file = _first_existing(
        (
            root / "cars_test_annos_withlabels.mat",
            root / "devkit" / "cars_test_annos_withlabels.mat",
            root / "car_devkit" / "devkit" / "cars_test_annos_withlabels.mat",
        )
    )
    meta_file = _first_existing(
        (
            root / "devkit" / "cars_meta.mat",
            root / "car_devkit" / "devkit" / "cars_meta.mat",
        )
    )
    if train_file.exists() and test_file.exists() and meta_file.exists():
        from scipy.io import loadmat

        class_meta = loadmat(meta_file)["class_names"][0]
        train_image_dir = _image_dir_with_images(root, ("cars_train", "cars_train/cars_train"))
        test_image_dir = _image_dir_with_images(root, ("cars_test", "cars_test/cars_test"))

        def class_name(label: int) -> str:
            raw = str(class_meta[label][0])
            parts = raw.split(" ")
            if len(parts) > 1:
                year = parts.pop(-1)
                parts.insert(0, year)
            return " ".join(parts)

        def read_annos(image_dir: str, anno_path: Path, split_name: str) -> list[ManifestRecord]:
            annotations = loadmat(anno_path)["annotations"][0]
            out = []
            for item in annotations:
                image_name = str(_mat_scalar(item["fname"]))
                label_id = int(_mat_scalar(item["class"])) - 1
                image_path = root / image_dir / image_name
                out.append(
                    ManifestRecord(
                        uid=_uid(dataset, root, image_path),
                        dataset=dataset,
                        image_path=str(image_path.resolve()),
                        label_id=label_id,
                        class_name=class_name(label_id),
                        source_split=split_name,
                        metadata={"source_layout": "stanford_cars_mat"},
                    )
                )
            return out

        trainval = read_annos(train_image_dir, train_file, "trainval")
        test = read_annos(test_image_dir, test_file, "test")
        return _trainval_test_records(trainval, test, val_ratio=0.2, seed=1, source="promptsrc_trainval_80_20")

    records: list[ManifestRecord] = []
    train_root = root / "cars_train"
    test_root = root / "cars_test"
    class_dirs = sorted({path.name for path in _class_dirs(train_root)})
    if not class_dirs:
        raise FileNotFoundError(f"Could not find Stanford Cars MAT or class-folder layout under {root}")
    label_map = {name: idx for idx, name in enumerate(class_dirs)}
    for split_root, split_name in ((train_root, "train"), (test_root, "test")):
        for class_dir in _class_dirs(split_root):
            for image_path in _images_under(class_dir):
                records.append(
                    ManifestRecord(
                        uid=_uid(dataset, root, image_path),
                        dataset=dataset,
                        image_path=str(image_path.resolve()),
                        label_id=label_map[class_dir.name],
                        class_name=_normalize_class_name(class_dir.name),
                        source_split=split_name,
                        metadata={"source_layout": "stanford_cars_class_folders"},
                    )
                )
    if not any(record.source_split == "val" for record in records):
        train = [record for record in records if record.source_split == "train"]
        test = [record for record in records if record.source_split == "test"]
        return _trainval_test_records(train, test, val_ratio=0.2, seed=1, source="class_folder_trainval_80_20")
    return sorted(records, key=lambda item: item.uid)


def build_manifest_for_dataset(data_root: str | Path, dataset: str) -> list[ManifestRecord]:
    dataset = canonical_dataset(dataset)
    root = extracted_dataset_dir(data_root, dataset)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    builders = {
        "eurosat": build_eurosat_manifest,
        "oxford_flowers": build_flowers_manifest,
        "stanford_cars": build_stanford_cars_manifest,
    }
    return builders[dataset](root)


def _group_by_label(records: Iterable[ManifestRecord]) -> dict[int, list[ManifestRecord]]:
    grouped: dict[int, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label_id].append(record)
    return {label: sorted(items, key=lambda item: item.uid) for label, items in grouped.items()}


def _sample_per_class(records: Sequence[ManifestRecord], shots: int, seed: int, allow_fewer: bool = False) -> list[ManifestRecord]:
    rng = random.Random(seed)
    selected: list[ManifestRecord] = []
    for label, items in _group_by_label(records).items():
        if len(items) < shots and not allow_fewer:
            raise ValueError(f"Class {label} has {len(items)} examples; requested {shots} shots")
        shuffled = list(items)
        rng.shuffle(shuffled)
        selected.extend(sorted(shuffled[: min(shots, len(shuffled))], key=lambda item: item.uid))
    return sorted(selected, key=lambda item: item.uid)


def make_few_shot_split(
    records: Sequence[ManifestRecord],
    dataset: str,
    shots: int,
    seed: int,
    protocol: str = "few_shot_all_classes",
) -> SplitSpec:
    if protocol != "few_shot_all_classes":
        raise ValueError("Only few_shot_all_classes is implemented for the primary PromptSRC-NC pipeline")
    dataset = canonical_dataset(dataset)
    train_pool = sorted([record for record in records if record.source_split == "train"], key=lambda item: item.uid)
    val_pool = sorted([record for record in records if record.source_split in {"val", "valid", "validation"}], key=lambda item: item.uid)
    test = sorted([record for record in records if record.source_split == "test"], key=lambda item: item.uid)
    if not train_pool or not val_pool or not test:
        raise ValueError(
            f"Manifest for {dataset} must contain train, val, and test records; "
            f"found train={len(train_pool)}, val={len(val_pool)}, test={len(test)}"
        )
    train = _sample_per_class(train_pool, shots=shots, seed=seed)
    val = _sample_per_class(val_pool, shots=min(shots, 4), seed=seed)
    train_ids = {record.uid for record in train}
    unlabeled = [record for record in train_pool if record.uid not in train_ids]
    class_names = {str(idx): name for idx, name in enumerate(class_names_from_records(records))}
    return SplitSpec(
        dataset=dataset,
        protocol=protocol,
        shots=shots,
        seed=seed,
        train_ids=tuple(record.uid for record in train),
        val_ids=tuple(record.uid for record in val),
        test_ids=tuple(record.uid for record in test),
        unlabeled_ids=tuple(record.uid for record in unlabeled),
        metadata={
            "num_classes": len(class_names),
            "class_names": class_names,
            "train_pool_size": len(train_pool),
            "unlabeled_policy": "full_training_split_minus_fewshot_labeled_train",
            "uses_test_images_for_unlabeled": False,
            "fewshot_val_shots": min(shots, 4),
        },
    )


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path, expected_md5: str | None = None) -> None:
    if destination.exists() and (expected_md5 is None or _md5(destination) == expected_md5):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f"{destination.name}.download")
    request = urllib.request.Request(url, headers={"User-Agent": "promptsrc-nc-data-prep/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60 * 30) as response, temp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from None
    if expected_md5 is not None:
        actual = _md5(temp_path)
        if actual != expected_md5:
            temp_path.unlink(missing_ok=True)
            raise ValueError(f"MD5 mismatch for {url}: expected {expected_md5}, found {actual}")
    temp_path.replace(destination)


def _safe_extract_tgz(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if member_path != destination_resolved and destination_resolved not in member_path.parents:
                raise ValueError(f"Unsafe path in archive {archive_path}: {member.name}")
        archive.extractall(destination)


def _flowers_cat_to_name() -> dict[str, str]:
    try:
        from torchvision.datasets import Flowers102

        return {str(index + 1): str(name) for index, name in enumerate(Flowers102.classes)}
    except Exception:
        return {}


def _is_official_flowers102_source(path: Path) -> bool:
    return (path / "jpg").is_dir() and (path / "imagelabels.mat").is_file()


def _prepare_official_flowers102_source(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "102flowers.tgz"
    images_url, images_md5 = FLOWERS102_DOWNLOADS["images"]
    labels_url, labels_md5 = FLOWERS102_DOWNLOADS["labels"]
    _download_file(images_url, archive_path, images_md5)
    _download_file(labels_url, destination / "imagelabels.mat", labels_md5)
    _safe_extract_tgz(archive_path, destination)
    cat_to_name = _flowers_cat_to_name()
    if cat_to_name:
        write_json(destination / "cat_to_name.json", cat_to_name)
    write_json(
        destination / "source_metadata.json",
        {
            "dataset": "oxford_flowers",
            "source": OFFICIAL_DATASET_SOURCES["oxford_flowers"],
            "files": sorted(FLOWERS102_DOWNLOADS),
            "split_source": "promptsrc_style_50_20_30_from_labels",
        },
    )
    if not _is_official_flowers102_source(destination):
        raise FileNotFoundError("Official Flowers102 download did not produce jpg/ and imagelabels.mat")


def _ensure_official_flowers102_downloaded(target: Path, download: bool) -> Path:
    if _is_official_flowers102_source(target):
        return target
    if not download:
        raise FileNotFoundError(
            f"{target} is missing the official Flowers102 layout; expected jpg/ and imagelabels.mat"
        )

    temp_target = target.with_name(f"{target.name}.official-download")
    if temp_target.exists():
        shutil.rmtree(temp_target)
    try:
        _prepare_official_flowers102_source(temp_target)
        if target.exists():
            shutil.rmtree(target)
        temp_target.replace(target)
    except Exception:
        if temp_target.exists():
            shutil.rmtree(temp_target)
        raise
    return target


def _has_stanford_cars_labeled_test_annos(target: Path) -> bool:
    return any(
        path.exists()
        for path in (
            target / "cars_test_annos_withlabels.mat",
            target / "devkit" / "cars_test_annos_withlabels.mat",
            target / "car_devkit" / "devkit" / "cars_test_annos_withlabels.mat",
        )
    )


def _ensure_stanford_cars_labeled_test_annos(target: Path, download: bool) -> None:
    if _has_stanford_cars_labeled_test_annos(target):
        return
    if not download:
        raise FileNotFoundError(
            f"{target} is missing cars_test_annos_withlabels.mat; labeled test annotations are required"
        )
    errors: list[str] = []
    for url, expected_md5 in STANFORD_CARS_TEST_ANNOS_WITH_LABELS_DOWNLOADS:
        try:
            _download_file(url, target / "cars_test_annos_withlabels.mat", expected_md5)
            return
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError(
        "Could not download Stanford Cars labeled test annotations from any configured source: "
        + " | ".join(errors)
    )


def ensure_dataset_downloaded(data_root: str | Path, dataset: str, download: bool = True) -> Path:
    dataset = canonical_dataset(dataset)
    target = extracted_dataset_dir(data_root, dataset)
    if dataset == "oxford_flowers":
        return _ensure_official_flowers102_downloaded(target, download)
    if target.exists() and any(target.iterdir()):
        if dataset == "stanford_cars":
            _ensure_stanford_cars_labeled_test_annos(target, download)
        return target
    if not download:
        raise FileNotFoundError(f"{target} is missing and download=False")
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub is required for Modal data preparation downloads") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    source = Path(kagglehub.dataset_download(KAGGLE_HANDLES[dataset]))
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    if dataset == "stanford_cars":
        _ensure_stanford_cars_labeled_test_annos(target, download)
    return target


def prepare_datasets(
    data_root: str | Path,
    datasets: Iterable[str],
    shots: Iterable[int] = (16,),
    seeds: Iterable[int] = (1, 2, 3),
    protocol: str = "few_shot_all_classes",
    download: bool = True,
    log_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for raw_name in datasets:
        dataset = canonical_dataset(raw_name)
        ensure_dataset_downloaded(data_root, dataset, download=download)
        records = build_manifest_for_dataset(data_root, dataset)
        validate_manifest_image_paths(records, dataset)
        mpath = manifest_path(data_root, dataset)
        write_manifest(records, mpath)
        split_files = []
        for shot in shots:
            for seed in seeds:
                split = make_few_shot_split(records, dataset, shots=int(shot), seed=int(seed), protocol=protocol)
                spath = split_path(data_root, dataset, protocol, int(shot), int(seed))
                write_split(split, spath)
                split_files.append(str(spath))
        summary = {
            "event": "data_prepared",
            "dataset": dataset,
            "manifest": str(mpath),
            "splits": split_files,
            "summary": summarize_manifest(records),
            "source": (
                f"official:{OFFICIAL_DATASET_SOURCES[dataset]}"
                if dataset in OFFICIAL_DATASET_SOURCES
                else f"kagglehub:{KAGGLE_HANDLES[dataset]}"
            ),
        }
        summaries.append(summary)
        if log_path is not None:
            append_jsonl(log_path, summary)
    return summaries


def load_split_records(
    data_root: str | Path,
    dataset: str,
    shots: int,
    seed: int,
    protocol: str = "few_shot_all_classes",
) -> tuple[list[ManifestRecord], list[ManifestRecord], list[ManifestRecord], list[ManifestRecord], SplitSpec, list[str]]:
    dataset = canonical_dataset(dataset)
    records = read_manifest(manifest_path(data_root, dataset))
    split = read_split(split_path(data_root, dataset, protocol, shots, seed))
    validate_split_integrity(split)
    validate_split_against_manifest(records, split)
    train = records_by_ids(records, split.train_ids)
    val = records_by_ids(records, split.val_ids)
    test = records_by_ids(records, split.test_ids)
    unlabeled = records_by_ids(records, split.unlabeled_ids)
    classnames = class_names_from_records(records)
    return train, val, test, unlabeled, split, classnames


def parse_int_list(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def parse_dataset_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [canonical_dataset(item.strip()) for item in value.split(",") if item.strip()]
    return [canonical_dataset(item) for item in value]


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare PromptSRC-NC manifests and few-shot splits.")
    parser.add_argument("--data-root", default="/vol/data")
    parser.add_argument("--datasets", default="eurosat,oxford_flowers,stanford_cars")
    parser.add_argument("--shots", default="16")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--log-path", default="")
    args = parser.parse_args(argv)

    summaries = prepare_datasets(
        data_root=args.data_root,
        datasets=parse_dataset_list(args.datasets),
        shots=parse_int_list(args.shots),
        seeds=parse_int_list(args.seeds),
        download=not args.no_download,
        log_path=args.log_path or None,
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
