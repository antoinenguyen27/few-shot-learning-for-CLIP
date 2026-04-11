"""External dataset sources used by the project."""

from __future__ import annotations

from .types import DatasetSource


DATASET_SOURCES: dict[str, DatasetSource] = {
    "eurosat": DatasetSource(
        key="eurosat",
        display_name="EuroSAT",
        kaggle_handle="apollo2506/eurosat-dataset",
        expected_layout=(
            "EuroSAT/<class_name>/*.jpg",
            "EuroSATallBands/<class_name>/*.tif (not used by default)",
        ),
        notes="Use the RGB JPG EuroSAT directory for OpenCLIP experiments.",
    ),
    "flowers102": DatasetSource(
        key="flowers102",
        display_name="Oxford 102 Flowers",
        kaggle_handle="nunenuh/pytorch-challange-flower-dataset",
        expected_layout=(
            "dataset/train/<class_id>/*.jpg",
            "dataset/valid/<class_id>/*.jpg",
            "dataset/test/<class_id>/*.jpg",
            "cat_to_name.json",
        ),
        notes="Class folders are numeric IDs; cat_to_name.json maps IDs to readable names.",
    ),
    "stanford_cars": DatasetSource(
        key="stanford_cars",
        display_name="Stanford Cars",
        kaggle_handle="eduardo4jesus/stanford-cars-dataset",
        expected_layout=(
            "cars_train/**/*.jpg",
            "cars_test/**/*.jpg",
            "devkit/cars_meta.mat",
            "devkit/cars_train_annos.mat",
            "cars_test_annos_withlabels.mat",
        ),
        notes="The loader supports common official .mat layouts and CSV annotation fallbacks.",
    ),
}

ALIASES = {
    "flowers": "flowers102",
    "oxford_flowers": "flowers102",
    "cars": "stanford_cars",
    "stanfordcars": "stanford_cars",
    "euro_sat": "eurosat",
}


def normalize_dataset_key(dataset_key: str) -> str:
    key = dataset_key.strip().lower().replace("-", "_")
    return ALIASES.get(key, key)


def get_dataset_source(dataset_key: str) -> DatasetSource:
    key = normalize_dataset_key(dataset_key)
    try:
        return DATASET_SOURCES[key]
    except KeyError as exc:
        known = ", ".join(sorted(DATASET_SOURCES))
        raise KeyError(f"Unknown dataset '{dataset_key}'. Known datasets: {known}") from exc

