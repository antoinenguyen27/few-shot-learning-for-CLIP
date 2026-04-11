"""Dataset utilities for the shared few-shot CLIP pipeline."""

from .manifest import class_names_from_records, read_manifest, summarize_records, write_manifest
from .registry import build_manifest, dataset_keys
from .splits import FewShotSplitError, make_few_shot_split, read_split, write_split
from .torch_dataset import ManifestImageDataset, SplitDatasets, build_data_loaders, build_split_datasets
from .types import ImageRecord, SplitSpec

__all__ = [
    "FewShotSplitError",
    "ImageRecord",
    "ManifestImageDataset",
    "SplitDatasets",
    "SplitSpec",
    "build_data_loaders",
    "build_manifest",
    "build_split_datasets",
    "class_names_from_records",
    "dataset_keys",
    "make_few_shot_split",
    "read_manifest",
    "read_split",
    "summarize_records",
    "write_manifest",
    "write_split",
]
