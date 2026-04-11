# Common Package

`common/` contains shared code that all method implementations should use.

## Files

- `methods.py`: lightweight Protocol and artifact dataclass for future experiment orchestration.
- `datasets/types.py`: dataclasses for dataset sources, image records, and split specs.
- `datasets/sources.py`: Kaggle handles and expected source layouts.
- `datasets/paths.py`: data-root and raw-root resolution, including environment-variable overrides.
- `datasets/download.py`: KaggleHub download helpers.
- `datasets/file_utils.py`: small filesystem helpers used by dataset adapters.
- `datasets/eurosat.py`: EuroSAT RGB JPG manifest builder.
- `datasets/flowers102.py`: Flowers102 Kaggle manifest builder.
- `datasets/stanford_cars.py`: Stanford Cars annotation parser and manifest builder.
- `datasets/manifest.py`: JSONL manifest read/write, class-name extraction, and summary utilities.
- `datasets/registry.py`: dataset adapter registry.
- `datasets/splits.py`: deterministic few-shot split generation.
- `datasets/templates.py`: dataset-specific CLIP prompt templates.
- `datasets/torch_dataset.py`: lazy image dataset wrapper and split-to-dataloader helpers.
- `models/openclip.py`: shared OpenCLIP model, transform construction, feature encoding, zero-shot classifier, and CLIP-logit helpers.
- `features/cache.py`: stable cache directory naming and metadata helpers.
- `evaluation/metrics.py`: shared top-1, macro, grouped, and base/new harmonic metrics.
- `evaluation/results.py`: persisted JSONL run-result schema and seed aggregation.

## Editing Rule

`common/` is shared infrastructure. Change it only when the change benefits every method or fixes a data/model contract bug. Method-specific experiments should stay in the method directory.
