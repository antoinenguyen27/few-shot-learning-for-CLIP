# LP++ Workspace

This folder is for the LP++ method owner.

Reference repo: https://github.com/FereshteShakeri/FewShot-CLIP-Strong-Baseline

## Your Job

Implement LP++ inside:

```text
LP++/lp_plus_plus/
```

The folder is named `LP++` for humans, but Python code should live in `lp_plus_plus` because `LP++` is not a valid Python package name.

## Use The Common Pipeline

Start with:

```python
from common.datasets.torch_dataset import build_data_loaders, build_split_datasets
from common.models.openclip import build_openclip_bundle

bundle = build_openclip_bundle(device="cpu")
datasets = build_split_datasets(
    dataset="eurosat",
    protocol="few_shot_all_classes",
    shots=1,
    seed=1,
    train_transform=bundle.preprocess_train,
    eval_transform=bundle.preprocess_eval,
)
loaders = build_data_loaders(datasets, batch_size=32)
```

## Feature Helpers

LP++ should use shared OpenCLIP feature helpers:

```python
from common.models.openclip import (
    build_zero_shot_classifier,
    clip_classification_logits,
    encode_image_features,
    encode_text_features,
)
```

Use `common.features.cache.feature_cache_dir` if you cache features.

## Method-Specific Notes

- Do not resample few-shot examples.
- Cache features under `data/cache/`, not inside this folder.
- Avoid hard-coded CLIP embedding dimensions.
- Persist results with `RunResult`.

## Before Pushing

Run:

```bash
python3 -m compileall 'LP++' common
python3 -m unittest discover -s tests -p 'test_*.py' -v
```
