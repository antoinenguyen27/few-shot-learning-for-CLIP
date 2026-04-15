# Method Contract

This page explains how method code should connect to the shared repo. If you are new, read `START_HERE.md` first.

## Your Job

Implement your method inside your assigned method folder:

- `Promptsrc/`
- `LP++/`
- `DPC/`
- `promptkd/`
- `ZeroShotCLIP/`

Do not create your own data split. Do not create your own OpenCLIP model setup. Use the common helpers below.

## The One Recommended Data Path

Use this in your method:

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

loaders = build_data_loaders(datasets, batch_size=32, num_workers=0)

train_loader = loaders["train"]
val_loader = loaders["val"]
test_loader = loaders["test"]
classnames = datasets.classnames
```

Use:

- `train_loader` for training.
- `val_loader` for choosing settings.
- `test_loader` for final reporting.
- `classnames` when building text prompts or classifiers.

For GPUs, change:

```python
bundle = build_openclip_bundle(device="cuda")
```

## Feature-Based Methods

LP++ and other feature methods should use:

```python
from common.models.openclip import (
    build_zero_shot_classifier,
    clip_classification_logits,
    encode_image_features,
    encode_text_features,
)
```

This keeps feature normalization and CLIP logit scaling consistent across methods.

## Result Logging

Every method should write one result record per dataset/shot/seed:

```python
from common.evaluation.results import RunResult, append_result, result_jsonl_path

append_result(
    RunResult(
        method="LP++",
        dataset="eurosat",
        protocol="few_shot_all_classes",
        model_name=bundle.model_name,
        pretrained=bundle.pretrained,
        shots=1,
        seed=1,
        metrics={
            "val/top1_accuracy": 0.0,
            "val/macro_accuracy": 0.0,
            "test/top1_accuracy": 0.0,
            "test/macro_accuracy": 0.0,
        },
        split_path=str(datasets.split_file),
        notes="",
    ),
    result_jsonl_path("vit_b32_256_few_shot_all_classes"),
)
```

Replace the zeros with your real metrics.
Replace `LP++` with your method name.

For the frozen zero-shot CLIP baseline, `fit` should only build the text
classifier from class names and templates. It should log `shots=0`, avoid
inspecting train images, and avoid choosing settings from validation/test
accuracy.

Summarize the file with:

```bash
python3 scripts/summarize_results.py results/vit_b32_256_few_shot_all_classes.jsonl --metric test/top1_accuracy
```

## Valid Experiment Rules

- Use shared split files.
- Use shared OpenCLIP transforms.
- Train on train data.
- Use validation data for choices.
- Use test data only for final reporting.
- Log any teacher model, extra annotation file, checkpoint, or unlabeled data source.

## Later Base-To-New Runs

The first protocol is `few_shot_all_classes`. Later, if we run base-to-new experiments, log:

- `test/base_accuracy`
- `test/new_accuracy`
- `test/harmonic_mean`

Do not mix base-to-new rows into the first all-class few-shot result table.
