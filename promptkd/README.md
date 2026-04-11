# PromptKD Workspace

This folder is for the PromptKD method owner.

Reference repo: https://github.com/zhengli97/PromptKD

## Your Job

Implement PromptKD inside:

```text
promptkd/promptkd/
```

PromptKD uses a teacher/student setup. Record the teacher model and distillation data source.

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

## Method-Specific Notes

- Do not silently use test images as unlabeled distillation data in strict few-shot mode.
- If you run a transductive setting, label it clearly and keep it separate from strict few-shot results.
- Record teacher model, teacher checkpoint, and distillation data source in `RunResult.notes` or `RunResult.extra`.
- Avoid hard-coded CLIP dimensions.

## Before Pushing

Run:

```bash
python3 -m compileall promptkd common
python3 -m unittest discover -s tests -p 'test_*.py' -v
```
