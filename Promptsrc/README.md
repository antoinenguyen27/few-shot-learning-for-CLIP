# PromptSRC Workspace

This folder is for the PromptSRC method owner.

Reference repo: https://github.com/muzairkhattak/PromptSRC

## Your Job

Implement PromptSRC inside:

```text
Promptsrc/promptsrc/
```

Do not create a private dataset loader or private few-shot sampler.

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

Use `loaders["train"]` for training, `loaders["val"]` for choices, and `loaders["test"]` for final reporting.

## Method-Specific Notes

- PromptSRC learns prompts while regularizing against frozen CLIP behavior.
- Keep frozen/reference CLIP behavior explicit in your code or result notes.
- Avoid hard-coded CLIP dimensions from the original repo. Infer dimensions from OpenCLIP.
- For this first protocol, report all-class few-shot metrics only.
- Base/new accuracy and harmonic mean are for a later base-to-new protocol.

## Before Pushing

Run:

```bash
python3 -m compileall Promptsrc common
python3 -m unittest discover -s tests -p 'test_*.py' -v
```
