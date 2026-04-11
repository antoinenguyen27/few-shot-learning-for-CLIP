# DPC Workspace

This folder is for the DPC method owner.

Reference repo: https://github.com/JREion/DPC

## Your Job

Implement DPC inside:

```text
DPC/dpc/
```

DPC is a plug-in over a prompt-tuned backbone method. Record which backbone method/checkpoint you use.

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

- Do not create DPC-only data splits.
- If you use DPC annotation JSON files, put them under an ignored data path and log the path in `RunResult.notes` or `RunResult.extra`.
- Record the backbone method, backbone checkpoint, and training split.
- Base/new accuracy and harmonic mean are for a later base-to-new protocol.

## Before Pushing

Run:

```bash
python3 -m compileall DPC common
python3 -m unittest discover -s tests -p 'test_*.py' -v
```
