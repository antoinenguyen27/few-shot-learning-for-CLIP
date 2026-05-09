# DPC Workspace

This folder is for the DPC method owner.

Reference repo: https://github.com/JREion/DPC

## Your Job

Implement DPC inside:

```text
DPC/dpc/
```

DPC is a plug-in over a prompt-tuned backbone method. Record which backbone method/checkpoint you use.

This workspace now contains an OpenCLIP-native DPC port. It uses a
PromptSRC-style learned text prompt as the first-stage backbone, clones that
prompt into a parallel DPC prompt, then trains the parallel prompt with a
hard-negative objective and weighted dual-prompt inference.

The official DPC repository is built around Dassl, OpenAI CLIP, backbone
checkpoints, and DPC-specific `SPLE_XXX.json` annotation files. This port keeps
the shared repo protocol intact: shared OpenCLIP, shared splits, shared metrics,
and standardized `RunResult` logging. It does not claim paper-number parity with
the official base-to-new DPC setup.

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

## Example Run

From the repo root, after data manifests and splits exist:

```bash
python3 -m DPC.dpc.runner --dataset eurosat --shots 1 --seed 1 --device cuda --backbone-epochs 20 --dpc-epochs 20
```

For a quick smoke test:

```bash
python3 -m DPC.dpc.runner --dataset eurosat --shots 1 --seed 1 --device cuda --backbone-epochs 1 --dpc-epochs 1 --max-train-batches 1 --max-eval-batches 2 --no-log
```

Use `--annotation-path` and `--backbone-checkpoint` to log provenance if you
adapt official DPC annotation files or a saved backbone prompt checkpoint later.
