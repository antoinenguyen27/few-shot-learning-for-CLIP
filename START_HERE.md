# Start Here

This is the beginner path through the repo. Follow this page first. The other docs are references.

## 1. What This Repo Does

We are comparing few-shot learning methods for CLIP. Everyone should use the same:

- datasets
- few-shot samples
- OpenCLIP model
- preprocessing
- validation/test split
- result format

That is why `common/` exists. It prevents each method owner from accidentally using different data or evaluation.

## 2. Where You Should Work

Use only your assigned folder:

```text
Promptsrc/   PromptSRC method owner works here
LP++/        LP++ method owner works here
DPC/         DPC method owner works here
promptkd/    PromptKD method owner works here
ZeroShotCLIP/ frozen zero-shot CLIP baseline
```

Most people should not edit:

```text
common/      shared data/model/eval code
scripts/     data and result commands
data/        local downloaded data and generated splits
results/     local result logs
```

## 3. Set Up Python

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If the PyTorch install is wrong for your machine, install the correct PyTorch build first, then run:

```bash
pip install -e .
```

## 4. Before You Edit: Create Your Own Branch

Use one branch per method or task:

```bash
git checkout main
git pull
git checkout -b method/<your-method-name>
```

Examples:

```bash
git checkout -b method/promptsrc
git checkout -b method/lp-plus-plus
git checkout -b method/dpc
git checkout -b method/promptkd
```

If you or a coding agent need to change `common/`, do it on your branch and mention the common change clearly when you ask for review.

## 5. Download And Prepare Data

Download the three Kaggle datasets:

```bash
python3 scripts/download_data.py --datasets eurosat flowers102 stanford_cars
```

Check that the repo understands the downloaded folder structure:

```bash
python3 scripts/inspect_data.py --datasets eurosat flowers102 stanford_cars
```

Build the shared manifest files:

```bash
python3 scripts/build_manifests.py --datasets eurosat flowers102 stanford_cars
```

Build the shared few-shot split files:

```bash
python3 scripts/build_splits.py --datasets eurosat flowers102 stanford_cars --shots 1 2 4 8 16 --seeds 1 2 3
```

These commands create generated files under `data/`. Do not commit those generated files.

## 6. Common Quick Check

A common quick check confirms that the shared repo code works. It does not test a method implementation.

Run:

```bash
python3 -m compileall common scripts Promptsrc 'LP++' DPC promptkd ZeroShotCLIP tests
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

If these pass, the shared Python code imports and the deterministic split/result helpers work.

## 7. Method Quick Check

A method quick check is for a method owner after they implement some code.

Use a tiny run first:

```text
dataset: eurosat
shots: 1
seed: 1
```

The goal is not to get a good number. The goal is to confirm:

- your method loads the shared data
- your method uses the shared OpenCLIP model
- your method trains without crashing
- your method evaluates on validation/test data
- your method writes a result record

For the frozen zero-shot baseline, use:

```bash
python3 ZeroShotCLIP/zero_shot_clip/runner.py --dataset eurosat --device cpu --no-log
```

## 8. How To Load The Shared Data In Your Method

Use this pattern inside your method code:

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

Use `train_loader` for training, `val_loader` for choosing settings, and `test_loader` for final reporting.

## 9. How To Log A Result

After evaluation, write one result record:

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
        notes="replace this with anything unusual about the run",
    ),
    result_jsonl_path("vit_b32_256_few_shot_all_classes"),
)
```

Replace the zeros with your real metrics.
Replace `LP++` with your method name.

Then summarize results:

```bash
python3 scripts/summarize_results.py results/vit_b32_256_few_shot_all_classes.jsonl --metric test/top1_accuracy
```

## 10. Commit And Publish Your Branch

Check what changed:

```bash
git status
git diff
```

Add and commit only the files you intended to change:

```bash
git add <files-you-changed>
git commit -m "Implement <method-name>"
```

Publish your branch:

```bash
git push -u origin <your-branch-name>
```

After you push to a published branch, let Antoine know. He will take a look at the code and create the pull request.

## 11. If You Are Stuck

Common issues:

- `ModuleNotFoundError`: activate the environment and run `pip install -e .`.
- Kaggle download fails: check your Kaggle login/API access.
- Data inspect fails: run the inspect command and share the printed folder path/error.
- Stanford Cars fails: its Kaggle folder layout may differ; do not hand-write a new split, ask before changing the loader.
- GPU memory issue: start on CPU or reduce batch size for a method quick check.

## 12. Words Used In This Repo

- **shot**: number of training images per class.
- **seed**: random seed used to choose the few-shot images.
- **manifest**: a generated list of every image, label, and split.
- **split**: generated train/validation/test sample IDs.
- **validation set**: used to choose settings.
- **test set**: used only for final reporting.
- **macro accuracy**: average of per-class accuracies.
- **base-to-new**: a later protocol where some classes are base classes and others are new classes.
- **harmonic mean**: a score often used for base-to-new results.
