# Repo Map

Start with `START_HERE.md`. This page is a reference for what each directory does.

## Main Files

- `README.md`: short project overview and key commands.
- `START_HERE.md`: beginner walkthrough.
- `AGENTS.md`: instructions for coding agents.
- `pyproject.toml`: Python dependencies and package metadata.
- `requirements.txt`: convenience install file.
- `.gitignore`: keeps data, caches, results, and Python artifacts out of git.

## Directories

- `common/`: shared Python code for data, OpenCLIP, metrics, and result logging. Most method owners should import this, not edit it.
- `scripts/`: direct Python commands for data download, inspection, manifest building, split building, and result summaries.
- `data/`: local generated data files. Ignored by git.
- `results/`: local result logs. Ignored by git.
- `docs/`: reference docs.
- `tests/`: tests for the shared code.
- `Promptsrc/`: PromptSRC method workspace.
- `LP++/`: LP++ method workspace. Python package code goes under `LP++/lp_plus_plus/`.
- `DPC/`: DPC method workspace.
- `promptkd/`: PromptKD method workspace.

## Common Code Reference

- `common/datasets/sources.py`: dataset names and Kaggle handles.
- `common/datasets/download.py`: KaggleHub download helper.
- `common/datasets/eurosat.py`: EuroSAT manifest builder.
- `common/datasets/flowers102.py`: Flowers102 manifest builder.
- `common/datasets/stanford_cars.py`: Stanford Cars manifest builder.
- `common/datasets/manifest.py`: manifest read/write and class-name helpers.
- `common/datasets/splits.py`: deterministic few-shot split generation.
- `common/datasets/torch_dataset.py`: train/val/test dataset and dataloader helpers.
- `common/datasets/templates.py`: dataset text templates for CLIP classifiers.
- `common/models/openclip.py`: OpenCLIP model, preprocessing, feature, and logit helpers.
- `common/evaluation/metrics.py`: top-1, macro, grouped, and harmonic-mean metrics.
- `common/evaluation/results.py`: result JSONL logging and seed aggregation.
- `common/features/cache.py`: feature-cache path helpers.
- `common/methods.py`: lightweight method interface notes.

## Scripts

- `scripts/download_data.py`: download Kaggle datasets.
- `scripts/inspect_data.py`: check raw dataset parsing.
- `scripts/build_manifests.py`: build generated manifest JSONL files.
- `scripts/build_splits.py`: build deterministic few-shot split JSON files.
- `scripts/summarize_results.py`: summarize result JSONL files across seeds.
