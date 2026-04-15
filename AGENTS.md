# Coding Agent Instructions

This repo is a research comparison repo. Preserve scientific validity before optimizing for speed.

## Non-Negotiable Constraints

- Do not invent new train/validation/test splits in a method folder.
- Do not tune hyperparameters on test results.
- Do not use notebooks for implementation.
- Do not download or commit large datasets, model weights, feature caches, checkpoints, or result dumps.
- Do not change `common/` behavior from inside a method task unless explicitly asked to work on the shared pipeline.
- Do not rename the top-level method folders without explicit approval.

## Shared Interfaces

Use these shared modules instead of duplicating logic:

- `common.datasets.registry.build_manifest`: parse raw datasets into normalized records.
- `common.datasets.splits.make_few_shot_split`: create deterministic few-shot splits.
- `common.datasets.torch_dataset.ManifestImageDataset`: load manifest records as images.
- `common.models.openclip.build_openclip_bundle`: construct the shared OpenCLIP model and transforms.
- `common.models.openclip.encode_image_features` and `encode_text_features`: normalized OpenCLIP features for feature-based methods.
- `common.datasets.torch_dataset.build_split_datasets`: construct train/val/test datasets from existing split files.
- `common.features.cache.feature_cache_dir`: choose feature-cache locations.
- `common.evaluation.metrics`: compute accuracy and per-class accuracy.
- `common.evaluation.results.RunResult`: persist standardized run results.

## Method Workspaces

- `Promptsrc/`: PromptSRC integration.
- `LP++/`: LP++ integration. Python code should live under `LP++/lp_plus_plus/` because `LP++` is not import-safe.
- `DPC/`: DPC integration.
- `promptkd/`: PromptKD integration.
- `ZeroShotCLIP/`: frozen zero-shot CLIP baseline.

Each method directory contains a README describing its method-specific constraints.

## Safe Implementation Pattern

1. Read `START_HERE.md`.
2. Read `docs/method-contract.md`.
3. Implement inside the assigned method folder.
4. Consume the common model bundle and common split files.
5. Log validation/test metrics with `RunResult`.
6. Run `python3 -m compileall common scripts <method-folder>`.
7. Add a short method README update if you introduce a new file or command.

If a paper repo uses Dassl, OpenAI CLIP internals, local `clip/` folders, or a different dataset split, adapt it carefully at the boundary. Do not import a full legacy training stack into `common/` unless the team agrees.
