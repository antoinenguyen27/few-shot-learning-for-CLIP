# Coding Agent Instructions

This repo is a research workspace for **PromptSRC-NC: Neighborhood-Consistent PromptSRC**, with archived method-reproduction scaffolding kept for context. Preserve scientific validity before optimizing for speed.

## Source-Of-Truth Order

Use sources in this order:

1. Official papers and official repositories.
2. `promptSRC-NC/neighborhood_promptsrc_research_proposal.md`.
3. `promptSRC-NC/neighborhood_promptsrc_technical_spec.md`.
4. Local shared infrastructure in `common/` and `scripts/`.
5. Archived reproduction notes and code under `method_reproductions/`.

The reproduction material may be incomplete or inaccurate. Do not treat `method_reproductions/` as evidence that a method is implemented correctly unless you verify it against the official paper and official repository.

## Non-Negotiable Constraints

- Do not invent new train/validation/test splits in a method folder.
- Do not tune hyperparameters on test results.
- Do not use test images as unlabeled data in the primary PromptSRC-NC setting.
- Do not mix strict few-shot, base-to-novel, and transductive results without explicit labels.
- Do not use notebooks for implementation.
- Do not download or commit large datasets, model weights, feature caches, neighbor caches, checkpoints, or result dumps.
- Do not change `common/` behavior from inside a method task unless explicitly asked to work on the shared pipeline.
- Do not rename top-level research or archive folders without explicit approval.
- Do not present archived reproduction code as a faithful paper implementation without checking official sources.

## PromptSRC-NC Method Rules

The main method is narrow by design:

- Start from official PromptSRC.
- Build frozen-CLIP nearest-neighbor pairs over unlabeled target training images.
- Continue prompt optimization with a symmetric neighborhood-consistency loss over those pairs.
- Keep PromptSRC losses active during Stage 2.
- Compare real frozen-CLIP neighbor pairs against a shuffled-neighbor control.

Do not add these to the main PromptSRC-NC method unless explicitly asked:

- pseudo-labeling;
- entropy minimization;
- confidence thresholding;
- EMA teachers;
- PromptKD-style teacher-student distillation;
- relation preservation losses;
- optimal transport;
- graph Laplacian normalization variants;
- periodically refreshed graphs;
- per-dataset hyperparameter tuning.

The main variants are:

1. `PromptSRC`: official baseline.
2. `PromptSRC-NC`: real frozen-CLIP neighbor pairs.
3. `PromptSRC-NC shuffled`: same Stage 2 setup with shuffled pairs.

The primary unlabeled pool is:

```text
full training split minus few-shot labeled training examples
```

Labels may be carried for bookkeeping and diagnostics, but Stage 0 feature construction and Stage 2 neighbor loss must not use labels as supervision.

## Active Workspaces

- `promptSRC-NC/`: active proposal and technical specification for the new method.
- `method_reproductions/`: archived prior reproduction/comparison scaffold. Read for context; verify against official sources before relying on it.
- `common/`: shared local data/model/evaluation utilities from the earlier scaffold.
- `scripts/`: command-line wrappers around `common/`.
- `data/`: local generated data, ignored by git.
- `results/`: local result logs, ignored by git.
- `tests/`: tests for shared infrastructure and archived method code.

If implementing PromptSRC-NC code, work inside the active method workspace or a clearly documented method-local subfolder. If a folder name is not import-safe, put Python packages under an import-safe subdirectory.

## Shared Interfaces

Use these shared modules when working on the local OpenCLIP scaffold or archived comparison code:

- `common.datasets.registry.build_manifest`: parse raw datasets into normalized records.
- `common.datasets.splits.make_few_shot_split`: create deterministic few-shot splits.
- `common.datasets.torch_dataset.ManifestImageDataset`: load manifest records as images.
- `common.models.openclip.build_openclip_bundle`: construct the shared OpenCLIP model and transforms.
- `common.models.openclip.encode_image_features` and `encode_text_features`: normalized OpenCLIP features for feature-based methods.
- `common.datasets.torch_dataset.build_split_datasets`: construct train/val/test datasets from existing split files.
- `common.features.cache.feature_cache_dir`: choose feature-cache locations.
- `common.evaluation.metrics`: compute accuracy and per-class accuracy.
- `common.evaluation.results.RunResult`: persist standardized run results.

For PromptSRC-NC work based on the official PromptSRC repository, the official PromptSRC paper/repo and `promptSRC-NC` technical spec take precedence over the older OpenCLIP scaffold if there is a conflict.

## Safe Implementation Pattern

1. Read `README.md`.
2. Read both docs in `promptSRC-NC/`.
3. Read the relevant official paper and official repository code.
4. Read archived docs in `method_reproductions/` only for local scaffold context.
5. Implement inside the assigned active method workspace.
6. Preserve official data splits, model/backbone choices, transforms, and method losses unless a documented compatibility fix is necessary.
7. Log validation/test metrics and all material deviations, including unlabeled-pool policy and checkpoint provenance.
8. Run the narrowest relevant checks, for example `python3 -m compileall common scripts <method-folder>`.
9. Update the relevant README/spec if you introduce a new file, command, protocol, or assumption.

If a paper repo uses Dassl, OpenAI CLIP internals, local `clip/` folders, or different dataset split semantics, adapt carefully at the boundary. Do not import a full legacy training stack into `common/` unless the team explicitly agrees.

## Result Validity

PromptSRC-NC results should report:

- dataset;
- shots;
- seed;
- split/protocol;
- PromptSRC checkpoint used for Stage 2;
- real or shuffled pair mode;
- neighbor construction metadata;
- validation metrics;
- final test metrics;
- diagnostics such as edge disagreement, mean JS, entropy, and confidence.

If the shuffled-neighbor control outperforms the real-neighbor variant, do not hide it. That weakens the geometry claim and should be reported as such.
