# PromptSRC-NC Standalone Implementation

This folder contains a self-contained PyTorch/OpenCLIP + Modal implementation of **PromptSRC-NC: Neighborhood-Consistent PromptSRC**. It does not vendor or depend on the official PromptSRC repository at runtime.

The supported primary variants are exactly:

- `PromptSRC`: Stage 1 PromptSRC baseline only.
- `PromptSRC-NC real`: Stage 1 followed by Stage 2 with real frozen-CLIP neighbor pairs.
- `PromptSRC-NC shuffled`: Stage 1 followed by Stage 2 with shuffled pairs, the same unlabeled images, the same loss, and the same schedule.

## Verified References

Before implementation, the code was checked against:

- official PromptSRC repository files: `trainers/promptsrc.py`, `clip/model.py`, dataset classes, and few-shot config;
- Modal docs for Apps/Functions, `Image.uv_sync`, `add_local_dir`, Volumes, Secrets, GPU functions, and pricing;
- OpenCLIP source/docs for `create_model_and_transforms`, tokenizer use, ViT/text transformer internals, and encode APIs;
- PyTorch docs for `torch.amp` and CUDA memory metrics;
- official dataset descriptions for EuroSAT, Flowers102, and Stanford Cars.

## Important OpenCLIP Port Notes

Official PromptSRC modifies the OpenAI CLIP package directly. This implementation ports the same method mechanics to OpenCLIP wrappers:

- frozen OpenCLIP image/text towers;
- trainable shallow and deep text prompts;
- trainable shallow and deep visual prompts;
- frozen text-template teacher features averaged over ImageNet-style templates;
- PromptSRC CE, text SCL, image SCL, and logit SCL losses;
- Stage 1 GPA over trainable prompt tensors;
- Stage 2 initialized from the Stage 1 GPA checkpoint;
- Stage 2 keeps PromptSRC losses active and adds only symmetric JS neighbor consistency.

For OpenAI-pretrained ViT backbones, the effective OpenCLIP model uses the matching `*-quickgelu` variant, for example requested `ViT-B-16` plus `pretrained=openai` resolves to `ViT-B-16-quickgelu`. This matches OpenAI CLIP's activation and avoids silently evaluating PromptSRC on a mismatched tower. Checkpoints intentionally store prompt tensors only, not frozen OpenCLIP weights. The requested backbone, effective OpenCLIP model name, pretrained source, config, and checkpoint provenance are logged so the full model can be reconstructed.

Prompt optimization defaults to `fp32`. AMP remains an explicit option for profiling only after it has been validated for the selected GPU/backbone; non-finite prompt gradients fail closed instead of writing checkpoints.

## Local Setup

Use `uv`, not conda:

```bash
cd promptSRC-NC
uv sync --extra dev
uv run python -m compileall promptsrc_nc modal_app.py
uv run pytest -q
```

Local CLI commands should be run from `promptSRC-NC/`. Modal commands below are run from the repository root because `modal_app.py` builds the image from `promptSRC-NC`.

## Data Preparation

Modal data prep downloads or verifies the documented data sources, builds normalized manifests, and writes few-shot splits. Flowers102 uses the official Oxford VGG images and labels, then creates the PromptSRC-compatible 50/20/30 source split when a Zhou split file is not present. EuroSAT and Stanford Cars use the documented Kaggle mirrors, with Stanford Cars augmented by checksum-verified labeled test annotations when the mirror only provides unlabeled test annotations. Existing unlabeled Kaggle-style Flowers test folders are rejected instead of being used as evaluation data.

```bash
uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py::prepare_data \
  --datasets eurosat,oxford_flowers,stanford_cars \
  --shots 1,16 \
  --seeds 1,2,3
```

Primary unlabeled pool:

```text
full training split minus few-shot labeled training examples
```

Test images are never used in the primary unlabeled pool.

## Smoke Test

Runs a tiny end-to-end path: imports, data loading, Stage 0 neighbors, one Stage 1 step, one real Stage 2 step, one shuffled Stage 2 step, and one validation evaluation batch.

```bash
RUN_ID=smoke-$(date +%Y%m%d-%H%M%S)
uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py::smoke_test \
  --run-id "$RUN_ID" \
  --dataset eurosat \
  --shots 1 \
  --seed 1 \
  --backbone ViT-B-16
```

## GPU Cost Profiling

Profile T4 first, then L4 via the local dispatcher. The code records runtime, PyTorch/NVML memory fields when available, and estimated cost per 1000 steps.
Stage 2 profiling expects neighbor artifacts and a Stage 1 GPA checkpoint for the same `RUN_ID`, dataset, shots, seed, and backbone; run the Stage 0 and Stage 1 commands below first for the profiling target.

```bash
RUN_ID=profile-$(date +%Y%m%d-%H%M%S)

uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py --action profile_gpu_cost \
  --run-id "$RUN_ID" \
  --dataset stanford_cars \
  --shots 16 \
  --seed 1 \
  --backbone ViT-B-16 \
  --gpu T4 \
  --stage stage1

uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py --action profile_gpu_cost \
  --run-id "$RUN_ID" \
  --dataset stanford_cars \
  --shots 16 \
  --seed 1 \
  --backbone ViT-B-16 \
  --gpu L4 \
  --stage stage1

uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py --action profile_gpu_cost \
  --run-id "$RUN_ID" \
  --dataset stanford_cars \
  --shots 16 \
  --seed 1 \
  --backbone ViT-B-16 \
  --gpu T4 \
  --stage stage2 \
  --pair-mode real \
  --pair-batch-size 8

uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py --action profile_gpu_cost \
  --run-id "$RUN_ID" \
  --dataset stanford_cars \
  --shots 16 \
  --seed 1 \
  --backbone ViT-B-16 \
  --gpu L4 \
  --stage stage2 \
  --pair-mode real \
  --pair-batch-size 8
```

Current Modal pricing constants used in logs:

```text
T4 = $0.000164/sec
L4 = $0.000222/sec
```

L4 is cheaper only if it is more than about `1.35x` faster end to end.

## Stage 0: Build Neighbors

```bash
uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py::build_neighbors \
  --run-id "$RUN_ID" \
  --dataset oxford_flowers \
  --shots 16 \
  --seed 1 \
  --backbone ViT-B-16
```

Outputs:

```text
/vol/runs/{run_id}/neighbors/{dataset}/shot{K}/seed{seed}/
  unlabeled_items.jsonl
  features.pt
  real_pairs.pt
  shuffled_pairs.pt
  metadata.json
```

Real pairs use reciprocal mutual top-5 frozen-CLIP neighbors for the final PromptSRC-NC protocol. Some code paths still expose a legacy mutual top-1 request plus top-5 fallback for backward compatibility and ablations; final reporting should use the recorded `neighbor_k_used` field and should treat `neighbor_k_used = 5` as a mutual top-5 run. Shuffled pairs use degree-preserving double-edge swaps.
Neighbor metadata records the full split hash, effective unlabeled ID hash, requested OpenCLIP backbone, effective OpenCLIP model name, pretrained source, and shuffled-control audit fields. Stage 2, diagnostics, and stage-2 profiling fail closed if these artifacts do not match the active config and split.

## Stage 1: PromptSRC Baseline

```bash
uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py::train_stage1 \
  --run-id "$RUN_ID" \
  --dataset oxford_flowers \
  --shots 16 \
  --seed 1 \
  --backbone ViT-B-16
```

Stage 1 writes `final.pt` and `gpa.pt`. Stage 2 should use `gpa.pt`.
Stage 2 also validates the Stage 1 checkpoint provenance, including dataset, shots, seed, protocol, backbone, pretrained source, and split hash.

## Stage 2: PromptSRC-NC Real and Shuffled

```bash
uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py::train_stage2 \
  --run-id "$RUN_ID" \
  --dataset oxford_flowers \
  --shots 16 \
  --seed 1 \
  --backbone ViT-B-16 \
  --pair-mode real

uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py::train_stage2 \
  --run-id "$RUN_ID" \
  --dataset oxford_flowers \
  --shots 16 \
  --seed 1 \
  --backbone ViT-B-16 \
  --pair-mode shuffled
```

Stage 2 uses a global fixed configuration:

```text
stage2_epochs = 5
stage2_lr = 0.00025
sgd_momentum = 0.9
weight_decay = 0.0005
lambda_nc_max = 1.0
lambda_nc_warmup_epochs = 1
neighbor_k = 5
fallback_k = 5  # legacy compatibility only
pair_batch_size = 8
```

No pseudo-labeling, entropy minimization, confidence thresholding, EMA teacher, PromptKD distillation, graph refresh, or per-dataset tuning is implemented.

## Evaluation

Evaluate all three variants:

```bash
uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py::evaluate \
  --run-id "$RUN_ID" --dataset oxford_flowers --shots 16 --seed 1 \
  --backbone ViT-B-16 --checkpoint-ref stage1 --split test

uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py::evaluate \
  --run-id "$RUN_ID" --dataset oxford_flowers --shots 16 --seed 1 \
  --backbone ViT-B-16 --checkpoint-ref stage2-real --split test

uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py::evaluate \
  --run-id "$RUN_ID" --dataset oxford_flowers --shots 16 --seed 1 \
  --backbone ViT-B-16 --checkpoint-ref stage2-shuffled --split test
```

## Diagnostics

Diagnostics are computed on real neighbor pairs for PromptSRC and PromptSRC-NC:

```bash
uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py::diagnostics \
  --run-id "$RUN_ID" --dataset oxford_flowers --shots 16 --seed 1 \
  --backbone ViT-B-16 --checkpoint-ref stage1

uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py::diagnostics \
  --run-id "$RUN_ID" --dataset oxford_flowers --shots 16 --seed 1 \
  --backbone ViT-B-16 --checkpoint-ref stage2-real
```

Recorded diagnostics include edge disagreement, mean JS, mean entropy, mean confidence, pair counts, and neighbor cosine metadata.

## Aggregation

```bash
uv run --project promptSRC-NC modal run promptSRC-NC/modal_app.py::aggregate_results \
  --run-id "$RUN_ID"
```

Outputs:

```text
/vol/runs/{run_id}/results/runs.jsonl
/vol/runs/{run_id}/results/eval_summary.csv
/vol/runs/{run_id}/results/eval_summary.json
/vol/runs/{run_id}/results/diagnostics_summary.csv
/vol/runs/{run_id}/results/runtime_summary.csv
/vol/runs/{run_id}/results/cost_profile_summary.csv
```

`eval_summary.csv` reports PromptSRC, PromptSRC-NC shuffled, PromptSRC-NC real, `real_minus_promptsrc`, and `real_minus_shuffled` side by side.

## Minimal Matrix

Do not run this full paid matrix without explicit approval:

```text
datasets = oxford_flowers, eurosat, stanford_cars
shots = 16
seeds = 1, 2, 3
variants = PromptSRC, PromptSRC-NC real, PromptSRC-NC shuffled
```

For every dataset/seed:

```bash
# 1. Stage 0
build_neighbors

# 2. Stage 1 baseline
train_stage1

# 3. Stage 2 variants
train_stage2 --pair-mode real
train_stage2 --pair-mode shuffled

# 4. Evaluate and diagnose
evaluate --checkpoint-ref stage1
evaluate --checkpoint-ref stage2-real
evaluate --checkpoint-ref stage2-shuffled
diagnostics --checkpoint-ref stage1
diagnostics --checkpoint-ref stage2-real
```

## Local CPU CLI Examples

For local dry runs against already prepared small data:

```bash
cd promptSRC-NC
uv run python -m promptsrc_nc.neighbors --data-root ../data/promptsrc_nc --run-root ../results/promptsrc_nc --run-id local-dev --dataset eurosat --shots 1 --seed 1 --max-unlabeled-images 256 --num-workers 0
uv run python -m promptsrc_nc.train --stage stage1 --data-root ../data/promptsrc_nc --run-root ../results/promptsrc_nc --run-id local-dev --dataset eurosat --shots 1 --seed 1 --epochs 1 --max-train-batches 1 --max-eval-batches 1 --num-workers 0 --precision fp32
```
