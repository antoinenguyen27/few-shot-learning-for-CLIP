# PromptSRC-NC End-to-End Runbook

This runbook describes how to run the standalone PromptSRC-NC pipeline on Modal from setup through data preparation, model-weight prewarming, smoke tests, calibration, parallel full runs, evaluation, diagnostics, aggregation, and result download.

The primary variants are:

```text
PromptSRC              Stage 1 PromptSRC baseline
PromptSRC-NC real      Stage 2 continuation with real frozen-CLIP mutual-neighbor pairs
PromptSRC-NC shuffled  Stage 2 continuation with degree-preserving shuffled-neighbor control
```

The core scientific comparisons are:

```text
PromptSRC-NC real vs PromptSRC
PromptSRC-NC real vs PromptSRC-NC shuffled
```

Do not use test images as unlabeled data. Do not tune hyperparameters on test results. Do not pass `--max-unlabeled-images` for final scientific runs.

## 1. Local Setup

Run from the repository root:

```bash
cd /Users/an/Documents/few-shot-learning-for-CLIP

PROJECT=promptSRC-NC
BACKBONE=ViT-B-16
PRETRAINED=openai
```

With `PRETRAINED=openai`, requested `ViT-B-16` resolves internally to OpenCLIP's `ViT-B-16-quickgelu` model so activation behavior matches OpenAI CLIP and official PromptSRC. Artifacts still use the requested backbone in paths, and metadata records the effective OpenCLIP model name for provenance.

Install dependencies:

```bash
uv sync --project "$PROJECT" --extra dev
```

This creates the local `uv` environment and installs PyTorch, OpenCLIP, Modal, KaggleHub, Hugging Face transfer support, and test dependencies.

Run local checks before paid Modal jobs:

```bash
uv run --project "$PROJECT" python -m compileall \
  "$PROJECT/promptsrc_nc" "$PROJECT/modal_app.py"

uv run --project "$PROJECT" pytest "$PROJECT/tests" -q

uv lock --project "$PROJECT" --check
```

These checks catch syntax errors, scientific-invariant regressions, dispatcher mistakes, provenance-validation regressions, and dependency-lock drift before remote execution.

## 2. Modal Auth, Secrets, and Volumes

Authenticate Modal:

```bash
uv run --project "$PROJECT" modal setup
uv run --project "$PROJECT" modal token info
```

`modal setup` opens the browser login flow. `modal token info` is the Modal CLI 1.4.x sanity check that shows the workspace/account used for runs.

Create the Kaggle secret used only by data preparation:

```bash
uv run --project "$PROJECT" modal secret create kaggle \
  KAGGLE_USERNAME="..." \
  KAGGLE_KEY="..."
```

Create the Hugging Face secret used by all OpenCLIP-loading jobs:

```bash
read -s HF_TOKEN
uv run --project "$PROJECT" modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
unset HF_TOKEN
```

Why this matters:

- OpenCLIP prefers Hugging Face mirrors when `huggingface_hub` is installed.
- The OpenAI ViT-B/16 checkpoint is about 570 MiB and is downloaded lazily unless prewarmed.
- Authenticated HF requests reduce rate-limit and cold-start risk.
- The code injects this secret only into functions that can load model weights.

The app uses three persistent Modal volumes:

```text
promptsrc-nc-data     raw/extracted datasets, manifests, few-shot splits
promptsrc-nc-weights  Hugging Face, OpenCLIP, and Torch caches
promptsrc-nc-runs     neighbor artifacts, checkpoints, logs, evals, summaries
```

They are created automatically by `modal.Volume.from_name(..., create_if_missing=True)`.

Optional checks:

```bash
uv run --project "$PROJECT" modal secret list
uv run --project "$PROJECT" modal volume list
```

## 3. Prepare Data

Prepare all datasets, shots, and seeds needed by smoke tests and primary runs:

```bash
uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
  --action prepare_data \
  --datasets eurosat,oxford_flowers,stanford_cars \
  --prepare-shots 1,16 \
  --prepare-seeds 1,2,3
```

Configuration:

```text
--datasets       datasets to download, verify, and normalize
--prepare-shots  few-shot split sizes to write
--prepare-seeds  deterministic split seeds
```

What this does:

- downloads/verifies dataset sources;
- uses official Oxford VGG Flowers102 images and labels, then creates the PromptSRC-compatible source split unless a Zhou split file is present;
- uses Kaggle mirrors for EuroSAT and Stanford Cars;
- augments Stanford Cars with checksum-verified labeled test annotations when needed;
- writes manifests and deterministic split JSON files;
- creates the primary unlabeled pool for each split.

The primary unlabeled pool is:

```text
full training split minus few-shot labeled training examples
```

Stage 1 uses only few-shot labeled training records. Stage 0 and Stage 2 use only the remaining training images as unlabeled data. Validation and test remain separate.

## 4. Prepare Weights

Prewarm the common OpenCLIP checkpoint before smoke, profiling, or training:

```bash
uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
  --action prepare_weights \
  --backbone "$BACKBONE" \
  --pretrained "$PRETRAINED"
```

What this does:

- loads the effective OpenCLIP model once on CPU;
- downloads the pretrained checkpoint through authenticated HF if missing;
- stores it in `promptsrc-nc-weights`;
- commits the weights volume for later GPU jobs.

This separates one-time checkpoint download time from training and cost-profile timing. Later jobs still reload the model, but they should read weights from the persistent Modal cache rather than silently downloading them.

Inspect the cache:

```bash
uv run --project "$PROJECT" modal volume ls promptsrc-nc-weights /hf/hub
```

## 5. Live Monitoring

The app writes durable JSON/JSONL artifacts and also emits compact one-line JSON status events to stdout. Modal can stream these logs:

```bash
uv run --project "$PROJECT" modal app logs promptsrc-nc -f --timestamps
```

Useful events include:

```text
modal_action_start / modal_action_complete
prepare_weights_start / prepare_weights_complete
neighbors_start / neighbors_feature_batch / neighbors_built
stage_start / train_step / epoch_end / stage_complete
gpu_profile_start / gpu_profile_model_ready / gpu_profile_complete
eval_start / eval_complete
diagnostics_start / diagnostics_complete
```

Training status is emitted every `log_interval` steps and at every epoch end. This avoids messy progress bars while still showing stage, epoch, step, losses, validation metrics, seconds per step, and checkpoint completion.

Use `--detach` for long jobs so a local terminal interruption does not kill the Modal app:

```bash
uv run --project "$PROJECT" modal run --detach --timestamps "$PROJECT/modal_app.py" \
  --action train_stage1 \
  --run-id example \
  --dataset eurosat \
  --shots 16 \
  --seed 1 \
  --backbone "$BACKBONE" \
  --pretrained "$PRETRAINED" \
  --gpu L4
```

Then monitor with `modal app logs`.

## 6. Smoke Test

Run a cheap end-to-end smoke test:

```bash
SMOKE_RUN_ID=smoke-$(date +%Y%m%d-%H%M%S)

uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
  --action smoke_test \
  --run-id "$SMOKE_RUN_ID" \
  --dataset eurosat \
  --shots 1 \
  --seed 1 \
  --backbone "$BACKBONE" \
  --pretrained "$PRETRAINED" \
  --gpu T4
```

Smoke settings:

```text
stage1_epochs = 1
stage2_epochs = 1
max_train_batches = 1
max_eval_batches = 1
max_unlabeled_images = 256
pair_batch_size <= 4
precision = fp32
lambda_nc_warmup_epochs = 0
```

Smoke is not a benchmark. It catches integration failures before full runs: data loading, OpenCLIP loading, neighbor construction, checkpoint writing/loading, Stage 2 initialization, real/shuffled branches, and validation evaluation.

## 7. Calibration and GPU Choice

Use Stanford Cars 16-shot seed 1 as the calibration target:

```bash
CALIB_RUN_ID=calib-$(date +%Y%m%d-%H%M%S)
CALIB_DATASET=stanford_cars
CALIB_SHOTS=16
CALIB_SEED=1
```

Stage 1 profiling:

```bash
for GPU in T4 L4; do
  uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
    --action profile_gpu_cost \
    --run-id "$CALIB_RUN_ID" \
    --dataset "$CALIB_DATASET" \
    --shots "$CALIB_SHOTS" \
    --seed "$CALIB_SEED" \
    --backbone "$BACKBONE" \
    --pretrained "$PRETRAINED" \
    --gpu "$GPU" \
    --stage stage1
done
```

Your Stage 1 calibration favored L4:

```text
T4: 1.0973 sec/step, $0.17995 / 1000 steps
L4: 0.6235 sec/step, $0.13842 / 1000 steps
```

Modal pricing at that run was:

```text
T4: $0.000164/sec
L4: $0.000222/sec
```

L4 costs about `1.35x` more per second but was measured at `1.76x` faster, so it was both faster and cheaper per optimizer step. Use `GPU=L4` for final runs unless capacity is unavailable.

Batch policy:

```text
labeled batch_size = 4      keep fixed; official PromptSRC schedule
pair_batch_size = 8         primary Stage 2 setting
pair_batch_size = 16        optional global ablation only
pair_batch_size = 4         OOM fallback
```

Do not raise the labeled PromptSRC batch to 8, 16, or 32 for primary results. It changes optimizer steps, gradient noise, and few-shot dynamics.

## 8. Full Run Variables

Set shared variables:

```bash
RUN_ID=promptsrc-nc-main-$(date +%Y%m%d-%H%M%S)
GPU=L4
SHOTS=16
PAIR_BATCH_SIZE=8
SEEDS=(1 2 3)
```

Do not pass `--max-unlabeled-images` in final runs. The full uncapped train-remain unlabeled pool is part of the protocol.

Stages are fixed-duration:

```text
Stage 1: 50 epochs, no early stopping, writes checkpoints/final.pt and checkpoints/gpa.pt
Stage 2: 5 epochs, no early stopping, writes checkpoints/final.pt
```

Validation is used for monitoring and provenance, not for early stopping or test tuning.

## 9. Parallel Execution DAG

Valid dependencies:

```text
prepare_data + prepare_weights
        |
        v
per dataset/shot/seed:
    build_neighbors  ----\
                          +--> train_stage2 real ----\
    train_stage1 gpa -----/                          \
                                                     +--> eval/diagnostics --> aggregate
    train_stage2 shuffled ---------------------------/
```

What can run in parallel:

```text
Different dataset/seed cells: yes
Same cell build_neighbors and train_stage1: yes
Same cell train_stage2 real and shuffled: yes, after Stage 0 and Stage 1 are done
Evaluation for different checkpoints/cells: yes, after checkpoints exist
Aggregation: last
```

The run artifacts are written under dataset/shot/seed/backbone/mode-specific paths. Aggregation scans per-artifact JSON files, so it does not rely on concurrent appends to shared JSONL logs.

Start with bounded concurrency. The recommended practical pattern is one continuous foreground command per dataset, run in separate terminals if you want dataset-level parallelism. Each dataset command below handles dependencies in the correct order, so you do not need to manually start the next stage when one finishes.

## 10. Recommended Dataset-Level Commands

Use these after `prepare_data`, `prepare_weights`, and smoke have succeeded. Open one terminal per dataset if you want to run datasets concurrently. Because these commands do not use `--detach`, their live logs stream in the same terminal. Use `modal app logs promptsrc-nc -f --timestamps` only for a detached run, a second monitor, or after the launching terminal is gone.

EuroSAT:

```bash
DATASET=eurosat
for SEED in "${SEEDS[@]}"; do
  echo "=== $DATASET seed=$SEED Stage 0: neighbors ==="
  uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
    --action build_neighbors \
    --run-id "$RUN_ID" \
    --dataset "$DATASET" \
    --shots "$SHOTS" \
    --seed "$SEED" \
    --backbone "$BACKBONE" \
    --pretrained "$PRETRAINED" \
    --gpu "$GPU"

  echo "=== $DATASET seed=$SEED Stage 1: PromptSRC ==="
  uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
    --action train_stage1 \
    --run-id "$RUN_ID" \
    --dataset "$DATASET" \
    --shots "$SHOTS" \
    --seed "$SEED" \
    --backbone "$BACKBONE" \
    --pretrained "$PRETRAINED" \
    --gpu "$GPU"

  for PAIR_MODE in real shuffled; do
    echo "=== $DATASET seed=$SEED Stage 2: $PAIR_MODE ==="
    uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
      --action train_stage2 \
      --run-id "$RUN_ID" \
      --dataset "$DATASET" \
      --shots "$SHOTS" \
      --seed "$SEED" \
      --backbone "$BACKBONE" \
      --pretrained "$PRETRAINED" \
      --gpu "$GPU" \
      --pair-mode "$PAIR_MODE" \
      --pair-batch-size "$PAIR_BATCH_SIZE"
  done

  for SPLIT in val test; do
    for CKPT in stage1 stage2-real stage2-shuffled; do
      echo "=== $DATASET seed=$SEED eval $CKPT $SPLIT ==="
      uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
        --action evaluate \
        --run-id "$RUN_ID" \
        --dataset "$DATASET" \
        --shots "$SHOTS" \
        --seed "$SEED" \
        --backbone "$BACKBONE" \
        --pretrained "$PRETRAINED" \
        --gpu "$GPU" \
        --checkpoint-ref "$CKPT" \
        --split "$SPLIT"
    done
  done

  for CKPT in stage1 stage2-real stage2-shuffled; do
    echo "=== $DATASET seed=$SEED diagnostics $CKPT ==="
    uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
      --action diagnostics \
      --run-id "$RUN_ID" \
      --dataset "$DATASET" \
      --shots "$SHOTS" \
      --seed "$SEED" \
      --backbone "$BACKBONE" \
      --pretrained "$PRETRAINED" \
      --gpu "$GPU" \
      --checkpoint-ref "$CKPT"
  done
done
```

Oxford Flowers:

```bash
DATASET=oxford_flowers
for SEED in "${SEEDS[@]}"; do
  echo "=== $DATASET seed=$SEED Stage 0: neighbors ==="
  uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
    --action build_neighbors \
    --run-id "$RUN_ID" \
    --dataset "$DATASET" \
    --shots "$SHOTS" \
    --seed "$SEED" \
    --backbone "$BACKBONE" \
    --pretrained "$PRETRAINED" \
    --gpu "$GPU"

  echo "=== $DATASET seed=$SEED Stage 1: PromptSRC ==="
  uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
    --action train_stage1 \
    --run-id "$RUN_ID" \
    --dataset "$DATASET" \
    --shots "$SHOTS" \
    --seed "$SEED" \
    --backbone "$BACKBONE" \
    --pretrained "$PRETRAINED" \
    --gpu "$GPU"

  for PAIR_MODE in real shuffled; do
    echo "=== $DATASET seed=$SEED Stage 2: $PAIR_MODE ==="
    uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
      --action train_stage2 \
      --run-id "$RUN_ID" \
      --dataset "$DATASET" \
      --shots "$SHOTS" \
      --seed "$SEED" \
      --backbone "$BACKBONE" \
      --pretrained "$PRETRAINED" \
      --gpu "$GPU" \
      --pair-mode "$PAIR_MODE" \
      --pair-batch-size "$PAIR_BATCH_SIZE"
  done

  for SPLIT in val test; do
    for CKPT in stage1 stage2-real stage2-shuffled; do
      echo "=== $DATASET seed=$SEED eval $CKPT $SPLIT ==="
      uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
        --action evaluate \
        --run-id "$RUN_ID" \
        --dataset "$DATASET" \
        --shots "$SHOTS" \
        --seed "$SEED" \
        --backbone "$BACKBONE" \
        --pretrained "$PRETRAINED" \
        --gpu "$GPU" \
        --checkpoint-ref "$CKPT" \
        --split "$SPLIT"
    done
  done

  for CKPT in stage1 stage2-real stage2-shuffled; do
    echo "=== $DATASET seed=$SEED diagnostics $CKPT ==="
    uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
      --action diagnostics \
      --run-id "$RUN_ID" \
      --dataset "$DATASET" \
      --shots "$SHOTS" \
      --seed "$SEED" \
      --backbone "$BACKBONE" \
      --pretrained "$PRETRAINED" \
      --gpu "$GPU" \
      --checkpoint-ref "$CKPT"
  done
done
```

Stanford Cars:

```bash
DATASET=stanford_cars
for SEED in "${SEEDS[@]}"; do
  echo "=== $DATASET seed=$SEED Stage 0: neighbors ==="
  uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
    --action build_neighbors \
    --run-id "$RUN_ID" \
    --dataset "$DATASET" \
    --shots "$SHOTS" \
    --seed "$SEED" \
    --backbone "$BACKBONE" \
    --pretrained "$PRETRAINED" \
    --gpu "$GPU"

  echo "=== $DATASET seed=$SEED Stage 1: PromptSRC ==="
  uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
    --action train_stage1 \
    --run-id "$RUN_ID" \
    --dataset "$DATASET" \
    --shots "$SHOTS" \
    --seed "$SEED" \
    --backbone "$BACKBONE" \
    --pretrained "$PRETRAINED" \
    --gpu "$GPU"

  for PAIR_MODE in real shuffled; do
    echo "=== $DATASET seed=$SEED Stage 2: $PAIR_MODE ==="
    uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
      --action train_stage2 \
      --run-id "$RUN_ID" \
      --dataset "$DATASET" \
      --shots "$SHOTS" \
      --seed "$SEED" \
      --backbone "$BACKBONE" \
      --pretrained "$PRETRAINED" \
      --gpu "$GPU" \
      --pair-mode "$PAIR_MODE" \
      --pair-batch-size "$PAIR_BATCH_SIZE"
  done

  for SPLIT in val test; do
    for CKPT in stage1 stage2-real stage2-shuffled; do
      echo "=== $DATASET seed=$SEED eval $CKPT $SPLIT ==="
      uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
        --action evaluate \
        --run-id "$RUN_ID" \
        --dataset "$DATASET" \
        --shots "$SHOTS" \
        --seed "$SEED" \
        --backbone "$BACKBONE" \
        --pretrained "$PRETRAINED" \
        --gpu "$GPU" \
        --checkpoint-ref "$CKPT" \
        --split "$SPLIT"
    done
  done

  for CKPT in stage1 stage2-real stage2-shuffled; do
    echo "=== $DATASET seed=$SEED diagnostics $CKPT ==="
    uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
      --action diagnostics \
      --run-id "$RUN_ID" \
      --dataset "$DATASET" \
      --shots "$SHOTS" \
      --seed "$SEED" \
      --backbone "$BACKBONE" \
      --pretrained "$PRETRAINED" \
      --gpu "$GPU" \
      --checkpoint-ref "$CKPT"
  done
done
```

After all dataset-level commands finish, run aggregation once:

```bash
uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
  --action aggregate_results \
  --run-id "$RUN_ID"
```

## 11. Optional Detached Wave Commands

The commands below are optional. Use them only if you want to launch stages as detached jobs and monitor separately with:

```bash
uv run --project "$PROJECT" modal app logs promptsrc-nc -f --timestamps
```

### Wave 1: Stage 0 and Stage 1

For EuroSAT:

```bash
DATASET=eurosat
for SEED in "${SEEDS[@]}"; do
  uv run --project "$PROJECT" modal run --detach --timestamps "$PROJECT/modal_app.py" \
    --action build_neighbors --run-id "$RUN_ID" --dataset "$DATASET" --shots "$SHOTS" \
    --seed "$SEED" --backbone "$BACKBONE" --pretrained "$PRETRAINED" --gpu "$GPU"

  uv run --project "$PROJECT" modal run --detach --timestamps "$PROJECT/modal_app.py" \
    --action train_stage1 --run-id "$RUN_ID" --dataset "$DATASET" --shots "$SHOTS" \
    --seed "$SEED" --backbone "$BACKBONE" --pretrained "$PRETRAINED" --gpu "$GPU"
done
```

For Oxford Flowers:

```bash
DATASET=oxford_flowers
for SEED in "${SEEDS[@]}"; do
  uv run --project "$PROJECT" modal run --detach --timestamps "$PROJECT/modal_app.py" \
    --action build_neighbors --run-id "$RUN_ID" --dataset "$DATASET" --shots "$SHOTS" \
    --seed "$SEED" --backbone "$BACKBONE" --pretrained "$PRETRAINED" --gpu "$GPU"

  uv run --project "$PROJECT" modal run --detach --timestamps "$PROJECT/modal_app.py" \
    --action train_stage1 --run-id "$RUN_ID" --dataset "$DATASET" --shots "$SHOTS" \
    --seed "$SEED" --backbone "$BACKBONE" --pretrained "$PRETRAINED" --gpu "$GPU"
done
```

For Stanford Cars:

```bash
DATASET=stanford_cars
for SEED in "${SEEDS[@]}"; do
  uv run --project "$PROJECT" modal run --detach --timestamps "$PROJECT/modal_app.py" \
    --action build_neighbors --run-id "$RUN_ID" --dataset "$DATASET" --shots "$SHOTS" \
    --seed "$SEED" --backbone "$BACKBONE" --pretrained "$PRETRAINED" --gpu "$GPU"

  uv run --project "$PROJECT" modal run --detach --timestamps "$PROJECT/modal_app.py" \
    --action train_stage1 --run-id "$RUN_ID" --dataset "$DATASET" --shots "$SHOTS" \
    --seed "$SEED" --backbone "$BACKBONE" --pretrained "$PRETRAINED" --gpu "$GPU"
done
```

Stage 0 is done for a cell when this exists:

```text
/vol/runs/{RUN_ID}/neighbors/{dataset}/shot16/seed{seed}/metadata.json
```

Stage 1 is done for a cell when this exists:

```text
/vol/runs/{RUN_ID}/stage1/{dataset}/shot16/seed{seed}/ViT-B-16/checkpoints/gpa.pt
```

Check remotely:

```bash
uv run --project "$PROJECT" modal volume ls promptsrc-nc-runs "$RUN_ID/neighbors"
uv run --project "$PROJECT" modal volume ls promptsrc-nc-runs "$RUN_ID/stage1"
```

### Wave 2: Stage 2 Real and Shuffled

Run only after the matching Stage 0 neighbor artifacts and Stage 1 GPA checkpoint exist.

For EuroSAT:

```bash
DATASET=eurosat
for SEED in "${SEEDS[@]}"; do
  for PAIR_MODE in real shuffled; do
    uv run --project "$PROJECT" modal run --detach --timestamps "$PROJECT/modal_app.py" \
      --action train_stage2 --run-id "$RUN_ID" --dataset "$DATASET" --shots "$SHOTS" \
      --seed "$SEED" --backbone "$BACKBONE" --pretrained "$PRETRAINED" --gpu "$GPU" \
      --pair-mode "$PAIR_MODE" --pair-batch-size "$PAIR_BATCH_SIZE"
  done
done
```

For Oxford Flowers:

```bash
DATASET=oxford_flowers
for SEED in "${SEEDS[@]}"; do
  for PAIR_MODE in real shuffled; do
    uv run --project "$PROJECT" modal run --detach --timestamps "$PROJECT/modal_app.py" \
      --action train_stage2 --run-id "$RUN_ID" --dataset "$DATASET" --shots "$SHOTS" \
      --seed "$SEED" --backbone "$BACKBONE" --pretrained "$PRETRAINED" --gpu "$GPU" \
      --pair-mode "$PAIR_MODE" --pair-batch-size "$PAIR_BATCH_SIZE"
  done
done
```

For Stanford Cars:

```bash
DATASET=stanford_cars
for SEED in "${SEEDS[@]}"; do
  for PAIR_MODE in real shuffled; do
    uv run --project "$PROJECT" modal run --detach --timestamps "$PROJECT/modal_app.py" \
      --action train_stage2 --run-id "$RUN_ID" --dataset "$DATASET" --shots "$SHOTS" \
      --seed "$SEED" --backbone "$BACKBONE" --pretrained "$PRETRAINED" --gpu "$GPU" \
      --pair-mode "$PAIR_MODE" --pair-batch-size "$PAIR_BATCH_SIZE"
  done
done
```

Stage 2 is done when these exist:

```text
/vol/runs/{RUN_ID}/stage2/{dataset}/shot16/seed{seed}/ViT-B-16/real/checkpoints/final.pt
/vol/runs/{RUN_ID}/stage2/{dataset}/shot16/seed{seed}/ViT-B-16/shuffled/checkpoints/final.pt
```

### Wave 3: Evaluation and Diagnostics

Run after all target checkpoints for the dataset/seed exist.

Evaluation for one dataset:

```bash
DATASET=eurosat
for SEED in "${SEEDS[@]}"; do
  for SPLIT in val test; do
    for CKPT in stage1 stage2-real stage2-shuffled; do
      uv run --project "$PROJECT" modal run --detach --timestamps "$PROJECT/modal_app.py" \
        --action evaluate --run-id "$RUN_ID" --dataset "$DATASET" --shots "$SHOTS" \
        --seed "$SEED" --backbone "$BACKBONE" --pretrained "$PRETRAINED" --gpu "$GPU" \
        --checkpoint-ref "$CKPT" --split "$SPLIT"
    done
  done
done
```

Repeat with:

```bash
DATASET=oxford_flowers
DATASET=stanford_cars
```

Diagnostics for one dataset:

```bash
DATASET=eurosat
for SEED in "${SEEDS[@]}"; do
  for CKPT in stage1 stage2-real stage2-shuffled; do
    uv run --project "$PROJECT" modal run --detach --timestamps "$PROJECT/modal_app.py" \
      --action diagnostics --run-id "$RUN_ID" --dataset "$DATASET" --shots "$SHOTS" \
      --seed "$SEED" --backbone "$BACKBONE" --pretrained "$PRETRAINED" --gpu "$GPU" \
      --checkpoint-ref "$CKPT"
  done
done
```

Repeat with:

```bash
DATASET=oxford_flowers
DATASET=stanford_cars
```

Expected output counts:

```text
9 Stage 0 neighbor builds
9 Stage 1 PromptSRC runs
18 Stage 2 runs
54 eval rows: 3 datasets x 3 seeds x 3 variants x val/test
27 diagnostics rows if diagnostics are run for all variants
```

## 12. Aggregation

Aggregate after evaluation and diagnostics complete:

```bash
uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
  --action aggregate_results \
  --run-id "$RUN_ID"
```

What this does:

- scans per-artifact evaluation JSON files and diagnostics JSON files;
- keeps validation and test metrics distinct;
- keeps PromptSRC, PromptSRC-NC real, and PromptSRC-NC shuffled distinct;
- computes mean/std over seeds;
- computes `real_minus_promptsrc` and `real_minus_shuffled`;
- writes `aggregate_warnings.json` if malformed convenience JSONL logs were skipped.

Main interpretation:

```text
real_minus_promptsrc > 0   PromptSRC-NC real improved over PromptSRC
real_minus_shuffled > 0    real frozen-CLIP geometry beat shuffled control
```

If shuffled outperforms real, report it. That weakens the frozen-CLIP geometry claim.

## 13. Download Results

Download summaries and logs:

```bash
mkdir -p "$PROJECT/results/$RUN_ID"

uv run --project "$PROJECT" modal volume get --force \
  promptsrc-nc-runs "$RUN_ID/results" "$PROJECT/results/$RUN_ID"

uv run --project "$PROJECT" modal volume get --force \
  promptsrc-nc-runs "$RUN_ID/logs" "$PROJECT/results/$RUN_ID/logs"
```

Primary files:

```text
promptSRC-NC/results/$RUN_ID/results/eval_summary.csv
promptSRC-NC/results/$RUN_ID/results/eval_summary.json
promptSRC-NC/results/$RUN_ID/results/runs.jsonl
promptSRC-NC/results/$RUN_ID/results/diagnostics_summary.csv
promptSRC-NC/results/$RUN_ID/results/runtime_summary.csv
promptSRC-NC/results/$RUN_ID/results/cost_profile_summary.csv
promptSRC-NC/results/$RUN_ID/results/aggregate_warnings.json
```

## 14. Validity Checklist

Before reporting:

```text
[ ] Modal auth works.
[ ] Kaggle secret exists.
[ ] huggingface-secret exists with HF_TOKEN.
[ ] prepare_data completed for all datasets, shots, and seeds.
[ ] prepare_weights completed for the chosen backbone/pretrained pair.
[ ] Smoke test completed.
[ ] L4 choice is documented from calibration.
[ ] No final command used --max-unlabeled-images.
[ ] Every dataset/seed has Stage 0 neighbor artifacts.
[ ] Every dataset/seed has Stage 1 gpa.pt.
[ ] Every dataset/seed has Stage 2 real final.pt.
[ ] Every dataset/seed has Stage 2 shuffled final.pt.
[ ] Val and test evaluations exist for all three variants.
[ ] Diagnostics exist for all intended variants.
[ ] Aggregation completed.
[ ] real_minus_promptsrc and real_minus_shuffled are both reported.
[ ] Any shuffled-control win is reported, not hidden.
```

Sanity commands:

```bash
uv run --project "$PROJECT" modal volume ls promptsrc-nc-runs "$RUN_ID/results"
uv run --project "$PROJECT" modal volume get promptsrc-nc-runs "$RUN_ID/results/eval_summary.csv" -
uv run --project "$PROJECT" modal volume get promptsrc-nc-runs "$RUN_ID/results/aggregate_warnings.json" -
```

## 15. Troubleshooting

### HF warning or slow model load

Check the secret:

```bash
uv run --project "$PROJECT" modal secret list
```

Recreate if needed:

```bash
read -s HF_TOKEN
uv run --project "$PROJECT" modal secret create huggingface-secret --force HF_TOKEN="$HF_TOKEN"
unset HF_TOKEN
```

Then rerun:

```bash
uv run --project "$PROJECT" modal run --timestamps "$PROJECT/modal_app.py" \
  --action prepare_weights \
  --backbone "$BACKBONE" \
  --pretrained "$PRETRAINED"
```

### Modal cannot download data

Check and recreate the Kaggle secret:

```bash
uv run --project "$PROJECT" modal secret list

uv run --project "$PROJECT" modal secret create kaggle --force \
  KAGGLE_USERNAME="..." \
  KAGGLE_KEY="..."
```

### Logs look quiet

Use the app log stream:

```bash
uv run --project "$PROJECT" modal app logs promptsrc-nc -f --timestamps
```

If using detached jobs, the launching terminal will not show the full training stream. The app log stream is the live view.

### Stage 2 runs out of memory

First fallback:

```text
--pair-batch-size 4
```

This changes only the unlabeled pair batch. Do not reduce labeled `batch_size=4` for primary runs.

### A job fails halfway through

Rerun the failed command with the same:

```text
RUN_ID
dataset
shots
seed
backbone
pretrained
pair_mode
```

The artifact paths are per run, dataset, shot, seed, backbone, and pair mode, so rerunning a failed stage is the intended recovery path.

### Evaluation rejects a checkpoint

Check provenance:

```text
dataset
shots
seed
protocol
backbone
pretrained
checkpoint role
split hash
```

Expected checkpoint roles:

```text
stage1 evaluation         stage1_gpa
stage2 real evaluation    stage2_real_final
stage2 shuffled eval      stage2_shuffled_final
```

### Neighbor validation rejects artifacts

Rebuild Stage 0 if any of these changed:

```text
dataset
shots
seed
backbone
pretrained
max_unlabeled_images
split file
manifest image paths
neighbor_k
fallback_k
```

## 16. Scientific Guardrails

Keep these fixed unless the research proposal/spec is explicitly revised:

```text
Primary setting: few-shot all-class
Primary shots: 16
Primary seeds: 1, 2, 3
Primary datasets: oxford_flowers, eurosat, stanford_cars
Primary backbone: ViT-B-16
Primary pretrained weights: OpenAI CLIP
Primary unlabeled pool: full training split minus few-shot labeled examples
Stage 1: PromptSRC only
Stage 2: PromptSRC losses remain active plus symmetric JS neighborhood consistency
Control: degree-preserving shuffled pairs
No test images as unlabeled data
No pseudo-labeling
No entropy minimization
No EMA teacher
No PromptKD
No graph refresh
No per-dataset hyperparameter tuning
```
