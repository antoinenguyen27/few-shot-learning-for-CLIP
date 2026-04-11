# Experiment Protocol

This repo currently supports one simple shared protocol.

## First Protocol

Name:

```text
few_shot_all_classes
```

Datasets:

- EuroSAT
- Oxford Flowers 102
- Stanford Cars

Model:

```text
OpenCLIP model: ViT-B-32-256
pretrained weights: datacomp_s34b_b86k
image size: 256
```

Shots:

```text
1, 2, 4, 8, 16
```

Seeds:

```text
1, 2, 3
```

Metrics:

- `test/top1_accuracy`
- `test/macro_accuracy`

Also log validation metrics:

- `val/top1_accuracy`
- `val/macro_accuracy`

## Commands

Build all shared splits:

```bash
python3 scripts/build_splits.py --datasets eurosat flowers102 stanford_cars --shots 1 2 4 8 16 --seeds 1 2 3
```

Summarize result logs:

```bash
python3 scripts/summarize_results.py results/vit_b32_256_few_shot_all_classes.jsonl --metric test/top1_accuracy
```

## Rules

- Do not create method-private split files.
- Do not choose hyperparameters from test accuracy.
- Do not change preprocessing for one method only.
- Do not mix strict few-shot and transductive results.

## Method Notes

- LP++ is feature-based. It should use shared OpenCLIP feature helpers and shared cache paths.
- PromptSRC should keep frozen CLIP/reference behavior explicit in result notes.
- PromptKD must record teacher model and unlabeled distillation data source.
- DPC must record the backbone method and any DPC-specific annotation files or prior checkpoints.

## Later Protocol

Base-to-new class generalization is useful for PromptSRC, PromptKD, and DPC, but it is not the first protocol. If we run it later, it should get its own split files and its own result table.
