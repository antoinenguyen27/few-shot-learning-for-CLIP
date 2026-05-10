# Technical Specification: PromptSRC-NC

## 0. Scope

This document specifies an implementation plan for **PromptSRC-NC: Neighborhood-Consistent PromptSRC**, a standalone PyTorch/OpenCLIP research implementation that follows the official PromptSRC paper and repository as the algorithmic reference.

The coding agent should implement:

1. standalone data preprocessing for Flowers102, EuroSAT, and Stanford Cars;
2. standalone PromptSRC-style Stage 1 prompt training and evaluation;
3. frozen-CLIP neighbor-pair construction over unlabeled training images;
4. Stage 2 PromptSRC-NC adaptation with neighborhood-consistency loss;
5. a shuffled-neighbor control;
6. Modal app functions for data prep, smoke tests, GPU cost profiling, training, evaluation, and aggregation;
7. comprehensive runtime, training, evaluation, and diagnostic logging.

The implementation should be modular and self-contained under `promptSRC-NC/`. The official PromptSRC repository should be read and matched carefully, but it should not be a runtime dependency for the cloud implementation unless a specific compatibility issue forces that choice.

---

## 1. Reference baseline

Use the official PromptSRC repository as the source of truth for method mechanics:

```text
https://github.com/muzairkhattak/PromptSRC
```

Important files in the official repository:

```text
train.py
trainers/promptsrc.py
clip/model.py
configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx.yaml
configs/trainers/PromptSRC/vit_b16_c2_ep50_batch4_4+4ctx_few_shot.yaml
configs/datasets/oxford_flowers.yaml
configs/datasets/eurosat.yaml
configs/datasets/stanford_cars.yaml
scripts/promptsrc/base2new_train.sh
scripts/promptsrc/base2new_test.sh
scripts/promptsrc/few_shot.sh
docs/TRAIN.md
docs/EVAL.md
docs/DATASETS.md
```

The official repository is based on Dassl.pytorch. For this project, treat Dassl and the official `clip/` fork as references to port from, not as infrastructure to build around. The active implementation should be standalone PyTorch/OpenCLIP code so Modal jobs can run from this repository without cloning and modifying a separate training stack.

---

## 2. Environment

### 2.1 Local environment

Local development should use `uv`. Do not use conda for this project.

```bash
cd promptSRC-NC
uv sync --extra dev
uv run python -m compileall promptsrc_nc modal_app.py
uv run pytest -q
uv run modal --help
```

Local should be used for editing, linting, import checks, small CPU-only unit tests, and Modal command submission. Full training runs should run on Modal.

### 2.2 Modal cloud environment

Use a Modal app as the cloud execution layer. Modal Apps group Functions and manage logs for code running in Modal. Define one app under `promptSRC-NC/`, for example:

```text
promptSRC-NC/
|-- modal_app.py
|-- promptsrc_nc/
|   |-- data.py
|   |-- splits.py
|   |-- model.py
|   |-- train.py
|   |-- neighbors.py
|   |-- eval.py
|   |-- logging.py
|   |-- aggregate.py
|   `-- cost_profile.py
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

The Modal image should install dependencies with Modal's uv-based image APIs, not conda:

```python
import modal

app = modal.App("promptsrc-nc")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .uv_pip_install(
        "torch",
        "torchvision",
        "open_clip_torch",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "Pillow",
        "tqdm",
        "kagglehub",
        "rich",
        "orjson",
        "psutil",
        "pynvml",
    )
    .add_local_dir("promptSRC-NC/promptsrc_nc", remote_path="/root/promptsrc_nc")
)
```

If a `pyproject.toml` and `uv.lock` are maintained inside `promptSRC-NC/`, prefer `Image.uv_sync(...)` and then add local source files. Modal's docs note that `uv_sync` installs dependencies from a uv project but does not install the project itself, so local Python source must still be added to the image.

### 2.3 Modal volumes

Use Modal Volumes for durable datasets, model caches, neighbor caches, checkpoints, logs, and result artifacts. Recommended mounts:

```python
data_vol = modal.Volume.from_name("promptsrc-nc-data", create_if_missing=True)
weights_vol = modal.Volume.from_name("promptsrc-nc-weights", create_if_missing=True)
runs_vol = modal.Volume.from_name("promptsrc-nc-runs", create_if_missing=True)

VOLUME_MOUNTS = {
    "/vol/data": data_vol,
    "/vol/weights": weights_vol,
    "/vol/runs": runs_vol,
}
```

Use these path conventions:

```text
/vol/data/raw_archives/              downloaded Kaggle archives or source snapshots
/vol/data/extracted/{dataset}/       extracted dataset trees when needed
/vol/data/processed/{dataset}/       manifests, split files, class maps
/vol/weights/openclip/               OpenCLIP and Hugging Face caches
/vol/runs/{run_id}/neighbors/        features and real/shuffled pair files
/vol/runs/{run_id}/checkpoints/      Stage 1 and Stage 2 checkpoints
/vol/runs/{run_id}/logs/             JSONL runtime/train/eval logs
/vol/runs/{run_id}/results/          per-run metrics and aggregate tables
```

Modal Volumes require explicit `commit()` calls for writes that must be visible outside the current container, and `reload()` calls to see changes written by another container. Avoid concurrent writes to the same file. Use per-run, per-dataset, per-seed output directories so parallel jobs never contend for the same artifacts.

Modal's Volume docs note that Volumes work best below tens of thousands of files. The three datasets together are around that threshold, so prefer storing raw downloads as archives and writing processed metadata separately. Training functions may extract a dataset archive to container-local scratch storage at startup if direct small-file reads from the Volume are slow.

### 2.4 Modal secrets

If data download needs Kaggle credentials, store them as a Modal Secret and attach the secret only to the data-preparation function:

```bash
modal secret create kaggle KAGGLE_USERNAME=... KAGGLE_KEY=...
```

```python
@app.function(image=image, volumes=VOLUME_MOUNTS, secrets=[modal.Secret.from_name("kaggle")])
def prepare_data(...):
    ...
```

Do not write secrets into logs, result JSON, config files, or Volume artifacts.

### 2.5 Modal references

- Apps and Functions: `https://modal.com/docs/guide/apps`
- Images and uv installs: `https://modal.com/docs/guide/images`, `https://modal.com/docs/reference/modal.Image`
- GPU functions: `https://modal.com/docs/guide/gpu`
- Volumes: `https://modal.com/docs/guide/volumes`, `https://modal.com/docs/reference/modal.Volume`
- Model weights on Volumes: `https://modal.com/docs/guide/model-weights`
- Secrets: `https://modal.com/docs/guide/secrets`
- GPU metrics: `https://modal.com/docs/guide/gpu-metrics`
- Timeouts and retries: `https://modal.com/docs/guide/timeouts`, `https://modal.com/docs/guide/retries`

---

## 3. Datasets

The project uses three datasets:

| Human name | PromptSRC dataset arg | Registered class | Folder under `$DATA` |
|---|---|---|---|
| Flowers102 | `oxford_flowers` | `OxfordFlowers` | `oxford_flowers/` |
| EuroSAT | `eurosat` | `EuroSAT` | `eurosat/` |
| Stanford Cars | `stanford_cars` | `StanfordCars` | `stanford_cars/` |

The standalone implementation must provide its own data preprocessing code under `promptSRC-NC/`. It may reuse ideas from the archived `method_reproductions/docs/data.md`, but it should not depend on scripts outside `promptSRC-NC/` for cloud runs.

Use these default data sources:

```text
Flowers102:    official Oxford VGG source, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
EuroSAT:       KaggleHub mirror, apollo2506/eurosat-dataset
Stanford Cars: KaggleHub mirror, eduardo4jesus/stanford-cars-dataset
```

The `prepare_data` Modal function should download or verify raw sources, build normalized manifests, write deterministic split files, and emit a data-preparation log record for every dataset. It should be idempotent: if the Volume already contains matching source metadata and split files, it should verify and return without redownloading.

### 3.1 Required dataset structure

Support both the official PromptSRC/Zhou-style dataset layout and the Kaggle layouts from the reproduction scaffold. The output of preprocessing should be a normalized manifest and split files that the standalone PyTorch dataloaders consume.

#### Flowers102

PromptSRC/Zhou-style:

```text
$DATA/oxford_flowers/
|-- cat_to_name.json
|-- imagelabels.mat
|-- jpg/
|-- split_zhou_OxfordFlowers.json
```

Official VGG fallback when the PromptSRC/Zhou split file is not present:

```text
$DATA/oxford_flowers/
|-- cat_to_name.json
|-- imagelabels.mat
|-- jpg/
```

The implementation should use `split_zhou_OxfordFlowers.json` when available, otherwise use the official VGG labels to create the PromptSRC-compatible 50/20/30 train/validation/test source split. Do not use the official VGG `setid.mat` split for the primary PromptSRC-NC few-shot protocol because its training partition has only 10 images per class and cannot support 16-shot runs. A flat Kaggle challenge-style `dataset/test/*.jpg` folder is unlabeled and must not be used as the evaluation test split.

Class-labeled Kaggle mirror layout, supported only if already provided manually:

```text
dataset/
|-- train/<class_id>/*.jpg
|-- valid/<class_id>/*.jpg
|-- test/<class_id>/*.jpg
cat_to_name.json
```

#### EuroSAT

PromptSRC/Zhou-style:

```text
$DATA/eurosat/
|-- 2750/
|-- split_zhou_EuroSAT.json
```

Kaggle mirror:

```text
EuroSAT/
|-- AnnualCrop/*.jpg
|-- Forest/*.jpg
|-- ...
EuroSATallBands/
```

Use only RGB JPG images for the main CLIP experiments.

#### Stanford Cars

PromptSRC/Zhou-style:

```text
$DATA/stanford_cars/
|-- cars_test/
|-- cars_test_annos_withlabels.mat
|-- cars_train/
|-- devkit/
|-- split_zhou_StanfordCars.json
```

Kaggle mirror plus official labeled test annotations:

```text
cars_train/**/*.jpg or cars_train/cars_train/*.jpg
cars_test/**/*.jpg or cars_test/cars_test/*.jpg
devkit/cars_meta.mat or car_devkit/devkit/cars_meta.mat
devkit/cars_train_annos.mat or car_devkit/devkit/cars_train_annos.mat
cars_test_annos_withlabels.mat
```

Some Kaggle mirrors include only `cars_test_annos.mat`, whose annotations do not contain class labels. That file is not sufficient for evaluation; data preparation must download or verify `cars_test_annos_withlabels.mat` from the official Stanford Cars source, with a checksum-verified public mirror fallback if the Stanford host is temporarily unavailable.

### 3.2 Dataset split behavior in PromptSRC

The official PromptSRC dataset classes load fixed Zhou-style splits when available. If the fixed split is missing, the dataset classes create train/val/test splits and save them. The standalone implementation should match this behavior as closely as possible and record the split source in metadata.

Few-shot behavior:

1. Read full split.
2. If `DATASET.NUM_SHOTS >= 1`, create or load `split_fewshot/shot_{K}-seed_{seed}.pkl`.
3. Few-shot train set is generated from the original train split.
4. Few-shot val set is generated from the original val split with `min(K, 4)` shots.
5. `DATASET.SUBSAMPLE_CLASSES` can be `all`, `base`, or `new`.

For this project’s primary few-shot setting, use:

```text
DATASET.SUBSAMPLE_CLASSES all
DATASET.NUM_SHOTS K
```

### 3.3 Standalone manifest format

Write one manifest JSONL per dataset:

```text
/vol/data/processed/{dataset}/manifest.jsonl
```

Each line should include:

```json
{
  "uid": "eurosat/AnnualCrop/AnnualCrop_1.jpg",
  "dataset": "eurosat",
  "image_path": "/vol/data/extracted/eurosat/EuroSAT/AnnualCrop/AnnualCrop_1.jpg",
  "label_id": 0,
  "class_name": "annual crop land",
  "source_split": "train",
  "metadata": {
    "source": "kagglehub:apollo2506/eurosat-dataset"
  }
}
```

Labels in manifests are allowed for supervised training, evaluation, and diagnostics. Stage 0 neighbor construction and Stage 2 neighborhood loss must not use labels as supervision.

### 3.4 Split files

Write split files under:

```text
/vol/data/processed/{dataset}/splits/{protocol}/shots_{K}/seed_{seed}.json
```

The split file should identify:

- labeled training image IDs;
- validation image IDs;
- test image IDs;
- unlabeled training-pool image IDs for PromptSRC-NC;
- class-name mapping;
- source split policy;
- seed and shot count.

For the primary setting, the unlabeled pool must be:

```text
full training split minus few-shot labeled training examples
```

Do not include test image IDs in the primary unlabeled pool.

---

## 4. Official PromptSRC mechanics

This section describes PromptSRC at a low level because the second-stage trainer must preserve these mechanics.

The standalone implementation should use PyTorch and OpenCLIP for model construction, tokenization, image preprocessing, mixed precision, and checkpointing. The official PromptSRC/Dassl code below is reference material for what must be matched, not a requirement to import Dassl or the official `clip/` package.

### 4.1 CLIP loading

`trainers/promptsrc.py` defines:

```python
load_clip_to_cpu(cfg, zero_shot_model=False)
```

When `zero_shot_model=False`, it builds a CLIP model with design details:

```python
design_details = {
    "trainer": "IVLP",
    "vision_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION,
    "language_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT,
    "vision_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_VISION,
    "language_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_TEXT,
}
```

When `zero_shot_model=True`, it builds an unprompted CLIP model with zero prompt depths and zero context tokens. This is used to compute frozen CLIP visual/textual features.

### 4.2 Visual prompts

`clip/model.py` modifies the CLIP VisionTransformer. If `design_details["vision_depth"] > 0`, the visual encoder creates `self.VPT`, a learnable visual prompt matrix with shape:

```text
[N_CTX_VISION, vision_width]
```

For ViT-B/16, `vision_width` is 768.

At the visual input level:

1. image is patchified;
2. class token is prepended;
3. positional embedding is added;
4. visual prompt tokens are appended;
5. transformer blocks process the sequence.

For deeper prompting, `ResidualAttentionBlock_IVLP` can replace visual prompt tokens at deeper layers up to the configured prompt depth.

### 4.3 Text prompts

`VLPromptLearner` creates learnable text context vectors:

```python
self.ctx = nn.Parameter(ctx_vectors)
```

Default text prompt count:

```text
N_CTX_TEXT = 4
```

If `CTX_INIT="a photo of a"` and `n_ctx <= 4`, the first-layer text context is initialized from CLIP token embeddings for the phrase. Otherwise, it is random-normal initialized.

For each class name, a prompt is constructed:

```text
{ctx_init} {class name}.
```

The token embedding is split into:

- `token_prefix`: SOS token embedding;
- learnable `ctx`;
- `token_suffix`: class-name tokens plus EOS.

At each forward pass:

```python
prompts = [prefix, ctx, suffix]
text_features = TextEncoder(prompts, tokenized_prompts)
```

For deep text prompting, `clip/model.py` inserts prompt tokens into text transformer layers through `ResidualAttentionBlock_IVLP`.

### 4.4 Frozen text features with textual diversity

In `VLPromptLearner.__init__`, PromptSRC loads a separate zero-shot CLIP model and computes frozen CLIP text embeddings using many templates from `IMAGENET_TEMPLATES`.

For each template:

```python
x = [single_template.replace("{}", name) for name in classnames]
text_features = clip_model_temp.encode_text(x_tokenized.cuda())
```

Then it stores:

```python
self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
```

This gives one averaged frozen text feature per class.

### 4.5 Forward pass during training

`CustomCLIP.forward(image, label)` computes:

1. prompted text features;
2. prompted image features;
3. normalized features;
4. prompted logits:

\[
\text{logits} =
\exp(\text{logit_scale})
\cdot
\hat{f}_p
\hat{g}_p^\top
\]

During training it also computes frozen CLIP image features with the zero-shot image encoder:

```python
zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
```

and zero-shot logits:

```python
zero_shot_logits = logit_scale * zero_shot_features @ fixed_embeddings.T
```

The training forward returns:

```python
loss_ce,
text_features,
fixed_embeddings,
zero_shot_features,
image_features,
zero_shot_logits,
logits
```

During evaluation it returns only logits.

### 4.6 PromptSRC loss

In `PromptSRC.forward_backward`, the official loss is:

```python
loss_ce = F.cross_entropy(logits, label)

loss_scl_text = F.l1_loss(
    normalized_text_features,
    zs_clip_text_embeddings.cuda(),
    reduction="mean"
) * TEXT_LOSS_WEIGHT

loss_scl_image = F.l1_loss(
    image_ft,
    zs_image_embedd.cuda(),
    reduction="mean"
) * IMAGE_LOSS_WEIGHT

L_SCL_logits = F.kl_div(
    F.log_softmax(logits / 1, dim=1),
    F.log_softmax(zero_shot_logits / 1, dim=1),
    reduction="sum",
    log_target=True
) / logits.numel()

loss = loss_ce + loss_scl_text + loss_scl_image + L_SCL_logits
```

Default PromptSRC weights:

```text
TEXT_LOSS_WEIGHT = 25
IMAGE_LOSS_WEIGHT = 10
```

### 4.7 Gaussian weighted prompt aggregation

At the end of each epoch, PromptSRC accumulates a Gaussian-weighted copy of the model state dict.

The Gaussian weights are computed over the epoch range:

```python
gauss = lambda x: (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu) / sigma)**2)
self.gauss = [gauss(a) for a in range(1, N+1)]
self.gauss = self.gauss / sum(self.gauss)
```

At the end of training, PromptSRC loads the accumulated GPA state dict for final inference.

The coding agent must verify whether the saved final checkpoint contains the GPA-loaded model. If not, modify the training code to explicitly save a `model-gpa.pth.tar` after GPA loading. Stage 2 should initialize from GPA weights, not from a non-aggregated final prompt state.

---

## 5. Standalone training defaults

### 5.1 Few-shot all-class setting

Use the official PromptSRC few-shot config as the reference:

```text
configs/trainers/PromptSRC/vit_b16_c2_ep50_batch4_4+4ctx_few_shot.yaml
```

Important values:

```yaml
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 4
  TEST:
    BATCH_SIZE: 100
  NUM_WORKERS: 8

INPUT:
  SIZE: (224, 224)
  INTERPOLATION: "bicubic"
  PIXEL_MEAN: [0.48145466, 0.4578275, 0.40821073]
  PIXEL_STD: [0.26862954, 0.26130258, 0.27577711]
  TRANSFORMS: ["random_resized_crop", "random_flip", "normalize"]

OPTIM:
  NAME: "sgd"
  LR: 0.0025
  MOMENTUM: 0.9
  WEIGHT_DECAY: 0.0005
  MAX_EPOCH: 50
  LR_SCHEDULER: "cosine"
  WARMUP_EPOCH: 1
  WARMUP_TYPE: "constant"
  WARMUP_CONS_LR: 1e-5

MODEL:
  BACKBONE:
    NAME: "ViT-B/16"

TRAINER:
  PROMPTSRC:
    N_CTX_VISION: 4
    N_CTX_TEXT: 4
    CTX_INIT: "a photo of a"
    PREC: "fp16"
    PROMPT_DEPTH_VISION: 9
    PROMPT_DEPTH_TEXT: 9
    TEXT_LOSS_WEIGHT: 25
    IMAGE_LOSS_WEIGHT: 10
```

For the three chosen datasets, the PromptSRC config comments recommend:

```yaml
GPA_MEAN: 45
GPA_STD: 5
```

for:

```text
StanfordCars, Flowers102, FGVCAircraft, DTD, EuroSAT
```

Therefore set `GPA_MEAN=45`, `GPA_STD=5` for all three project datasets in the few-shot setting.

For the standalone PyTorch/OpenCLIP implementation:

- default backbone should remain `ViT-B/16` for the cleanest comparison;
- keep labeled batch size at `4` unless a smoke or profile run shows it cannot fit;
- use fp32 prompt optimization by default; AMP is allowed only as an explicitly validated performance variant because non-finite prompt gradients or skipped optimizer steps invalidate checkpoints;
- keep image size and normalization equivalent to CLIP/OpenCLIP's pretrained preprocessing;
- when `pretrained=openai`, resolve OpenCLIP ViT model names to the matching `*-quickgelu` variant and record that effective model name in provenance;
- keep PromptSRC's prompt counts, prompt depths, loss weights, optimizer, LR, epoch count, and GPA settings unless a documented compatibility issue is found.
- implement both textual and visual prompting behavior locally. Do not silently downgrade to text-only prompt tuning; if OpenCLIP internals require a wrapper for visual prompt tokens, build that wrapper inside `promptSRC-NC/` and document any unsupported PromptSRC component.

If the $10 Modal budget cannot support the full minimal run with `ViT-B/16`, the first fallback is `ViT-B/32` with the same experimental matrix. This changes the absolute accuracy target, but still allows a valid relative comparison between PromptSRC, PromptSRC-NC, and the shuffled-neighbor control. Record the backbone in every run artifact.

### 5.2 Base-to-novel setting

Use:

```text
configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx.yaml
```

Important differences:

```yaml
OPTIM:
  MAX_EPOCH: 20

TRAINER:
  PROMPTSRC:
    GPA_MEAN: 15
    GPA_STD: 1
```

Use this only if running the secondary base-to-novel benchmark.

---

## 6. New method configuration

Add standalone config fields, for example in a dataclass or YAML file:

```python
neighbor_k = 5
min_pairs_fraction = 0.25
fallback_k = 5  # legacy compatibility only; final protocol should already use 5
lambda_nc_max = 1.0
lambda_nc_warmup_epochs = 1
stage2_epochs = 5
stage2_lr = 0.00025
pair_batch_size = 8
pair_mode = "real"  # real | shuffled
neighbor_cache_dir = ""
unlabeled_split = "train_remain"
use_test_images = False
```

Default values should not be tuned per dataset for the main experiment.

### 6.1 Minimal knobs

The method should be kept simple. Only the following should be considered active hyperparameters:

1. `LAMBDA_NC`
2. `LAMBDA_NC_WARMUP_EPOCHS`
3. `STAGE2_EPOCHS`
4. `NEIGHBOR_K`

For the official project runs, fix them globally:

```text
LAMBDA_NC_MAX = 1.0
LAMBDA_NC_WARMUP_EPOCHS = 1
STAGE2_EPOCHS = 5 for few-shot
NEIGHBOR_K = 5 reciprocal mutual neighbors
PAIR_BATCH_SIZE = 8 for Modal budget mode, reduce to 4 only if T4 runs out of memory
```

`NEIGHBOR_K = 5` is the primary intended implementation. It is not a dataset-specific tuning choice. It is fixed globally because reciprocal top-5 balances three constraints: preserve local frozen-CLIP geometry, reject one-way hub edges through mutuality, and maintain enough connected unlabeled examples for Stage 2 to learn from.

The current codebase may still expose a legacy `neighbor_k = 1, fallback_k = 5` path. That path exists only for backward compatibility and ablation/debugging. It should not be used to define the final protocol. When analyzing artifacts, treat `neighbor_k_used` as authoritative; final PromptSRC-NC runs should be reported as mutual top-5 whenever `neighbor_k_used = 5`, even if a legacy artifact also records `neighbor_k_requested = 1`.

Use a linear Stage 2 consistency-weight warmup:

```text
lambda_nc(epoch_progress) = LAMBDA_NC_MAX * min(1.0, epoch_progress / LAMBDA_NC_WARMUP_EPOCHS)
```

where `epoch_progress` is measured from the start of Stage 2 in epochs. This is a fixed stability rule, not a sweep. It keeps the PromptSRC supervised/self-regularized anchor dominant at the start of refinement, then turns on the neighborhood consistency loss over the first epoch.

Do not add entropy minimization, pseudo-labeling, EMA teacher, relation preservation, class-balance loss, or graph weights to the main method.

---

## 7. Modal smoke tests and GPU cost profiling

### 7.1 Modal app functions

The Modal app should expose small, composable Functions:

```text
prepare_data(datasets)
smoke_test(dataset, shots, seed, backbone, gpu)
profile_gpu_cost(dataset, shots, seed, backbone, gpu, stage, steps)
build_neighbors(dataset, shots, seed, backbone)
train_stage1(dataset, shots, seed, backbone)
train_stage2(dataset, shots, seed, backbone, pair_mode)
evaluate(model_ref, dataset, shots, seed, backbone)
aggregate_results(run_id)
```

Do not make one giant Modal function that performs the entire study. Separate functions make retries cheaper and make logs easier to interpret.

### 7.2 Smoke-test policy

Smoke tests answer: "Does the code path work?" They should run before any cost-profiling or full training job.

Minimum remote smoke test:

1. attach Volumes;
2. import all project modules;
3. verify data manifest and split files exist for one dataset;
4. load one train, val, test, and unlabeled batch;
5. build the OpenCLIP model and transforms;
6. run one Stage 1 forward/backward/optimizer step;
7. build a tiny neighbor subset from at most 256 unlabeled images;
8. run one Stage 2 real-pair step and one shuffled-pair step with NC active;
9. run one evaluation batch;
10. write a smoke-test JSON artifact under `/vol/runs/{run_id}/smoke/`.

Use a cheap, representative smoke target first:

```text
dataset = eurosat
shots = 1
seed = 1
backbone = ViT-B/16
gpu = T4
max_train_batches = 1
max_eval_batches = 1
max_unlabeled_images = 256
```

Smoke should run both before and after major implementation changes. It is also the first Modal function to run after `prepare_data`.

Smoke tests are not cost benchmarks. They include setup overhead, imports, cache misses, and tiny batch effects. Use the profiling function below for T4-vs-L4 cost decisions.

### 7.3 GPU cost-profiling function

Add a dedicated profiling function:

```text
profile_gpu_cost(dataset="stanford_cars", shots=16, seed=1, backbone="ViT-B/16", gpu="T4")
```

The profiling function should run on the worst practical case, because Stanford Cars has the most classes and the largest 16-shot labeled set:

```text
dataset = stanford_cars
shots = 16
seed = 1
backbone = ViT-B/16
Stage 1 labeled batch size = 4
Stage 2 pair batch size = 4 and then 8 if memory permits
warmup_steps = 10
timed_steps = 100 or one full epoch, whichever is smaller
```

Log these fields:

```json
{
  "event": "gpu_profile",
  "gpu": "T4",
  "modal_price_per_second": 0.000164,
  "dataset": "stanford_cars",
  "shots": 16,
  "seed": 1,
  "backbone": "ViT-B/16",
  "stage": "stage1",
  "batch_size": 4,
  "pair_batch_size": null,
  "warmup_steps": 10,
  "timed_steps": 100,
  "seconds_total": 123.4,
  "seconds_per_step": 1.234,
  "images_per_second": 6.48,
  "max_cuda_memory_allocated_mb": 8200,
  "max_cuda_memory_reserved_mb": 10400,
  "cost_per_1000_steps_usd": 0.2024,
  "estimated_full_matrix_cost_usd": 7.31
}
```

Compute:

```text
cost_per_1000_steps = seconds_per_step * 1000 * modal_price_per_second
```

Run at least two profiles before selecting the main GPU:

```text
T4, Stage 1, batch 4
L4, Stage 1, batch 4
T4, Stage 2, batch 4, pair batch 8 if it fits
L4, Stage 2, batch 4, pair batch 8
```

### 7.4 T4 vs L4 decision rule

As of the Modal pricing checked for this plan:

| GPU | Modal price | Budget implication |
|---|---:|---:|
| T4 | `$0.000164/sec`, about `$0.590/hr` | cheapest hourly rate |
| L4 | `$0.000222/sec`, about `$0.799/hr` | 1.35x T4 hourly cost |

L4 is cheaper per completed run only if:

```text
runtime_L4 < 0.738 * runtime_T4
```

or equivalently if L4 is more than:

```text
1.35x faster than T4
```

The main app should therefore try T4 first for smoke tests and profiling:

```python
@app.function(gpu="T4", ...)
def smoke_test(...):
    ...
```

For production matrix runs, either:

1. set `gpu="T4"` if T4 has lower measured `cost_per_1000_steps`;
2. set `gpu="L4"` if L4 is more than 1.35x faster in the profile;
3. set `gpu=["T4", "L4"]` only when availability matters more than exact cost reproducibility.

Do not lower labeled batch size from 4 to 2 just to fit T4 unless required. Halving the labeled batch doubles optimizer steps per epoch under the current epoch-based schedule and changes the effective training setup. For budget pressure, reduce Stage 2 `PAIR_BATCH_SIZE` from 8 to 4 first, then consider switching to L4, then consider a `ViT-B/32` budget run.

### 7.5 Expected VRAM profile

PromptSRC trains only prompt parameters, so optimizer state is tiny. However, gradients still flow through the frozen image and text encoders to the prompt tokens. Activation memory remains meaningful.

Expected fit:

| Configuration | T4 16 GB expectation |
|---|---|
| Stage 1, labeled batch 4, ViT-B/16, fp16 | should fit |
| Stage 2, labeled batch 4 + pair batch 4 | should fit |
| Stage 2, labeled batch 4 + pair batch 8 | likely fit, profile first |
| Stage 2, labeled batch 4 + pair batch 16 | avoid for budget mode |
| Stage 2, labeled batch 4 + pair batch 32 | avoid on T4 |

If T4 runs out of memory:

1. reduce Stage 2 pair batch size to 4;
2. enable gradient checkpointing only if it is implemented without changing model behavior;
3. use L4;
4. demote the backbone to `ViT-B/32` for a clearly labeled budget run.

---

## 8. Unlabeled pool construction

### 8.1 Primary few-shot setting

For dataset \(D\), shot count \(K\), and seed \(s\):

1. Load the official full train split.
2. Load or generate PromptSRC’s few-shot labeled train split.
3. Define the unlabeled pool as:

\[
U = \text{full train split} \setminus \text{few-shot labeled train split}
\]

Use image paths as identifiers for set subtraction.

Do not use test images.

Do not use labels in the loss. The implementation may carry labels in `Datum` objects for bookkeeping, but they must not be used in Stage 0 or Stage 2 neighbor loss.

### 8.2 Optional base-to-novel setting

Strict default:

\[
U = \text{base-class full train split} \setminus \text{base-class 16-shot labeled train split}
\]

Do not include novel-class images in strict base-to-novel training.

If a transductive experiment is deliberately run, store it in a separate output directory and name it explicitly:

```text
base2new_transductive_unlabeled_all_classes
```

---

## 9. Neighbor-pair construction

### 9.1 Script

Create:

```text
promptsrc_nc/neighbors.py
```

Suggested CLI:

```bash
uv run python -m promptsrc_nc.neighbors \
  --data-root /vol/data \
  --run-root /vol/runs/${RUN_ID} \
  --dataset oxford_flowers \
  --seed 1 \
  --shots 16 \
  --backbone ViT-B-16 \
  --neighbor-k 5 \
  --fallback-k 5 \
  --min-pairs-fraction 0.25
```

Outputs:

```text
/vol/runs/{run_id}/neighbors/{dataset}/shot{K}/seed{seed}/
|-- unlabeled_items.jsonl
|-- features.pt
|-- real_pairs.pt
|-- shuffled_pairs.pt
|-- metadata.json
```

### 9.2 Data format

`unlabeled_items.jsonl`:

```json
{"uid": 0, "impath": "...", "label": 17, "classname": "rose"}
{"uid": 1, "impath": "...", "label": 42, "classname": "sunflower"}
```

Labels/classnames are allowed for diagnostics only. Stage 2 loss must not use them.

`features.pt`:

```python
{
    "features": torch.FloatTensor[N, 512],  # L2-normalized
    "uids": List[int],
    "impaths": List[str],
}
```

For ViT-B/16 CLIP, output feature dimension is 512.

`real_pairs.pt`:

```python
{
    "pairs": torch.LongTensor[M, 2],
    "cosine": torch.FloatTensor[M],
    "neighbor_k": int,
    "mutual": True,
}
```

`shuffled_pairs.pt`:

```python
{
    "pairs": torch.LongTensor[M, 2],
    "source": "degree_preserving_edge_swap",
    "num_swaps": int,
}
```

`metadata.json`:

```json
{
  "dataset": "oxford_flowers",
  "shots": 16,
  "seed": 1,
  "unlabeled_policy": "train_remain",
  "clip_backbone": "ViT-B/16",
  "feature_source": "frozen_unprompted_clip_before_promptsrc",
  "num_unlabeled": 7342,
  "neighbor_k_requested": 5,
  "neighbor_k_used": 5,
  "num_real_pairs": 12706,
  "mean_real_cosine": 0.87,
  "mean_shuffled_cosine": 0.43
}
```

### 9.3 Image transform for neighbor construction

Use deterministic CLIP evaluation preprocessing, not random training augmentation.

Recommended:

```text
resize shortest side / bicubic to CLIP input convention
center crop 224
normalize with CLIP mean/std
```

Use the deterministic OpenCLIP evaluation preprocessing for cached features. Do not use `random_resized_crop` or `random_flip` for cached features.

### 9.4 Feature extraction

Pseudocode:

```python
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained=pretrained,
)
clip_model.float()
clip_model.to(device)
clip_model.eval()

all_features = []
for images in unlabeled_loader:
    with torch.no_grad():
        feats = clip_model.encode_image(images.to(device))
        feats = feats / feats.norm(dim=-1, keepdim=True)
    all_features.append(feats.cpu().float())

features = torch.cat(all_features, dim=0)
```

### 9.5 Real mutual-neighbor pairs

For normalized features, cosine similarity is dot product.

For each image \(i\):

1. compute top-\(k+1\) similarities;
2. remove self;
3. store top-\(k\) neighbors.

For mutual top-\(k\):

\[
(i,j) \in \mathcal{P}
\quad \text{iff} \quad
j \in \text{kNN}(i)
\quad \text{and} \quad
i \in \text{kNN}(j).
\]

Store undirected pairs with `i < j`.

Primary final protocol:

```text
neighbor_k = 5
```

This value is chosen up front and fixed globally. Mutual top-5 is more appropriate than mutual top-1 for PromptSRC-NC because Stage 2 needs both precision and coverage:

1. Mutuality rejects one-way neighbor links and hub artifacts.
2. Top-5 keeps the graph local while giving each image more chances to form a reciprocal edge.
3. The resulting graph usually covers enough unlabeled images to make the NC loss a meaningful training signal.
4. It avoids adding extra graph hyperparameters such as edge weights, confidence thresholds, refresh schedules, or Laplacian normalization.

Legacy compatibility:

```text
neighbor_k = 1
fallback_k = 5
```

Some code paths still build mutual top-1 first and rebuild with mutual top-5 when:

```text
num_pairs < MIN_PAIRS_FRACTION * num_unlabeled
```

This should be treated as a legacy safety path or a secondary ablation, not as the final method definition. For final reports, use the metadata field `neighbor_k_used`; if it is 5, the result is a mutual top-5 result. Do not claim that such a run used strict mutual top-1.

### 9.6 Shuffled-neighbor control

Implement a degree-preserving edge shuffle by double-edge swaps.

Input:

```python
edges = set((i, j) where i < j)
```

Algorithm:

```python
for step in range(num_swaps):
    sample two distinct edges (a,b), (c,d)
    randomly choose proposed swap:
        (a,d), (c,b)
    canonicalize each edge so low index first
    reject if:
        any self-loop
        duplicate edge
        edge already exists
    otherwise:
        remove old edges
        add new edges
```

Use:

```text
num_swaps = 10 * len(edges)
```

This preserves the node degree distribution and number of edges while destroying semantic neighborhood topology.

If double-edge swapping is difficult, use endpoint permutation as a fallback, but document that it preserves edge count more strongly than exact degree.

---

## 10. Stage 1: PromptSRC training

### 10.1 Few-shot all-class training commands

Stage 1 should be implemented inside the standalone package:

```text
promptsrc_nc/train.py
```

Suggested local CLI used by the Modal function:

```bash
uv run python -m promptsrc_nc.train \
  --stage stage1 \
  --data-root /vol/data \
  --run-root /vol/runs/${RUN_ID} \
  --dataset oxford_flowers \
  --shots 16 \
  --seed 1 \
  --backbone ViT-B-16 \
  --batch-size 4 \
  --epochs 50
```

Suggested Modal call:

```bash
uv run modal run promptSRC-NC/modal_app.py::app.train_stage1 \
  --dataset oxford_flowers \
  --shots 16 \
  --seed 1 \
  --backbone ViT-B-16 \
  --gpu T4
```

Ensure the standalone config uses the correct GPA settings for these three datasets:

```yaml
GPA_MEAN: 45
GPA_STD: 5
```

### 10.2 Output directories

Stage 1 should write:

```text
/vol/runs/{run_id}/stage1/{dataset}/shot{K}/seed{seed}/{backbone}/
|-- checkpoints/
|   |-- epoch_001.pt
|   |-- ...
|   |-- final.pt
|   `-- gpa.pt
|-- logs/
|   |-- runtime.jsonl
|   |-- train.jsonl
|   `-- eval_val.jsonl
|-- config.json
`-- metrics_val.json
```

Example:

```text
/vol/runs/2026-05-10-main/stage1/oxford_flowers/shot16/seed1/ViT-B-16/
```

### 10.3 Checkpoints

Save checkpoints as plain PyTorch dictionaries:

```python
{
    "method": "PromptSRC",
    "stage": "stage1",
    "epoch": epoch,
    "backbone": "ViT-B/16",
    "model_state": model.state_dict(),
    "prompt_state": prompt_state_dict,
    "optimizer_state": optimizer.state_dict(),
    "gpa_state": gpa_state_dict,
    "config": config_dict,
    "metrics": metrics_dict,
}
```

Stage 2 should initialize from the Stage 1 GPA prompt weights, not from the non-aggregated final prompt state. Always write `gpa.pt` and record its path in Stage 2 metadata.

---

## 11. Stage 2 trainer

### 11.1 New file

Create:

```text
promptsrc_nc/train.py
promptsrc_nc/losses.py
promptsrc_nc/pair_dataset.py
```

Implement a standalone trainer class:

```python
class PromptSRCNCTrainer:
    ...
```

The trainer should share Stage 1 model code and add only neighbor-pair loading and the Stage 2 neighborhood-consistency term.

### 11.2 Loading Stage 1 checkpoint

Stage 2 should be invoked with:

```bash
uv run python -m promptsrc_nc.train \
  --stage stage2 \
  --data-root /vol/data \
  --run-root /vol/runs/${RUN_ID} \
  --dataset oxford_flowers \
  --shots 16 \
  --seed 1 \
  --backbone ViT-B-16 \
  --init-checkpoint /vol/runs/${RUN_ID}/stage1/oxford_flowers/shot16/seed1/ViT-B-16/checkpoints/gpa.pt \
  --neighbor-dir /vol/runs/${RUN_ID}/neighbors/oxford_flowers/shot16/seed1 \
  --pair-mode real \
  --batch-size 4 \
  --pair-batch-size 8 \
  --epochs 5
```

The Modal function should validate that the checkpoint method is `PromptSRC`, stage is `stage1`, backbone matches the requested backbone, and the split metadata matches the requested dataset/shot/seed.

### 11.3 Stage 2 optimizer

Use a smaller LR than Stage 1 for stability:

```text
STAGE2_LR = 0.00025
```

This is exactly one tenth of the official PromptSRC LR.

Use:

```text
OPTIM.NAME = "sgd"
OPTIM.LR = 0.00025
OPTIM.MAX_EPOCH = 5
OPTIM.LR_SCHEDULER = "cosine"
WARMUP_EPOCH = 0 or 1
```

Use a separate linear warmup for `lambda_nc` from `0` to `1.0` over the first Stage 2 epoch. This does not change the LR schedule.

Keep all model/prompt hyperparameters identical to Stage 1.

### 11.4 Pair loader

Create a lightweight dataset:

```python
class NeighborPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_path, unlabeled_items, transform):
        self.pairs = torch.load(pairs_path)["pairs"]
        self.items = load_jsonl(unlabeled_items)
        self.transform = transform

    def __getitem__(self, idx):
        i, j = self.pairs[idx].tolist()
        img_i = read_image(self.items[i]["impath"])
        img_j = read_image(self.items[j]["impath"])
        return {
            "img_i": self.transform(img_i),
            "img_j": self.transform(img_j),
            "uid_i": i,
            "uid_j": j,
        }

    def __len__(self):
        return len(self.pairs)
```

Use the same stochastic train transforms as the labeled Stage 2 training images:

```text
random_resized_crop
random_flip
normalize
```

Rationale: Stage 0 uses deterministic features to define pairs; Stage 2 trains with standard augmentations to stay aligned with PromptSRC training.

If stochasticity makes the loss noisy, switch pair images to deterministic eval transforms, but use the same choice for real and shuffled variants.

### 11.5 Training loop

Modify the Stage 1 training step in `PromptSRCNCTrainer`:

1. parse labeled batch as official PromptSRC;
2. compute official PromptSRC loss;
3. get next unlabeled pair batch;
4. compute logits for both images;
5. compute JS divergence;
6. total loss = PromptSRC loss + lambda * NC loss;
7. optimize prompts.

Pseudocode:

```python
def forward_backward(self, batch):
    image, label = self.parse_batch_train(batch)

    (
        loss_ce,
        normalized_text_features,
        zs_clip_text_embeddings,
        zs_image_embedd,
        image_ft,
        zero_shot_logits,
        logits
    ) = self.model(image, label)

    loss_scl_text = ...
    loss_scl_image = ...
    loss_scl_logits = ...
    loss_promptsrc = loss_ce + loss_scl_text + loss_scl_image + loss_scl_logits

    pair_batch = next(self.pair_iter)
    img_i = pair_batch["img_i"].to(self.device)
    img_j = pair_batch["img_j"].to(self.device)

    logits_i = self.model(img_i)  # model eval-mode path? See warning below.
    logits_j = self.model(img_j)

    p_i = F.softmax(logits_i, dim=-1)
    p_j = F.softmax(logits_j, dim=-1)
    loss_nc = js_divergence(p_i, p_j)

    lambda_nc = config.lambda_nc_max * min(1.0, epoch_progress / config.lambda_nc_warmup_epochs)
    loss = loss_promptsrc + lambda_nc * loss_nc

    optim.zero_grad()
    loss.backward()
    optim.step()
```

### 11.6 Important warning about logits-only forward

In the official `CustomCLIP.forward`, behavior depends on:

```python
if self.prompt_learner.training:
    return training tuple
else:
    return logits
```

During Stage 2, the model is in train mode, so calling:

```python
self.model(img_i)
```

without labels may not work, because the training path expects labels and returns a tuple with CE.

Standalone implementation fix:

Add a method to the model class:

```python
def forward_logits(self, image):
    tokenized_prompts = self.tokenized_prompts
    logit_scale = self.logit_scale.exp()
    prompts = self.prompt_learner()
    text_features = self.text_encoder(prompts, tokenized_prompts)
    image_features = self.image_encoder(image.type(self.dtype))
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t()
    return logits
```

Use this for unlabeled neighbor images:

```python
logits_i = self.model.forward_logits(img_i)
logits_j = self.model.forward_logits(img_j)
```

Do not set the whole model to eval mode inside training, because prompts need gradients.

### 11.7 JS divergence implementation

Use a stable symmetric implementation:

```python
def js_divergence_from_logits(logits_a, logits_b, eps=1e-8):
    p = F.softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    m = 0.5 * (p + q)

    log_p = torch.log(p.clamp_min(eps))
    log_q = torch.log(q.clamp_min(eps))
    log_m = torch.log(m.clamp_min(eps))

    kl_pm = torch.sum(p * (log_p - log_m), dim=-1)
    kl_qm = torch.sum(q * (log_q - log_m), dim=-1)

    js = 0.5 * (kl_pm + kl_qm)
    return js.mean()
```

No stop-gradient. No sharpening. No temperature by default.

### 11.8 GPA in Stage 2

Recommended:

- Disable new GPA aggregation during Stage 2 unless the implementation naturally supports a short second-stage GPA.
- The Stage 2 checkpoint should simply be the final Stage 2 prompt weights.

Rationale:

- Stage 1 already uses PromptSRC’s GPA and initializes Stage 2 from GPA weights.
- A second GPA introduces another schedule/knob and muddies interpretation.
- Stage 2 is short and primarily a local refinement.

Implementation options:

1. Add config:
   ```yaml
   TRAINER:
     PROMPTSRC_NC:
       USE_STAGE2_GPA: False
   ```
2. Override GPA update logic in `PromptSRC_NC.forward_backward` to skip `previous_model_gpa` accumulation.

If skipping GPA requires too much code change, keep GPA but use a fixed mean centered in the Stage 2 schedule and document it. The cleaner choice is no Stage 2 GPA.

---

## 12. Stage 2 configs and Modal functions

### 12.1 Config file

Create:

```text
promptSRC-NC/configs/vit_b16_ep5_batch4_4ctx_nc.yaml
```

Base it on the PromptSRC few-shot config, with changes:

```yaml
OPTIM:
  NAME: "sgd"
  LR: 0.00025
  MOMENTUM: 0.9
  WEIGHT_DECAY: 0.0005
  MAX_EPOCH: 5
  LR_SCHEDULER: "cosine"
  WARMUP_EPOCH: 0

TRAINER:
  NAME: "PromptSRC_NC"

  PROMPTSRC:
    N_CTX_VISION: 4
    N_CTX_TEXT: 4
    CTX_INIT: "a photo of a"
    PREC: "fp16"
    PROMPT_DEPTH_VISION: 9
    PROMPT_DEPTH_TEXT: 9
    TEXT_LOSS_WEIGHT: 25
    IMAGE_LOSS_WEIGHT: 10
    GPA_MEAN: 45
    GPA_STD: 5

  PROMPTSRC_NC:
    NEIGHBOR_K: 1
    FALLBACK_K: 5
    MIN_PAIRS_FRACTION: 0.25
    LAMBDA_NC_MAX: 1.0
    LAMBDA_NC_WARMUP_EPOCHS: 1
    STAGE2_EPOCHS: 5
    STAGE2_LR: 0.00025
    PAIR_BATCH_SIZE: 8
    PAIR_MODE: "real"
    NEIGHBOR_CACHE_DIR: ""
    UNLABELED_SPLIT: "train_remain"
    USE_TEST_IMAGES: False
    USE_STAGE2_GPA: False
```

### 12.2 Stage 2 real-neighbor Modal call

```bash
uv run modal run promptSRC-NC/modal_app.py::app.train_stage2 \
  --dataset oxford_flowers \
  --shots 16 \
  --seed 1 \
  --backbone ViT-B-16 \
  --pair-mode real \
  --gpu T4
```

The Modal function should dispatch the equivalent internal command:

```bash
uv run python -m promptsrc_nc.train \
  --stage stage2 \
  --data-root /vol/data \
  --run-root /vol/runs/${RUN_ID} \
  --dataset ${DATASET} \
  --shots ${SHOTS} \
  --seed ${SEED} \
  --backbone ${BACKBONE} \
  --pair-mode real
```

### 12.3 Stage 2 shuffled-neighbor Modal call

Same as above, except:

```bash
--pair-mode shuffled
```

and output directory:

```text
/vol/runs/{run_id}/stage2/{dataset}/shot{K}/seed{seed}/{backbone}/shuffled/
```

---

## 13. Evaluation

### 13.1 Few-shot all-class evaluation

For each dataset, seed, and shot count, evaluate:

1. PromptSRC baseline checkpoint.
2. Real-neighbor Stage 2 checkpoint.
3. Shuffled-neighbor Stage 2 checkpoint.

Use the same test split and the standalone evaluation code.

Suggested Modal call:

```bash
uv run modal run promptSRC-NC/modal_app.py::app.evaluate \
  --dataset oxford_flowers \
  --shots 16 \
  --seed 1 \
  --backbone ViT-B-16 \
  --checkpoint-ref stage2-real
```

Internal command:

```bash
uv run python -m promptsrc_nc.eval \
  --data-root /vol/data \
  --run-root /vol/runs/${RUN_ID} \
  --dataset ${DATASET} \
  --shots ${SHOTS} \
  --seed ${SEED} \
  --backbone ${BACKBONE} \
  --checkpoint ${CHECKPOINT}
```

### 13.2 Metrics

Primary metric:

```text
Top-1 accuracy on official test split
```

Report:

```text
mean accuracy over seeds
standard deviation over seeds
```

For each dataset:

```text
Flowers102
EuroSAT
Stanford Cars
Average over 3 datasets
```

### 13.3 Base-to-novel evaluation if included

Metrics:

```text
Base accuracy
Novel accuracy
Harmonic mean = 2 * Base * Novel / (Base + Novel)
```

Use PromptSRC’s base-to-novel output structure.

---

## 14. Diagnostics implementation

Create:

```text
promptsrc_nc/diagnostics.py
```

Inputs:

```bash
uv run python -m promptsrc_nc.diagnostics \
  --checkpoint ${CHECKPOINT} \
  --neighbors /vol/runs/${RUN_ID}/neighbors/${DATASET}/shot${SHOTS}/seed${SEED}/real_pairs.pt \
  --items /vol/runs/${RUN_ID}/neighbors/${DATASET}/shot${SHOTS}/seed${SEED}/unlabeled_items.jsonl \
  --dataset ${DATASET} \
  --run-root /vol/runs/${RUN_ID}
```

Outputs JSON:

```json
{
  "edge_disagree": 0.21,
  "mean_js": 0.034,
  "mean_entropy": 1.74,
  "mean_confidence": 0.62,
  "num_pairs": 2801
}
```

### 14.1 Edge disagreement

```python
pred_i = logits_i.argmax(dim=-1)
pred_j = logits_j.argmax(dim=-1)
edge_disagree = (pred_i != pred_j).float().mean()
```

### 14.2 Mean JS

Reuse `js_divergence_from_logits`, but collect per-pair values.

### 14.3 Entropy

```python
p = F.softmax(logits, dim=-1)
entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1)
```

---

## 15. Logging and artifacts

Logging is a first-class requirement. Every Modal function should write human-readable stdout logs and durable structured JSONL logs under `/vol/runs/{run_id}/logs/` or the function-specific output directory.

### 15.1 Runtime logs

Every function should emit JSONL records with:

```json
{
  "event": "runtime_step",
  "run_id": "2026-05-10-main",
  "function": "train_stage1",
  "dataset": "stanford_cars",
  "shots": 16,
  "seed": 1,
  "backbone": "ViT-B/16",
  "gpu": "T4",
  "modal_function_call_id": "...",
  "modal_region": "...",
  "hostname": "...",
  "pid": 123,
  "timestamp": "2026-05-10T00:00:00Z",
  "step": 250,
  "epoch": 1,
  "seconds_since_start": 312.5,
  "seconds_per_step_ema": 1.21,
  "cpu_percent": 78.3,
  "rss_mb": 4200,
  "cuda_memory_allocated_mb": 7600,
  "cuda_memory_reserved_mb": 9300,
  "gpu_utilization_percent": 62,
  "gpu_memory_used_mb": 10120
}
```

Use `time.perf_counter()` for timing, `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()` for PyTorch memory, and NVML through `pynvml` when available for GPU utilization and total memory use.

### 15.2 Training logs

Stage 1 and Stage 2 should log one JSON object at a fixed interval, for example every 20 optimizer steps and every epoch end:

```json
{
  "event": "train_step",
  "stage": "stage2",
  "pair_mode": "real",
  "dataset": "oxford_flowers",
  "shots": 16,
  "seed": 1,
  "epoch": 3,
  "step": 740,
  "global_step": 4820,
  "lr": 0.00018,
  "loss_total": 1.234,
  "loss_ce": 0.456,
  "loss_scl_text": 0.111,
  "loss_scl_image": 0.222,
  "loss_scl_logits": 0.033,
  "loss_nc": 0.041,
  "lambda_nc": 0.74,
  "lambda_nc_max": 1.0,
  "lambda_nc_warmup_epochs": 1,
  "batch_size": 4,
  "pair_batch_size": 8,
  "grad_norm_prompt": 0.87,
  "seconds_per_step": 1.19
}
```

Stage 2 logs must include `loss_nc` for both real and shuffled runs so the two controls can be compared directly.

### 15.3 Evaluation and diagnostics logs

Evaluation should write:

```json
{
  "event": "eval_summary",
  "method": "PromptSRC-NC",
  "pair_mode": "real",
  "dataset": "eurosat",
  "shots": 16,
  "seed": 1,
  "split": "test",
  "top1_accuracy": 0.0,
  "macro_accuracy": 0.0,
  "num_examples": 0,
  "checkpoint": "/vol/runs/.../checkpoints/final.pt"
}
```

Diagnostics should write:

```json
{
  "event": "neighbor_diagnostics",
  "method": "PromptSRC-NC",
  "pair_mode": "real",
  "dataset": "stanford_cars",
  "shots": 16,
  "seed": 1,
  "num_unlabeled": 5008,
  "num_real_pairs": 2801,
  "neighbor_k": 5,
  "edge_disagreement": 0.21,
  "mean_js": 0.034,
  "mean_entropy": 1.74,
  "mean_confidence": 0.62,
  "mean_real_cosine": 0.87,
  "mean_shuffled_cosine": 0.43
}
```

### 15.4 Aggregation outputs

The `aggregate_results` function should write machine-readable files that can be plotted without scraping console logs:

```text
/vol/runs/{run_id}/results/runs.jsonl
/vol/runs/{run_id}/results/eval_summary.csv
/vol/runs/{run_id}/results/eval_summary.json
/vol/runs/{run_id}/results/diagnostics_summary.csv
/vol/runs/{run_id}/results/runtime_summary.csv
/vol/runs/{run_id}/results/cost_profile_summary.csv
```

Minimum aggregate columns:

```text
run_id, method, pair_mode, dataset, shots, seed, backbone, gpu,
checkpoint, val_top1, test_top1, test_macro, edge_disagreement,
mean_js, mean_entropy, mean_confidence, train_seconds,
eval_seconds, estimated_gpu_cost_usd
```

### 15.5 Modal runtime logs

Keep Modal stdout/stderr useful. Use concise progress lines with dataset, seed, stage, epoch, step, loss, accuracy, seconds/step, and GPU memory. Do not rely on stdout as the only record; structured JSONL logs are the source of truth for plotting and analysis.

---

## 16. Run matrix

### 16.1 Minimal capstone run

Datasets:

```text
oxford_flowers
eurosat
stanford_cars
```

Shots:

```text
16
```

Seeds:

```text
1, 2, 3
```

Variants:

```text
PromptSRC
PromptSRC_NC real
PromptSRC_NC shuffled
```

Total training runs:

```text
3 datasets × 3 seeds × 1 PromptSRC = 9 Stage 1 runs
3 datasets × 3 seeds × 2 Stage 2 variants = 18 Stage 2 runs
Total = 27 runs
```

Stage 2 runs are short and initialized from Stage 1.

### 16.2 Expanded run if compute allows

Add shots:

```text
1, 4
```

This tests whether gains are larger in more label-scarce settings.

---

## 17. Expected output tables

### 17.1 Few-shot result table

| Dataset | Shots | PromptSRC | PromptSRC-NC shuffled | PromptSRC-NC real | Real - PromptSRC | Real - Shuffled |
|---|---:|---:|---:|---:|---:|---:|
| Flowers102 | 16 | mean±std | mean±std | mean±std | Δ | Δ |
| EuroSAT | 16 | mean±std | mean±std | mean±std | Δ | Δ |
| Stanford Cars | 16 | mean±std | mean±std | mean±std | Δ | Δ |
| Average | 16 | mean | mean | mean | Δ | Δ |

### 17.2 Diagnostics table

| Dataset | Variant | Edge disagreement | Mean JS | Mean entropy | Mean confidence |
|---|---|---:|---:|---:|---:|
| Flowers102 | PromptSRC | | | | |
| Flowers102 | PromptSRC-NC real | | | | |
| EuroSAT | PromptSRC | | | | |
| EuroSAT | PromptSRC-NC real | | | | |
| Stanford Cars | PromptSRC | | | | |
| Stanford Cars | PromptSRC-NC real | | | | |

Compute diagnostics on the **real neighbor pairs** for both PromptSRC and real-neighbor-adapted PromptSRC.

---

## 18. Failure handling

### 18.1 If Stage 2 diverges

Symptoms:

- loss becomes NaN;
- entropy collapses;
- accuracy sharply drops;
- predictions become one-class dominated.

Actions, in order:

1. Confirm the one-epoch `lambda_nc` warmup is active.
2. Reduce `LAMBDA_NC_MAX` from 1.0 to 0.1.
3. Reduce Stage 2 LR from 2.5e-4 to 1e-4.
4. Use deterministic transforms for pair images.
5. Reduce pair batch size if memory issue, not for stability.
6. Do not add new losses.

### 18.2 If a run records `neighbor_k_requested = 1`

Interpret this as the legacy construction path. The final protocol is determined by `neighbor_k_used`. If `neighbor_k_used = 5`, report the run as mutual top-5. If `neighbor_k_used = 1`, mark it as a secondary top-1 ablation and do not mix it with the primary PromptSRC-NC results.

### 18.3 If shuffled graph outperforms real graph

Do not hide it. This means the geometry claim is not supported on that dataset or setting. Interpret as generic smoothing/extra training, or possible CLIP-neighborhood mismatch.

### 18.4 If real graph helps EuroSAT but hurts Stanford Cars

This is a scientifically plausible outcome. The conclusion should be dataset-conditional:

> Unlabeled CLIP-neighborhood consistency is helpful when frozen CLIP neighborhoods align with task semantics, but can hurt in fine-grained regimes where visual neighbors cross label boundaries.

---

## 19. Implementation checklist

### Modal infrastructure

- [ ] Add `promptSRC-NC/modal_app.py`.
- [ ] Add uv-based Modal image definition.
- [ ] Add Modal Volumes for data, weights, and runs.
- [ ] Add Kaggle Modal Secret support for data preparation.
- [ ] Add `prepare_data` function.
- [ ] Add `smoke_test` function.
- [ ] Add `profile_gpu_cost` function.
- [ ] Add run-id and artifact path conventions.

### Standalone package

- [ ] Add `promptSRC-NC/promptsrc_nc/` package.
- [ ] Add standalone data preprocessing.
- [ ] Add standalone split creation/loading.
- [ ] Add PyTorch/OpenCLIP model implementation.
- [ ] Add standalone Stage 1 trainer.
- [ ] Add standalone Stage 2 trainer.
- [ ] Add standalone evaluation module.
- [ ] Add structured logging utilities.

### Stage 0

- [ ] Build deterministic unlabeled loader.
- [ ] Extract frozen CLIP image features.
- [ ] Save normalized features.
- [ ] Build real reciprocal mutual top-5 neighbor pairs.
- [ ] Build degree-preserving shuffled pairs.
- [ ] Save metadata and sanity stats.

### Stage 1

- [ ] Run official PromptSRC for each dataset/seed/shot.
- [ ] Confirm GPA checkpoint is saved or add explicit GPA save.
- [ ] Evaluate PromptSRC baseline.
- [ ] Parse results over seeds.

### Stage 2

- [ ] Add `PromptSRC_NC` trainer.
- [ ] Add config node for neighbor settings.
- [ ] Add `forward_logits` to `CustomCLIP`.
- [ ] Add neighbor-pair dataloader.
- [ ] Implement JS loss.
- [ ] Implement real/shuffled pair selection.
- [ ] Initialize from Stage 1 checkpoint.
- [ ] Train real-neighbor adaptation.
- [ ] Train shuffled-neighbor adaptation.
- [ ] Evaluate both.

### Diagnostics

- [ ] Edge disagreement.
- [ ] Mean JS on real edges.
- [ ] Mean entropy on unlabeled pool.
- [ ] Neighbor metadata table.
- [ ] Runtime and cost profile summaries.
- [ ] Qualitative nearest-neighbor visualization, optional but useful.

---

## 20. Non-goals

Do not implement these in the main version:

- pseudo-labeling;
- entropy minimization;
- confidence thresholding;
- EMA teacher;
- PromptKD-style teacher-student distillation;
- relation preservation;
- optimal transport;
- graph Laplacian normalization variants;
- periodically refreshed graphs;
- per-dataset hyperparameter tuning.
- conda-based environments;
- code that depends on editing an external clone of the official PromptSRC repo;
- cloud runs that rely on files outside `promptSRC-NC/` except mounted Modal Volumes.

These may be future work, but they weaken the central experiment if included now.

---

## 21. Final intended claim

If results support the method, the final report can claim:

> A short, teacher-free unlabeled adaptation stage improves PromptSRC when it uses meaningful frozen-CLIP neighbor structure. The improvement is not explained by extra training alone, because the shuffled-neighbor control uses the same extra training and same unlabeled images but destroys the local geometry.

If results are mixed, the final report can claim:

> The usefulness of unlabeled CLIP-neighborhood regularization depends on whether frozen CLIP neighborhoods align with downstream class semantics; this appears dataset-dependent and is especially fragile for fine-grained categories.

Both conclusions are scientifically useful.
