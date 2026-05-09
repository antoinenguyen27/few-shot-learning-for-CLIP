"""Configuration objects and path conventions for PromptSRC-NC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import exp, pi, sqrt
from pathlib import Path
from typing import Any


DATASETS = ("oxford_flowers", "eurosat", "stanford_cars")
PRIMARY_VARIANTS = ("promptsrc", "promptsrc_nc_real", "promptsrc_nc_shuffled")

MODAL_GPU_PRICE_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
}


DATASET_ALIASES = {
    "flowers102": "oxford_flowers",
    "flowers": "oxford_flowers",
    "oxfordflowers": "oxford_flowers",
    "oxford_flowers": "oxford_flowers",
    "eurosat": "eurosat",
    "stanfordcars": "stanford_cars",
    "stanford_cars": "stanford_cars",
    "cars": "stanford_cars",
}


DISPLAY_NAMES = {
    "oxford_flowers": "Flowers102",
    "eurosat": "EuroSAT",
    "stanford_cars": "Stanford Cars",
}


KAGGLE_HANDLES = {
    "eurosat": "apollo2506/eurosat-dataset",
    "oxford_flowers": "nunenuh/pytorch-challange-flower-dataset",
    "stanford_cars": "eduardo4jesus/stanford-cars-dataset",
}


def canonical_dataset(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace("/", "_")
    try:
        return DATASET_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset {name!r}; expected one of {', '.join(DATASETS)}") from exc


def canonical_backbone(backbone: str) -> str:
    key = backbone.strip().replace("_", "-")
    aliases = {
        "ViT-B/16": "ViT-B-16",
        "ViT-B-16": "ViT-B-16",
        "vit-b-16": "ViT-B-16",
        "ViT-B/32": "ViT-B-32",
        "ViT-B-32": "ViT-B-32",
        "vit-b-32": "ViT-B-32",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("Supported backbones are ViT-B/16, ViT-B-16, ViT-B/32, and ViT-B-32") from exc


def backbone_slug(backbone: str) -> str:
    return canonical_backbone(backbone).replace("/", "-")


def now_run_id(prefix: str = "manual") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


def gaussian_epoch_weights(epochs: int, mean: float, std: float) -> list[float]:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if std <= 0:
        raise ValueError("std must be > 0")
    coeff = 1.0 / (std * sqrt(2.0 * pi))
    weights = [coeff * exp(-0.5 * ((epoch - mean) / std) ** 2) for epoch in range(1, epochs + 1)]
    total = sum(weights)
    if total <= 0:
        return [1.0 / epochs for _ in range(epochs)]
    return [weight / total for weight in weights]


@dataclass(frozen=True)
class PromptSRCNCConfig:
    """Global method configuration for the three PromptSRC-NC variants."""

    dataset: str = "eurosat"
    shots: int = 16
    seed: int = 1
    protocol: str = "few_shot_all_classes"
    backbone: str = "ViT-B-16"
    pretrained: str = "openai"
    image_size: int = 224
    batch_size: int = 4
    eval_batch_size: int = 100
    pair_batch_size: int = 8
    num_workers: int = 8
    stage1_epochs: int = 50
    stage2_epochs: int = 5
    stage1_lr: float = 0.0025
    stage2_lr: float = 0.00025
    weight_decay: float = 0.0
    warmup_epochs: int = 1
    warmup_cons_lr: float = 1e-5
    n_ctx_text: int = 4
    n_ctx_vision: int = 4
    ctx_init: str = "a photo of a"
    prompt_depth_text: int = 9
    prompt_depth_vision: int = 9
    text_loss_weight: float = 25.0
    image_loss_weight: float = 10.0
    logit_loss_weight: float = 1.0
    gpa_mean: float = 45.0
    gpa_std: float = 5.0
    precision: str = "amp"
    neighbor_k: int = 1
    fallback_k: int = 5
    min_pairs_fraction: float = 0.25
    lambda_nc_max: float = 1.0
    lambda_nc_warmup_epochs: float = 1.0
    pair_mode: str = "real"
    use_stage2_gpa: bool = False
    unlabeled_split: str = "train_remain"
    use_test_images: bool = False
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    max_unlabeled_images: int | None = None
    log_interval: int = 20
    save_every: int = 1
    run_id: str = field(default_factory=now_run_id)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset", canonical_dataset(self.dataset))
        object.__setattr__(self, "backbone", canonical_backbone(self.backbone))
        if self.shots < 1:
            raise ValueError("shots must be >= 1")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.batch_size < 1 or self.eval_batch_size < 1 or self.pair_batch_size < 1:
            raise ValueError("batch sizes must be >= 1")
        if self.stage1_epochs < 1 or self.stage2_epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.precision not in {"fp32", "fp16", "amp"}:
            raise ValueError("precision must be one of fp32, fp16, amp")
        if self.neighbor_k < 1 or self.fallback_k < 1:
            raise ValueError("neighbor k values must be >= 1")
        if self.pair_mode not in {"real", "shuffled"}:
            raise ValueError("pair_mode must be real or shuffled")
        if self.use_test_images:
            raise ValueError("Primary PromptSRC-NC setting forbids test images in the unlabeled pool")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stage1_dir(run_root: str | Path, run_id: str, dataset: str, shots: int, seed: int, backbone: str) -> Path:
    return (
        Path(run_root)
        / run_id
        / "stage1"
        / canonical_dataset(dataset)
        / f"shot{shots}"
        / f"seed{seed}"
        / backbone_slug(backbone)
    )


def stage2_dir(
    run_root: str | Path,
    run_id: str,
    dataset: str,
    shots: int,
    seed: int,
    backbone: str,
    pair_mode: str,
) -> Path:
    return (
        Path(run_root)
        / run_id
        / "stage2"
        / canonical_dataset(dataset)
        / f"shot{shots}"
        / f"seed{seed}"
        / backbone_slug(backbone)
        / pair_mode
    )


def neighbor_dir(run_root: str | Path, run_id: str, dataset: str, shots: int, seed: int) -> Path:
    return Path(run_root) / run_id / "neighbors" / canonical_dataset(dataset) / f"shot{shots}" / f"seed{seed}"


def results_dir(run_root: str | Path, run_id: str) -> Path:
    return Path(run_root) / run_id / "results"


def logs_dir(run_root: str | Path, run_id: str) -> Path:
    return Path(run_root) / run_id / "logs"

