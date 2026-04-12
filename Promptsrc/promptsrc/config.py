"""Configuration for the PromptSRC OpenCLIP port."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, pi, sqrt


@dataclass(frozen=True)
class PromptSRCConfig:
    """Hyperparameters for the PromptSRC method-local implementation.

    The defaults mirror the public PromptSRC few-shot/base-to-new configs where
    possible, while preserving this repo's shared OpenCLIP model and splits.
    """

    epochs: int = 50
    lr: float = 0.0025
    weight_decay: float = 0.0
    warmup_epochs: int = 1
    batch_size: int = 4
    n_ctx_text: int = 4
    ctx_init: str = "a photo of a"
    text_loss_weight: float = 25.0
    image_loss_weight: float = 10.0
    logit_loss_weight: float = 1.0
    gpa_mean: float = 15.0
    gpa_std: float = 1.0
    use_gpa: bool = True
    precision: str = "fp32"
    trainable_vision_prompts: bool = False
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    show_progress: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.n_ctx_text < 1:
            raise ValueError("n_ctx_text must be >= 1")
        if self.gpa_std <= 0:
            raise ValueError("gpa_std must be > 0")
        if self.precision not in {"fp32", "amp"}:
            raise ValueError("precision must be one of: fp32, amp")
        if self.trainable_vision_prompts:
            raise ValueError(
                "trainable_vision_prompts=True is not supported by this stock OpenCLIP port. "
                "Keep it False unless a method-local visual prompt wrapper is added."
            )
        if self.max_train_batches is not None and self.max_train_batches < 1:
            raise ValueError("max_train_batches must be >= 1 when set")
        if self.max_eval_batches is not None and self.max_eval_batches < 1:
            raise ValueError("max_eval_batches must be >= 1 when set")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def gaussian_epoch_weights(epochs: int, mean: float, std: float) -> list[float]:
    """Return normalized Gaussian epoch weights used for GPA."""

    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if std <= 0:
        raise ValueError("std must be > 0")
    coeff = 1.0 / (std * sqrt(2.0 * pi))
    weights = [coeff * exp(-0.5 * ((epoch - mean) / std) ** 2) for epoch in range(1, epochs + 1)]
    total = sum(weights)
    if total == 0.0:
        return [1.0 / epochs] * epochs
    return [weight / total for weight in weights]
