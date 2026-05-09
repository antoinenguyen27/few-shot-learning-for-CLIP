"""Configuration for the DPC OpenCLIP port."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DPCConfig:
    """Hyperparameters for the repo-native DPC implementation.

    DPC is a plug-in over a tuned prompt backbone. This port uses a
    PromptSRC-style text prompt as the default backbone because the repo's first
    implemented prompt backbone is PromptSRC.
    """

    backbone_method: str = "PromptSRC-text"
    backbone_epochs: int = 20
    dpc_epochs: int = 20
    lr: float = 0.0025
    weight_decay: float = 0.0
    warmup_epochs: int = 1
    batch_size: int = 4
    n_ctx_text: int = 4
    ctx_init: str = "a photo of a"
    stack_weight: float = 0.2
    hard_negative_weight: float = 1.0
    retain_weight: float = 1.0
    prompt_proximity_weight: float = 0.1
    hard_negative_topk: int = 8
    precision: str = "fp32"
    annotation_path: str | None = None
    backbone_checkpoint: str | None = None
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    show_progress: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.backbone_epochs < 0:
            raise ValueError("backbone_epochs must be >= 0")
        if self.dpc_epochs < 1:
            raise ValueError("dpc_epochs must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.n_ctx_text < 1:
            raise ValueError("n_ctx_text must be >= 1")
        if not 0.0 <= self.stack_weight <= 1.0:
            raise ValueError("stack_weight must be in [0, 1]")
        if self.hard_negative_topk < 1:
            raise ValueError("hard_negative_topk must be >= 1")
        if self.precision not in {"fp32", "amp"}:
            raise ValueError("precision must be one of: fp32, amp")
        if self.max_train_batches is not None and self.max_train_batches < 1:
            raise ValueError("max_train_batches must be >= 1 when set")
        if self.max_eval_batches is not None and self.max_eval_batches < 1:
            raise ValueError("max_eval_batches must be >= 1 when set")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
