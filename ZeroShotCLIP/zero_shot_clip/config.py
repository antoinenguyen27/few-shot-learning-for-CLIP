"""Configuration for the zero-shot CLIP baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ZeroShotCLIPConfig:
    """Runtime options for frozen zero-shot evaluation."""

    precision: str = "fp32"
    max_eval_batches: int | None = None
    show_progress: bool = True

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "amp"}:
            raise ValueError("precision must be one of: fp32, amp")
        if self.max_eval_batches is not None and self.max_eval_batches < 1:
            raise ValueError("max_eval_batches must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
