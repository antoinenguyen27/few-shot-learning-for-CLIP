"""PromptKD integration placeholder."""

from __future__ import annotations


class PromptKDMethod:
    """Adapter boundary for PromptKD.

    Log teacher model, distillation data source, and strict/transductive setting
    in RunResult.extra when implementing this class.
    """

    method_name = "PromptKD"

    def fit(self, train_loader, val_loader, classnames, model_bundle):
        """Train/distill PromptKD without leaking test-only information."""

        raise NotImplementedError("Implement PromptKD training against the common data/model contract.")

    def evaluate(self, artifact, test_loader, classnames, model_bundle):
        """Return standardized metrics for test_loader."""

        raise NotImplementedError("Implement PromptKD evaluation against the common data/model contract.")
