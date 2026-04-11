"""PromptSRC integration placeholder.

Method owners should connect their implementation to common.methods.FewShotMethod.
"""

from __future__ import annotations


class PromptSRCMethod:
    """Adapter boundary for PromptSRC.

    train_loader, val_loader, and test_loader should come from
    common.datasets.torch_dataset.build_data_loaders. model_bundle should come
    from common.models.openclip.build_openclip_bundle.
    """

    method_name = "PromptSRC"

    def fit(self, train_loader, val_loader, classnames, model_bundle):
        """Train PromptSRC prompts using only train_loader and val_loader."""

        raise NotImplementedError("Implement PromptSRC training against the common data/model contract.")

    def evaluate(self, artifact, test_loader, classnames, model_bundle):
        """Return standardized metrics for test_loader."""

        raise NotImplementedError("Implement PromptSRC evaluation against the common data/model contract.")
