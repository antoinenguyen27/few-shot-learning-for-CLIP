"""DPC integration placeholder."""

from __future__ import annotations


class DPCMethod:
    """Adapter boundary for DPC.

    DPC is a plug-in over another prompt method. Store the backbone method and
    checkpoint provenance in RunResult.extra when implementing this class.
    """

    method_name = "DPC"

    def fit(self, train_loader, val_loader, classnames, model_bundle):
        """Fit DPC using the approved train/validation loaders."""

        raise NotImplementedError("Implement DPC training against the common data/model contract.")

    def evaluate(self, artifact, test_loader, classnames, model_bundle):
        """Return standardized metrics for test_loader."""

        raise NotImplementedError("Implement DPC evaluation against the common data/model contract.")
