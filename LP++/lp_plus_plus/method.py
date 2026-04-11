"""LP++ integration placeholder.

The top-level LP++ directory is kept for readability; this package name is
Python-import safe.
"""

from __future__ import annotations


class LPPlusPlusMethod:
    """Adapter boundary for LP++.

    LP++ should use the common OpenCLIP feature helpers and feature-cache paths.
    Do not sample few-shot data inside this class; consume split-derived loaders.
    """

    method_name = "LP++"

    def fit(self, train_loader, val_loader, classnames, model_bundle):
        """Fit the LP++ classifier from shared OpenCLIP features."""

        raise NotImplementedError("Implement LP++ training against the common data/model contract.")

    def evaluate(self, artifact, test_loader, classnames, model_bundle):
        """Return standardized metrics for test_loader."""

        raise NotImplementedError("Implement LP++ evaluation against the common data/model contract.")
