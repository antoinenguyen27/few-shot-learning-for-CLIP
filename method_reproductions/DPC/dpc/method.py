"""DPC method adapter for the shared few-shot CLIP pipeline."""

from __future__ import annotations

from typing import Any

from common.datasets.templates import get_templates

from .config import DPCConfig
from .trainer import DPCArtifact, DPCTrainer


class DPCMethod:
    """DPC OpenCLIP port over a PromptSRC-style text-prompt backbone.

    DPC is a plug-in over another prompt method. Store the backbone method and
    checkpoint provenance in RunResult.extra when implementing this class.
    """

    method_name = "DPC"

    def __init__(self, config: DPCConfig | None = None, templates: list[str] | None = None):
        self.config = config or DPCConfig()
        self.templates = templates
        self.trainer = DPCTrainer(self.config)
        self.model: Any | None = None

    def fit(self, train_loader, val_loader, classnames, model_bundle) -> DPCArtifact:
        """Fit DPC using the approved train/validation loaders."""

        templates = self.templates or list(get_templates(self._infer_dataset_name(train_loader)))
        self.model = self.trainer.build_model(model_bundle, list(classnames), templates)
        artifact = self.trainer.train(self.model, train_loader, val_loader)
        artifact.metadata.update(
            {
                "templates": templates,
                "model_name": model_bundle.model_name,
                "pretrained": model_bundle.pretrained,
            }
        )
        return artifact

    def evaluate(self, artifact, test_loader, classnames, model_bundle, metric_prefix: str = "test"):
        """Return standardized metrics for test_loader."""

        if self.model is None:
            templates = self.templates or ["a photo of a {}."]
            self.model = self.trainer.build_model(model_bundle, list(classnames), templates)
        self.model.load_trainable_state_dict(artifact.state_dict)
        return self.trainer.evaluate(self.model, test_loader, prefix=metric_prefix)

    def _infer_dataset_name(self, train_loader) -> str:
        dataset = getattr(train_loader, "dataset", None)
        records = getattr(dataset, "records", None)
        if records:
            return str(records[0].dataset)
        raise ValueError(
            "DPCMethod could not infer the dataset name from the train loader. "
            "Pass dataset-specific templates with DPCMethod(templates=[...])."
        )
