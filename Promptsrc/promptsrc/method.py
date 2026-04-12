"""PromptSRC method adapter for the shared few-shot CLIP pipeline."""

from __future__ import annotations

from typing import Any

from common.datasets.templates import get_templates

from .config import PromptSRCConfig
from .trainer import PromptSRCArtifact, PromptSRCTrainer


class PromptSRCMethod:
    """PromptSRC OpenCLIP port.

    train_loader, val_loader, and test_loader should come from
    common.datasets.torch_dataset.build_data_loaders. model_bundle should come
    from common.models.openclip.build_openclip_bundle.
    """

    method_name = "PromptSRC"

    def __init__(self, config: PromptSRCConfig | None = None, templates: list[str] | None = None):
        self.config = config or PromptSRCConfig()
        self.templates = templates
        self.trainer = PromptSRCTrainer(self.config)
        self.model: Any | None = None

    def fit(self, train_loader, val_loader, classnames, model_bundle) -> PromptSRCArtifact:
        """Train PromptSRC prompts using only train_loader and val_loader."""

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
            "PromptSRCMethod could not infer the dataset name from the train loader. "
            "Pass dataset-specific templates with PromptSRCMethod(templates=[...])."
        )
