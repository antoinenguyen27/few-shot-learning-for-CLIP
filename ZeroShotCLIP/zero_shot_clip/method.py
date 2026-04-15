"""Frozen zero-shot OpenCLIP baseline for the shared few-shot protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from common.datasets.templates import get_templates
from common.evaluation.metrics import accuracy, macro_accuracy
from common.models.openclip import (
    build_zero_shot_classifier,
    clip_classification_logits,
    encode_image_features,
)

from .config import ZeroShotCLIPConfig


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Zero-shot CLIP requires torch. Install dependencies with `pip install -e .`.") from exc
    return torch


def batch_images_labels(batch: Any, device: str):
    """Extract images and labels from repo dict batches or tuple batches."""

    torch = _require_torch()
    if isinstance(batch, dict):
        image = batch.get("image")
        if image is None:
            image = batch.get("img")
        label = batch.get("label")
    else:
        image, label = batch
    if image is None or label is None:
        raise KeyError("Batch must contain image/img and label fields.")
    if not torch.is_tensor(label):
        label = torch.as_tensor(label)
    return image.to(device), label.to(device).long()


@dataclass
class ZeroShotCLIPArtifact:
    """In-memory zero-shot classifier plus metadata for evaluation."""

    method_name: str
    text_classifier: Any
    classnames: list[str]
    templates: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class ZeroShotCLIPMethod:
    """Zero-shot CLIP adapter for the repo's method contract."""

    method_name = "Zero-shot CLIP"

    def __init__(
        self,
        config: ZeroShotCLIPConfig | None = None,
        templates: Iterable[str] | None = None,
    ) -> None:
        self.config = config or ZeroShotCLIPConfig()
        self.templates = list(templates) if templates is not None else None
        if self.templates is not None and not self.templates:
            raise ValueError("templates must not be empty when provided")

    def fit(self, train_loader: Any, val_loader: Any, classnames: list[str], model_bundle: Any) -> ZeroShotCLIPArtifact:
        """Build the frozen text classifier without fitting on train images."""

        template_list = self._resolve_templates(train_loader, val_loader)
        text_classifier = build_zero_shot_classifier(model_bundle, list(classnames), template_list)
        return ZeroShotCLIPArtifact(
            method_name=self.method_name,
            text_classifier=text_classifier,
            classnames=list(classnames),
            templates=template_list,
            metadata={
                "templates": template_list,
                "model_name": model_bundle.model_name,
                "pretrained": model_bundle.pretrained,
                "uses_train_images": False,
                "port_note": "Frozen OpenCLIP zero-shot classifier averaged over dataset-specific templates.",
            },
        )

    def evaluate(
        self,
        artifact: ZeroShotCLIPArtifact,
        data_loader: Any,
        classnames: list[str],
        model_bundle: Any,
        metric_prefix: str = "test",
    ) -> dict[str, float]:
        """Evaluate the frozen classifier and return standardized metrics."""

        if data_loader is None:
            return {}
        if list(classnames) != artifact.classnames:
            raise ValueError("Evaluation classnames must match the zero-shot classifier classnames.")

        torch = _require_torch()
        model_bundle.model.eval()
        text_classifier = artifact.text_classifier.to(model_bundle.device)
        predictions: list[int] = []
        targets: list[int] = []

        eval_iter = self._progress(
            data_loader,
            desc=f"{metric_prefix} zero-shot eval",
            total=self._limited_total(data_loader, self.config.max_eval_batches),
            leave=False,
        )
        with torch.no_grad():
            for batch_index, batch in enumerate(eval_iter, start=1):
                images, labels = batch_images_labels(batch, model_bundle.device)
                with torch.autocast(device_type=self._device_type(model_bundle), enabled=self._use_amp(model_bundle)):
                    image_features = encode_image_features(model_bundle, images, normalize=True)
                    logits = clip_classification_logits(model_bundle, image_features, text_classifier)
                predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
                targets.extend(labels.detach().cpu().tolist())
                if self.config.max_eval_batches is not None and batch_index >= self.config.max_eval_batches:
                    break

        if not targets:
            raise ValueError("Cannot evaluate zero-shot CLIP on an empty loader.")
        return {
            f"{metric_prefix}/top1_accuracy": accuracy(predictions, targets),
            f"{metric_prefix}/macro_accuracy": macro_accuracy(predictions, targets),
        }

    def _resolve_templates(self, *loaders: Any) -> list[str]:
        if self.templates is not None:
            return list(self.templates)
        return list(get_templates(self._infer_dataset_name(*loaders)))

    def _infer_dataset_name(self, *loaders: Any) -> str:
        for loader in loaders:
            dataset = getattr(loader, "dataset", None)
            records = getattr(dataset, "records", None)
            if records:
                return str(records[0].dataset)
        raise ValueError(
            "ZeroShotCLIPMethod could not infer the dataset name from the loaders. "
            "Pass templates with ZeroShotCLIPMethod(templates=[...])."
        )

    def _device_type(self, model_bundle: Any) -> str:
        device_name = str(getattr(model_bundle, "device", "cpu"))
        return "cuda" if device_name.startswith("cuda") else "cpu"

    def _use_amp(self, model_bundle: Any) -> bool:
        torch = _require_torch()
        return self.config.precision == "amp" and self._device_type(model_bundle) == "cuda" and torch.cuda.is_available()

    def _limited_total(self, data_loader: Any, limit: int | None) -> int | None:
        try:
            total = len(data_loader)
        except TypeError:
            return limit
        if limit is None:
            return total
        return min(total, limit)

    def _progress(self, iterable: Any, **kwargs: Any) -> Any:
        if not self.config.show_progress:
            return iterable
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return iterable
        return tqdm(iterable, **kwargs)
