"""Training utilities for the PromptSRC OpenCLIP port."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import PromptSRCConfig, gaussian_epoch_weights
from .model import PromptSRCModel, build_teacher_text_features


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("PromptSRC requires torch. Install the repo dependencies with `pip install -e .`.") from exc
    return torch, F


@dataclass
class PromptSRCArtifact:
    method_name: str
    state_dict: dict[str, Any]
    config: PromptSRCConfig
    classnames: list[str]
    history: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def batch_images_labels(batch: Any, device: str):
    """Extract images and labels from this repo's dict batches or Dassl-style batches."""

    torch, _ = _require_torch()
    if isinstance(batch, dict):
        image = batch.get("image", batch.get("img"))
        label = batch.get("label")
    else:
        image, label = batch
    if image is None or label is None:
        raise KeyError("Batch must contain image/img and label fields.")
    if not torch.is_tensor(label):
        label = torch.as_tensor(label)
    return image.to(device), label.to(device).long()


class PromptSRCTrainer:
    """Small trainer that keeps PromptSRC independent from the shared pipeline."""

    def __init__(self, config: PromptSRCConfig):
        self.config = config

    def build_model(self, bundle: Any, classnames: list[str], templates: list[str]) -> Any:
        fixed_text_features = build_teacher_text_features(bundle, classnames, templates)
        return PromptSRCModel(
            bundle=bundle,
            classnames=classnames,
            fixed_text_features=fixed_text_features,
            n_ctx_text=self.config.n_ctx_text,
            ctx_init=self.config.ctx_init,
        )

    def train(self, model: Any, train_loader: Any, val_loader: Any | None = None) -> PromptSRCArtifact:
        torch, F = _require_torch()
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.SGD(trainable_parameters, lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = self._build_scheduler(optimizer)
        gpa_weights = gaussian_epoch_weights(self.config.epochs, self.config.gpa_mean, self.config.gpa_std)
        use_amp = self._use_amp(model)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        history: list[dict[str, float]] = []
        best_state = model.trainable_state_dict()
        best_val = -1.0
        gpa_state = None

        epoch_iter = self._progress(
            range(1, self.config.epochs + 1),
            desc="PromptSRC epochs",
            total=self.config.epochs,
        )
        for epoch in epoch_iter:
            model.train()
            epoch_loss = 0.0
            epoch_examples = 0
            epoch_parts: dict[str, float] = {}

            train_total = self._limited_total(train_loader, self.config.max_train_batches)
            train_iter = self._progress(
                train_loader,
                desc=f"epoch {epoch} train",
                total=train_total,
                leave=False,
            )
            for batch_index, batch in enumerate(train_iter, start=1):
                images, labels = batch_images_labels(batch, model.device_name)
                with torch.autocast(device_type=self._device_type(model), enabled=use_amp):
                    outputs = model(images)
                    loss, loss_parts = self._loss(outputs, labels, F)

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                batch_size = int(labels.shape[0])
                epoch_loss += float(loss.detach().cpu()) * batch_size
                epoch_examples += batch_size
                for key, value in loss_parts.items():
                    epoch_parts[key] = epoch_parts.get(key, 0.0) + value * batch_size
                if self.config.max_train_batches is not None and batch_index >= self.config.max_train_batches:
                    break

            scheduler.step()
            train_loss = epoch_loss / max(epoch_examples, 1)
            averaged_parts = {key: value / max(epoch_examples, 1) for key, value in epoch_parts.items()}
            val_metrics = self.evaluate(model, val_loader, prefix="val") if val_loader is not None else {}
            val_top1 = val_metrics.get("val/top1_accuracy", -train_loss)

            if val_top1 >= best_val:
                best_val = val_top1
                best_state = model.trainable_state_dict()

            current_state = model.trainable_state_dict()
            weighted_state = self._weight_state(current_state, gpa_weights[epoch - 1])
            gpa_state = weighted_state if gpa_state is None else self._add_state(gpa_state, weighted_state)

            history_row = {
                "epoch": float(epoch),
                "train/loss": train_loss,
                **{key: float(value) for key, value in averaged_parts.items()},
                **val_metrics,
            }
            history.append(history_row)
            if hasattr(epoch_iter, "set_postfix"):
                postfix = {"loss": f"{train_loss:.4f}"}
                if "val/top1_accuracy" in val_metrics:
                    postfix["val_top1"] = f"{val_metrics['val/top1_accuracy']:.4f}"
                epoch_iter.set_postfix(postfix)

        final_state = gpa_state if self.config.use_gpa and gpa_state is not None else best_state
        model.load_trainable_state_dict(final_state)

        return PromptSRCArtifact(
            method_name="PromptSRC",
            state_dict=final_state,
            config=self.config,
            classnames=list(model.prompt_learner.classnames),
            history=history,
            metadata={
                "best_val_top1_accuracy": best_val if best_val >= 0 else None,
                "gpa_enabled": self.config.use_gpa,
                "vision_prompting": False,
                "port_note": "OpenCLIP port with learned text prompts, PromptSRC regularizers, and prompt-only GPA.",
            },
        )

    def evaluate(self, model: Any, data_loader: Any, prefix: str = "test") -> dict[str, float]:
        if data_loader is None:
            return {}
        torch, _ = _require_torch()
        from common.evaluation.metrics import accuracy, macro_accuracy

        model.eval()
        predictions: list[int] = []
        targets: list[int] = []
        with torch.no_grad():
            eval_total = self._limited_total(data_loader, self.config.max_eval_batches)
            eval_iter = self._progress(
                data_loader,
                desc=f"{prefix} eval",
                total=eval_total,
                leave=False,
            )
            for batch_index, batch in enumerate(eval_iter, start=1):
                images, labels = batch_images_labels(batch, model.device_name)
                with torch.autocast(device_type=self._device_type(model), enabled=self._use_amp(model)):
                    logits = model(images)["logits"]
                predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
                targets.extend(labels.detach().cpu().tolist())
                if self.config.max_eval_batches is not None and batch_index >= self.config.max_eval_batches:
                    break
        if not targets:
            raise ValueError("Cannot evaluate PromptSRC on an empty loader.")
        return {
            f"{prefix}/top1_accuracy": accuracy(predictions, targets),
            f"{prefix}/macro_accuracy": macro_accuracy(predictions, targets),
        }

    def _loss(self, outputs: dict[str, Any], labels: Any, F: Any) -> tuple[Any, dict[str, float]]:
        loss_ce = F.cross_entropy(outputs["logits"], labels)
        loss_text = F.l1_loss(outputs["text_features"], outputs["fixed_text_features"], reduction="mean")
        loss_image = F.l1_loss(
            outputs["image_features"],
            outputs["zero_shot_image_features"],
            reduction="mean",
        )
        loss_logits = F.kl_div(
            F.log_softmax(outputs["logits"], dim=1),
            F.log_softmax(outputs["zero_shot_logits"], dim=1),
            reduction="batchmean",
            log_target=True,
        )
        total = (
            loss_ce
            + self.config.text_loss_weight * loss_text
            + self.config.image_loss_weight * loss_image
            + self.config.logit_loss_weight * loss_logits
        )
        return total, {
            "train/loss_ce": float(loss_ce.detach().cpu()),
            "train/loss_text": float(loss_text.detach().cpu()),
            "train/loss_image": float(loss_image.detach().cpu()),
            "train/loss_logits": float(loss_logits.detach().cpu()),
        }

    def _build_scheduler(self, optimizer: Any):
        torch, _ = _require_torch()

        def lr_lambda(epoch_index: int) -> float:
            import math

            if self.config.warmup_epochs > 0 and epoch_index < self.config.warmup_epochs:
                return max((epoch_index + 1) / self.config.warmup_epochs, 1e-5)
            progress = (epoch_index - self.config.warmup_epochs + 1) / max(
                self.config.epochs - self.config.warmup_epochs,
                1,
            )
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _weight_state(self, state: dict[str, Any], weight: float) -> dict[str, Any]:
        return {key: value.detach().clone() * weight for key, value in state.items()}

    def _add_state(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        return {key: left[key] + right[key] for key in left}

    def _device_type(self, model: Any) -> str:
        device_name = str(getattr(model, "device_name", "cpu"))
        return "cuda" if device_name.startswith("cuda") else "cpu"

    def _use_amp(self, model: Any) -> bool:
        torch, _ = _require_torch()
        return self.config.precision == "amp" and self._device_type(model) == "cuda" and torch.cuda.is_available()

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
