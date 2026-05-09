"""Training utilities for the DPC OpenCLIP port."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from Promptsrc.promptsrc.model import build_teacher_text_features
from Promptsrc.promptsrc.trainer import batch_images_labels

from .config import DPCConfig
from .model import DPCDualPromptModel


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("DPC requires torch. Install the repo dependencies with `pip install -e .`.") from exc
    return torch, F


@dataclass
class DPCArtifact:
    method_name: str
    state_dict: dict[str, Any]
    config: DPCConfig
    classnames: list[str]
    history: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DPCTrainer:
    """Two-stage trainer for DPC over a PromptSRC-style text backbone."""

    def __init__(self, config: DPCConfig):
        self.config = config

    def build_model(self, bundle: Any, classnames: list[str], templates: list[str]) -> Any:
        fixed_text_features = build_teacher_text_features(bundle, classnames, templates)
        return DPCDualPromptModel(
            bundle=bundle,
            classnames=classnames,
            fixed_text_features=fixed_text_features,
            n_ctx_text=self.config.n_ctx_text,
            ctx_init=self.config.ctx_init,
            stack_weight=self.config.stack_weight,
        )

    def train(self, model: Any, train_loader: Any, val_loader: Any | None = None) -> DPCArtifact:
        torch, F = _require_torch()
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

        history: list[dict[str, float]] = []
        if self.config.backbone_epochs > 0:
            model.unfreeze_backbone_prompt()
            history.extend(
                self._run_stage(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    stage="backbone",
                    epochs=self.config.backbone_epochs,
                    loss_fn=lambda outputs, labels: self._backbone_loss(outputs, labels, F),
                )
            )
            model.clone_backbone_to_dpc()

        model.freeze_backbone_prompt()
        history.extend(
            self._run_stage(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                stage="dpc",
                epochs=self.config.dpc_epochs,
                loss_fn=lambda outputs, labels: self._dpc_loss(outputs, labels, F),
            )
        )

        return DPCArtifact(
            method_name="DPC",
            state_dict=model.trainable_state_dict(),
            config=self.config,
            classnames=list(model.backbone_prompt.classnames),
            history=history,
            metadata={
                "backbone_method": self.config.backbone_method,
                "backbone_checkpoint": self.config.backbone_checkpoint,
                "annotation_path": self.config.annotation_path,
                "stack_weight": self.config.stack_weight,
                "official_source": "https://github.com/JREion/DPC",
                "port_note": (
                    "OpenCLIP text-prompt DPC port: PromptSRC-style backbone prompt, cloned parallel prompt, "
                    "hard-negative DPC loss, and weighted dual-prompt inference."
                ),
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
        eval_iter = self._progress(
            data_loader,
            desc=f"{prefix} eval",
            total=self._limited_total(data_loader, self.config.max_eval_batches),
            leave=False,
        )
        with torch.no_grad():
            for batch_index, batch in enumerate(eval_iter, start=1):
                images, labels = batch_images_labels(batch, model.device_name)
                with torch.autocast(device_type=self._device_type(model), enabled=self._use_amp(model)):
                    logits = model(images)["logits"]
                predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
                targets.extend(labels.detach().cpu().tolist())
                if self.config.max_eval_batches is not None and batch_index >= self.config.max_eval_batches:
                    break
        if not targets:
            raise ValueError("Cannot evaluate DPC on an empty loader.")
        return {
            f"{prefix}/top1_accuracy": accuracy(predictions, targets),
            f"{prefix}/macro_accuracy": macro_accuracy(predictions, targets),
        }

    def _run_stage(self, model: Any, train_loader: Any, val_loader: Any | None, stage: str, epochs: int, loss_fn: Any):
        torch, _ = _require_torch()
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.SGD(trainable_parameters, lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = self._build_scheduler(optimizer, epochs)
        use_amp = self._use_amp(model)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rows: list[dict[str, float]] = []

        epoch_iter = self._progress(range(1, epochs + 1), desc=f"DPC {stage} epochs", total=epochs)
        for epoch in epoch_iter:
            model.train()
            epoch_loss = 0.0
            epoch_examples = 0
            epoch_parts: dict[str, float] = {}
            train_iter = self._progress(
                train_loader,
                desc=f"{stage} epoch {epoch} train",
                total=self._limited_total(train_loader, self.config.max_train_batches),
                leave=False,
            )

            for batch_index, batch in enumerate(train_iter, start=1):
                images, labels = batch_images_labels(batch, model.device_name)
                with torch.autocast(device_type=self._device_type(model), enabled=use_amp):
                    outputs = model(images)
                    loss, loss_parts = loss_fn(outputs, labels)

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
                    epoch_parts[f"{stage}/{key}"] = epoch_parts.get(f"{stage}/{key}", 0.0) + value * batch_size
                if self.config.max_train_batches is not None and batch_index >= self.config.max_train_batches:
                    break

            scheduler.step()
            train_loss = epoch_loss / max(epoch_examples, 1)
            averaged_parts = {key: value / max(epoch_examples, 1) for key, value in epoch_parts.items()}
            val_metrics = self.evaluate(model, val_loader, prefix=f"val/{stage}") if val_loader is not None else {}
            row = {
                "epoch": float(epoch),
                f"{stage}/loss": train_loss,
                **{key: float(value) for key, value in averaged_parts.items()},
                **val_metrics,
            }
            rows.append(row)
            if hasattr(epoch_iter, "set_postfix"):
                postfix = {"loss": f"{train_loss:.4f}"}
                metric_name = f"val/{stage}/top1_accuracy"
                if metric_name in val_metrics:
                    postfix["val_top1"] = f"{val_metrics[metric_name]:.4f}"
                epoch_iter.set_postfix(postfix)
        return rows

    def _backbone_loss(self, outputs: dict[str, Any], labels: Any, F: Any):
        loss_ce = F.cross_entropy(outputs["backbone_logits"], labels)
        loss_retain = F.kl_div(
            F.log_softmax(outputs["backbone_logits"], dim=1),
            F.log_softmax(outputs["fixed_logits"], dim=1),
            reduction="batchmean",
            log_target=True,
        )
        total = loss_ce + self.config.retain_weight * loss_retain
        return total, {
            "loss_ce": float(loss_ce.detach().cpu()),
            "loss_retain": float(loss_retain.detach().cpu()),
        }

    def _dpc_loss(self, outputs: dict[str, Any], labels: Any, F: Any):
        loss_ce = F.cross_entropy(outputs["logits"], labels)
        loss_hard = self._hard_negative_loss(outputs, labels, F)
        loss_retain = F.kl_div(
            F.log_softmax(outputs["logits"], dim=1),
            F.log_softmax(outputs["backbone_logits"].detach(), dim=1),
            reduction="batchmean",
            log_target=True,
        )
        loss_proximity = F.l1_loss(outputs["dpc_text_features"], outputs["backbone_text_features"].detach())
        total = (
            loss_ce
            + self.config.hard_negative_weight * loss_hard
            + self.config.retain_weight * loss_retain
            + self.config.prompt_proximity_weight * loss_proximity
        )
        return total, {
            "loss_ce": float(loss_ce.detach().cpu()),
            "loss_hard_negative": float(loss_hard.detach().cpu()),
            "loss_retain": float(loss_retain.detach().cpu()),
            "loss_prompt_proximity": float(loss_proximity.detach().cpu()),
        }

    def _hard_negative_loss(self, outputs: dict[str, Any], labels: Any, F: Any):
        torch, _ = _require_torch()
        query_logits = outputs["backbone_logits"].detach()
        dpc_logits = outputs["dpc_logits"]
        num_classes = dpc_logits.shape[1]
        topk = min(self.config.hard_negative_topk, max(num_classes - 1, 1))
        masked = query_logits.clone()
        masked[torch.arange(labels.shape[0], device=labels.device), labels] = float("-inf")
        hard_indices = masked.topk(topk, dim=1).indices
        positive_logits = dpc_logits.gather(1, labels.view(-1, 1))
        negative_logits = dpc_logits.gather(1, hard_indices)
        hard_logits = torch.cat([positive_logits, negative_logits], dim=1)
        hard_targets = torch.zeros(labels.shape[0], dtype=torch.long, device=labels.device)
        return F.cross_entropy(hard_logits, hard_targets)

    def _build_scheduler(self, optimizer: Any, epochs: int):
        torch, _ = _require_torch()

        def lr_lambda(epoch_index: int) -> float:
            import math

            if self.config.warmup_epochs > 0 and epoch_index < self.config.warmup_epochs:
                return max((epoch_index + 1) / self.config.warmup_epochs, 1e-5)
            progress = (epoch_index - self.config.warmup_epochs + 1) / max(epochs - self.config.warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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
