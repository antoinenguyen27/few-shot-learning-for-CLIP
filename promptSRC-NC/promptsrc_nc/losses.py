"""PromptSRC and neighborhood-consistency losses."""

from __future__ import annotations

from typing import Any


def js_divergence_from_logits(logits_a: Any, logits_b: Any, eps: float = 1e-8, reduction: str = "mean"):
    import torch
    import torch.nn.functional as F

    p = F.softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    m = 0.5 * (p + q)
    log_p = torch.log(p.clamp_min(eps))
    log_q = torch.log(q.clamp_min(eps))
    log_m = torch.log(m.clamp_min(eps))
    js = 0.5 * (
        torch.sum(p * (log_p - log_m), dim=-1)
        + torch.sum(q * (log_q - log_m), dim=-1)
    )
    if reduction == "none":
        return js
    if reduction == "sum":
        return js.sum()
    if reduction != "mean":
        raise ValueError("reduction must be none, mean, or sum")
    return js.mean()


def entropy_from_logits(logits: Any, eps: float = 1e-8):
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs.clamp_min(eps)), dim=-1)


def promptsrc_loss(outputs: dict[str, Any], labels: Any, config: Any) -> tuple[Any, dict[str, float]]:
    import torch.nn.functional as F

    loss_ce = F.cross_entropy(outputs["logits"], labels)
    loss_text_raw = F.l1_loss(outputs["text_features"], outputs["fixed_text_features"], reduction="mean")
    loss_image_raw = F.l1_loss(outputs["image_features"], outputs["zero_shot_image_features"], reduction="mean")
    loss_logits_raw = F.kl_div(
        F.log_softmax(outputs["logits"], dim=1),
        F.log_softmax(outputs["zero_shot_logits"], dim=1),
        reduction="sum",
        log_target=True,
    ) / outputs["logits"].numel()
    loss_text = config.text_loss_weight * loss_text_raw
    loss_image = config.image_loss_weight * loss_image_raw
    loss_logits = config.logit_loss_weight * loss_logits_raw
    total = loss_ce + loss_text + loss_image + loss_logits
    return total, {
        "loss_ce": float(loss_ce.detach().cpu()),
        "loss_scl_text": float(loss_text.detach().cpu()),
        "loss_scl_image": float(loss_image.detach().cpu()),
        "loss_scl_logits": float(loss_logits.detach().cpu()),
        "loss_scl_text_raw": float(loss_text_raw.detach().cpu()),
        "loss_scl_image_raw": float(loss_image_raw.detach().cpu()),
        "loss_scl_logits_raw": float(loss_logits_raw.detach().cpu()),
    }


def lambda_nc_for_progress(epoch_progress: float, lambda_max: float, warmup_epochs: float) -> float:
    if warmup_epochs <= 0:
        return float(lambda_max)
    return float(lambda_max) * min(1.0, max(0.0, float(epoch_progress)) / float(warmup_epochs))

