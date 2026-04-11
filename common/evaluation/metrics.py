"""Metrics shared by method implementations.

The primary metric for the first protocol is top-1 accuracy. Prompt-learning
papers such as PromptSRC, PromptKD, and DPC also report base/new class accuracy
and their harmonic mean in base-to-new settings, so those helpers live here too.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not targets:
        raise ValueError("targets must not be empty")
    correct = sum(int(pred == target) for pred, target in zip(predictions, targets))
    return correct / len(targets)


def per_class_accuracy(predictions: Sequence[int], targets: Sequence[int]) -> dict[int, float]:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    totals: dict[int, int] = defaultdict(int)
    correct: dict[int, int] = defaultdict(int)
    for pred, target in zip(predictions, targets):
        totals[int(target)] += 1
        correct[int(target)] += int(pred == target)
    return {label: correct[label] / total for label, total in sorted(totals.items())}


def macro_accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    """Mean per-class accuracy over classes present in targets."""

    return mean(per_class_accuracy(predictions, targets).values())


def harmonic_mean(first: float, second: float) -> float:
    """Harmonic mean used by base-to-new prompt-learning reports."""

    if first < 0 or second < 0:
        raise ValueError("harmonic_mean inputs must be non-negative")
    if first == 0 or second == 0:
        return 0.0
    return 2 * first * second / (first + second)


def grouped_accuracy(
    predictions: Sequence[int],
    targets: Sequence[int],
    groups: Mapping[str, Iterable[int]],
) -> dict[str, float]:
    """Accuracy over named class groups, for example base and new classes."""

    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    output: dict[str, float] = {}
    for group_name, labels in groups.items():
        label_set = {int(label) for label in labels}
        group_predictions = [pred for pred, target in zip(predictions, targets) if int(target) in label_set]
        group_targets = [target for target in targets if int(target) in label_set]
        if group_targets:
            output[group_name] = accuracy(group_predictions, group_targets)
    return output


def base_new_harmonic_mean(base_accuracy: float, new_accuracy: float) -> float:
    """Named wrapper for the base/new harmonic mean reported in prompt papers."""

    return harmonic_mean(base_accuracy, new_accuracy)


def topk_accuracy_from_logits(logits, targets, topk: tuple[int, ...] = (1,)) -> dict[str, float]:
    """Compute top-k accuracy from a torch-style logits tensor.

    This helper imports torch lazily so the metrics module remains importable in
    lightweight test environments. targets may be a tensor or a sequence of ints.
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for topk_accuracy_from_logits.") from exc

    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)
    if not torch.is_tensor(targets):
        targets = torch.as_tensor(list(targets), device=logits.device)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [num_examples, num_classes]")
    if targets.ndim != 1:
        raise ValueError("targets must have shape [num_examples]")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must contain the same number of examples")
    if logits.shape[0] == 0:
        raise ValueError("logits must not be empty")

    max_k = max(topk)
    if max_k < 1 or max_k > logits.shape[1]:
        raise ValueError(f"top-k values must be in [1, {logits.shape[1]}]")
    _, predictions = logits.topk(max_k, dim=1)
    correct = predictions.eq(targets.view(-1, 1))
    return {
        f"top{k}_accuracy": correct[:, :k].any(dim=1).float().mean().item()
        for k in topk
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)
