"""Common method interface used by the method workspaces.

The repo does not force every method to subclass a base class. The Protocols
below define the shape expected by future experiment orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class MethodArtifact:
    """Serializable pointer to trained method state."""

    method_name: str
    artifact_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class FewShotMethod(Protocol):
    """Minimal contract for methods connected to the common pipeline.

    Implementations may use plain classes; subclassing is not required. The
    experiment orchestration layer will call these methods with shared loaders,
    class names, and an OpenCLIPBundle.
    """

    method_name: str

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
        classnames: list[str],
        model_bundle: Any,
    ) -> MethodArtifact:
        """Fit a method using only the approved training and validation data."""

    def evaluate(
        self,
        artifact: MethodArtifact,
        test_loader: Any,
        classnames: list[str],
        model_bundle: Any,
    ) -> Mapping[str, float]:
        """Evaluate a fitted method on the approved test split.

        Return metric keys such as test/top1_accuracy and test/macro_accuracy.
        """
