"""Zero-shot CLIP baseline package."""

from .config import ZeroShotCLIPConfig
from .method import ZeroShotCLIPArtifact, ZeroShotCLIPMethod

__all__ = ["ZeroShotCLIPArtifact", "ZeroShotCLIPConfig", "ZeroShotCLIPMethod"]
