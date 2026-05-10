"""PromptSRC-NC standalone implementation.

Heavy dependencies such as torch, torchvision, OpenCLIP, Modal, and pandas are
imported inside runtime modules so that a plain package import remains cheap.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"

