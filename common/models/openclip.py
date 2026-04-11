"""OpenCLIP model construction used by all methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class OpenCLIPBundle:
    model: Any
    preprocess_train: Any
    preprocess_eval: Any
    tokenizer: Any
    model_name: str
    pretrained: str
    device: str
    precision: str


def build_openclip_bundle(
    model_name: str = "ViT-B-32-256",
    pretrained: str = "datacomp_s34b_b86k",
    device: str = "cpu",
    precision: str = "fp32",
) -> OpenCLIPBundle:
    """Create the shared OpenCLIP model and transforms.

    The model is returned in eval mode. Method implementations that train
    adapters/prompts should explicitly switch only their own trainable modules.
    """

    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "open_clip_torch and torch are required. Install dependencies with `pip install -e .`."
        ) from exc

    model, preprocess_train, preprocess_eval = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision=precision,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return OpenCLIPBundle(
        model=model,
        preprocess_train=preprocess_train,
        preprocess_eval=preprocess_eval,
        tokenizer=tokenizer,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        precision=precision,
    )


def _normalize_features(features):
    return features / features.norm(dim=-1, keepdim=True)


def encode_image_features(bundle: OpenCLIPBundle, images, normalize: bool = True):
    """Encode images with the shared OpenCLIP image tower."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to encode OpenCLIP features. Install with `pip install -e .`.") from exc

    with torch.no_grad():
        features = bundle.model.encode_image(images.to(bundle.device))
        return _normalize_features(features) if normalize else features


def encode_text_features(bundle: OpenCLIPBundle, texts: Iterable[str], normalize: bool = True):
    """Tokenize and encode text strings with the shared OpenCLIP text tower."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to encode OpenCLIP features. Install with `pip install -e .`.") from exc

    tokens = bundle.tokenizer(list(texts)).to(bundle.device)
    with torch.no_grad():
        features = bundle.model.encode_text(tokens)
        return _normalize_features(features) if normalize else features


def build_zero_shot_classifier(
    bundle: OpenCLIPBundle,
    classnames: list[str],
    templates: Iterable[str],
):
    """Build a normalized text-feature matrix for zero-shot classification.

    Returns a tensor with shape [num_classes, embedding_dim]. For inference,
    normalized image features can be multiplied by classifier.T and scaled by
    model.logit_scale.exp().
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to build a zero-shot classifier. Install with `pip install -e .`.") from exc

    templates = list(templates)
    if not templates:
        raise ValueError("templates must not be empty")
    class_features = []
    for classname in classnames:
        texts = [template.format(classname) for template in templates]
        text_features = encode_text_features(bundle, texts, normalize=True)
        text_features = text_features.mean(dim=0)
        class_features.append(_normalize_features(text_features))
    return torch.stack(class_features, dim=0)


def clip_classification_logits(bundle: OpenCLIPBundle, image_features, text_classifier):
    """Compute standard CLIP classification logits from normalized features.

    image_features shape: [batch, embedding_dim]
    text_classifier shape: [num_classes, embedding_dim]
    """

    return bundle.model.logit_scale.exp() * image_features @ text_classifier.t()
