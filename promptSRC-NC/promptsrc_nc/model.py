"""PromptSRC-style model over frozen OpenCLIP towers.

The official PromptSRC repository modifies OpenAI CLIP internals to add
independent deep text and visual prompt tokens. This module ports that behavior
to OpenCLIP wrappers without depending on the external PromptSRC repository at
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .config import canonical_backbone
from .imagenet_templates import IMAGENET_TEMPLATES


def _normalize(features: Any):
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _openclip_pretrained_arg(pretrained: str) -> str | None:
    tag = str(pretrained).strip()
    if tag.lower() in {"", "none", "null", "random"}:
        return None
    return tag


@dataclass(frozen=True)
class OpenCLIPBundle:
    model: Any
    tokenizer: Any
    train_preprocess: Any
    eval_preprocess: Any
    model_name: str
    pretrained: str
    device: str
    precision: str


def build_openclip_bundle(
    backbone: str,
    pretrained: str = "openai",
    device: str = "cuda",
    precision: str = "amp",
) -> OpenCLIPBundle:
    import open_clip
    import torch

    model_name = canonical_backbone(backbone)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    # OpenCLIP's fp16 construction is useful on GPU; CPU falls back to fp32.
    openclip_precision = "fp16" if precision == "fp16" and device.startswith("cuda") else "fp32"
    model, train_preprocess, eval_preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=_openclip_pretrained_arg(pretrained),
        device=device,
        precision=openclip_precision,
    )
    validate_openclip_compatibility(model)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return OpenCLIPBundle(
        model=model,
        tokenizer=tokenizer,
        train_preprocess=train_preprocess,
        eval_preprocess=eval_preprocess,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        precision=precision,
    )


def validate_openclip_compatibility(model: Any) -> None:
    """Fail early if OpenCLIP internals no longer match the prompt wrappers."""

    required_model_attrs = ("token_embedding", "transformer", "positional_embedding", "ln_final", "visual")
    missing = [name for name in required_model_attrs if not hasattr(model, name)]
    if missing:
        raise RuntimeError(f"OpenCLIP model is missing required attributes for PromptSRC-NC: {missing}")
    for name in ("encode_image", "encode_text"):
        if not callable(getattr(model, name, None)):
            raise RuntimeError(f"OpenCLIP model must expose callable {name}()")

    text_transformer = model.transformer
    if getattr(text_transformer, "batch_first", None) is not True:
        raise RuntimeError("PromptSRC-NC OpenCLIP text wrapper requires a batch_first text transformer")
    if not hasattr(text_transformer, "resblocks"):
        raise RuntimeError("OpenCLIP text transformer must expose resblocks")

    visual = model.visual
    required_visual_attrs = ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post")
    missing_visual = [name for name in required_visual_attrs if not hasattr(visual, name)]
    if missing_visual:
        raise RuntimeError(f"OpenCLIP visual tower is missing required ViT attributes: {missing_visual}")
    visual_transformer = visual.transformer
    if getattr(visual_transformer, "batch_first", None) is not True:
        raise RuntimeError("PromptSRC-NC OpenCLIP visual wrapper requires a batch_first visual transformer")
    if not hasattr(visual_transformer, "resblocks"):
        raise RuntimeError("OpenCLIP visual transformer must expose resblocks")


class TextPromptLearner:
    def __new__(cls, *args: Any, **kwargs: Any):
        import torch
        from torch import nn

        class _TextPromptLearner(nn.Module):
            def __init__(
                self,
                classnames: list[str],
                clip_model: Any,
                tokenizer: Any,
                n_ctx: int,
                ctx_init: str,
                device: str,
            ) -> None:
                super().__init__()
                if not hasattr(clip_model, "token_embedding"):
                    raise AttributeError("OpenCLIP model must expose token_embedding for prompt learning")
                self.classnames = [name.replace("_", " ") for name in classnames]
                self.n_cls = len(self.classnames)
                self.n_ctx = int(n_ctx)
                self.ctx_init = ctx_init

                token_embedding = clip_model.token_embedding
                dtype = token_embedding.weight.dtype
                ctx_dim = token_embedding.weight.shape[1]

                prompt_prefix = " ".join(["X"] * self.n_ctx)
                ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=dtype, device=device)
                nn.init.normal_(ctx_vectors, std=0.02)

                ctx_words = ctx_init.replace("_", " ").split() if ctx_init else []
                if ctx_words and len(ctx_words) <= self.n_ctx:
                    prompt_prefix = ctx_init.replace("_", " ")
                    init_tokens = tokenizer([prompt_prefix]).to(device)
                    with torch.no_grad():
                        init_embedding = token_embedding(init_tokens).to(dtype=dtype)
                    ctx_vectors[: len(ctx_words)].copy_(init_embedding[0, 1 : 1 + len(ctx_words), :])

                prompts = [f"{prompt_prefix} {name}." for name in self.classnames]
                tokenized_prompts = tokenizer(prompts).to(device)
                with torch.no_grad():
                    embedding = token_embedding(tokenized_prompts).to(dtype=dtype)

                self.ctx = nn.Parameter(ctx_vectors)
                self.register_buffer("token_prefix", embedding[:, :1, :])
                self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])
                self.register_buffer("tokenized_prompts", tokenized_prompts)
                self.prompt_prefix = prompt_prefix

            def forward(self):
                ctx = self.ctx
                if ctx.dim() == 2:
                    ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
                return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

        return _TextPromptLearner(*args, **kwargs)


class PromptedTextEncoder:
    def __new__(cls, *args: Any, **kwargs: Any):
        import torch
        from torch import nn

        class _PromptedTextEncoder(nn.Module):
            def __init__(self, clip_model: Any, n_ctx_text: int, prompt_depth_text: int) -> None:
                super().__init__()
                self.transformer = clip_model.transformer
                self.positional_embedding = clip_model.positional_embedding
                self.ln_final = clip_model.ln_final
                self.text_projection = getattr(clip_model, "text_projection", None)
                self.attn_mask = getattr(clip_model, "attn_mask", None)
                self.pool_type = getattr(clip_model, "text_pool_type", "argmax")
                self.eos_id = getattr(clip_model, "text_eos_id", None)
                self.n_ctx_text = int(n_ctx_text)
                self.prompt_depth_text = int(prompt_depth_text)
                self.deep_prompts = nn.ParameterList()
                width = self.positional_embedding.shape[-1]
                layers = len(self.transformer.resblocks)
                for _ in range(max(0, min(self.prompt_depth_text, layers) - 1)):
                    prompt = nn.Parameter(torch.empty(self.n_ctx_text, width))
                    nn.init.normal_(prompt, std=0.02)
                    self.deep_prompts.append(prompt)

            def _pool(self, x, tokenized_prompts):
                if self.pool_type == "first":
                    return x[:, 0]
                if self.pool_type == "last":
                    return x[:, -1]
                if self.pool_type == "eos" and self.eos_id is not None:
                    idx = (tokenized_prompts == self.eos_id).int().argmax(dim=-1)
                    return x[torch.arange(x.shape[0], device=x.device), idx]
                return x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)]

            def forward(self, prompts, tokenized_prompts):
                seq_len = prompts.shape[1]
                x = prompts + self.positional_embedding[:seq_len].to(device=prompts.device, dtype=prompts.dtype)
                attn_mask = self.attn_mask
                if attn_mask is not None:
                    attn_mask = attn_mask[:seq_len, :seq_len].to(device=x.device, dtype=x.dtype)
                for layer_index, block in enumerate(self.transformer.resblocks):
                    if 0 < layer_index <= len(self.deep_prompts):
                        prefix = x[:, :1, :]
                        suffix = x[:, 1 + self.n_ctx_text :, :]
                        ctx = self.deep_prompts[layer_index - 1].to(device=x.device, dtype=x.dtype)
                        ctx = ctx.unsqueeze(0).expand(x.shape[0], -1, -1)
                        x = torch.cat([prefix, ctx, suffix], dim=1)
                    x = block(x, attn_mask=attn_mask)
                x = self.ln_final(x).to(dtype=prompts.dtype)
                pooled = self._pool(x, tokenized_prompts)
                if self.text_projection is not None:
                    if isinstance(self.text_projection, nn.Linear):
                        pooled = self.text_projection(pooled)
                    else:
                        pooled = pooled @ self.text_projection.to(device=pooled.device, dtype=pooled.dtype)
                return pooled

        return _PromptedTextEncoder(*args, **kwargs)


class PromptedVisionEncoder:
    def __new__(cls, *args: Any, **kwargs: Any):
        import torch
        from torch import nn

        class _PromptedVisionEncoder(nn.Module):
            def __init__(self, visual: Any, n_ctx_vision: int, prompt_depth_vision: int) -> None:
                super().__init__()
                required = ("conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post")
                missing = [name for name in required if not hasattr(visual, name)]
                if missing:
                    raise TypeError(
                        "PromptSRC-NC visual prompting currently supports OpenCLIP VisionTransformer only; "
                        f"missing attributes: {missing}"
                    )
                self.visual = visual
                self.n_ctx_vision = int(n_ctx_vision)
                self.prompt_depth_vision = int(prompt_depth_vision)
                width = visual.conv1.out_channels
                layers = len(visual.transformer.resblocks)
                self.shallow_prompt = None
                if self.prompt_depth_vision > 0:
                    self.shallow_prompt = nn.Parameter(torch.empty(self.n_ctx_vision, width))
                    nn.init.normal_(self.shallow_prompt, std=0.02)
                self.deep_prompts = nn.ParameterList()
                for _ in range(max(0, min(self.prompt_depth_vision, layers) - 1)):
                    prompt = nn.Parameter(torch.empty(self.n_ctx_vision, width))
                    nn.init.normal_(prompt, std=0.02)
                    self.deep_prompts.append(prompt)

            def _embeds_with_prompts(self, image):
                visual = self.visual
                image = image.to(dtype=visual.conv1.weight.dtype)
                x = visual.conv1(image)
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)
                class_embedding = visual.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1)
                x = torch.cat([class_embedding.to(x.dtype), x], dim=1)
                x = x + visual.positional_embedding.to(device=x.device, dtype=x.dtype)
                if hasattr(visual, "patch_dropout"):
                    x = visual.patch_dropout(x)
                if self.shallow_prompt is not None:
                    prompt = self.shallow_prompt.to(device=x.device, dtype=x.dtype)
                    x = torch.cat([x, prompt.unsqueeze(0).expand(x.shape[0], -1, -1)], dim=1)
                x = visual.ln_pre(x)
                return x

            def _pool(self, x):
                visual = self.visual
                if self.shallow_prompt is not None:
                    x = x[:, : -self.n_ctx_vision, :]
                if hasattr(visual, "_pool"):
                    pooled, _tokens = visual._pool(x)
                else:
                    x = visual.ln_post(x)
                    pooled = x[:, 0]
                if getattr(visual, "proj", None) is not None:
                    pooled = pooled @ visual.proj.to(device=pooled.device, dtype=pooled.dtype)
                return pooled

            def forward(self, image):
                x = self._embeds_with_prompts(image)
                for layer_index, block in enumerate(self.visual.transformer.resblocks):
                    if 0 < layer_index <= len(self.deep_prompts):
                        prefix = x[:, : -self.n_ctx_vision, :]
                        prompt = self.deep_prompts[layer_index - 1].to(device=x.device, dtype=x.dtype)
                        x = torch.cat([prefix, prompt.unsqueeze(0).expand(x.shape[0], -1, -1)], dim=1)
                    x = block(x)
                return self._pool(x)

        return _PromptedVisionEncoder(*args, **kwargs)


class PromptSRCModel:
    def __new__(cls, *args: Any, **kwargs: Any):
        import torch
        from torch import nn

        class _PromptSRCModel(nn.Module):
            def __init__(self, bundle: OpenCLIPBundle, classnames: list[str], config: Any) -> None:
                super().__init__()
                self.bundle = bundle
                self.clip_model = bundle.model
                self.device_name = bundle.device
                self.config = config
                for parameter in self.clip_model.parameters():
                    parameter.requires_grad_(False)
                self.prompt_learner = TextPromptLearner(
                    classnames=classnames,
                    clip_model=self.clip_model,
                    tokenizer=bundle.tokenizer,
                    n_ctx=config.n_ctx_text,
                    ctx_init=config.ctx_init,
                    device=bundle.device,
                )
                self.text_encoder = PromptedTextEncoder(
                    self.clip_model,
                    n_ctx_text=config.n_ctx_text,
                    prompt_depth_text=config.prompt_depth_text,
                )
                self.image_encoder = PromptedVisionEncoder(
                    self.clip_model.visual,
                    n_ctx_vision=config.n_ctx_vision,
                    prompt_depth_vision=config.prompt_depth_vision,
                )
                fixed_text = build_teacher_text_features(bundle, classnames, IMAGENET_TEMPLATES)
                self.register_buffer("fixed_text_features", fixed_text)
                self.to(bundle.device)

            def train(self, mode: bool = True):
                super().train(mode)
                self.clip_model.eval()
                return self

            def prompt_state_dict(self) -> dict[str, Any]:
                text_deep = {
                    str(index): prompt.detach().clone()
                    for index, prompt in enumerate(self.text_encoder.deep_prompts)
                }
                image_deep = {
                    str(index): prompt.detach().clone()
                    for index, prompt in enumerate(self.image_encoder.deep_prompts)
                }
                image_state: dict[str, Any] = {"deep_prompts": image_deep}
                if self.image_encoder.shallow_prompt is not None:
                    image_state["shallow_prompt"] = self.image_encoder.shallow_prompt.detach().clone()
                return {
                    "prompt_learner": {"ctx": self.prompt_learner.ctx.detach().clone()},
                    "text_encoder": {"deep_prompts": text_deep},
                    "image_encoder": image_state,
                }

            def load_prompt_state_dict(self, state_dict: MappingLike) -> None:
                with torch.no_grad():
                    prompt_state = state_dict.get("prompt_learner", {})
                    if "ctx" in prompt_state:
                        self.prompt_learner.ctx.copy_(prompt_state["ctx"].to(self.prompt_learner.ctx.device))
                    text_state = state_dict.get("text_encoder", {}).get("deep_prompts", {})
                    for index, tensor in text_state.items():
                        prompt = self.text_encoder.deep_prompts[int(index)]
                        prompt.copy_(tensor.to(device=prompt.device, dtype=prompt.dtype))
                    image_state = state_dict.get("image_encoder", {})
                    if "shallow_prompt" in image_state and self.image_encoder.shallow_prompt is not None:
                        self.image_encoder.shallow_prompt.copy_(
                            image_state["shallow_prompt"].to(
                                device=self.image_encoder.shallow_prompt.device,
                                dtype=self.image_encoder.shallow_prompt.dtype,
                            )
                        )
                    for index, tensor in image_state.get("deep_prompts", {}).items():
                        prompt = self.image_encoder.deep_prompts[int(index)]
                        prompt.copy_(tensor.to(device=prompt.device, dtype=prompt.dtype))

            def encode_prompted_text(self):
                prompts = self.prompt_learner()
                text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
                return _normalize(text_features)

            def encode_prompted_image(self, images):
                images = images.to(self.device_name)
                features = self.image_encoder(images)
                return _normalize(features)

            def encode_zero_shot_image(self, images):
                images = images.to(self.device_name)
                with torch.no_grad():
                    try:
                        features = self.clip_model.encode_image(images, normalize=True)
                    except TypeError:
                        features = _normalize(self.clip_model.encode_image(images))
                return features.detach()

            def forward_logits(self, images):
                image_features = self.encode_prompted_image(images)
                text_features = self.encode_prompted_text()
                logit_scale = self.clip_model.logit_scale.exp()
                return logit_scale * image_features @ text_features.t()

            def forward(self, images):
                images = images.to(self.device_name)
                image_features = self.encode_prompted_image(images)
                text_features = self.encode_prompted_text()
                logit_scale = self.clip_model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()
                fixed_text = _normalize(self.fixed_text_features)
                zero_shot_features = self.encode_zero_shot_image(images)
                zero_shot_logits = logit_scale * zero_shot_features @ fixed_text.t()
                return {
                    "logits": logits,
                    "zero_shot_logits": zero_shot_logits.detach(),
                    "image_features": image_features,
                    "zero_shot_image_features": zero_shot_features.detach(),
                    "text_features": text_features,
                    "fixed_text_features": fixed_text.detach(),
                }

        return _PromptSRCModel(*args, **kwargs)


MappingLike = dict[str, Any]


def build_teacher_text_features(bundle: OpenCLIPBundle, classnames: list[str], templates: Iterable[str]):
    import torch

    clean_names = [name.replace("_", " ") for name in classnames]
    features = []
    bundle.model.eval()
    for template in templates:
        texts = [template.format(name) for name in clean_names]
        tokens = bundle.tokenizer(texts).to(bundle.device)
        with torch.no_grad():
            try:
                text_features = bundle.model.encode_text(tokens, normalize=False)
            except TypeError:
                text_features = bundle.model.encode_text(tokens)
        features.append(text_features.float())
    fixed = _normalize(torch.stack(features, dim=0).mean(dim=0))
    return fixed.to(bundle.device)


def trainable_parameters(model: Any):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
