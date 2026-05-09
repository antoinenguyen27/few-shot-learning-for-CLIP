"""PromptSRC model components adapted to the repo's OpenCLIP contract."""

from __future__ import annotations

from typing import Any, Iterable


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("PromptSRC requires torch. Install the repo dependencies with `pip install -e .`.") from exc
    return torch, nn, F


def _normalize(features: Any) -> Any:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _tokenize(tokenizer: Any, texts: Iterable[str], device: str):
    tokens = tokenizer(list(texts))
    return tokens.to(device) if hasattr(tokens, "to") else tokens


def build_teacher_text_features(bundle: Any, classnames: list[str], templates: Iterable[str]):
    """Build frozen zero-shot text features averaged over templates."""

    torch, _, _ = _require_torch()
    from common.models.openclip import encode_text_features

    template_list = list(templates)
    if not template_list:
        template_list = ["a photo of a {}."]

    all_features = []
    clean_names = [name.replace("_", " ") for name in classnames]
    for template in template_list:
        texts = [template.format(name) for name in clean_names]
        all_features.append(encode_text_features(bundle, texts, normalize=True))
    features = torch.stack(all_features, dim=0).mean(dim=0)
    return _normalize(features).detach()


class TextPromptLearner:
    """Learn CoOp-style context tokens for each class prompt."""

    def __new__(cls, *args, **kwargs):
        _, nn, _ = _require_torch()

        class _TextPromptLearner(nn.Module):
            def __init__(self, classnames: list[str], clip_model: Any, tokenizer: Any, n_ctx: int, ctx_init: str, device: str):
                super().__init__()
                torch, nn, _ = _require_torch()
                if not hasattr(clip_model, "token_embedding"):
                    raise AttributeError("OpenCLIP model must expose token_embedding for prompt learning.")

                self.classnames = [name.replace("_", " ") for name in classnames]
                self.n_cls = len(classnames)
                self.n_ctx = n_ctx
                self.device_name = device

                token_embedding = clip_model.token_embedding
                dtype = token_embedding.weight.dtype
                ctx_dim = token_embedding.weight.shape[1]

                prompt_prefix = " ".join(["X"] * n_ctx)
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
                nn.init.normal_(ctx_vectors, std=0.02)

                ctx_words = ctx_init.replace("_", " ").split() if ctx_init else []
                if ctx_words and len(ctx_words) == n_ctx:
                    prompt_prefix = " ".join(ctx_words)
                    with torch.no_grad():
                        init_tokens = _tokenize(tokenizer, [prompt_prefix], device)
                        init_embedding = token_embedding(init_tokens).to(dtype=dtype)
                        ctx_vectors.copy_(init_embedding[0, 1 : 1 + n_ctx, :])

                prompts = [f"{prompt_prefix} {name}." for name in self.classnames]
                tokenized_prompts = _tokenize(tokenizer, prompts, device)
                with torch.no_grad():
                    embedding = token_embedding(tokenized_prompts).to(dtype=dtype)

                self.ctx = nn.Parameter(ctx_vectors)
                self.register_buffer("token_prefix", embedding[:, :1, :])
                self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
                self.register_buffer("tokenized_prompts", tokenized_prompts)
                self.prompt_prefix = prompt_prefix

            def forward(self):
                torch, _, _ = _require_torch()
                ctx = self.ctx
                if ctx.dim() == 2:
                    ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
                return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

        return _TextPromptLearner(*args, **kwargs)


class OpenCLIPPromptTextEncoder:
    """Text encoder that accepts prompt embeddings instead of token IDs."""

    def __new__(cls, *args, **kwargs):
        _, nn, _ = _require_torch()

        class _OpenCLIPPromptTextEncoder(nn.Module):
            def __init__(self, clip_model: Any, batch_first: bool | None = None):
                super().__init__()
                self.transformer = clip_model.transformer
                self.positional_embedding = clip_model.positional_embedding
                self.ln_final = clip_model.ln_final
                self.text_projection = getattr(clip_model, "text_projection", None)
                self.attn_mask = getattr(clip_model, "attn_mask", None)
                self.batch_first = self._infer_batch_first(batch_first)

            def _infer_batch_first(self, batch_first: bool | None) -> bool:
                if batch_first is not None:
                    return batch_first
                transformer_batch_first = getattr(self.transformer, "batch_first", None)
                if transformer_batch_first is not None:
                    return bool(transformer_batch_first)
                resblocks = getattr(self.transformer, "resblocks", None)
                if resblocks:
                    attn = getattr(resblocks[0], "attn", None)
                    attn_batch_first = getattr(attn, "batch_first", None)
                    if attn_batch_first is not None:
                        return bool(attn_batch_first)
                return False

            def _call_transformer(self, x):
                if self.attn_mask is not None:
                    try:
                        return self.transformer(x, attn_mask=self.attn_mask)
                    except TypeError:
                        pass
                return self.transformer(x)

            def forward(self, prompts, tokenized_prompts):
                torch, _, _ = _require_torch()
                dtype = prompts.dtype
                positional = self.positional_embedding[: prompts.shape[1]].to(device=prompts.device, dtype=dtype)
                x = prompts + positional
                if self.batch_first:
                    x = self._call_transformer(x)
                else:
                    x = x.permute(1, 0, 2)
                    x = self._call_transformer(x)
                    x = x.permute(1, 0, 2)
                x = self.ln_final(x).to(dtype=dtype)
                eot_indices = tokenized_prompts.argmax(dim=-1)
                x = x[torch.arange(x.shape[0], device=x.device), eot_indices]
                if self.text_projection is not None:
                    x = x @ self.text_projection.to(device=x.device, dtype=x.dtype)
                return x

        return _OpenCLIPPromptTextEncoder(*args, **kwargs)


class PromptSRCModel:
    """PromptSRC-style text-prompt model over a frozen OpenCLIP image tower."""

    def __new__(cls, *args, **kwargs):
        _, nn, _ = _require_torch()

        class _PromptSRCModel(nn.Module):
            def __init__(self, bundle: Any, classnames: list[str], fixed_text_features: Any, n_ctx_text: int, ctx_init: str):
                super().__init__()
                self.bundle = bundle
                self.clip_model = bundle.model
                self.device_name = bundle.device
                for parameter in self.clip_model.parameters():
                    parameter.requires_grad_(False)

                self.prompt_learner = TextPromptLearner(
                    classnames=classnames,
                    clip_model=self.clip_model,
                    tokenizer=bundle.tokenizer,
                    n_ctx=n_ctx_text,
                    ctx_init=ctx_init,
                    device=bundle.device,
                )
                self.text_encoder = OpenCLIPPromptTextEncoder(self.clip_model)
                self.register_buffer("fixed_text_features", fixed_text_features.to(bundle.device))
                self.to(bundle.device)

            def trainable_state_dict(self) -> dict[str, Any]:
                return {"ctx": self.prompt_learner.ctx.detach().clone()}

            def load_trainable_state_dict(self, state_dict: dict[str, Any]) -> None:
                self.prompt_learner.load_state_dict(state_dict, strict=False)

            def train(self, mode: bool = True):
                super().train(mode)
                self.clip_model.eval()
                return self

            def encode_prompted_text(self):
                prompts = self.prompt_learner()
                text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
                return _normalize(text_features)

            def encode_frozen_image(self, images):
                torch, _, _ = _require_torch()
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(images.to(self.device_name))
                return _normalize(image_features)

            def forward(self, images):
                image_features = self.encode_frozen_image(images)
                text_features = self.encode_prompted_text()
                logit_scale = self.clip_model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()
                fixed_text = _normalize(self.fixed_text_features)
                zero_shot_logits = logit_scale * image_features @ fixed_text.t()
                return {
                    "logits": logits,
                    "zero_shot_logits": zero_shot_logits.detach(),
                    "image_features": image_features.detach(),
                    "zero_shot_image_features": image_features.detach(),
                    "text_features": text_features,
                    "fixed_text_features": fixed_text.detach(),
                }

        return _PromptSRCModel(*args, **kwargs)
