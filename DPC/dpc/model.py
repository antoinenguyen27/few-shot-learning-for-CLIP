"""Dual-prompt model for the repo-native DPC port."""

from __future__ import annotations

from typing import Any

from Promptsrc.promptsrc.model import OpenCLIPPromptTextEncoder, TextPromptLearner, _normalize


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("DPC requires torch. Install the repo dependencies with `pip install -e .`.") from exc
    return torch, nn


class DPCDualPromptModel:
    """Frozen OpenCLIP image tower with backbone and parallel DPC text prompts."""

    def __new__(cls, *args, **kwargs):
        _, nn = _require_torch()

        class _DPCDualPromptModel(nn.Module):
            def __init__(
                self,
                bundle: Any,
                classnames: list[str],
                fixed_text_features: Any,
                n_ctx_text: int,
                ctx_init: str,
                stack_weight: float,
            ):
                super().__init__()
                self.bundle = bundle
                self.clip_model = bundle.model
                self.device_name = bundle.device
                self.stack_weight = stack_weight

                for parameter in self.clip_model.parameters():
                    parameter.requires_grad_(False)

                self.backbone_prompt = TextPromptLearner(
                    classnames=classnames,
                    clip_model=self.clip_model,
                    tokenizer=bundle.tokenizer,
                    n_ctx=n_ctx_text,
                    ctx_init=ctx_init,
                    device=bundle.device,
                )
                self.dpc_prompt = TextPromptLearner(
                    classnames=classnames,
                    clip_model=self.clip_model,
                    tokenizer=bundle.tokenizer,
                    n_ctx=n_ctx_text,
                    ctx_init=ctx_init,
                    device=bundle.device,
                )
                self.text_encoder = OpenCLIPPromptTextEncoder(self.clip_model)
                self.register_buffer("fixed_text_features", fixed_text_features.to(bundle.device))
                self.clone_backbone_to_dpc()
                self.to(bundle.device)

            def clone_backbone_to_dpc(self) -> None:
                self.dpc_prompt.ctx.data.copy_(self.backbone_prompt.ctx.data)

            def freeze_backbone_prompt(self) -> None:
                self.backbone_prompt.ctx.requires_grad_(False)
                self.dpc_prompt.ctx.requires_grad_(True)

            def unfreeze_backbone_prompt(self) -> None:
                self.backbone_prompt.ctx.requires_grad_(True)
                self.dpc_prompt.ctx.requires_grad_(False)

            def trainable_state_dict(self) -> dict[str, Any]:
                return {
                    "backbone_ctx": self.backbone_prompt.ctx.detach().clone(),
                    "dpc_ctx": self.dpc_prompt.ctx.detach().clone(),
                }

            def load_trainable_state_dict(self, state_dict: dict[str, Any]) -> None:
                if "backbone_ctx" in state_dict:
                    self.backbone_prompt.ctx.data.copy_(state_dict["backbone_ctx"].to(self.device_name))
                if "dpc_ctx" in state_dict:
                    self.dpc_prompt.ctx.data.copy_(state_dict["dpc_ctx"].to(self.device_name))

            def train(self, mode: bool = True):
                super().train(mode)
                self.clip_model.eval()
                return self

            def encode_prompt(self, prompt_learner):
                prompts = prompt_learner()
                return _normalize(self.text_encoder(prompts, prompt_learner.tokenized_prompts))

            def encode_frozen_image(self, images):
                torch, _ = _require_torch()
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(images.to(self.device_name))
                return _normalize(image_features)

            def forward(self, images):
                image_features = self.encode_frozen_image(images)
                backbone_text = self.encode_prompt(self.backbone_prompt)
                dpc_text = self.encode_prompt(self.dpc_prompt)
                fixed_text = _normalize(self.fixed_text_features)
                logit_scale = self.clip_model.logit_scale.exp()

                backbone_logits = logit_scale * image_features @ backbone_text.t()
                dpc_logits = logit_scale * image_features @ dpc_text.t()
                fixed_logits = logit_scale * image_features @ fixed_text.t()
                combined_logits = (1.0 - self.stack_weight) * backbone_logits + self.stack_weight * dpc_logits

                return {
                    "logits": combined_logits,
                    "backbone_logits": backbone_logits,
                    "dpc_logits": dpc_logits,
                    "fixed_logits": fixed_logits.detach(),
                    "backbone_text_features": backbone_text,
                    "dpc_text_features": dpc_text,
                    "image_features": image_features.detach(),
                }

        return _DPCDualPromptModel(*args, **kwargs)
