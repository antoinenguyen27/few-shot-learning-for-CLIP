from __future__ import annotations

import pytest
import torch

from promptsrc_nc.config import PromptSRCNCConfig
from promptsrc_nc.imagenet_templates import IMAGENET_TEMPLATES
from promptsrc_nc.model import openclip_model_name_for_weights
from promptsrc_nc import train as train_module
from promptsrc_nc.train import _build_scheduler


def test_promptsrc_uses_official_60_teacher_templates() -> None:
    assert len(IMAGENET_TEMPLATES) == 60
    assert IMAGENET_TEMPLATES[0] == "a photo of a {}."
    assert IMAGENET_TEMPLATES[-1] == "itap of the {}."


def test_promptsrc_sgd_defaults_match_dassl_promptsrc() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    config = PromptSRCNCConfig(run_id="test")

    assert hasattr(train_module, "build_prompt_optimizer")
    optimizer = train_module.build_prompt_optimizer([parameter], config, lr=config.stage1_lr)

    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(0.0025)
    assert group["momentum"] == pytest.approx(0.9)
    assert group["weight_decay"] == pytest.approx(5e-4)
    assert group["dampening"] == pytest.approx(0.0)
    assert group["nesterov"] is False


def test_default_precision_is_fp32_for_prompt_optimization() -> None:
    assert PromptSRCNCConfig(run_id="test").precision == "fp32"


def test_openai_vit_b16_uses_quickgelu_openclip_variant() -> None:
    assert openclip_model_name_for_weights("ViT-B-16", "openai") == "ViT-B-16-quickgelu"
    assert openclip_model_name_for_weights("ViT-B/16", "openai") == "ViT-B-16-quickgelu"


def test_non_openai_vit_b16_keeps_requested_openclip_model_name() -> None:
    assert openclip_model_name_for_weights("ViT-B-16", "laion400m_e31") == "ViT-B-16"


def test_nonfinite_prompt_gradients_fail_closed() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    parameter.grad = torch.tensor(float("nan"))

    with pytest.raises(RuntimeError, match="Non-finite prompt gradients"):
        train_module.ensure_finite_gradients([parameter])


def test_stage2_cosine_schedule_starts_at_declared_lr() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.00025)

    _build_scheduler(optimizer, lr=0.00025, epochs=5, warmup_epochs=0, warmup_cons_lr=1e-5)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.00025)


def test_constant_warmup_hands_off_to_full_lr_after_first_epoch() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=0.0025)
    scheduler = _build_scheduler(optimizer, lr=0.0025, epochs=50, warmup_epochs=1, warmup_cons_lr=1e-5)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0025)
