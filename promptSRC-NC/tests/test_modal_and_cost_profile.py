from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from promptsrc_nc import cost_profile as cost_profile_module
from promptsrc_nc.config import MODAL_GPU_PRICE_PER_SECOND


def test_cost_profile_prices_detected_gpu_over_requested_label() -> None:
    assert hasattr(cost_profile_module, "_price_for_profile")
    assert cost_profile_module._price_for_profile("L4", "T4") == pytest.approx(MODAL_GPU_PRICE_PER_SECOND["T4"])
    assert cost_profile_module._price_for_profile("T4", "L4") == pytest.approx(MODAL_GPU_PRICE_PER_SECOND["L4"])


def test_modal_dispatcher_exposes_checkpoint_split_and_prepare_args() -> None:
    spec = importlib.util.spec_from_file_location("modal_app", Path("promptSRC-NC/modal_app.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    signature = inspect.signature(module.run.info.raw_f)

    assert "checkpoint_ref" in signature.parameters
    assert "split" in signature.parameters
    assert "datasets" in signature.parameters
    assert "prepare_shots" in signature.parameters
    assert "prepare_seeds" in signature.parameters


def test_modal_app_wires_huggingface_secret_and_prepare_weights() -> None:
    source = Path("promptSRC-NC/modal_app.py").read_text()

    assert 'modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])' in source
    assert "def prepare_weights(" in source
    assert 'elif action == "prepare_weights":' in source
    assert "secrets=HF_SECRETS" in source


def test_modal_image_adds_local_source_after_build_steps() -> None:
    source = Path("promptSRC-NC/modal_app.py").read_text()

    add_local_dir = source.index('.add_local_dir("promptSRC-NC/promptsrc_nc"')

    assert source.index('.uv_sync("promptSRC-NC")') < add_local_dir
    assert source.index('.workdir("/root")') < add_local_dir
    assert source.index(".env(") < add_local_dir


def test_modal_dispatcher_does_not_call_removed_function_with_options() -> None:
    source = Path("promptSRC-NC/modal_app.py").read_text()

    assert ".with_options(" not in source


def test_modal_dispatcher_routes_gpu_specific_smoke_function(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("modal_app", Path("promptSRC-NC/modal_app.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeFunction:
        def __init__(self, name: str) -> None:
            self.name = name
            self.calls: list[tuple[object, ...]] = []

        def remote(self, *args: object) -> str:
            self.calls.append(args)
            return f"{self.name}-ok"

    smoke_t4 = FakeFunction("smoke_t4")
    smoke_l4 = FakeFunction("smoke_l4")
    monkeypatch.setattr(module, "smoke_test", smoke_t4)
    monkeypatch.setattr(module, "smoke_test_l4", smoke_l4, raising=False)

    module.run.info.raw_f(
        action="smoke_test",
        run_id="smoke-test",
        dataset="eurosat",
        shots=1,
        seed=1,
        backbone="ViT-B-16",
        pretrained="openai",
        gpu="L4",
    )

    assert smoke_t4.calls == []
    assert smoke_l4.calls == [("smoke-test", "eurosat", 1, 1, "ViT-B-16", "openai")]
    assert "smoke_l4-ok" in capsys.readouterr().out


def test_modal_dispatcher_routes_prepare_weights(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("modal_app", Path("promptSRC-NC/modal_app.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeFunction:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def remote(self, *args: object) -> str:
            self.calls.append(args)
            return "prepare-weights-ok"

    fake_prepare_weights = FakeFunction()
    monkeypatch.setattr(module, "prepare_weights", fake_prepare_weights, raising=False)

    module.run.info.raw_f(
        action="prepare_weights",
        run_id="unused",
        dataset="eurosat",
        shots=1,
        seed=1,
        backbone="ViT-B-16",
        pretrained="openai",
    )

    assert fake_prepare_weights.calls == [("ViT-B-16", "openai")]
    assert "prepare-weights-ok" in capsys.readouterr().out
