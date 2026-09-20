"""Regression tests for mcore architecture dispatch."""

from types import SimpleNamespace

import pytest

from verl.models.mcore import registry


OMNI_ARCHITECTURE = "Qwen3_5OmniMoeForConditionalGeneration"
VL_ARCHITECTURE = "Qwen3_5MoeForConditionalGeneration"


@pytest.mark.parametrize(
    "selector_name, factory_name",
    [
        ("get_mcore_forward_fn", "model_forward_gen"),
        ("get_mcore_forward_fused_fn", "fused_forward_model_gen"),
        ("get_mcore_forward_fused_model_engine_fn", "fused_forward_model_engine"),
    ],
)
def test_omni_architecture_uses_vision_forward(selector_name, factory_name, monkeypatch):
    calls = []

    def fake_factory(vision_model):
        calls.append(vision_model)
        return object()

    monkeypatch.setattr(registry, factory_name, fake_factory)
    selector = getattr(registry, selector_name)

    selector(SimpleNamespace(architectures=[OMNI_ARCHITECTURE]))
    assert calls == [True]


def test_ordinary_qwen35_vl_dispatch_remains_vision_aware(monkeypatch):
    calls = []

    def fake_factory(vision_model):
        calls.append(vision_model)
        return object()

    monkeypatch.setattr(registry, "fused_forward_model_engine", fake_factory)
    registry.get_mcore_forward_fused_model_engine_fn(SimpleNamespace(architectures=[VL_ARCHITECTURE]))
    assert calls == [True]


def test_language_architecture_still_uses_language_forward(monkeypatch):
    calls = []

    def fake_factory(vision_model):
        calls.append(vision_model)
        return object()

    monkeypatch.setattr(registry, "fused_forward_model_engine", fake_factory)
    registry.get_mcore_forward_fused_model_engine_fn(SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"]))
    assert calls == [False]
