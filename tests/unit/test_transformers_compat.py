"""Tests for composite Transformers checkpoint loading."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from stateset_agents.core.transformers_compat import load_generation_model


class _CausalSuccess:
    @staticmethod
    def from_pretrained(name, **kwargs):
        return ("causal", name, kwargs)


class _CausalRejected:
    @staticmethod
    def from_pretrained(name, **kwargs):
        raise ValueError(f"{name} is not a causal auto-model")


def test_uses_causal_loader_when_checkpoint_is_registered() -> None:
    model, model_cls = load_generation_model(
        _CausalSuccess, "acme/model", {"trust_remote_code": True}
    )

    assert model == ("causal", "acme/model", {"trust_remote_code": True})
    assert model_cls is _CausalSuccess


def test_falls_back_to_new_multimodal_auto_class(monkeypatch) -> None:
    fake_transformers = ModuleType("transformers")

    class Multimodal:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return ("multimodal", name, kwargs)

    fake_transformers.AutoModelForMultimodalLM = Multimodal
    fake_transformers.Glm5NextForConditionalGeneration = Multimodal
    from types import SimpleNamespace

    fake_transformers.FineGrainedFP8Config = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model, model_cls = load_generation_model(
        _CausalRejected,
        "zai-org/GLM-5.3-Flash",
        {"device_map": "auto"},
    )

    assert model[:2] == ("multimodal", "zai-org/GLM-5.3-Flash")
    assert model[2]["device_map"] == "auto"
    assert model[2]["revision"] == "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
    assert "quantization_config" not in model[2]
    assert model_cls is Multimodal


def test_preserves_original_error_when_no_composite_loader_accepts(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))

    with pytest.raises(ValueError, match="not a causal auto-model"):
        load_generation_model(_CausalRejected, "acme/unsupported", {})


def test_frontier_training_requires_the_explicit_fp8_dequantization_path():
    with pytest.raises(RuntimeError, match="frontier-sft"):
        load_generation_model(
            _CausalSuccess, "zai-org/GLM-5.3-Flash", {}, for_training=True
        )


def test_inference_preserves_explicit_native_quantization(monkeypatch):
    fake = ModuleType("transformers")
    fake.Glm5NextForConditionalGeneration = _CausalSuccess
    monkeypatch.setitem(sys.modules, "transformers", fake)
    quantization = object()
    original = {"quantization_config": quantization, "revision": None}
    model, _ = load_generation_model(_CausalSuccess, "zai-org/GLM-5.3-Flash", original)
    assert model[2]["quantization_config"] is quantization
    assert model[2]["revision"] == "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
    assert original["revision"] is None
