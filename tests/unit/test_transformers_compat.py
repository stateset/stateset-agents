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
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    model, model_cls = load_generation_model(
        _CausalRejected,
        "zai-org/GLM-5.3-Flash",
        {"device_map": "auto"},
    )

    assert model == ("multimodal", "zai-org/GLM-5.3-Flash", {"device_map": "auto"})
    assert model_cls is Multimodal


def test_preserves_original_error_when_no_composite_loader_accepts(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))

    with pytest.raises(ValueError, match="not a causal auto-model"):
        load_generation_model(_CausalRejected, "acme/unsupported", {})
