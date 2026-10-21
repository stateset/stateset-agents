"""Reference models must load the same pinned snapshot as the policy."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from stateset_agents.training import base_trainer
from stateset_agents.training.base_trainer import BaseModelManager
from stateset_agents.training.gspo_trainer import GSPOModelManager

REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))


class _RecordingModelClass:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, name: str, **kwargs: Any) -> _FakeModel:
        self.calls.append((name, kwargs))
        return _FakeModel()


def _config(**overrides: Any) -> SimpleNamespace:
    values = {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": REVISION,
        "beta": 0.02,
        "use_reference_model": True,
        "trust_remote_code": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _manager(cls: type, config: SimpleNamespace) -> Any:
    manager = object.__new__(cls)
    manager.config = config
    manager.ref_model = None
    return manager


def test_gspo_reference_model_uses_policy_revision_and_is_frozen() -> None:
    model_cls = _RecordingModelClass()
    manager = _manager(GSPOModelManager, _config())

    manager._load_reference_model(model_cls, {"torch_dtype": torch.float32})

    [(name, kwargs)] = model_cls.calls
    assert name == "Qwen/Qwen2.5-7B-Instruct"
    assert kwargs["revision"] == REVISION
    assert not manager.ref_model.training
    assert all(not p.requires_grad for p in manager.ref_model.parameters())


def test_gspo_reference_model_skipped_without_kl() -> None:
    model_cls = _RecordingModelClass()
    manager = _manager(GSPOModelManager, _config(beta=0.0))

    manager._load_reference_model(model_cls, {"torch_dtype": torch.float32})

    assert model_cls.calls == []
    assert manager.ref_model is None


def test_base_reference_model_uses_policy_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_cls = _RecordingModelClass()
    monkeypatch.setattr(base_trainer, "_AutoModelForCausalLM", model_cls)
    manager = _manager(BaseModelManager, _config())

    manager._load_reference_model(model_cls, {"torch_dtype": torch.float32})

    [(_, kwargs)] = model_cls.calls
    assert kwargs["revision"] == REVISION
    assert kwargs["trust_remote_code"] is False
    assert all(not p.requires_grad for p in manager.ref_model.parameters())
