"""``stateset_agents.training`` resolves real submodules via ``__getattr__``."""

from __future__ import annotations

import importlib

import pytest


def test_submodule_attribute_resolves() -> None:
    training = importlib.import_module("stateset_agents.training")
    trainer_mod = training.trainer
    assert trainer_mod.__name__ == "stateset_agents.training.trainer"
    assert hasattr(trainer_mod, "GRPOTrainer")


def test_unknown_attribute_raises_attribute_error() -> None:
    training = importlib.import_module("stateset_agents.training")
    with pytest.raises(AttributeError):
        _ = training.definitely_not_a_module


def test_unimportable_submodule_reports_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A submodule whose import fails must look absent, not explode."""
    training = importlib.import_module("stateset_agents.training")
    name = "vapo_trainer"
    training.__dict__.pop(name, None)
    real_import_module = importlib.import_module

    def fake_import_module(module_name: str, package: str | None = None):
        if module_name == f"stateset_agents.training.{name}":
            raise ImportError("no torch")
        return real_import_module(module_name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    assert hasattr(training, name) is False
