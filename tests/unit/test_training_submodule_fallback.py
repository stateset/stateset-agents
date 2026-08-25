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
