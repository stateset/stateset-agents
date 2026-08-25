"""``core.types.TrainingConfig`` remains importable as a deprecation shim."""

from __future__ import annotations

import pytest


def test_training_config_shim_warns_and_returns_canonical_class() -> None:
    import stateset_agents.core.types as types_mod
    from stateset_agents.training.config import TrainingConfig as Canonical

    with pytest.warns(DeprecationWarning, match="stateset_agents.training.config"):
        shimmed = types_mod.TrainingConfig

    assert shimmed is Canonical


def test_unknown_attribute_still_raises() -> None:
    import stateset_agents.core.types as types_mod

    with pytest.raises(AttributeError):
        _ = types_mod.NoSuchNameAtAll
