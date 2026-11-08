"""The retired TrainingConfig alias cannot recreate a core-to-training edge."""

from __future__ import annotations

import pytest


def test_training_config_alias_is_removed_and_canonical_class_remains() -> None:
    import stateset_agents.core.types as types_mod
    from stateset_agents.training.config import TrainingConfig as Canonical

    with pytest.raises(AttributeError):
        _ = types_mod.TrainingConfig
    assert Canonical.__module__ == "stateset_agents.training.config"


def test_unknown_attribute_still_raises() -> None:
    import stateset_agents.core.types as types_mod

    with pytest.raises(AttributeError):
        _ = types_mod.NoSuchNameAtAll
