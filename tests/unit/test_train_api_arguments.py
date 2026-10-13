"""``train()`` must honour or reject every argument instead of silently dropping it."""

from __future__ import annotations

from typing import Any

import pytest

from stateset_agents.training import train as train_fn
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.train import _build_training_config


def _build(**kwargs: Any) -> TrainingConfig:
    params: dict[str, Any] = {
        "profile": "balanced",
        "num_episodes": 3,
        "config_overrides": None,
        "resume_from_checkpoint": None,
        "objective": None,
        "objective_overrides": None,
        "extra_config": {},
    }
    params.update(kwargs)
    return _build_training_config(**params)


def test_objective_argument_selects_preset():
    config = _build(objective="dapo", objective_overrides={"clip_high": 0.3})
    assert config.objective == "dapo"
    assert config.objective_overrides == {"clip_high": 0.3}


def test_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="not a known preset"):
        _build(objective="not-a-preset")


def test_config_field_keyword_arguments_reach_the_config():
    config = _build(extra_config={"learning_rate": 1e-5, "seed": 7})
    assert config.learning_rate == 1e-5
    assert config.seed == 7
    assert config.num_episodes == 3


def test_config_overrides_still_apply():
    config = _build(config_overrides={"learning_rate": 2e-5})
    assert config.learning_rate == 2e-5


def test_conflicting_duplicate_settings_are_rejected():
    with pytest.raises(TypeError, match="learning_rate"):
        _build(
            config_overrides={"learning_rate": 2e-5},
            extra_config={"learning_rate": 1e-5},
        )
    with pytest.raises(TypeError, match="objective"):
        _build(objective="dapo", config_overrides={"objective": "gspo"})


def test_unknown_keyword_argument_is_rejected_with_suggestion():
    with pytest.raises(TypeError, match="learning_rate"):
        _build(extra_config={"learnig_rate": 1e-5})


@pytest.mark.asyncio
async def test_train_rejects_unknown_keyword_before_training():
    with pytest.raises(TypeError, match="learnig_rate"):
        await train_fn(agent=object(), environment=object(), learnig_rate=1e-5)


@pytest.mark.asyncio
async def test_train_rejects_unknown_training_mode():
    with pytest.raises(ValueError, match="training_mode"):
        await train_fn(agent=object(), environment=object(), training_mode="multiturn")
