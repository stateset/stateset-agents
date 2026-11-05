"""Algorithm configs reject values that would divide by zero or train nothing.

``VAPOConfig(value_warmup_steps=0)`` passed validation and then raised
``ZeroDivisionError`` at the end of value warmup. The audit covered every
numeric field the dedicated trainers read from their own configs; each now
fails fast in ``validate_for_training`` (run by every trainer's constructor)
with a :class:`ConfigurationError` naming the field.

A duck-typed (legacy) config used to skip validation entirely
(``runtime_state.validate_training_config`` only checked ``TrainingConfig``
instances); it is now validated by the trainer's own config class.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from stateset_agents.core.errors import ConfigurationError
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.dapo_config import DAPOConfig
from stateset_agents.training.gspo_config import GSPOConfig
from stateset_agents.training.runtime_state import validate_training_config
from stateset_agents.training.vapo_config import VAPOConfig


def _gepo_config(**values: Any) -> Any:
    from stateset_agents.training.gepo_trainer import GEPOConfig

    return GEPOConfig(**values)


BAD_VALUES: list[tuple[Any, str, Any]] = [
    # VAPO
    (VAPOConfig, "value_warmup_steps", 0),
    (VAPOConfig, "value_warmup_steps", -1),
    (VAPOConfig, "value_warmup_steps", 2.0),
    (VAPOConfig, "value_hidden_size", 0),
    (VAPOConfig, "value_num_layers", 0),
    (VAPOConfig, "lambda_policy_alpha", 0.0),
    (VAPOConfig, "lambda_policy_alpha", math.inf),
    (VAPOConfig, "lambda_critic", 1.5),
    (VAPOConfig, "gamma", -0.1),
    (VAPOConfig, "clip_eps_low", 1.0),
    (VAPOConfig, "clip_eps_high", -0.1),
    (VAPOConfig, "positive_lm_weight", math.nan),
    (VAPOConfig, "value_loss_coef", -1.0),
    (VAPOConfig, "actor_learning_rate", 0.0),
    (VAPOConfig, "critic_learning_rate", -1e-6),
    # DAPO
    (DAPOConfig, "clip_eps_low", 1.0),
    (DAPOConfig, "clip_eps_high", math.nan),
    (DAPOConfig, "dynamic_sampling_buffer_size", 0),
    (DAPOConfig, "min_accuracy_threshold", -0.5),
    (DAPOConfig, "max_accuracy_threshold", 1.5),
    (DAPOConfig, "max_generation_length", 0),
    (DAPOConfig, "overlong_cache_length", -1),
    (DAPOConfig, "overlong_penalty", math.inf),
    (DAPOConfig, "mini_batch_size", 0),
    # GEPO
    (_gepo_config, "clip_eps", 1.0),
    (_gepo_config, "clip_eps", -0.2),
    # GSPO
    (GSPOConfig, "num_outer_iterations", 0),
    (GSPOConfig, "num_iterations", 0),
    (GSPOConfig, "generations_per_iteration", 0),
    (GSPOConfig, "clip_range_left", 1.0),
    (GSPOConfig, "clip_range_right", -1e-4),
    (GSPOConfig, "mini_batch_size", 0),
]


@pytest.mark.parametrize(
    "factory,field,value",
    BAD_VALUES,
    ids=[f"{getattr(f, '__name__', f)}-{n}={v!r}" for f, n, v in BAD_VALUES],
)
def test_bad_numeric_fields_fail_validation(
    factory: Any, field: str, value: Any
) -> None:
    with pytest.raises(ConfigurationError, match=field):
        factory(**{field: value}).validate_for_training()


def test_min_accuracy_above_max_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="min_accuracy_threshold"):
        DAPOConfig(
            min_accuracy_threshold=0.8, max_accuracy_threshold=0.2
        ).validate_for_training()


@pytest.mark.parametrize(
    "factory", [VAPOConfig, DAPOConfig, GSPOConfig, _gepo_config, TrainingConfig]
)
def test_defaults_validate(factory: Any) -> None:
    factory().validate_for_training()


def test_vapo_trainer_rejects_zero_warmup_before_allocating() -> None:
    torch = pytest.importorskip("torch")
    from stateset_agents.training.vapo_trainer import VAPOTrainer

    with pytest.raises(ConfigurationError, match="value_warmup_steps"):
        VAPOTrainer(
            config=VAPOConfig(value_warmup_steps=0, report_to="none"),
            model=torch.nn.Linear(2, 2),
            tokenizer=None,
            reward_fn=lambda prompt, response: 0.0,
        )


def test_duck_typed_config_is_validated_by_the_trainers_config_class() -> None:
    legacy = SimpleNamespace(**{**VAPOConfig().to_dict(), "value_warmup_steps": 0})
    with pytest.raises(ConfigurationError, match="value_warmup_steps"):
        validate_training_config(legacy, VAPOConfig)
    legacy = SimpleNamespace(num_episodes=0)
    with pytest.raises(ConfigurationError, match="num_episodes"):
        validate_training_config(legacy)


def test_duck_typed_config_with_valid_fields_passes() -> None:
    validate_training_config(SimpleNamespace(num_episodes=3, learning_rate=1e-5))
    validate_training_config(SimpleNamespace(**GSPOConfig().to_dict()), GSPOConfig)
