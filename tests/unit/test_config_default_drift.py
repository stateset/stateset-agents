"""A legacy duck-typed config that omits a field trains with the
``TrainingConfig`` default, not with a default restated (and drifted) in the
trainer. Each case pinned here used to differ from the dataclass.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import single_turn_trainer  # noqa: E402
from stateset_agents.training.config import TrainingConfig  # noqa: E402
from stateset_agents.training.multi_turn_trainer import (  # noqa: E402
    MultiTurnGRPOTrainer,
)

DEFAULTS = TrainingConfig()


def _single_turn(**config: object) -> single_turn_trainer.SingleTurnGRPOTrainer:
    return single_turn_trainer.SingleTurnGRPOTrainer(
        SimpleNamespace(model=torch.nn.Linear(2, 1)),
        SimpleNamespace(reset=AsyncMock(), step=AsyncMock()),
        config=SimpleNamespace(bf16=False, **config),
    )


def _multi_turn(**config: object) -> MultiTurnGRPOTrainer:
    return MultiTurnGRPOTrainer(
        SimpleNamespace(model=torch.nn.Linear(2, 1), tokenizer=None),
        SimpleNamespace(reset=AsyncMock(), step=AsyncMock()),
        config=SimpleNamespace(bf16=False, **config),
    )


def test_single_turn_optimizer_uses_the_dataclass_learning_rate_and_betas() -> None:
    """It used 5e-5 and beta2 0.999 against the dataclass's 5e-6 and 0.99."""
    trainer = _single_turn()
    trainer._setup_optimizer()
    group = trainer.optimizer.param_groups[0]
    assert group["lr"] == DEFAULTS.learning_rate == 5e-6
    assert group["betas"] == (DEFAULTS.adam_beta1, DEFAULTS.adam_beta2)
    assert group["betas"][1] == 0.99


@pytest.mark.asyncio
async def test_single_turn_samples_the_dataclass_group_size() -> None:
    """It sampled 8 replies per prompt against the dataclass's 16."""
    trainer = _single_turn()
    seen = []

    async def group(*, prompt: str, num_generations: int, context: object) -> object:
        seen.append(num_generations)
        raise RuntimeError("stop after sampling")

    trainer._generate_trajectory_group = group
    with pytest.raises(RuntimeError, match="stop after sampling"):
        await trainer._train_turn({"prompt": "q"}, None)
    assert seen == [DEFAULTS.num_generations] == [16]


@pytest.mark.parametrize("factory", [_single_turn, _multi_turn])
def test_episode_and_accumulation_budgets_use_the_dataclass(factory) -> None:
    """Single-turn assumed 10 episodes, multi-turn 100 (and turned an
    explicit 0 into 100); both assumed no gradient accumulation."""
    trainer = factory()
    assert trainer._get_grad_accum_steps() == DEFAULTS.gradient_accumulation_steps
    if factory is _single_turn:
        assert trainer._prepare_training_loop()[0] == DEFAULTS.num_episodes
    else:
        assert trainer._planned_optimizer_steps() > 100
        assert _multi_turn(num_episodes=0)._planned_optimizer_steps() == 0


@pytest.mark.asyncio
async def test_multi_turn_rejects_an_explicit_empty_group() -> None:
    """``num_generations=0`` fell through an ``or`` chain to the config."""
    trainer = _multi_turn(num_generations=4)
    with pytest.raises(ValueError, match="num_generations must be a positive"):
        await trainer.generate_trajectories([{}], num_generations=0)
