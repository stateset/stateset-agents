"""Reward failures are counted by default and fatal under ``strict_rewards``."""

from __future__ import annotations

from typing import Any

import pytest

from stateset_agents.core.environment_base import (
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.training.config import TrainingConfig


class BrokenJudge:
    """A reward function whose judge call always fails."""

    async def compute_reward(self, turns: Any, context: Any = None) -> Any:
        raise RuntimeError("judge unavailable")


class OneTurn(Environment):
    """Ends after one agent turn; optionally fails inside the rollout."""

    def __init__(self, *, fail_step: bool = False, **kwargs: Any) -> None:
        super().__init__(max_turns=1, **kwargs)
        self.fail_step = fail_step

    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
        return EnvironmentState("episode", 0, EpisodeStatus.ONGOING, {})

    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
        if self.fail_step:
            raise RuntimeError("checker unavailable")
        state.status = EpisodeStatus.COMPLETED
        return state, 1.0, True, {}


async def agent(history: Any, context: Any) -> ConversationTurn:
    return ConversationTurn(role="assistant", content="ok")


@pytest.mark.asyncio
async def test_failed_reward_is_counted_and_flagged_by_default() -> None:
    env = OneTurn(reward_fn=BrokenJudge())
    trajectory = await env.run_episode(agent)
    assert trajectory.metadata["reward_error"] is True
    assert env.reward_failures == 1
    assert await env.get_reward(trajectory) == 0.5
    assert env.reward_failures == 2


@pytest.mark.asyncio
async def test_strict_rewards_raise_instead_of_substituting() -> None:
    env = OneTurn(reward_fn=BrokenJudge(), strict_rewards=True)
    with pytest.raises(RuntimeError, match="judge unavailable"):
        await env.run_episode(agent)
    healthy = OneTurn()
    trajectory = await healthy.run_episode(agent)
    assert trajectory.metadata["reward_error"] is False
    with pytest.raises(RuntimeError, match="judge unavailable"):
        await env.get_reward(trajectory)
    assert env.reward_failures == 0


class Agent:
    model = None
    tokenizer = None

    async def generate_turn(self, history: Any, context: Any) -> ConversationTurn:
        return ConversationTurn(role="assistant", content="ok")


def trainer(environment: Environment, *, strict: bool) -> Any:
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    config = TrainingConfig(
        model_name="local", num_generations=2, report_to="none", strict_rewards=strict
    )
    return MultiTurnGRPOTrainer(Agent(), environment, config=config)


@pytest.mark.asyncio
async def test_trainer_drops_failed_rollouts_unless_strict() -> None:
    pytest.importorskip("torch")
    lenient = trainer(OneTurn(fail_step=True), strict=False)
    assert await lenient.generate_trajectories([{"id": "a"}]) == []
    with pytest.raises(RuntimeError, match="checker unavailable"):
        await trainer(OneTurn(fail_step=True), strict=True).generate_trajectories(
            [{"id": "a"}]
        )


@pytest.mark.asyncio
async def test_strict_trainer_rejects_a_substituted_reward() -> None:
    pytest.importorskip("torch")
    strict = trainer(OneTurn(reward_fn=BrokenJudge()), strict=True)
    with pytest.raises(RuntimeError, match="reward could not be computed"):
        await strict.generate_trajectories([{"id": "a"}])
    lenient = trainer(OneTurn(reward_fn=BrokenJudge()), strict=False)
    groups = await lenient.generate_trajectories([{"id": "a"}])
    assert all(t.metadata["reward_error"] for t in groups[0].trajectories)
