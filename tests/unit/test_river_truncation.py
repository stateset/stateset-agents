"""Truncation must close the episode, preserve attribution, and prevent success."""

import json
import sys
from types import SimpleNamespace

import pytest

from stateset_agents.core.environment_base import EpisodeStatus
from stateset_agents.core.environments.refund_environment import RefundEnvironment
from stateset_agents.remote.river_environment import (
    river_environment_factory,
    trajectory_truncation,
)
from stateset_agents.training.river_refund import EVALUATION_SETTINGS


@pytest.mark.parametrize(
    "value", [True, None, "-1", float("nan"), float("inf"), -float("inf")]
)
def test_invalid_truncation_reward_fails_before_sdk_import(monkeypatch, value):
    monkeypatch.setitem(sys.modules, "river_client", None)
    with pytest.raises(ValueError, match="finite number"):
        river_environment_factory(RefundEnvironment, truncation_reward=value)


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=SimpleNamespace(Env=object))
    )
    return river_environment_factory(
        RefundEnvironment, truncation_reward=EVALUATION_SETTINGS["truncation_reward"]
    )()


@pytest.mark.asyncio
async def test_truncation_replaces_partial_rewards_and_closes_episode(bridge):
    row = {"order_id": "A", "amount_cents": 100, "eligible": True}
    await bridge.reset(row)
    trajectory = SimpleNamespace(
        truncated="generated_tokens", metrics={"task_success": 1}
    )
    bridge.total_reward = 0.5  # Partial credit cannot survive terminal truncation.
    assert await bridge.on_truncated(trajectory, row, "generated_tokens") == -1
    assert trajectory.metrics["task_success"] == 0
    assert trajectory.stateset_episode_id == "A"
    assert bridge.state.status == EpisodeStatus.TIMEOUT
    assert trajectory_truncation(trajectory) == "generated_tokens"
    assert await bridge.reward(trajectory, row) == -1
    with pytest.raises(RuntimeError, match="reset"):
        await bridge.on_turn(trajectory)
    await bridge.reset(row)
    assert not bridge.done and bridge.total_reward == 0


@pytest.mark.asyncio
async def test_environment_timeout_is_visible_without_overwriting_engine_state(bridge):
    row = {"order_id": "A", "amount_cents": 100, "eligible": True}
    await bridge.reset(row)
    trajectory = SimpleNamespace(truncated=None, metrics={}, messages=[])
    for turn in range(4):
        tool = "refund" if turn == 1 else "lookup_order"
        args = {"order_id": "A", **({"amount_cents": 100} if turn == 1 else {})}
        trajectory.messages.append(
            {"role": "assistant", "content": json.dumps({"tool": tool, "args": args})}
        )
        observations = await bridge.on_turn(trajectory)
    assert observations is None
    assert await bridge.reward(trajectory, row) == -1
    assert trajectory.truncated is None
    assert trajectory_truncation(trajectory) == "environment_timeout"
    assert trajectory.metrics["environment_timeout"] == 1
    assert trajectory.metrics["task_success"] == 0


def test_effective_cause_prioritizes_engine_and_rejects_invalid_values():
    assert trajectory_truncation(SimpleNamespace()) is None
    assert (
        trajectory_truncation(
            SimpleNamespace(truncated="turns", stateset_truncated="environment_timeout")
        )
        == "turns"
    )
    for field in ("truncated", "stateset_truncated"):
        for value in (False, "", 5):
            with pytest.raises(ValueError, match="nonempty text"):
                trajectory_truncation(SimpleNamespace(**{field: value}))
