"""Tests for TRL entrypoint sync/async boundary handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from stateset_agents.training.trl_grpo_entrypoints import _build_sync_reward_function


def _reward_wrapper() -> MagicMock:
    wrapper = MagicMock()
    wrapper.compute_rewards = AsyncMock(return_value=[0.25, 0.75])
    return wrapper


def test_sync_reward_callback_without_running_loop() -> None:
    wrapper = _reward_wrapper()
    callback = _build_sync_reward_function(wrapper)
    assert callback(["a", "b"], ["p1", "p2"]) == [0.25, 0.75]
    wrapper.compute_rewards.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_reward_callback_inside_running_loop_uses_worker() -> None:
    wrapper = _reward_wrapper()
    callback = _build_sync_reward_function(wrapper)
    assert callback(["a", "b"], ["p1", "p2"], scenario_index=[0, 1]) == [
        0.25,
        0.75,
    ]
    wrapper.compute_rewards.assert_awaited_once_with(
        ["a", "b"], ["p1", "p2"], scenario_index=[0, 1]
    )
