"""Regression tests for the quick-start example."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from examples import quick_start


def test_create_quickstart_config_uses_stub_backend() -> None:
    config = quick_start.create_quickstart_config("You are helpful.")

    assert config.model_name == "stub://quickstart"
    assert config.use_stub_model is True
    assert config.attn_implementation == "eager"
    assert config.stub_responses == quick_start.QUICKSTART_STUB_RESPONSES


@pytest.mark.asyncio
async def test_basic_example_uses_single_turn_stub_training() -> None:
    created: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, config):
            self.config = config
            created["agent"] = self

        async def initialize(self) -> None:
            created["initialized"] = True

    train_mock = AsyncMock(side_effect=lambda **kwargs: kwargs["agent"])
    test_conversation_mock = AsyncMock()

    with (
        patch.object(quick_start, "MultiTurnAgent", FakeAgent),
        patch.object(quick_start, "train", train_mock),
        patch.object(quick_start, "test_conversation", test_conversation_mock),
    ):
        await quick_start.basic_example()

    agent = created["agent"]
    assert created["initialized"] is True
    assert agent.config.use_stub_model is True
    assert train_mock.await_count == 1
    kwargs = train_mock.await_args.kwargs
    assert kwargs["training_mode"] == "single_turn"
    assert kwargs["num_episodes"] == 4
    assert kwargs["environment"].max_turns == 4
    test_conversation_mock.assert_awaited_once_with(agent)


@pytest.mark.asyncio
async def test_main_runs_only_basic_example() -> None:
    basic_example_mock = AsyncMock()
    auto_training_example_mock = AsyncMock()
    custom_reward_example_mock = AsyncMock()

    with (
        patch.object(quick_start, "basic_example", basic_example_mock),
        patch.object(quick_start, "auto_training_example", auto_training_example_mock),
        patch.object(quick_start, "custom_reward_example", custom_reward_example_mock),
    ):
        await quick_start.main()

    basic_example_mock.assert_awaited_once()
    auto_training_example_mock.assert_not_awaited()
    custom_reward_example_mock.assert_not_awaited()
