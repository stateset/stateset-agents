"""Lockstep batched episodes must reproduce run_episode exactly."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.environment_base import (
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from stateset_agents.core.trajectory import ConversationTurn

SCENARIOS = [
    {"id": "a", "user_turns": ["hi", "more", "thanks"]},
    {"id": "b", "user_turns": ["refund please"]},
    {"id": "c", "user_query": "where is my order?", "user_responses": ["ok"]},
    {"id": "d", "user_turns": ["x", "y"]},
]


class LengthReward:
    async def compute_reward(self, turns: list[ConversationTurn], context: Any):
        return float(len(turns[-1].content)) / 100.0


def _reply(history: list[dict[str, str]]) -> str:
    """Deterministic policy: depends only on the visible conversation."""
    last_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), ""
    )
    return f"re[{len(history)}]:{last_user}"


def _signature(trajectory: Any) -> tuple[Any, ...]:
    return (
        [(t.role, t.content) for t in trajectory.turns],
        trajectory.turn_rewards,
        trajectory.total_reward,
        trajectory.metadata["status"],
        trajectory.metadata["scenario"],
    )


def _env() -> ConversationEnvironment:
    return ConversationEnvironment(
        scenarios=SCENARIOS, max_turns=2, reward_fn=LengthReward()  # type: ignore[arg-type]
    )


def test_batched_episodes_match_sequential_run_episode() -> None:
    env = _env()
    batch_sizes: list[int] = []

    async def one(history: list[dict[str, str]], context: Any) -> str:
        return _reply(history)

    async def many(
        histories: list[list[dict[str, str]]], contexts: list[Any]
    ) -> list[str]:
        assert len(histories) == len(contexts)
        batch_sizes.append(len(histories))
        return [_reply(h) for h in histories]

    async def run() -> tuple[list[Any], list[Any]]:
        sequential = [await env.run_episode(one, s) for s in SCENARIOS]
        batched = await env.run_episodes_batched(many, SCENARIOS)
        return sequential, batched

    sequential, batched = asyncio.run(run())

    assert [_signature(t) for t in batched] == [_signature(t) for t in sequential]
    # One call per round; finished episodes drop out of later rounds.
    assert batch_sizes == [4, 3]


def test_batched_episodes_preserve_turn_objects_from_the_agent() -> None:
    env = _env()

    async def many(histories: list[Any], contexts: list[Any]) -> list[Any]:
        return [
            ConversationTurn(
                role="assistant", content=_reply(h), metadata={"token_ids": [i]}
            )
            for i, h in enumerate(histories)
        ]

    [trajectory] = asyncio.run(env.run_episodes_batched(many, [SCENARIOS[1]]))

    assistant = [t for t in trajectory.turns if t.role == "assistant"]
    assert assistant[0].metadata["token_ids"] == [0]


def test_wrong_batch_length_is_rejected() -> None:
    env = _env()

    async def many(histories: list[Any], contexts: list[Any]) -> list[str]:
        return ["only one"]

    with pytest.raises(ValueError, match="for 2 episodes"):
        asyncio.run(env.run_episodes_batched(many, SCENARIOS[:2]))


class _SharedStateEnvironment(Environment):
    """Keeps per-episode state on the instance, so it is not concurrency-safe."""

    def __init__(self) -> None:
        super().__init__(max_turns=2)
        self.turn = 0

    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
        self.turn = 0
        return EnvironmentState(
            episode_id=str((scenario or {}).get("id")),
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={"scenario": scenario or {}},
        )

    async def step(self, state: EnvironmentState, action: Any) -> Any:
        self.turn += 1
        new_state = state.copy()
        new_state.turn_count = self.turn
        done = self.turn >= 2
        if done:
            new_state.status = EpisodeStatus.COMPLETED
        return new_state, float(self.turn), done, {}


def test_environments_without_concurrency_support_run_sequentially() -> None:
    env = _SharedStateEnvironment()
    calls: list[int] = []

    async def many(histories: list[Any], contexts: list[Any]) -> list[str]:
        calls.append(len(histories))
        return ["ok"] * len(histories)

    trajectories = asyncio.run(
        env.run_episodes_batched(many, [{"id": "p"}, {"id": "q"}])
    )

    assert calls == [1, 1, 1, 1]
    assert [t.turn_rewards for t in trajectories] == [[1.0, 2.0], [1.0, 2.0]]
