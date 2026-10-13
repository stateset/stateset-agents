"""MultiTurnGRPOTrainer rolls groups out in lockstep batches when it can."""

from __future__ import annotations

import asyncio
from typing import Any

from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

SCENARIOS = [
    {"id": "a", "user_turns": ["hi", "more", "thanks"]},
    {"id": "b", "user_turns": ["refund please"]},
]


def _reply(history: list[dict[str, str]]) -> str:
    last = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    return f"re[{len(history)}]:{last}"


class LengthReward:
    async def compute_reward(self, turns: list[ConversationTurn], context: Any):
        return float(sum(len(t.content) for t in turns if t.role == "assistant"))


class FakeAgent:
    """Deterministic agent exposing both single and batched generation."""

    def __init__(self) -> None:
        self.single_calls = 0
        self.batch_sizes: list[int] = []

    async def reset(self) -> None:
        return None

    async def generate_turn(self, history: Any, context: Any) -> ConversationTurn:
        self.single_calls += 1
        return ConversationTurn(role="assistant", content=_reply(history))

    async def generate_turns_batch(
        self, histories: list[Any], contexts: list[Any] | None = None
    ) -> list[ConversationTurn]:
        self.batch_sizes.append(len(histories))
        return [
            ConversationTurn(role="assistant", content=_reply(h)) for h in histories
        ]

    async def generate_response(self, history: Any, context: Any) -> str:
        return _reply(history)


def _trainer(agent: FakeAgent, **config: Any) -> MultiTurnGRPOTrainer:
    return MultiTurnGRPOTrainer(
        agent=agent,  # type: ignore[arg-type]
        environment=ConversationEnvironment(scenarios=SCENARIOS, max_turns=4),
        reward_fn=LengthReward(),  # type: ignore[arg-type]
        config=TrainingConfig(report_to="none", **config),
    )


def _signature(groups: list[Any]) -> list[Any]:
    return [
        (
            g.scenario_id,
            [
                ([(t.role, t.content) for t in traj.turns], traj.total_reward)
                for traj in g.trajectories
            ],
        )
        for g in groups
    ]


def test_batched_rollouts_match_sequential_and_batch_each_round() -> None:
    batched_agent = FakeAgent()
    sequential_agent = FakeAgent()

    batched = asyncio.run(
        _trainer(batched_agent).generate_trajectories(SCENARIOS, num_generations=3)
    )
    sequential = asyncio.run(
        _trainer(sequential_agent, batched_rollouts=False).generate_trajectories(
            SCENARIOS, num_generations=3
        )
    )

    assert _signature(batched) == _signature(sequential)
    assert [len(g.trajectories) for g in batched] == [3, 3]
    # Round 1: all 6 episodes; scenario b ends after one reply.
    assert batched_agent.batch_sizes == [6, 3, 3]
    assert batched_agent.single_calls == 0
    assert sequential_agent.batch_sizes == []


def test_batched_rollout_failure_falls_back_to_sequential() -> None:
    agent = FakeAgent()

    async def broken(*_: Any, **__: Any) -> list[ConversationTurn]:
        raise RuntimeError("engine down")

    agent.generate_turns_batch = broken  # type: ignore[method-assign]

    groups = asyncio.run(
        _trainer(agent).generate_trajectories(SCENARIOS, num_generations=2)
    )

    assert [len(g.trajectories) for g in groups] == [2, 2]
    assert agent.single_calls > 0
