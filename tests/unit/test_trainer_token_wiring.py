"""Both GRPO trainers ask the agent for a full turn (token ids + log-probs)
when it offers ``generate_turn``, and keep that metadata on the trajectory."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from stateset_agents.core.trajectory import ConversationTurn


class _TurnAgent:
    """Minimal agent exposing generate_turn (and generate_response)."""

    def __init__(self):
        self.turn_calls = 0
        self.response_calls = 0
        self.model = None

    async def reset(self):
        return None

    async def generate_turn(self, messages, context=None):
        self.turn_calls += 1
        return ConversationTurn(
            role="assistant",
            content="ok",
            metadata={
                "prompt_token_ids": [1, 2, 3],
                "token_ids": [4, 5],
                "sampler_log_probs": [-0.5, -0.7],
            },
        )

    async def generate_response(self, messages, context=None):
        self.response_calls += 1
        return "ok"


class _TextAgent:
    def __init__(self):
        self.model = None

    async def reset(self):
        return None

    async def generate_response(self, messages, context=None):
        return "ok"


def _multi_turn_trainer(agent):
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    env = ConversationEnvironment(
        scenarios=[
            {
                "id": "s1",
                "topic": "t",
                "context": "c",
                "user_responses": ["more please"],
            }
        ],
        max_turns=2,
    )
    config = SimpleNamespace(
        seed=0,
        bf16=False,
        fp16=False,
        use_reference_model=False,
        report_to=None,
        learning_rate=1e-4,
        weight_decay=0.0,
        max_grad_norm=1.0,
        num_generations=2,
        continual_strategy="none",
    )
    reward_fn = MagicMock()
    reward_fn.compute_reward = AsyncMock(return_value=0.5)
    return MultiTurnGRPOTrainer(
        agent=agent, environment=env, reward_fn=reward_fn, config=config
    )


@pytest.mark.asyncio
async def test_multi_turn_trainer_uses_generate_turn_and_keeps_metadata():
    agent = _TurnAgent()
    trainer = _multi_turn_trainer(agent)
    groups = await trainer.generate_trajectories(
        trainer.environment.scenarios, num_generations=2
    )
    assert agent.turn_calls > 0 and agent.response_calls == 0
    assistant_turns = [
        t
        for g in groups
        for tr in g.trajectories
        for t in tr.turns
        if t.role == "assistant"
    ]
    assert assistant_turns
    assert all(t.metadata.get("token_ids") == [4, 5] for t in assistant_turns)


@pytest.mark.asyncio
async def test_multi_turn_trainer_falls_back_to_generate_response():
    trainer = _multi_turn_trainer(_TextAgent())
    groups = await trainer.generate_trajectories(
        trainer.environment.scenarios, num_generations=1
    )
    assistant_turns = [
        t
        for g in groups
        for tr in g.trajectories
        for t in tr.turns
        if t.role == "assistant"
    ]
    assert assistant_turns and all(
        "token_ids" not in t.metadata for t in assistant_turns
    )


@pytest.mark.asyncio
async def test_single_turn_trainer_uses_generate_turn_and_keeps_metadata():
    from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer

    agent = _TurnAgent()
    trainer = object.__new__(SingleTurnGRPOTrainer)
    trainer.agent = agent
    trainer.environment = SimpleNamespace()
    trainer.reward_fn = None
    group, best = await trainer._generate_trajectory_group("hello", 3)
    assert agent.turn_calls == 3 and agent.response_calls == 0
    assert best == "ok"
    for traj in group.trajectories:
        assistant = traj.turns[1]
        assert assistant.role == "assistant"
        assert assistant.metadata["token_ids"] == [4, 5]
