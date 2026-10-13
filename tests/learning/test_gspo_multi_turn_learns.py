"""On-policy multi-turn GSPO learns from a reward on a later turn.

Episodes are genuine two-turn conversations: ``ConversationEnvironment`` opens
with the scenario's first scripted user message, the real ``MultiTurnAgent``
replies, the environment answers with the second scripted user message, and
the agent replies again. Only that second reply is rewarded, so the policy can
improve only through the multi-turn rollout path. ``MultiTurnGRPOTrainer``
optimizes the ``gspo`` objective preset (length-normalized sequence ratio per
assistant turn) on its batched token path.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("transformers")

from tests.learning._toy import (  # noqa: E402
    HELDOUT_PROMPTS,
    TRAIN_PROMPTS,
    DigitReward,
    make_agent,
    make_model,
    make_tokenizer,
    seeds,
    target_mass,
)

pytestmark = pytest.mark.learning

STEPS = 10
FOLLOW_UP = "and again"


def _second_turn_contexts(agent: Any) -> list[str]:
    """Held-out conversations rendered right before the rewarded reply."""
    return [
        agent._render_prompt(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": FOLLOW_UP},
            ]
        )[0]
        for prompt in HELDOUT_PROMPTS
    ]


def _train(seed: int, *, sign: float) -> tuple[float, float, list[Any], list[Any]]:
    import torch

    from stateset_agents.core.conversation_environment import (
        ConversationEnvironment,
    )
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    tokenizer = make_tokenizer()
    model = make_model(seed)
    agent = make_agent(model, tokenizer)
    scenarios = [
        {"id": f"s{i}", "user_turns": [prompt, FOLLOW_UP]}
        for i, prompt in enumerate(TRAIN_PROMPTS)
    ]
    config = TrainingConfig(
        num_episodes=STEPS,
        num_generations=8,
        learning_rate=1e-2,
        weight_decay=0.0,
        gradient_accumulation_steps=1,
        max_prompt_length=64,
        max_completion_length=8,
        beta=0.0,
        bf16=False,
        fp16=False,
        report_to="none",
        seed=seed,
    )
    config.objective = "gspo"
    trainer = MultiTurnGRPOTrainer(
        agent=agent,
        environment=ConversationEnvironment(scenarios=scenarios, max_turns=4),
        reward_fn=DigitReward(sign),
        config=config,
    )
    contexts = _second_turn_contexts(agent)
    before = target_mass(model, tokenizer, contexts)
    metrics: list[Any] = []
    episodes: list[Any] = []

    async def run() -> None:
        await trainer.initialize()
        torch.manual_seed(seed)  # fixed sampling stream
        for step in range(STEPS):
            batch = [scenarios[(2 * step) % 4], scenarios[(2 * step + 1) % 4]]
            groups = await trainer.generate_trajectories(batch, num_generations=8)
            episodes.extend(t for group in groups for t in group.trajectories)
            metrics.append(await trainer.training_step(groups))

    asyncio.run(run())
    return before, target_mass(model, tokenizer, contexts), metrics, episodes


def _non_system_roles(trajectory: Any) -> list[str]:
    return [turn.role for turn in trajectory.turns if turn.role != "system"]


@pytest.mark.parametrize("seed", seeds())
def test_multi_turn_gspo_learns_from_a_later_turn_reward(seed: int) -> None:
    before, after, metrics, episodes = _train(seed, sign=1.0)

    assert all(m["path"] == "token" for m in metrics), [m.get("path") for m in metrics]
    assert all(m["optimizer_step"] for m in metrics)
    for trajectory in episodes:
        assert _non_system_roles(trajectory) == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        users = [t.content for t in trajectory.turns if t.role == "user"]
        assert users[0] in TRAIN_PROMPTS and users[1] == FOLLOW_UP
    assert after > before + 0.15, (before, after)


@pytest.mark.parametrize("seed", seeds())
def test_multi_turn_gspo_negated_reward_decreases_target_probability(
    seed: int,
) -> None:
    before, after, metrics, _ = _train(seed, sign=-1.0)

    assert all(m["path"] == "token" for m in metrics)
    assert after < before - 0.025, (before, after)
