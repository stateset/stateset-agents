"""Composed and averaged rewards are exact and stable across calls.

Hosted CI (Python < 3.12, whose float ``sum`` is not compensated) saw
``0.5384615384615385 != 0.5384615384615384`` between two calls of the same
reward, and a 20-step average of 0.2 come out as 0.20000000000000004.
"""

from __future__ import annotations

import functools
import math

import pytest

from stateset_agents.core.basic_rewards import HelpfulnessReward, SafetyReward
from stateset_agents.core.conversation_environment import ConversationEnvironment
from stateset_agents.core.reward_base import CompositeReward
from stateset_agents.core.trajectory import ConversationTurn

TURNS = [
    ConversationTurn(role="user", content="Where is my order?"),
    ConversationTurn(role="assistant", content="Let me check that for you."),
    ConversationTurn(role="user", content="Thanks"),
    ConversationTurn(role="assistant", content="It ships tomorrow. Anything else?"),
]


def _naive_sum(values, start=0):
    return functools.reduce(lambda a, b: a + b, values, start)


async def test_multi_objective_scores_are_bitwise_stable_across_calls(monkeypatch):
    """Renormalizing stored weights on every call drifted them by an ulp per
    call under a naive float sum; emulate that summation."""
    import stateset_agents.rewards.multi_objective_reward as mor
    from stateset_agents.rewards.kimi_k25_reward import (
        create_kimi_k25_conversational_reward,
    )

    monkeypatch.setattr(mor, "sum", _naive_sum, raising=False)
    reward = create_kimi_k25_conversational_reward()
    weights = [component.weight for component in reward.components]
    scores = {repr((await reward.compute_reward(TURNS)).score) for _ in range(6)}
    assert len(scores) == 1
    assert [component.weight for component in reward.components] == weights


async def test_composite_weighted_sum_is_exactly_rounded():
    composite = CompositeReward(
        [HelpfulnessReward(weight=0.1), SafetyReward(weight=0.2)]
    )
    result = await composite.compute_reward(TURNS)
    parts = result.metadata["component_scores"]
    assert result.score == math.fsum(
        s * w for s, w in zip(parts, (0.1, 0.2), strict=True)
    )


@pytest.mark.parametrize("steps", [1, 3, 7, 20])
async def test_averaged_episode_rewards_are_exact_for_equal_step_scores(steps):
    scenario = {"id": "s", "user_query": "Close my ticket, please."}
    environment = ConversationEnvironment([scenario], max_turns=steps)

    async def pad(history, context):
        return "Let me keep looking into this for you; a few more checks."

    trajectory = await environment.run_episode(pad, scenario)
    assert len(trajectory.turn_rewards) == steps
    assert trajectory.total_reward == 0.2
    assert math.fsum(trajectory.turn_rewards) == trajectory.total_reward
