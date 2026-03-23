from __future__ import annotations

import pytest

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards import multi_objective_components as components_module
from stateset_agents.rewards import multi_objective_reward as reward_module


class StaticScoreComponent(components_module.BaseRewardComponent):
    def __init__(self, name: str, weight: float, score: float):
        super().__init__(name=name, weight=weight)
        self._score = score

    async def compute_score(self, turns, context=None) -> float:
        return self._score


def test_multi_objective_reward_reexports_component_classes():
    assert reward_module.BaseRewardComponent is components_module.BaseRewardComponent
    assert reward_module.LengthRewardComponent is components_module.LengthRewardComponent
    assert (
        reward_module.ReasoningRewardComponent
        is components_module.ReasoningRewardComponent
    )


@pytest.mark.asyncio
async def test_weighted_average_uses_weights_instead_of_component_count():
    reward = reward_module.MultiObjectiveRewardFunction(
        components=[
            StaticScoreComponent("low", weight=0.25, score=0.2),
            StaticScoreComponent("high", weight=0.75, score=1.0),
        ],
        normalization_method="weighted_average",
    )

    result = await reward.compute_reward(
        turns=[{"role": "assistant", "content": "Response"}]
    )

    assert result.score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_reasoning_component_preserves_conversation_turn_metadata():
    reward = reward_module.MultiObjectiveRewardFunction(
        components=[
            reward_module.ReasoningRewardComponent(
                weight=1.0,
                min_length=1,
                optimal_length=4,
            )
        ]
    )

    result = await reward.compute_reward(
        turns=[
            ConversationTurn(
                role="assistant",
                content="Final answer",
                metadata={"reasoning": "step one step two"},
            )
        ]
    )

    assert result.score == pytest.approx(1.0)
