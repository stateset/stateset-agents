"""Multi-objective reward combination and length scoring."""

import pytest

from stateset_agents.rewards.multi_objective_components import LengthRewardComponent


@pytest.mark.asyncio
@pytest.mark.parametrize("words", [200, 201, 350, 500])
async def test_length_optimal_range_beyond_default_max_scores_one(words):
    component = LengthRewardComponent(optimal_range=(200, 500))
    turns = [{"role": "assistant", "content": "w " * words}]
    assert await component.compute_score(turns) == 1.0


@pytest.mark.asyncio
async def test_geometric_mean_of_perfect_components_is_one():
    from stateset_agents.rewards.multi_objective_reward import (
        MultiObjectiveRewardFunction,
    )

    components = [
        LengthRewardComponent(name=f"len{i}", weight=0.25, optimal_range=(1, 5))
        for i in range(4)
    ]
    reward = MultiObjectiveRewardFunction(
        components=components, normalization_method="geometric_mean"
    )
    result = await reward.compute_reward(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a b c"}]
    )
    assert result.score == pytest.approx(1.0)
