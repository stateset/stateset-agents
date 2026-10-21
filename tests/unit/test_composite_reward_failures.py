"""A failing component must not count as a zero score in CompositeReward."""

import pytest

from stateset_agents.core.reward_base import (
    CompositeReward,
    RewardFunction,
    RewardResult,
)


class _Fixed(RewardFunction):
    def __init__(self, score, weight=0.5, fail=False, name="fixed"):
        super().__init__(weight=weight, name=name)
        self.score, self.fail = score, fail

    async def compute_reward(self, turns, context=None):
        if self.fail:
            raise ValueError("boom")
        return RewardResult(score=self.score, breakdown={})


@pytest.mark.asyncio
async def test_failed_component_weight_is_redistributed():
    healthy = CompositeReward([_Fixed(1.0, name="a"), _Fixed(1.0, name="b")])
    degraded = CompositeReward(
        [_Fixed(1.0, name="a"), _Fixed(1.0, fail=True, name="b")]
    )
    assert (await healthy.compute_reward([])).score == pytest.approx(1.0)
    result = await degraded.compute_reward([])
    assert result.score == pytest.approx(1.0)
    assert result.metadata["failed_components"] == ["b"]


@pytest.mark.asyncio
async def test_unequal_weights_keep_configured_scale():
    reward = CompositeReward(
        [
            _Fixed(0.4, weight=2.0, name="a"),
            _Fixed(1.0, weight=1.0, fail=True, name="b"),
        ]
    )
    # survivor carries the full configured weight of 3.0
    assert (await reward.compute_reward([])).score == pytest.approx(1.2)
