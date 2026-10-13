"""Calibrated rewards must be a fixed monotone transform between refreshes.

Updating running statistics on every call made the score depend on call order:
within one GRPO group, equal raw rewards got different calibrated values and a
lower raw reward could outrank a higher one."""

import pytest

from stateset_agents.core.reward_base import RewardFunction, RewardResult
from stateset_agents.training.reward_calibration import (
    CalibratedRewardFunction,
    RewardNormalizer,
)


class _Fixed(RewardFunction):
    def __init__(self):
        super().__init__(weight=1.0, name="fixed")
        self.queue: list[float] = []

    async def compute_reward(self, turns, context=None):
        return RewardResult(score=self.queue.pop(0), breakdown={}, components={})


async def _scores(fn, base, raw):
    base.queue.extend(raw)
    return [(await fn.compute_reward([])).score for _ in raw]


@pytest.mark.asyncio
async def test_scores_within_an_interval_preserve_order_and_ties():
    base = _Fixed()
    fn = CalibratedRewardFunction(
        base, RewardNormalizer(method="z_score"), calibration_interval=4
    )
    await _scores(fn, base, [0.0, 1.0, 2.0, 3.0])  # first interval: stats freeze
    group = [0.1, 0.0, 0.0]
    scores = await _scores(fn, base, group)
    assert scores[1] == scores[2]
    assert scores[0] > scores[1]
    # Frozen stats from the first interval: mean 1.5, population std ~1.118.
    assert scores[1] == pytest.approx((0.0 - 1.5) / 1.118033988749895)


@pytest.mark.asyncio
async def test_first_interval_is_identity():
    base = _Fixed()
    fn = CalibratedRewardFunction(
        base, RewardNormalizer(method="z_score"), calibration_interval=10
    )
    assert await _scores(fn, base, [0.1, 0.0, 0.0, 0.09]) == [0.1, 0.0, 0.0, 0.09]
