"""Bounded checks connecting the Lean arithmetic models to Python tensors."""

from __future__ import annotations

import itertools
import math
from typing import Any

import pytest

from stateset_agents.rewards.multi_objective_components import BaseRewardComponent
from stateset_agents.rewards.multi_objective_reward import MultiObjectiveRewardFunction


class FixedComponent(BaseRewardComponent):
    """A deterministic component for checking weighted reward composition."""

    def __init__(
        self, name: str, weight: float, score: float, fails: bool = False
    ) -> None:
        super().__init__(name, weight)
        self.score = score
        self.fails = fails

    async def compute_score(
        self, turns: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> float:
        if self.fails:
            raise ValueError("component failed")
        return self.score


@pytest.mark.asyncio
@pytest.mark.parametrize("normalization_method", ["weighted_sum", "weighted_average"])
async def test_bounded_weighted_reward_matches_integer_model(
    normalization_method: str,
) -> None:
    """Enumerate both sum and average paths within the Lean domain."""
    for weight_a, weight_b, score_a, score_b, fail_a, fail_b in itertools.product(
        (0, 1, 2),
        (0, 1, 2),
        (0.0, 0.5, 1.0),
        (0.0, 0.5, 1.0),
        (False, True),
        (False, True),
    ):
        reward = MultiObjectiveRewardFunction(
            components=[
                FixedComponent("a", weight_a, score_a, fail_a),
                FixedComponent("b", weight_b, score_b, fail_b),
            ],
            normalization_method=normalization_method,
        )
        result = await reward.compute_reward(turns=[])
        total = weight_a + weight_b
        expected = (
            (
                (0 if fail_a else weight_a * score_a)
                + (0 if fail_b else weight_b * score_b)
            )
            / total
            if total
            else 0.0
        )
        assert result.score == pytest.approx(expected)
        assert 0.0 <= result.score <= 1.0


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_reward_rejects_weights_outside_lean_domain(weight: float) -> None:
    """The runtime enforces the proof's nonnegative finite weight premise."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        MultiObjectiveRewardFunction(
            components=[FixedComponent("invalid", weight, 0.5)]
        )
    reward = MultiObjectiveRewardFunction(
        components=[FixedComponent("valid", 1.0, 0.5)]
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        reward.add_component(FixedComponent("invalid", weight, 0.5))
    assert len(reward.components) == 1


@pytest.mark.asyncio
async def test_nonfinite_component_score_contributes_zero() -> None:
    """An invalid component is a failed component in the Lean abstraction."""
    reward = MultiObjectiveRewardFunction(
        components=[
            FixedComponent("invalid", 1.0, float("nan")),
            FixedComponent("valid", 1.0, 1.0),
        ]
    )
    result = await reward.compute_reward(turns=[])
    assert result.score == pytest.approx(0.5)
    assert result.components == {"invalid": 0.0, "valid": 1.0}


def test_bounded_clipping_and_surrogate_match_integer_model() -> None:
    """Compare the Lean clipping formula with the production tensor helpers."""
    torch = pytest.importorskip("torch")
    from stateset_agents.training import rl_losses

    for log_ratio in (-50, -20, -2, 0, 2, 20, 50):
        ratio = rl_losses.safe_exp_ratio(
            torch.tensor(float(log_ratio), dtype=torch.float64), clamp=20.0
        )
        assert ratio.item() == pytest.approx(math.exp(max(-20, min(20, log_ratio))))

    for ratio_value, advantage in itertools.product(
        (0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0), (-2.0, -1.0, 0.0, 1.0, 2.0)
    ):
        ratio = torch.tensor(ratio_value, dtype=torch.float64)
        adv = torch.tensor(advantage, dtype=torch.float64)
        loss = rl_losses.clipped_surrogate(ratio, adv, clip_low=0.2, clip_high=0.2)
        clipped = max(0.8, min(1.2, ratio_value))
        expected = -min(ratio_value * advantage, clipped * advantage)
        assert loss.item() == pytest.approx(expected)
        assert loss.item() >= -ratio_value * advantage - 1e-12


def test_bounded_group_centering_and_zero_mask_match_integer_model() -> None:
    """Check exact small integer groups and masked losses against Lean laws."""
    torch = pytest.importorskip("torch")
    from stateset_agents.training import rl_losses

    for size in (1, 2, 3):
        for rewards in itertools.product((-2, -1, 0, 1, 2), repeat=size):
            tensor = torch.tensor(rewards, dtype=torch.float64)
            advantages = rl_losses.group_advantages(tensor, normalize=False)
            expected = [value - sum(rewards) / size for value in rewards]
            torch.testing.assert_close(
                advantages, torch.tensor(expected, dtype=torch.float32)
            )
            assert advantages.sum().item() == pytest.approx(0.0, abs=1e-6)

    scores = torch.tensor([[3.0, -4.0]])
    zero_mask = torch.zeros_like(scores)
    for mode in ("token", "seq"):
        assert rl_losses.masked_mean(scores, zero_mask, mode=mode).item() == 0.0
