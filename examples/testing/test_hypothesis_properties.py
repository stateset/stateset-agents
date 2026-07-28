"""Property-based tests using the bundled `stateset_agents.testing` strategies.

The framework ships hypothesis strategies for `ConversationTurn`, reward
values, and trajectory configs. Property tests pay off when the surface
area is large or adversarial — e.g., "no matter what the model produces,
the reward must stay in [0, 1]."

Run a focused fuzz:

    pytest test_hypothesis_properties.py --hypothesis-seed=42 -q
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.testing import conversation_turns, reward_values
from stateset_agents.testing.matchers import RewardMatcher


class ClampedSimilarityReward(RewardFunction):
    """Toy reward that should *always* be in [0, 1] regardless of input.

    The whole point of the property test below: prove the clamp holds.
    """

    name = "clamped_similarity"

    def __init__(self) -> None:
        super().__init__(weight=1.0, reward_type=RewardType.IMMEDIATE, name=self.name)

    async def compute_reward(self, turns, context=None):
        if not turns:
            return RewardResult(score=0.0)
        # An intentionally noisy calculation — but we clamp at the end.
        raw = len((turns[-1].content or "").split()) / 20.0
        clamped = max(0.0, min(1.0, raw))
        return RewardResult(score=clamped, breakdown={"raw": raw, "clamped": clamped})


@given(turn=conversation_turns())
@settings(max_examples=50, deadline=None)
def test_conversation_turns_strategy_produces_valid_shape(turn):
    """The bundled strategy always emits a {role, content, metadata?} dict."""
    assert turn["role"] in {"user", "assistant", "system"}
    assert isinstance(turn["content"], str)
    assert turn["content"].strip()  # the strategy filter rejects whitespace


@given(value=reward_values())
@settings(max_examples=100, deadline=None)
def test_reward_values_in_documented_range(value):
    """The `reward_values()` strategy must obey the published [-10, 10] envelope."""
    assert RewardMatcher.is_within_range(value, -10.0, 10.0)


@given(content=st.text(min_size=0, max_size=2000))
@settings(max_examples=100, deadline=None)
async def test_clamped_reward_always_in_unit_interval(content):
    """For any string the model could produce, the clamp must hold."""
    reward = ClampedSimilarityReward()
    turn = ConversationTurn(role="assistant", content=content)
    result = await reward.compute_reward([turn], context=None)
    assert (
        0.0 <= result.score <= 1.0
    ), f"Clamp violated: {result.score} for content len={len(content)}"
