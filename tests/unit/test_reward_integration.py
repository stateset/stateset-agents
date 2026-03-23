"""
Reward system integration tests using real reward functions (not mocks).

Tests that reward scores move in expected directions and that composite
rewards properly weight and combine component scores.
"""

import pytest

from stateset_agents.core.reward import (
    CompositeReward,
    HelpfulnessReward,
    RewardFunction,
    RewardResult,
    SafetyReward,
)
from stateset_agents.core.trajectory import ConversationTurn


def _make_turns(messages):
    """Convert dicts to ConversationTurn objects."""
    turns = []
    for msg in messages:
        turns.append(
            ConversationTurn(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
            )
        )
    return turns


class TestHelpfulnessRewardBehavior:
    """Test that helpfulness reward scores respond to content quality."""

    @pytest.mark.asyncio
    async def test_helpful_response_scores_higher(self):
        reward = HelpfulnessReward()

        helpful_turns = _make_turns(
            [
                {"role": "user", "content": "How do I fix a flat tire?"},
                {
                    "role": "assistant",
                    "content": (
                        "Here's how to fix a flat tire: First, pull over safely. "
                        "Then, loosen the lug nuts with a wrench. Jack up the car, "
                        "remove the flat tire, and put on the spare. Tighten the "
                        "lug nuts and lower the car. Check the tire pressure."
                    ),
                },
            ]
        )

        terse_turns = _make_turns(
            [
                {"role": "user", "content": "How do I fix a flat tire?"},
                {"role": "assistant", "content": "Fix it."},
            ]
        )

        helpful_result = await reward.compute_reward(helpful_turns)
        terse_result = await reward.compute_reward(terse_turns)

        assert helpful_result.score > terse_result.score

    @pytest.mark.asyncio
    async def test_empty_turns_returns_low_score(self):
        reward = HelpfulnessReward()
        result = await reward.compute_reward([])
        assert result.score >= 0.0
        assert result.score <= 0.5


class TestSafetyRewardBehavior:
    """Test that safety reward detects unsafe content."""

    @pytest.mark.asyncio
    async def test_safe_content_scores_high(self):
        reward = SafetyReward()
        safe_turns = _make_turns(
            [
                {"role": "user", "content": "Tell me about gardening"},
                {
                    "role": "assistant",
                    "content": "Gardening is a wonderful hobby! Start with easy plants like herbs.",
                },
            ]
        )
        result = await reward.compute_reward(safe_turns)
        assert result.score >= 0.5

    @pytest.mark.asyncio
    async def test_score_is_bounded(self):
        reward = SafetyReward()
        turns = _make_turns(
            [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ]
        )
        result = await reward.compute_reward(turns)
        assert 0.0 <= result.score <= 1.0


class TestCompositeRewardIntegration:
    """Test CompositeReward with real reward functions."""

    @pytest.mark.asyncio
    async def test_composite_with_real_rewards(self):
        """Composite of real rewards produces a valid combined score."""
        reward = CompositeReward(
            [
                HelpfulnessReward(weight=0.5),
                SafetyReward(weight=0.5),
            ]
        )

        turns = _make_turns(
            [
                {"role": "user", "content": "What is Python?"},
                {
                    "role": "assistant",
                    "content": "Python is a high-level programming language known for readability.",
                },
            ]
        )

        result = await reward.compute_reward(turns)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.breakdown, dict)
        assert len(result.breakdown) >= 2

    @pytest.mark.asyncio
    async def test_zero_weight_component_contributes_nothing(self):
        """A component with weight=0 should not affect the final score."""

        # Create a custom reward that always returns 0.0
        class ZeroReward(RewardFunction):
            async def compute_reward(self, turns, context=None):
                return RewardResult(score=0.0, breakdown={}, metadata={})

        reward = CompositeReward(
            [
                HelpfulnessReward(weight=1.0),
                ZeroReward(weight=0.0),
            ]
        )

        turns = _make_turns(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there! How can I help?"},
            ]
        )

        result_with_zero = await reward.compute_reward(turns)

        # Same score as helpfulness alone
        helpfulness_only = await HelpfulnessReward(weight=1.0).compute_reward(turns)
        assert abs(result_with_zero.score - helpfulness_only.score) < 0.15

    @pytest.mark.asyncio
    async def test_reward_result_serialization(self):
        """RewardResult should be convertible to dict."""
        result = RewardResult(
            score=0.75,
            breakdown={"helpfulness": 0.8, "safety": 0.7},
            metadata={"model": "test"},
        )
        assert result.score == 0.75
        assert result.breakdown["helpfulness"] == 0.8
        assert result.metadata["model"] == "test"

    @pytest.mark.asyncio
    async def test_composite_reward_empty_turns(self):
        """Composite reward handles empty turns gracefully."""
        reward = CompositeReward(
            [
                HelpfulnessReward(weight=0.5),
                SafetyReward(weight=0.5),
            ]
        )
        result = await reward.compute_reward([])
        assert 0.0 <= result.score <= 1.0
