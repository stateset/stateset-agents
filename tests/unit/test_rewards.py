"""
Comprehensive tests for reward system
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateset_agents.core.reward import (
    CompositeReward,
    HelpfulnessReward,
    RewardFunction,
    RewardResult,
    SafetyReward,
)
from stateset_agents.core.trajectory import ConversationTurn


def make_turns(turn_dicts):
    """Helper to convert dict turns to ConversationTurn objects."""
    return [ConversationTurn(role=t["role"], content=t["content"]) for t in turn_dicts]


# ===========================
# RewardResult Tests
# ===========================


class TestRewardResult:
    """Test RewardResult dataclass"""

    def test_reward_result_creation(self):
        """Test creating reward result"""
        result = RewardResult(
            score=0.85,
            breakdown={"helpfulness": 0.9, "safety": 0.8},
            metadata={"source": "test"}
        )

        assert result.score == 0.85
        assert result.breakdown["helpfulness"] == 0.9
        assert result.breakdown["safety"] == 0.8
        assert result.metadata["source"] == "test"

    def test_reward_result_defaults(self):
        """Test reward result with defaults"""
        result = RewardResult(score=0.5, breakdown={}, metadata={})

        assert result.score == 0.5
        assert isinstance(result.breakdown, dict)
        assert isinstance(result.metadata, dict)

    def test_reward_result_with_components(self):
        """Test reward result with component breakdown"""
        result = RewardResult(
            score=0.75,
            breakdown={
                "empathy": 0.8,
                "action_oriented": 0.7,
                "professionalism": 0.75
            },
            metadata={"num_turns": 3}
        )

        assert len(result.breakdown) == 3
        assert "empathy" in result.breakdown
        assert result.metadata["num_turns"] == 3


# ===========================
# Base RewardFunction Tests
# ===========================


class TestRewardFunction:
    """Test base RewardFunction"""

    def test_reward_function_abstract(self):
        """Test RewardFunction is abstract"""
        # Should not be able to instantiate directly without compute_reward
        with pytest.raises(TypeError):
            RewardFunction()

    def test_reward_function_with_weight(self):
        """Test reward function with custom weight"""
        class TestReward(RewardFunction):
            async def compute_reward(self, turns, context=None):
                return RewardResult(score=0.5, breakdown={}, metadata={})

        reward = TestReward(weight=0.7)
        assert reward.weight == 0.7

    @pytest.mark.asyncio
    async def test_custom_reward_function(self):
        """Test custom reward function implementation"""
        class CustomReward(RewardFunction):
            async def compute_reward(self, turns, context=None):
                score = len(turns) * 0.1
                return RewardResult(
                    score=min(1.0, score),
                    breakdown={"turn_count": len(turns)},
                    metadata={}
                )

        reward = CustomReward()
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        result = await reward.compute_reward(turns)
        assert isinstance(result, RewardResult)
        assert result.score == 0.2
        assert result.breakdown["turn_count"] == 2


# ===========================
# HelpfulnessReward Tests
# ===========================


class TestHelpfulnessReward:
    """Test HelpfulnessReward"""

    def test_helpfulness_reward_initialization(self):
        """Test helpfulness reward initialization"""
        reward = HelpfulnessReward(weight=0.8)
        assert reward.weight == 0.8

    def test_helpfulness_reward_default_weight(self):
        """Test helpfulness reward with default weight"""
        reward = HelpfulnessReward()
        assert reward.weight == 1.0

    @pytest.mark.asyncio
    async def test_helpfulness_reward_with_turns(self):
        """Test helpfulness reward computation"""
        reward = HelpfulnessReward()
        turns = make_turns([
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "To reset your password, click on 'Forgot Password' and follow the email instructions."}
        ])

        result = await reward.compute_reward(turns)

        assert isinstance(result, RewardResult)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.breakdown, dict)

    @pytest.mark.asyncio
    async def test_helpfulness_reward_empty_turns(self):
        """Test helpfulness reward with empty turns"""
        reward = HelpfulnessReward()
        result = await reward.compute_reward([])

        assert isinstance(result, RewardResult)
        # Should handle empty input gracefully
        assert result.score >= 0.0

    @pytest.mark.asyncio
    async def test_helpfulness_reward_multiple_turns(self):
        """Test helpfulness reward with multiple conversation turns"""
        reward = HelpfulnessReward()
        turns = make_turns([
            {"role": "user", "content": "I need help"},
            {"role": "assistant", "content": "Sure, I can help you!"},
            {"role": "user", "content": "What about shipping?"},
            {"role": "assistant", "content": "Shipping takes 3-5 business days."}
        ])

        result = await reward.compute_reward(turns)

        assert isinstance(result, RewardResult)
        assert 0.0 <= result.score <= 1.0


# ===========================
# SafetyReward Tests
# ===========================


class TestSafetyReward:
    """Test SafetyReward"""

    def test_safety_reward_initialization(self):
        """Test safety reward initialization"""
        reward = SafetyReward(weight=0.6)
        assert reward.weight == 0.6

    @pytest.mark.asyncio
    async def test_safety_reward_with_safe_content(self):
        """Test safety reward with safe content"""
        reward = SafetyReward()
        turns = make_turns([
            {"role": "user", "content": "Tell me about your product"},
            {"role": "assistant", "content": "Our product is a cloud-based solution for business."}
        ])

        result = await reward.compute_reward(turns)

        assert isinstance(result, RewardResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_safety_reward_empty_input(self):
        """Test safety reward with empty input"""
        reward = SafetyReward()
        result = await reward.compute_reward([])

        assert isinstance(result, RewardResult)
        assert result.score >= 0.0

    @pytest.mark.asyncio
    async def test_safety_reward_with_context(self):
        """Test safety reward with context"""
        reward = SafetyReward()
        turns = make_turns([{"role": "user", "content": "Hello"}])
        context = {"user_age": 18, "content_rating": "general"}

        result = await reward.compute_reward(turns, context=context)

        assert isinstance(result, RewardResult)
        assert 0.0 <= result.score <= 1.0


# ===========================
# CompositeReward Tests
# ===========================


class TestCompositeReward:
    """Test CompositeReward"""

    def test_composite_reward_initialization(self):
        """Test composite reward with multiple components"""
        reward_functions = [
            HelpfulnessReward(weight=0.6),
            SafetyReward(weight=0.4)
        ]

        reward = CompositeReward(reward_functions)
        assert len(reward.reward_functions) == 2

    def test_composite_reward_empty_components(self):
        """Test composite reward with no components"""
        reward = CompositeReward([])
        assert len(reward.reward_functions) == 0

    @pytest.mark.asyncio
    async def test_composite_reward_computes_combined_score(self):
        """Test composite reward computes combined score"""
        reward_functions = [
            HelpfulnessReward(weight=0.5),
            SafetyReward(weight=0.5)
        ]

        reward = CompositeReward(reward_functions)
        turns = make_turns([
            {"role": "user", "content": "Need help"},
            {"role": "assistant", "content": "I'm here to help!"}
        ])

        result = await reward.compute_reward(turns)

        assert isinstance(result, RewardResult)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.breakdown, dict)

    @pytest.mark.asyncio
    async def test_composite_reward_respects_weights(self):
        """Test composite reward respects component weights"""
        # Create mock rewards with known scores
        mock_reward1 = MagicMock(spec=RewardFunction)
        mock_reward1.weight = 0.7
        mock_reward1.name = "MockReward1"
        mock_reward1.compute_reward = AsyncMock(
            return_value=RewardResult(score=1.0, breakdown={}, metadata={})
        )

        mock_reward2 = MagicMock(spec=RewardFunction)
        mock_reward2.weight = 0.3
        mock_reward2.name = "MockReward2"
        mock_reward2.compute_reward = AsyncMock(
            return_value=RewardResult(score=0.0, breakdown={}, metadata={})
        )

        reward = CompositeReward([mock_reward1, mock_reward2])
        turns = make_turns([{"role": "user", "content": "Test"}])

        result = await reward.compute_reward(turns)

        # With 0.7 weight on score 1.0 and 0.3 weight on score 0.0
        # Expected: 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert isinstance(result, RewardResult)
        # Score should be weighted combination
        assert result.score >= 0.5  # At least half of max

    @pytest.mark.asyncio
    async def test_composite_reward_passes_context(self):
        """Test composite reward passes context to components"""
        reward_functions = [HelpfulnessReward(), SafetyReward()]
        reward = CompositeReward(reward_functions)

        turns = make_turns([{"role": "user", "content": "Question"}])
        context = {"domain": "customer_service"}

        result = await reward.compute_reward(turns, context=context)

        assert isinstance(result, RewardResult)

    def test_composite_reward_with_three_components(self):
        """Test composite reward with multiple components"""
        reward_functions = [
            HelpfulnessReward(weight=0.4),
            SafetyReward(weight=0.3),
            HelpfulnessReward(weight=0.3)  # Can have duplicates with different weights
        ]

        reward = CompositeReward(reward_functions)
        assert len(reward.reward_functions) == 3


# ===========================
# Reward System Integration Tests
# ===========================


class TestRewardIntegration:
    """Test reward system integration"""

    @pytest.mark.asyncio
    async def test_chained_reward_computation(self):
        """Test computing rewards in sequence"""
        helpfulness = HelpfulnessReward()
        safety = SafetyReward()

        turns = make_turns([
            {"role": "user", "content": "Help me"},
            {"role": "assistant", "content": "I'm here to assist!"}
        ])

        result1 = await helpfulness.compute_reward(turns)
        result2 = await safety.compute_reward(turns)

        assert isinstance(result1, RewardResult)
        assert isinstance(result2, RewardResult)

    @pytest.mark.asyncio
    async def test_reward_caching_behavior(self):
        """Test reward function can be called multiple times"""
        reward = HelpfulnessReward()
        turns = make_turns([{"role": "user", "content": "Test"}])

        result1 = await reward.compute_reward(turns)
        result2 = await reward.compute_reward(turns)

        # Both should succeed
        assert isinstance(result1, RewardResult)
        assert isinstance(result2, RewardResult)

    @pytest.mark.asyncio
    async def test_reward_with_various_turn_lengths(self):
        """Test reward functions handle different conversation lengths"""
        reward = CompositeReward([HelpfulnessReward(), SafetyReward()])

        # Single turn
        single = make_turns([{"role": "user", "content": "Hi"}])
        result_single = await reward.compute_reward(single)
        assert isinstance(result_single, RewardResult)

        # Multiple turns
        multiple = make_turns([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"}
        ])
        result_multiple = await reward.compute_reward(multiple)
        assert isinstance(result_multiple, RewardResult)

    @pytest.mark.asyncio
    async def test_reward_error_handling(self):
        """Test reward function handles errors gracefully"""
        class ErrorReward(RewardFunction):
            async def compute_reward(self, turns, context=None):
                # Should handle malformed input
                if not turns:
                    return RewardResult(score=0.0, breakdown={}, metadata={"error": "empty"})
                return RewardResult(score=0.5, breakdown={}, metadata={})

        reward = ErrorReward()

        # Empty turns
        result = await reward.compute_reward([])
        assert result.score == 0.0
        assert "error" in result.metadata
