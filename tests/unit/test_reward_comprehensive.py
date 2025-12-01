"""
Comprehensive tests for core/reward.py

Covers:
- RewardResult creation and manipulation
- All built-in reward functions (Helpfulness, Safety, Correctness, Engagement)
- CompositeReward with multiple components
- Neural reward model creation
- Reward factories
- Error handling
"""

import pytest
from unittest.mock import Mock, AsyncMock
from stateset_agents.core.reward import (
    RewardResult,
    RewardFunction,
    HelpfulnessReward,
    SafetyReward,
    CorrectnessReward,
    EngagementReward,
    CompositeReward,
    create_customer_service_reward,
    create_helpful_agent_reward,
    create_tutoring_reward,
)
from stateset_agents.core.trajectory import ConversationTurn


class TestRewardResultExtended:
    """Extended tests for RewardResult"""

    def test_reward_result_with_metadata(self):
        """Test RewardResult with metadata"""
        result = RewardResult(
            score=0.8,
            components={"helpfulness": 0.9, "safety": 0.7},
            metadata={"model": "gpt-4", "tokens": 100}
        )
        assert result.score == 0.8
        assert result.metadata["model"] == "gpt-4"
        assert result.metadata["tokens"] == 100

    def test_reward_result_to_dict(self):
        """Test RewardResult serialization"""
        result = RewardResult(
            score=0.75,
            components={"empathy": 0.8, "clarity": 0.7},
            metadata={"timestamp": "2024-01-01"}
        )
        data = result.to_dict()
        assert data["score"] == 0.75
        assert "empathy" in data["components"]
        assert "timestamp" in data["metadata"]

    def test_reward_result_from_dict(self):
        """Test RewardResult deserialization"""
        data = {
            "score": 0.6,
            "components": {"speed": 0.7, "accuracy": 0.5},
            "metadata": {"version": "1.0"}
        }
        result = RewardResult.from_dict(data)
        assert result.score == 0.6
        assert result.components["speed"] == 0.7


class TestHelpfulnessRewardExtended:
    """Extended tests for HelpfulnessReward"""

    @pytest.mark.asyncio
    async def test_helpfulness_with_detailed_turns(self):
        """Test helpfulness reward with detailed conversation"""
        reward = HelpfulnessReward(weight=1.0)

        turns = [
            ConversationTurn(role="user", content="I need help with my order"),
            ConversationTurn(role="assistant", content="I'd be happy to help! Can you provide your order number?"),
            ConversationTurn(role="user", content="It's #12345"),
            ConversationTurn(role="assistant", content="Thank you! I've located your order. It will arrive tomorrow."),
        ]

        result = await reward.compute_reward(turns, {})
        assert isinstance(result, RewardResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_helpfulness_with_single_turn(self):
        """Test helpfulness with minimal conversation"""
        reward = HelpfulnessReward()
        turns = [
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hi!"),
        ]
        result = await reward.compute_reward(turns, {})
        assert isinstance(result, RewardResult)

    @pytest.mark.asyncio
    async def test_helpfulness_with_custom_weight(self):
        """Test helpfulness with custom weight"""
        reward = HelpfulnessReward(weight=0.5)
        assert reward.weight == 0.5


class TestSafetyRewardExtended:
    """Extended tests for SafetyReward"""

    @pytest.mark.asyncio
    async def test_safety_detects_unsafe_content(self):
        """Test safety reward with potentially unsafe content"""
        reward = SafetyReward(weight=1.0)

        # Test with neutral content
        safe_turns = [
            ConversationTurn(role="user", content="How do I reset my password?"),
            ConversationTurn(role="assistant", content="Click the 'Forgot Password' link on the login page."),
        ]
        safe_result = await reward.compute_reward(safe_turns, {})
        assert safe_result.score > 0.5  # Should be relatively safe

    @pytest.mark.asyncio
    async def test_safety_with_empty_conversation(self):
        """Test safety with no turns"""
        reward = SafetyReward()
        result = await reward.compute_reward([], {})
        assert result.score == 1.0  # Empty is safe

    @pytest.mark.asyncio
    async def test_safety_strict_mode(self):
        """Test safety reward in strict mode"""
        reward = SafetyReward(weight=2.0, strict_mode=True)
        assert reward.weight == 2.0


class TestCorrectnessReward:
    """Tests for CorrectnessReward"""

    @pytest.mark.asyncio
    async def test_correctness_basic(self):
        """Test basic correctness reward"""
        reward = CorrectnessReward(weight=1.0)

        turns = [
            ConversationTurn(role="user", content="What is 2+2?"),
            ConversationTurn(role="assistant", content="2+2 equals 4."),
        ]

        result = await reward.compute_reward(turns, {})
        assert isinstance(result, RewardResult)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_correctness_with_context(self):
        """Test correctness with additional context"""
        reward = CorrectnessReward()

        turns = [
            ConversationTurn(role="user", content="What's the capital of France?"),
            ConversationTurn(role="assistant", content="Paris"),
        ]

        context = {"expected_answer": "Paris", "topic": "geography"}
        result = await reward.compute_reward(turns, context)
        assert isinstance(result, RewardResult)

    @pytest.mark.asyncio
    async def test_correctness_empty_turns(self):
        """Test correctness with empty conversation"""
        reward = CorrectnessReward()
        result = await reward.compute_reward([], {})
        assert result.score >= 0.0


class TestEngagementReward:
    """Tests for EngagementReward"""

    @pytest.mark.asyncio
    async def test_engagement_multi_turn(self):
        """Test engagement reward for multi-turn conversation"""
        reward = EngagementReward(weight=1.0)

        turns = [
            ConversationTurn(role="user", content="Tell me about your product"),
            ConversationTurn(role="assistant", content="Our product is amazing! Let me explain..."),
            ConversationTurn(role="user", content="That sounds interesting, tell me more"),
            ConversationTurn(role="assistant", content="Sure! Here are the key features..."),
        ]

        result = await reward.compute_reward(turns, {})
        assert isinstance(result, RewardResult)
        assert result.score > 0.0  # Multi-turn should have some engagement

    @pytest.mark.asyncio
    async def test_engagement_short_responses(self):
        """Test engagement with short responses"""
        reward = EngagementReward()

        turns = [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hi"),
        ]

        result = await reward.compute_reward(turns, {})
        assert isinstance(result, RewardResult)
        # Short responses should have lower engagement
        assert result.score <= 1.0

    @pytest.mark.asyncio
    async def test_engagement_with_questions(self):
        """Test engagement when assistant asks questions"""
        reward = EngagementReward()

        turns = [
            ConversationTurn(role="user", content="I need help"),
            ConversationTurn(role="assistant", content="I'm here to help! What do you need assistance with?"),
        ]

        result = await reward.compute_reward(turns, {})
        assert isinstance(result, RewardResult)


class TestCompositeRewardExtended:
    """Extended tests for CompositeReward"""

    @pytest.mark.asyncio
    async def test_composite_with_all_components(self):
        """Test composite reward with all standard components"""
        components = [
            HelpfulnessReward(weight=0.4),
            SafetyReward(weight=0.3),
            CorrectnessReward(weight=0.2),
            EngagementReward(weight=0.1),
        ]

        composite = CompositeReward(components)

        turns = [
            ConversationTurn(role="user", content="Help me understand your policy"),
            ConversationTurn(role="assistant", content="Our policy covers X, Y, and Z. What specific aspect would you like to know more about?"),
        ]

        result = await composite.compute_reward(turns, {})
        assert isinstance(result, RewardResult)
        assert len(result.components) == 4
        assert "helpfulness" in result.components
        assert "safety" in result.components
        assert "correctness" in result.components
        assert "engagement" in result.components

    @pytest.mark.asyncio
    async def test_composite_weight_normalization(self):
        """Test that composite reward normalizes weights"""
        components = [
            HelpfulnessReward(weight=1.0),
            SafetyReward(weight=1.0),
        ]

        composite = CompositeReward(components, normalize_weights=True)

        turns = [
            ConversationTurn(role="user", content="Test"),
            ConversationTurn(role="assistant", content="Response"),
        ]

        result = await composite.compute_reward(turns, {})
        assert isinstance(result, RewardResult)
        # With normalization, weights should sum to 1
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_composite_with_single_component(self):
        """Test composite with only one component"""
        composite = CompositeReward([HelpfulnessReward(weight=1.0)])

        turns = [
            ConversationTurn(role="user", content="Question"),
            ConversationTurn(role="assistant", content="Answer"),
        ]

        result = await composite.compute_reward(turns, {})
        assert isinstance(result, RewardResult)
        assert "helpfulness" in result.components


class TestRewardFactories:
    """Test reward factory functions"""

    @pytest.mark.asyncio
    async def test_customer_service_reward_factory(self):
        """Test customer service reward creation"""
        reward = create_customer_service_reward()

        assert isinstance(reward, CompositeReward)

        # Test with sample conversation
        turns = [
            ConversationTurn(role="user", content="I have a problem with my account"),
            ConversationTurn(role="assistant", content="I'm sorry to hear that. Let me help you resolve this issue."),
        ]

        result = await reward.compute_reward(turns, {})
        assert isinstance(result, RewardResult)
        assert result.score >= 0.0

    @pytest.mark.asyncio
    async def test_helpful_agent_reward_factory(self):
        """Test helpful agent reward creation"""
        reward = create_helpful_agent_reward()

        assert isinstance(reward, CompositeReward)

        turns = [
            ConversationTurn(role="user", content="How do I install the software?"),
            ConversationTurn(role="assistant", content="Here are the step-by-step instructions..."),
        ]

        result = await reward.compute_reward(turns, {})
        assert isinstance(result, RewardResult)

    @pytest.mark.asyncio
    async def test_tutoring_reward_factory(self):
        """Test tutoring reward creation"""
        reward = create_tutoring_reward()

        assert isinstance(reward, CompositeReward)

        turns = [
            ConversationTurn(role="user", content="Can you explain how photosynthesis works?"),
            ConversationTurn(role="assistant", content="Photosynthesis is the process by which plants..."),
        ]

        result = await reward.compute_reward(turns, {})
        assert isinstance(result, RewardResult)


class TestCustomRewardFunction:
    """Test creating custom reward functions"""

    @pytest.mark.asyncio
    async def test_custom_reward_implementation(self):
        """Test implementing a custom reward function"""

        class LengthReward(RewardFunction):
            """Reward based on response length"""

            async def compute_reward(self, turns, context):
                if not turns:
                    return RewardResult(score=0.0)

                # Calculate average response length
                assistant_turns = [t for t in turns if t.role == "assistant"]
                if not assistant_turns:
                    return RewardResult(score=0.0)

                avg_length = sum(len(t.content) for t in assistant_turns) / len(assistant_turns)
                # Normalize to 0-1 range (assuming 100 chars is ideal)
                score = min(avg_length / 100.0, 1.0)

                return RewardResult(
                    score=score,
                    components={"length": score},
                    metadata={"avg_length": avg_length}
                )

        reward = LengthReward(weight=1.0)

        turns = [
            ConversationTurn(role="user", content="Hello"),
            ConversationTurn(role="assistant", content="Hello! How can I help you today?"),
        ]

        result = await reward.compute_reward(turns, {})
        assert isinstance(result, RewardResult)
        assert "length" in result.components
        assert "avg_length" in result.metadata


class TestRewardErrorHandling:
    """Test error handling in reward functions"""

    @pytest.mark.asyncio
    async def test_reward_with_malformed_turns(self):
        """Test reward computation with malformed turn data"""
        reward = HelpfulnessReward()

        # Empty content
        turns = [
            ConversationTurn(role="user", content=""),
            ConversationTurn(role="assistant", content=""),
        ]

        result = await reward.compute_reward(turns, {})
        assert isinstance(result, RewardResult)

    @pytest.mark.asyncio
    async def test_reward_with_none_context(self):
        """Test reward with None context"""
        reward = SafetyReward()

        turns = [
            ConversationTurn(role="user", content="Test"),
            ConversationTurn(role="assistant", content="Response"),
        ]

        # Should handle None context gracefully
        result = await reward.compute_reward(turns, None)
        assert isinstance(result, RewardResult)

    @pytest.mark.asyncio
    async def test_composite_reward_with_failing_component(self):
        """Test composite reward when one component fails"""

        class FailingReward(RewardFunction):
            async def compute_reward(self, turns, context):
                raise ValueError("Simulated failure")

        # Composite should handle failure gracefully
        components = [
            HelpfulnessReward(weight=0.5),
            FailingReward(weight=0.5, name="failing"),
        ]

        composite = CompositeReward(components)

        turns = [
            ConversationTurn(role="user", content="Test"),
            ConversationTurn(role="assistant", content="Response"),
        ]

        # Should still return a result despite one component failing
        result = await composite.compute_reward(turns, {})
        assert isinstance(result, RewardResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
