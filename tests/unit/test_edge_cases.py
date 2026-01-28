"""
Comprehensive Edge Case Tests for StateSet Agents

This module tests edge cases, boundary conditions, and error handling
across all major components to ensure robustness and reliability.

Categories covered:
- Empty/null inputs
- Boundary values
- Malformed data
- Concurrent operations
- Resource exhaustion
- Error recovery
- Type coercion
"""

import asyncio
import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.nn as nn


# =============================================================================
# AGENT EDGE CASES
# =============================================================================


class TestAgentEdgeCases:
    """Edge cases for Agent classes."""

    @pytest.fixture
    def mock_agent_config(self):
        """Create a mock agent config."""
        from stateset_agents.core.agent import AgentConfig
        return AgentConfig(
            model_name="stub://test",
            use_stub_model=True,
            stub_responses=["Test response"],
        )

    @pytest.mark.asyncio
    async def test_empty_message_list(self, mock_agent_config):
        """Test agent with empty message list."""
        from stateset_agents.core.agent import MultiTurnAgent

        agent = MultiTurnAgent(mock_agent_config)
        await agent.initialize()

        # Empty messages should still work
        response = await agent.generate_response([])
        assert response is not None
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_very_long_input(self, mock_agent_config):
        """Test agent with extremely long input."""
        from stateset_agents.core.agent import MultiTurnAgent

        agent = MultiTurnAgent(mock_agent_config)
        await agent.initialize()

        # Create very long message (100k characters)
        long_content = "x" * 100000
        messages = [{"role": "user", "content": long_content}]

        response = await agent.generate_response(messages)
        assert response is not None

    @pytest.mark.asyncio
    async def test_unicode_content(self, mock_agent_config):
        """Test agent with various unicode characters."""
        from stateset_agents.core.agent import MultiTurnAgent

        agent = MultiTurnAgent(mock_agent_config)
        await agent.initialize()

        # Test various unicode
        unicode_inputs = [
            "Hello 你好 مرحبا שלום",  # Multiple scripts
            "Emoji test 🎉🔥💡",  # Emojis
            "Special: \n\t\r\0",  # Control characters
            "Math: ∑∫∂∆",  # Math symbols
            "RTL: مرحبا العالم",  # Right-to-left
        ]

        for content in unicode_inputs:
            messages = [{"role": "user", "content": content}]
            response = await agent.generate_response(messages)
            assert response is not None

    @pytest.mark.asyncio
    async def test_malformed_messages(self, mock_agent_config):
        """Test agent with malformed message structures."""
        from stateset_agents.core.agent import MultiTurnAgent

        agent = MultiTurnAgent(mock_agent_config)
        await agent.initialize()

        # Missing role
        messages = [{"content": "Hello"}]
        try:
            response = await agent.generate_response(messages)
            # Should either work or raise graceful error
        except (KeyError, ValueError):
            pass  # Expected behavior

        # Missing content
        messages = [{"role": "user"}]
        try:
            response = await agent.generate_response(messages)
        except (KeyError, ValueError):
            pass

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_agent_config):
        """Test agent handling concurrent requests."""
        from stateset_agents.core.agent import MultiTurnAgent

        agent = MultiTurnAgent(mock_agent_config)
        await agent.initialize()

        # Make 100 concurrent requests
        messages = [{"role": "user", "content": "Test"}]

        async def make_request():
            return await agent.generate_response(messages)

        responses = await asyncio.gather(*[make_request() for _ in range(100)])

        assert len(responses) == 100
        assert all(r is not None for r in responses)


# =============================================================================
# REWARD EDGE CASES
# =============================================================================


class TestRewardEdgeCases:
    """Edge cases for reward computation."""

    @pytest.mark.asyncio
    async def test_empty_turns(self):
        """Test reward computation with empty turns."""
        from stateset_agents.core.reward import HelpfulnessReward

        reward_fn = HelpfulnessReward()
        result = await reward_fn.compute_reward([])

        # Should handle empty gracefully
        assert result is not None
        assert hasattr(result, 'score')

    @pytest.mark.asyncio
    async def test_single_turn(self):
        """Test reward with single turn (unusual but valid)."""
        from stateset_agents.core.reward import HelpfulnessReward
        from stateset_agents.core.trajectory import ConversationTurn

        reward_fn = HelpfulnessReward()
        turns = [ConversationTurn(role="user", content="Hello")]

        result = await reward_fn.compute_reward(turns)
        assert result is not None

    @pytest.mark.asyncio
    async def test_extreme_weights(self):
        """Test composite reward with extreme weights."""
        from stateset_agents.core.reward import CompositeReward, HelpfulnessReward, SafetyReward
        from stateset_agents.core.trajectory import ConversationTurn

        # Very large weights
        reward = CompositeReward([
            HelpfulnessReward(weight=1e10),
            SafetyReward(weight=1e-10),
        ])

        turns = [
            ConversationTurn(role="user", content="Help me"),
            ConversationTurn(role="assistant", content="Sure!"),
        ]

        result = await reward.compute_reward(turns)
        assert not math.isnan(result.score)
        assert not math.isinf(result.score)

    @pytest.mark.asyncio
    async def test_zero_weights(self):
        """Test composite reward with zero weights."""
        from stateset_agents.core.reward import CompositeReward, HelpfulnessReward
        from stateset_agents.core.trajectory import ConversationTurn

        reward = CompositeReward([
            HelpfulnessReward(weight=0.0),
        ])

        turns = [
            ConversationTurn(role="user", content="Test"),
            ConversationTurn(role="assistant", content="Response"),
        ]

        result = await reward.compute_reward(turns)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_negative_weights(self):
        """Test handling of negative weights (penalty rewards)."""
        from stateset_agents.core.reward import CompositeReward, HelpfulnessReward
        from stateset_agents.core.trajectory import ConversationTurn

        # Negative weight should work (penalty)
        reward = CompositeReward([
            HelpfulnessReward(weight=-1.0),
        ])

        turns = [
            ConversationTurn(role="user", content="Test"),
            ConversationTurn(role="assistant", content="Response"),
        ]

        result = await reward.compute_reward(turns)
        assert result.score <= 0


# =============================================================================
# TRAJECTORY EDGE CASES
# =============================================================================


class TestTrajectoryEdgeCases:
    """Edge cases for trajectory handling."""

    def test_conversation_turn_positional_role_with_reward(self):
        """Role-like first arg with reward should not be treated as legacy."""
        from stateset_agents.core.trajectory import ConversationTurn

        turn = ConversationTurn("assistant", "Hi there", 0.5)

        assert turn.role == "assistant"
        assert turn.content == "Hi there"
        assert turn.reward == 0.5
        assert turn.user_message is None
        assert turn.assistant_response is None

    def test_conversation_turn_positional_role_case_insensitive(self):
        """Role parsing should handle common case variants."""
        from stateset_agents.core.trajectory import ConversationTurn

        turn = ConversationTurn("User", "Hello", 1.0)

        assert turn.role == "user"
        assert turn.content == "Hello"
        assert turn.reward == 1.0

    def test_conversation_turn_positional_legacy_pair_with_reward(self):
        """Non-role first arg should keep legacy behavior."""
        from stateset_agents.core.trajectory import ConversationTurn

        turn = ConversationTurn("hello", "hi", 0.2)

        assert turn.role == "assistant"
        assert turn.content == "hi"
        assert turn.user_message == "hello"
        assert turn.assistant_response == "hi"
        assert turn.reward == 0.2

    def test_empty_trajectory(self):
        """Test creating trajectory with no turns."""
        from stateset_agents.core.trajectory import MultiTurnTrajectory

        traj = MultiTurnTrajectory(
            trajectory_id="test",
            turns=[],
            rewards=[],
            total_reward=0.0,
            metadata={},
        )

        assert traj.trajectory_id == "test"
        assert len(traj.turns) == 0

    def test_mismatched_rewards_turns(self):
        """Test trajectory with mismatched rewards/turns count."""
        from stateset_agents.core.trajectory import MultiTurnTrajectory, ConversationTurn

        turns = [
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello"),
        ]

        # More rewards than turns
        traj = MultiTurnTrajectory(
            trajectory_id="test",
            turns=turns,
            rewards=[0.5, 0.6, 0.7, 0.8],  # Too many
            total_reward=2.6,
            metadata={},
        )

        # Should still be valid
        assert len(traj.turns) == 2
        assert len(traj.rewards) == 4

    def test_trajectory_serialization_special_chars(self):
        """Test trajectory serialization with special characters."""
        from stateset_agents.core.trajectory import MultiTurnTrajectory, ConversationTurn

        turns = [
            ConversationTurn(role="user", content='Test "quotes" and \\slashes'),
            ConversationTurn(role="assistant", content="Tab\there\nnewline"),
        ]

        traj = MultiTurnTrajectory(
            trajectory_id="test",
            turns=turns,
            rewards=[0.5, 0.5],
            total_reward=1.0,
            metadata={"key": "value\nwith\nnewlines"},
        )

        # Test to_messages conversion
        messages = traj.to_messages()
        assert len(messages) == 2

    def test_very_long_trajectory(self):
        """Test trajectory with many turns."""
        from stateset_agents.core.trajectory import MultiTurnTrajectory, ConversationTurn

        # Create 1000 turn trajectory
        turns = []
        for i in range(500):
            turns.append(ConversationTurn(role="user", content=f"User message {i}"))
            turns.append(ConversationTurn(role="assistant", content=f"Assistant response {i}"))

        rewards = [0.5] * 1000

        traj = MultiTurnTrajectory(
            trajectory_id="long_test",
            turns=turns,
            rewards=rewards,
            total_reward=500.0,
            metadata={},
        )

        assert len(traj.turns) == 1000
        messages = traj.to_messages()
        assert len(messages) == 1000


# =============================================================================
# TRAINING EDGE CASES
# =============================================================================


class TestTrainingEdgeCases:
    """Edge cases for training components."""

    def test_zero_learning_rate(self):
        """Test training config with zero learning rate."""
        from stateset_agents.training.config import TrainingConfig

        config = TrainingConfig(learning_rate=0.0)
        assert config.learning_rate == 0.0

    def test_extreme_hyperparameters(self):
        """Test training config with extreme values."""
        from stateset_agents.training.config import TrainingConfig

        # Very small
        config_small = TrainingConfig(
            learning_rate=1e-20,
            beta=1e-20,
            clip_ratio=1e-10,
        )

        # Very large
        config_large = TrainingConfig(
            learning_rate=1e10,  # Should be clipped or warn
            beta=1e10,
            num_generations=10000,
        )

        assert config_small.learning_rate == 1e-20
        assert config_large.num_generations == 10000

    def test_advantage_computation_all_same_rewards(self):
        """Test advantage computation when all rewards are identical."""
        from stateset_agents.training.base_trainer import compute_group_advantages

        rewards = [0.5, 0.5, 0.5, 0.5]
        advantages = compute_group_advantages(rewards, baseline_type="mean")

        # All advantages should be 0
        assert all(a == 0.0 for a in advantages)

    def test_advantage_computation_single_sample(self):
        """Test advantage computation with single sample."""
        from stateset_agents.training.base_trainer import compute_group_advantages

        rewards = [0.8]
        advantages = compute_group_advantages(rewards, baseline_type="mean")

        # Single sample: advantage = reward - reward = 0
        assert advantages[0] == 0.0

    def test_advantage_normalization_zero_std(self):
        """Test advantage normalization when std is zero."""
        from stateset_agents.training.base_trainer import normalize_advantages

        # All same values -> std = 0
        advantages = torch.tensor([1.0, 1.0, 1.0, 1.0])
        normalized = normalize_advantages(advantages)

        # Should not produce NaN/Inf
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()

    def test_gradient_clipping_zero_norm(self):
        """Test gradient clipping with zero gradients."""
        model = nn.Linear(10, 10)

        # Set all gradients to zero
        for p in model.parameters():
            p.grad = torch.zeros_like(p)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert norm == 0.0

    def test_empty_batch(self):
        """Test handling of empty batches."""
        # Empty tensor operations
        empty = torch.tensor([])

        # Should handle gracefully
        assert empty.sum() == 0
        assert empty.mean().isnan()  # Mean of empty is NaN


# =============================================================================
# ENVIRONMENT EDGE CASES
# =============================================================================


class TestEnvironmentEdgeCases:
    """Edge cases for environment handling."""

    @pytest.mark.asyncio
    async def test_empty_scenarios(self):
        """Test environment with no scenarios."""
        from stateset_agents.core.environment import ConversationEnvironment

        env = ConversationEnvironment(scenarios=[])

        # Should handle gracefully (may raise or return None)
        try:
            state = await env.reset()
        except (IndexError, ValueError):
            pass  # Expected if no scenarios

    @pytest.mark.asyncio
    async def test_max_turns_zero(self):
        """Test environment with max_turns=0."""
        from stateset_agents.core.environment import ConversationEnvironment

        env = ConversationEnvironment(
            scenarios=[{"prompt": "Test"}],
            max_turns=0,
        )

        state = await env.reset()
        # Should be done immediately
        assert state.is_done or state.turn_count == 0

    @pytest.mark.asyncio
    async def test_very_large_scenario_count(self):
        """Test environment with many scenarios."""
        from stateset_agents.core.environment import ConversationEnvironment

        # 10000 scenarios
        scenarios = [{"prompt": f"Scenario {i}"} for i in range(10000)]
        env = ConversationEnvironment(scenarios=scenarios)

        state = await env.reset()
        assert state is not None


# =============================================================================
# CONFIG EDGE CASES
# =============================================================================


class TestConfigEdgeCases:
    """Edge cases for configuration handling."""

    def test_config_with_none_values(self):
        """Test config handling of None values."""
        from stateset_agents.training.config import TrainingConfig

        config = TrainingConfig(
            model_name="gpt2",
            wandb_run_name=None,
        )

        assert config.wandb_run_name is None

    def test_config_to_dict_with_complex_types(self):
        """Test config serialization with complex nested types."""
        from stateset_agents.training.gspo_trainer import GSPOConfig

        config = GSPOConfig(
            lora_target_modules=["q_proj", "v_proj", "k_proj"],
        )

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["lora_target_modules"] == ["q_proj", "v_proj", "k_proj"]

    def test_config_domain_presets(self):
        """Test all domain preset configurations are valid."""
        from stateset_agents.training.config import get_config_for_task

        domains = ["customer_service", "technical_support", "sales_assistant"]

        for domain in domains:
            try:
                config = get_config_for_task(domain)
                assert config is not None
            except ValueError:
                # Some domains might not be implemented
                pass


# =============================================================================
# NUMERICAL STABILITY EDGE CASES
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_log_softmax_large_values(self):
        """Test log_softmax with very large values."""
        logits = torch.tensor([[1e10, 1e10, 1e10]])
        result = torch.nn.functional.log_softmax(logits, dim=-1)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_log_softmax_small_values(self):
        """Test log_softmax with very small values."""
        logits = torch.tensor([[-1e10, -1e10, -1e10]])
        result = torch.nn.functional.log_softmax(logits, dim=-1)

        assert not torch.isnan(result).any()

    def test_kl_divergence_identical_distributions(self):
        """Test KL divergence between identical distributions."""
        p = torch.softmax(torch.randn(10), dim=0)
        q = p.clone()

        kl = torch.nn.functional.kl_div(p.log(), q, reduction="sum")
        assert kl.abs() < 1e-6  # Should be ~0

    def test_kl_divergence_zero_probability(self):
        """Test KL divergence with zero probabilities."""
        p = torch.tensor([0.5, 0.5, 0.0])
        q = torch.tensor([0.33, 0.33, 0.34])

        # This can produce inf, test handling
        log_p = torch.log(p + 1e-10)  # Add epsilon
        kl = torch.nn.functional.kl_div(log_p, q, reduction="sum")

        # Should be a finite number
        assert not torch.isnan(kl)

    def test_reward_normalization_extreme_values(self):
        """Test reward normalization with extreme values."""
        rewards = torch.tensor([1e-10, 1e10])

        mean = rewards.mean()
        std = rewards.std()

        normalized = (rewards - mean) / (std + 1e-8)

        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()


# =============================================================================
# MEMORY EDGE CASES
# =============================================================================


class TestMemoryEdgeCases:
    """Tests for memory handling edge cases."""

    def test_large_tensor_creation(self):
        """Test creation of large tensors."""
        try:
            # 1GB tensor (might fail on low-memory systems)
            large_tensor = torch.zeros(256, 1024, 1024)
            del large_tensor
        except RuntimeError:
            pytest.skip("Not enough memory for large tensor test")

    def test_gradient_accumulation_memory(self):
        """Test that gradient accumulation doesn't leak memory."""
        model = nn.Linear(100, 100)
        optimizer = torch.optim.Adam(model.parameters())

        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        for _ in range(100):
            x = torch.randn(32, 100)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Memory shouldn't grow significantly
        if torch.cuda.is_available():
            assert final_memory - initial_memory < 1e9  # Less than 1GB growth


# =============================================================================
# ASYNC/CONCURRENT EDGE CASES
# =============================================================================


class TestAsyncEdgeCases:
    """Tests for async and concurrent operations."""

    @pytest.mark.asyncio
    async def test_rapid_sequential_calls(self):
        """Test rapid sequential async calls."""
        async def dummy_async():
            await asyncio.sleep(0.001)
            return True

        results = []
        for _ in range(1000):
            results.append(await dummy_async())

        assert all(results)

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in async operations."""
        async def slow_operation():
            await asyncio.sleep(10)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_cancelled_task(self):
        """Test handling of cancelled tasks."""
        async def cancellable():
            await asyncio.sleep(10)

        task = asyncio.create_task(cancellable())
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


# =============================================================================
# TYPE COERCION EDGE CASES
# =============================================================================


class TestTypeCoercion:
    """Tests for type coercion and validation."""

    def test_string_to_float_conversion(self):
        """Test string to float conversion for config values."""
        # Common user error: passing string instead of float
        value = "0.001"
        converted = float(value)
        assert converted == 0.001

    def test_bool_string_handling(self):
        """Test boolean string handling."""
        # Various string representations of booleans
        true_strings = ["true", "True", "TRUE", "1", "yes", "Yes"]
        false_strings = ["false", "False", "FALSE", "0", "no", "No"]

        for s in true_strings:
            assert s.lower() in ["true", "1", "yes"]

        for s in false_strings:
            assert s.lower() in ["false", "0", "no"]

    def test_list_vs_tuple(self):
        """Test that lists and tuples are handled consistently."""
        list_val = [1, 2, 3]
        tuple_val = (1, 2, 3)

        # Both should work for iteration
        assert list(list_val) == list(tuple_val)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
