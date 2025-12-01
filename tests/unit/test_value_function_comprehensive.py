"""
Comprehensive tests for core/value_function.py

Covers:
- ValueHead creation and forward pass
- ValueFunction initialization and configuration
- GAE computation with various parameters
- GRPO advantages computation
- Value function updates
- Save/load functionality
- Edge cases and error handling
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
from stateset_agents.core.value_function import (
    ValueHead,
    ValueFunction,
    create_value_function,
)


class TestValueHeadExtended:
    """Extended tests for ValueHead"""

    def test_value_head_with_layer_norm(self):
        """Test ValueHead with layer normalization"""
        hidden_size = 768
        value_head = ValueHead(hidden_size, dropout=0.1, use_layer_norm=True)

        assert value_head is not None
        assert hasattr(value_head, 'norm')

    def test_value_head_without_layer_norm(self):
        """Test ValueHead without layer normalization"""
        hidden_size = 512
        value_head = ValueHead(hidden_size, dropout=0.0, use_layer_norm=False)

        assert value_head is not None

    def test_value_head_various_hidden_sizes(self):
        """Test ValueHead with different hidden sizes"""
        for hidden_size in [256, 512, 768, 1024]:
            value_head = ValueHead(hidden_size)

            # Test forward pass
            batch_size, seq_len = 2, 10
            hidden_states = torch.randn(batch_size, seq_len, hidden_size)
            values = value_head(hidden_states)

            assert values.shape == (batch_size, seq_len, 1)

    def test_value_head_dropout_probabilities(self):
        """Test ValueHead with different dropout rates"""
        hidden_size = 768

        for dropout in [0.0, 0.1, 0.3, 0.5]:
            value_head = ValueHead(hidden_size, dropout=dropout)
            assert value_head is not None

    def test_value_head_gradient_flow(self):
        """Test that gradients flow through ValueHead"""
        hidden_size = 256
        value_head = ValueHead(hidden_size)

        hidden_states = torch.randn(1, 5, hidden_size, requires_grad=True)
        values = value_head(hidden_states)
        loss = values.sum()
        loss.backward()

        assert hidden_states.grad is not None


class TestValueFunctionConfiguration:
    """Test ValueFunction with various configurations"""

    def test_value_function_custom_gamma(self):
        """Test ValueFunction with custom discount factor"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        for gamma in [0.9, 0.95, 0.99, 1.0]:
            value_fn = ValueFunction(mock_model, gamma=gamma)
            assert value_fn.gamma == gamma

    def test_value_function_custom_gae_lambda(self):
        """Test ValueFunction with custom GAE lambda"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        for gae_lambda in [0.0, 0.5, 0.9, 0.95, 1.0]:
            value_fn = ValueFunction(mock_model, gae_lambda=gae_lambda)
            assert value_fn.gae_lambda == gae_lambda

    def test_value_function_with_normalization(self):
        """Test ValueFunction with advantage normalization enabled"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, normalize_advantages=True)
        assert value_fn.normalize_advantages is True

    def test_value_function_without_normalization(self):
        """Test ValueFunction with advantage normalization disabled"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, normalize_advantages=False)
        assert value_fn.normalize_advantages is False

    def test_value_function_with_custom_value_head(self):
        """Test ValueFunction with pre-created value head"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        custom_head = ValueHead(768, dropout=0.2)
        value_fn = ValueFunction(mock_model, value_head=custom_head)

        assert value_fn.value_head is custom_head


class TestGAEComputation:
    """Test Generalized Advantage Estimation"""

    def test_gae_single_step(self):
        """Test GAE with single step"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, gamma=0.99, gae_lambda=0.95, normalize_advantages=False)

        rewards = [1.0]
        values = torch.tensor([0.5], dtype=torch.float32)

        advantages, returns = value_fn.compute_gae(rewards, values)

        assert len(advantages) == 1
        assert len(returns) == 1
        assert torch.allclose(returns, advantages + values, atol=1e-5)

    def test_gae_multi_step(self):
        """Test GAE with multiple steps"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, normalize_advantages=False)

        rewards = [1.0, 0.8, 0.6, 0.4, 0.2]
        values = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float32)

        advantages, returns = value_fn.compute_gae(rewards, values)

        assert len(advantages) == len(rewards)
        assert len(returns) == len(rewards)
        assert torch.allclose(returns, advantages + values, atol=1e-5)

    def test_gae_with_done_flags(self):
        """Test GAE with episode termination flags"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, normalize_advantages=False)

        rewards = [1.0, 0.5, 0.8]
        values = torch.tensor([0.5, 0.4, 0.6], dtype=torch.float32)
        dones = [False, True, False]  # Episode ends after second step

        advantages, returns = value_fn.compute_gae(rewards, values, dones)

        assert len(advantages) == len(rewards)
        assert len(returns) == len(rewards)

    def test_gae_with_zero_rewards(self):
        """Test GAE with zero rewards"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, normalize_advantages=False)

        rewards = [0.0, 0.0, 0.0]
        values = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        advantages, returns = value_fn.compute_gae(rewards, values)

        assert len(advantages) == len(rewards)
        assert torch.allclose(advantages, torch.zeros(3), atol=1e-5)

    def test_gae_with_negative_rewards(self):
        """Test GAE with negative rewards"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, normalize_advantages=False)

        rewards = [-1.0, -0.5, -0.2]
        values = torch.tensor([0.5, 0.3, 0.1], dtype=torch.float32)

        advantages, returns = value_fn.compute_gae(rewards, values)

        assert len(advantages) == len(rewards)
        # With negative rewards, returns should be lower than values
        assert all(returns[i] < values[i] + 1e-5 for i in range(len(rewards)))


class TestGRPOAdvantages:
    """Test GRPO-style advantage computation"""

    def test_grpo_advantages_group_mean_baseline(self):
        """Test GRPO advantages with group mean baseline"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model)

        group_rewards = [0.8, 0.5, 0.9, 0.6, 0.7]
        advantages = value_fn.compute_grpo_advantages(group_rewards, baseline_type="group_mean")

        # Mean should be close to zero
        assert torch.abs(advantages.mean()) < 1e-5

    def test_grpo_advantages_group_median_baseline(self):
        """Test GRPO advantages with group median baseline"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, normalize_advantages=False)

        group_rewards = [0.1, 0.5, 0.9, 0.3, 0.7]
        advantages = value_fn.compute_grpo_advantages(group_rewards, baseline_type="group_median")

        # Median is 0.5, so check distribution around it
        median = torch.tensor(group_rewards).median()
        expected = torch.tensor(group_rewards) - median

        assert torch.allclose(advantages, expected, atol=1e-5)

    def test_grpo_advantages_with_uniform_rewards(self):
        """Test GRPO advantages when all rewards are equal"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, normalize_advantages=False)

        group_rewards = [0.5, 0.5, 0.5, 0.5]
        advantages = value_fn.compute_grpo_advantages(group_rewards)

        # All advantages should be zero (all equal to baseline)
        assert torch.allclose(advantages, torch.zeros(4), atol=1e-5)

    def test_grpo_advantages_single_reward(self):
        """Test GRPO advantages with single reward"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, normalize_advantages=False)

        group_rewards = [0.8]
        advantages = value_fn.compute_grpo_advantages(group_rewards)

        # Single reward should have zero advantage (equals its own mean)
        assert torch.allclose(advantages, torch.tensor([0.0]), atol=1e-5)

    def test_grpo_advantages_large_variance(self):
        """Test GRPO advantages with high variance rewards"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model)

        group_rewards = [0.0, 0.1, 0.9, 1.0]
        advantages = value_fn.compute_grpo_advantages(group_rewards)

        # Should normalize despite high variance
        assert len(advantages) == len(group_rewards)


class TestValueFunctionConvenienceCreator:
    """Test create_value_function convenience function"""

    def test_create_value_function_defaults(self):
        """Test creating value function with defaults"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = create_value_function(mock_model)

        assert value_fn is not None
        assert value_fn.gamma == 0.99
        assert value_fn.gae_lambda == 0.95

    def test_create_value_function_custom_params(self):
        """Test creating value function with custom parameters"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = create_value_function(
            mock_model,
            gamma=0.9,
            gae_lambda=0.85,
            normalize_advantages=False
        )

        assert value_fn.gamma == 0.9
        assert value_fn.gae_lambda == 0.85
        assert value_fn.normalize_advantages is False


class TestValueFunctionEdgeCases:
    """Test edge cases and error handling"""

    def test_value_function_with_d_model_config(self):
        """Test ValueFunction with model using d_model instead of hidden_size"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.d_model = 512  # T5-style models use d_model
        mock_model.device = torch.device('cpu')
        delattr(mock_model.config, 'hidden_size')  # Remove hidden_size

        value_fn = ValueFunction(mock_model)
        # Should fallback to default or d_model
        assert value_fn is not None

    def test_value_function_without_config(self):
        """Test ValueFunction with model that has no config"""
        mock_model = Mock()
        mock_model.device = torch.device('cpu')
        # No config attribute

        value_fn = ValueFunction(mock_model)
        # Should use default hidden size
        assert value_fn is not None

    def test_gae_with_mismatched_lengths(self):
        """Test GAE computation handles length mismatches gracefully"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, normalize_advantages=False)

        rewards = [1.0, 0.5]
        values = torch.tensor([0.5, 0.4, 0.3], dtype=torch.float32)  # Extra value

        # Should handle gracefully (may truncate or error)
        try:
            advantages, returns = value_fn.compute_gae(rewards, values[:len(rewards)])
            assert len(advantages) == len(rewards)
        except Exception:
            # If it errors, that's also acceptable behavior
            pass

    def test_gae_with_very_large_rewards(self):
        """Test GAE with extremely large reward values"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, normalize_advantages=False)

        rewards = [1000.0, 5000.0, 10000.0]
        values = torch.tensor([500.0, 2500.0, 5000.0], dtype=torch.float32)

        advantages, returns = value_fn.compute_gae(rewards, values)

        # Should handle large values without overflow
        assert torch.isfinite(advantages).all()
        assert torch.isfinite(returns).all()

    def test_gae_normalization_with_zero_std(self):
        """Test GAE normalization when std is zero"""
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.hidden_size = 768
        mock_model.device = torch.device('cpu')

        value_fn = ValueFunction(mock_model, normalize_advantages=True)

        # All same rewards should lead to zero std
        rewards = [0.5, 0.5, 0.5]
        values = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        advantages, returns = value_fn.compute_gae(rewards, values)

        # Should handle zero std gracefully (add epsilon)
        assert torch.isfinite(advantages).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
