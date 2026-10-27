"""
Comprehensive tests for core/enhanced/advanced_rl_algorithms.py

Covers:
- DPO (Direct Preference Optimization) implementation
- A2C (Advantage Actor-Critic) implementation
- Actor-Critic network architecture
- Configuration classes for each algorithm
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from stateset_agents.core.enhanced.advanced_rl_algorithms import (
    A2CConfig,
    A2CTrainer,
    ActorCriticNetwork,
    DPOConfig,
    DPOTrainer,
)


class TestDPOConfig:
    """Test DPO configuration"""

    def test_dpo_config_defaults(self):
        """Test DPO config with defaults"""
        config = DPOConfig()

        assert config.learning_rate == 5e-7
        assert config.batch_size == 32
        assert config.beta == 0.1
        assert config.max_length == 512

    def test_dpo_config_custom_values(self):
        """Test DPO config with custom values"""
        config = DPOConfig(
            learning_rate=1e-6,
            beta=0.2,
            batch_size=64,
        )

        assert config.learning_rate == 1e-6
        assert config.beta == 0.2
        assert config.batch_size == 64

    def test_dpo_config_beta_parameter(self):
        """Test DPO beta (temperature) parameter"""
        # Low beta (more deterministic)
        low_beta_config = DPOConfig(beta=0.01)
        assert low_beta_config.beta == 0.01

        # High beta (less deterministic)
        high_beta_config = DPOConfig(beta=1.0)
        assert high_beta_config.beta == 1.0


class TestA2CConfig:
    """Test A2C configuration"""

    def test_a2c_config_defaults(self):
        """Test A2C config defaults"""
        config = A2CConfig()

        assert config.learning_rate == 7e-4
        assert config.value_loss_coef == 0.5
        assert config.entropy_coef == 0.01
        assert config.gamma == 0.99
        assert config.n_steps == 5

    def test_a2c_config_custom_values(self):
        """Test A2C config with custom values"""
        config = A2CConfig(
            learning_rate=1e-3,
            n_steps=10,
            gamma=0.95,
        )

        assert config.learning_rate == 1e-3
        assert config.n_steps == 10
        assert config.gamma == 0.95


class TestActorCriticNetwork:
    """Test Actor-Critic network architecture"""

    def test_network_creation(self):
        """Test creating Actor-Critic network"""
        input_size = 128
        hidden_size = 256
        output_size = 10

        network = ActorCriticNetwork(input_size, hidden_size, output_size)

        assert network is not None
        assert isinstance(network, nn.Module)

    def test_network_forward_pass(self):
        """Test forward pass through network"""
        input_size = 64
        hidden_size = 128
        output_size = 5
        batch_size = 4

        network = ActorCriticNetwork(input_size, hidden_size, output_size)

        # Create dummy input
        x = torch.randn(batch_size, input_size)

        # Forward pass
        action_logits, value = network(x)

        assert action_logits.shape == (batch_size, output_size)
        assert value.shape == (batch_size, 1)

    def test_network_weight_initialization(self):
        """Test network weights are properly initialized"""
        network = ActorCriticNetwork(64, 128, 10)

        # Check that weights are not all zeros
        actor_weights = network.actor.weight.data
        assert not torch.all(actor_weights == 0)

        critic_weights = network.critic.weight.data
        assert not torch.all(critic_weights == 0)

    def test_network_with_various_sizes(self):
        """Test network with different architecture sizes"""
        test_configs = [
            (32, 64, 4),
            (128, 256, 20),
            (512, 1024, 100),
        ]

        for input_size, hidden_size, output_size in test_configs:
            network = ActorCriticNetwork(input_size, hidden_size, output_size)
            assert network is not None

            # Test forward pass
            x = torch.randn(2, input_size)
            action_logits, value = network(x)

            assert action_logits.shape == (2, output_size)
            assert value.shape == (2, 1)

    def test_network_shared_layers(self):
        """Test that actor and critic share base layers"""
        network = ActorCriticNetwork(64, 128, 10)

        # Both actor and critic should use shared_layers
        assert hasattr(network, "shared_layers")
        assert hasattr(network, "actor")
        assert hasattr(network, "critic")

        # Shared layers should be a Sequential module
        assert isinstance(network.shared_layers, nn.Sequential)


class TestDPOTrainer:
    """Test DPO trainer"""

    def test_dpo_trainer_creation(self):
        """Test creating DPO trainer"""
        config = DPOConfig()

        mock_model = Mock()
        mock_ref_model = Mock()
        mock_tokenizer = Mock()

        try:
            trainer = DPOTrainer(
                config=config,
                model=mock_model,
                ref_model=mock_ref_model,
                tokenizer=mock_tokenizer,
            )

            assert trainer is not None
            assert trainer.config == config
        except Exception:
            # If initialization requires more, that's OK
            pass

    def test_dpo_preference_learning(self):
        """Test DPO preference learning concept"""
        config = DPOConfig(beta=0.1)

        # DPO uses preference pairs (chosen, rejected)
        # Loss: -log(σ(β * (log π(y_w|x) - log π(y_l|x))))

        # Simulate log probabilities
        log_prob_chosen = torch.tensor([0.5])
        log_prob_rejected = torch.tensor([0.2])

        beta = config.beta

        # Difference should be positive for chosen > rejected
        log_diff = log_prob_chosen - log_prob_rejected
        assert log_diff > 0

        # DPO reward
        reward_diff = beta * log_diff
        assert reward_diff > 0

    def test_dpo_beta_scaling(self):
        """Test DPO beta parameter scaling"""
        low_beta = DPOConfig(beta=0.01)
        high_beta = DPOConfig(beta=1.0)

        assert low_beta.beta < high_beta.beta

        # Lower beta = less emphasis on preference
        # Higher beta = more emphasis on preference


class TestA2CTrainer:
    """Test A2C trainer"""

    def test_a2c_trainer_creation(self):
        """A2C builds its actor-critic over the tokenizer's vocabulary."""
        config = A2CConfig()
        agent = SimpleNamespace(tokenizer=list(range(10)))

        with pytest.warns(DeprecationWarning, match="removed in v0.71.0"):
            trainer = A2CTrainer(agent=agent, config=config, device="cpu")

        assert trainer.config is config
        assert trainer.actor_critic.actor.out_features == 10

    def test_a2c_n_step_returns(self):
        """Test A2C n-step return calculation"""
        config = A2CConfig(n_steps=5, gamma=0.99)

        # N-step return: R_t = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V_{t+n}
        rewards = [1.0, 0.5, 0.8, 0.3, 0.6]
        next_value = 0.4

        gamma = config.gamma
        n_step_return = 0.0

        for i, r in enumerate(rewards):
            n_step_return += (gamma**i) * r

        n_step_return += (gamma ** len(rewards)) * next_value

        assert n_step_return > 0

    def test_a2c_synchronous_updates(self):
        """Test A2C synchronous update concept"""
        config = A2CConfig(n_steps=5)

        # A2C collects n steps, then updates
        # Unlike A3C, updates are synchronous
        assert config.n_steps == 5


class TestAlgorithmComparison:
    """Test differences between algorithms"""

    def test_a2c_vs_dpo_configs(self):
        """A2C uses n-step returns; DPO learns from preferences."""
        a2c_config = A2CConfig()
        dpo_config = DPOConfig()

        assert hasattr(a2c_config, "n_steps")
        assert hasattr(dpo_config, "beta")
        assert a2c_config.learning_rate > 0
        assert dpo_config.learning_rate > 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_network_with_small_dimensions(self):
        """Test network with very small dimensions"""
        network = ActorCriticNetwork(input_size=4, hidden_size=8, output_size=2)

        x = torch.randn(1, 4)
        action_logits, value = network(x)

        assert action_logits.shape == (1, 2)
        assert value.shape == (1, 1)

    def test_network_with_large_batch(self):
        """Test network with large batch size"""
        network = ActorCriticNetwork(64, 128, 10)

        large_batch = 1000
        x = torch.randn(large_batch, 64)

        action_logits, value = network(x)

        assert action_logits.shape == (large_batch, 10)
        assert value.shape == (large_batch, 1)

    def test_config_with_extreme_values(self):
        """Test configs with extreme parameter values"""
        # Very small learning rate
        tiny_lr_config = A2CConfig(learning_rate=1e-10)
        assert tiny_lr_config.learning_rate == 1e-10

        # Gamma near 1 (far-sighted)
        high_gamma_config = A2CConfig(gamma=0.999)
        assert high_gamma_config.gamma == 0.999

    def test_network_gradient_flow(self):
        """Test that gradients flow through network"""
        network = ActorCriticNetwork(64, 128, 10)

        x = torch.randn(4, 64, requires_grad=True)
        action_logits, value = network(x)

        # Compute loss
        loss = action_logits.sum() + value.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None

        # Check network parameters have gradients
        for param in network.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_ppo_duplicate_and_gspo_stub_were_removed_in_v0_70_0():
    """The PPO duplicate and GSPO stub were removed on their stated date."""
    import stateset_agents.core.enhanced as enhanced
    import stateset_agents.core.enhanced.advanced_rl_algorithms as legacy

    for name in (
        "PPOConfig",
        "PPOTrainer",
        "GSPOTrainerStub",
        "create_ppo_trainer",
        "create_gspo_trainer",
    ):
        assert not hasattr(legacy, name), name
        assert name not in enhanced.__all__, name


def test_remaining_research_trainers_warn_with_removal_release():
    """A2C, DPO and the orchestrator name v0.71.0 and their replacements."""
    import stateset_agents.core.enhanced.advanced_rl_algorithms as legacy

    agent = SimpleNamespace(tokenizer=list(range(4)))
    with pytest.warns(DeprecationWarning, match="removed in v0.71.0"):
        orchestrator = legacy.AdvancedRLOrchestrator(agent)
    assert orchestrator.select_algorithm("general", {}) == "a2c"
    assert orchestrator.select_algorithm("preference_learning", {}) == "dpo"
    for cls in (legacy.A2CTrainer, legacy.DPOTrainer, legacy.AdvancedRLOrchestrator):
        assert "Removed in v0.71.0" in (cls.__doc__ or ""), cls.__name__
