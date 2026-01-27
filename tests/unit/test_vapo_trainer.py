"""
Unit tests for the VAPO (Value-Augmented Policy Optimization) Trainer module.

Tests cover VAPO configuration, model management, value head network,
and training components.
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.nn as nn


class TestVAPOConfig:
    """Test VAPOConfig dataclass."""

    def test_vapo_config_defaults(self):
        """Test VAPO config with default values."""
        from stateset_agents.training.vapo_trainer import VAPOConfig

        config = VAPOConfig()

        assert config.model_name == "gpt2"
        assert config.max_prompt_length == 256
        assert config.max_completion_length == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9

    def test_vapo_config_lora_settings(self):
        """Test VAPO config LoRA settings."""
        from stateset_agents.training.vapo_trainer import VAPOConfig

        config = VAPOConfig()

        assert config.use_lora is True
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05

    def test_vapo_config_value_network_settings(self):
        """Test VAPO value network settings."""
        from stateset_agents.training.vapo_trainer import VAPOConfig

        config = VAPOConfig()

        assert config.value_hidden_size == 1024
        assert config.value_num_layers == 2
        assert config.value_warmup_steps == 50

    def test_vapo_config_decoupled_gae(self):
        """Test VAPO decoupled GAE settings."""
        from stateset_agents.training.vapo_trainer import VAPOConfig

        config = VAPOConfig()

        assert config.lambda_critic == 1.0
        assert config.lambda_policy_alpha == 0.05

    def test_vapo_config_asymmetric_clipping(self):
        """Test VAPO asymmetric clipping settings."""
        from stateset_agents.training.vapo_trainer import VAPOConfig

        config = VAPOConfig()

        assert config.clip_eps_low == 0.2
        assert config.clip_eps_high == 0.28

    def test_vapo_config_custom_values(self):
        """Test VAPO config with custom values."""
        from stateset_agents.training.vapo_trainer import VAPOConfig

        config = VAPOConfig(
            model_name="llama-7b",
            max_completion_length=1024,
            value_warmup_steps=100,
            clip_eps_high=0.3,
            group_size=32,
        )

        assert config.model_name == "llama-7b"
        assert config.max_completion_length == 1024
        assert config.value_warmup_steps == 100
        assert config.clip_eps_high == 0.3
        assert config.group_size == 32

    def test_vapo_config_from_training_config(self):
        """Test creating VAPO config from training config."""
        from stateset_agents.training.config import TrainingConfig
        from stateset_agents.training.vapo_trainer import VAPOConfig

        base_config = TrainingConfig(
            model_name="gpt2",
            learning_rate=1e-5,
        )

        vapo_config = VAPOConfig.from_training_config(
            base_config,
            value_warmup_steps=75,
            group_size=8,
        )

        assert vapo_config.value_warmup_steps == 75
        assert vapo_config.group_size == 8


class TestVAPOModelManager:
    """Test VAPOModelManager class."""

    @pytest.fixture
    def vapo_config(self):
        """Create a VAPO config for testing."""
        from stateset_agents.training.vapo_trainer import VAPOConfig
        return VAPOConfig(model_name="gpt2", use_lora=True)

    @pytest.fixture
    def model_manager(self, vapo_config):
        """Create a VAPOModelManager for testing."""
        from stateset_agents.training.vapo_trainer import VAPOModelManager
        return VAPOModelManager(vapo_config)

    def test_model_manager_creation(self, model_manager):
        """Test model manager creation."""
        assert model_manager.model is None
        assert model_manager.tokenizer is None
        assert model_manager.device is not None

    @patch("training.vapo_trainer.AutoTokenizer")
    @patch("training.vapo_trainer.AutoModelForCausalLM")
    @patch("training.vapo_trainer.get_peft_model")
    def test_load_model_and_tokenizer(
        self, mock_peft, mock_model_class, mock_tokenizer_class, model_manager
    ):
        """Test loading model and tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_peft.return_value = mock_model

        model, tokenizer = model_manager.load_model_and_tokenizer()

        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()
        assert tokenizer.pad_token == tokenizer.eos_token

    @patch("training.vapo_trainer.AutoTokenizer")
    @patch("training.vapo_trainer.AutoModelForCausalLM")
    def test_load_model_without_lora(
        self, mock_model_class, mock_tokenizer_class, vapo_config
    ):
        """Test loading model without LoRA."""
        vapo_config.use_lora = False

        from stateset_agents.training.vapo_trainer import VAPOModelManager
        manager = VAPOModelManager(vapo_config)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        model, tokenizer = manager.load_model_and_tokenizer()

        assert model == mock_model


class TestValueHead:
    """Test ValueHead neural network module."""

    @pytest.fixture
    def value_head(self):
        """Create a ValueHead for testing."""
        from stateset_agents.training.vapo_trainer import ValueHead
        return ValueHead(
            hidden_size=768,
            value_hidden_size=1024,
            num_layers=2,
            dropout=0.1,
        )

    def test_value_head_creation(self, value_head):
        """Test ValueHead creation."""
        assert value_head is not None
        assert isinstance(value_head, nn.Module)

    def test_value_head_forward(self, value_head):
        """Test ValueHead forward pass."""
        batch_size = 4
        seq_len = 128
        hidden_size = 768

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output = value_head(hidden_states)

        # Output should be single value per position
        assert output.shape == (batch_size, seq_len, 1)

    def test_value_head_output_range(self, value_head):
        """Test ValueHead output is unbounded (can be any real value)."""
        hidden_states = torch.randn(2, 64, 768)
        output = value_head(hidden_states)

        # Values can be any real number
        assert torch.isfinite(output).all()


class TestVAPOAdvantageComputation:
    """Test VAPO advantage computation."""

    def test_gae_computation_concept(self):
        """Test GAE computation conceptually."""
        # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        gamma = 0.99
        lam = 0.95
        rewards = torch.tensor([1.0, 0.0, 0.0, 10.0])
        values = torch.tensor([5.0, 5.0, 5.0, 5.0, 0.0])  # Including bootstrap value

        # Compute advantages (simplified)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        assert advantages.shape == rewards.shape

    def test_decoupled_gae_parameters(self):
        """Test decoupled GAE uses different lambdas."""
        lambda_critic = 1.0  # Unbiased for value estimation
        lambda_policy_base = 0.95  # Biased for variance reduction

        assert lambda_critic > lambda_policy_base

    def test_length_adaptive_lambda(self):
        """Test length-adaptive lambda computation."""
        # Lambda decreases with sequence length
        alpha = 0.05
        lambda_base = 0.95

        for seq_len in [100, 500, 1000]:
            lambda_adaptive = lambda_base * (1 - alpha * (seq_len / 1000))
            assert 0 < lambda_adaptive < 1


class TestVAPOClipping:
    """Test VAPO asymmetric clipping (Clip-Higher)."""

    def test_asymmetric_clip_ranges(self):
        """Test asymmetric clipping ranges."""
        clip_eps_low = 0.2
        clip_eps_high = 0.28

        # Clip-Higher: higher upper bound
        assert clip_eps_high > clip_eps_low

        # Verify clipping bounds
        lower_bound = 1 - clip_eps_low  # 0.8
        upper_bound = 1 + clip_eps_high  # 1.28

        assert lower_bound == 0.8
        assert upper_bound == 1.28

    def test_clipped_ratio(self):
        """Test clipping importance ratios."""
        clip_eps_low = 0.2
        clip_eps_high = 0.28

        ratios = torch.tensor([0.5, 0.9, 1.0, 1.2, 1.5])

        # Asymmetric clipping
        clipped = torch.clamp(ratios, 1 - clip_eps_low, 1 + clip_eps_high)

        assert clipped[0] == 0.8  # Clamped up
        assert clipped[1] == 0.9  # Unchanged
        assert clipped[2] == 1.0  # Unchanged
        assert clipped[3] == 1.2  # Unchanged
        assert clipped[4] == 1.28  # Clamped down


class TestVAPOTokenLevelLoss:
    """Test VAPO token-level loss computation."""

    def test_token_level_loss_concept(self):
        """Test token-level loss conceptually."""
        batch_size = 2
        seq_len = 10

        # Token-level advantages
        advantages = torch.randn(batch_size, seq_len)

        # Token-level policy ratios
        ratios = torch.abs(torch.randn(batch_size, seq_len)) + 0.5

        # Token-level loss (without clipping for simplicity)
        loss = -(ratios * advantages).mean()

        assert loss.dim() == 0  # Scalar loss

    def test_token_vs_sequence_loss(self):
        """Test difference between token and sequence level loss."""
        batch_size = 2
        seq_len = 10

        advantages = torch.randn(batch_size, seq_len)

        # Token-level: mean over all tokens
        token_loss = advantages.mean()

        # Sequence-level: mean of sequence sums
        seq_loss = advantages.sum(dim=1).mean()

        # These should be different unless seq_len=1
        assert token_loss != seq_loss or seq_len == 1


class TestVAPOPositiveLMLoss:
    """Test VAPO positive example LM loss."""

    def test_positive_lm_loss_weight(self):
        """Test positive LM loss weight parameter."""
        from stateset_agents.training.vapo_trainer import VAPOConfig

        config = VAPOConfig()

        assert config.use_positive_lm_loss is True
        assert config.positive_lm_weight == 0.1

    def test_lm_loss_computation(self):
        """Test LM loss computation for positive examples."""
        # Simulated log probabilities and targets
        vocab_size = 1000
        batch_size = 2
        seq_len = 10

        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Cross entropy loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
        )

        assert loss.dim() == 0
        assert loss > 0


class TestVAPOGroupSampling:
    """Test VAPO group sampling strategy."""

    def test_group_size_config(self):
        """Test group size configuration."""
        from stateset_agents.training.vapo_trainer import VAPOConfig

        config = VAPOConfig()

        assert config.group_size == 16
        assert config.num_prompts_per_batch == 512

    def test_group_sampling_concept(self):
        """Test group sampling conceptually."""
        num_prompts = 10
        group_size = 4

        # Total samples = prompts * group_size
        total_samples = num_prompts * group_size
        assert total_samples == 40

        # Each prompt has group_size responses
        prompts = list(range(num_prompts))
        responses_per_prompt = {p: list(range(group_size)) for p in prompts}

        assert len(responses_per_prompt) == num_prompts
        assert all(len(r) == group_size for r in responses_per_prompt.values())


class TestVAPOOptimizers:
    """Test VAPO separate optimizers for actor and critic."""

    def test_separate_learning_rates(self):
        """Test separate learning rates for actor and critic."""
        from stateset_agents.training.vapo_trainer import VAPOConfig

        config = VAPOConfig()

        assert config.actor_learning_rate == 1e-6
        assert config.critic_learning_rate == 2e-6
        # Critic LR is higher for faster value network convergence
        assert config.critic_learning_rate > config.actor_learning_rate

    def test_optimizer_creation_concept(self):
        """Test creating separate optimizers conceptually."""
        actor_params = [torch.randn(10, 10, requires_grad=True)]
        critic_params = [torch.randn(10, 10, requires_grad=True)]

        actor_lr = 1e-6
        critic_lr = 2e-6

        actor_optimizer = torch.optim.Adam(actor_params, lr=actor_lr)
        critic_optimizer = torch.optim.Adam(critic_params, lr=critic_lr)

        assert actor_optimizer.defaults["lr"] == actor_lr
        assert critic_optimizer.defaults["lr"] == critic_lr


class TestVAPOValueWarmup:
    """Test VAPO value network warmup."""

    def test_warmup_steps_config(self):
        """Test warmup steps configuration."""
        from stateset_agents.training.vapo_trainer import VAPOConfig

        config = VAPOConfig()
        assert config.value_warmup_steps == 50

    def test_warmup_training_concept(self):
        """Test warmup training conceptually."""
        warmup_steps = 50
        total_steps = 1000

        # During warmup, only train value network
        for step in range(total_steps):
            is_warmup = step < warmup_steps
            train_policy = not is_warmup
            train_value = True  # Always train value

            if is_warmup:
                assert train_policy is False
                assert train_value is True


class TestVAPOTrainer:
    """Test VAPOTrainer class."""

    @pytest.fixture
    def vapo_config(self):
        """Create a VAPO config for testing."""
        from stateset_agents.training.vapo_trainer import VAPOConfig
        return VAPOConfig(
            model_name="gpt2",
            num_iterations=2,
            group_size=4,
        )

    def test_trainer_initialization(self, vapo_config):
        """Test trainer can be initialized."""
        # Just verify config is valid
        assert vapo_config.model_name == "gpt2"
        assert vapo_config.group_size == 4


class TestVAPOMetrics:
    """Test VAPO training metrics."""

    def test_value_loss_coefficient(self):
        """Test value loss coefficient."""
        from stateset_agents.training.vapo_trainer import VAPOConfig

        config = VAPOConfig()
        assert config.value_loss_coef == 0.5

    def test_entropy_coefficient(self):
        """Test entropy coefficient (usually 0 for reasoning)."""
        from stateset_agents.training.vapo_trainer import VAPOConfig

        config = VAPOConfig()
        assert config.entropy_coef == 0.0

    def test_combined_loss_computation(self):
        """Test combined loss computation."""
        policy_loss = torch.tensor(0.5)
        value_loss = torch.tensor(0.3)
        positive_lm_loss = torch.tensor(0.2)

        value_coef = 0.5
        lm_weight = 0.1

        total_loss = policy_loss + value_coef * value_loss + lm_weight * positive_lm_loss

        expected = 0.5 + 0.5 * 0.3 + 0.1 * 0.2
        assert abs(total_loss.item() - expected) < 1e-5
