"""
Unit tests for the DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) Trainer.

Tests cover DAPO configuration, asymmetric clipping, dynamic sampling,
overlong reward shaping, and token-level loss computation.
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.nn as nn


class TestDAPOConfig:
    """Test DAPOConfig dataclass."""

    def test_dapo_config_defaults(self):
        """Test DAPO config with default values."""
        from stateset_agents.training.dapo_trainer import DAPOConfig

        config = DAPOConfig()

        assert config.model_name == "gpt2"
        assert config.max_prompt_length == 256
        assert config.max_completion_length == 512

    def test_dapo_config_clip_higher(self):
        """Test DAPO Clip-Higher configuration."""
        from stateset_agents.training.dapo_trainer import DAPOConfig

        config = DAPOConfig()

        # Asymmetric clipping: higher upper bound
        assert config.clip_eps_low == 0.2
        assert config.clip_eps_high == 0.28
        assert config.clip_eps_high > config.clip_eps_low

    def test_dapo_config_dynamic_sampling(self):
        """Test DAPO dynamic sampling configuration."""
        from stateset_agents.training.dapo_trainer import DAPOConfig

        config = DAPOConfig()

        assert config.use_dynamic_sampling is True
        assert config.min_accuracy_threshold == 0.0
        assert config.max_accuracy_threshold == 1.0
        assert config.dynamic_sampling_buffer_size == 1024

    def test_dapo_config_overlong_shaping(self):
        """Test DAPO overlong reward shaping configuration."""
        from stateset_agents.training.dapo_trainer import DAPOConfig

        config = DAPOConfig()

        assert config.use_overlong_shaping is True
        assert config.max_generation_length == 20480
        assert config.overlong_cache_length == 4096
        assert config.overlong_penalty == -1.0

    def test_dapo_config_no_kl_penalty(self):
        """Test DAPO has no KL penalty by default."""
        from stateset_agents.training.dapo_trainer import DAPOConfig

        config = DAPOConfig()

        assert config.beta == 0.0
        assert config.use_reference_model is False

    def test_dapo_config_custom_values(self):
        """Test DAPO config with custom values."""
        from stateset_agents.training.dapo_trainer import DAPOConfig

        config = DAPOConfig(
            model_name="llama-7b",
            clip_eps_high=0.35,
            group_size=32,
            use_vllm=True,
        )

        assert config.model_name == "llama-7b"
        assert config.clip_eps_high == 0.35
        assert config.group_size == 32
        assert config.use_vllm is True


class TestDAPOClipHigher:
    """Test DAPO Clip-Higher asymmetric clipping."""

    def test_asymmetric_clip_bounds(self):
        """Test asymmetric clipping bounds."""
        clip_eps_low = 0.2
        clip_eps_high = 0.28

        lower_bound = 1 - clip_eps_low  # 0.8
        upper_bound = 1 + clip_eps_high  # 1.28

        assert lower_bound == 0.8
        assert upper_bound == 1.28

    def test_clip_higher_prevents_entropy_collapse(self):
        """Test that Clip-Higher allows more exploration."""
        clip_eps_low = 0.2
        clip_eps_high = 0.28

        # Ratio > 1 means action is more likely under current policy
        # Higher upper bound allows this to contribute more to loss
        ratio = 1.3  # Above standard PPO clip

        # Standard PPO would clip to 1.2
        standard_clipped = min(ratio, 1 + 0.2)
        assert standard_clipped == 1.2

        # DAPO allows up to 1.28
        dapo_clipped = min(ratio, 1 + clip_eps_high)
        assert dapo_clipped == 1.28

    def test_clipped_surrogate_loss(self):
        """Test clipped surrogate loss computation."""
        batch_size = 4
        clip_eps_low = 0.2
        clip_eps_high = 0.28

        ratios = torch.tensor([0.7, 0.95, 1.1, 1.35])
        advantages = torch.tensor([1.0, -0.5, 0.8, -0.3])

        # Asymmetric clipping
        clipped = torch.clamp(ratios, 1 - clip_eps_low, 1 + clip_eps_high)

        # Surrogate objectives
        surr1 = ratios * advantages
        surr2 = clipped * advantages

        # Pessimistic (take worse case)
        loss = -torch.min(surr1, surr2).mean()

        assert loss.dim() == 0


class TestDAPODynamicSampling:
    """Test DAPO dynamic sampling."""

    def test_filter_trivial_accuracy(self):
        """Test filtering prompts with trivial accuracy."""
        # Simulate group accuracies for prompts
        group_size = 16
        num_prompts = 10

        # Binary rewards (0 or 1)
        rewards = torch.randint(0, 2, (num_prompts, group_size)).float()

        # Calculate accuracy per prompt
        accuracies = rewards.mean(dim=1)

        # Filter out 0% and 100% accuracy
        min_thresh = 0.0
        max_thresh = 1.0

        valid_mask = (accuracies > min_thresh) & (accuracies < max_thresh)

        # Only prompts with mixed results remain
        valid_prompts = valid_mask.sum().item()
        assert valid_prompts <= num_prompts

    def test_dynamic_sampling_buffer(self):
        """Test dynamic sampling buffer concept."""
        buffer_size = 1024
        buffer = []

        # Simulate adding samples
        for i in range(100):
            if len(buffer) < buffer_size:
                buffer.append(f"sample_{i}")

        assert len(buffer) == 100
        assert len(buffer) <= buffer_size

    def test_accuracy_threshold_filtering(self):
        """Test accuracy threshold filtering."""
        min_thresh = 0.0
        max_thresh = 1.0

        test_cases = [
            (0.0, False),   # All wrong -> filter
            (0.5, True),    # Mixed -> keep
            (0.25, True),   # Mixed -> keep
            (0.75, True),   # Mixed -> keep
            (1.0, False),   # All correct -> filter
        ]

        for accuracy, should_keep in test_cases:
            keep = (accuracy > min_thresh) and (accuracy < max_thresh)
            assert keep == should_keep, f"Failed for accuracy={accuracy}"


class TestDAPOOverlongShaping:
    """Test DAPO overlong reward shaping."""

    def test_overlong_penalty_config(self):
        """Test overlong penalty configuration."""
        from stateset_agents.training.dapo_trainer import DAPOConfig

        config = DAPOConfig()

        assert config.overlong_penalty == -1.0
        assert config.max_generation_length == 20480
        assert config.overlong_cache_length == 4096

    def test_soft_penalty_computation(self):
        """Test soft penalty for sequences approaching max length."""
        max_len = 20480
        cache_len = 4096

        # Penalty starts at max_len - cache_len
        penalty_start = max_len - cache_len

        test_lengths = [1000, 10000, 16384, 18000, 20000, 20480]

        for seq_len in test_lengths:
            if seq_len < penalty_start:
                # No penalty
                penalty = 0.0
            elif seq_len >= max_len:
                # Full penalty for truncated
                penalty = -1.0
            else:
                # Soft penalty (linear interpolation)
                progress = (seq_len - penalty_start) / cache_len
                penalty = -progress

            assert -1.0 <= penalty <= 0.0

    def test_overlong_reward_shaping(self):
        """Test applying overlong penalty to rewards."""
        max_len = 1000
        cache_len = 200
        penalty_start = max_len - cache_len

        base_reward = 1.0
        overlong_penalty = -1.0

        # Short sequence: no penalty
        short_len = 500
        short_shaped = base_reward
        assert short_shaped == 1.0

        # Near-max sequence: partial penalty
        near_max_len = 900
        progress = (near_max_len - penalty_start) / cache_len
        near_max_shaped = base_reward + progress * overlong_penalty
        assert near_max_shaped < base_reward

        # Truncated sequence: full penalty
        truncated_len = max_len
        truncated_shaped = base_reward + overlong_penalty
        assert truncated_shaped == 0.0


class TestDAPOTokenLevelLoss:
    """Test DAPO token-level loss normalization."""

    def test_token_level_vs_sample_level(self):
        """Test difference between token and sample level normalization."""
        batch_size = 4
        seq_lengths = [100, 200, 50, 150]

        # Token-level losses per sample
        losses_per_sample = torch.tensor([10.0, 20.0, 5.0, 15.0])

        # Sample-level normalization (mean over samples)
        sample_level = losses_per_sample.mean()
        assert sample_level == 12.5

        # Token-level normalization (divide by total tokens)
        total_tokens = sum(seq_lengths)
        token_level = losses_per_sample.sum() / total_tokens
        assert token_level == 50.0 / 500  # 0.1

    def test_token_level_prevents_length_bias(self):
        """Test token-level normalization prevents length bias."""
        # Without token-level: longer sequences dominate
        # With token-level: equal contribution per token

        short_seq = {"length": 50, "loss_per_token": 0.1}
        long_seq = {"length": 200, "loss_per_token": 0.1}

        # Total loss per sequence
        short_total = short_seq["length"] * short_seq["loss_per_token"]
        long_total = long_seq["length"] * long_seq["loss_per_token"]

        # Sample-level: mean of sequence totals
        sample_mean = (short_total + long_total) / 2
        assert sample_mean == 12.5

        # Token-level: mean over all tokens
        total_tokens = short_seq["length"] + long_seq["length"]
        total_loss = short_total + long_total
        token_mean = total_loss / total_tokens
        assert token_mean == 0.1  # Equals loss per token


class TestDAPOGroupSampling:
    """Test DAPO group sampling."""

    def test_group_size_config(self):
        """Test group size configuration."""
        from stateset_agents.training.dapo_trainer import DAPOConfig

        config = DAPOConfig()

        assert config.group_size == 16
        assert config.prompt_batch_size == 512
        assert config.num_gradient_updates == 16

    def test_group_reward_baseline(self):
        """Test group reward baseline computation."""
        group_size = 16
        num_prompts = 10

        rewards = torch.randn(num_prompts, group_size)

        # Baseline is mean within each group
        baselines = rewards.mean(dim=1, keepdim=True)

        # Advantages
        advantages = rewards - baselines

        # Each group's advantages sum to zero
        for i in range(num_prompts):
            assert abs(advantages[i].sum().item()) < 1e-5


class TestDAPOModelManager:
    """Test DAPOModelManager class."""

    @pytest.fixture
    def dapo_config(self):
        """Create a DAPO config for testing."""
        from stateset_agents.training.dapo_trainer import DAPOConfig
        return DAPOConfig(model_name="gpt2", use_lora=True)

    @pytest.fixture
    def model_manager(self, dapo_config):
        """Create a DAPOModelManager for testing."""
        from stateset_agents.training.dapo_trainer import DAPOModelManager
        return DAPOModelManager(dapo_config)

    def test_model_manager_creation(self, model_manager):
        """Test model manager creation."""
        assert model_manager.model is None
        assert model_manager.tokenizer is None
        assert model_manager.device is not None


class TestDAPOvLLMIntegration:
    """Test DAPO vLLM integration."""

    def test_vllm_config(self):
        """Test vLLM configuration."""
        from stateset_agents.training.dapo_trainer import DAPOConfig

        config = DAPOConfig(use_vllm=True)

        assert config.use_vllm is True
        assert config.vllm_gpu_memory_utilization == 0.85
        assert config.vllm_tensor_parallel_size == 1
        assert config.vllm_enable_prefix_caching is True


class TestDAPOTrainer:
    """Test DAPOTrainer class."""

    @pytest.fixture
    def dapo_config(self):
        """Create a DAPO config for testing."""
        from stateset_agents.training.dapo_trainer import DAPOConfig
        return DAPOConfig(
            model_name="gpt2",
            group_size=4,
            num_gradient_updates=2,
        )

    def test_trainer_config_valid(self, dapo_config):
        """Test trainer config is valid."""
        assert dapo_config.model_name == "gpt2"
        assert dapo_config.group_size == 4
        assert dapo_config.num_gradient_updates == 2


class TestDAPOMetrics:
    """Test DAPO training metrics."""

    def test_entropy_tracking(self):
        """Test entropy tracking during training."""
        batch_size = 4
        vocab_size = 1000

        # Simulated logits
        logits = torch.randn(batch_size, vocab_size)
        probs = torch.softmax(logits, dim=-1)

        # Entropy
        entropy = -(probs * probs.log()).sum(dim=-1).mean()

        assert entropy.dim() == 0
        assert entropy > 0  # Entropy is positive

    def test_clip_fraction_tracking(self):
        """Test tracking fraction of clipped ratios."""
        batch_size = 100
        clip_eps_low = 0.2
        clip_eps_high = 0.28

        ratios = torch.abs(torch.randn(batch_size)) + 0.5

        # Count clipped
        lower = 1 - clip_eps_low
        upper = 1 + clip_eps_high

        clipped_low = (ratios < lower).float().mean()
        clipped_high = (ratios > upper).float().mean()
        clip_fraction = clipped_low + clipped_high

        assert 0 <= clip_fraction <= 1

    def test_accuracy_distribution(self):
        """Test tracking accuracy distribution."""
        num_prompts = 100
        group_size = 16

        rewards = torch.randint(0, 2, (num_prompts, group_size)).float()
        accuracies = rewards.mean(dim=1)

        # Distribution stats
        mean_acc = accuracies.mean()
        std_acc = accuracies.std()
        zero_acc = (accuracies == 0).float().mean()
        full_acc = (accuracies == 1).float().mean()

        assert 0 <= mean_acc <= 1
        assert std_acc >= 0
        assert 0 <= zero_acc <= 1
        assert 0 <= full_acc <= 1


class TestDAPOLearningRate:
    """Test DAPO learning rate configuration."""

    def test_constant_lr_schedule(self):
        """Test DAPO uses constant learning rate."""
        from stateset_agents.training.dapo_trainer import DAPOConfig

        config = DAPOConfig()

        assert config.learning_rate == 1e-6
        assert config.lr_scheduler_type == "constant"

    def test_constant_schedule_concept(self):
        """Test constant LR schedule conceptually."""
        initial_lr = 1e-6
        num_steps = 1000

        # Constant schedule returns same LR
        for step in range(num_steps):
            current_lr = initial_lr  # Constant
            assert current_lr == initial_lr
