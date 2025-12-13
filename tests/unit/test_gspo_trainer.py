"""
Unit tests for the GSPO (Group Sequence Policy Optimization) Trainer module.

Tests cover GSPO configuration, model management, sequence-level optimization,
and training components.
"""

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.nn as nn


class TestGSPOConfig:
    """Test GSPOConfig dataclass."""

    def test_gspo_config_defaults(self):
        """Test GSPO config with default values."""
        from training.gspo_trainer import GSPOConfig

        config = GSPOConfig()

        assert config.num_generations == 4
        assert config.beta == 0.0
        assert config.clip_range_left == 3e-4
        assert config.clip_range_right == 4e-4

    def test_gspo_config_generation_params(self):
        """Test GSPO generation parameters."""
        from training.gspo_trainer import GSPOConfig

        config = GSPOConfig()

        assert config.max_prompt_length == 256
        assert config.max_completion_length == 256
        assert config.temperature == 0.7
        assert config.top_p == 0.9

    def test_gspo_config_lora_settings(self):
        """Test GSPO LoRA configuration."""
        from training.gspo_trainer import GSPOConfig

        config = GSPOConfig()

        assert config.use_lora is True
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05

    def test_gspo_config_memory_optimization(self):
        """Test GSPO memory optimization settings."""
        from training.gspo_trainer import GSPOConfig

        config = GSPOConfig()

        assert config.gradient_checkpointing is True
        assert config.use_8bit is False
        assert config.use_4bit is False

    def test_gspo_config_custom_values(self):
        """Test GSPO config with custom values."""
        from training.gspo_trainer import GSPOConfig

        config = GSPOConfig(
            num_generations=8,
            beta=0.01,
            clip_range_left=5e-4,
            clip_range_right=6e-4,
            use_vllm=True,
        )

        assert config.num_generations == 8
        assert config.beta == 0.01
        assert config.clip_range_left == 5e-4
        assert config.use_vllm is True

    def test_gspo_config_from_training_config(self):
        """Test creating GSPO config from training config."""
        from training.config import TrainingConfig
        from training.gspo_trainer import GSPOConfig

        base_config = TrainingConfig(
            model_name="gpt2",
            learning_rate=1e-5,
        )

        gspo_config = GSPOConfig.from_training_config(
            base_config,
            num_generations=16,
            beta=0.05,
        )

        assert gspo_config.num_generations == 16
        assert gspo_config.beta == 0.05


class TestGSPOModelManager:
    """Test GSPOModelManager class."""

    @pytest.fixture
    def gspo_config(self):
        """Create a GSPO config for testing."""
        from training.gspo_trainer import GSPOConfig
        return GSPOConfig(model_name="gpt2", use_lora=True)

    @pytest.fixture
    def model_manager(self, gspo_config):
        """Create a GSPOModelManager for testing."""
        from training.gspo_trainer import GSPOModelManager
        return GSPOModelManager(gspo_config)

    def test_model_manager_creation(self, model_manager):
        """Test model manager creation."""
        assert model_manager.model is None
        assert model_manager.tokenizer is None
        assert model_manager.ref_model is None
        assert model_manager.device is not None

    @patch("training.gspo_trainer.AutoTokenizer")
    @patch("training.gspo_trainer.AutoModelForCausalLM")
    @patch("training.gspo_trainer.get_peft_model")
    def test_load_model_and_tokenizer(
        self, mock_peft, mock_model_class, mock_tokenizer_class, model_manager
    ):
        """Test loading model and tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.gradient_checkpointing_enable = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_peft.return_value = mock_model

        model, tokenizer = model_manager.load_model_and_tokenizer()

        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch("training.gspo_trainer.AutoTokenizer")
    @patch("training.gspo_trainer.AutoModelForCausalLM")
    def test_load_model_with_quantization(
        self, mock_model_class, mock_tokenizer_class, gspo_config
    ):
        """Test loading model with quantization."""
        gspo_config.use_8bit = True
        gspo_config.use_lora = False

        from training.gspo_trainer import GSPOModelManager
        manager = GSPOModelManager(gspo_config)

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.gradient_checkpointing_enable = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        with patch("training.gspo_trainer._require_bitsandbytes"), patch(
            "training.gspo_trainer.prepare_model_for_kbit_training"
        ) as mock_kbit:
            mock_kbit.return_value = mock_model
            model, tokenizer = manager.load_model_and_tokenizer()

            # Should call prepare_model_for_kbit_training
            mock_kbit.assert_called_once()


class TestGSPOSequenceLevelOptimization:
    """Test GSPO sequence-level optimization."""

    def test_sequence_level_ratio(self):
        """Test sequence-level importance ratio computation."""
        batch_size = 4
        seq_len = 100

        # Log probabilities under current and reference policies
        log_probs_current = torch.randn(batch_size, seq_len)
        log_probs_ref = torch.randn(batch_size, seq_len)

        # Sequence-level log ratio
        seq_log_ratio = log_probs_current.sum(dim=1) - log_probs_ref.sum(dim=1)

        # Sequence-level ratio
        seq_ratio = torch.exp(seq_log_ratio)

        assert seq_ratio.shape == (batch_size,)

    def test_sequence_level_clipping(self):
        """Test GSPO sequence-level clipping."""
        # GSPO uses smaller clipping ranges than token-level methods
        clip_left = 3e-4  # 1 - epsilon_l
        clip_right = 4e-4  # 1 + epsilon_r

        # The clipping is around 1
        lower = 1 - clip_left
        upper = 1 + clip_right

        ratios = torch.tensor([0.9, 0.9999, 1.0, 1.0001, 1.1])

        clipped = torch.clamp(ratios, lower, upper)

        # Ratios far from 1 get clipped
        assert clipped[0] == lower  # Clipped up
        assert clipped[4] == upper  # Clipped down

    def test_gspo_vs_grpo_clipping_ranges(self):
        """Test that GSPO uses smaller clipping ranges than GRPO."""
        # GSPO clipping ranges (sequence-level)
        gspo_clip_left = 3e-4
        gspo_clip_right = 4e-4

        # GRPO clipping ranges (token-level, typically much larger)
        grpo_clip = 0.2

        # GSPO uses much smaller ranges
        assert gspo_clip_left < grpo_clip
        assert gspo_clip_right < grpo_clip


class TestGSPOGroupSampling:
    """Test GSPO group sampling."""

    def test_group_size_config(self):
        """Test group size configuration."""
        from training.gspo_trainer import GSPOConfig

        config = GSPOConfig()

        assert config.num_generations == 4  # Group size G

    def test_group_reward_normalization(self):
        """Test reward normalization within groups."""
        group_size = 4
        num_prompts = 10

        # Simulate rewards for groups
        rewards = torch.randn(num_prompts, group_size)

        # Normalize within each group (per prompt)
        group_mean = rewards.mean(dim=1, keepdim=True)
        group_std = rewards.std(dim=1, keepdim=True)

        normalized = (rewards - group_mean) / (group_std + 1e-8)

        # Each group should have mean ~0 and std ~1
        for i in range(num_prompts):
            assert abs(normalized[i].mean().item()) < 0.1
            assert abs(normalized[i].std().item() - 1.0) < 0.1

    def test_group_advantage_computation(self):
        """Test advantage computation within groups."""
        group_size = 4
        num_prompts = 5

        rewards = torch.randn(num_prompts, group_size)

        # Baseline is mean reward within group
        baselines = rewards.mean(dim=1, keepdim=True)

        # Advantage is reward minus baseline
        advantages = rewards - baselines

        # Advantages within each group should sum to ~0
        for i in range(num_prompts):
            assert abs(advantages[i].sum().item()) < 1e-5


class TestGSPOKLPenalty:
    """Test GSPO KL divergence penalty."""

    def test_beta_config(self):
        """Test KL penalty coefficient (beta) configuration."""
        from training.gspo_trainer import GSPOConfig

        config = GSPOConfig()
        # Default beta is 0 (no KL penalty in basic GSPO)
        assert config.beta == 0.0

    def test_kl_divergence_computation(self):
        """Test KL divergence computation."""
        batch_size = 4
        seq_len = 50

        log_probs_current = torch.randn(batch_size, seq_len)
        log_probs_ref = torch.randn(batch_size, seq_len)

        # KL(current || ref) = sum(p_current * (log p_current - log p_ref))
        # For sequence level: sum over tokens, then mean over batch
        kl = (torch.exp(log_probs_current) * (log_probs_current - log_probs_ref)).sum(dim=1).mean()

        assert kl.dim() == 0  # Scalar

    def test_kl_penalty_effect(self):
        """Test KL penalty effect on loss."""
        policy_loss = torch.tensor(0.5)
        kl = torch.tensor(0.1)
        beta = 0.01

        total_loss = policy_loss + beta * kl

        assert total_loss > policy_loss


class TestGSPOLoss:
    """Test GSPO loss computation."""

    def test_clipped_loss_computation(self):
        """Test clipped GSPO loss computation."""
        batch_size = 8

        # Sequence-level ratios
        ratios = torch.abs(torch.randn(batch_size)) + 0.8

        # Advantages
        advantages = torch.randn(batch_size)

        clip_left = 3e-4
        clip_right = 4e-4

        # Clipped ratios
        clipped_ratios = torch.clamp(ratios, 1 - clip_left, 1 + clip_right)

        # Loss is negative advantage weighted by clipped ratio
        loss1 = -(ratios * advantages)
        loss2 = -(clipped_ratios * advantages)

        # Take minimum (pessimistic)
        loss = torch.max(loss1, loss2).mean()

        assert loss.dim() == 0

    def test_gspo_objective(self):
        """Test full GSPO objective."""
        batch_size = 4

        ratios = torch.abs(torch.randn(batch_size)) + 0.9
        advantages = torch.randn(batch_size)

        clip_left = 3e-4
        clip_right = 4e-4
        beta = 0.01

        # Clip ratios
        clipped = torch.clamp(ratios, 1 - clip_left, 1 + clip_right)

        # Policy loss (clipped surrogate)
        surr1 = ratios * advantages
        surr2 = clipped * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty (simplified)
        kl_penalty = (ratios.log()).mean()

        total_loss = policy_loss + beta * kl_penalty

        assert total_loss.dim() == 0


class TestGSPOGeneration:
    """Test GSPO response generation."""

    def test_generation_config(self):
        """Test generation configuration."""
        from training.gspo_trainer import GSPOConfig

        config = GSPOConfig()

        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_completion_length == 256

    def test_multiple_generation_concept(self):
        """Test generating multiple responses per prompt."""
        num_prompts = 5
        group_size = 4

        # Simulate generating G responses per prompt
        prompts = [f"Prompt {i}" for i in range(num_prompts)]
        responses = {
            p: [f"Response {j} for {p}" for j in range(group_size)]
            for p in prompts
        }

        assert len(responses) == num_prompts
        assert all(len(r) == group_size for r in responses.values())


class TestGSPOIterativeTraining:
    """Test GSPO iterative/online training."""

    def test_outer_iterations_config(self):
        """Test outer iteration configuration."""
        from training.gspo_trainer import GSPOConfig

        config = GSPOConfig()

        assert config.num_outer_iterations == 1
        assert config.generations_per_iteration == 100

    def test_iterative_training_concept(self):
        """Test iterative training conceptually."""
        num_outer = 3
        prompts_per_iter = 100
        group_size = 4

        total_samples = 0
        for outer in range(num_outer):
            # Generate phase
            samples_generated = prompts_per_iter * group_size

            # Train phase
            total_samples += samples_generated

        expected_total = num_outer * prompts_per_iter * group_size
        assert total_samples == expected_total


class TestGSPOvLLMIntegration:
    """Test GSPO vLLM backend integration."""

    def test_vllm_config(self):
        """Test vLLM configuration."""
        from training.gspo_trainer import GSPOConfig

        config = GSPOConfig(use_vllm=True)
        assert config.use_vllm is True

    @patch("training.gspo_trainer.VLLM_AVAILABLE", True)
    def test_vllm_generation_concept(self):
        """Test vLLM generation conceptually."""
        # vLLM provides faster batched generation
        prompts = ["Prompt 1", "Prompt 2"]
        num_generations = 4

        # Would generate num_generations per prompt
        expected_outputs = len(prompts) * num_generations
        assert expected_outputs == 8


class TestGSPOTokenVariant:
    """Test GSPO-token variant."""

    def test_gspo_token_config(self):
        """Test GSPO-token variant configuration."""
        from training.gspo_trainer import GSPOConfig

        config = GSPOConfig(use_gspo_token=True)
        assert config.use_gspo_token is True

    def test_token_level_advantages(self):
        """Test token-level advantage computation for GSPO-token."""
        batch_size = 4
        seq_len = 50

        # Token-level log probs
        log_probs = torch.randn(batch_size, seq_len)

        # Token-level rewards (e.g., from reward model)
        token_rewards = torch.randn(batch_size, seq_len)

        # Token-level advantages
        advantages = token_rewards  # Simplified

        assert advantages.shape == (batch_size, seq_len)


class TestGSPOTrainer:
    """Test GSPOTrainer class."""

    @pytest.fixture
    def gspo_config(self):
        """Create a GSPO config for testing."""
        from training.gspo_trainer import GSPOConfig
        return GSPOConfig(
            model_name="gpt2",
            num_generations=4,
            num_iterations=2,
        )

    def test_trainer_config_valid(self, gspo_config):
        """Test trainer config is valid."""
        assert gspo_config.model_name == "gpt2"
        assert gspo_config.num_generations == 4


class TestGSPOTrainerComputeGroupAdvantages:
    """Tests for GSPOTrainer.compute_group_advantages edge cases."""

    def test_constant_rewards_no_item_attribute_error(self):
        """Constant rewards should not raise and should return zero advantages."""
        from training.gspo_trainer import GSPOTrainer

        rewards = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        advantages, stats = GSPOTrainer.compute_group_advantages(None, rewards)

        assert torch.allclose(advantages, torch.zeros_like(rewards))
        assert stats["mean_reward"] == 1.0
        assert stats["std_reward"] == 0.0

    def test_single_reward_no_nan(self):
        """Single-element groups should not produce NaNs."""
        from training.gspo_trainer import GSPOTrainer

        rewards = torch.tensor([2.0], dtype=torch.float32)
        advantages, stats = GSPOTrainer.compute_group_advantages(None, rewards)

        assert not torch.isnan(advantages).any()
        assert torch.allclose(advantages, torch.zeros_like(rewards))
        assert stats["mean_reward"] == 2.0


class TestGSPOTrainerSequenceLogProbs:
    """Tests for GSPOTrainer._compute_group_sequence_log_probs."""

    class _DummyTokenizer:
        def __init__(self, vocab_size: int = 64):
            self.vocab_size = vocab_size
            self.pad_token_id = 0
            self.padding_side = "right"

        def __call__(
            self,
            texts,
            return_tensors: str = "pt",
            padding: bool = False,
            truncation: bool = False,
            max_length: int = 128,
            add_special_tokens: bool = False,
        ):
            if isinstance(texts, str):
                texts = [texts]

            encoded = []
            for text in texts:
                ids = [((ord(ch) % (self.vocab_size - 1)) + 1) for ch in text]
                if truncation and max_length is not None:
                    ids = ids[:max_length]
                encoded.append(torch.tensor(ids, dtype=torch.long))

            if padding:
                max_len = max((t.numel() for t in encoded), default=0)
                input_ids = torch.full(
                    (len(encoded), max_len), self.pad_token_id, dtype=torch.long
                )
                attention_mask = torch.zeros_like(input_ids)
                for i, ids in enumerate(encoded):
                    length = ids.numel()
                    if length == 0:
                        continue
                    if self.padding_side == "right":
                        input_ids[i, :length] = ids
                        attention_mask[i, :length] = 1
                    else:
                        input_ids[i, -length:] = ids
                        attention_mask[i, -length:] = 1
            else:
                input_ids = torch.stack(encoded, dim=0) if encoded else torch.empty(0)
                attention_mask = torch.ones_like(input_ids)

            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class _DummyCausalLM(nn.Module):
        def __init__(self, vocab_size: int = 64, hidden_size: int = 16):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        def forward(self, input_ids, attention_mask=None):
            hidden = self.embed(input_ids)
            logits = self.lm_head(hidden)
            return SimpleNamespace(logits=logits)

    def test_log_probs_require_grad(self):
        """Sequence log probs should require grad for policy optimization."""
        from training.gspo_trainer import GSPOTrainer, GSPOConfig

        model = self._DummyCausalLM()
        tokenizer = self._DummyTokenizer()
        config = GSPOConfig(max_prompt_length=64, max_completion_length=64)

        trainer = SimpleNamespace(model=model, tokenizer=tokenizer, config=config)
        log_probs, lengths = GSPOTrainer._compute_group_sequence_log_probs(
            trainer, "Hello", ["world", "there"]
        )

        assert log_probs.shape == (2,)
        assert lengths.shape == (2,)
        assert log_probs.requires_grad is True

        log_probs.sum().backward()
        assert any(p.grad is not None for p in model.parameters())


class TestGSPOMetrics:
    """Test GSPO training metrics."""

    def test_reward_metrics(self):
        """Test reward metrics computation."""
        rewards = torch.tensor([0.5, 0.7, 0.3, 0.9])

        mean_reward = rewards.mean()
        std_reward = rewards.std()
        max_reward = rewards.max()
        min_reward = rewards.min()

        assert 0.5 < mean_reward < 0.7
        assert std_reward > 0
        assert max_reward == 0.9
        assert min_reward == 0.3

    def test_ratio_metrics(self):
        """Test importance ratio metrics."""
        ratios = torch.tensor([0.95, 1.0, 1.05, 0.98])

        mean_ratio = ratios.mean()
        max_ratio = ratios.max()
        min_ratio = ratios.min()

        # Ratios should be around 1
        assert 0.9 < mean_ratio < 1.1

    def test_kl_metrics(self):
        """Test KL divergence metrics."""
        batch_size = 10
        seq_len = 50

        log_probs_current = torch.randn(batch_size, seq_len)
        log_probs_ref = torch.randn(batch_size, seq_len)

        # Approximate KL per sequence
        kl_per_seq = (log_probs_current - log_probs_ref).sum(dim=1)

        mean_kl = kl_per_seq.mean()
        max_kl = kl_per_seq.max()

        assert mean_kl.dim() == 0
        assert max_kl.dim() == 0
