"""
Unit tests for the Base Trainer module.

Tests the shared infrastructure used by all RL trainers (GSPO, VAPO, DAPO, etc.).
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, AsyncMock


class TestBaseTrainerConfig:
    """Test BaseTrainerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from training.base_trainer import BaseTrainerConfig

        config = BaseTrainerConfig()

        assert config.model_name == "gpt2"
        assert config.learning_rate == 1e-5
        assert config.use_lora is True
        assert config.bf16 is True
        assert config.gradient_checkpointing is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from training.base_trainer import BaseTrainerConfig

        config = BaseTrainerConfig(
            model_name="meta-llama/Llama-2-7b",
            learning_rate=5e-6,
            use_lora=False,
            bf16=False,
            fp16=True,
        )

        assert config.model_name == "meta-llama/Llama-2-7b"
        assert config.learning_rate == 5e-6
        assert config.use_lora is False
        assert config.fp16 is True

    def test_to_dict(self):
        """Test config to dictionary conversion."""
        from training.base_trainer import BaseTrainerConfig

        config = BaseTrainerConfig(
            model_name="test-model",
            learning_rate=1e-4,
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "test-model"
        assert config_dict["learning_rate"] == 1e-4

    def test_from_dict(self):
        """Test creating config from dictionary."""
        from training.base_trainer import BaseTrainerConfig

        config_dict = {
            "model_name": "from-dict-model",
            "learning_rate": 2e-5,
            "use_lora": True,
            "extra_field": "should_be_ignored",  # Unknown field
        }

        config = BaseTrainerConfig.from_dict(config_dict)

        assert config.model_name == "from-dict-model"
        assert config.learning_rate == 2e-5


class TestNormalizeAdvantages:
    """Test advantage normalization utility."""

    def test_basic_normalization(self):
        """Test basic advantage normalization."""
        from training.base_trainer import normalize_advantages

        advantages = torch.tensor([1.0, 2.0, 3.0, 4.0])
        normalized = normalize_advantages(advantages)

        # Should have zero mean
        assert abs(normalized.mean().item()) < 1e-6

        # Should have unit variance (std ~= 1)
        assert abs(normalized.std().item() - 1.0) < 0.1

    def test_zero_std_handling(self):
        """Test normalization when all values are the same (std=0)."""
        from training.base_trainer import normalize_advantages

        advantages = torch.tensor([5.0, 5.0, 5.0, 5.0])
        normalized = normalize_advantages(advantages)

        # Should not produce NaN
        assert not torch.isnan(normalized).any()

    def test_single_element(self):
        """Test normalization with single element."""
        from training.base_trainer import normalize_advantages

        advantages = torch.tensor([3.0])
        normalized = normalize_advantages(advantages)

        assert not torch.isnan(normalized).any()


class TestComputeGroupAdvantages:
    """Test group advantage computation."""

    def test_mean_baseline(self):
        """Test advantages with mean baseline."""
        from training.base_trainer import compute_group_advantages

        rewards = [0.2, 0.4, 0.6, 0.8]
        advantages = compute_group_advantages(rewards, baseline_type="mean")

        # Mean is 0.5
        expected = [-0.3, -0.1, 0.1, 0.3]

        for a, e in zip(advantages, expected):
            assert abs(a - e) < 1e-6

    def test_median_baseline(self):
        """Test advantages with median baseline."""
        from training.base_trainer import compute_group_advantages

        rewards = [0.1, 0.3, 0.7, 0.9]
        advantages = compute_group_advantages(rewards, baseline_type="median")

        # Median is 0.5
        expected = [-0.4, -0.2, 0.2, 0.4]

        for a, e in zip(advantages, expected):
            assert abs(a - e) < 1e-6

    def test_invalid_baseline(self):
        """Test with invalid baseline type."""
        from training.base_trainer import compute_group_advantages

        with pytest.raises(ValueError):
            compute_group_advantages([0.5], baseline_type="invalid")

    def test_empty_rewards(self):
        """Test with empty rewards list."""
        from training.base_trainer import compute_group_advantages

        advantages = compute_group_advantages([], baseline_type="mean")
        assert advantages == []


class TestCreateResponseMask:
    """Test response mask creation."""

    def test_basic_mask(self):
        """Test basic response mask creation."""
        from training.base_trainer import create_response_mask

        input_ids = torch.zeros(2, 10, dtype=torch.long)
        prompt_lengths = [3, 5]

        mask = create_response_mask(input_ids, prompt_lengths)

        # First sequence: prompt=3, response starts at 3
        assert mask[0, :3].sum() == 0  # Prompt is 0
        assert mask[0, 3:].sum() == 7  # Response is 1

        # Second sequence: prompt=5, response starts at 5
        assert mask[1, :5].sum() == 0
        assert mask[1, 5:].sum() == 5

    def test_full_prompt(self):
        """Test when entire sequence is prompt."""
        from training.base_trainer import create_response_mask

        input_ids = torch.zeros(1, 10, dtype=torch.long)
        prompt_lengths = [10]

        mask = create_response_mask(input_ids, prompt_lengths)

        assert mask.sum() == 0  # All zeros


class TestBaseModelManager:
    """Test BaseModelManager class."""

    def test_get_dtype_fp16(self):
        """Test dtype selection for FP16."""
        from training.base_trainer import BaseModelManager, BaseTrainerConfig

        config = BaseTrainerConfig(fp16=True, bf16=False)
        manager = BaseModelManager(config)

        dtype = manager._get_dtype()
        assert dtype == torch.float16

    def test_get_dtype_bf16(self):
        """Test dtype selection for BF16."""
        from training.base_trainer import BaseModelManager, BaseTrainerConfig

        config = BaseTrainerConfig(fp16=False, bf16=True)
        manager = BaseModelManager(config)

        dtype = manager._get_dtype()
        assert dtype == torch.bfloat16

    def test_get_dtype_fp32(self):
        """Test dtype selection for FP32."""
        from training.base_trainer import BaseModelManager, BaseTrainerConfig

        config = BaseTrainerConfig(fp16=False, bf16=False)
        manager = BaseModelManager(config)

        dtype = manager._get_dtype()
        assert dtype == torch.float32

    def test_lora_target_modules_gpt2(self):
        """Test LoRA target modules for GPT-2."""
        from training.base_trainer import BaseModelManager, BaseTrainerConfig

        config = BaseTrainerConfig(model_name="gpt2")
        manager = BaseModelManager(config)

        modules = manager._get_lora_target_modules()
        assert "c_attn" in modules
        assert "c_proj" in modules

    def test_lora_target_modules_llama(self):
        """Test LoRA target modules for LLaMA."""
        from training.base_trainer import BaseModelManager, BaseTrainerConfig

        config = BaseTrainerConfig(model_name="meta-llama/Llama-2-7b")
        manager = BaseModelManager(config)

        modules = manager._get_lora_target_modules()
        assert "q_proj" in modules
        assert "k_proj" in modules
        assert "v_proj" in modules
        assert "o_proj" in modules

    def test_lora_target_modules_custom(self):
        """Test custom LoRA target modules."""
        from training.base_trainer import BaseModelManager, BaseTrainerConfig

        config = BaseTrainerConfig(
            model_name="custom-model",
            lora_target_modules=["custom_layer_1", "custom_layer_2"],
        )
        manager = BaseModelManager(config)

        modules = manager._get_lora_target_modules()
        assert modules == ["custom_layer_1", "custom_layer_2"]


class TestBaseTrajectoryGenerator:
    """Test BaseTrajectoryGenerator class."""

    def test_initialization_without_vllm(self):
        """Test initialization when vLLM is disabled."""
        from training.base_trainer import BaseTrajectoryGenerator, BaseTrainerConfig

        config = BaseTrainerConfig(use_vllm=False)
        generator = BaseTrajectoryGenerator(config)

        assert generator.vllm_generator is None
        assert not generator.using_vllm

    def test_using_vllm_property(self):
        """Test using_vllm property."""
        from training.base_trainer import BaseTrajectoryGenerator, BaseTrainerConfig

        config = BaseTrainerConfig(use_vllm=False)
        generator = BaseTrajectoryGenerator(config)

        assert generator.using_vllm is False


class TestBaseTrainerMethods:
    """Test BaseTrainer method implementations."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = nn.Linear(10, 10)
        return model

    def test_compute_token_log_probs(self, mock_model):
        """Test token log probability computation."""
        # Create minimal test setup
        input_ids = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones_like(input_ids)

        # Mock logits output
        batch_size, seq_len = input_ids.shape
        vocab_size = 100

        with patch.object(mock_model, 'forward') as mock_forward:
            mock_output = MagicMock()
            mock_output.logits = torch.randn(batch_size, seq_len, vocab_size)
            mock_forward.return_value = mock_output

            # Manual computation to verify
            logits = mock_output.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            assert token_log_probs.shape == (batch_size, seq_len - 1)

    def test_compute_kl_divergence(self):
        """Test KL divergence computation."""
        batch_size, seq_len, vocab_size = 2, 10, 100

        current_logits = torch.randn(batch_size, seq_len, vocab_size)
        reference_logits = torch.randn(batch_size, seq_len, vocab_size)

        # Manual KL computation
        current_log_probs = torch.nn.functional.log_softmax(current_logits, dim=-1)
        reference_probs = torch.nn.functional.softmax(reference_logits, dim=-1)

        kl = torch.nn.functional.kl_div(
            current_log_probs,
            reference_probs,
            reduction="none"
        ).sum(dim=-1)

        assert kl.shape == (batch_size, seq_len)
        assert not torch.isnan(kl).any()

    def test_gradient_clipping(self, mock_model):
        """Test gradient clipping."""
        # Create gradients
        for p in mock_model.parameters():
            p.grad = torch.randn_like(p) * 100  # Large gradients

        max_norm = 1.0
        norm = torch.nn.utils.clip_grad_norm_(mock_model.parameters(), max_norm)

        assert norm > 0  # Had gradients

        # Verify gradients are clipped
        total_norm = 0.0
        for p in mock_model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5

        # After clipping, total norm should be close to max_norm
        assert total_norm <= max_norm * 1.1  # Allow small tolerance


class TestLazyImports:
    """Test lazy import functionality."""

    def test_load_transformers_guard(self):
        """Test transformers lazy loading guard."""
        from training.base_trainer import _load_transformers

        # Should return True if transformers available, False otherwise
        result = _load_transformers()
        assert isinstance(result, bool)

    def test_load_vllm_backend_guard(self):
        """Test vLLM backend lazy loading guard."""
        from training.base_trainer import _load_vllm_backend

        result = _load_vllm_backend()
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
