"""
Unit tests for Kimi-K2.5 integration

These tests verify the configuration and setup functions work correctly.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from examples.kimi_k25_config import (
    get_kimi_k25_config,
    get_kimi_k25_conversational_config,
    get_kimi_k25_vision_config,
    KIMI_K25_SUPPORTED_VARIANTS,
    KIMI_K25_MODEL_SPECS,
)


class TestKimiK25Config:
    """Test Kimi-K2.5 configuration functions."""

    def test_supported_variants(self):
        """Test that supported variants are defined."""
        assert len(KIMI_K25_SUPPORTED_VARIANTS) > 0
        assert "moonshotai/Kimi-K2.5" in KIMI_K25_SUPPORTED_VARIANTS

    def test_model_specs(self):
        """Test model specifications."""
        assert "moonshotai/Kimi-K2.5" in KIMI_K25_MODEL_SPECS
        spec = KIMI_K25_MODEL_SPECS["moonshotai/Kimi-K2.5"]
        assert spec["total_params"] == "1T"
        assert spec["activated_params"] == "32B"
        assert spec["architecture"] == "MoE"

    def test_get_kimi_k25_config_default(self):
        """Test default configuration."""
        config = get_kimi_k25_config()
        assert config.model_name == "moonshotai/Kimi-K2.5"
        assert config.use_lora is True
        assert config.use_vllm is True
        assert config.num_generations == 8

    def test_get_kimi_k25_config_with_quantization(self):
        """Test configuration with 4-bit quantization."""
        config = get_kimi_k25_config(use_4bit=True)
        assert config.use_4bit is True
        assert config.use_8bit is False
        assert config.lora_r == 128  # Lower rank for quantized

    def test_get_kimi_k25_config_without_vllm(self):
        """Test configuration without vLLM."""
        config = get_kimi_k25_config(use_vllm=False)
        assert config.use_vllm is False
        assert config.vllm_gpu_memory_utilization == 0.85  # Default value

    def test_get_kimi_k25_conversational_config(self):
        """Test conversational-specific configuration."""
        config = get_kimi_k25_conversational_config()
        assert config.max_completion_length == 2048
        assert config.max_prompt_length == 4096
        assert config.temperature == 0.7
        assert "conversational" in config.wandb_tags

    def test_get_kimi_k25_vision_config(self):
        """Test vision-specific configuration."""
        config = get_kimi_k25_vision_config()
        assert config.max_completion_length == 4096  # Longer for vision
        assert config.max_prompt_length == 8192
        assert "vision" in config.wandb_tags
        assert temperature == 1.0  # Higher for vision

    def test_custom_overrides(self):
        """Test that custom overrides are applied."""
        config = get_kimi_k25_config(
            learning_rate=1e-5,
            num_outer_iterations=200,
            lora_r=128,
        )
        assert config.learning_rate == 1e-5
        assert config.num_outer_iterations == 200
        assert config.lora_r == 128

    def test_model_name_validation(self):
        """Test that unsupported model names raise error."""
        with pytest.raises(ValueError, match="Unsupported Kimi-K2.5 model"):
            get_kimi_k25_config(model_name="unsupported/model")


class TestKimiK25Training:
    """Test Kimi-K2.5 training setup."""

    @patch("examples.kimi_k25_config.MultiTurnAgent")
    @patch("examples.kimi_k25_config.ConversationEnvironment")
    @patch("examples.kimi_k25_config.create_domain_reward")
    @patch("examples.kimi_k25_config.train_with_gspo")
    async def test_setup_kimi_k25_training(
        self, mock_train, mock_reward, mock_env, mock_agent
    ):
        """Test training setup function."""
        from examples.kimi_k25_config import setup_kimi_k25_training

        # Mock returns
        mock_agent_instance = Mock()
        mock_agent.return_value = mock_agent_instance
        mock_agent_instance.initialize = Mock(return_value=None)

        mock_env_instance = Mock()
        mock_env.return_value = mock_env_instance

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0

        mock_trained_agent = Mock()
        mock_train.return_value = mock_trained_agent

        # Call setup
        agent, env, reward_model, gspo_config = await setup_kimi_k25_training(
            task="customer_service",
            model_name="moonshotai/Kimi-K2.5",
            use_lora=True,
            use_vllm=True,
        )

        # Verify calls
        mock_agent.assert_called_once()
        mock_env.assert_called_once()
        mock_reward.assert_called_once_with("customer_service")
        mock_train.assert_called_once()

        # Verify returns
        assert agent is not None
        assert env is not None
        assert reward_model is not None
        assert gspo_config is not None

    def test_compute_effective_batch_size(self):
        """Test effective batch size calculation."""
        config = get_kimi_k25_config(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
        )
        effective = config.get_effective_batch_size()
        assert effective == 16

    def test_config_validation(self):
        """Test configuration validation."""
        config = get_kimi_k25_config()
        warnings = config.validate()
        # Should not have critical warnings
        assert len(warnings) == 0

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = get_kimi_k25_config()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "model_name" in config_dict
        assert "learning_rate" in config_dict
        assert config_dict["model_name"] == "moonshotai/Kimi-K2.5"


class TestKimiK25HardwareOptimization:
    """Test hardware-specific optimizations."""

    def test_large_gpu_optimization(self):
        """Test configuration for large GPU (80GB)."""
        config = get_kimi_k25_config(gpu_memory_gb=80)
        assert config.per_device_train_batch_size == 1  # Still 1 for MoE
        assert config.gradient_accumulation_steps == 8

    def test_medium_gpu_optimization(self):
        """Test configuration for medium GPU (40GB)."""
        config = get_kimi_k25_config(gpu_memory_gb=40)
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 16

    def test_small_gpu_fallback(self):
        """Test that small GPU triggers quantization warning."""
        config = get_kimi_k25_config(gpu_memory_gb=24, use_4bit=False)
        warnings = config.validate()
        # Should warn about small GPU without quantization
        assert any("GPU memory" in str(w) for w in warnings)

    def test_multi_gpu_optimization(self):
        """Test multi-GPU configuration."""
        config = get_kimi_k25_config(num_gpus=4, gpu_memory_gb=80)
        assert config.gradient_accumulation_steps >= 16  # Adjusted for multi-GPU


class TestKimiK25WandbIntegration:
    """Test Weights & Biases integration."""

    def test_wandb_enabled_by_default(self):
        """Test that W&B is enabled by default."""
        config = get_kimi_k25_config()
        assert config.report_to == "wandb"
        assert config.wandb_project is not None

    def test_wandb_can_be_disabled(self):
        """Test that W&B can be disabled."""
        config = get_kimi_k25_config(wandb_enabled=False)
        assert config.report_to == "none"

    def test_wandb_custom_project(self):
        """Test custom W&B project name."""
        config = get_kimi_k25_config(
            wandb_project="my-custom-project",
            wandb_entity="my-entity",
        )
        assert config.wandb_project == "my-custom-project"
        assert config.wandb_entity == "my-entity"

    def testwandb_tags_include_model_info(self):
        """Test that W&B tags include model information."""
        config = get_kimi_k25_conversational_config()
        assert "kimi-k25" in config.wandb_tags
        assert "conversational" in config.wandb_tags
        assert "gspo" in config.wandb_tags


class TestKimiK25EdgeCases:
    """Test edge cases and error handling."""

    def test_zero_iterations(self):
        """Test configuration with zero iterations (should fail validation)."""
        config = get_kimi_k25_config(num_outer_iterations=0)
        warnings = config.validate()
        # Should warn about zero iterations
        assert len(warnings) > 0

    def test_negative_learning_rate(self):
        """Test negative learning rate (should fail validation)."""
        config = get_kimi_k25_config(learning_rate=-1e-6)
        warnings = config.validate()
        # Should warn about negative learning rate
        assert any("learning rate" in str(w).lower() for w in warnings)

    def test_large_lora_rank(self):
        """Test very large LoRA rank."""
        config = get_kimi_k25_config(lora_r=512)
        assert config.lora_r == 512
        # Large rank should not fail, but may be inefficient

    def test_no_lora_without_quantization(self):
        """Test no LoRA without quantization (should use LoRA)."""
        config = get_kimi_k25_config(use_lora=False, use_4bit=False)
        warnings = config.validate()
        # Should recommend LoRA for large models
        assert any("lora" in str(w).lower() for w in warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
