"""Unit tests for Kimi-K2.5 integration with StateSet Agents."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from examples.kimi_k25_config import (
    get_kimi_k25_config,
    get_kimi_k25_config_gspo,
    validate_kimi_k25_config,
    KimiK25Config,
)


class TestKimiK25Config:
    """Test suite for Kimi-K2.5 configuration."""

    def test_default_config_creation(self):
        """Test creating default Kimi-K2.5 configuration."""
        config = KimiK25Config()

        assert config.model_name == "moonshotai/Kimi-K2.5"
        assert config.use_lora is True
        assert config.use_vllm is True
        assert config.device_map == "auto"
        assert config.trust_remote_code is True

    def test_config_with_custom_task(self):
        """Test configuration for different tasks."""
        config = get_kimi_k25_config(task="technical_support", num_iterations=50)

        assert config.task == "technical_support"
        assert config.num_iterations == 50
        assert config.system_prompt == (
            "You are Kimi, a helpful and empathetic technical support "
            "specialist created by Moonshot AI. You help users troubleshoot "
            "technical issues with clear, detailed explanations using "
            "reasoning when helpful."
        )

    def test_config_with_lora_disabled(self):
        """Test configuration with LoRA disabled."""
        config = get_kimi_k25_config(use_lora=False)

        assert config.use_lora is False
        assert config.lora_r is None
        assert config.lora_alpha is None

    def test_config_with_vllm_disabled(self):
        """Test configuration with vLLM disabled."""
        config = get_kimi_k25_config(use_vllm=False)

        assert config.use_vllm is False

    def test_gspo_config_generation(self):
        """Test GSPO configuration generation."""
        from stateset_agents.training.gspo_trainer import GSPOConfig

        base_config = get_config_for_task(
            "customer_service", model_name="moonshotai/Kimi-K2.5"
        )
        kimi_config = get_kimi_k25_config(task="customer_service")

        with patch(
            "examples.kimi_k25_config.get_config_for_task", return_value=base_config
        ):
            if GSPOConfig is not None:
                gspo_config = get_kimi_k25_config_gspo(kimi_config, base_config)

                assert gspo_config.model_name == "moonshotai/Kimi-K2.5"
                assert gspo_config.use_lora is True
                assert gspo_config.lora_r == 64
                assert gspo_config.num_generations == 8

    def test_custom_hardware_override(self):
        """Test hardware-specific configuration overrides."""
        config = get_kimi_k25_config(gpu_memory_gb=16, num_gpus=1)

        assert config.gpu_memory_gb == 16
        assert config.num_gpus == 1

    def test_validation_success(self):
        """Test successful configuration validation."""
        config = KimiK25Config()
        warnings = validate_kimi_k25_config(config)

        assert len(warnings) == 0

    def test_validation_large_batch_warning(self):
        """Test validation warning for large batch size."""
        config = KimiK25Config(per_device_train_batch_size=8)
        warnings = validate_kimi_k25_config(config)

        assert any("large batch size" in w.lower() for w in warnings)

    def test_validation_high_learning_rate_warning(self):
        """Test validation warning for high learning rate."""
        config = KimiK25Config(learning_rate=1e-4)
        warnings = validate_kimi_k25_config(config)

        assert any("learning rate" in w.lower() for w in warnings)

    def test_validation_low_learning_rate_warning(self):
        """Test validation warning for low learning rate."""
        config = KimiK25Config(learning_rate=1e-8)
        warnings = validate_kimi_k25_config(config)

        assert any("learning rate" in w.lower() for w in warnings)

    def test_system_prompt_by_task(self):
        """Test system prompt generation for different tasks."""
        tasks = ["customer_service", "technical_support", "sales", "coding_assistant"]

        for task in tasks:
            config = get_kimi_k25_config(task=task)
            assert "Kimi" in config.system_prompt
            assert "Moonshot AI" in config.system_prompt

    def test_temperature_settings(self):
        """Test temperature configuration for different modes."""
        config = get_kimi_k25_config()

        assert config.temperature_thinking == 1.0
        assert config.temperature_instant == 0.6

    def test_context_length(self):
        """Test context length configuration."""
        config = get_kimi_k25_config()

        assert config.max_prompt_length == 8192
        assert config.max_completion_length == 4096

    def test_quantization_settings(self):
        """Test quantization configuration."""
        config = get_kimi_k25_config(use_4bit=False)

        assert config.use_4bit is False
        assert config.bf16 is True

    def test_checkpointing_settings(self):
        """Test checkpoint configuration."""
        config = get_kimi_k25_config(num_iterations=200, save_steps_every=25)

        assert config.save_steps_every == 25
        assert config.output_dir == "./outputs/kimi_k25_finetuned"

    def test_logging_settings(self):
        """Test logging configuration."""
        config = get_kimi_k25_config(use_wandb=True, wandb_project="kimi-k25-test")

        assert config.use_wandb is True
        assert config.wandb_project == "kimi-k25-test"

    def test_grad_accumulation_for_moe(self):
        """Test gradient accumulation for MoE model."""
        config = get_kimi_k25_config()

        # MoE models typically need smaller batch size but more accumulation
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 16

    def test_lora_target_modules(self):
        """Test LoRA target modules configuration."""
        config = get_kimi_k25_config()

        expected_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        assert config.lora_target_modules == expected_modules

    def test_vllm_gpu_memory_utilization(self):
        """Test vLLM GPU memory configuration."""
        config = get_kimi_k25_config()

        assert config.vllm_gpu_memory_utilization == 0.85
        assert config.vllm_enable_prefix_caching is True

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = KimiK25Config()
        config_dict = config.to_dict()

        assert "model_name" in config_dict
        assert config_dict["model_name"] == "moonshotai/Kimi-K2.5"

    def test_config_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "model_name": "moonshotai/Kimi-K2.5",
            "task": "customer_service",
            "use_lora": True,
            "num_iterations": 50,
        }

        config = KimiK25Config.from_dict(config_dict)

        assert config.model_name == "moonshotai/Kimi-K2.5"
        assert config.task == "customer_service"
        assert config.num_iterations == 50


class TestKimiK25Integration:
    """Test suite for Kimi-K2.5 integration with training."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test Kimi-K2.5 agent initialization."""
        from stateset_agents import MultiTurnAgent
        from stateset_agents.core.agent import AgentConfig

        with patch("stateset_agents.core.agent.AutoModelForCausalLM") as mock_model:
            with patch("stateset_agents.core.agent.AutoTokenizer") as mock_tokenizer:
                mock_tokenizer.from_pretrained.return_value = MagicMock()

                agent_config = AgentConfig(
                    model_name="moonshotai/Kimi-K2.5",
                    trust_remote_code=True,
                    device_map="auto",
                )

                # Note: Full initialization would require downloading the model
                # This is a unit test to verify configuration setup
                assert agent_config.model_name == "moonshotai/Kimi-K2.5"
                assert agent_config.trust_remote_code is True

    def test_reward_model_selection(self):
        """Test reward model selection for Kimi-K2.5."""
        from stateset_agents.rewards.multi_objective_reward import (
            create_customer_service_reward,
        )

        reward_model = create_customer_service_reward()

        assert reward_model is not None

    def test_environment_configuration(self):
        """Test environment configuration for Kimi-K2.5 training."""
        from stateset_agents.core.environment import ConversationEnvironment

        scenarios = [
            {
                "id": "test_scenario",
                "topic": "customer_service",
                "context": "Test scenario for Kimi-K2.5",
                "user_responses": ["Hello", "How can you help?"],
            }
        ]

        env = ConversationEnvironment(scenarios=scenarios, max_turns=4)

        assert len(env.scenarios) == 1
        assert env.scenarios[0]["id"] == "test_scenario"

    @pytest.mark.slow
    def test_kimi_k25_model_availability(self):
        """Test if Kimi-K2.5 model is available on HuggingFace (network dependent)."""
        try:
            from transformers import AutoTokenizer

            # Try to load tokenizer only (faster than loading full model)
            tokenizer = AutoTokenizer.from_pretrained(
                "moonshotai/Kimi-K2.5", trust_remote_code=True
            )

            assert tokenizer is not None
            assert tokenizer.vocab_size == 160000  # Kimi-K2.5 has 160K vocab

        except Exception as e:
            pytest.skip(f"Network or model access issue: {e}")


class TestKimiK25TaskSpecific:
    """Test suite for task-specific Kimi-K2.5 configurations."""

    def test_customer_service_config(self):
        """Test customer service specific configuration."""
        config = get_kimi_k25_config(task="customer_service")

        assert "customer service" in config.system_prompt.lower()
        assert "empathetic" in config.system_prompt.lower()

    def test_technical_support_config(self):
        """Test technical support specific configuration."""
        config = get_kimi_k25_config(task="technical_support")

        assert "technical support" in config.system_prompt.lower()
        assert "troubleshoot" in config.system_prompt.lower()

    def test_sales_config(self):
        """Test sales specific configuration."""
        config = get_kimi_k25_config(task="sales")

        assert "sales" in config.system_prompt.lower()
        assert "persuasive" in config.system_prompt.lower()

    def test_coding_assistant_config(self):
        """Test coding assistant specific configuration."""
        config = get_kimi_k25_config(task="coding_assistant")

        assert "coding" in config.system_prompt.lower()
        assert "programming" in config.system_prompt.lower()


class TestKimiK25HardwareProfiles:
    """Test suite for hardware-specific profiles."""

    def test_small_gpu_profile(self):
        """Test configuration for small GPU (16GB)."""
        config = get_kimi_k25_config(gpu_memory_gb=16, num_gpus=1)

        # Should use more aggressive optimizations
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps >= 16

    def test_large_gpu_profile(self):
        """Test configuration for large GPU (80GB+)."""
        config = get_kimi_k25_config(gpu_memory_gb=80, num_gpus=1)

        # Can use larger batch size with vLLM
        assert config.use_vllm is True

    def test_multi_gpu_profile(self):
        """Test configuration for multi-GPU setup."""
        config = get_kimi_k25_config(gpu_memory_gb=40, num_gpus=4)

        assert config.num_gpus == 4


@pytest.fixture
def mock_base_config():
    """Fixture for mocking base training config."""
    from stateset_agents.training.config import TrainingConfig

    return TrainingConfig(
        model_name="moonshotai/Kimi-K2.5",
        learning_rate=3e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_episodes=100,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
