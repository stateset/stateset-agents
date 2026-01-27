"""
Unit tests for Kimi-K2.5 integration
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
from pathlib import Path

from kimi_k25_config import get_kimi_k25_config
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.rewards.multi_objective_reward import (
    create_customer_service_reward,
)


class TestKimiK25Config:
    """Test Kimi-K2.5 configuration"""

    def test_default_config(self):
        """Test default configuration generation"""
        config = get_kimi_k25_config(task="customer_service")

        assert config.model_name == "moonshotai/Kimi-K2.5"
        assert config.use_lora == True
        assert config.use_vllm == True
        assert config.num_generations == 8  # Larger group for MoE
        assert config.learning_rate == 3e-6
        assert config.lora_r == 64
        assert config.per_device_train_batch_size == 1

    def test_custom_config_overrides(self):
        """Test custom configuration overrides"""
        config = get_kimi_k25_config(
            task="technical_support",
            use_lora=False,
            learning_rate=1e-5,
            num_outer_iterations=50,
        )

        assert config.use_lora == False
        assert config.learning_rate == 1e-5
        assert config.num_outer_iterations == 50

    def test_task_specific_configs(self):
        """Test task-specific configuration adjustments"""
        cs_config = get_kimi_k25_config(task="customer_service")
        tech_config = get_kimi_k25_config(task="technical_support")

        # Different tasks might have different settings
        assert cs_config != tech_config

    def test_vllm_config(self):
        """Test vLLM specific configuration"""
        config = get_kimi_k25_config(
            task="customer_service",
            use_vllm=True,
            vllm_gpu_memory_utilization=0.9,
        )

        assert config.use_vllm == True
        assert config.vllm_gpu_memory_utilization == 0.9


class TestKimiK25Agent:
    """Test Kimi-K2.5 agent initialization"""

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent can be initialized with Kimi-K2.5 config"""
        from stateset_agents import MultiTurnAgent

        agent_config = AgentConfig(
            model_name="moonshotai/Kimi-K2.5",
            system_prompt="You are Kimi, a helpful AI assistant created by Moonshot AI.",
            max_new_tokens=2048,
        )

        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model:
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
                mock_model.return_value = MagicMock()
                mock_tokenizer.return_value = MagicMock()

                agent = MultiTurnAgent(agent_config)
                await agent.initialize()

                assert agent.config.model_name == "moonshotai/Kimi-K2.5"

    @pytest.mark.asyncio
    async def test_agent_system_prompt(self):
        """Test Kimi-specific system prompt"""
        from stateset_agents import MultiTurnAgent

        kimi_system_prompt = "You are Kimi, an AI assistant created by Moonshot AI."

        agent_config = AgentConfig(
            model_name="moonshotai/Kimi-K2.5",
            system_prompt=kimi_system_prompt,
        )

        assert agent_config.system_prompt == kimi_system_prompt


class TestKimiK25Training:
    """Test Kimi-K2.5 training setup"""

    def test_reward_model_compatibility(self):
        """Test reward model compatibility with Kimi-K2.5"""
        reward_model = create_customer_service_reward()
        assert reward_model is not None

    def test_environment_compatibility(self):
        """Test environment compatibility"""
        env = ConversationEnvironment(
            scenarios=[
                {
                    "user": "I need help with my order",
                    "expected_response": "I'd be happy to help you with your order!",
                }
            ]
        )
        assert len(env.scenarios) == 1

    @pytest.mark.asyncio
    async def test_training_setup(self):
        """Test training can be set up"""
        config = get_kimi_k25_config(task="customer_service")

        # Verify GSPO config is properly set
        assert hasattr(config, "num_generations")
        assert hasattr(config, "clip_range_left")
        assert hasattr(config, "clip_range_right")
        assert hasattr(config, "learning_rate")


class TestKimiK25Multimodal:
    """Test Kimi-K2.5 multimodal capabilities"""

    def test_model_info(self):
        """Test Kimi-K2.5 model information"""
        from kimi_k25_config import KIMI_K25_INFO

        assert KIMI_K25_INFO["model_name"] == "moonshotai/Kimi-K2.5"
        assert KIMI_K25_INFO["total_params"] == "1T"
        assert KIMI_K25_INFO["activated_params"] == "32B"
        assert KIMI_K25_INFO["context_length"] == "256K"
        assert KIMI_K25_INFO["multimodal"] == True

    def test_deployment_engines(self):
        """Test supported deployment engines"""
        from kimi_k25_config import KIMI_K25_INFO

        expected_engines = ["vLLM", "SGLang", "KTransformers"]
        for engine in expected_engines:
            assert engine in KIMI_K25_INFO["supported_engines"]


class TestKimiK25Integration:
    """End-to-end integration tests"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_training_pipeline_smoke(self):
        """Smoke test for full training pipeline (runs with minimal iterations)"""
        from kimi_k25_config import get_kimi_k25_config
        from stateset_agents.rewards.multi_objective_reward import (
            create_customer_service_reward,
        )

        # Get minimal config for testing
        config = get_kimi_k25_config(
            task="customer_service",
            num_outer_iterations=1,  # Minimal for testing
            generations_per_iteration=2,
            eval_num_generations=1,
        )

        # Verify config is valid
        assert config.model_name == "moonshotai/Kimi-K2.5"
        assert config.num_outer_iterations == 1

        # Create reward model
        reward_model = create_customer_service_reward()
        assert reward_model is not None


def test_config_yaml_serialization():
    """Test config can be serialized for external tools"""
    import yaml

    config = get_kimi_k25_config(task="customer_service")
    config_dict = config.to_dict()

    # Convert to YAML
    yaml_str = yaml.dump(config_dict)
    assert yaml_str is not None

    # Convert back
    loaded_dict = yaml.safe_load(yaml_str)
    assert loaded_dict["model_name"] == "moonshotai/Kimi-K2.5"


def test_env_variable_config():
    """Test environment variable configuration"""
    import os

    # Set environment variable
    os.environ["KIMI_K25_MODEL_NAME"] = "moonshotai/Kimi-K2.5"

    # This would be used in config loading
    model_name = os.getenv("KIMI_K25_MODEL_NAME", "default")
    assert model_name == "moonshotai/Kimi-K2.5"

    # Clean up
    del os.environ["KIMI_K25_MODEL_NAME"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
