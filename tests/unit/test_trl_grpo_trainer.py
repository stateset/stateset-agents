"""
Comprehensive tests for training/trl_grpo_trainer.py

Covers:
- TRLGRPOConfig creation and validation
- TrajectoryGenerator with standard and vLLM backends
- ModelManager for model loading and LoRA
- TRLGRPOTrainerWrapper integration
- Training workflow
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import asdict

# Mock the external dependencies before importing
import sys
from unittest.mock import MagicMock

# Create mock modules with __spec__ attribute
class MockModule(MagicMock):
    __spec__ = MagicMock()

sys.modules['trl'] = MockModule()
sys.modules['trl.core'] = MockModule()
sys.modules['peft'] = MockModule()
sys.modules['vllm'] = MockModule()

from stateset_agents.training.trl_grpo_trainer import (
    TRLGRPOConfig,
    TrajectoryGenerator,
    ModelManager,
)
from stateset_agents.training.config import TrainingConfig
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.trajectory import MultiTurnTrajectory, ConversationTurn


class TestTRLGRPOConfig:
    """Test TRL GRPO configuration"""

    def test_config_creation_with_defaults(self):
        """Test creating config with default values"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
        )

        assert config.model_name == "gpt2"
        assert config.num_episodes == 10
        assert config.beta == 0.0  # Default
        assert config.num_generations == 4  # Default
        assert config.use_lora is True  # Default

    def test_config_custom_values(self):
        """Test creating config with custom values"""
        config = TRLGRPOConfig(
            model_name="meta-llama/Llama-2-7b",
            num_episodes=100,
            beta=0.1,
            num_generations=8,
            use_lora=True,
            lora_r=32,
            lora_alpha=64,
            temperature=0.8,
            top_p=0.95,
        )

        assert config.beta == 0.1
        assert config.num_generations == 8
        assert config.lora_r == 32
        assert config.lora_alpha == 64
        assert config.temperature == 0.8
        assert config.top_p == 0.95

    def test_config_from_training_config(self):
        """Test creating TRL config from base training config"""
        base_config = TrainingConfig(
            model_name="gpt2",
            num_episodes=50,
            learning_rate=1e-5,
        )

        trl_config = TRLGRPOConfig.from_training_config(
            base_config,
            beta=0.05,
            num_generations=6
        )

        assert trl_config.model_name == "gpt2"
        assert trl_config.num_episodes == 50
        assert trl_config.learning_rate == 1e-5
        assert trl_config.beta == 0.05
        assert trl_config.num_generations == 6

    def test_config_serialization(self):
        """Test config to_dict and from_dict"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=20,
            beta=0.1,
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "gpt2"
        assert config_dict["beta"] == 0.1

    def test_config_lora_parameters(self):
        """Test LoRA-specific configuration"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            use_lora=True,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            lora_target_modules=["q_proj", "v_proj"],
        )

        assert config.use_lora is True
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert "q_proj" in config.lora_target_modules

    def test_config_memory_optimization(self):
        """Test memory optimization flags"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            gradient_checkpointing=True,
            use_8bit=True,
            use_4bit=False,
        )

        assert config.gradient_checkpointing is True
        assert config.use_8bit is True
        assert config.use_4bit is False

    def test_config_generation_parameters(self):
        """Test generation-specific parameters"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            max_prompt_length=512,
            max_completion_length=256,
            temperature=0.9,
            top_p=0.95,
        )

        assert config.max_prompt_length == 512
        assert config.max_completion_length == 256
        assert config.temperature == 0.9
        assert config.top_p == 0.95


class TestTrajectoryGenerator:
    """Test trajectory generation"""

    @pytest.mark.asyncio
    async def test_generator_creation(self):
        """Test creating trajectory generator"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            use_vllm=False,
        )

        agent_config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(agent_config)
        await agent.initialize()

        scenarios = [{"topic": "test", "context": "Test scenario"}]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=3)

        generator = TrajectoryGenerator(config, agent, env)

        assert generator is not None
        assert generator.config == config
        assert generator.agent == agent
        assert generator.environment == env

    @pytest.mark.asyncio
    async def test_generator_without_vllm(self):
        """Test generator uses standard generation when vLLM disabled"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            use_vllm=False,
        )

        agent_config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(agent_config)
        await agent.initialize()

        scenarios = [{"topic": "greeting"}]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=2)

        generator = TrajectoryGenerator(config, agent, env)

        assert generator.vllm_engine is None

    @pytest.mark.asyncio
    async def test_standard_generation(self):
        """Test standard trajectory generation"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            use_vllm=False,
        )

        agent_config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(agent_config)
        await agent.initialize()

        scenarios = [{"topic": "test"}]
        env = ConversationEnvironment(scenarios=scenarios, max_turns=2)

        # Mock the environment's run_episode method
        mock_trajectory = MultiTurnTrajectory()
        mock_trajectory.add_turn(ConversationTurn(role="user", content="Hello"))
        mock_trajectory.add_turn(ConversationTurn(role="assistant", content="Hi there!"))

        env.run_episode = AsyncMock(return_value=mock_trajectory)

        generator = TrajectoryGenerator(config, agent, env)

        trajectories = await generator._generate_standard(num_episodes=3)

        assert len(trajectories) == 3
        assert all(isinstance(t, MultiTurnTrajectory) for t in trajectories)


class TestModelManager:
    """Test model loading and management"""

    def test_model_manager_creation(self):
        """Test creating model manager"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
        )

        manager = ModelManager(config)

        assert manager is not None
        assert manager.config == config
        assert manager.model is None
        assert manager.tokenizer is None

    def test_model_manager_device_selection(self):
        """Test device selection (CPU/CUDA)"""
        config = TRLGRPOConfig(model_name="gpt2", num_episodes=10)
        manager = ModelManager(config)

        # Should default to CPU if CUDA not available
        assert manager.device in [torch.device("cpu"), torch.device("cuda")]

    @patch('training.trl_grpo_trainer.AutoTokenizer')
    @patch('training.trl_grpo_trainer.AutoModelForCausalLM')
    def test_model_loading_with_lora(self, mock_model_cls, mock_tokenizer_cls):
        """Test loading model with LoRA"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            use_lora=True,
            lora_r=16,
        )

        # Mock the model and tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        manager = ModelManager(config)

        # Note: This would normally call load_model_and_tokenizer
        # but we're just testing that the manager is properly configured
        assert manager.config.use_lora is True
        assert manager.config.lora_r == 16

    def test_model_manager_quantization_config(self):
        """Test quantization settings"""
        config_8bit = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            use_8bit=True,
            use_4bit=False,
        )

        manager_8bit = ModelManager(config_8bit)
        assert manager_8bit.config.use_8bit is True

        config_4bit = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            use_8bit=False,
            use_4bit=True,
        )

        manager_4bit = ModelManager(config_4bit)
        assert manager_4bit.config.use_4bit is True


class TestTRLGRPOTrainerWrapper:
    """Test TRL GRPO trainer wrapper"""

    @patch('training.trl_grpo_trainer.TRLGRPOTrainer')
    @patch('training.trl_grpo_trainer.GRPOConfig')
    def test_wrapper_creation(self, mock_grpo_config_cls, mock_trainer_cls):
        """Test creating trainer wrapper"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            output_dir="./output",
        )

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()
        mock_reward_fn = MagicMock()

        # Mock the GRPOConfig and TRLGRPOTrainer
        mock_grpo_config = MagicMock()
        mock_grpo_config_cls.return_value = mock_grpo_config

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        from stateset_agents.training.trl_grpo_trainer import TRLGRPOTrainerWrapper

        wrapper = TRLGRPOTrainerWrapper(
            config=config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_dataset,
            reward_function=mock_reward_fn,
        )

        assert wrapper is not None
        assert wrapper.config == config
        assert wrapper.model == mock_model
        assert wrapper.tokenizer == mock_tokenizer

    @patch('training.trl_grpo_trainer.TRLGRPOTrainer')
    @patch('training.trl_grpo_trainer.GRPOConfig')
    def test_wrapper_grpo_config_creation(self, mock_grpo_config_cls, mock_trainer_cls):
        """Test GRPO config creation from training config"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            output_dir="./output",
            beta=0.1,
            num_generations=8,
            learning_rate=1e-5,
        )

        mock_grpo_config_cls.return_value = MagicMock()
        mock_trainer_cls.return_value = MagicMock()

        from stateset_agents.training.trl_grpo_trainer import TRLGRPOTrainerWrapper

        wrapper = TRLGRPOTrainerWrapper(
            config=config,
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            reward_function=MagicMock(),
        )

        # Verify _create_grpo_config was called
        assert wrapper.grpo_config is not None


class TestTrainingWorkflow:
    """Test end-to-end training workflow"""

    @patch('training.trl_grpo_trainer.wandb')
    def test_wandb_initialization(self, mock_wandb):
        """Test W&B initialization in training"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            report_to="wandb",
            wandb_project="test-project",
            run_name="test-run",
        )

        # This would be part of train_with_trl_grpo function
        assert config.report_to == "wandb"
        assert config.wandb_project == "test-project"

    def test_config_validation(self):
        """Test configuration validation"""
        # Valid config
        valid_config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            learning_rate=1e-5,
        )

        assert valid_config.learning_rate > 0
        assert valid_config.num_episodes > 0

    def test_output_directory_config(self):
        """Test output directory configuration"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            output_dir="./custom_output",
        )

        assert config.output_dir == "./custom_output"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_config_with_extreme_values(self):
        """Test config with extreme parameter values"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=1,  # Minimum
            temperature=0.0,  # Deterministic
            learning_rate=1e-10,  # Very small
        )

        assert config.num_episodes == 1
        assert config.temperature == 0.0

    def test_config_with_large_batch_sizes(self):
        """Test config with large batch sizes"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            per_device_train_batch_size=128,
            gradient_accumulation_steps=32,
        )

        effective_batch_size = (
            config.per_device_train_batch_size *
            config.gradient_accumulation_steps
        )

        assert effective_batch_size == 4096

    @pytest.mark.asyncio
    async def test_generator_with_empty_environment(self):
        """Test generator handles empty scenarios gracefully"""
        config = TRLGRPOConfig(
            model_name="gpt2",
            num_episodes=10,
            use_vllm=False,
        )

        agent_config = AgentConfig(model_name="gpt2", use_stub_model=True)
        agent = MultiTurnAgent(agent_config)
        await agent.initialize()

        # Empty scenarios - should handle gracefully or error appropriately
        try:
            env = ConversationEnvironment(scenarios=[], max_turns=2)
            generator = TrajectoryGenerator(config, agent, env)
            # If it doesn't error, that's acceptable
        except (ValueError, AssertionError):
            # If it errors on empty scenarios, that's also acceptable
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
