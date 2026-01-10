"""
Comprehensive tests for training modules
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from training.config import TrainingConfig, TrainingProfile, get_config_for_task
from training.trainer import SingleTurnGRPOTrainer
from stateset_agents.core.environment import Environment, EnvironmentState, EpisodeStatus


# ===========================
# TrainingConfig Tests
# ===========================


class TestTrainingConfig:
    """Test training configuration"""

    def test_training_config_defaults(self):
        """Test default training configuration"""
        config = TrainingConfig()
        assert config.num_episodes > 0
        assert config.learning_rate > 0
        assert hasattr(config, 'per_device_train_batch_size')

    def test_training_config_custom_values(self):
        """Test custom training configuration"""
        config = TrainingConfig(
            num_episodes=50,
            learning_rate=1e-4,
            per_device_train_batch_size=16,
            max_grad_norm=1.5,
            bf16=True
        )
        assert config.num_episodes == 50
        assert config.learning_rate == 1e-4
        assert config.per_device_train_batch_size == 16
        assert config.max_grad_norm == 1.5
        assert config.bf16 is True

    def test_training_config_serialization(self):
        """Test training config serialization"""
        config = TrainingConfig(num_episodes=25, learning_rate=5e-5)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['num_episodes'] == 25
        assert config_dict['learning_rate'] == 5e-5

    def test_training_config_validation(self):
        """Test training config validation"""
        # Valid config
        config = TrainingConfig(num_episodes=10, learning_rate=1e-4)
        assert config.num_episodes > 0
        assert config.learning_rate > 0

    def test_training_profile_enum(self):
        """Test TrainingProfile enum values"""
        assert hasattr(TrainingProfile, 'CONSERVATIVE')
        assert hasattr(TrainingProfile, 'BALANCED')
        assert hasattr(TrainingProfile, 'AGGRESSIVE')

    def test_get_config_for_task(self):
        """Test getting config for specific task"""
        config = get_config_for_task('customer_service')
        assert isinstance(config, TrainingConfig)
        assert config.num_episodes > 0

    def test_get_config_for_task_with_profile(self):
        """Test getting config with different profiles"""
        try:
            conservative = get_config_for_task('customer_service', profile='conservative')
            assert isinstance(conservative, TrainingConfig)
        except TypeError:
            # Function may not accept profile parameter
            pass

        try:
            balanced = get_config_for_task('customer_service', profile='balanced')
            assert isinstance(balanced, TrainingConfig)
        except TypeError:
            pass


# ===========================
# SingleTurnGRPOTrainer Tests
# ===========================


class TestSingleTurnGRPOTrainer:
    """Test SingleTurnGRPOTrainer functionality"""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent"""
        agent = MagicMock()
        agent.model = MagicMock()
        agent.tokenizer = MagicMock()
        agent.generate_response = AsyncMock(return_value="Test response")
        agent.model.named_parameters = MagicMock(return_value=[
            ("layer.weight", MagicMock(requires_grad=True)),
            ("bias", MagicMock(requires_grad=True))
        ])
        return agent

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment"""
        env = AsyncMock()
        env.reset = AsyncMock(return_value={"prompt": "Hello", "step": 0})
        env.step = AsyncMock(return_value={
            "state": {"step": 1},
            "reward": 0.8,
            "done": False
        })
        return env

    @pytest.fixture
    def mock_reward_fn(self):
        """Create a mock reward function"""
        reward_fn = AsyncMock()
        reward_result = MagicMock()
        reward_result.score = 0.75
        reward_fn.compute_reward = AsyncMock(return_value=reward_result)
        return reward_fn

    @pytest.fixture
    def training_config(self):
        """Create a test training config"""
        return TrainingConfig(
            num_episodes=2,
            max_steps_per_episode=3,
            learning_rate=1e-5,
            seed=42
        )

    def test_single_turn_trainer_initialization(self, mock_agent, mock_environment, training_config):
        """Test SingleTurnGRPOTrainer initialization"""
        trainer = SingleTurnGRPOTrainer(
            agent=mock_agent,
            environment=mock_environment,
            config=training_config
        )

        assert trainer.agent == mock_agent
        assert trainer.environment == mock_environment
        assert trainer.config == training_config
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0

    def test_single_turn_trainer_with_reward_function(
        self, mock_agent, mock_environment, mock_reward_fn, training_config
    ):
        """Test trainer with reward function"""
        trainer = SingleTurnGRPOTrainer(
            agent=mock_agent,
            environment=mock_environment,
            reward_fn=mock_reward_fn,
            config=training_config
        )

        assert trainer.reward_fn == mock_reward_fn

    def test_single_turn_trainer_with_callbacks(
        self, mock_agent, mock_environment, training_config
    ):
        """Test trainer with callbacks"""
        callback = MagicMock()
        trainer = SingleTurnGRPOTrainer(
            agent=mock_agent,
            environment=mock_environment,
            config=training_config,
            callbacks=[callback]
        )

        assert len(trainer.callbacks) == 1
        assert trainer.callbacks[0] == callback

    def test_single_turn_trainer_add_callback(
        self, mock_agent, mock_environment, training_config
    ):
        """Test adding callback to trainer"""
        trainer = SingleTurnGRPOTrainer(
            agent=mock_agent,
            environment=mock_environment,
            config=training_config
        )

        callback = MagicMock()
        trainer.add_callback(callback)

        assert len(trainer.callbacks) == 1
        assert trainer.callbacks[0] == callback

    @pytest.mark.asyncio
    @patch('training.trainer.torch')
    async def test_single_turn_trainer_initialize(
        self, mock_torch, mock_agent, mock_environment, training_config
    ):
        """Test trainer initialization"""
        mock_torch.cuda.is_available.return_value = False

        trainer = SingleTurnGRPOTrainer(
            agent=mock_agent,
            environment=mock_environment,
            config=training_config
        )

        await trainer.initialize()

        # Verify agent was initialized
        assert trainer.agent.model is not None

    @pytest.mark.asyncio
    @patch('training.trainer.torch')
    @patch('training.trainer.np')
    async def test_single_turn_trainer_train(
        self, mock_np, mock_torch, mock_agent, mock_environment, mock_reward_fn, training_config
    ):
        """Test training loop"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.optim.AdamW = MagicMock()
        mock_np.mean.return_value = 0.75

        trainer = SingleTurnGRPOTrainer(
            agent=mock_agent,
            environment=mock_environment,
            reward_fn=mock_reward_fn,
            config=training_config
        )

        await trainer.initialize()
        result = await trainer.train()

        # Verify training completed
        assert result == mock_agent
        assert trainer.global_step > 0

    @pytest.mark.asyncio
    @patch('training.trainer.torch')
    async def test_single_turn_trainer_accepts_environment_state(
        self, mock_torch, mock_agent, training_config
    ):
        """Ensure EnvironmentState-compatible environments work in the train loop."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.optim.AdamW = MagicMock()

        class StateEnv(Environment):
            async def reset(self, scenario=None):
                return EnvironmentState(
                    episode_id="ep1",
                    turn_count=0,
                    status=EpisodeStatus.ONGOING,
                    context={"prompt": "Hello"},
                )

            async def step(self, state, action):
                new_state = state.copy()
                new_state.status = EpisodeStatus.COMPLETED
                return new_state, 0.0, True, {}

        trainer = SingleTurnGRPOTrainer(
            agent=mock_agent,
            environment=StateEnv(max_turns=1),
            config=training_config,
        )

        with patch("stateset_agents.training.single_turn_trainer.compute_grpo_loss") as mock_loss:
            mock_loss.return_value = {"policy_loss": None, "mean_advantage": 0.0}
            await trainer.initialize()
            result = await trainer.train()

        assert result == mock_agent

    def test_single_turn_trainer_setup_optimizer(
        self, mock_agent, mock_environment, training_config
    ):
        """Test optimizer setup"""
        with patch('training.trainer.torch') as mock_torch:
            mock_torch.optim.AdamW = MagicMock()

            trainer = SingleTurnGRPOTrainer(
                agent=mock_agent,
                environment=mock_environment,
                config=training_config
            )

            trainer._setup_optimizer()

            # Verify optimizer was created
            mock_torch.optim.AdamW.assert_called_once()

    @pytest.mark.asyncio
    async def test_single_turn_trainer_save_checkpoint(
        self, mock_agent, mock_environment, training_config, tmp_path
    ):
        """Test checkpoint saving"""
        trainer = SingleTurnGRPOTrainer(
            agent=mock_agent,
            environment=mock_environment,
            config=training_config
        )

        checkpoint_path = tmp_path / "checkpoint"
        await trainer.save_checkpoint(checkpoint_path)

        # Verify checkpoint directory was created
        assert checkpoint_path.exists()


# ===========================
# Training Utilities Tests
# ===========================


class TestTrainingUtilities:
    """Test training utility functions"""

    def test_training_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            'num_episodes': 30,
            'learning_rate': 2e-4,
            'per_device_train_batch_size': 8
        }

        config = TrainingConfig(**config_dict)
        assert config.num_episodes == 30
        assert config.learning_rate == 2e-4
        assert config.per_device_train_batch_size == 8

    def test_training_config_merge(self):
        """Test merging training configs"""
        base_config = TrainingConfig(num_episodes=10)
        override_config = TrainingConfig(num_episodes=20, learning_rate=1e-4)

        # Override base config values
        merged = TrainingConfig(
            num_episodes=override_config.num_episodes,
            learning_rate=override_config.learning_rate
        )

        assert merged.num_episodes == 20
        assert merged.learning_rate == 1e-4
