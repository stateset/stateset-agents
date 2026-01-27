"""
Unit tests for the Multi-Turn GRPO Trainer module.

Tests cover multi-turn conversation training, trajectory handling,
and GRPO-specific training components.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


class TestMultiTurnGRPOTrainer:
    """Test MultiTurnGRPOTrainer class."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock()
        agent.model = None
        agent.initialize = AsyncMock()
        return agent

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment."""
        env = MagicMock()
        env.reset = AsyncMock(return_value={"state": "initial"})
        env.step = AsyncMock(return_value={
            "state": {"step": 1},
            "reward": 0.5,
            "done": False,
        })
        return env

    @pytest.fixture
    def mock_reward_fn(self):
        """Create a mock reward function."""
        reward_fn = MagicMock()
        reward_fn.compute_reward = AsyncMock(return_value=0.8)
        return reward_fn

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = MagicMock()
        config.seed = 42
        config.bf16 = False
        config.fp16 = False
        config.use_reference_model = False
        config.report_to = None
        config.learning_rate = 1e-4
        config.weight_decay = 0.01
        config.max_grad_norm = 1.0
        return config

    @pytest.fixture
    def trainer(self, mock_agent, mock_environment, mock_reward_fn, mock_config):
        """Create a MultiTurnGRPOTrainer for testing."""
        from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
        return MultiTurnGRPOTrainer(
            agent=mock_agent,
            environment=mock_environment,
            reward_fn=mock_reward_fn,
            config=mock_config,
        )

    def test_trainer_creation(self, trainer):
        """Test trainer creation."""
        assert trainer.agent is not None
        assert trainer.environment is not None
        assert trainer.reward_fn is not None
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0

    def test_trainer_initial_state(self, trainer):
        """Test trainer initial state."""
        assert trainer.optimizer is None
        assert trainer.lr_scheduler is None
        assert trainer.scaler is None
        assert trainer.best_eval_metric == float("-inf")
        assert trainer.steps_without_improvement == 0

    def test_trainer_grpo_enhancements(self, trainer):
        """Test GRPO-specific enhancements."""
        assert trainer._global_reward_mean == 0.0
        assert trainer._global_reward_count == 0
        assert trainer.reference_model is None

    @pytest.mark.asyncio
    async def test_trainer_initialize(self, trainer, mock_agent):
        """Test trainer initialization."""
        with patch("training.multi_turn_trainer.require_torch") as mock_torch, \
             patch("training.multi_turn_trainer.require_transformers"):
            mock_torch.return_value = MagicMock()

            # Mock optimizer setup
            trainer._setup_optimizer = MagicMock()

            await trainer.initialize()

            mock_agent.initialize.assert_called_once()

    def test_trainer_load_checkpoint(self, trainer, tmp_path):
        """Test loading checkpoint training state."""
        trainer.optimizer = MagicMock()
        trainer.lr_scheduler = MagicMock()

        checkpoint_path = tmp_path / "checkpoint"
        checkpoint_path.mkdir()
        (checkpoint_path / "training_state.pt").write_text("placeholder")

        state = {
            "global_step": 7,
            "current_epoch": 3,
            "best_eval_metric": 0.2,
            "steps_without_improvement": 2,
            "grad_accum_step": 4,
            "optimizer_state_dict": {"opt": 1},
            "scheduler_state_dict": {"sched": 2},
        }

        with patch(
            "stateset_agents.training.multi_turn_trainer.require_torch"
        ) as mock_torch:
            mock_torch.return_value = MagicMock(load=MagicMock(return_value=state))
            loaded = trainer.load_checkpoint(checkpoint_path)

        assert loaded is True
        assert trainer.global_step == 7
        assert trainer.current_epoch == 3
        trainer.optimizer.load_state_dict.assert_called_once_with(
            state["optimizer_state_dict"]
        )
        trainer.lr_scheduler.load_state_dict.assert_called_once_with(
            state["scheduler_state_dict"]
        )


class TestTrainerOptimizer:
    """Test trainer optimizer setup."""

    @pytest.fixture
    def trainer(self):
        """Create a trainer for testing."""
        from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

        mock_agent = MagicMock()
        mock_agent.model = MagicMock()
        mock_agent.model.parameters = MagicMock(return_value=[
            MagicMock(requires_grad=True)
        ])

        config = MagicMock()
        config.learning_rate = 1e-4
        config.weight_decay = 0.01

        return MultiTurnGRPOTrainer(
            agent=mock_agent,
            environment=MagicMock(),
            config=config,
        )

    def test_optimizer_config_params(self, trainer):
        """Test optimizer configuration parameters."""
        assert trainer.config.learning_rate == 1e-4
        assert trainer.config.weight_decay == 0.01


class TestTrajectoryHandling:
    """Test multi-turn trajectory handling."""

    def test_multi_turn_trajectory_creation(self):
        """Test creating multi-turn trajectories."""
        from stateset_agents.core.trajectory import MultiTurnTrajectory, ConversationTurn

        trajectory = MultiTurnTrajectory(trajectory_id="test_123")

        # Add turns
        turn1 = ConversationTurn(
            user_message="Hello",
            assistant_response="Hi there!",
            reward=0.8,
        )
        turn2 = ConversationTurn(
            user_message="How are you?",
            assistant_response="I'm doing great!",
            reward=0.9,
        )

        trajectory.add_turn(turn1)
        trajectory.add_turn(turn2)

        assert len(trajectory.turns) == 2
        assert trajectory.total_reward == 1.7

    def test_trajectory_average_reward(self):
        """Test trajectory average reward calculation."""
        from stateset_agents.core.trajectory import MultiTurnTrajectory, ConversationTurn

        trajectory = MultiTurnTrajectory()
        trajectory.add_turn(ConversationTurn("msg1", "resp1", 0.5))
        trajectory.add_turn(ConversationTurn("msg2", "resp2", 0.7))
        trajectory.add_turn(ConversationTurn("msg3", "resp3", 0.9))

        assert trajectory.average_reward == pytest.approx(0.7, rel=0.01)


def test_multi_turn_trainer_uses_environment_scenarios():
    """Trainer should prefer environment scenarios when available."""
    from types import SimpleNamespace
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    env = MagicMock()
    env.scenarios = [
        {"id": "s1", "context": "alpha"},
        {"id": "s2", "context": "beta"},
    ]
    config = SimpleNamespace(
        num_episodes=3,
        task_schedule=None,
        task_switch_steps=0,
        task_id_key="task_id",
        eval_split_size=0.0,
        max_examples=None,
    )

    trainer = MultiTurnGRPOTrainer(
        agent=MagicMock(),
        environment=env,
        config=config,
    )

    scenarios = trainer._get_training_scenarios()

    assert len(scenarios) == 3
    assert scenarios[0]["context"] == "alpha"
    assert scenarios[1]["context"] == "beta"


def test_multi_turn_trainer_stratifies_eval_by_task():
    """Eval split should include each task when stratify_by_task is enabled."""
    from types import SimpleNamespace
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    env = MagicMock()
    env.scenarios = [
        {"id": "a1", "task_id": "A"},
        {"id": "a2", "task_id": "A"},
        {"id": "b1", "task_id": "B"},
        {"id": "b2", "task_id": "B"},
    ]
    config = SimpleNamespace(
        num_episodes=4,
        task_schedule=None,
        task_switch_steps=0,
        task_id_key="task_id",
        eval_split_size=0.5,
        max_examples=None,
        stratify_by_task=True,
    )

    trainer = MultiTurnGRPOTrainer(
        agent=MagicMock(),
        environment=env,
        config=config,
    )

    eval_scenarios = trainer._get_eval_scenarios()
    tasks = {scenario.get("task_id") for scenario in eval_scenarios}

    assert tasks == {"A", "B"}


class TestTrajectoryGroup:
    """Test trajectory group handling for GRPO."""

    def test_trajectory_group_creation(self):
        """Test creating trajectory groups."""
        from stateset_agents.core.trajectory import TrajectoryGroup, MultiTurnTrajectory

        group = TrajectoryGroup()

        # Add trajectories
        for i in range(4):
            traj = MultiTurnTrajectory(trajectory_id=f"traj_{i}")
            group.add_trajectory(traj)

        assert len(group.trajectories) == 4

    def test_group_reward_statistics(self):
        """Test computing group reward statistics."""
        from stateset_agents.core.trajectory import TrajectoryGroup, MultiTurnTrajectory, ConversationTurn

        group = TrajectoryGroup()

        rewards = [0.5, 0.7, 0.3, 0.9]
        for i, reward in enumerate(rewards):
            traj = MultiTurnTrajectory(trajectory_id=f"traj_{i}")
            traj.add_turn(ConversationTurn(f"msg_{i}", f"resp_{i}", reward))
            group.add_trajectory(traj)

        # Compute statistics
        group_rewards = [t.total_reward for t in group.trajectories]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)

        assert mean_reward == pytest.approx(0.6, rel=0.01)
        assert std_reward > 0


class TestGRPOLossComputation:
    """Test GRPO loss computation."""

    def test_grpo_loss_import(self):
        """Test GRPO loss functions can be imported."""
        from stateset_agents.training.loss_computation import compute_grpo_loss, compute_enhanced_grpo_loss

        assert compute_grpo_loss is not None
        assert compute_enhanced_grpo_loss is not None

    def test_advantage_normalization_concept(self):
        """Test advantage normalization conceptually."""
        rewards = np.array([0.5, 0.7, 0.3, 0.9])

        # Group-relative advantage
        mean_reward = rewards.mean()
        std_reward = rewards.std()

        advantages = (rewards - mean_reward) / (std_reward + 1e-8)

        # Normalized advantages have mean ~0 and std ~1
        assert abs(advantages.mean()) < 0.01
        assert abs(advantages.std() - 1.0) < 0.1


class TestCallbacks:
    """Test trainer callbacks."""

    @pytest.fixture
    def trainer_with_callbacks(self):
        """Create a trainer with callbacks."""
        from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

        callbacks = [MagicMock(), MagicMock()]

        return MultiTurnGRPOTrainer(
            agent=MagicMock(),
            environment=MagicMock(),
            callbacks=callbacks,
        )

    def test_callbacks_registered(self, trainer_with_callbacks):
        """Test callbacks are registered."""
        assert len(trainer_with_callbacks.callbacks) == 2


class TestWandbIntegration:
    """Test W&B integration."""

    @pytest.fixture
    def trainer_with_wandb(self):
        """Create a trainer with W&B logger."""
        from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

        config = MagicMock()
        config.report_to = "wandb"

        return MultiTurnGRPOTrainer(
            agent=MagicMock(),
            environment=MagicMock(),
            config=config,
            wandb_logger=MagicMock(),
        )

    def test_wandb_logger_configured(self, trainer_with_wandb):
        """Test W&B logger is configured."""
        assert trainer_with_wandb.wandb_logger is not None
        assert trainer_with_wandb.config.report_to == "wandb"


class TestReferenceModel:
    """Test reference model for KL regularization."""

    @pytest.fixture
    def trainer(self):
        """Create a trainer for testing."""
        from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

        config = MagicMock()
        config.use_reference_model = True

        return MultiTurnGRPOTrainer(
            agent=MagicMock(),
            environment=MagicMock(),
            config=config,
        )

    def test_reference_model_config(self, trainer):
        """Test reference model configuration."""
        assert trainer.config.use_reference_model is True
        assert trainer.reference_model is None  # Before initialization


class TestMixedPrecision:
    """Test mixed precision training."""

    @pytest.fixture
    def trainer_fp16(self):
        """Create a trainer with FP16."""
        from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

        config = MagicMock()
        config.fp16 = True
        config.bf16 = False

        return MultiTurnGRPOTrainer(
            agent=MagicMock(),
            environment=MagicMock(),
            config=config,
        )

    def test_fp16_config(self, trainer_fp16):
        """Test FP16 configuration."""
        assert trainer_fp16.config.fp16 is True
        assert trainer_fp16.scaler is None  # Before initialization


class TestEarlyStopping:
    """Test early stopping mechanism."""

    @pytest.fixture
    def trainer(self):
        """Create a trainer for testing."""
        from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

        return MultiTurnGRPOTrainer(
            agent=MagicMock(),
            environment=MagicMock(),
        )

    def test_initial_best_metric(self, trainer):
        """Test initial best metric value."""
        assert trainer.best_eval_metric == float("-inf")

    def test_steps_without_improvement(self, trainer):
        """Test steps without improvement tracking."""
        assert trainer.steps_without_improvement == 0

    def test_improvement_tracking_concept(self, trainer):
        """Test improvement tracking conceptually."""
        # Simulate improvement
        trainer.best_eval_metric = 0.5
        new_metric = 0.6

        if new_metric > trainer.best_eval_metric:
            trainer.best_eval_metric = new_metric
            trainer.steps_without_improvement = 0
        else:
            trainer.steps_without_improvement += 1

        assert trainer.best_eval_metric == 0.6
        assert trainer.steps_without_improvement == 0


class TestTrainingMetrics:
    """Test training metrics tracking."""

    @pytest.fixture
    def trainer(self):
        """Create a trainer for testing."""
        from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

        return MultiTurnGRPOTrainer(
            agent=MagicMock(),
            environment=MagicMock(),
        )

    def test_global_step_tracking(self, trainer):
        """Test global step tracking."""
        assert trainer.global_step == 0

        trainer.global_step += 1
        assert trainer.global_step == 1

    def test_epoch_tracking(self, trainer):
        """Test epoch tracking."""
        assert trainer.current_epoch == 0

        trainer.current_epoch += 1
        assert trainer.current_epoch == 1

    def test_global_reward_mean_tracking(self, trainer):
        """Test global reward mean tracking."""
        assert trainer._global_reward_mean == 0.0
        assert trainer._global_reward_count == 0

        # Update running mean
        new_reward = 0.8
        trainer._global_reward_count += 1
        trainer._global_reward_mean += (new_reward - trainer._global_reward_mean) / trainer._global_reward_count

        assert trainer._global_reward_mean == 0.8
        assert trainer._global_reward_count == 1


class TestSchedulerSetup:
    """Test learning rate scheduler setup."""

    def test_scheduler_imports(self):
        """Test scheduler utilities can be imported."""
        from stateset_agents.training.trainer_utils import (
            get_cosine_schedule_with_warmup,
            get_linear_schedule_with_warmup,
        )

        assert get_cosine_schedule_with_warmup is not None
        assert get_linear_schedule_with_warmup is not None


class TestTaskSchedule:
    """Test task scheduling in training scenarios."""

    def test_task_schedule_assignment(self):
        from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

        config = MagicMock()
        config.num_episodes = 5
        config.task_schedule = ["task_a", "task_b"]
        config.task_switch_steps = 2
        config.task_id_key = "task_id"

        trainer = MultiTurnGRPOTrainer(
            agent=MagicMock(),
            environment=MagicMock(),
            config=config,
        )

        scenarios = trainer._get_training_scenarios()
        assert scenarios[0]["task_id"] == "task_a"
        assert scenarios[1]["task_id"] == "task_a"
        assert scenarios[2]["task_id"] == "task_b"
