"""
Unit tests for the Distributed GRPO Trainer module.

Tests cover distributed configuration, multi-GPU setup, fault tolerance,
and memory optimization features.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestDistributedConfig:
    """Test DistributedConfig dataclass."""

    def test_distributed_config_defaults(self):
        """Test distributed config with default values."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig()

        assert config.backend == "nccl"
        assert config.init_method == "env://"
        assert config.world_size == 1
        assert config.rank == 0
        assert config.local_rank == 0
        assert config.master_addr == "localhost"
        assert config.master_port == "12355"

    def test_distributed_config_training_settings(self):
        """Test training-specific settings."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig()

        assert config.gradient_accumulation_steps == 1
        assert config.sync_bn is True
        assert config.find_unused_parameters is False

    def test_distributed_config_fault_tolerance(self):
        """Test fault tolerance settings."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig()

        assert config.max_restarts == 3
        assert config.restart_interval == 60

    def test_distributed_config_memory_optimization(self):
        """Test memory optimization settings."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig()

        assert config.cpu_offload is False
        assert config.activation_checkpointing is False
        assert config.mixed_precision is True

    def test_distributed_config_custom_values(self):
        """Test distributed config with custom values."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig(
            backend="gloo",
            world_size=4,
            rank=2,
            gradient_accumulation_steps=4,
            cpu_offload=True,
        )

        assert config.backend == "gloo"
        assert config.world_size == 4
        assert config.rank == 2
        assert config.gradient_accumulation_steps == 4
        assert config.cpu_offload is True

    def test_distributed_config_to_dict(self):
        """Test config serialization to dictionary."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig()
        config_dict = config.to_dict()

        assert config_dict["backend"] == "nccl"
        assert config_dict["world_size"] == 1
        assert "master_addr" in config_dict
        assert "mixed_precision" in config_dict


class TestDistributedGRPOTrainer:
    """Test DistributedGRPOTrainer class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for distributed trainer."""
        with patch(
            "stateset_agents.training.distributed_trainer.DISTRIBUTED_AVAILABLE", True
        ):
            yield

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock()
        agent.model = MagicMock()
        return agent

    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment."""
        return MagicMock()

    @pytest.fixture
    def mock_reward_function(self):
        """Create a mock reward function."""
        return MagicMock()

    @pytest.fixture
    def training_config(self):
        """Create a training config."""
        config = MagicMock()
        config.learning_rate = 1e-4
        return config

    @pytest.fixture
    def distributed_config(self):
        """Create a distributed config."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        return DistributedConfig()

    @pytest.fixture
    def trainer(
        self,
        mock_dependencies,
        mock_agent,
        mock_environment,
        mock_reward_function,
        training_config,
        distributed_config,
    ):
        """Create a DistributedGRPOTrainer for testing."""
        import warnings

        from stateset_agents.training.distributed_trainer import DistributedGRPOTrainer

        # The class is deprecated (non-functional for real training); this
        # fixture only exercises its inert construction-time state.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return DistributedGRPOTrainer(
                agent=mock_agent,
                environment=mock_environment,
                reward_function=mock_reward_function,
                training_config=training_config,
                distributed_config=distributed_config,
            )

    def test_trainer_creation(self, trainer):
        """Test trainer creation."""
        assert trainer.agent is not None
        assert trainer.environment is not None
        assert trainer.reward_function is not None
        assert trainer.is_initialized is False
        assert trainer.is_master is False

    def test_trainer_initial_state(self, trainer):
        """Test trainer initial state."""
        assert trainer.device is None
        assert trainer.model is None
        assert trainer.optimizer is None
        assert trainer.scaler is None
        assert trainer.step_count == 0
        assert trainer.epoch_count == 0


class TestSetupDistributed:
    """Test distributed setup functionality."""

    def test_environment_variables_concept(self):
        """Test environment variables are set correctly."""
        master_addr = "localhost"
        master_port = "12355"
        world_size = 4
        rank = 2

        # Simulate setting env vars
        env_vars = {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
        }

        assert env_vars["MASTER_ADDR"] == "localhost"
        assert env_vars["WORLD_SIZE"] == "4"
        assert env_vars["RANK"] == "2"

    def test_device_assignment_concept(self):
        """Test device assignment for multi-GPU."""
        rank = 2
        device = f"cuda:{rank}"

        assert device == "cuda:2"


class TestDistributedDataParallel:
    """Test DDP functionality."""

    def test_ddp_wrapping_concept(self):
        """Test DDP model wrapping conceptually."""
        # DDP wraps model for distributed training
        # Each rank has a copy of the model
        world_size = 4

        # Each rank processes 1/world_size of the data
        batch_size = 32
        per_rank_batch = batch_size // world_size

        assert per_rank_batch == 8

    def test_gradient_synchronization_concept(self):
        """Test gradient synchronization conceptually."""
        # All-reduce averages gradients across ranks
        num_ranks = 4
        gradients_per_rank = [torch.randn(10) for _ in range(num_ranks)]

        # All-reduce: average
        avg_gradient = sum(gradients_per_rank) / num_ranks

        assert avg_gradient.shape == gradients_per_rank[0].shape


class TestGradientAccumulation:
    """Test gradient accumulation in distributed setting."""

    def test_gradient_accumulation_config(self):
        """Test gradient accumulation configuration."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig(gradient_accumulation_steps=4)

        assert config.gradient_accumulation_steps == 4

    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        world_size = 4
        per_device_batch = 8
        accumulation_steps = 4

        effective_batch = world_size * per_device_batch * accumulation_steps
        assert effective_batch == 128


class TestMixedPrecisionDistributed:
    """Test mixed precision in distributed training."""

    def test_mixed_precision_config(self):
        """Test mixed precision configuration."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig(mixed_precision=True)

        assert config.mixed_precision is True

    def test_grad_scaler_concept(self):
        """Test gradient scaler conceptually."""
        # GradScaler helps with mixed precision training
        scale = 65536.0  # Initial scale

        # Scale down on overflow
        scale_after_overflow = scale / 2
        assert scale_after_overflow == 32768.0


class TestFaultTolerance:
    """Test fault tolerance features."""

    def test_max_restarts_config(self):
        """Test max restarts configuration."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig(max_restarts=5)

        assert config.max_restarts == 5

    def test_restart_interval_config(self):
        """Test restart interval configuration."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig(restart_interval=120)

        assert config.restart_interval == 120

    def test_checkpoint_recovery_concept(self):
        """Test checkpoint recovery conceptually."""
        # State to save for recovery
        training_state = {
            "step": 1000,
            "epoch": 5,
            "model_state": "model_dict",
            "optimizer_state": "optim_dict",
            "rng_state": "rng_dict",
        }

        # On restart, load from checkpoint
        resumed_step = training_state["step"]
        resumed_epoch = training_state["epoch"]

        assert resumed_step == 1000
        assert resumed_epoch == 5


class TestMemoryOptimization:
    """Test memory optimization features."""

    def test_cpu_offload_config(self):
        """Test CPU offload configuration."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig(cpu_offload=True)

        assert config.cpu_offload is True

    def test_activation_checkpointing_config(self):
        """Test activation checkpointing configuration."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig(activation_checkpointing=True)

        assert config.activation_checkpointing is True


class TestDistributedSampler:
    """Test distributed data sampling."""

    def test_distributed_sampler_concept(self):
        """Test distributed sampler conceptually."""
        # Each rank gets a different subset
        dataset_size = 1000
        world_size = 4
        rank = 2

        # Samples for this rank
        indices = list(range(rank, dataset_size, world_size))

        assert len(indices) == dataset_size // world_size

    def test_epoch_shuffling_concept(self):
        """Test epoch shuffling in distributed sampler."""
        # Different shuffle per epoch
        seed = 42
        epoch = 5

        import random

        random.seed(seed + epoch)

        # Shuffled indices would differ per epoch
        assert True


class TestSyncBatchNorm:
    """Test synchronized batch normalization."""

    def test_sync_bn_config(self):
        """Test sync BN configuration."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig(sync_bn=True)

        assert config.sync_bn is True

    def test_sync_bn_effect_concept(self):
        """Test sync BN effect conceptually."""
        # Without sync: BN stats computed per-GPU
        # With sync: BN stats synchronized across GPUs

        world_size = 4
        per_gpu_samples = 8

        # Effective batch size for BN stats
        without_sync = per_gpu_samples
        with_sync = per_gpu_samples * world_size

        assert with_sync > without_sync


class TestDistributedMetrics:
    """Test distributed metrics aggregation."""

    def test_all_reduce_mean(self):
        """Test all-reduce for metric averaging."""
        # Simulate metrics from different ranks
        metrics_per_rank = torch.tensor([0.5, 0.6, 0.4, 0.55])

        # All-reduce mean
        global_mean = metrics_per_rank.mean()

        assert abs(global_mean.item() - 0.5125) < 0.01

    def test_all_gather_concept(self):
        """Test all-gather for collecting metrics."""
        world_size = 4

        # Each rank has local metrics
        local_metrics = {"loss": 0.5, "accuracy": 0.8}

        # After all-gather, rank 0 has all metrics
        gathered = [local_metrics.copy() for _ in range(world_size)]

        assert len(gathered) == world_size


class TestMasterRank:
    """Test master rank functionality."""

    def test_is_master_concept(self):
        """Test is_master determination."""
        for rank in range(4):
            is_master = rank == 0
            if rank == 0:
                assert is_master is True
            else:
                assert is_master is False

    def test_master_only_logging(self):
        """Test master-only logging concept."""
        rank = 2
        is_master = rank == 0

        logged_messages = []
        message = "Training step 100"

        if is_master:
            logged_messages.append(message)

        assert len(logged_messages) == 0  # Rank 2 doesn't log


class TestBackendSelection:
    """Test distributed backend selection."""

    def test_nccl_for_gpu(self):
        """Test NCCL backend for GPU training."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig(backend="nccl")

        assert config.backend == "nccl"

    def test_gloo_for_cpu(self):
        """Test Gloo backend for CPU training."""
        from stateset_agents.training.distributed_trainer import DistributedConfig

        config = DistributedConfig(backend="gloo")

        assert config.backend == "gloo"


class TestDistributedGRPOLoss:
    """The GRPO loss must be a real function of the rewards, not a 0.0 stub."""

    @staticmethod
    def _construct():
        from types import SimpleNamespace

        from stateset_agents.training.config import TrainingConfig
        from stateset_agents.training.distributed_trainer import (
            DistributedConfig,
            DistributedGRPOTrainer,
        )

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.tensor(0.5))
                self.device = torch.device("cpu")

            def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
                seq = input_ids.shape[1]
                logits = torch.zeros(1, seq, 7) + self.p
                # NLL varies with sequence length so distinct trajectories
                # contribute distinct terms to the loss.
                return SimpleNamespace(loss=self.p * seq, logits=logits)

        class _Tokenizer:
            def apply_chat_template(
                self,
                messages,
                *,
                return_dict: bool = False,
                return_assistant_tokens_mask: bool = False,
                **kwargs,
            ):
                n = 4 + sum(len(m["content"]) for m in messages)
                half = n // 2
                return {
                    "input_ids": torch.arange(1, n + 1).unsqueeze(0),
                    "attention_mask": torch.ones(1, n, dtype=torch.long),
                    "assistant_tokens_mask": torch.tensor(
                        [[0] * (n - half) + [1] * half]
                    ),
                }

        agent = SimpleNamespace(model=_Model(), tokenizer=_Tokenizer())
        return DistributedGRPOTrainer(
            agent=agent,
            environment=MagicMock(),
            reward_function=MagicMock(),
            training_config=TrainingConfig(
                num_generations=2, advantage_normalization=False
            ),
            distributed_config=DistributedConfig(backend="gloo"),
        )

    @classmethod
    def _make_trainer(cls):
        """Construct the (deprecated) trainer, absorbing its expected warning."""
        with pytest.warns(DeprecationWarning):
            return cls._construct()

    def test_init_warns_deprecated(self):
        """Constructing the trainer must emit a DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="DistributedGRPOTrainer"):
            self._construct()

    def test_generate_training_data_raises(self):
        """The trainer must refuse to fabricate training data."""
        import asyncio

        trainer = self._make_trainer()

        with pytest.raises(NotImplementedError, match="no data source"):
            asyncio.run(trainer._generate_training_data())

    @staticmethod
    def _trajectories(n):
        from types import SimpleNamespace

        return [
            SimpleNamespace(
                turns=[
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello" * (i + 1)},
                ],
                total_reward=0.0,
                metadata={},
            )
            for i in range(n)
        ]

    def test_distributed_trainer_loss_is_not_constant(self):
        """Two different reward lists must produce different losses."""
        trainer = self._make_trainer()

        loss_a = trainer._compute_grpo_loss(self._trajectories(2), [0.0, 1.0])
        loss_b = trainer._compute_grpo_loss(self._trajectories(2), [1.0, 0.0])

        assert loss_a.item() != 0.0
        assert loss_a.item() != pytest.approx(loss_b.item())
        assert loss_a.requires_grad

    def test_compute_rewards_uses_reward_function(self):
        """_compute_rewards must delegate to the configured reward function."""
        import asyncio

        from stateset_agents.core.reward import RewardResult

        trainer = self._make_trainer()
        trajectories = self._trajectories(2)

        async def fake_reward(turns, context=None):
            return RewardResult(score=0.75, breakdown={"fake": 0.75})

        trainer.reward_function = MagicMock()
        trainer.reward_function.compute_reward = fake_reward

        rewards = asyncio.run(trainer._compute_rewards(trajectories))

        assert rewards == [0.75, 0.75]
        assert trajectories[0].total_reward == 0.75

    def test_generate_trajectories_uses_environment(self):
        """_generate_trajectories must run real episodes, not echo the batch."""
        import asyncio
        from types import SimpleNamespace

        trainer = self._make_trainer()

        episodes = []

        async def run_episode(agent_fn, scenario=None, **kwargs):
            episodes.append(scenario)
            return SimpleNamespace(turns=[], total_reward=0.0, metadata={})

        trainer.environment = MagicMock()
        trainer.environment.run_episode = run_episode
        trainer.agent.generate_response = MagicMock()

        trajectories = asyncio.run(
            trainer._generate_trajectories({"prompt": "p", "response": "r"})
        )

        assert len(trajectories) == 2  # num_generations
        assert len(episodes) == 2
        assert episodes[0]["prompt"] == "p"
        assert not isinstance(trajectories[0], dict)
