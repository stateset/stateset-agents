"""
Unit tests for the Advanced Training Orchestrator module.

Tests cover training job management, resource allocation, job scheduling,
experiment tracking, and training workers.
"""

import asyncio
import pickle
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Handle import errors gracefully (e.g., torchvision compatibility issues)
try:
    from training.advanced_training_orchestrator import (
        AdvancedTrainingOrchestrator,
        ExperimentTracker,
        JobScheduler,
        ResourceManager,
        ResourceRequirement,
        ResourceType,
        SchedulingStrategy,
        TrainingConfig,
        TrainingJob,
        TrainingStatus,
        TrainingWorker,
        get_training_orchestrator,
    )
    ORCHESTRATOR_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    ORCHESTRATOR_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ORCHESTRATOR_AVAILABLE,
    reason="Training orchestrator not available (check transformers/torchvision compatibility)"
)


class TestResourceType:
    """Test ResourceType enum."""

    def test_resource_type_values(self):
        """Test that all resource types have expected values."""
        assert ResourceType.CPU.value == "cpu"
        assert ResourceType.GPU.value == "gpu"
        assert ResourceType.MEMORY.value == "memory"
        assert ResourceType.STORAGE.value == "storage"
        assert ResourceType.NETWORK.value == "network"


class TestTrainingStatus:
    """Test TrainingStatus enum."""

    def test_training_status_values(self):
        """Test that all training statuses have expected values."""
        assert TrainingStatus.PENDING.value == "pending"
        assert TrainingStatus.QUEUED.value == "queued"
        assert TrainingStatus.RUNNING.value == "running"
        assert TrainingStatus.PAUSED.value == "paused"
        assert TrainingStatus.COMPLETED.value == "completed"
        assert TrainingStatus.FAILED.value == "failed"
        assert TrainingStatus.CANCELLED.value == "cancelled"


class TestSchedulingStrategy:
    """Test SchedulingStrategy enum."""

    def test_scheduling_strategy_values(self):
        """Test that all scheduling strategies have expected values."""
        assert SchedulingStrategy.FIFO.value == "fifo"
        assert SchedulingStrategy.PRIORITY.value == "priority"
        assert SchedulingStrategy.FAIR_SHARE.value == "fair_share"
        assert SchedulingStrategy.SHORTEST_JOB_FIRST.value == "shortest_job_first"
        assert SchedulingStrategy.RESOURCE_AWARE.value == "resource_aware"


class TestResourceRequirement:
    """Test ResourceRequirement dataclass."""

    def test_resource_requirement_creation(self):
        """Test creating a ResourceRequirement."""
        req = ResourceRequirement(
            resource_type=ResourceType.CPU,
            amount=4.0,
            min_amount=2.0,
            max_amount=8.0,
            priority=3,
        )

        assert req.resource_type == ResourceType.CPU
        assert req.amount == 4.0
        assert req.min_amount == 2.0
        assert req.max_amount == 8.0
        assert req.priority == 3

    def test_resource_requirement_defaults(self):
        """Test ResourceRequirement default values."""
        req = ResourceRequirement(
            resource_type=ResourceType.MEMORY,
            amount=8.0,
        )

        assert req.min_amount is None
        assert req.max_amount is None
        assert req.priority == 1


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_training_config_creation(self):
        """Test creating a TrainingConfig."""
        config = TrainingConfig(
            experiment_name="test_experiment",
            agent_type="MultiTurnAgent",
            model_config={"model_type": "gpt2"},
            training_data="/path/to/data",
        )

        assert config.experiment_name == "test_experiment"
        assert config.agent_type == "MultiTurnAgent"
        assert config.num_epochs == 10
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4

    def test_training_config_custom(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            experiment_name="custom_experiment",
            agent_type="CustomAgent",
            model_config={"model_type": "custom"},
            training_data="/path/to/data",
            num_epochs=20,
            batch_size=64,
            learning_rate=5e-5,
            grpo_epsilon=0.3,
            enable_checkpointing=False,
        )

        assert config.num_epochs == 20
        assert config.batch_size == 64
        assert config.grpo_epsilon == 0.3
        assert config.enable_checkpointing is False


class TestTrainingJob:
    """Test TrainingJob dataclass."""

    @pytest.fixture
    def training_config(self):
        """Create a test training config."""
        return TrainingConfig(
            experiment_name="test",
            agent_type="TestAgent",
            model_config={},
            training_data="/data",
            num_epochs=5,
        )

    def test_training_job_creation(self, training_config):
        """Test creating a TrainingJob."""
        job = TrainingJob(
            job_id="job_001",
            config=training_config,
        )

        assert job.job_id == "job_001"
        assert job.status == TrainingStatus.PENDING
        assert job.priority == 1
        assert job.current_epoch == 0
        assert job.current_step == 0

    def test_training_job_runtime_not_started(self, training_config):
        """Test runtime property when job hasn't started."""
        job = TrainingJob(job_id="job_001", config=training_config)
        assert job.runtime is None

    def test_training_job_runtime_running(self, training_config):
        """Test runtime property for running job."""
        job = TrainingJob(job_id="job_001", config=training_config)
        job.started_at = time.time() - 100  # Started 100 seconds ago

        assert 99 <= job.runtime <= 101

    def test_training_job_runtime_completed(self, training_config):
        """Test runtime property for completed job."""
        job = TrainingJob(job_id="job_001", config=training_config)
        job.started_at = 1000.0
        job.completed_at = 1500.0

        assert job.runtime == 500.0

    def test_training_job_estimated_completion(self, training_config):
        """Test estimated completion calculation."""
        job = TrainingJob(job_id="job_001", config=training_config)
        job.started_at = time.time() - 100
        job.current_epoch = 2

        # 2 epochs in 100 seconds = 50 seconds per epoch
        # 3 remaining epochs * 50 = 150 seconds remaining
        estimated = job.estimated_completion
        assert estimated is not None
        current_time = time.time()
        assert estimated > current_time


class TestResourceManager:
    """Test ResourceManager class."""

    @pytest.fixture
    def resource_manager(self):
        """Create a ResourceManager for testing."""
        # Create ResourceManager directly - it detects resources automatically
        manager = ResourceManager()
        # Override with test values
        manager.available_resources = {
            ResourceType.CPU: 8.0,
            ResourceType.GPU: 2.0,
            ResourceType.MEMORY: 16.0,
            ResourceType.STORAGE: 100.0,
            ResourceType.NETWORK: 1000.0,
        }
        return manager

    @pytest.mark.asyncio
    async def test_can_allocate_sufficient_resources(self, resource_manager):
        """Test can_allocate with sufficient resources."""
        requirements = [
            ResourceRequirement(ResourceType.CPU, 2.0),
            ResourceRequirement(ResourceType.MEMORY, 4.0),
        ]

        result = await resource_manager.can_allocate(requirements)
        assert result is True

    @pytest.mark.asyncio
    async def test_can_allocate_insufficient_resources(self, resource_manager):
        """Test can_allocate with insufficient resources."""
        requirements = [
            ResourceRequirement(ResourceType.CPU, 100.0),  # More than available
        ]

        result = await resource_manager.can_allocate(requirements)
        assert result is False

    @pytest.mark.asyncio
    async def test_allocate_resources(self, resource_manager):
        """Test allocating resources."""
        requirements = [
            ResourceRequirement(ResourceType.CPU, 2.0),
            ResourceRequirement(ResourceType.MEMORY, 4.0),
        ]

        result = await resource_manager.allocate_resources("job_001", requirements)

        assert result is True
        assert "job_001" in resource_manager.allocated_resources
        assert resource_manager.allocated_resources["job_001"][ResourceType.CPU] == 2.0

    @pytest.mark.asyncio
    async def test_deallocate_resources(self, resource_manager):
        """Test deallocating resources."""
        requirements = [ResourceRequirement(ResourceType.CPU, 2.0)]
        await resource_manager.allocate_resources("job_001", requirements)

        await resource_manager.deallocate_resources("job_001")

        assert "job_001" not in resource_manager.allocated_resources

    def test_get_resource_utilization(self, resource_manager):
        """Test getting resource utilization."""
        resource_manager.allocated_resources["job_001"] = {
            ResourceType.CPU: 4.0,  # Half of 8 available
        }

        utilization = resource_manager.get_resource_utilization()

        assert utilization[ResourceType.CPU] == 0.5


class TestJobScheduler:
    """Test JobScheduler class."""

    @pytest.fixture
    def training_config(self):
        """Create a test training config."""
        return TrainingConfig(
            experiment_name="test",
            agent_type="TestAgent",
            model_config={},
            training_data="/data",
            num_epochs=5,
        )

    @pytest.fixture
    def scheduler(self):
        """Create a JobScheduler for testing."""
        return JobScheduler(SchedulingStrategy.FIFO)

    @pytest.mark.asyncio
    async def test_submit_job(self, scheduler, training_config):
        """Test submitting a job."""
        job = TrainingJob(job_id="job_001", config=training_config)

        await scheduler.submit_job(job)

        assert len(scheduler.job_queue) == 1
        assert scheduler.job_queue[0].status == TrainingStatus.QUEUED

    @pytest.mark.asyncio
    async def test_get_next_job(self, scheduler, training_config):
        """Test getting next job from queue."""
        job = TrainingJob(job_id="job_001", config=training_config)
        await scheduler.submit_job(job)

        # Create a mock resource manager that allows allocation
        resource_manager = MagicMock()
        resource_manager.can_allocate = AsyncMock(return_value=True)

        next_job = await scheduler.get_next_job(resource_manager)

        assert next_job is not None
        assert next_job.job_id == "job_001"
        assert len(scheduler.job_queue) == 0

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self, scheduler, training_config):
        """Test cancelling a queued job."""
        job = TrainingJob(job_id="job_001", config=training_config)
        await scheduler.submit_job(job)

        result = await scheduler.cancel_job("job_001")

        assert result is True
        assert len(scheduler.job_queue) == 0

    @pytest.mark.asyncio
    async def test_cancel_running_job(self, scheduler, training_config):
        """Test cancelling a running job."""
        job = TrainingJob(job_id="job_001", config=training_config)
        job.status = TrainingStatus.RUNNING
        scheduler.running_jobs["job_001"] = job

        result = await scheduler.cancel_job("job_001")

        assert result is True
        assert scheduler.running_jobs["job_001"].status == TrainingStatus.CANCELLED

    def test_sort_queue_priority(self, training_config):
        """Test priority-based queue sorting."""
        scheduler = JobScheduler(SchedulingStrategy.PRIORITY)

        job1 = TrainingJob(job_id="job_001", config=training_config, priority=1)
        job2 = TrainingJob(job_id="job_002", config=training_config, priority=5)
        job3 = TrainingJob(job_id="job_003", config=training_config, priority=3)

        scheduler.job_queue = [job1, job2, job3]
        scheduler._sort_queue()

        assert scheduler.job_queue[0].priority == 5
        assert scheduler.job_queue[1].priority == 3
        assert scheduler.job_queue[2].priority == 1

    def test_get_queue_status(self, scheduler, training_config):
        """Test getting queue status."""
        job = TrainingJob(job_id="job_001", config=training_config)
        scheduler.job_queue.append(job)
        scheduler.running_jobs["job_002"] = TrainingJob(
            job_id="job_002", config=training_config
        )
        scheduler.completed_jobs["job_003"] = TrainingJob(
            job_id="job_003", config=training_config
        )

        status = scheduler.get_queue_status()

        assert status["queued_jobs"] == 1
        assert status["running_jobs"] == 1
        assert status["completed_jobs"] == 1
        assert status["strategy"] == "fifo"


class TestExperimentTracker:
    """Test ExperimentTracker class."""

    @pytest.fixture
    def training_config(self):
        """Create a test training config."""
        return TrainingConfig(
            experiment_name="test_experiment",
            agent_type="TestAgent",
            model_config={},
            training_data="/data",
        )

    @pytest.fixture
    def tracker(self):
        """Create an ExperimentTracker for testing."""
        return ExperimentTracker(enable_wandb=False, enable_mlflow=False)

    @pytest.mark.asyncio
    async def test_start_experiment(self, tracker, training_config):
        """Test starting an experiment."""
        job = TrainingJob(job_id="job_001", config=training_config)

        experiment_id = await tracker.start_experiment(job)

        assert experiment_id is not None
        assert experiment_id in tracker.experiments
        assert tracker.experiments[experiment_id]["job_id"] == "job_001"

    @pytest.mark.asyncio
    async def test_log_metrics(self, tracker, training_config):
        """Test logging metrics."""
        job = TrainingJob(job_id="job_001", config=training_config)
        experiment_id = await tracker.start_experiment(job)

        metrics = {"loss": 0.5, "accuracy": 0.85}
        await tracker.log_metrics(experiment_id, metrics, step=10)

        experiment = tracker.experiments[experiment_id]
        assert "loss" in experiment["metrics"]
        assert experiment["metrics"]["loss"][0]["value"] == 0.5
        assert experiment["metrics"]["loss"][0]["step"] == 10

    @pytest.mark.asyncio
    async def test_log_artifact(self, tracker, training_config):
        """Test logging artifacts."""
        job = TrainingJob(job_id="job_001", config=training_config)
        experiment_id = await tracker.start_experiment(job)

        await tracker.log_artifact(experiment_id, "/path/to/model.pt", "model")

        experiment = tracker.experiments[experiment_id]
        assert len(experiment["artifacts"]) == 1
        assert experiment["artifacts"][0]["path"] == "/path/to/model.pt"
        assert experiment["artifacts"][0]["type"] == "model"

    @pytest.mark.asyncio
    async def test_finish_experiment(self, tracker, training_config):
        """Test finishing an experiment."""
        job = TrainingJob(job_id="job_001", config=training_config)
        experiment_id = await tracker.start_experiment(job)

        final_metrics = {"final_loss": 0.1, "final_accuracy": 0.95}
        await tracker.finish_experiment(experiment_id, final_metrics)

        experiment = tracker.experiments[experiment_id]
        assert "finished_at" in experiment
        assert experiment["final_metrics"] == final_metrics


class TestTrainingWorker:
    """Test TrainingWorker class."""

    @pytest.fixture
    def training_config(self):
        """Create a test training config."""
        return TrainingConfig(
            experiment_name="test",
            agent_type="TestAgent",
            model_config={},
            training_data="/data",
            num_epochs=2,
            enable_checkpointing=False,
            enable_wandb=False,
        )

    @pytest.fixture
    def worker(self):
        """Create a TrainingWorker for testing."""
        with patch("training.advanced_training_orchestrator.get_monitoring_service") as mock:
            mock.return_value = MagicMock()
            mock.return_value.record_training_iteration = MagicMock()
            return TrainingWorker("worker_001")

    def test_worker_creation(self, worker):
        """Test worker creation."""
        assert worker.worker_id == "worker_001"
        assert worker.current_job is None

    def test_create_optimizer(self, worker, training_config):
        """Test optimizer creation."""
        model = MagicMock()
        optimizer = worker._create_optimizer(model, training_config)

        assert optimizer["type"] == "adamw"
        assert optimizer["lr"] == 1e-4

    def test_create_scheduler(self, worker, training_config):
        """Test scheduler creation."""
        optimizer = MagicMock()
        scheduler = worker._create_scheduler(optimizer, training_config)

        assert scheduler["type"] == "cosine"

    def test_compute_final_metrics(self, worker, training_config):
        """Test computing final metrics."""
        job = TrainingJob(job_id="job_001", config=training_config)
        job.current_epoch = 5
        job.current_step = 500
        job.started_at = time.time() - 100
        job.completed_at = time.time()

        metrics = worker._compute_final_metrics(job)

        assert metrics["final_epoch"] == 5
        assert metrics["total_steps"] == 500
        assert "training_time" in metrics


class TestAdvancedTrainingOrchestrator:
    """Test AdvancedTrainingOrchestrator class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies."""
        with patch("training.advanced_training_orchestrator.get_monitoring_service") as mock_monitoring, \
             patch("training.advanced_training_orchestrator.get_state_service") as mock_state, \
             patch("training.advanced_training_orchestrator.psutil") as mock_psutil:
            mock_monitoring.return_value = MagicMock()
            mock_monitoring.return_value.metrics_collector = MagicMock()
            mock_state.return_value = MagicMock()
            mock_state.return_value.state_manager = MagicMock()
            mock_state.return_value.state_manager.set = AsyncMock()
            mock_state.return_value.state_manager.get = AsyncMock(return_value=None)
            mock_psutil.cpu_count.return_value = 8
            mock_psutil.virtual_memory.return_value = MagicMock(total=16 * (1024**3))
            mock_psutil.disk_usage.return_value = MagicMock(free=100 * (1024**3))
            yield {
                "monitoring": mock_monitoring,
                "state": mock_state,
                "psutil": mock_psutil,
            }

    @pytest.fixture
    def training_config(self):
        """Create a test training config."""
        return TrainingConfig(
            experiment_name="test",
            agent_type="TestAgent",
            model_config={},
            training_data="/data",
        )

    @pytest.fixture
    def orchestrator(self, mock_dependencies):
        """Create an orchestrator for testing."""
        orch = AdvancedTrainingOrchestrator(
            max_concurrent_jobs=2,
            scheduling_strategy=SchedulingStrategy.FIFO,
            enable_experiment_tracking=False,
        )
        # Cancel background tasks for testing
        if orch._orchestration_task:
            orch._orchestration_task.cancel()
        if orch._monitoring_task:
            orch._monitoring_task.cancel()
        return orch

    def test_orchestrator_creation(self, orchestrator):
        """Test orchestrator creation."""
        assert orchestrator.max_concurrent_jobs == 2
        assert orchestrator.scheduler is not None
        assert orchestrator.resource_manager is not None

    @pytest.mark.asyncio
    async def test_submit_training_job(self, orchestrator, training_config, mock_dependencies):
        """Test submitting a training job."""
        job_id = await orchestrator.submit_training_job(
            training_config, priority=2, user_id="user_001"
        )

        assert job_id is not None
        assert len(orchestrator.scheduler.job_queue) == 1
        assert orchestrator.scheduler.job_queue[0].priority == 2

    @pytest.mark.asyncio
    async def test_cancel_job(self, orchestrator, training_config, mock_dependencies):
        """Test cancelling a job."""
        job_id = await orchestrator.submit_training_job(training_config)

        result = await orchestrator.cancel_job(job_id)

        assert result is True
        assert len(orchestrator.scheduler.job_queue) == 0

    @pytest.mark.asyncio
    async def test_get_system_status(self, orchestrator, mock_dependencies):
        """Test getting system status."""
        status = await orchestrator.get_system_status()

        assert "resource_utilization" in status
        assert "queue_status" in status
        assert "max_concurrent_jobs" in status
        assert status["max_concurrent_jobs"] == 2

    @pytest.mark.asyncio
    async def test_shutdown(self, orchestrator, mock_dependencies):
        """Test orchestrator shutdown."""
        # Add a running job
        config = TrainingConfig(
            experiment_name="test",
            agent_type="TestAgent",
            model_config={},
            training_data="/data",
        )
        job = TrainingJob(job_id="job_001", config=config)
        orchestrator.scheduler.running_jobs["job_001"] = job

        await orchestrator.shutdown()

        # Verify job was cancelled
        assert orchestrator.scheduler.running_jobs["job_001"].status == TrainingStatus.CANCELLED


class TestGetTrainingOrchestrator:
    """Test global orchestrator instance."""

    @patch("training.advanced_training_orchestrator._orchestrator", None)
    @patch("training.advanced_training_orchestrator.get_monitoring_service")
    @patch("training.advanced_training_orchestrator.get_state_service")
    @patch("training.advanced_training_orchestrator.psutil")
    def test_get_training_orchestrator_creates_instance(
        self, mock_psutil, mock_state, mock_monitoring
    ):
        """Test that get_training_orchestrator creates a new instance."""
        mock_monitoring.return_value = MagicMock()
        mock_monitoring.return_value.metrics_collector = MagicMock()
        mock_state.return_value = MagicMock()
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * (1024**3))
        mock_psutil.disk_usage.return_value = MagicMock(free=100 * (1024**3))

        orchestrator = get_training_orchestrator()

        assert orchestrator is not None
        assert isinstance(orchestrator, AdvancedTrainingOrchestrator)
