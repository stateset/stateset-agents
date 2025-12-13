"""
Advanced Training Orchestrator for GRPO Agent Framework

This module provides sophisticated training orchestration with dynamic resource allocation,
fault tolerance, experiment tracking, and intelligent scheduling capabilities.
"""

import asyncio
import json
import logging
import pickle
import shutil
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import psutil  # type: ignore[import-not-found]

    PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False

from stateset_agents.core.advanced_monitoring import get_monitoring_service, monitor_async_function
from stateset_agents.core.enhanced_state_management import get_state_service
from stateset_agents.core.error_handling import ErrorHandler, GRPOException, RetryConfig, retry_async
from stateset_agents.core.performance_optimizer import OptimizationLevel, PerformanceOptimizer

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Training job status"""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResourceType(Enum):
    """Resource types for training"""

    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


class SchedulingStrategy(Enum):
    """Job scheduling strategies"""

    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based
    FAIR_SHARE = "fair_share"  # Fair resource sharing
    SHORTEST_JOB_FIRST = "shortest_job_first"  # Shortest estimated time first
    RESOURCE_AWARE = "resource_aware"  # Resource utilization aware


@dataclass
class ResourceRequirement:
    """Resource requirement specification"""

    resource_type: ResourceType
    amount: float
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    priority: int = 1  # 1 = low, 5 = high


@dataclass
class TrainingConfig:
    """Comprehensive training configuration"""

    # Basic configuration
    experiment_name: str
    agent_type: str
    model_config: Dict[str, Any]
    training_data: Union[str, List[str]]  # Path or list of paths

    # Training parameters
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    # GRPO specific
    grpo_epsilon: float = 0.2
    grpo_value_loss_coef: float = 0.5
    grpo_entropy_coef: float = 0.01

    # Resource requirements
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    max_runtime: Optional[int] = None  # Maximum runtime in seconds

    # Fault tolerance
    enable_checkpointing: bool = True
    checkpoint_frequency: int = 100  # Steps between checkpoints
    max_retries: int = 3

    # Monitoring and logging
    enable_wandb: bool = True
    enable_mlflow: bool = False
    log_frequency: int = 10

    # Advanced features
    enable_early_stopping: bool = True
    early_stopping_patience: int = 5
    enable_model_pruning: bool = False
    enable_quantization: bool = False


@dataclass
class TrainingJob:
    """Training job definition"""

    job_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    priority: int = 1
    user_id: Optional[str] = None

    # Runtime information
    assigned_resources: Dict[str, Any] = field(default_factory=dict)
    worker_nodes: List[str] = field(default_factory=list)
    checkpoint_path: Optional[str] = None

    # Progress tracking
    current_epoch: int = 0
    current_step: int = 0
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    # Error handling
    retry_count: int = 0
    last_error: Optional[str] = None

    @property
    def runtime(self) -> Optional[float]:
        """Get job runtime in seconds"""
        if self.started_at is None:
            return None
        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    @property
    def estimated_completion(self) -> Optional[float]:
        """Estimate completion time based on progress"""
        if self.current_epoch == 0 or self.runtime is None:
            return None

        epochs_remaining = self.config.num_epochs - self.current_epoch
        time_per_epoch = self.runtime / self.current_epoch
        return time.time() + (epochs_remaining * time_per_epoch)


class ResourceManager:
    """Dynamic resource allocation and management"""

    def __init__(self):
        self.available_resources: Dict[ResourceType, float] = {}
        self.allocated_resources: Dict[
            str, Dict[ResourceType, float]
        ] = {}  # job_id -> resources
        self.resource_locks: Dict[ResourceType, asyncio.Lock] = {}

        # Initialize locks
        for resource_type in ResourceType:
            self.resource_locks[resource_type] = asyncio.Lock()

        # Detect available resources
        self._detect_resources()

    def _detect_resources(self):
        """Detect available system resources"""
        try:
            import psutil

            # CPU cores
            self.available_resources[ResourceType.CPU] = psutil.cpu_count()

            # Memory in GB
            memory_gb = psutil.virtual_memory().total / (1024**3)
            self.available_resources[ResourceType.MEMORY] = memory_gb

            # Storage in GB (available space)
            disk_gb = psutil.disk_usage("/").free / (1024**3)
            self.available_resources[ResourceType.STORAGE] = disk_gb

            # Network bandwidth (simplified)
            self.available_resources[ResourceType.NETWORK] = 1000.0  # Mbps

            # GPU detection
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.available_resources[ResourceType.GPU] = torch.cuda.device_count()
            else:
                self.available_resources[ResourceType.GPU] = 0.0

        except ImportError:
            # Fallback values
            self.available_resources = {
                ResourceType.CPU: 4.0,
                ResourceType.GPU: 0.0,
                ResourceType.MEMORY: 8.0,
                ResourceType.STORAGE: 100.0,
                ResourceType.NETWORK: 100.0,
            }

    async def can_allocate(self, requirements: List[ResourceRequirement]) -> bool:
        """Check if resources can be allocated"""
        for req in requirements:
            async with self.resource_locks[req.resource_type]:
                available = self.available_resources.get(req.resource_type, 0)
                allocated = sum(
                    alloc.get(req.resource_type, 0)
                    for alloc in self.allocated_resources.values()
                )

                if available - allocated < req.amount:
                    return False

        return True

    async def allocate_resources(
        self, job_id: str, requirements: List[ResourceRequirement]
    ) -> bool:
        """Allocate resources for a job"""
        # Check availability first
        if not await self.can_allocate(requirements):
            return False

        # Allocate resources
        allocated = {}
        for req in requirements:
            async with self.resource_locks[req.resource_type]:
                allocated[req.resource_type] = req.amount

        self.allocated_resources[job_id] = allocated
        return True

    async def deallocate_resources(self, job_id: str):
        """Deallocate resources for a job"""
        if job_id in self.allocated_resources:
            del self.allocated_resources[job_id]

    def get_resource_utilization(self) -> Dict[ResourceType, float]:
        """Get current resource utilization"""
        utilization = {}

        for resource_type in ResourceType:
            available = self.available_resources.get(resource_type, 0)
            allocated = sum(
                alloc.get(resource_type, 0)
                for alloc in self.allocated_resources.values()
            )

            if available > 0:
                utilization[resource_type] = allocated / available
            else:
                utilization[resource_type] = 0.0

        return utilization


class JobScheduler:
    """Intelligent job scheduling"""

    def __init__(
        self, strategy: SchedulingStrategy = SchedulingStrategy.RESOURCE_AWARE
    ):
        self.strategy = strategy
        self.job_queue: List[TrainingJob] = []
        self.running_jobs: Dict[str, TrainingJob] = {}
        self.completed_jobs: Dict[str, TrainingJob] = {}
        self.queue_lock = asyncio.Lock()

    async def submit_job(self, job: TrainingJob):
        """Submit a job to the scheduler"""
        async with self.queue_lock:
            job.status = TrainingStatus.QUEUED
            self.job_queue.append(job)
            self._sort_queue()

    async def get_next_job(
        self, resource_manager: ResourceManager
    ) -> Optional[TrainingJob]:
        """Get the next job to run"""
        async with self.queue_lock:
            for i, job in enumerate(self.job_queue):
                if await resource_manager.can_allocate(
                    job.config.resource_requirements
                ):
                    return self.job_queue.pop(i)
            return None

    def _sort_queue(self):
        """Sort job queue based on scheduling strategy"""
        if self.strategy == SchedulingStrategy.FIFO:
            # Already in FIFO order
            pass
        elif self.strategy == SchedulingStrategy.PRIORITY:
            self.job_queue.sort(key=lambda job: -job.priority)
        elif self.strategy == SchedulingStrategy.SHORTEST_JOB_FIRST:
            self.job_queue.sort(key=lambda job: job.config.num_epochs)
        # Add more strategies as needed

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        async with self.queue_lock:
            # Remove from queue
            for i, job in enumerate(self.job_queue):
                if job.job_id == job_id:
                    job.status = TrainingStatus.CANCELLED
                    self.job_queue.pop(i)
                    return True

            # Cancel running job
            if job_id in self.running_jobs:
                job = self.running_jobs[job_id]
                job.status = TrainingStatus.CANCELLED
                # Note: Actual cancellation would need to stop the training process
                return True

        return False

    def get_queue_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "queued_jobs": len(self.job_queue),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": len(self.completed_jobs),
            "strategy": self.strategy.value,
        }


class ExperimentTracker:
    """Advanced experiment tracking and management"""

    def __init__(self, enable_wandb: bool = True, enable_mlflow: bool = False):
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.enable_mlflow = enable_mlflow and MLFLOW_AVAILABLE
        self.experiments: Dict[str, Dict[str, Any]] = {}

    async def start_experiment(self, job: TrainingJob) -> str:
        """Start tracking an experiment"""
        experiment_id = f"{job.config.experiment_name}_{job.job_id}"

        experiment_data = {
            "experiment_id": experiment_id,
            "job_id": job.job_id,
            "config": job.config.__dict__,
            "started_at": time.time(),
            "metrics": {},
            "artifacts": [],
        }

        self.experiments[experiment_id] = experiment_data

        # Initialize external trackers
        if self.enable_wandb:
            try:
                wandb.init(
                    project=job.config.experiment_name,
                    name=job.job_id,
                    config=job.config.__dict__,
                )
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")

        if self.enable_mlflow:
            try:
                mlflow.start_run(run_name=job.job_id)
                mlflow.log_params(job.config.__dict__)
            except Exception as e:
                logger.error(f"Failed to initialize MLflow: {e}")

        return experiment_id

    async def log_metrics(
        self, experiment_id: str, metrics: Dict[str, float], step: int
    ):
        """Log metrics for an experiment"""
        if experiment_id not in self.experiments:
            return

        experiment = self.experiments[experiment_id]

        # Store metrics
        for metric_name, value in metrics.items():
            if metric_name not in experiment["metrics"]:
                experiment["metrics"][metric_name] = []
            experiment["metrics"][metric_name].append({"step": step, "value": value})

        # Log to external trackers
        if self.enable_wandb:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.error(f"Failed to log to W&B: {e}")

        if self.enable_mlflow:
            try:
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value, step=step)
            except Exception as e:
                logger.error(f"Failed to log to MLflow: {e}")

    async def log_artifact(
        self, experiment_id: str, artifact_path: str, artifact_type: str = "model"
    ):
        """Log an artifact"""
        if experiment_id not in self.experiments:
            return

        experiment = self.experiments[experiment_id]
        experiment["artifacts"].append(
            {"path": artifact_path, "type": artifact_type, "logged_at": time.time()}
        )

        # Log to external trackers
        if self.enable_wandb:
            try:
                wandb.save(artifact_path)
            except Exception as e:
                logger.error(f"Failed to log artifact to W&B: {e}")

        if self.enable_mlflow:
            try:
                mlflow.log_artifact(artifact_path)
            except Exception as e:
                logger.error(f"Failed to log artifact to MLflow: {e}")

    async def finish_experiment(
        self, experiment_id: str, final_metrics: Dict[str, float] = None
    ):
        """Finish tracking an experiment"""
        if experiment_id not in self.experiments:
            return

        experiment = self.experiments[experiment_id]
        experiment["finished_at"] = time.time()

        if final_metrics:
            experiment["final_metrics"] = final_metrics

        # Finish external trackers
        if self.enable_wandb:
            try:
                if final_metrics:
                    wandb.log(final_metrics)
                wandb.finish()
            except Exception as e:
                logger.error(f"Failed to finish W&B: {e}")

        if self.enable_mlflow:
            try:
                if final_metrics:
                    for metric_name, value in final_metrics.items():
                        mlflow.log_metric(metric_name, value)
                mlflow.end_run()
            except Exception as e:
                logger.error(f"Failed to finish MLflow: {e}")


class TrainingWorker:
    """Individual training worker"""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.current_job: Optional[TrainingJob] = None
        self.error_handler = ErrorHandler()
        self.performance_optimizer = PerformanceOptimizer(OptimizationLevel.BALANCED)
        self.monitoring = get_monitoring_service()

    @monitor_async_function("training_worker.execute_job")
    async def execute_job(
        self,
        job: TrainingJob,
        experiment_tracker: ExperimentTracker,
        checkpoint_callback: Optional[Callable] = None,
    ) -> bool:
        """Execute a training job"""
        self.current_job = job
        job.status = TrainingStatus.RUNNING
        job.started_at = time.time()

        try:
            # Start experiment tracking
            experiment_id = await experiment_tracker.start_experiment(job)

            # Setup checkpointing
            if job.config.enable_checkpointing:
                checkpoint_dir = Path(f"checkpoints/{job.job_id}")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                job.checkpoint_path = str(checkpoint_dir)

            # Execute training with retries
            success = await self._execute_training_with_retries(
                job, experiment_tracker, experiment_id
            )

            if success:
                job.status = TrainingStatus.COMPLETED
                job.completed_at = time.time()

                # Log final metrics
                final_metrics = self._compute_final_metrics(job)
                await experiment_tracker.log_metrics(
                    experiment_id, final_metrics, job.current_step
                )

                # Save final model
                if job.checkpoint_path:
                    model_path = Path(job.checkpoint_path) / "final_model.pt"
                    await self._save_model(job, str(model_path))
                    await experiment_tracker.log_artifact(
                        experiment_id, str(model_path), "final_model"
                    )
            else:
                job.status = TrainingStatus.FAILED
                job.completed_at = time.time()

            # Finish experiment
            await experiment_tracker.finish_experiment(experiment_id)

            return success

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.completed_at = time.time()
            job.last_error = str(e)

            error_context = self.error_handler.handle_error(
                e, "training_worker", "execute_job"
            )
            logger.error(f"Job {job.job_id} failed: {error_context.error_id}")

            return False
        finally:
            self.current_job = None

    async def _execute_training_with_retries(
        self,
        job: TrainingJob,
        experiment_tracker: ExperimentTracker,
        experiment_id: str,
    ) -> bool:
        """Execute training with automatic retries"""

        @retry_async(
            RetryConfig(
                max_attempts=job.config.max_retries + 1,
                base_delay=10.0,
                exponential_backoff=True,
            )
        )
        async def _training_loop():
            return await self._run_training_loop(job, experiment_tracker, experiment_id)

        try:
            return await _training_loop()
        except Exception as e:
            logger.error(f"Training failed after {job.config.max_retries} retries: {e}")
            return False

    async def _run_training_loop(
        self,
        job: TrainingJob,
        experiment_tracker: ExperimentTracker,
        experiment_id: str,
    ) -> bool:
        """Main training loop"""

        # Load or initialize model
        model = await self._load_or_create_model(job)

        # Setup optimizer and scheduler
        optimizer = self._create_optimizer(model, job.config)
        scheduler = self._create_scheduler(optimizer, job.config)

        # Load training data
        train_loader = await self._create_data_loader(
            job.config.training_data, job.config.batch_size
        )

        # Early stopping setup
        best_metric = float("-inf")
        patience_counter = 0

        # Training loop
        for epoch in range(job.current_epoch, job.config.num_epochs):
            job.current_epoch = epoch

            # Check for cancellation
            if job.status == TrainingStatus.CANCELLED:
                return False

            epoch_metrics = await self._train_epoch(
                model, optimizer, train_loader, job, experiment_tracker, experiment_id
            )

            # Update scheduler
            if scheduler:
                scheduler.step()

            # Log epoch metrics
            await experiment_tracker.log_metrics(experiment_id, epoch_metrics, epoch)

            # Early stopping check
            if job.config.enable_early_stopping:
                current_metric = epoch_metrics.get(
                    "validation_reward", epoch_metrics.get("train_reward", 0)
                )
                if current_metric > best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= job.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered for job {job.job_id}")
                    break

            # Checkpointing
            if (
                job.config.enable_checkpointing
                and epoch % job.config.checkpoint_frequency == 0
            ):
                checkpoint_path = (
                    Path(job.checkpoint_path) / f"checkpoint_epoch_{epoch}.pt"
                )
                await self._save_checkpoint(job, model, optimizer, str(checkpoint_path))
                await experiment_tracker.log_artifact(
                    experiment_id, str(checkpoint_path), "checkpoint"
                )

        return True

    async def _load_or_create_model(self, job: TrainingJob):
        """Load existing model or create new one"""
        # This is a simplified implementation
        # In practice, you would load the actual model based on config

        if job.checkpoint_path and Path(job.checkpoint_path).exists():
            # Load from checkpoint
            checkpoint_files = list(
                Path(job.checkpoint_path).glob("checkpoint_epoch_*.pt")
            )
            if checkpoint_files:
                latest_checkpoint = max(
                    checkpoint_files, key=lambda p: p.stat().st_mtime
                )
                logger.info(f"Loading model from checkpoint: {latest_checkpoint}")
                # Load checkpoint logic here

        # Create new model (placeholder)
        logger.info(f"Creating new model for job {job.job_id}")
        return {"type": "placeholder_model", "config": job.config.model_config}

    def _create_optimizer(self, model, config: TrainingConfig):
        """Create optimizer"""
        # Placeholder implementation
        return {"type": config.optimizer, "lr": config.learning_rate}

    def _create_scheduler(self, optimizer, config: TrainingConfig):
        """Create learning rate scheduler"""
        # Placeholder implementation
        return {"type": config.scheduler}

    async def _create_data_loader(self, training_data, batch_size: int):
        """Create data loader"""
        # Placeholder implementation
        return {"data_path": training_data, "batch_size": batch_size}

    async def _train_epoch(
        self,
        model,
        optimizer,
        train_loader,
        job: TrainingJob,
        experiment_tracker: ExperimentTracker,
        experiment_id: str,
    ) -> Dict[str, float]:
        """Train for one epoch"""

        # Simulate training steps
        num_steps = 100  # Placeholder
        epoch_loss = 0.0
        epoch_reward = 0.0

        for step in range(num_steps):
            job.current_step += 1

            # Simulate training step
            await asyncio.sleep(0.01)  # Simulate computation time

            # Simulate metrics
            step_loss = 0.5 + (0.1 * (1 - step / num_steps))  # Decreasing loss
            step_reward = 0.3 + (0.2 * (step / num_steps))  # Increasing reward

            epoch_loss += step_loss
            epoch_reward += step_reward

            # Log step metrics periodically
            if step % job.config.log_frequency == 0:
                step_metrics = {
                    "step_loss": step_loss,
                    "step_reward": step_reward,
                    "learning_rate": optimizer["lr"],
                }
                await experiment_tracker.log_metrics(
                    experiment_id, step_metrics, job.current_step
                )

            # Monitor performance
            self.monitoring.record_training_iteration(
                job.config.agent_type,
                "grpo",
                {"loss": step_loss, "reward": step_reward},
            )

        # Return epoch metrics
        return {
            "train_loss": epoch_loss / num_steps,
            "train_reward": epoch_reward / num_steps,
            "epoch": job.current_epoch,
        }

    async def _save_checkpoint(
        self, job: TrainingJob, model, optimizer, checkpoint_path: str
    ):
        """Save training checkpoint"""
        checkpoint_data = {
            "job_id": job.job_id,
            "epoch": job.current_epoch,
            "step": job.current_step,
            "model_state": model,  # Placeholder
            "optimizer_state": optimizer,  # Placeholder
            "config": job.config.__dict__,
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    async def _save_model(self, job: TrainingJob, model_path: str):
        """Save final trained model"""
        # Placeholder implementation
        model_data = {
            "job_id": job.job_id,
            "config": job.config.__dict__,
            "final_metrics": job.metrics,
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved: {model_path}")

    def _compute_final_metrics(self, job: TrainingJob) -> Dict[str, float]:
        """Compute final training metrics"""
        return {
            "final_epoch": job.current_epoch,
            "total_steps": job.current_step,
            "training_time": job.runtime or 0,
            "avg_loss": 0.3,  # Placeholder
            "final_reward": 0.8,  # Placeholder
        }


class AdvancedTrainingOrchestrator:
    """Main training orchestrator service"""

    def __init__(
        self,
        max_concurrent_jobs: int = 4,
        scheduling_strategy: SchedulingStrategy = SchedulingStrategy.RESOURCE_AWARE,
        enable_experiment_tracking: bool = True,
        start_background_tasks: bool = True,
    ):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.resource_manager = ResourceManager()
        self.scheduler = JobScheduler(scheduling_strategy)
        self.experiment_tracker = (
            ExperimentTracker() if enable_experiment_tracking else None
        )
        self._background_tasks_enabled = start_background_tasks

        # Worker management
        self.workers: Dict[str, TrainingWorker] = {}
        self.worker_tasks: Dict[str, asyncio.Task] = {}

        # State management
        self.state_service = get_state_service()
        self.monitoring = get_monitoring_service()

        # Background tasks
        self._orchestration_task = None
        self._monitoring_task = None

        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background orchestration tasks"""
        if not self._background_tasks_enabled:
            return
        if self._orchestration_task or self._monitoring_task:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover
            # Constructed outside an event loop (tests or import-time); start later.
            logger.debug("No running event loop; skipping background task startup.")
            return

        self._orchestration_task = loop.create_task(self._orchestration_loop())
        self._monitoring_task = loop.create_task(self._monitoring_loop())

    async def submit_training_job(
        self, config: TrainingConfig, priority: int = 1, user_id: str = None
    ) -> str:
        """Submit a new training job"""
        self._start_background_tasks()
        job_id = str(uuid.uuid4())

        job = TrainingJob(
            job_id=job_id, config=config, priority=priority, user_id=user_id
        )

        # Store job in state
        await self.state_service.state_manager.set(
            f"training_job:{job_id}", job.__dict__
        )

        # Submit to scheduler
        await self.scheduler.submit_job(job)

        logger.info(f"Training job submitted: {job_id}")
        return job_id

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get training job status"""
        self._start_background_tasks()
        job_data = await self.state_service.state_manager.get(f"training_job:{job_id}")
        if not job_data:
            return None

        # Convert back to TrainingJob object for runtime calculation
        job = TrainingJob(**job_data)

        return {
            "job_id": job_id,
            "status": job.status.value,
            "progress": {
                "current_epoch": job.current_epoch,
                "total_epochs": job.config.num_epochs,
                "current_step": job.current_step,
                "progress_percent": (job.current_epoch / job.config.num_epochs) * 100,
            },
            "runtime": job.runtime,
            "estimated_completion": job.estimated_completion,
            "assigned_resources": job.assigned_resources,
            "metrics": job.metrics,
            "last_error": job.last_error,
        }

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        self._start_background_tasks()
        success = await self.scheduler.cancel_job(job_id)

        if success:
            # Update job status in state
            job_data = await self.state_service.state_manager.get(
                f"training_job:{job_id}"
            )
            if job_data:
                job_data["status"] = TrainingStatus.CANCELLED.value
                await self.state_service.state_manager.set(
                    f"training_job:{job_id}", job_data
                )

        return success

    async def get_system_status(self) -> Dict[str, Any]:
        """Get orchestrator system status"""
        self._start_background_tasks()
        resource_utilization = self.resource_manager.get_resource_utilization()
        queue_status = self.scheduler.get_queue_status()

        return {
            "resource_utilization": {
                rt.value: util for rt, util in resource_utilization.items()
            },
            "queue_status": queue_status,
            "active_workers": len(self.workers),
            "running_tasks": len(self.worker_tasks),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "available_resources": {
                rt.value: amount
                for rt, amount in self.resource_manager.available_resources.items()
            },
        }

    async def _orchestration_loop(self):
        """Main orchestration loop"""
        while True:
            try:
                # Check if we can start new jobs
                active_jobs = len(
                    [w for w in self.workers.values() if w.current_job is not None]
                )

                if active_jobs < self.max_concurrent_jobs:
                    # Get next job from scheduler
                    next_job = await self.scheduler.get_next_job(self.resource_manager)

                    if next_job:
                        # Allocate resources
                        if await self.resource_manager.allocate_resources(
                            next_job.job_id, next_job.config.resource_requirements
                        ):
                            # Start job
                            await self._start_job(next_job)
                        else:
                            # Put job back in queue
                            await self.scheduler.submit_job(next_job)

                # Clean up completed tasks
                await self._cleanup_completed_tasks()

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(30)

    async def _start_job(self, job: TrainingJob):
        """Start a training job"""
        worker_id = f"worker_{job.job_id}"
        worker = TrainingWorker(worker_id)

        self.workers[worker_id] = worker
        self.scheduler.running_jobs[job.job_id] = job

        # Create and start worker task
        task = asyncio.create_task(worker.execute_job(job, self.experiment_tracker))
        self.worker_tasks[worker_id] = task

        logger.info(f"Started training job {job.job_id} on worker {worker_id}")

    async def _cleanup_completed_tasks(self):
        """Clean up completed worker tasks"""
        completed_workers = []

        for worker_id, task in self.worker_tasks.items():
            if task.done():
                completed_workers.append(worker_id)

                # Get job and deallocate resources
                worker = self.workers[worker_id]
                if worker.current_job:
                    job = worker.current_job
                    await self.resource_manager.deallocate_resources(job.job_id)

                    # Move job to completed
                    if job.job_id in self.scheduler.running_jobs:
                        completed_job = self.scheduler.running_jobs.pop(job.job_id)
                        self.scheduler.completed_jobs[job.job_id] = completed_job

                    # Update job in state
                    await self.state_service.state_manager.set(
                        f"training_job:{job.job_id}", job.__dict__
                    )

        # Remove completed workers
        for worker_id in completed_workers:
            del self.workers[worker_id]
            del self.worker_tasks[worker_id]

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                # Record system metrics
                resource_util = self.resource_manager.get_resource_utilization()

                for resource_type, utilization in resource_util.items():
                    self.monitoring.metrics_collector.record_metric(
                        f"orchestrator.resource_utilization.{resource_type.value}",
                        utilization,
                    )

                # Record queue metrics
                queue_status = self.scheduler.get_queue_status()
                for metric_name, value in queue_status.items():
                    if isinstance(value, (int, float)):
                        self.monitoring.metrics_collector.record_metric(
                            f"orchestrator.queue.{metric_name}", value
                        )

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down training orchestrator...")

        # Cancel all background tasks
        if self._orchestration_task:
            self._orchestration_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()

        # Cancel all running jobs
        for job_id in list(self.scheduler.running_jobs.keys()):
            await self.cancel_job(job_id)

        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks.values(), return_exceptions=True)

        logger.info("Training orchestrator shutdown complete")


# Global orchestrator instance
_orchestrator: Optional[AdvancedTrainingOrchestrator] = None


def get_training_orchestrator() -> AdvancedTrainingOrchestrator:
    """Get or create global training orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AdvancedTrainingOrchestrator()
    return _orchestrator


if __name__ == "__main__":
    # Example usage
    async def main():
        orchestrator = AdvancedTrainingOrchestrator()

        # Create training configuration
        config = TrainingConfig(
            experiment_name="test_grpo_training",
            agent_type="MultiTurnAgent",
            model_config={"model_type": "gpt2", "model_name": "gpt2"},
            training_data="/path/to/training/data",
            num_epochs=5,
            batch_size=16,
            resource_requirements=[
                ResourceRequirement(ResourceType.CPU, 2.0),
                ResourceRequirement(ResourceType.MEMORY, 4.0),
                ResourceRequirement(ResourceType.GPU, 1.0),
            ],
        )

        # Submit job
        job_id = await orchestrator.submit_training_job(config, priority=1)
        print(f"Submitted job: {job_id}")

        # Monitor progress
        while True:
            status = await orchestrator.get_job_status(job_id)
            if status:
                print(
                    f"Job {job_id}: {status['status']} - {status['progress']['progress_percent']:.1f}%"
                )

                if status["status"] in ["completed", "failed", "cancelled"]:
                    break

            await asyncio.sleep(5)

        # Get system status
        system_status = await orchestrator.get_system_status()
        print("System Status:", json.dumps(system_status, indent=2))

        # Shutdown
        await orchestrator.shutdown()

    asyncio.run(main())
