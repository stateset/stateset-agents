"""
Value objects and serialization helpers for the advanced training orchestrator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrainingStatus(Enum):
    """Training job status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResourceType(Enum):
    """Resource types for training."""

    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


class SchedulingStrategy(Enum):
    """Job scheduling strategies."""

    FIFO = "fifo"
    PRIORITY = "priority"
    FAIR_SHARE = "fair_share"
    SHORTEST_JOB_FIRST = "shortest_job_first"
    RESOURCE_AWARE = "resource_aware"


@dataclass
class ResourceRequirement:
    """Resource requirement specification."""

    resource_type: ResourceType
    amount: float
    min_amount: float | None = None
    max_amount: float | None = None
    priority: int = 1


@dataclass
class TrainingConfig:
    """Comprehensive training configuration."""

    experiment_name: str
    agent_type: str
    model_config: dict[str, Any]
    training_data: str | list[str]

    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    grpo_epsilon: float = 0.2
    grpo_value_loss_coef: float = 0.5
    grpo_entropy_coef: float = 0.01

    resource_requirements: list[ResourceRequirement] = field(default_factory=list)
    max_runtime: int | None = None

    enable_checkpointing: bool = True
    checkpoint_frequency: int = 100
    max_retries: int = 3

    enable_wandb: bool = True
    enable_mlflow: bool = False
    log_frequency: int = 10

    enable_early_stopping: bool = True
    early_stopping_patience: int = 5
    enable_model_pruning: bool = False
    enable_quantization: bool = False


@dataclass
class TrainingJob:
    """Training job definition."""

    job_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    priority: int = 1
    user_id: str | None = None

    assigned_resources: dict[str, Any] = field(default_factory=dict)
    worker_nodes: list[str] = field(default_factory=list)
    checkpoint_path: str | None = None

    current_epoch: int = 0
    current_step: int = 0
    metrics: dict[str, list[float]] = field(default_factory=dict)

    retry_count: int = 0
    last_error: str | None = None

    @property
    def runtime(self) -> float | None:
        """Get job runtime in seconds."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    @property
    def estimated_completion(self) -> float | None:
        """Estimate completion time based on progress."""
        if self.current_epoch == 0 or self.runtime is None:
            return None

        epochs_remaining = self.config.num_epochs - self.current_epoch
        time_per_epoch = self.runtime / self.current_epoch
        return time.time() + (epochs_remaining * time_per_epoch)


def _coerce_resource_type(value: ResourceType | str) -> ResourceType:
    if isinstance(value, ResourceType):
        return value
    return ResourceType(str(value))


def _serialize_resource_requirements(
    requirements: list[ResourceRequirement],
) -> list[dict[str, Any]]:
    serialized = []
    for requirement in requirements:
        serialized.append(
            {
                "resource_type": requirement.resource_type.value,
                "amount": requirement.amount,
                "min_amount": requirement.min_amount,
                "max_amount": requirement.max_amount,
                "priority": requirement.priority,
            }
        )
    return serialized


def deserialize_training_config(config_data: TrainingConfig | dict[str, Any]) -> TrainingConfig:
    """Convert serialized config data back into a TrainingConfig."""
    if isinstance(config_data, TrainingConfig):
        return config_data

    raw_requirements = list(config_data.get("resource_requirements", []))
    requirements = []
    for requirement in raw_requirements:
        if isinstance(requirement, ResourceRequirement):
            requirements.append(requirement)
            continue
        requirement_dict = dict(requirement)
        requirement_dict["resource_type"] = _coerce_resource_type(
            requirement_dict["resource_type"]
        )
        requirements.append(ResourceRequirement(**requirement_dict))

    config_kwargs = dict(config_data)
    config_kwargs["resource_requirements"] = requirements
    return TrainingConfig(**config_kwargs)


def serialize_training_config(config: TrainingConfig) -> dict[str, Any]:
    """Convert a TrainingConfig into a JSON-friendly dict."""
    config_dict = dict(config.__dict__)
    config_dict["resource_requirements"] = _serialize_resource_requirements(
        list(config.resource_requirements)
    )
    return config_dict


def deserialize_training_job(job_data: TrainingJob | dict[str, Any]) -> TrainingJob:
    """Convert serialized job data back into a TrainingJob."""
    if isinstance(job_data, TrainingJob):
        return job_data

    job_kwargs = dict(job_data)
    job_kwargs["config"] = deserialize_training_config(job_kwargs["config"])
    status_value = job_kwargs.get("status", TrainingStatus.PENDING)
    if not isinstance(status_value, TrainingStatus):
        job_kwargs["status"] = TrainingStatus(str(status_value))
    return TrainingJob(**job_kwargs)


def serialize_training_job(job: TrainingJob) -> dict[str, Any]:
    """Convert a TrainingJob into a JSON-friendly dict."""
    job_dict = dict(job.__dict__)
    job_dict["status"] = job.status.value
    job_dict["config"] = serialize_training_config(job.config)
    return job_dict


__all__ = [
    "ResourceRequirement",
    "ResourceType",
    "SchedulingStrategy",
    "TrainingConfig",
    "TrainingJob",
    "TrainingStatus",
    "deserialize_training_config",
    "deserialize_training_job",
    "serialize_training_config",
    "serialize_training_job",
]
