"""
Distributed Multi-GPU Training for GRPO Agent Framework

This module provides advanced distributed training capabilities with
proper rank handling, memory optimization, and fault tolerance.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from torch.distributed.elastic.multiprocessing.errors import record
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from stateset_agents.core.agent import Agent
from stateset_agents.core.environment import Environment
from stateset_agents.core.reward import RewardFunction
from utils.monitoring import MonitoringService

from .config import TrainingConfig
from .trainer import GRPOTrainer

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training"""

    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"

    # Training specific
    gradient_accumulation_steps: int = 1
    sync_bn: bool = True
    find_unused_parameters: bool = False

    # Fault tolerance
    max_restarts: int = 3
    restart_interval: int = 60

    # Memory optimization
    cpu_offload: bool = False
    activation_checkpointing: bool = False
    mixed_precision: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "init_method": self.init_method,
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "sync_bn": self.sync_bn,
            "find_unused_parameters": self.find_unused_parameters,
            "max_restarts": self.max_restarts,
            "restart_interval": self.restart_interval,
            "cpu_offload": self.cpu_offload,
            "activation_checkpointing": self.activation_checkpointing,
            "mixed_precision": self.mixed_precision,
        }


class DistributedGRPOTrainer:
    """
    Distributed GRPO trainer with advanced multi-GPU capabilities
    """

    def __init__(
        self,
        agent: Agent,
        environment: Environment,
        reward_function: RewardFunction,
        training_config: TrainingConfig,
        distributed_config: DistributedConfig,
        monitoring_service: Optional[MonitoringService] = None,
    ):
        if not DISTRIBUTED_AVAILABLE:
            raise ImportError("PyTorch distributed training is not available")

        self.agent = agent
        self.environment = environment
        self.reward_function = reward_function
        self.training_config = training_config
        self.distributed_config = distributed_config
        self.monitoring = monitoring_service

        # Training state
        self.is_initialized = False
        self.is_master = False
        self.device = None
        self.model = None
        self.optimizer = None
        self.scaler = None

        # Metrics
        self.training_metrics = {}
        self.step_count = 0
        self.epoch_count = 0

    def setup_distributed(self, rank: int, world_size: int):
        """Setup distributed training environment"""
        # Set environment variables
        os.environ["MASTER_ADDR"] = self.distributed_config.master_addr
        os.environ["MASTER_PORT"] = self.distributed_config.master_port
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)

        # Initialize process group
        dist.init_process_group(
            backend=self.distributed_config.backend,
            init_method=self.distributed_config.init_method,
            world_size=world_size,
            rank=rank,
        )

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # Set master process
        self.is_master = rank == 0

        # Update configs
        self.distributed_config.rank = rank
        self.distributed_config.world_size = world_size
        self.distributed_config.local_rank = rank

        self.is_initialized = True

        if self.is_master:
            logger.info(
                f"Initialized distributed training: rank={rank}, world_size={world_size}"
            )

    def setup_model(self):
        """Setup model for distributed training"""
        if not self.is_initialized:
            raise RuntimeError("Distributed training not initialized")

        # Get model from agent
        self.model = self.agent.get_model()

        # Move to device
        self.model.to(self.device)

        # Enable gradient checkpointing for memory efficiency
        if self.distributed_config.activation_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Wrap with DDP
        self.model = DDP(
            self.model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            output_device=self.device.index if self.device.type == "cuda" else None,
            find_unused_parameters=self.distributed_config.find_unused_parameters,
        )

        # Setup optimizer
        self.optimizer = self.agent.get_optimizer()

        # Setup mixed precision scaler
        if self.distributed_config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()

        if self.is_master:
            logger.info("Model setup complete for distributed training")

    def setup_wandb(self):
        """Setup Weights & Biases for distributed training"""
        if not WANDB_AVAILABLE or not self.is_master:
            return

        # Initialize wandb only on master process
        wandb.init(
            project=self.training_config.wandb_project,
            name=f"grpo-distributed-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                **self.training_config.to_dict(),
                **self.distributed_config.to_dict(),
            },
        )

        # Log model architecture
        if hasattr(self.model, "module"):
            wandb.watch(self.model.module)
        else:
            wandb.watch(self.model)

    async def train(self) -> Dict[str, Any]:
        """
        Main training loop with distributed coordination
        """
        if not self.is_initialized:
            raise RuntimeError("Distributed training not initialized")

        # Setup wandb
        self.setup_wandb()

        # Training loop
        for epoch in range(self.training_config.num_epochs):
            self.epoch_count = epoch

            # Train epoch
            epoch_metrics = await self._train_epoch()

            # Log metrics (only master process)
            if self.is_master:
                await self._log_metrics(epoch_metrics, epoch)

            # Synchronize processes
            dist.barrier()

            # Save checkpoint (only master process)
            if self.is_master and epoch % self.training_config.save_interval == 0:
                await self._save_checkpoint(epoch)

        # Final cleanup
        if self.is_master and WANDB_AVAILABLE:
            wandb.finish()

        return self.training_metrics

    async def _train_epoch(self) -> Dict[str, Any]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {
            "loss": 0.0,
            "rewards": [],
            "trajectories": 0,
            "step_time": 0.0,
            "gpu_memory": 0.0,
        }

        # Generate training data
        training_data = await self._generate_training_data()

        # Distribute data across processes
        local_data = self._distribute_data(training_data)

        # Training steps
        for step_idx, batch in enumerate(local_data):
            step_start = asyncio.get_event_loop().time()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(
                enabled=self.distributed_config.mixed_precision
            ):
                loss, step_metrics = await self._training_step(batch)

            # Backward pass
            if self.distributed_config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient synchronization and update
            if (
                step_idx + 1
            ) % self.distributed_config.gradient_accumulation_steps == 0:
                if self.distributed_config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.step_count += 1

            # Update metrics
            epoch_metrics["loss"] += loss.item()
            epoch_metrics["rewards"].extend(step_metrics.get("rewards", []))
            epoch_metrics["trajectories"] += step_metrics.get("trajectories", 0)
            epoch_metrics["step_time"] += asyncio.get_event_loop().time() - step_start

            # Memory tracking
            if torch.cuda.is_available():
                epoch_metrics["gpu_memory"] = (
                    torch.cuda.max_memory_allocated() / 1024**3
                )

        # Average metrics across processes
        epoch_metrics = self._reduce_metrics(epoch_metrics)

        return epoch_metrics

    async def _generate_training_data(self) -> List[Dict[str, Any]]:
        """Generate training data for the epoch"""
        # This should be implemented based on your specific training data requirements
        # For now, return placeholder data
        return [
            {"prompt": f"Training prompt {i}", "response": f"Response {i}"}
            for i in range(self.training_config.batch_size)
        ]

    def _distribute_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Distribute training data across processes"""
        # Simple round-robin distribution
        local_data = []
        for i, item in enumerate(data):
            if i % self.distributed_config.world_size == self.distributed_config.rank:
                local_data.append(item)
        return local_data

    async def _training_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Execute one training step"""
        # Generate trajectories
        trajectories = await self._generate_trajectories(batch)

        # Compute rewards
        rewards = await self._compute_rewards(trajectories)

        # Compute GRPO loss
        loss = self._compute_grpo_loss(trajectories, rewards)

        step_metrics = {
            "rewards": rewards,
            "trajectories": len(trajectories),
            "loss": loss.item(),
        }

        return loss, step_metrics

    async def _generate_trajectories(
        self, batch: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate trajectories for the batch"""
        # Placeholder implementation
        return [{"prompt": batch["prompt"], "response": batch["response"]}]

    async def _compute_rewards(self, trajectories: List[Dict[str, Any]]) -> List[float]:
        """Compute rewards for trajectories"""
        # Placeholder implementation
        return [0.5 for _ in trajectories]

    def _compute_grpo_loss(
        self, trajectories: List[Dict[str, Any]], rewards: List[float]
    ) -> torch.Tensor:
        """Compute GRPO loss"""
        # Placeholder implementation
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _reduce_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce metrics across all processes"""
        # Gather metrics from all processes
        gathered_metrics = [None] * self.distributed_config.world_size
        dist.all_gather_object(gathered_metrics, metrics)

        # Aggregate metrics
        if self.is_master:
            aggregated = {}
            for key in metrics:
                if key == "rewards":
                    aggregated[key] = [r for m in gathered_metrics for r in m[key]]
                elif key == "trajectories":
                    aggregated[key] = sum(m[key] for m in gathered_metrics)
                elif isinstance(metrics[key], (int, float)):
                    aggregated[key] = sum(m[key] for m in gathered_metrics) / len(
                        gathered_metrics
                    )
                else:
                    aggregated[key] = metrics[key]

            return aggregated
        else:
            return metrics

    async def _log_metrics(self, metrics: Dict[str, Any], epoch: int):
        """Log metrics to monitoring services"""
        # Log to wandb
        if WANDB_AVAILABLE and self.is_master:
            wandb_metrics = {
                "epoch": epoch,
                "loss": metrics["loss"],
                "average_reward": sum(metrics["rewards"]) / len(metrics["rewards"])
                if metrics["rewards"]
                else 0,
                "trajectories": metrics["trajectories"],
                "step_time": metrics["step_time"],
                "gpu_memory_gb": metrics["gpu_memory"],
            }
            wandb.log(wandb_metrics)

        # Log to monitoring service
        if self.monitoring and self.is_master:
            await self.monitoring.log_metric("grpo.distributed.loss", metrics["loss"])
            await self.monitoring.log_metric(
                "grpo.distributed.trajectories", metrics["trajectories"]
            )
            await self.monitoring.log_metric(
                "grpo.distributed.gpu_memory", metrics["gpu_memory"]
            )

    async def _save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.training_config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        # Save model state
        model_state = (
            self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_config": self.training_config.to_dict(),
            "distributed_config": self.distributed_config.to_dict(),
            "step_count": self.step_count,
            "training_metrics": self.training_metrics,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def cleanup(self):
        """Cleanup distributed training"""
        if self.is_initialized:
            dist.destroy_process_group()
            logger.info("Distributed training cleanup complete")


@record
def run_distributed_training(
    rank: int,
    world_size: int,
    agent: Agent,
    environment: Environment,
    reward_function: RewardFunction,
    training_config: TrainingConfig,
    distributed_config: DistributedConfig,
    monitoring_service: Optional[MonitoringService] = None,
):
    """
    Entry point for distributed training process
    """
    # Create trainer
    trainer = DistributedGRPOTrainer(
        agent=agent,
        environment=environment,
        reward_function=reward_function,
        training_config=training_config,
        distributed_config=distributed_config,
        monitoring_service=monitoring_service,
    )

    try:
        # Setup distributed environment
        trainer.setup_distributed(rank, world_size)

        # Setup model
        trainer.setup_model()

        # Run training
        asyncio.run(trainer.train())

    except Exception as e:
        logger.error(f"Distributed training failed on rank {rank}: {e}")
        raise
    finally:
        # Cleanup
        trainer.cleanup()


def launch_distributed_training(
    agent: Agent,
    environment: Environment,
    reward_function: RewardFunction,
    training_config: TrainingConfig,
    distributed_config: DistributedConfig,
    monitoring_service: Optional[MonitoringService] = None,
) -> Dict[str, Any]:
    """
    Launch distributed training across multiple GPUs
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for distributed training")

    world_size = distributed_config.world_size
    if world_size <= 0:
        world_size = torch.cuda.device_count()
        distributed_config.world_size = world_size

    logger.info(f"Launching distributed training with {world_size} processes")

    # Launch training processes
    mp.spawn(
        run_distributed_training,
        args=(
            world_size,
            agent,
            environment,
            reward_function,
            training_config,
            distributed_config,
            monitoring_service,
        ),
        nprocs=world_size,
        join=True,
    )

    return {
        "status": "completed",
        "world_size": world_size,
        "distributed_config": distributed_config.to_dict(),
    }


# Utility functions
def get_optimal_distributed_config(
    num_gpus: Optional[int] = None, memory_per_gpu: Optional[float] = None
) -> DistributedConfig:
    """Get optimal distributed configuration for available hardware"""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if memory_per_gpu is None:
        memory_per_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # Determine optimal settings based on hardware
    config = DistributedConfig()
    config.world_size = num_gpus

    # Memory optimization based on GPU memory
    if memory_per_gpu < 8:  # Less than 8GB
        config.gradient_accumulation_steps = 4
        config.mixed_precision = True
        config.activation_checkpointing = True
    elif memory_per_gpu < 16:  # Less than 16GB
        config.gradient_accumulation_steps = 2
        config.mixed_precision = True
        config.activation_checkpointing = False
    else:  # 16GB or more
        config.gradient_accumulation_steps = 1
        config.mixed_precision = False
        config.activation_checkpointing = False

    return config


def estimate_memory_usage(
    model_parameters: int,
    batch_size: int,
    sequence_length: int,
    mixed_precision: bool = True,
) -> float:
    """Estimate GPU memory usage in GB"""
    # Model parameters
    param_memory = model_parameters * 4 / 1024**3  # 4 bytes per param

    # Gradients
    grad_memory = param_memory

    # Optimizer states (Adam)
    optimizer_memory = param_memory * 2

    # Activations (rough estimate)
    activation_memory = batch_size * sequence_length * 1024 * 4 / 1024**3

    # Mixed precision reduces some memory usage
    if mixed_precision:
        total_memory = (
            param_memory
            + grad_memory
            + optimizer_memory * 0.5
            + activation_memory * 0.5
        )
    else:
        total_memory = param_memory + grad_memory + optimizer_memory + activation_memory

    # Add 20% safety margin
    return total_memory * 1.2
