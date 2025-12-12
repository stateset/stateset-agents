"""
Distributed Training Support for GRPO Agent Framework

This module provides distributed training capabilities including:
- Multi-GPU training with DDP
- Multi-node training
- Gradient accumulation across devices
- Efficient communication strategies
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_checkpoint
from accelerate import Accelerator, DistributedType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from stateset_agents.core.agent import MultiTurnAgent
from stateset_agents.core.environment import Environment
from stateset_agents.core.reward import RewardFunction

from .config import TrainingConfig
from .trainer import MultiTurnGRPOTrainer

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training"""

    # Basic distributed settings
    backend: str = "nccl"  # nccl, gloo, mpi
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"

    # Training strategy
    strategy: str = "ddp"  # ddp, fsdp, deepspeed
    gradient_accumulation_steps: int = 1

    # FSDP specific
    fsdp_sharding_strategy: str = "full_shard"  # full_shard, shard_grad_op, no_shard
    fsdp_cpu_offload: bool = False
    fsdp_backward_prefetch: bool = True
    fsdp_mixed_precision: bool = True

    # DeepSpeed specific
    deepspeed_config: Optional[Dict[str, Any]] = None

    # Communication optimization
    gradient_as_bucket_view: bool = True
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True

    # Checkpointing
    checkpoint_parallel: bool = True
    checkpoint_dir: str = "./distributed_checkpoints"


class DistributedTrainer(MultiTurnGRPOTrainer):
    """
    Distributed GRPO trainer supporting multi-GPU and multi-node training
    """

    def __init__(
        self,
        agent: MultiTurnAgent,
        environment: Environment,
        reward_fn: Optional[RewardFunction] = None,
        config: Optional[TrainingConfig] = None,
        distributed_config: Optional[DistributedConfig] = None,
        **kwargs,
    ):
        self.distributed_config = distributed_config or DistributedConfig()
        self.accelerator = None
        self.is_main_process = True

        # Initialize distributed environment
        self._init_distributed()

        # Initialize base trainer
        super().__init__(agent, environment, reward_fn, config, **kwargs)

    def _init_distributed(self):
        """Initialize distributed training environment"""

        if self.distributed_config.strategy == "accelerate":
            # Use Accelerate for automatic distributed setup
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.distributed_config.gradient_accumulation_steps,
                mixed_precision="bf16"
                if self.config.bf16
                else "fp16"
                if self.config.fp16
                else "no",
                log_with=["wandb"] if self.config.report_to == "wandb" else None,
            )
            self.is_main_process = self.accelerator.is_main_process

        else:
            # Manual distributed setup
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                self.distributed_config.rank = int(os.environ["RANK"])
                self.distributed_config.world_size = int(os.environ["WORLD_SIZE"])
                self.distributed_config.local_rank = int(os.environ["LOCAL_RANK"])

            if self.distributed_config.world_size > 1:
                torch.cuda.set_device(self.distributed_config.local_rank)
                dist.init_process_group(
                    backend=self.distributed_config.backend,
                    world_size=self.distributed_config.world_size,
                    rank=self.distributed_config.rank,
                )
                self.is_main_process = self.distributed_config.rank == 0

                logger.info(
                    f"Initialized distributed training: "
                    f"rank {self.distributed_config.rank}/{self.distributed_config.world_size}"
                )

    async def initialize(self):
        """Initialize distributed trainer"""
        await super().initialize()

        # Wrap model for distributed training
        if self.distributed_config.world_size > 1:
            if self.distributed_config.strategy == "fsdp":
                self._wrap_model_fsdp()
            elif self.distributed_config.strategy == "ddp":
                self._wrap_model_ddp()
            elif self.distributed_config.strategy == "deepspeed":
                self._wrap_model_deepspeed()
            elif self.distributed_config.strategy == "accelerate":
                self._wrap_model_accelerate()

    def _wrap_model_ddp(self):
        """Wrap model with DistributedDataParallel"""
        self.agent.model = DDP(
            self.agent.model,
            device_ids=[self.distributed_config.local_rank],
            output_device=self.distributed_config.local_rank,
            gradient_as_bucket_view=self.distributed_config.gradient_as_bucket_view,
            find_unused_parameters=self.distributed_config.find_unused_parameters,
            broadcast_buffers=self.distributed_config.broadcast_buffers,
        )
        logger.info("Model wrapped with DistributedDataParallel")

    def _wrap_model_fsdp(self):
        """Wrap model with FullyShardedDataParallel"""

        # Configure FSDP
        cpu_offload = CPUOffload(
            offload_params=self.distributed_config.fsdp_cpu_offload
        )

        backward_prefetch = None
        if self.distributed_config.fsdp_backward_prefetch:
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE

        mixed_precision = None
        if self.distributed_config.fsdp_mixed_precision:
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                reduce_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                buffer_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            )

        # Wrap model
        self.agent.model = FSDP(
            self.agent.model,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            mixed_precision=mixed_precision,
            sharding_strategy=self.distributed_config.fsdp_sharding_strategy,
            device_id=torch.cuda.current_device(),
        )
        logger.info("Model wrapped with FullyShardedDataParallel")

    def _wrap_model_deepspeed(self):
        """Wrap model with DeepSpeed"""
        try:
            import deepspeed

            # Initialize DeepSpeed
            (
                self.agent.model,
                self.optimizer,
                _,
                self.lr_scheduler,
            ) = deepspeed.initialize(
                model=self.agent.model,
                optimizer=self.optimizer,
                config=self.distributed_config.deepspeed_config,
                model_parameters=self.agent.model.parameters(),
            )
            logger.info("Model wrapped with DeepSpeed")

        except ImportError:
            logger.error("DeepSpeed not installed. Install with: pip install deepspeed")
            raise

    def _wrap_model_accelerate(self):
        """Wrap model with Accelerate"""
        self.agent.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.agent.model, self.optimizer, self.lr_scheduler
        )
        logger.info("Model wrapped with Accelerate")

    async def training_step(self, trajectory_groups: List[Any]) -> Dict[str, Any]:
        """Execute distributed training step"""

        if self.distributed_config.strategy == "accelerate":
            # Use Accelerate's gradient accumulation
            with self.accelerator.accumulate(self.agent.model):
                metrics = await super().training_step(trajectory_groups)
        else:
            metrics = await super().training_step(trajectory_groups)

        # Synchronize metrics across processes
        if self.distributed_config.world_size > 1:
            metrics = self._sync_metrics(metrics)

        return metrics

    def _sync_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize metrics across all processes"""

        synced_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor = torch.tensor(value, device=self.agent.model.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                synced_metrics[key] = tensor.item()
            else:
                synced_metrics[key] = value

        return synced_metrics

    async def save_checkpoint(
        self, is_best: bool = False, checkpoint_name: Optional[str] = None
    ):
        """Save distributed checkpoint"""

        if not self.is_main_process and not self.distributed_config.checkpoint_parallel:
            return

        if self.distributed_config.checkpoint_parallel:
            # Parallel checkpoint saving
            await self._save_distributed_checkpoint(is_best, checkpoint_name)
        else:
            # Only main process saves
            await super().save_checkpoint(is_best, checkpoint_name)

    async def _save_distributed_checkpoint(
        self, is_best: bool = False, checkpoint_name: Optional[str] = None
    ):
        """Save checkpoint in distributed manner"""

        checkpoint_name = checkpoint_name or f"checkpoint-{self.global_step}"
        checkpoint_path = Path(self.distributed_config.checkpoint_dir) / checkpoint_name

        # Create checkpoint directory
        if self.is_main_process:
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Wait for directory creation
        if self.distributed_config.world_size > 1:
            dist.barrier()

        # Save model state
        if self.distributed_config.strategy == "fsdp":
            # FSDP checkpoint saving
            dist_checkpoint.save_state_dict(
                state_dict=self.agent.model.state_dict(),
                storage_writer=dist_checkpoint.FileSystemWriter(str(checkpoint_path)),
            )
        else:
            # Regular checkpoint saving
            if self.is_main_process:
                model_to_save = (
                    self.agent.model.module
                    if hasattr(self.agent.model, "module")
                    else self.agent.model
                )
                model_to_save.save_pretrained(checkpoint_path)
                self.agent.tokenizer.save_pretrained(checkpoint_path)

        # Save training state (main process only)
        if self.is_main_process:
            training_state = {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "best_eval_metric": self.best_eval_metric,
                "distributed_config": self.distributed_config.__dict__,
            }
            torch.save(training_state, checkpoint_path / "training_state.pt")

        logger.info(f"Distributed checkpoint saved: {checkpoint_path}")

    def cleanup(self):
        """Cleanup distributed resources"""
        if self.distributed_config.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

        if self.accelerator:
            self.accelerator.end_training()


# Utility functions for distributed training


def get_distributed_config(
    num_gpus: Optional[int] = None, strategy: str = "ddp", **kwargs
) -> DistributedConfig:
    """Create distributed configuration based on available resources"""

    config = DistributedConfig(strategy=strategy, **kwargs)

    # Auto-detect GPUs if not specified
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    config.world_size = num_gpus

    # Set strategy-specific defaults
    if strategy == "fsdp" and num_gpus >= 4:
        config.fsdp_sharding_strategy = "full_shard"
        config.fsdp_cpu_offload = True
    elif strategy == "deepspeed":
        config.deepspeed_config = get_default_deepspeed_config(num_gpus)

    return config


def get_default_deepspeed_config(num_gpus: int) -> Dict[str, Any]:
    """Get default DeepSpeed configuration"""

    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "zero_optimization": {
            "stage": 2 if num_gpus >= 4 else 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
    }


async def train_distributed(
    agent: MultiTurnAgent,
    environment: Environment,
    config: TrainingConfig,
    num_gpus: Optional[int] = None,
    strategy: str = "ddp",
    **kwargs,
) -> MultiTurnAgent:
    """High-level distributed training function"""

    # Create distributed config
    dist_config = get_distributed_config(num_gpus, strategy)

    # Create distributed trainer
    trainer = DistributedTrainer(
        agent=agent,
        environment=environment,
        config=config,
        distributed_config=dist_config,
        **kwargs,
    )

    try:
        # Initialize and train
        await trainer.initialize()
        trained_agent = await trainer.train()
        return trained_agent

    finally:
        # Cleanup
        trainer.cleanup()
