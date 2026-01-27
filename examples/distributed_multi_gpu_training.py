"""
Distributed Multi-GPU Training Example for StateSet Agents

This example demonstrates how to train a conversational agent using multiple GPUs
with PyTorch Distributed Data Parallel (DDP). It covers:

1. Setting up distributed training across multiple GPUs
2. Configuring gradient accumulation and mixed precision
3. Synchronizing training across processes
4. Saving and loading distributed checkpoints
5. Monitoring and logging in a distributed environment

Requirements:
    - Multiple GPUs (or can be tested with CPU)
    - PyTorch with distributed support
    - pip install stateset-agents[dev]

Usage:
    # Single machine, multiple GPUs
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        examples/distributed_multi_gpu_training.py \
        --model gpt2 \
        --task customer_service

    # Multiple machines (run on each machine)
    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr="192.168.1.1" \
        --master_port=12355 \
        examples/distributed_multi_gpu_training.py

    # Test with CPU (for debugging)
    python examples/distributed_multi_gpu_training.py --cpu --world-size 2
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Framework imports
from stateset_agents import MultiTurnAgent
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import (
    CompositeReward,
    create_customer_service_reward,
    create_domain_reward,
)

try:
    from stateset_agents.training.distributed_trainer import DistributedGRPOTrainer, DistributedConfig
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.utils.monitoring import MonitoringService
except ImportError:
    print("Error: This example requires a full installation from source.")
    print("Please run: pip install -e .[dev]")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Task-Specific Scenarios
# ============================================================================

CUSTOMER_SERVICE_SCENARIOS = [
    {
        "topic": "refund_request",
        "user_goal": "Get a refund for delayed order",
        "context": "Order #12345, delayed by 2 weeks, high-value customer",
        "difficulty": "medium"
    },
    {
        "topic": "technical_issue",
        "user_goal": "Resolve login problem",
        "context": "Account locked after multiple failed attempts",
        "difficulty": "easy"
    },
    {
        "topic": "product_inquiry",
        "user_goal": "Learn about premium features",
        "context": "Existing customer considering upgrade",
        "difficulty": "medium"
    },
    {
        "topic": "complaint_handling",
        "user_goal": "Express dissatisfaction with service",
        "context": "Multiple support tickets unresolved",
        "difficulty": "hard"
    },
]

TECHNICAL_SUPPORT_SCENARIOS = [
    {
        "topic": "debugging",
        "user_goal": "Fix memory leak in Python app",
        "context": "Application crashes after 24 hours",
        "difficulty": "hard"
    },
    {
        "topic": "installation",
        "user_goal": "Install package with dependencies",
        "context": "Complex dependency conflicts",
        "difficulty": "medium"
    },
    {
        "topic": "api_usage",
        "user_goal": "Understand API authentication",
        "context": "Getting 401 errors consistently",
        "difficulty": "easy"
    },
]

SALES_SCENARIOS = [
    {
        "topic": "lead_qualification",
        "user_goal": "Determine if prospect is a good fit",
        "context": "Enterprise customer with complex needs",
        "difficulty": "medium"
    },
    {
        "topic": "product_demo",
        "user_goal": "Showcase key features",
        "context": "Technical decision maker, time-constrained",
        "difficulty": "medium"
    },
]


def get_scenarios(task: str) -> List[Dict]:
    """Get task-specific training scenarios"""
    scenarios_map = {
        "customer_service": CUSTOMER_SERVICE_SCENARIOS,
        "technical_support": TECHNICAL_SUPPORT_SCENARIOS,
        "sales": SALES_SCENARIOS,
    }
    return scenarios_map.get(task, CUSTOMER_SERVICE_SCENARIOS)


# ============================================================================
# Distributed Training Setup
# ============================================================================

def setup_process(rank: int, world_size: int, backend: str = "nccl"):
    """
    Initialize distributed process group

    Args:
        rank: Rank of current process
        world_size: Total number of processes
        backend: Communication backend (nccl for GPU, gloo for CPU)
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Set device for this process
    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    logger.info(f"Process {rank}/{world_size} initialized on {device}")
    return device


def cleanup_process():
    """Clean up distributed process group"""
    dist.destroy_process_group()


# ============================================================================
# Training Function (runs on each process)
# ============================================================================

async def train_distributed(
    rank: int,
    world_size: int,
    args: argparse.Namespace
):
    """
    Main training function that runs on each distributed process

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        args: Command line arguments
    """
    is_master = (rank == 0)

    # Setup distributed environment
    backend = "gloo" if args.cpu else "nccl"
    device = setup_process(rank, world_size, backend)

    if is_master:
        logger.info("=" * 80)
        logger.info("Distributed Training Configuration")
        logger.info("=" * 80)
        logger.info(f"World Size: {world_size}")
        logger.info(f"Backend: {backend}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Task: {args.task}")
        logger.info(f"Mixed Precision: {not args.no_mixed_precision}")
        logger.info(f"Gradient Accumulation Steps: {args.grad_accumulation}")
        logger.info("=" * 80)

    try:
        # ====================================================================
        # 1. Create Agent
        # ====================================================================

        if is_master:
            logger.info(f"[Rank {rank}] Creating agent...")

        agent_config = AgentConfig(
            model_name=args.model,
            temperature=0.7,
            max_new_tokens=256,
            use_lora=args.use_lora,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        agent = MultiTurnAgent(agent_config)
        await agent.initialize()

        # ====================================================================
        # 2. Create Environment
        # ====================================================================

        scenarios = get_scenarios(args.task)

        if is_master:
            logger.info(f"[Rank {rank}] Creating environment with {len(scenarios)} scenarios...")

        # Create task-specific reward function
        if args.task == "customer_service":
            reward_fn = create_customer_service_reward()
        else:
            reward_fn = create_domain_reward(args.task)

        environment = ConversationEnvironment(
            scenarios=scenarios,
            max_turns=6,
            reward_fn=reward_fn,
        )

        # ====================================================================
        # 3. Configure Distributed Training
        # ====================================================================

        training_config = TrainingConfig(
            num_episodes=args.num_episodes // world_size,  # Divide episodes across processes
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_grad_norm=1.0,
            num_epochs=args.num_epochs,
            save_interval=args.save_interval,
            log_interval=10,
            checkpoint_dir=args.checkpoint_dir,
            wandb_project=args.wandb_project if is_master else None,
        )

        distributed_config = DistributedConfig(
            backend=backend,
            world_size=world_size,
            rank=rank,
            local_rank=rank,
            gradient_accumulation_steps=args.grad_accumulation,
            mixed_precision=not args.no_mixed_precision,
            activation_checkpointing=args.checkpoint_activations,
            cpu_offload=args.cpu_offload,
            sync_bn=True,
            find_unused_parameters=False,
        )

        # ====================================================================
        # 4. Create Distributed Trainer
        # ====================================================================

        if is_master:
            logger.info(f"[Rank {rank}] Setting up distributed trainer...")

        # Optional: Setup monitoring (only on master)
        monitoring = None
        if is_master and not args.no_monitoring:
            monitoring = MonitoringService()
            await monitoring.start()

        trainer = DistributedGRPOTrainer(
            agent=agent,
            environment=environment,
            reward_function=reward_fn,
            training_config=training_config,
            distributed_config=distributed_config,
            monitoring_service=monitoring,
        )

        # Setup distributed training
        trainer.setup_distributed(rank, world_size)
        trainer.setup_model()

        # ====================================================================
        # 5. Train
        # ====================================================================

        if is_master:
            logger.info("=" * 80)
            logger.info("Starting Distributed Training")
            logger.info("=" * 80)

        # Synchronize all processes before starting
        dist.barrier()

        training_start = datetime.now()

        # Run training
        metrics = await trainer.train()

        # Synchronize all processes after training
        dist.barrier()

        training_duration = (datetime.now() - training_start).total_seconds()

        # ====================================================================
        # 6. Report Results (Master Only)
        # ====================================================================

        if is_master:
            logger.info("=" * 80)
            logger.info("Training Complete!")
            logger.info("=" * 80)
            logger.info(f"Duration: {training_duration:.2f}s")
            logger.info(f"Final Metrics: {metrics}")
            logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")
            logger.info("=" * 80)

        # Cleanup
        if monitoring:
            await monitoring.stop()

    except Exception as e:
        logger.error(f"[Rank {rank}] Training failed: {e}", exc_info=True)
        raise

    finally:
        # Clean up distributed process group
        cleanup_process()


# ============================================================================
# Process Launcher
# ============================================================================

def run_distributed_training(args: argparse.Namespace):
    """
    Launch distributed training processes

    This function spawns multiple processes (one per GPU) and coordinates
    the distributed training.
    """
    # Determine world size
    if args.world_size is not None:
        world_size = args.world_size
    elif torch.cuda.is_available() and not args.cpu:
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    if world_size == 1:
        logger.warning("Only 1 GPU/process detected. Running in single-process mode.")
        logger.warning("For true distributed training, use multiple GPUs or --world-size > 1")

    logger.info(f"Launching distributed training with world_size={world_size}")

    # Create output directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Launch training on each process
    if world_size > 1:
        # Use torch multiprocessing to spawn processes
        mp.spawn(
            lambda rank: asyncio.run(train_distributed(rank, world_size, args)),
            args=(world_size, args),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single process mode
        asyncio.run(train_distributed(0, 1, args))


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Distributed Multi-GPU Training for StateSet Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name or path (default: gpt2)"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training"
    )

    # Task arguments
    parser.add_argument(
        "--task",
        type=str,
        default="customer_service",
        choices=["customer_service", "technical_support", "sales"],
        help="Training task (default: customer_service)"
    )

    # Training arguments
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Total number of training episodes (default: 100)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per GPU (default: 8)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--grad-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)"
    )

    # Distributed arguments
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Number of processes (default: auto-detect GPU count)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (useful for testing)"
    )

    # Optimization arguments
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training"
    )
    parser.add_argument(
        "--checkpoint-activations",
        action="store_true",
        help="Enable gradient checkpointing for memory efficiency"
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Offload optimizer states to CPU (saves GPU memory)"
    )

    # Logging and checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/distributed",
        help="Directory for saving checkpoints (default: ./checkpoints/distributed)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1)"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="stateset-agents-distributed",
        help="W&B project name (default: stateset-agents-distributed)"
    )
    parser.add_argument(
        "--no-monitoring",
        action="store_true",
        help="Disable monitoring service"
    )

    args = parser.parse_args()

    # Run distributed training
    run_distributed_training(args)


if __name__ == "__main__":
    main()
