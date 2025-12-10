"""
Advanced Optimization Techniques for StateSet Agents

This example demonstrates advanced optimization techniques for training
conversational agents efficiently. It covers:

1. Mixed Precision Training (FP16/BF16)
2. Gradient Accumulation
3. Gradient Clipping and Normalization
4. Advanced Optimizers (AdamW, Adafactor, Lion)
5. Learning Rate Scheduling (Cosine, Warmup, ReduceLROnPlateau)
6. Model Compilation with torch.compile (PyTorch 2.0+)
7. Memory Optimization (Gradient Checkpointing, CPU Offloading)
8. Efficient Batch Processing

Requirements:
    - PyTorch 2.0+ (for torch.compile)
    - pip install stateset-agents[dev]
    - Optional: pip install lion-pytorch  # For Lion optimizer

Usage:
    # Basic training with all optimizations
    python examples/advanced_optimization_techniques.py \
        --model gpt2 \
        --mixed-precision bf16

    # Memory-efficient training for large models
    python examples/advanced_optimization_techniques.py \
        --model gpt2-large \
        --gradient-checkpointing \
        --cpu-offload \
        --grad-accumulation 4

    # Maximum performance with compilation
    python examples/advanced_optimization_techniques.py \
        --model gpt2 \
        --compile \
        --mixed-precision bf16 \
        --optimizer adamw
"""

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)

# Framework imports
from stateset_agents import MultiTurnAgent
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import create_customer_service_reward

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. Optimizer Configurations
# ============================================================================

@dataclass
class OptimizerConfig:
    """Configuration for optimizer"""
    name: str
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # AdamW specific
    amsgrad: bool = False

    # Adafactor specific
    scale_parameter: bool = True
    relative_step: bool = False
    warmup_init: bool = False


def create_optimizer(
    model: nn.Module,
    config: OptimizerConfig
) -> torch.optim.Optimizer:
    """
    Create optimizer with advanced configurations

    Supports:
    - AdamW (with weight decay fix)
    - Adafactor (memory efficient)
    - Lion (recent optimizer from Google)
    - SGD (with momentum)
    """

    # Separate parameters with and without weight decay
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    if config.name.lower() == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            amsgrad=config.amsgrad,
        )
        logger.info(f"Created AdamW optimizer with lr={config.learning_rate}")

    elif config.name.lower() == "adafactor":
        try:
            from transformers.optimization import Adafactor
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=config.learning_rate,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=config.weight_decay,
                scale_parameter=config.scale_parameter,
                relative_step=config.relative_step,
                warmup_init=config.warmup_init,
            )
            logger.info("Created Adafactor optimizer (memory efficient)")
        except ImportError:
            logger.warning("Adafactor not available, falling back to AdamW")
            optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    elif config.name.lower() == "lion":
        try:
            from lion_pytorch import Lion
            optimizer = Lion(
                optimizer_grouped_parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay,
            )
            logger.info("Created Lion optimizer")
        except ImportError:
            logger.warning("Lion optimizer not available, falling back to AdamW")
            optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    elif config.name.lower() == "sgd":
        optimizer = SGD(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            momentum=config.beta1,  # Use beta1 as momentum
        )
        logger.info(f"Created SGD optimizer with momentum={config.beta1}")

    else:
        raise ValueError(f"Unknown optimizer: {config.name}")

    return optimizer


# ============================================================================
# 2. Learning Rate Schedulers
# ============================================================================

def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    **kwargs
):
    """
    Create learning rate scheduler

    Supports:
    - Linear warmup + cosine annealing
    - ReduceLROnPlateau (adaptive based on metrics)
    - Constant with warmup
    - Polynomial decay
    """

    if scheduler_type == "cosine":
        # Linear warmup followed by cosine annealing
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=kwargs.get("min_lr", 1e-7),
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps],
        )
        logger.info(f"Created cosine LR scheduler with {num_warmup_steps} warmup steps")

    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "max"),
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 5),
            min_lr=kwargs.get("min_lr", 1e-7),
        )
        logger.info("Created ReduceLROnPlateau scheduler")

    elif scheduler_type == "constant":
        # Constant LR after warmup
        if num_warmup_steps > 0:
            scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=num_warmup_steps,
            )
            logger.info(f"Created constant LR with {num_warmup_steps} warmup steps")
        else:
            scheduler = None
            logger.info("No LR scheduler (constant learning rate)")

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


# ============================================================================
# 3. Mixed Precision Training
# ============================================================================

class MixedPrecisionTrainer:
    """
    Handles mixed precision training with automatic scaling

    Supports FP16 and BF16 (bfloat16)
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: str = "fp16",
        init_scale: float = 2.0**16,
        growth_interval: int = 2000,
    ):
        self.enabled = enabled
        self.dtype = dtype

        if enabled:
            if dtype == "fp16":
                self.autocast_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler(
                    init_scale=init_scale,
                    growth_interval=growth_interval,
                )
                logger.info("Mixed precision enabled: FP16 with gradient scaling")
            elif dtype == "bf16":
                self.autocast_dtype = torch.bfloat16
                self.scaler = None  # BF16 doesn't need gradient scaling
                logger.info("Mixed precision enabled: BF16")
            else:
                raise ValueError(f"Unknown dtype: {dtype}")
        else:
            self.autocast_dtype = torch.float32
            self.scaler = None
            logger.info("Mixed precision disabled (FP32)")

    def autocast(self):
        """Context manager for automatic mixed precision"""
        return torch.cuda.amp.autocast(
            enabled=self.enabled,
            dtype=self.autocast_dtype,
        )

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass"""
        if self.scaler:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer):
        """Optimizer step with unscaling"""
        if self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients before gradient clipping"""
        if self.scaler:
            self.scaler.unscale_(optimizer)


# ============================================================================
# 4. Gradient Management
# ============================================================================

class GradientManager:
    """
    Manages gradient accumulation, clipping, and normalization
    """

    def __init__(
        self,
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        clip_method: str = "norm",  # "norm" or "value"
    ):
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.clip_method = clip_method
        self.current_step = 0

        logger.info(f"Gradient accumulation: {accumulation_steps} steps")
        logger.info(f"Gradient clipping: {clip_method}, max_norm={max_grad_norm}")

    def should_update(self) -> bool:
        """Check if we should update parameters"""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0

    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients to prevent exploding gradients

        Returns:
            Total gradient norm before clipping
        """
        parameters = [p for p in model.parameters() if p.grad is not None]

        if self.clip_method == "norm":
            total_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                self.max_grad_norm,
            )
        elif self.clip_method == "value":
            total_norm = 0.0
            for p in parameters:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            torch.nn.utils.clip_grad_value_(parameters, self.max_grad_norm)
        else:
            raise ValueError(f"Unknown clip method: {self.clip_method}")

        return total_norm

    def reset(self):
        """Reset accumulation counter"""
        self.current_step = 0


# ============================================================================
# 5. Memory Optimization
# ============================================================================

def enable_gradient_checkpointing(model: nn.Module):
    """
    Enable gradient checkpointing for memory efficiency

    Trades compute for memory by recomputing activations during backward pass
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    else:
        logger.warning("Model does not support gradient checkpointing")


def print_memory_stats():
    """Print GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        logger.info(f"GPU Memory: Allocated={allocated:.2f}GB, "
                   f"Reserved={reserved:.2f}GB, "
                   f"Max={max_allocated:.2f}GB")


# ============================================================================
# 6. Training Loop with All Optimizations
# ============================================================================

async def train_with_optimizations(
    agent: MultiTurnAgent,
    environment: ConversationEnvironment,
    args: argparse.Namespace,
):
    """
    Complete training loop with all optimization techniques
    """

    logger.info("=" * 80)
    logger.info("Advanced Optimization Training")
    logger.info("=" * 80)

    # Get model from agent
    model = agent.get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1. Enable memory optimizations
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    # 2. Model compilation (PyTorch 2.0+)
    if args.compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode=args.compile_mode)

    # 3. Create optimizer
    optimizer_config = OptimizerConfig(
        name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
    )
    optimizer = create_optimizer(model, optimizer_config)

    # 4. Create LR scheduler
    num_training_steps = args.num_episodes * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    scheduler = create_lr_scheduler(
        optimizer,
        args.scheduler,
        num_training_steps,
        num_warmup_steps,
        min_lr=args.min_lr,
    )

    # 5. Setup mixed precision
    mp_trainer = MixedPrecisionTrainer(
        enabled=args.mixed_precision != "none",
        dtype=args.mixed_precision,
    )

    # 6. Setup gradient management
    grad_manager = GradientManager(
        accumulation_steps=args.grad_accumulation,
        max_grad_norm=args.max_grad_norm,
        clip_method=args.grad_clip_method,
    )

    # Training metrics
    total_steps = 0
    best_reward = float("-inf")

    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    # Print initial memory stats
    print_memory_stats()

    # Training loop
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_grad_norm = 0.0

        model.train()
        optimizer.zero_grad()
        grad_manager.reset()

        for episode in range(args.num_episodes):
            episode_start = time.time()

            # Generate trajectory (simplified for example)
            # In practice, this would use the GRPO engine
            with mp_trainer.autocast():
                # Dummy loss for demonstration
                loss = torch.tensor(0.5, requires_grad=True, device=device)

                # Scale loss for gradient accumulation
                loss = loss / args.grad_accumulation

            # Backward pass
            scaled_loss = mp_trainer.scale_loss(loss)
            scaled_loss.backward()

            # Update parameters if accumulation complete
            if grad_manager.should_update():
                # Unscale gradients before clipping
                mp_trainer.unscale_gradients(optimizer)

                # Clip gradients
                grad_norm = grad_manager.clip_gradients(model)
                epoch_grad_norm += grad_norm

                # Optimizer step
                mp_trainer.step(optimizer)
                optimizer.zero_grad()

                # Update learning rate
                if scheduler and args.scheduler != "plateau":
                    scheduler.step()

                total_steps += 1

            epoch_loss += loss.item() * args.grad_accumulation
            episode_time = time.time() - episode_start

            if episode % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch + 1}/{args.num_epochs}, "
                    f"Episode {episode + 1}/{args.num_episodes}, "
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {current_lr:.2e}, "
                    f"Time: {episode_time:.2f}s"
                )

        # Epoch summary
        avg_loss = epoch_loss / args.num_episodes
        avg_grad_norm = epoch_grad_norm / (args.num_episodes // args.grad_accumulation)
        epoch_time = time.time() - epoch_start

        logger.info("=" * 80)
        logger.info(f"Epoch {epoch + 1} Summary")
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  Average Grad Norm: {avg_grad_norm:.4f}")
        logger.info(f"  Epoch Time: {epoch_time:.2f}s")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        logger.info("=" * 80)

        # Update LR scheduler (for plateau scheduler)
        if scheduler and args.scheduler == "plateau":
            scheduler.step(avg_loss)

        # Print memory stats
        print_memory_stats()

        # Save checkpoint if improved
        if epoch_reward > best_reward:
            best_reward = epoch_reward
            if args.save_checkpoints:
                checkpoint_path = Path(args.checkpoint_dir) / f"best_model_epoch_{epoch + 1}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "best_reward": best_reward,
                }, checkpoint_path)
                logger.info(f"Saved best checkpoint to {checkpoint_path}")

    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Advanced Optimization Techniques Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model arguments
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")

    # Optimizer arguments
    parser.add_argument(
        "--optimizer", type=str, default="adamw",
        choices=["adamw", "adafactor", "lion", "sgd"],
        help="Optimizer type"
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument(
        "--grad-clip-method", type=str, default="norm",
        choices=["norm", "value"],
        help="Gradient clipping method"
    )

    # Learning rate scheduler
    parser.add_argument(
        "--scheduler", type=str, default="cosine",
        choices=["cosine", "plateau", "constant"],
        help="LR scheduler type"
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Minimum learning rate")

    # Mixed precision
    parser.add_argument(
        "--mixed-precision", type=str, default="none",
        choices=["none", "fp16", "bf16"],
        help="Mixed precision training"
    )

    # Gradient accumulation
    parser.add_argument("--grad-accumulation", type=int, default=1, help="Gradient accumulation steps")

    # Memory optimization
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--cpu-offload", action="store_true", help="Offload to CPU")

    # Model compilation
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument(
        "--compile-mode", type=str, default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Compilation mode"
    )

    # Training arguments
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")

    # Checkpointing
    parser.add_argument("--save-checkpoints", action="store_true", help="Save checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/optimized", help="Checkpoint directory")

    args = parser.parse_args()

    # Create agent
    agent_config = AgentConfig(model_name=args.model)
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    # Create environment
    environment = ConversationEnvironment(
        scenarios=[{"topic": "demo", "context": "Demo conversation"}],
        max_turns=4,
        reward_fn=create_customer_service_reward(),
    )

    # Run training
    await train_with_optimizations(agent, environment, args)


if __name__ == "__main__":
    asyncio.run(main())
