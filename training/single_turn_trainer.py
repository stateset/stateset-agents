"""
Single-turn GRPO trainer implementation.

This module provides the SingleTurnGRPOTrainer class for training
agents on single-turn interactions (one prompt, one response).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .trainer_utils import get_amp, get_torch, require_torch, require_transformers

logger = logging.getLogger(__name__)


class SingleTurnGRPOTrainer:
    """
    GRPO trainer for single-turn agents with HuggingFace and W&B integration

    This trainer implements Group Relative Policy Optimization for simple
    single-turn interactions (one prompt, one response), making it ideal
    for supervised fine-tuning and basic RL scenarios.
    """

    def __init__(
        self,
        agent: Any,  # Base Agent, not MultiTurnAgent
        environment: Any,
        reward_fn: Optional[Any] = None,
        config: Optional[Any] = None,
        wandb_logger: Optional[Any] = None,
        callbacks: Optional[List] = None,
    ):
        self.agent = agent
        self.environment = environment
        self.reward_fn = reward_fn
        self.config = config
        self.wandb_logger = wandb_logger
        self.callbacks = callbacks or []

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_metric = float("-inf")
        self.steps_without_improvement = 0

        # HuggingFace components
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None

        # GRPO enhancements
        self._global_reward_mean: float = 0.0
        self._global_reward_count: int = 0

        logger.info(f"Single-Turn GRPO Trainer initialized with {type(agent).__name__}")

    async def initialize(self):
        """Initialize trainer components"""
        logger.info("Initializing Single-Turn GRPO trainer...")

        torch = require_torch()
        require_transformers()
        amp = get_amp()

        # Initialize agent if not already done
        if not hasattr(self.agent, "model") or self.agent.model is None:
            await self.agent.initialize()

        # Set seeds for reproducibility
        try:
            import random

            seed_value = getattr(self.config, "seed", None)
            if seed_value is not None and torch is not None:
                random.seed(seed_value)
                np.random.seed(seed_value)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_value)
                torch.manual_seed(seed_value)
        except Exception as seed_err:
            logger.warning(f"Failed to set seeds: {seed_err}")

        # Initialize optimizer
        self._setup_optimizer()

        # Initialize mixed precision scaler (CUDA only)
        if torch is not None and amp is not None and torch.cuda.is_available():
            bf16_enabled = getattr(self.config, "bf16", False)
            fp16_enabled = getattr(self.config, "fp16", False)
            if bf16_enabled or fp16_enabled:
                self.scaler = amp.GradScaler("cuda")

        logger.info("Single-Turn GRPO trainer initialized successfully")

    def _setup_optimizer(self):
        """Set up optimizer with HuggingFace best practices"""
        torch = get_torch()
        if self.agent.model is None or torch is None:
            return

        learning_rate = getattr(self.config, "learning_rate", 5e-5)
        weight_decay = getattr(self.config, "weight_decay", 0.01)

        # Separate parameters that should not have weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.agent.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.agent.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        logger.info(f"Optimizer initialized with lr={learning_rate}")

    async def train(self) -> Any:
        """Run single-turn training loop"""
        logger.info("Starting single-turn GRPO training...")

        num_episodes = getattr(self.config, "num_episodes", 10)
        max_steps = getattr(self.config, "max_steps_per_episode", 100)

        for episode in range(num_episodes):
            self.current_epoch = episode

            # Reset environment
            state = await self.environment.reset()
            episode_rewards = []

            for step in range(max_steps):
                # Generate single-turn interaction
                prompt = state.get("prompt", "Hello")
                response = await self.agent.generate_response(prompt)

                # Compute reward
                turns = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]

                if self.reward_fn is not None:
                    reward_result = await self.reward_fn.compute_reward(turns)
                    reward = (
                        reward_result.score
                        if hasattr(reward_result, "score")
                        else reward_result
                    )
                else:
                    reward = 0.5  # Default reward

                episode_rewards.append(reward)

                # Update model (simplified GRPO update)
                torch = get_torch()
                if self.optimizer is not None and torch is not None:
                    self.optimizer.zero_grad()
                    # Placeholder for actual policy gradient computation
                    # In a full implementation, this would compute advantages and policy loss
                    self.optimizer.step()

                self.global_step += 1

                # Get next state
                next_state = await self.environment.step(response)
                state = next_state.get("state", state)

                if next_state.get("done", False):
                    break

            # Log episode metrics
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            logger.info(
                f"Episode {episode + 1}/{num_episodes}: "
                f"Steps={len(episode_rewards)}, AvgReward={avg_reward:.3f}"
            )

            if self.wandb_logger is not None:
                self.wandb_logger.log(
                    {
                        "episode": episode,
                        "episode_reward": avg_reward,
                        "episode_steps": len(episode_rewards),
                    }
                )

        logger.info("Single-turn GRPO training completed")
        return self.agent

    async def save_checkpoint(self, path: Union[str, Path]):
        """Save training checkpoint"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save agent model
        if hasattr(self.agent, "model") and self.agent.model is not None:
            model_path = path / "model"
            self.agent.model.save_pretrained(model_path)
            if hasattr(self.agent, "tokenizer") and self.agent.tokenizer is not None:
                self.agent.tokenizer.save_pretrained(model_path)

        logger.info(f"Checkpoint saved to {path}")

    def add_callback(self, callback: Any):
        """Add training callback"""
        self.callbacks.append(callback)
