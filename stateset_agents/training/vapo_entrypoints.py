"""
Training entrypoints for VAPO workflows.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any
from collections.abc import Callable

import numpy as np

from .vapo_config import VAPOConfig

logger = logging.getLogger(__name__)


async def train_with_vapo(
    model_name: str,
    reward_fn: Callable[[str, str], float],
    train_prompts: list[str],
    config: VAPOConfig | None = None,
    verifier_fn: Callable[[str, str], bool] | None = None,
    output_dir: str = "./outputs/vapo",
    use_wandb: bool = False,
    wandb_project: str | None = None,
) -> tuple[Any, Any, dict[str, list[float]]]:
    """
    Train a model using VAPO algorithm.

    VAPO is the current SOTA for long-CoT reasoning (60.4 on AIME 2024).
    """
    from .vapo_trainer import (
        VAPOModelManager,
        VAPOTrainer,
        _require_wandb,
        wandb,
    )

    logger.info("=" * 60)
    logger.info("VAPO Training - Value-Augmented Policy Optimization")
    logger.info("=" * 60)
    logger.info("SOTA: 60.4 on AIME 2024")
    logger.info("Key: Value warmup + Decoupled GAE + Length-adaptive lambda")

    if config is None:
        config = VAPOConfig(
            model_name=model_name,
            output_dir=output_dir,
        )

    if use_wandb and wandb_project:
        _require_wandb()
        wandb.init(
            project=wandb_project,
            name=f"vapo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.to_dict(),
            tags=["vapo", "rl-training", "value-based", "reasoning"],
        )

    logger.info("Loading model: %s", model_name)
    _is_stub = model_name.startswith("stub://") or getattr(config, "use_stub_model", False)
    if _is_stub:
        from stateset_agents.core.agent_backends import StubModel, StubTokenizer
        model = StubModel()
        tokenizer = StubTokenizer()
    else:
        model_manager = VAPOModelManager(config)
        model, tokenizer = model_manager.load_model_and_tokenizer()

    trainer = VAPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        verifier_fn=verifier_fn,
    )

    logger.info("Starting training with %s prompts", len(train_prompts))
    logger.info("Value warmup steps: %s", config.value_warmup_steps)
    logger.info("Group size: %s", config.group_size)

    os.makedirs(output_dir, exist_ok=True)

    for iteration in range(config.num_episodes):
        batch_size = min(config.per_device_train_batch_size, len(train_prompts))
        if batch_size <= 0:
            break
        batch_indices = np.random.choice(len(train_prompts), batch_size, replace=False)
        batch_prompts = [train_prompts[i] for i in batch_indices]

        metrics = await trainer.train_step(batch_prompts)

        if iteration % config.logging_steps == 0:
            if "warmup_value_loss" in metrics:
                logger.info(
                    "Value warmup | Loss: %.4f", metrics["warmup_value_loss"]
                )
            else:
                logger.info(
                    "Iter %s/%s | Policy: %.4f | Value: %.4f | Reward: %.4f | Acc: %.2f%% | EV: %.4f",
                    iteration,
                    config.num_episodes,
                    metrics.get("policy_loss", 0.0),
                    metrics.get("value_loss", 0.0),
                    metrics.get("average_reward", 0.0),
                    metrics.get("accuracy", 0.0) * 100,
                    metrics.get("explained_variance", 0.0),
                )

            if use_wandb:
                wandb.log(metrics, step=iteration)

        if (iteration + 1) % config.save_steps == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{iteration + 1}")
            trainer.save_checkpoint(checkpoint_dir)

    final_dir = os.path.join(output_dir, "final")
    trainer.save_checkpoint(final_dir)

    if use_wandb:
        wandb.finish()

    logger.info("=" * 60)
    logger.info("VAPO Training Complete!")
    logger.info("=" * 60)

    return model, tokenizer, trainer.metrics_history


__all__ = ["train_with_vapo"]
