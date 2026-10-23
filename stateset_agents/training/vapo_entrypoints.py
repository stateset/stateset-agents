"""
Training entrypoints for VAPO workflows.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import numpy as np

from .spine.loop import run_dedicated_loop
from .vapo_config import VAPOConfig

logger = logging.getLogger(__name__)


async def train_with_vapo(
    model_name: str,
    reward_fn: Callable[[str, str], float | Awaitable[float]],
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
    from .vapo_trainer import VAPOModelManager, VAPOTrainer, _require_wandb, wandb

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
    _is_stub = model_name.startswith("stub://") or getattr(
        config, "use_stub_model", False
    )
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

    def batch(iteration: int) -> list[str] | None:
        batch_size = min(config.per_device_train_batch_size, len(train_prompts))
        if batch_size <= 0:
            return None
        batch_indices = np.random.choice(len(train_prompts), batch_size, replace=False)
        return [train_prompts[i] for i in batch_indices]

    def report(iteration: int, metrics: dict[str, Any]) -> None:
        if iteration % config.logging_steps == 0:
            if "warmup_value_loss" in metrics:
                logger.info("Value warmup | Loss: %.4f", metrics["warmup_value_loss"])
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

    # The spine's dedicated outer loop (periodic and final saves included).
    await run_dedicated_loop(
        iterations=config.num_episodes,
        batch=batch,
        step=trainer.train_step,
        save=trainer.save_checkpoint,
        output_dir=output_dir,
        save_steps=config.save_steps,
        final_name="final",
        report=report,
    )
    trainer.metrics_history["rollout_samples_total"] = [
        float(trainer.rollout_samples_total)
    ]

    if use_wandb:
        wandb.finish()

    logger.info("=" * 60)
    logger.info("VAPO Training Complete!")
    logger.info("=" * 60)

    return model, tokenizer, trainer.metrics_history


__all__ = ["train_with_vapo"]
