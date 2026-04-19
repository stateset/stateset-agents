"""
Training entrypoints for DAPO workflows.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any
from collections.abc import Awaitable, Callable

import numpy as np

from .dapo_config import DAPOConfig

logger = logging.getLogger(__name__)


async def train_with_dapo(
    model_name: str,
    reward_fn: Callable[[str, str], float | Awaitable[float]],
    train_prompts: list[str],
    config: DAPOConfig | None = None,
    verifier_fn: Callable[[str, str], bool] | None = None,
    output_dir: str = "./outputs/dapo",
    use_wandb: bool = False,
    wandb_project: str | None = None,
) -> tuple[Any, Any, dict[str, list[float]]]:
    """
    Train a model using the DAPO algorithm.

    DAPO is optimized for long chain-of-thought reasoning tasks like math.
    """
    from .dapo_trainer import DAPOModelManager, DAPOTrainer, _require_wandb, wandb

    logger.info("=" * 60)
    logger.info("DAPO Training - Decoupled Clip and Dynamic Sampling")
    logger.info("=" * 60)
    logger.info(
        "Key techniques: Clip-Higher, Dynamic Sampling, Token-Level Loss, Overlong Shaping"
    )

    if config is None:
        config = DAPOConfig(
            model_name=model_name,
            output_dir=output_dir,
        )

    wandb_enabled = use_wandb and wandb_project is not None
    if use_wandb and not wandb_enabled:
        logger.warning(
            "use_wandb=True but no wandb_project was provided; disabling wandb logging."
        )

    if wandb_enabled:
        _require_wandb()
        wandb.init(
            project=wandb_project,
            name=f"dapo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.to_dict(),
            tags=["dapo", "rl-training", "reasoning"],
        )

    logger.info("Loading model: %s", model_name)
    _is_stub = model_name.startswith("stub://") or getattr(config, "use_stub_model", False)
    if _is_stub:
        from stateset_agents.core.agent_backends import StubModel, StubTokenizer
        model = StubModel()
        tokenizer = StubTokenizer()
    else:
        model_manager = DAPOModelManager(config)
        model, tokenizer = model_manager.load_model_and_tokenizer()

    trainer = DAPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        verifier_fn=verifier_fn,
    )

    if config.use_vllm:
        logger.info("Initializing vLLM for fast DAPO generation...")
        vllm_success = await trainer.initialize_vllm()
        if vllm_success:
            logger.info("vLLM initialized - generation will be 5-20x faster!")
        else:
            logger.warning("vLLM initialization failed - using HuggingFace generation")

    logger.info("Starting training with %s prompts", len(train_prompts))
    logger.info("Group size: %s", config.group_size)
    logger.info(
        "Clip range: [%.2f, %.2f]",
        1 - config.clip_eps_low,
        1 + config.clip_eps_high,
    )
    logger.info("Dynamic sampling: %s", config.use_dynamic_sampling)

    os.makedirs(output_dir, exist_ok=True)

    for iteration in range(config.num_episodes):
        batch_size = min(config.per_device_train_batch_size, len(train_prompts))
        if batch_size <= 0:
            break
        batch_indices = np.random.choice(len(train_prompts), batch_size, replace=False)
        batch_prompts = [train_prompts[i] for i in batch_indices]

        metrics = await trainer.train_step(batch_prompts)

        if iteration % config.logging_steps == 0:
            logger.info(
                "Iter %s/%s | Loss: %.4f | Reward: %.4f | Acc: %.2f%% | Filtered: %.1f%%",
                iteration,
                config.num_episodes,
                metrics.get("policy_loss", 0.0),
                metrics.get("average_reward", 0.0),
                metrics.get("accuracy", 0.0) * 100,
                metrics.get("filtered_ratio", 0.0) * 100,
            )

            if wandb_enabled:
                wandb.log(metrics, step=iteration)

        if (iteration + 1) % config.save_steps == 0:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{iteration + 1}")
            trainer.save_checkpoint(checkpoint_dir)

    final_dir = os.path.join(output_dir, "final")
    trainer.save_checkpoint(final_dir)

    if wandb_enabled:
        wandb.finish()

    logger.info("=" * 60)
    logger.info("DAPO Training Complete!")
    logger.info("=" * 60)

    return model, tokenizer, trainer.metrics_history


async def train_reasoning_with_dapo(
    model_name: str,
    math_problems: list[dict[str, str]],
    output_dir: str = "./outputs/dapo-math",
    **kwargs,
) -> tuple[Any, Any, dict]:
    """Train a reasoning model using DAPO."""
    prompts = [problem["problem"] for problem in math_problems]
    answers = {problem["problem"]: problem["answer"] for problem in math_problems}

    def verifier(prompt: str, response: str) -> bool:
        expected = answers.get(prompt, "")
        return expected.lower() in response.lower()

    def reward_fn(prompt: str, response: str) -> float:
        if verifier(prompt, response):
            return 1.0
        return 0.0

    config = DAPOConfig(
        model_name=model_name,
        output_dir=output_dir,
        use_dynamic_sampling=True,
        use_overlong_shaping=True,
        use_token_level_loss=True,
        **kwargs,
    )

    return await train_with_dapo(
        model_name=model_name,
        reward_fn=reward_fn,
        train_prompts=prompts,
        config=config,
        verifier_fn=verifier,
        output_dir=output_dir,
    )


__all__ = ["train_with_dapo", "train_reasoning_with_dapo"]
