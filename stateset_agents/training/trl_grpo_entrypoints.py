"""
Training entrypoints for TRL-based GRPO workflows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import nullcontext
from datetime import datetime
from typing import Any

from stateset_agents.core.agent import Agent, AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.rewards.multi_objective_reward import MultiObjectiveRewardFunction

from .config import get_config_for_task
from .trl_grpo_config import TRLGRPOConfig

logger = logging.getLogger(__name__)


def _should_report_to_wandb(report_to: str | None) -> bool:
    """Return whether wandb logging is enabled in a report_to string."""
    if not report_to or report_to == "none":
        return False
    return "wandb" in {target.strip() for target in report_to.split(",")}


def _build_sync_reward_function(reward_wrapper: Any):
    """Wrap the async reward interface for TRL's synchronous callback."""

    def sync_reward_function(completions, prompts, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                reward_wrapper.compute_rewards(completions, prompts, **kwargs)
            )
        finally:
            loop.close()

    return sync_reward_function


async def train_with_trl_grpo(
    config: TRLGRPOConfig,
    agent: Agent,
    environment: ConversationEnvironment,
    reward_model: MultiObjectiveRewardFunction,
    train_data: list[dict[str, Any]] | None = None,
    eval_data: list[dict[str, Any]] | None = None,
) -> Agent:
    """Main training function using TRL GRPO."""
    from .trl_grpo_trainer import (
        ModelManager,
        TRLGRPODatasetBuilder,
        TRLGRPORewardFunction,
        TRLGRPOTrainerWrapper,
        _require_wandb,
        wandb,
    )

    logger.info("Initializing TRL GRPO training")
    logger.info("Configuration: %s", json.dumps(config.to_dict(), indent=2))

    os.makedirs(config.output_dir, exist_ok=True)

    wants_wandb = _should_report_to_wandb(config.report_to)
    wandb_enabled = wants_wandb and config.wandb_project is not None
    if wants_wandb and not wandb_enabled:
        logger.warning(
            "report_to includes wandb but no wandb_project was provided; disabling wandb logging."
        )

    if wandb_enabled:
        _require_wandb()
        wandb.init(
            project=config.wandb_project,
            name=config.run_name
            or f"trl-grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.to_dict(),
            tags=config.wandb_tags,
        )

    model_manager = ModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer()

    dataset_builder = TRLGRPODatasetBuilder(tokenizer, config)

    if train_data:
        train_dataset = dataset_builder.build_from_conversations(train_data)
    else:
        logger.info("Generating training trajectories...")
        trajectories = []

        for _episode in range(min(100, config.num_episodes)):
            trajectory = await environment.run_episode(agent)
            trajectories.append(trajectory)

        train_dataset = dataset_builder.build_from_trajectories(trajectories)

    if eval_data:
        logger.info(
            "eval_data was provided but is not currently consumed by the TRL GRPO entrypoint."
        )

    logger.info("Training dataset size: %s", len(train_dataset))

    reward_wrapper = TRLGRPORewardFunction(reward_model, agent, environment)
    sync_reward_function = _build_sync_reward_function(reward_wrapper)

    trainer = TRLGRPOTrainerWrapper(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        reward_function=sync_reward_function,
        ref_model=model_manager.ref_model,
    )

    trainer.train()

    final_model_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_model_path)
    logger.info("Model saved to %s", final_model_path)

    agent.model = trainer.model
    agent.tokenizer = tokenizer

    if wandb_enabled:
        wandb.finish()

    logger.info("TRL GRPO training completed successfully.")
    return agent


async def train_customer_service_with_trl(
    model_name: str = "openai/gpt-oss-120b",
    train_data: list[dict[str, Any]] | None = None,
    num_episodes: int = 1000,
    output_dir: str = "./outputs/trl_grpo",
    **kwargs,
) -> Agent:
    """Train a customer service agent using TRL GRPO."""
    base_config = get_config_for_task("customer_service", model_name=model_name)

    config = TRLGRPOConfig.from_training_config(
        base_config, num_episodes=num_episodes, output_dir=output_dir, **kwargs
    )

    agent_config = AgentConfig(
        model_name=model_name,
        system_prompt="You are a helpful and empathetic customer service representative.",
        **kwargs,
    )
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    from stateset_agents.core.environment import CONVERSATION_CONFIGS

    env_config = CONVERSATION_CONFIGS["customer_service"].copy()
    environment = ConversationEnvironment(**env_config)

    from stateset_agents.rewards.multi_objective_reward import (
        create_customer_service_reward,
    )

    reward_model = create_customer_service_reward()

    return await train_with_trl_grpo(
        config=config,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
        train_data=train_data,
    )


async def train_iterative_grpo(
    config: TRLGRPOConfig,
    agent: Agent,
    environment: ConversationEnvironment,
    reward_model: MultiObjectiveRewardFunction,
) -> Agent:
    """
    Run iterative (online) GRPO training.

    Loop:
    1. Generate trajectories using current policy.
    2. Train for N epochs.
    3. Repeat.
    """
    from .trl_grpo_trainer import (
        ModelManager,
        TRLGRPODatasetBuilder,
        TRLGRPORewardFunction,
        TRLGRPOTrainerWrapper,
        TrajectoryGenerator,
        _require_wandb,
        wandb,
    )

    logger.info(
        "Starting Iterative GRPO Training (%s iterations)",
        config.num_outer_iterations,
    )

    wants_wandb = _should_report_to_wandb(config.report_to)
    wandb_enabled = wants_wandb and config.wandb_project is not None
    if wants_wandb and not wandb_enabled:
        logger.warning(
            "report_to includes wandb but no wandb_project was provided; disabling wandb logging."
        )

    if wandb_enabled:
        _require_wandb()
        wandb.init(
            project=config.wandb_project,
            name=config.run_name
            or f"iterative-grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.to_dict(),
            tags=["iterative", "grpo"] + (config.wandb_tags or []),
        )

    model_manager = ModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer()

    agent.model = model
    agent.tokenizer = tokenizer

    generator = TrajectoryGenerator(config, agent, environment)
    dataset_builder = TRLGRPODatasetBuilder(tokenizer, config)

    reward_wrapper = TRLGRPORewardFunction(reward_model, agent, environment)
    sync_reward_function = _build_sync_reward_function(reward_wrapper)

    try:
        from rich import box
        from rich.layout import Layout
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
        from rich.table import Table

        rich_available = True
    except ImportError:
        rich_available = False

    if rich_available:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        status_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        task_id = status_progress.add_task(
            "[green]Iterative Training", total=config.num_outer_iterations
        )

        metrics_table = Table(title="Training Metrics", box=box.SIMPLE)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="magenta")

    live_ctx = Live(layout, refresh_per_second=4) if rich_available else nullcontext()

    with live_ctx:
        for iteration in range(config.num_outer_iterations):
            current_iter_label = (
                f"Iteration {iteration + 1}/{config.num_outer_iterations}"
            )
            logger.info("=== %s ===", current_iter_label)

            if rich_available:
                layout["header"].update(
                    Panel(
                        f"StateSet Agents - GRPO Training\n{current_iter_label}",
                        style="bold white on blue",
                    )
                )
                layout["left"].update(Panel(status_progress, title="Progress"))
                layout["right"].update(Panel(metrics_table, title="Metrics"))

            logger.info("Phase 1: Generating Trajectories...")
            if rich_available:
                status_progress.update(
                    task_id, description=f"{current_iter_label}: Generating Data"
                )

            trajectories = await generator.generate_batch(
                config.generations_per_iteration
            )

            train_dataset = dataset_builder.build_from_trajectories(trajectories)

            logger.info("Phase 2: Training...")
            if rich_available:
                status_progress.update(
                    task_id, description=f"{current_iter_label}: Training"
                )

            trainer_wrapper = TRLGRPOTrainerWrapper(
                config=config,
                model=agent.model,
                tokenizer=agent.tokenizer,
                train_dataset=train_dataset,
                reward_function=sync_reward_function,
                ref_model=model_manager.ref_model,
            )

            trainer_wrapper.train()

            agent.model = trainer_wrapper.trainer.model

            if rich_available:
                metrics_table = Table(title="Training Metrics", box=box.SIMPLE)
                metrics_table.add_column("Metric", style="cyan")
                metrics_table.add_column("Value", style="magenta")
                metrics_table.add_row("Trajectories", str(len(trajectories)))
                metrics_table.add_row("Dataset Size", str(len(train_dataset)))
                metrics_table.add_row("Last Iteration", str(iteration + 1))
                layout["right"].update(Panel(metrics_table, title="Metrics"))
                status_progress.advance(task_id)

            if generator.vllm_engine:
                logger.warning(
                    "vLLM engine weights are not automatically updated in this loop implementation yet."
                )

            checkpoint_dir = os.path.join(config.output_dir, f"iter_{iteration}")
            trainer_wrapper.save_model(checkpoint_dir)

    logger.info("Iterative training completed.")
    if wandb_enabled:
        wandb.finish()

    return agent


__all__ = [
    "train_with_trl_grpo",
    "train_customer_service_with_trl",
    "train_iterative_grpo",
]
