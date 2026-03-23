"""
Training entrypoints for GSPO workflows.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from stateset_agents.core.agent import Agent, AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.rewards.multi_objective_reward import (
    MultiObjectiveRewardFunction as MultiObjectiveReward,
)

from .config import get_config_for_task
from .gspo_config import GSPOConfig

logger = logging.getLogger(__name__)


async def train_with_gspo(
    config: GSPOConfig,
    agent: Agent,
    environment: ConversationEnvironment,
    reward_model: MultiObjectiveReward,
    train_queries: list[str | dict[str, Any]] | None = None,
    callbacks: list[Any] | None = None,
) -> Agent:
    """Main training function using GSPO."""
    from .gspo_trainer import GSPOModelManager, GSPOTrainer, _require_wandb, wandb

    logger.info("Initializing GSPO training")
    logger.info("Configuration: %s", json.dumps(config.to_dict(), indent=2))

    os.makedirs(config.output_dir, exist_ok=True)

    use_wandb = config.report_to == "wandb"
    if use_wandb:
        _require_wandb()
        if config.wandb_project:
            wandb.init(
                project=config.wandb_project,
                name=config.run_name
                or f"gspo-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=config.to_dict(),
                tags=["gspo"] + (config.wandb_tags or []),
            )
        else:
            logger.warning(
                "report_to='wandb' but no wandb_project set; disabling wandb logging."
            )
            use_wandb = False

    model_manager = GSPOModelManager(config)
    model, tokenizer = model_manager.load_model_and_tokenizer()

    agent.model = model
    agent.tokenizer = tokenizer
    agent.generation_config = agent._build_generation_config()

    if not train_queries:
        logger.info("Generating training queries from environment scenarios...")
        train_queries = []
        for scenario in environment.scenarios[: config.generations_per_iteration]:
            if isinstance(scenario, dict):
                query = scenario.get("context", "Hello")
                query_context = None
                if "task" in scenario:
                    query_context = scenario.get("task")
                elif "metadata" in scenario:
                    query_context = scenario.get("metadata")
                if query_context is not None:
                    train_queries.append({"prompt": query, "context": query_context})
                else:
                    train_queries.append(query)
            else:
                train_queries.append(str(scenario))

    logger.info("Training with %s queries", len(train_queries))

    trainer = GSPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
        ref_model=model_manager.ref_model,
    )

    if config.use_vllm:
        logger.info("Initializing vLLM for fast generation...")
        vllm_success = await trainer.generator.initialize_vllm()
        if vllm_success:
            logger.info("vLLM initialized - generation will be 5-20x faster!")
        else:
            logger.warning("vLLM initialization failed - using HuggingFace generation")

    _callbacks = callbacks or []
    aborted = False

    for iteration in range(config.num_outer_iterations):
        logger.info("=== Iteration %s/%s ===", iteration + 1, config.num_outer_iterations)

        metrics = await trainer.train_step(
            queries=train_queries, num_groups=min(len(train_queries), 10)
        )

        logger.info("Metrics: %s", json.dumps(metrics, indent=2))

        if use_wandb:
            wandb.log(metrics, step=iteration)

        # Periodic checkpoint
        if (iteration + 1) % config.save_steps == 0:
            checkpoint_dir = os.path.join(
                config.output_dir, f"checkpoint-{iteration + 1}"
            )
            trainer.save_model(checkpoint_dir)

        # Notify callbacks via the dispatch layer
        if _callbacks:
            from .callbacks import notify_step_end

            await notify_step_end(_callbacks, step=iteration, metrics=metrics)

            # Check for early abort signals from any callback
            for cb in _callbacks:
                if getattr(cb, "should_abort", False):
                    reason = getattr(cb, "abort_reason", "callback requested abort")
                    logger.warning("Training aborted by callback: %s", reason)
                    aborted = True
                    break

        if aborted:
            break

    final_model_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_model_path)

    if use_wandb:
        wandb.finish()

    if aborted:
        logger.info("GSPO training aborted early.")
    else:
        logger.info("GSPO training completed successfully.")
    return agent


async def train_customer_service_with_gspo(
    model_name: str = "gpt2",
    num_episodes: int = 100,
    output_dir: str = "./outputs/gspo",
    **kwargs,
) -> Agent:
    """Train a customer service agent using GSPO."""
    base_config = get_config_for_task("customer_service", model_name=model_name)

    config = GSPOConfig.from_training_config(
        base_config, num_episodes=num_episodes, output_dir=output_dir, **kwargs
    )

    agent_config = AgentConfig(
        model_name=model_name,
        system_prompt="You are a helpful and empathetic customer service representative.",
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

    return await train_with_gspo(
        config=config,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
    )


__all__ = ["train_with_gspo", "train_customer_service_with_gspo"]
