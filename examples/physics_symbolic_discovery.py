"""
Symbolic physics discovery demo using GSPO.

Usage:
    python examples/physics_symbolic_discovery.py --tasks examples/data/symbolic_physics_tasks.jsonl
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import List

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.environments.symbolic_physics import (
    SymbolicPhysicsEnvironment,
    SymbolicPhysicsTask,
    load_symbolic_tasks,
)
from stateset_agents.rewards.symbolic_physics_reward import (
    SymbolicPhysicsRewardFunction,
    SymbolicRewardConfig,
)
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.gspo_trainer import GSPOConfig, train_with_gspo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Symbolic physics discovery demo.")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("examples/data/symbolic_physics_tasks.jsonl"),
        help="Path to JSONL/JSON task file.",
    )
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument("--output-dir", type=str, default="./outputs/symbolic_physics")
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--generations-per-iteration", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Score targets only.")
    return parser


async def score_targets_only(
    tasks: List[SymbolicPhysicsTask], reward_fn: SymbolicPhysicsRewardFunction
) -> None:
    for task in tasks:
        if not task.target_expression:
            logger.info("Task %s has no target expression.", task.task_id)
            continue
        turn = ConversationTurn(role="assistant", content=task.target_expression)
        result = await reward_fn.compute_reward(
            turns=[turn], context=task.to_dict()
        )
        logger.info("Task %s target score: %.3f", task.task_id, result.score)


async def main() -> None:
    args = build_arg_parser().parse_args()
    tasks = load_symbolic_tasks(args.tasks)

    reward_config = SymbolicRewardConfig(num_samples=args.num_samples)
    reward_fn = SymbolicPhysicsRewardFunction(config=reward_config)
    environment = SymbolicPhysicsEnvironment(
        tasks=tasks,
        max_turns=args.max_turns,
        reward_fn=reward_fn,
    )

    if args.dry_run:
        await score_targets_only(tasks, reward_fn)
        return

    agent = MultiTurnAgent(
        AgentConfig(
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            system_prompt="Return a single expression. Do not include prose.",
        )
    )
    await agent.initialize()

    base_config = TrainingConfig(
        run_name="symbolic_physics_demo",
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
    )
    gspo_config = GSPOConfig.from_training_config(
        base_config,
        num_generations=args.num_generations,
        generations_per_iteration=args.generations_per_iteration,
        learning_rate=1e-5,
        report_to="none",
    )

    train_queries = [
        {"prompt": task.prompt, "context": task.to_dict()} for task in tasks
    ]

    await train_with_gspo(
        config=gspo_config,
        agent=agent,
        environment=environment,
        reward_model=reward_fn,
        train_queries=train_queries,
    )


if __name__ == "__main__":
    asyncio.run(main())
