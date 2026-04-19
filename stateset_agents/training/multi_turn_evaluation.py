"""
Evaluation and reward formatting helpers for the multi-turn trainer.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


async def coerce_reward_result(
    reward_result: Any,
    trainer_exceptions: tuple[type[BaseException], ...],
) -> tuple[float, dict[str, float]]:
    """Normalize reward outputs into a numeric score plus breakdown."""
    if inspect.isawaitable(reward_result):
        try:
            reward_result = await reward_result
        except trainer_exceptions:
            reward_result = None

    breakdown: dict[str, float] = {}

    if isinstance(reward_result, (int, float)) and not isinstance(reward_result, bool):
        return float(reward_result), breakdown

    if isinstance(reward_result, dict):
        raw_score = reward_result.get("score")
        score = 0.0
        if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
            score = float(raw_score)
        raw_breakdown = reward_result.get("breakdown")
        if isinstance(raw_breakdown, dict):
            for key, value in raw_breakdown.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    breakdown[str(key)] = float(value)
        return score, breakdown

    raw_score = getattr(reward_result, "score", None)
    if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
        score = float(raw_score)
    else:
        score = 0.0

    raw_breakdown = getattr(reward_result, "breakdown", None)
    if isinstance(raw_breakdown, dict):
        for key, value in raw_breakdown.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                breakdown[str(key)] = float(value)

    return score, breakdown


def format_trajectory_for_model(agent: Any, trajectory: Any) -> str:
    """Format a trajectory into a model-consumable transcript."""

    def _as_message(turn: Any) -> dict[str, str]:
        if isinstance(turn, dict):
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", ""))
            return {"role": role, "content": content}
        role = getattr(turn, "role", None) or "user"
        content = getattr(turn, "content", None) or ""
        return {"role": str(role), "content": str(content)}

    if hasattr(agent.tokenizer, "apply_chat_template"):
        messages = [_as_message(turn) for turn in trajectory.turns]
        rendered: object = agent.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return rendered if isinstance(rendered, str) else str(rendered)

    parts = []
    for turn in trajectory.turns:
        msg = _as_message(turn)
        if msg["role"] == "user":
            parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            parts.append(f"Assistant: {msg['content']}")

    return "\n".join(parts)


async def run_post_training_evaluation(
    trainer: Any,
    eval_scenarios: list[dict[str, Any]],
    num_samples: int,
    detailed: bool,
    trainer_exceptions: tuple[type[BaseException], ...],
    coerce_reward_result_fn: Any,
) -> dict[str, Any]:
    """Run comprehensive post-training evaluation."""
    logger.info("Running post-training evaluation on %s samples...", num_samples)

    trainer.agent.model.eval()
    eval_results = []

    sample_scenarios = (
        eval_scenarios[:num_samples]
        if len(eval_scenarios) > num_samples
        else eval_scenarios
    )

    for scenario_idx, scenario in enumerate(sample_scenarios):
        scenario_results = {
            "scenario_id": scenario.get("id", f"eval_{scenario_idx}"),
            "trajectories": [],
        }

        num_eval_generations = min(4, getattr(trainer.config, "num_generations", 4))

        for gen_idx in range(num_eval_generations):
            try:

                async def agent_fn(history: Any, context: Any) -> Any:
                    return await trainer.agent.generate_response(history, context)

                trajectory = await trainer.environment.run_episode(agent_fn, scenario)

                if trainer.reward_fn:
                    reward_result = await trainer.reward_fn.compute_reward(
                        trajectory.turns, scenario
                    )
                    score, reward_breakdown = await coerce_reward_result_fn(
                        reward_result
                    )
                    trajectory.total_reward = score
                else:
                    trajectory.total_reward = 0.0
                    reward_breakdown = {}

                traj_info = {
                    "generation_idx": gen_idx,
                    "reward": trajectory.total_reward,
                    "reward_breakdown": reward_breakdown,
                    "num_turns": len(trajectory.turns),
                    "episode_length": trajectory.episode_length,
                }

                if detailed:
                    conversation = []
                    for turn in trajectory.turns:
                        if isinstance(turn, dict):
                            role = str(turn.get("role", "user"))
                            content = str(turn.get("content", ""))
                        else:
                            role = str(getattr(turn, "role", "user"))
                            content = str(getattr(turn, "content", ""))
                        conversation.append(
                            {
                                "role": role,
                                "content": content[:200] + "..."
                                if len(content) > 200
                                else content,
                            }
                        )
                    traj_info["conversation"] = conversation

                scenario_results["trajectories"].append(traj_info)

            except trainer_exceptions as exc:
                logger.warning("Failed to evaluate trajectory: %s", exc)
                continue

        if scenario_results["trajectories"]:
            rewards = [t["reward"] for t in scenario_results["trajectories"]]
            scenario_results["stats"] = {
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "max_reward": np.max(rewards),
                "min_reward": np.min(rewards),
            }

        eval_results.append(scenario_results)

    all_rewards = []
    all_lengths = []
    for result in eval_results:
        for traj in result.get("trajectories", []):
            all_rewards.append(traj["reward"])
            all_lengths.append(traj["episode_length"])

    overall_stats = {
        "num_scenarios_evaluated": len(eval_results),
        "total_trajectories": len(all_rewards),
        "overall_mean_reward": np.mean(all_rewards) if all_rewards else 0.0,
        "overall_std_reward": np.std(all_rewards) if all_rewards else 0.0,
        "overall_mean_length": np.mean(all_lengths) if all_lengths else 0.0,
        "reward_distribution": {
            "p25": np.percentile(all_rewards, 25) if all_rewards else 0.0,
            "p50": np.percentile(all_rewards, 50) if all_rewards else 0.0,
            "p75": np.percentile(all_rewards, 75) if all_rewards else 0.0,
            "p90": np.percentile(all_rewards, 90) if all_rewards else 0.0,
        },
    }

    logger.info("Post-training evaluation complete:")
    logger.info("  Mean reward: %.4f", overall_stats["overall_mean_reward"])
    logger.info("  Std reward: %.4f", overall_stats["overall_std_reward"])
    logger.info("  Mean episode length: %.2f", overall_stats["overall_mean_length"])

    return {
        "overall_stats": overall_stats,
        "scenario_results": eval_results if detailed else None,
    }


__all__ = [
    "coerce_reward_result",
    "format_trajectory_for_model",
    "run_post_training_evaluation",
]
