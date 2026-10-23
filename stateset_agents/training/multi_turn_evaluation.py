"""
Evaluation and reward formatting helpers for the multi-turn trainer.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from stateset_agents.core.reward_context import (
    RewardUnavailableError,
    reward_score,
    score_episode,
)

logger = logging.getLogger(__name__)


async def coerce_reward_result(
    reward_result: Any,
    trainer_exceptions: tuple[type[BaseException], ...],
) -> tuple[float, dict[str, float]]:
    """Normalize a reward output into a numeric score plus breakdown.

    The score comes from ``reward_context.reward_score``, the one reward
    parser: an unparseable result raises ``TypeError``, a non-finite one
    ``ValueError``, and an awaitable that fails propagates, so no failure is
    read as 0.0. ``trainer_exceptions`` is accepted for compatibility.
    """
    if inspect.isawaitable(reward_result):
        reward_result = await reward_result
    score = reward_score(reward_result)

    breakdown: dict[str, float] = {}
    if isinstance(reward_result, Mapping):
        raw_breakdown = reward_result.get("breakdown")
    else:
        raw_breakdown = getattr(reward_result, "breakdown", None)
    if isinstance(raw_breakdown, Mapping):
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
    """Run comprehensive post-training evaluation.

    Episodes are scored with ``reward_context.score_episode``, the path
    training uses; ``coerce_reward_result_fn`` is accepted for compatibility
    and no longer used.
    """
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

                # The scoring path training uses (reward_context.score_episode).
                await score_episode(
                    trajectory, trainer.reward_fn, trainer.environment, scenario
                )
                metadata = getattr(trajectory, "metadata", None) or {}
                reward_breakdown = dict(metadata.get("reward_breakdown") or {})

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
                                "content": (
                                    content[:200] + "..."
                                    if len(content) > 200
                                    else content
                                ),
                            }
                        )
                    traj_info["conversation"] = conversation

                scenario_results["trajectories"].append(traj_info)

            except RewardUnavailableError as exc:
                # Only a reward outage skips the episode; a bug, bad data or a
                # None/NaN reward propagates instead of shrinking the sample.
                logger.warning("Skipping an evaluation episode: %s", exc)
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
