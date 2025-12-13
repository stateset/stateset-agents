"""
Reusable evaluation utilities for RL agents.

This module intentionally avoids depending on any specific algorithm trainer.
It evaluates an agent against an Environment by running episodes via the
environment's ``run_episode`` helper and aggregating rewards/length metrics.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from stateset_agents.core.environment import Environment
from stateset_agents.core.trajectory import ConversationTurn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluating an agent."""

    num_episodes: int = 10
    num_generations: int = 1
    max_turns: Optional[int] = None
    seed: Optional[int] = 42
    concurrency: int = 1


async def _compute_reward(
    reward_fn: Any,
    turns: List[ConversationTurn],
    context: Optional[Dict[str, Any]],
) -> float:
    if reward_fn is None:
        return 0.0

    compute = getattr(reward_fn, "compute_reward", None)
    if compute is None:
        raise TypeError("reward_fn must expose compute_reward(...)")

    result = compute(turns, context)
    if asyncio.iscoroutine(result):
        result = await result

    if hasattr(result, "score"):
        return float(result.score)
    if isinstance(result, dict) and "score" in result:
        return float(result["score"])
    if isinstance(result, (int, float)):
        return float(result)
    raise TypeError(f"Unexpected reward result type: {type(result)!r}")


async def evaluate_agent(
    *,
    agent: Any,
    environment: Environment,
    scenarios: Optional[Sequence[Dict[str, Any]]] = None,
    reward_fn: Optional[Any] = None,
    config: Optional[EvaluationConfig] = None,
) -> Dict[str, float]:
    """Evaluate an agent over a set of scenarios.

    Notes:
        - If ``reward_fn`` is provided, it overrides the environment-computed reward.
        - When ``concurrency > 1``, this function will attempt to use
          ``environment.clone()`` to avoid shared mutable state between episodes.
          The agent object is still shared; only enable concurrency if your agent
          backend is safe for concurrent calls.
    """

    cfg = config or EvaluationConfig()
    if cfg.num_episodes <= 0:
        return {
            "eval_reward": 0.0,
            "eval_reward_std": 0.0,
            "eval_episode_length": 0.0,
            "eval_success_rate": 0.0,
            "eval_num_episodes": 0.0,
        }

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    scenario_list: List[Dict[str, Any]] = list(scenarios or [])

    async def _run_one(episode_idx: int) -> Optional[List[Dict[str, Any]]]:
        scenario: Optional[Dict[str, Any]] = None
        if scenario_list:
            scenario = scenario_list[episode_idx % len(scenario_list)]

        env = environment
        if cfg.concurrency > 1:
            try:
                env = environment.clone()
            except NotImplementedError:
                # Fall back to sequential evaluation when clone is unavailable.
                env = environment

        async def agent_fn(history: Any, context: Any) -> Any:
            generate = getattr(agent, "generate_response", None)
            if generate is None or not callable(generate):
                raise TypeError("agent must implement generate_response(history, context)")
            return await generate(history, context)

        samples: List[Dict[str, Any]] = []
        for _ in range(max(1, cfg.num_generations)):
            try:
                trajectory = await env.run_episode(
                    agent_fn,
                    scenario=scenario,
                    max_turns=cfg.max_turns,
                )
            except Exception as exc:
                logger.debug("Evaluation episode %s failed: %s", episode_idx, exc)
                continue

            reward = float(getattr(trajectory, "total_reward", 0.0))
            if reward_fn is not None:
                try:
                    reward = await _compute_reward(
                        reward_fn, list(getattr(trajectory, "turns", [])), scenario
                    )
                except Exception as exc:
                    logger.debug("Reward override failed: %s", exc)

            episode_length = float(getattr(trajectory, "episode_length", 0))
            samples.append({"reward": reward, "episode_length": episode_length})

        return samples or None

    if cfg.concurrency <= 1:
        results = [await _run_one(i) for i in range(cfg.num_episodes)]
    else:
        sem = asyncio.Semaphore(cfg.concurrency)

        async def _run_guarded(i: int) -> Optional[Dict[str, Any]]:
            async with sem:
                return await _run_one(i)

        results = await asyncio.gather(*[_run_guarded(i) for i in range(cfg.num_episodes)])

    rewards: List[float] = []
    lengths: List[float] = []
    for per_episode in results:
        if not per_episode:
            continue
        for sample in per_episode:
            rewards.append(sample["reward"])
            lengths.append(sample["episode_length"])

    if not rewards:
        return {
            "eval_reward": 0.0,
            "eval_reward_std": 0.0,
            "eval_episode_length": 0.0,
            "eval_success_rate": 0.0,
            "eval_num_episodes": 0.0,
        }

    rewards_arr = np.array(rewards, dtype=float)
    lengths_arr = np.array(lengths, dtype=float) if lengths else np.array([0.0], dtype=float)

    return {
        "eval_reward": float(rewards_arr.mean()),
        "eval_reward_std": float(rewards_arr.std()),
        "eval_episode_length": float(lengths_arr.mean()),
        "eval_success_rate": float((rewards_arr > 0.5).mean()),
        "eval_num_episodes": float(len(rewards)),
    }
