"""
Reusable evaluation utilities for RL agents.

This module intentionally avoids depending on any specific algorithm trainer.
It evaluates an agent against an Environment by running episodes via the
environment's ``run_episode`` helper and aggregating rewards/length metrics.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from stateset_agents.core.environment import Environment
from stateset_agents.core.trajectory import ConversationTurn

logger = logging.getLogger(__name__)

EVAL_ENV_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
EVAL_REWARD_EXCEPTIONS = (
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluating an agent."""

    num_episodes: int = 10
    num_generations: int = 1
    max_turns: int | None = None
    seed: int | None = 42
    concurrency: int = 4  # Default to parallel evaluation for better performance
    fail_on_error: bool = True


async def _compute_reward(
    reward_fn: Any,
    turns: list[ConversationTurn],
    context: dict[str, Any] | None,
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
        value = float(result.score)
    elif isinstance(result, dict) and "score" in result:
        value = float(result["score"])
    elif isinstance(result, (int, float)):
        value = float(result)
    else:
        raise TypeError(f"Unexpected reward result type: {type(result)!r}")
    if not math.isfinite(value):
        raise ValueError("Evaluation reward must be finite")
    return value


class _EvaluationMetrics(dict[str, float]):
    """JSON-compatible metrics retaining the exact integer success count."""

    def __init__(self, values: dict[str, float], *, successes: int) -> None:
        super().__init__(values)
        self.successes = successes


def _empty_metrics(*, attempted: int = 0) -> dict[str, float]:
    return _EvaluationMetrics(
        {
            "eval_reward": 0.0,
            "eval_reward_std": 0.0,
            "eval_episode_length": 0.0,
            "eval_success_rate": 0.0,
            "eval_num_episodes": 0.0,
            "eval_attempted_episodes": float(attempted),
            "eval_failed_episodes": float(attempted),
            "eval_error_rate": float(attempted > 0),
            "eval_complete": 0.0,
        },
        successes=0,
    )


async def evaluate_agent(
    *,
    agent: Any,
    environment: Environment,
    scenarios: Sequence[dict[str, Any]] | None = None,
    reward_fn: Any | None = None,
    config: EvaluationConfig | None = None,
) -> dict[str, float]:
    """Evaluate an agent over a set of scenarios.

    Notes:
        - If ``reward_fn`` is provided, it overrides the environment-computed reward.
        - When ``concurrency > 1``, this function will attempt to use
          ``environment.clone()`` to avoid shared mutable state between episodes.
          The agent object is still shared; only enable concurrency if your agent
          backend is safe for concurrent calls.
        - Errors abort evaluation by default. ``fail_on_error=False`` is a
          diagnostic mode: failed attempts remain in coverage and success-rate
          denominators; reward/length means describe valid samples only.
          Required reward failures and OOM always propagate.
        - ``eval_success_rate`` remains the reward > 0.5 heuristic, not verified
          tool success. Use held-out outcome reports for quality claims.
    """

    cfg = config or EvaluationConfig()
    if type(cfg.fail_on_error) is not bool:
        raise ValueError("fail_on_error must be boolean")
    if cfg.num_episodes <= 0:
        return _empty_metrics()

    if cfg.concurrency < 1:
        cfg = replace(cfg, concurrency=1)

    if cfg.concurrency > 1 and type(environment).clone is Environment.clone:
        logger.warning(
            "Environment clone() is not implemented; forcing sequential evaluation."
        )
        cfg = replace(cfg, concurrency=1)

    # Seed for reproducible evaluation, but hand the caller's RNG streams back
    # afterwards: re-seeding mid-training would replay the same environment
    # randomness after every evaluation.
    saved_rng = None
    if cfg.seed is not None:
        saved_rng = (random.getstate(), np.random.get_state())
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
    try:
        return await _evaluate_seeded(agent, environment, scenarios, reward_fn, cfg)
    finally:
        if saved_rng is not None:
            random.setstate(saved_rng[0])
            np.random.set_state(saved_rng[1])


async def _evaluate_seeded(
    agent: Any,
    environment: Environment,
    scenarios: Sequence[dict[str, Any]] | None,
    reward_fn: Any | None,
    cfg: EvaluationConfig,
) -> dict[str, float]:
    shared_env_lock = asyncio.Lock()
    scenario_list: list[dict[str, Any]] = list(scenarios or [])
    required_reward = any(
        getattr(reward, "fail_on_error", False)
        for reward in (reward_fn, getattr(environment, "reward_fn", None))
    )

    def must_raise(exc: BaseException) -> bool:
        # Detect Torch OOM without importing the optional training stack.
        oom = getattr(sys.modules.get("torch"), "OutOfMemoryError", None)
        return (
            cfg.fail_on_error
            or required_reward
            or isinstance(exc, MemoryError)
            or (isinstance(oom, type) and isinstance(exc, oom))
        )

    async def _run_one(episode_idx: int) -> list[dict[str, Any]] | None:
        scenario: dict[str, Any] | None = None
        if scenario_list:
            scenario = scenario_list[episode_idx % len(scenario_list)]

        env = environment
        uses_shared_env = False
        if cfg.concurrency > 1:
            try:
                env = environment.clone()
            except NotImplementedError:
                # Fall back to sequential evaluation when clone is unavailable.
                env = environment
                uses_shared_env = True
            except (
                EVAL_ENV_EXCEPTIONS
            ) as exc:  # pragma: no cover - defensive for custom envs
                logger.warning(
                    "Environment clone() failed; using shared instance: %s", exc
                )
                env = environment
                uses_shared_env = True

        async def agent_fn(history: Any, context: Any) -> Any:
            generate = getattr(agent, "generate_response", None)
            if generate is None or not callable(generate):
                raise TypeError(
                    "agent must implement generate_response(history, context)"
                )
            return await generate(history, context)

        samples: list[dict[str, Any]] = []
        for _ in range(max(1, cfg.num_generations)):
            try:
                if uses_shared_env:
                    async with shared_env_lock:
                        trajectory = await env.run_episode(
                            agent_fn,
                            scenario=scenario,
                            max_turns=cfg.max_turns,
                        )
                else:
                    trajectory = await env.run_episode(
                        agent_fn,
                        scenario=scenario,
                        max_turns=cfg.max_turns,
                    )
            except EVAL_ENV_EXCEPTIONS as exc:
                if must_raise(exc):
                    raise
                logger.warning("Evaluation episode %s failed: %s", episode_idx, exc)
                continue

            try:
                reward = float(getattr(trajectory, "total_reward", 0.0))
                if reward_fn is not None:
                    context = (
                        getattr(trajectory, "final_state", None)
                        if getattr(reward_fn, "requires_environment_context", False)
                        else scenario
                    )
                    reward = await _compute_reward(
                        reward_fn, list(getattr(trajectory, "turns", [])), context
                    )
                episode_length = float(getattr(trajectory, "episode_length", 0))
                if (
                    not math.isfinite(reward)
                    or not math.isfinite(episode_length)
                    or episode_length < 0
                ):
                    raise ValueError(
                        "Evaluation reward and episode length must be finite"
                    )
            except EVAL_REWARD_EXCEPTIONS + (OverflowError,) as exc:
                if must_raise(exc):
                    raise
                logger.warning(
                    "Evaluation scoring failed for episode %s: %s", episode_idx, exc
                )
                continue
            samples.append({"reward": reward, "episode_length": episode_length})

        return samples or None

    if cfg.concurrency <= 1:
        results = [await _run_one(i) for i in range(cfg.num_episodes)]
    else:
        sem = asyncio.Semaphore(cfg.concurrency)

        async def _run_guarded(i: int) -> list[dict[str, Any]] | None:
            async with sem:
                return await _run_one(i)

        tasks = [asyncio.create_task(_run_guarded(i)) for i in range(cfg.num_episodes)]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    rewards: list[float] = []
    lengths: list[float] = []
    for per_episode in results:
        if not per_episode:
            continue
        for sample in per_episode:
            rewards.append(sample["reward"])
            lengths.append(sample["episode_length"])

    attempted = cfg.num_episodes * max(1, cfg.num_generations)
    if not rewards:
        return _empty_metrics(attempted=attempted)

    scale = max(abs(reward) for reward in rewards)
    reward_std = (
        float(np.std(np.array(rewards, dtype=float) / scale)) * scale if scale else 0.0
    )

    successes = sum(reward > 0.5 for reward in rewards)
    return _EvaluationMetrics(
        {
            "eval_reward": math.fsum(reward / len(rewards) for reward in rewards),
            "eval_reward_std": reward_std,
            "eval_episode_length": math.fsum(
                length / len(lengths) for length in lengths
            ),
            "eval_success_rate": successes / attempted,
            "eval_num_episodes": float(len(rewards)),
            "eval_attempted_episodes": float(attempted),
            "eval_failed_episodes": float(attempted - len(rewards)),
            "eval_error_rate": float((attempted - len(rewards)) / attempted),
            "eval_complete": float(len(rewards) == attempted),
        },
        successes=successes,
    )
