"""What a run actually does. Loaded by the worker once it can report a failure.

Three kinds, all driven by the same seed and matrix parameters:

``train``
    The framework's own training loop, in-process, followed by an evaluation.
``command``
    Any program. It receives its seed and output directory as placeholders and
    may write ``metrics.json`` there. Every starter and study in this repo is
    reachable this way, without the worker having to know about it.
``workflow_regression``
    The offline sampled tool-workflow task: real policy updates on a small model
    on CPU, no downloads. It validates training mechanics, not domain quality.

Heavy dependencies are imported inside the function that needs them.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stateset_agents.fanout.worker import METRICS_FILE, WorkloadOutcome, emit

__all__ = ["build_reward", "run_workload"]

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

_DEFAULT_REWARD = {"helpfulness": 0.6, "safety": 0.4}
_DEFAULT_SCENARIOS = [
    {
        "id": "demo",
        "topic": "general_help",
        "context": "Demo",
        "user_responses": ["Thanks, tell me more.", "Interesting, go on."],
    }
]


def build_reward(weights: dict[str, float]) -> Any:
    """Compose named reward components, refusing a name it cannot build."""
    from stateset_agents.core.reward import (
        CompositeReward,
        ConcisenessReward,
        EngagementReward,
        HelpfulnessReward,
        SafetyReward,
    )

    constructors: dict[str, Callable[..., Any]] = {
        "helpfulness": HelpfulnessReward,
        "safety": SafetyReward,
        "conciseness": ConcisenessReward,
        "engagement": EngagementReward,
    }
    unknown = sorted(set(weights) - set(constructors))
    if unknown:
        raise ValueError(
            f"unknown reward(s): {', '.join(unknown)}; "
            f"available: {', '.join(sorted(constructors))}"
        )
    return CompositeReward(
        [constructors[name](weight=weight) for name, weight in weights.items()]
    )


class _EpisodeRecorder:
    """Trainer callback that keeps episode rewards and reports progress."""

    def __init__(self, *, min_interval_s: float = 1.0) -> None:
        self.rewards: list[float] = []
        self.optimizer_updates = 0
        self._min_interval_s = min_interval_s
        self._last_emit = 0.0

    def on_episode_end(self, episode: int, metrics: dict[str, Any]) -> None:
        reward = metrics.get("episode_reward")
        if isinstance(reward, (int, float)):
            self.rewards.append(float(reward))
        updates = metrics.get("global_step")
        if isinstance(updates, int):
            self.optimizer_updates = updates
        now = time.monotonic()
        if now - self._last_emit >= self._min_interval_s:
            self._last_emit = now
            emit({"type": "episode", "episode": episode, "episode_reward": reward})


async def _train(
    spec: dict[str, Any], output_dir: Path
) -> dict[str, Any]:  # pragma: no cover - exercised through the worker subprocess
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.training.evaluation import EvaluationConfig, evaluate_agent
    from stateset_agents.training.train import train

    workload = spec["workload"]
    seed = spec["seed"]
    settings = {**workload.get("training", {}), **spec.get("params", {})}
    num_episodes = settings.pop("num_episodes", 100)
    profile = settings.pop("profile", "balanced")
    objective = settings.pop("objective", None)
    settings.setdefault("output_dir", str(output_dir / "trainer"))

    agent = MultiTurnAgent(AgentConfig(**workload["agent"]))
    await agent.initialize()
    environment_config = workload.get("environment", {})
    scenarios = environment_config.get("scenarios") or _DEFAULT_SCENARIOS
    environment = ConversationEnvironment(
        scenarios=scenarios, max_turns=environment_config.get("max_turns", 3)
    )
    reward = build_reward(workload.get("reward") or _DEFAULT_REWARD)
    recorder = _EpisodeRecorder()

    await train(
        agent=agent,
        environment=environment,
        reward_fn=reward,
        num_episodes=num_episodes,
        profile=profile,
        objective=objective,
        callbacks=[recorder],
        save_path=(
            str(output_dir / "checkpoint") if workload.get("save_checkpoint") else None
        ),
        seed=seed,
        **settings,
    )

    metrics: dict[str, Any] = {
        "episodes": len(recorder.rewards),
        "optimizer_updates": recorder.optimizer_updates,
    }
    if recorder.rewards:
        metrics["episode_reward_mean"] = sum(recorder.rewards) / len(recorder.rewards)
        metrics["episode_reward_last"] = recorder.rewards[-1]
    eval_episodes = workload.get("eval_episodes", len(scenarios))
    if eval_episodes:
        metrics.update(
            await evaluate_agent(
                agent=agent,
                environment=environment,
                scenarios=scenarios,
                reward_fn=reward,
                config=EvaluationConfig(
                    num_episodes=min(eval_episodes, len(scenarios)),
                    num_generations=1,
                    max_turns=settings.get("max_steps_per_episode"),
                    seed=seed,
                    concurrency=1,
                ),
            )
        )
    return metrics


def _run_train(spec: dict[str, Any], output_dir: Path) -> WorkloadOutcome:
    return WorkloadOutcome(metrics=asyncio.run(_train(spec, output_dir)))


def _read_metrics_file(output_dir: Path) -> tuple[dict[str, Any], str | None]:
    path = output_dir / METRICS_FILE
    if not path.is_file():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, f"{METRICS_FILE} is unreadable: {exc}"
    if not isinstance(value, dict):
        return {}, f"{METRICS_FILE} must contain a JSON object"
    return value, None


def _run_command(
    spec: dict[str, Any], output_dir: Path, stage_dir: Path
) -> WorkloadOutcome:
    values = {
        **{key: str(value) for key, value in spec.get("params", {}).items()},
        "seed": str(spec["seed"]),
        "run_id": str(spec["run_id"]),
        "output_dir": str(output_dir),
        "stage_dir": str(stage_dir),
        "python": sys.executable,
    }

    def fill(match: re.Match[str]) -> str:
        return values[match.group(1)]

    argv = [_PLACEHOLDER.sub(fill, arg) for arg in spec["workload"]["argv"]]
    sys.stdout.flush()
    try:
        returncode = subprocess.run(argv, check=False, cwd=output_dir).returncode
    except OSError as exc:
        return WorkloadOutcome(exit_code=127, error=f"cannot start {argv[0]}: {exc}")
    # A process killed by a signal reports -N; the shell convention is 128+N.
    exit_code = returncode if returncode >= 0 else 128 - returncode
    metrics, error = _read_metrics_file(output_dir)
    if exit_code != 0 and error is None:
        error = f"command exited with status {exit_code}"
    return WorkloadOutcome(exit_code=exit_code, metrics=metrics, error=error)


def _run_workflow_regression(
    spec: dict[str, Any], output_dir: Path
) -> WorkloadOutcome:  # pragma: no cover - exercised through the worker subprocess
    import torch

    from stateset_agents.training.sampled_workflow import run_seed

    settings = {key: value for key, value in spec["workload"].items() if key != "kind"}
    settings.update(spec.get("params", {}))
    # Matches the study's own entry point, so results stay comparable to it.
    torch.set_num_threads(1)
    report = asyncio.run(
        run_seed(
            spec["seed"],
            output_dir / "workflow",
            updates=settings.get("updates", 80),
            device=settings.get("device", "cpu"),
            gradient_accumulation_steps=settings.get("gradient_accumulation_steps", 1),
        )
    )

    def success_rate(rows: list[dict[str, Any]]) -> float:
        return sum(bool(row["success"]) for row in rows) / len(rows)

    return WorkloadOutcome(
        metrics={
            "improvement": report["improvement"],
            "baseline_success": success_rate(report["baseline"]),
            "candidate_success": success_rate(report["candidate"]),
            "optimizer_updates": report["optimizer_updates"],
            "reload_verified": float(report["reload_verified"]),
            "train_seconds": report["duration_seconds"],
        }
    )


def run_workload(
    spec: dict[str, Any], output_dir: Path, stage_dir: Path
) -> WorkloadOutcome:
    """Dispatch on the workload kind. Unknown kinds fail rather than no-op."""
    workload = spec.get("workload")
    kind = workload.get("kind") if isinstance(workload, dict) else None
    if kind == "train":
        return _run_train(spec, output_dir)
    if kind == "command":
        return _run_command(spec, output_dir, stage_dir)
    if kind == "workflow_regression":
        return _run_workflow_regression(spec, output_dir)
    raise ValueError(f"unknown workload kind: {kind!r}")
