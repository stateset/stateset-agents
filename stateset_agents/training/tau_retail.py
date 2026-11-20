"""τ-bench retail as a training and evaluation environment.

Wraps the official ``tau2`` gym environment so every rollout is driven by the
official customer simulator and scored by the official checker. The checker's
reference actions never reach the policy. A rollout whose official score is
missing raises :class:`EvaluationUnavailable` instead of counting as a failed
task: upstream reports reward 0.0 with no detail when its simulator or judge
call fails, which would otherwise be indistinguishable from a wrong answer.

``tau2`` is an optional dependency (Python 3.12+). Import errors surface when
an environment is constructed, not when this module is imported.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from stateset_agents.core.environment_base import (
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from stateset_agents.core.trajectory import ConversationTurn

logger = logging.getLogger(__name__)

DOMAIN = "retail"
STOP_ACTION = "done()"
SUCCESS_THRESHOLD = 1.0 - 1e-6

ACTION_FORMAT = (
    "Reply with exactly one action per turn.\n"
    "To call a tool, reply with only a JSON object: "
    '{"name": "<tool name>", "arguments": {<arguments>}}\n'
    "To speak to the customer, reply with plain text.\n"
    "Tool results and customer messages arrive prefixed with `tool:` and `user:`."
)


class EvaluationUnavailable(RuntimeError):
    """The official checker produced no score for a finished rollout."""


@dataclass(frozen=True)
class TauRetailSettings:
    """Models and limits that define one comparable evaluation setup.

    ``judge_llm=None`` keeps the judge model pinned by ``tau2``. Any other
    value replaces it process-wide and is a deviation from official scoring
    that reports must state.
    """

    user_llm: str
    user_llm_args: Mapping[str, Any] = field(default_factory=dict)
    judge_llm: str | None = None
    judge_llm_args: Mapping[str, Any] | None = None
    max_turns: int = 30
    simulator_max_steps: int = 100

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON form recorded with results."""
        return {
            "domain": DOMAIN,
            "user_llm": self.user_llm,
            "user_llm_args": dict(self.user_llm_args),
            "judge_llm": self.judge_llm,
            "judge_llm_args": (
                None if self.judge_llm_args is None else dict(self.judge_llm_args)
            ),
            "official_judge": self.judge_llm is None,
            "max_turns": self.max_turns,
            "simulator_max_steps": self.simulator_max_steps,
        }


def retail_task_ids(split: str) -> list[str]:
    """Return the official retail task IDs for ``train`` or ``test``."""
    from tau2.domains.retail.environment import get_tasks_split

    splits = get_tasks_split()
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    train, test = list(splits["train"]), list(splits["test"])
    if set(train) & set(test):
        raise ValueError("Official train and test splits overlap")
    return train if split == "train" else test


def configure_judge(settings: TauRetailSettings) -> None:
    """Point the natural-language-assertion judge at ``settings.judge_llm``."""
    if settings.judge_llm is None:
        return
    from tau2.evaluator import evaluator_nl_assertions as module

    module.DEFAULT_LLM_NL_ASSERTIONS = settings.judge_llm
    module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(settings.judge_llm_args or {})


def _official_env(settings: TauRetailSettings, task_id: str) -> Any:
    from tau2.gym.gym_agent import AgentGymEnv

    return AgentGymEnv(
        DOMAIN,
        task_id,
        max_steps=settings.simulator_max_steps,
        user_llm=settings.user_llm,
        user_llm_args=dict(settings.user_llm_args),
    )


def build_system_prompt(policy: str, tools: Sequence[Any]) -> str:
    """Render the domain policy, tool schemas and the text action format."""
    schemas = [getattr(tool, "openai_schema", tool) for tool in tools]
    return (
        "You are a customer service agent. Follow the policy below.\n\n"
        f"<policy>\n{policy}\n</policy>\n\n"
        f"<tools>\n{json.dumps(schemas, sort_keys=True)}\n</tools>\n\n"
        f"{ACTION_FORMAT}"
    )


def _score(info: Mapping[str, Any], reward: float) -> dict[str, Any]:
    raw = info.get("reward_info")
    detail = json.loads(raw) if isinstance(raw, str) and raw.strip() else raw
    if not isinstance(detail, dict) or "reward" not in detail:
        raise EvaluationUnavailable(
            "The official checker returned no reward detail; the customer "
            "simulator or judge call probably failed"
        )
    if float(detail["reward"]) != float(reward):
        raise EvaluationUnavailable("Reward and reward detail disagree")
    return detail


class TauRetailEnvironment(Environment):
    """One official τ-bench retail conversation per episode."""

    def __init__(
        self,
        settings: TauRetailSettings,
        *,
        env_factory: Callable[[str], Any] | None = None,
    ) -> None:
        super().__init__(max_turns=settings.max_turns)
        self.settings = settings
        self._env_factory = env_factory or (lambda t: _official_env(settings, t))
        self._sessions: dict[str, Any] = {}
        self._system_prompt: str | None = None
        if env_factory is None:
            configure_judge(settings)

    def clone(self) -> TauRetailEnvironment:
        """Return an independent environment with the same configuration."""
        clone = TauRetailEnvironment(self.settings, env_factory=self._env_factory)
        clone._system_prompt = self._system_prompt
        return clone

    def observation_context(self, state: EnvironmentState) -> dict[str, Any]:
        """Expose only the task ID and the latest observation to the policy."""
        return {
            "task_id": state.context["task_id"],
            "observation": state.context["observation"],
        }

    async def get_initial_prompt(self, scenario: dict[str, Any] | None = None) -> str:
        """Return the shared retail policy and tool prompt."""
        if self._system_prompt is None:
            raise RuntimeError("reset() must run before the first prompt is built")
        return self._system_prompt

    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
        """Start the official simulation for ``scenario['id']``."""
        if not scenario or not isinstance(scenario.get("id"), str):
            raise ValueError("scenario needs a string task 'id'")
        task_id = scenario["id"]
        session = self._env_factory(task_id)
        observation, info = await asyncio.to_thread(
            session.reset, seed=scenario.get("seed")
        )
        if not observation:
            raise EvaluationUnavailable(
                f"Task {task_id} ended before the agent's first turn"
            )
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt(info["policy"], info["tools"])
        episode_id = f"{task_id}-{uuid.uuid4().hex}"
        self._sessions[episode_id] = session
        return EnvironmentState(
            episode_id,
            0,
            EpisodeStatus.ONGOING,
            {
                "task_id": task_id,
                "observation": observation,
                "messages": [{"role": "user", "content": observation}],
                "task_success": False,
                "forced_stop": False,
                "reward_detail": None,
            },
        )

    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
        """Send one agent action; score with the official checker at the end."""
        if state.is_done:
            raise ValueError("Episode already ended")
        session = self._sessions[state.episode_id]
        state.turn_count += 1
        observation, reward, done, _, info = await asyncio.to_thread(
            session.step, (action.content or "").strip()
        )
        if not done and state.turn_count >= self.max_turns:
            # Turn budget spent: end the conversation so the official checker
            # scores the database as the agent left it.
            state.context["forced_stop"] = True
            observation, reward, done, _, info = await asyncio.to_thread(
                session.step, STOP_ACTION
            )
            if not done:
                raise EvaluationUnavailable("Simulation ignored the stop action")
        state.context["observation"] = observation
        if not done:
            return (
                state,
                0.0,
                False,
                {"env_response": ConversationTurn(role="user", content=observation)},
            )
        del self._sessions[state.episode_id]
        detail = _score(info, reward)
        success = float(reward) >= SUCCESS_THRESHOLD
        state.status = EpisodeStatus.COMPLETED if success else EpisodeStatus.FAILED
        state.context["task_success"] = success
        state.context["reward_detail"] = detail
        return state, float(reward), True, {}


AgentFn = Callable[[Any, dict[str, Any]], Awaitable[Any]]


async def evaluate_policy(
    environment: TauRetailEnvironment,
    agent_fn: AgentFn,
    task_ids: Sequence[str],
    *,
    trials: int,
    seed: int,
    concurrency: int = 1,
) -> list[dict[str, Any]]:
    """Run ``trials`` scored conversations per task and return one row each.

    Trial ``k`` of every task uses simulator seed ``seed + k`` so two policies
    evaluated with the same arguments face the same customer seeds.
    """
    if trials < 1 or concurrency < 1:
        raise ValueError("trials and concurrency must be positive")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be unique")
    limit = asyncio.Semaphore(concurrency)

    async def one(task_id: str, trial: int) -> dict[str, Any]:
        async with limit:
            trajectory = await environment.run_episode(
                agent_fn, {"id": task_id, "seed": seed + trial}
            )
        final = trajectory.final_state or {}
        if final.get("reward_detail") is None:
            raise EvaluationUnavailable(f"Task {task_id} trial {trial} was not scored")
        return {
            "task_id": task_id,
            "trial": trial,
            "simulator_seed": seed + trial,
            "success": bool(final["task_success"]),
            "reward": float(trajectory.total_reward),
            "forced_stop": bool(final["forced_stop"]),
            "agent_turns": sum(t.role == "assistant" for t in trajectory.turns),
        }

    return list(
        await asyncio.gather(
            *(one(task_id, trial) for task_id in task_ids for trial in range(trials))
        )
    )


def success_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count successes per task from :func:`evaluate_policy` rows."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["task_id"]] = counts.get(row["task_id"], 0) + int(row["success"])
    return counts
