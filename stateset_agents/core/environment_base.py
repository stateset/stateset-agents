"""
Base environment abstractions and state containers.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .reward import RewardFunction
from .reward_base import RewardType
from .trajectory import ConversationTurn, MultiTurnTrajectory

logger = logging.getLogger(__name__)

# Narrowed to transient/runtime failures only. TypeError, KeyError, and
# AttributeError typically indicate programming bugs and should propagate
# so they surface during testing.
ENVIRONMENT_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    OSError,
    asyncio.TimeoutError,
)


class EpisodeStatus(Enum):
    """Status of an episode"""

    ONGOING = "ongoing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class EnvironmentState:
    """Represents the state of an environment at a given time"""

    episode_id: str
    turn_count: int
    status: EpisodeStatus
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_done(self) -> bool:
        """Compatibility helper used by some tests."""
        return self.status != EpisodeStatus.ONGOING

    def copy(self) -> EnvironmentState:
        return EnvironmentState(
            episode_id=self.episode_id,
            turn_count=self.turn_count,
            status=self.status,
            context=self.context.copy(),
            metadata=self.metadata.copy(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "turn_count": self.turn_count,
            "step": self.turn_count,
            "status": self.status.value,
            "context": self.context,
            "metadata": self.metadata,
            "scenario": self.context.get("scenario"),
        }

    def __contains__(self, key: str) -> bool:
        return key in self.as_dict() or key in self.context

    def __getitem__(self, key: str) -> Any:
        data = self.as_dict()
        if key in data:
            return data[key]
        if key in self.context:
            return self.context[key]
        raise KeyError(key)


class Environment(ABC):
    """Abstract base class for all environments"""

    def __init__(
        self,
        max_turns: int = 10,
        reward_fn: RewardFunction | None = None,
        reward_function: RewardFunction | None = None,
        timeout_seconds: float | None = None,
    ):
        self.max_turns = max_turns
        self.reward_fn = reward_fn if reward_fn is not None else reward_function
        self.timeout_seconds = timeout_seconds
        self.active_episodes: dict[str, EnvironmentState] = {}

    @property
    def reward_function(self) -> RewardFunction | None:
        return self.reward_fn

    @reward_function.setter
    def reward_function(self, value: RewardFunction | None) -> None:
        self.reward_fn = value

    # Whether step() applies IMMEDIATE/DENSE (and untyped) reward functions.
    scores_steps: bool = False

    def _get_reward_type(self) -> RewardType | None:
        """Return configured reward type when available."""
        reward_fn = self.reward_fn
        return getattr(reward_fn, "reward_type", None)

    def _should_compute_step_reward(self) -> bool:
        """Determine if reward should be computed per-step."""
        reward_fn = self.reward_fn
        if reward_fn is None:
            return False
        reward_type = self._get_reward_type()
        if reward_type is None:
            return True
        return reward_type in (RewardType.IMMEDIATE, RewardType.DENSE)

    def _should_compute_final_reward(self) -> bool:
        """Determine if reward should be computed at episode end."""
        reward_fn = self.reward_fn
        if reward_fn is None:
            return False
        reward_type = self._get_reward_type()
        if reward_type is None:
            # An untyped reward function is applied once: per step where the
            # environment scores steps, otherwise at episode end. Doing both
            # double counts it.
            return not self.scores_steps
        return reward_type in (RewardType.CUMULATIVE, RewardType.SPARSE)

    @abstractmethod
    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
        """Reset environment and return initial state"""

    @abstractmethod
    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
        """Execute one step in the environment."""

    async def get_initial_prompt(self, scenario: dict[str, Any] | None = None) -> str:
        """Get initial prompt/context (override in subclasses)"""
        return ""

    async def get_initial_turns(
        self, state: EnvironmentState
    ) -> list[ConversationTurn]:
        """Return non-agent turns that open an episode, after the system prompt.

        Override to start an episode with, for example, the user's first
        message, so the agent replies to it instead of speaking first.
        """
        return [
            ConversationTurn(
                role=message.get("role", "user"), content=message["content"]
            )
            for message in state.context.get("messages", [])
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        ]

    def clone(self) -> Environment:
        """Return a new environment instance with the same configuration."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement clone(); run sequentially or "
            "provide an environment factory that returns fresh instances."
        )

    async def get_reward(self, trajectory: Any) -> float:
        """Compute reward for a trajectory."""
        if self.reward_fn is not None:
            try:
                turns = getattr(trajectory, "turns", [])
                if not turns and hasattr(trajectory, "__iter__"):
                    turns = list(trajectory)

                if getattr(self.reward_fn, "requires_environment_context", False):
                    result = await self.reward_fn.compute_reward(
                        turns, getattr(trajectory, "final_state", None)
                    )
                else:
                    result = await self.reward_fn.compute_reward(turns)
                if hasattr(result, "score"):
                    return float(result.score)
                if isinstance(result, dict) and "score" in result:
                    return float(result["score"])
                if isinstance(result, (int, float)):
                    return float(result)
                logger.warning("Unexpected reward result type: %s", type(result))
                return 0.5
            except ENVIRONMENT_EXCEPTIONS as e:
                if getattr(self.reward_fn, "fail_on_error", False):
                    raise
                logger.warning("Failed to compute reward from reward_fn: %s", e)
                return 0.5

        return 0.5

    def observation_context(self, state: EnvironmentState) -> dict[str, Any]:
        """Expose public context while retaining evaluator references privately."""
        context = dict(state.context)
        private = {"expected_response", "expected_tool_calls"}
        for key in private:
            context.pop(key, None)
        if isinstance(context.get("scenario"), dict):
            context["scenario"] = {
                key: value
                for key, value in context["scenario"].items()
                if key not in private
            }
        return context

    #: Whether ``reset``/``step`` keep all per-episode state in the
    #: ``EnvironmentState`` they return, so several episodes can be advanced
    #: concurrently on one instance. ``run_episodes_batched`` falls back to one
    #: episode at a time when this is ``False``.
    supports_concurrent_episodes: bool = False

    async def run_episode(
        self,
        agent_fn: Callable,
        scenario: dict[str, Any] | None = None,
        max_turns: int | None = None,
    ) -> MultiTurnTrajectory:
        """Run a complete episode using the standard step signature."""
        max_turns = max_turns or self.max_turns
        episode = await self._start_episode(scenario)
        for _ in range(max_turns):
            history = [{"role": t.role, "content": t.content} for t in episode.turns]
            policy_context = copy.deepcopy(self.observation_context(episode.state))
            agent_response = await agent_fn(history, policy_context)
            if await self._advance_episode(episode, agent_response):
                break
        return await self._finish_episode(episode)

    async def run_episodes_batched(
        self,
        batch_agent_fn: Callable,
        scenarios: Sequence[dict[str, Any] | None],
        max_turns: int | None = None,
    ) -> list[MultiTurnTrajectory]:
        """Run one episode per scenario, generating each round in one batch.

        ``batch_agent_fn(histories, contexts)`` receives the visible history and
        context of every still-running episode and returns one reply (a string
        or ``ConversationTurn``) per episode, in order. Each episode follows
        exactly the same sequence of calls as :meth:`run_episode`; finished
        episodes drop out of later rounds. Environments that do not declare
        ``supports_concurrent_episodes`` run their episodes one at a time.
        """
        max_turns = max_turns or self.max_turns
        if not self.supports_concurrent_episodes:
            results = []
            for scenario in scenarios:

                async def single(history: Any, context: Any) -> Any:
                    [reply] = await _batch_replies(batch_agent_fn, [history], [context])
                    return reply

                results.append(await self.run_episode(single, scenario, max_turns))
            return results

        episodes = [await self._start_episode(s) for s in scenarios]
        active = list(episodes)
        for _ in range(max_turns):
            if not active:
                break
            histories = [
                [{"role": t.role, "content": t.content} for t in e.turns]
                for e in active
            ]
            replies = await _batch_replies(
                batch_agent_fn,
                histories,
                [copy.deepcopy(self.observation_context(e.state)) for e in active],
            )
            still_running = []
            for episode, reply in zip(active, replies, strict=True):
                if not await self._advance_episode(episode, reply):
                    still_running.append(episode)
            active = still_running
        return [await self._finish_episode(e) for e in episodes]

    async def _start_episode(self, scenario: dict[str, Any] | None) -> _Episode:
        state = await self.reset(scenario)
        episode = _Episode(scenario=scenario, state=state)
        initial_prompt = ""
        # With no scenario, reset() chose one; use the episode's own choice
        # rather than whatever the instance saw last.
        prompt_scenario = scenario
        if prompt_scenario is None and isinstance(state.context, dict):
            prompt_scenario = state.context.get("scenario")
        try:
            initial_prompt = await self.get_initial_prompt(prompt_scenario)
        except ENVIRONMENT_EXCEPTIONS:
            if getattr(self.reward_fn, "fail_on_error", False):
                raise
            pass
        if initial_prompt:
            episode.turns.append(
                ConversationTurn(
                    role="system",
                    content=initial_prompt,
                    metadata={"scenario": scenario},
                )
            )
        episode.turns.extend(await self.get_initial_turns(state))
        return episode

    async def _advance_episode(self, episode: _Episode, agent_response: Any) -> bool:
        """Apply one agent reply; return True when the episode has ended."""
        agent_turn = (
            agent_response
            if isinstance(agent_response, ConversationTurn)
            else ConversationTurn(role="assistant", content=str(agent_response))
        )
        episode.turns.append(agent_turn)
        raw_step_result = await self.step(episode.state, agent_turn)
        new_state, step_reward, done, env_response, info = _parse_step_result(
            raw_step_result
        )
        episode.total_reward += float(step_reward)
        episode.turn_rewards.append(float(step_reward))
        if isinstance(env_response, ConversationTurn):
            episode.turns.append(env_response)
        for message in info.get("messages", []):
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                episode.turns.append(
                    ConversationTurn(
                        role=message.get("role", "user"), content=message["content"]
                    )
                )
        episode.state = new_state
        return bool(done or new_state.status != EpisodeStatus.ONGOING)

    async def _finish_episode(self, episode: _Episode) -> MultiTurnTrajectory:
        turns, state, scenario = episode.turns, episode.state, episode.scenario
        turn_rewards = episode.turn_rewards
        total_reward = episode.total_reward
        if self.reward_fn and self._should_compute_final_reward():
            try:
                final_reward = await self.reward_fn.compute_reward(turns, state.context)
                if hasattr(final_reward, "score"):
                    final_value = float(final_reward.score)
                elif isinstance(final_reward, dict) and "score" in final_reward:
                    final_value = float(final_reward["score"])
                else:
                    final_value = None
                if final_value is not None:
                    total_reward += final_value
                    if turn_rewards:
                        turn_rewards[-1] += final_value
                    else:
                        turn_rewards.append(final_value)
            except ENVIRONMENT_EXCEPTIONS:
                if getattr(self.reward_fn, "fail_on_error", False):
                    raise
                pass

        return MultiTurnTrajectory(
            turns=turns,
            total_reward=total_reward,
            turn_rewards=turn_rewards,
            initial_state={"scenario": scenario} if scenario else None,
            final_state=state.context,
            metadata={
                "episode_id": state.episode_id,
                "status": state.status.value,
                "scenario": scenario,
            },
        )


@dataclass
class _Episode:
    """Mutable bookkeeping for one episode while it runs."""

    scenario: dict[str, Any] | None
    state: EnvironmentState
    turns: list[ConversationTurn] = field(default_factory=list)
    turn_rewards: list[float] = field(default_factory=list)
    total_reward: float = 0.0


def _is_reward(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_step_result(
    raw_step_result: Any,
) -> tuple[EnvironmentState, float, bool, ConversationTurn | None, dict[str, Any]]:
    """Normalize step returns to ``(state, reward, done, reply, info)``.

    Accepts ``(state, reward, done, info)`` and the legacy
    ``(state, env_response, reward, done)``.
    """
    if not isinstance(raw_step_result, tuple) or len(raw_step_result) != 4:
        raise TypeError(
            "Environment.step() must return a 4-tuple "
            "(state, reward, done, info) or legacy "
            "(state, env_response, reward, done)"
        )
    step_result: tuple[Any, Any, Any, Any] = raw_step_result
    env_response: ConversationTurn | None = None
    if _is_reward(step_result[1]) and isinstance(step_result[2], bool):
        info = dict(step_result[3]) if isinstance(step_result[3], dict) else {}
        maybe = info.get("env_response")
        if isinstance(maybe, ConversationTurn):
            env_response = maybe
        return (
            step_result[0],
            float(step_result[1]),
            bool(step_result[2]),
            env_response,
            info,
        )
    if _is_reward(step_result[2]) and isinstance(step_result[3], bool):
        if isinstance(step_result[1], ConversationTurn):
            env_response = step_result[1]
        return (
            step_result[0],
            float(step_result[2]),
            bool(step_result[3]),
            env_response,
            {},
        )
    if isinstance(step_result[1], ConversationTurn):
        return (
            step_result[0],
            float(step_result[2]),
            bool(step_result[3]),
            step_result[1],
            {},
        )
    logger.warning(
        "Ambiguous Environment.step() return signature; "
        "assuming (state, reward, done, info)."
    )
    info = dict(step_result[3]) if isinstance(step_result[3], dict) else {}
    maybe = info.get("env_response")
    if isinstance(maybe, ConversationTurn):
        env_response = maybe
    return (
        step_result[0],
        float(step_result[1]),
        bool(step_result[2]),
        env_response,
        info,
    )


async def _batch_replies(
    batch_agent_fn: Callable, histories: list[Any], contexts: list[Any]
) -> list[Any]:
    replies = batch_agent_fn(histories, contexts)
    if inspect.isawaitable(replies):
        replies = await replies
    replies = list(replies)
    if len(replies) != len(histories):
        raise ValueError(
            f"batch agent returned {len(replies)} replies for "
            f"{len(histories)} episodes"
        )
    return replies
