"""
Base environment abstractions and state containers.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .reward import RewardFunction
from .reward_base import RewardType
from .reward_context import (
    REWARD_FALLBACK_KEY,
    build_reward_context,
    normalize_turns,
    note_fallback,
    reward_score,
    trajectory_reward_context,
)
from .trajectory import ConversationTurn, MultiTurnTrajectory

logger = logging.getLogger(__name__)

#: ``EnvironmentState.metadata`` key a step uses to report the reward
#: function's (or default heuristic's) share of its step reward.
STEP_REWARD_FN_SCORE = "_step_reward_fn_score"
#: ``EnvironmentState.metadata`` key a step may use to report the
#: environment's own share of its step reward exactly.
STEP_ENVIRONMENT_REWARD = "_step_environment_reward"

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

    #: Whether step() applies untyped reward functions (no ``reward_type``)
    #: per step. Environments that score steps always apply IMMEDIATE/DENSE
    #: rewards per step; an untyped one is applied per step only when a
    #: subclass sets this, and otherwise once at episode end (the default for
    #: every built-in environment, matching how a trainer applies it).
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
            return self.scores_steps
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
        """Score a finished trajectory with this environment's ``reward_fn``.

        There is no stand-in score: an environment without a ``reward_fn``
        raises ``ValueError`` (its own episode reward is
        ``trajectory.total_reward``), and a reward that fails, returns a
        non-finite score or an unknown result type raises too.
        """
        reward_fn = self.reward_fn
        if reward_fn is None:
            raise ValueError(
                f"{type(self).__name__} has no reward_fn to score the trajectory "
                "with; use trajectory.total_reward for the environment's own reward"
            )
        turns = getattr(trajectory, "turns", [])
        if not turns and hasattr(trajectory, "__iter__"):
            turns = list(trajectory)
        # Dict turns (Trajectory.add_turn) would make rewards that read
        # turn.role score 0; normalize them.
        turns = normalize_turns(turns)

        if getattr(reward_fn, "requires_environment_context", False):
            # Evidence rewards judge the recorded final state as is.
            context = getattr(trajectory, "final_state", None)
        else:
            context = trajectory_reward_context(trajectory, reward_fn=reward_fn)
        return reward_score(await reward_fn.compute_reward(turns, context))

    async def _apply_step_reward(
        self, state: EnvironmentState, turns: list[ConversationTurn]
    ) -> float:
        """Score the newest assistant turn of ``turns`` with ``reward_fn``.

        The whole conversation so far is passed for context, marked so rewards
        that aggregate over assistant turns score only the newest one. The
        call is counted in ``state.metadata["reward_fn_calls"]``; a failure is
        logged, recorded in ``state.metadata["reward_error"]`` and scores 0.0
        unless the reward sets ``fail_on_error``. The score is also recorded
        under :data:`STEP_REWARD_FN_SCORE`, so the episode can average the
        reward function's step scores separately from the environment's own
        step rewards.
        """
        reward_fn = self.reward_fn
        if reward_fn is None:
            return 0.0
        context = build_reward_context(
            state.context, reward_fn=reward_fn, latest_turn_only=True
        )
        state.metadata["reward_fn_calls"] = (
            int(state.metadata.get("reward_fn_calls", 0) or 0) + 1
        )
        # An unknown result type (None, a string) raises TypeError from
        # reward_score and propagates: it is a bug, not an outage.
        score = 0.0
        try:
            result = await reward_fn.compute_reward(turns, context)
            score = reward_score(result)
        except ENVIRONMENT_EXCEPTIONS as error:
            if getattr(reward_fn, "fail_on_error", False):
                raise
            _record_reward_failure(state.metadata, reward_fn, error, "step")
        else:
            breakdown = getattr(result, "breakdown", None)
            if isinstance(breakdown, dict):
                state.metadata["reward_breakdown"] = dict(breakdown)
            note_fallback(state.metadata, result)
        state.metadata[STEP_REWARD_FN_SCORE] = score
        return score

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
        # Split the step reward into the reward function's score (averaged
        # over the episode at the end) and the environment's own reward.
        reward_fn_part = float(new_state.metadata.pop(STEP_REWARD_FN_SCORE, 0.0))
        intrinsic = new_state.metadata.pop(STEP_ENVIRONMENT_REWARD, None)
        episode.reward_fn_scores.append(reward_fn_part)
        episode.environment_rewards.append(
            float(step_reward) - reward_fn_part if intrinsic is None else intrinsic
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
        """Total the episode: environment rewards plus one reward-function score.

        Each step keeps the environment's own reward; a per-step reward
        function (or the built-in default heuristic) contributes the mean of
        its step scores, spread as ``score / steps`` over the steps, so it
        never grows with episode length; an episode-end reward is added to
        the last step. ``total_reward == fsum(turn_rewards)``.
        """
        turns, state, scenario = episode.turns, episode.state, episode.scenario
        steps = len(episode.environment_rewards)
        turn_rewards = [
            intrinsic + score / steps
            for intrinsic, score in zip(
                episode.environment_rewards, episode.reward_fn_scores, strict=True
            )
        ]
        reward_metadata = {
            key: state.metadata[key]
            for key in (
                "reward_fn_calls",
                "reward_error",
                "reward_breakdown",
                REWARD_FALLBACK_KEY,
            )
            if state.metadata.get(key)
        }
        reward_fn = self.reward_fn
        if reward_fn and self._should_compute_final_reward():
            context = build_reward_context(
                state.context,
                scenario if isinstance(scenario, dict) else None,
                reward_fn=reward_fn,
            )
            reward_metadata["reward_fn_calls"] = (
                int(reward_metadata.get("reward_fn_calls", 0)) + 1
            )
            try:
                final_reward = await reward_fn.compute_reward(turns, context)
                final_value = reward_score(final_reward)
            except ENVIRONMENT_EXCEPTIONS as error:
                if getattr(reward_fn, "fail_on_error", False):
                    raise
                _record_reward_failure(reward_metadata, reward_fn, error, "episode")
            else:
                if turn_rewards:
                    turn_rewards[-1] += final_value
                else:
                    turn_rewards.append(final_value)
                breakdown = getattr(final_reward, "breakdown", None)
                if isinstance(breakdown, dict):
                    reward_metadata["reward_breakdown"] = dict(breakdown)
                note_fallback(reward_metadata, final_reward)

        metadata: dict[str, Any] = {
            "episode_id": state.episode_id,
            "status": state.status.value,
            "scenario": scenario,
            # The environment's own step rewards, without any reward function:
            # a trainer scoring with another reward adds its score to these.
            "environment_rewards": list(episode.environment_rewards),
        }
        if reward_metadata.get("reward_fn_calls"):
            # Lets a trainer holding the same reward_fn reuse this reward
            # instead of scoring the episode a second time.
            metadata["reward_fn_applied"] = True
        metadata.update(reward_metadata)
        return MultiTurnTrajectory(
            turns=turns,
            total_reward=math.fsum(turn_rewards),
            turn_rewards=turn_rewards,
            initial_state={"scenario": scenario} if scenario else None,
            final_state=state.context,
            metadata=metadata,
        )


@dataclass
class _Episode:
    """Mutable bookkeeping for one episode while it runs."""

    scenario: dict[str, Any] | None
    state: EnvironmentState
    turns: list[ConversationTurn] = field(default_factory=list)
    turn_rewards: list[float] = field(default_factory=list)
    total_reward: float = 0.0
    #: Per step: the reward function's (or default heuristic's) raw score.
    reward_fn_scores: list[float] = field(default_factory=list)
    #: Per step: the environment's own reward, without the reward function.
    environment_rewards: list[float] = field(default_factory=list)


def _record_reward_failure(
    metadata: dict[str, Any], reward_fn: Any, error: BaseException, where: str
) -> None:
    """Log a swallowed reward failure and mark it on the episode."""
    name = getattr(reward_fn, "name", None) or type(reward_fn).__name__
    logger.warning(
        "Reward function %s failed scoring the %s; it contributes 0.0: %s",
        name,
        where,
        error,
    )
    metadata["reward_error"] = f"{type(error).__name__}: {error}"


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
