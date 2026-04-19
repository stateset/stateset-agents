"""
Base environment abstractions and state containers.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import Callable

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
            return True
        return reward_type in (RewardType.CUMULATIVE, RewardType.SPARSE)

    @abstractmethod
    async def reset(
        self, scenario: dict[str, Any] | None = None
    ) -> EnvironmentState:
        """Reset environment and return initial state"""

    @abstractmethod
    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
        """Execute one step in the environment."""

    async def get_initial_prompt(
        self, scenario: dict[str, Any] | None = None
    ) -> str:
        """Get initial prompt/context (override in subclasses)"""
        return ""

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
                logger.warning("Failed to compute reward from reward_fn: %s", e)
                return 0.5

        return 0.5

    async def run_episode(
        self,
        agent_fn: Callable,
        scenario: dict[str, Any] | None = None,
        max_turns: int | None = None,
    ) -> MultiTurnTrajectory:
        """Run a complete episode using the standard step signature."""
        max_turns = max_turns or self.max_turns
        state = await self.reset(scenario)
        turns: list[ConversationTurn] = []
        turn_rewards: list[float] = []
        total_reward = 0.0

        def _is_reward(value: Any) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        def _is_done(value: Any) -> bool:
            return isinstance(value, bool)

        initial_prompt = ""
        try:
            initial_prompt = await self.get_initial_prompt(scenario)
        except ENVIRONMENT_EXCEPTIONS:
            pass
        if initial_prompt:
            turns.append(
                ConversationTurn(
                    role="system",
                    content=initial_prompt,
                    metadata={"scenario": scenario},
                )
            )

        for _ in range(max_turns):
            history = [{"role": t.role, "content": t.content} for t in turns]
            agent_response = await agent_fn(history, state.context)
            agent_turn = (
                agent_response
                if isinstance(agent_response, ConversationTurn)
                else ConversationTurn(role="assistant", content=str(agent_response))
            )
            turns.append(agent_turn)

            raw_step_result = await self.step(state, agent_turn)
            if not isinstance(raw_step_result, tuple) or len(raw_step_result) != 4:
                raise TypeError(
                    "Environment.step() must return a 4-tuple "
                    "(state, reward, done, info) or legacy "
                    "(state, env_response, reward, done)"
                )

            step_result: tuple[Any, Any, Any, Any] = raw_step_result
            new_state: EnvironmentState
            step_reward: float
            done: bool
            info: dict[str, Any]
            env_response: ConversationTurn | None = None

            if _is_reward(step_result[1]) and _is_done(step_result[2]):
                new_state = step_result[0]
                step_reward = float(step_result[1])
                done = bool(step_result[2])
                info = (
                    dict(step_result[3]) if isinstance(step_result[3], dict) else {}
                )
                maybe_env_response = info.get("env_response")
                if isinstance(maybe_env_response, ConversationTurn):
                    env_response = maybe_env_response
            elif _is_reward(step_result[2]) and _is_done(step_result[3]):
                new_state = step_result[0]
                env_response = (
                    step_result[1] if isinstance(step_result[1], ConversationTurn) else None
                )
                step_reward = float(step_result[2])
                done = bool(step_result[3])
                info = {}
            elif isinstance(step_result[1], ConversationTurn):
                new_state = step_result[0]
                env_response = step_result[1]
                step_reward = float(step_result[2])
                done = bool(step_result[3])
                info = {}
            else:
                logger.warning(
                    "Ambiguous Environment.step() return signature; "
                    "assuming (state, reward, done, info)."
                )
                new_state = step_result[0]
                step_reward = float(step_result[1])
                done = bool(step_result[2])
                info = (
                    dict(step_result[3]) if isinstance(step_result[3], dict) else {}
                )
                maybe_env_response = info.get("env_response")
                if isinstance(maybe_env_response, ConversationTurn):
                    env_response = maybe_env_response

            total_reward += float(step_reward)
            turn_rewards.append(float(step_reward))
            if isinstance(env_response, ConversationTurn):
                turns.append(env_response)
            state = new_state
            if done or state.status != EpisodeStatus.ONGOING:
                break

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
