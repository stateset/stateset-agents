"""
Conversation environment implementation.
"""

from __future__ import annotations

import logging
import random
import re
import uuid
from typing import Any, cast

from .environment_base import (
    ENVIRONMENT_EXCEPTIONS,
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from .reward import RewardFunction
from .trajectory import ConversationTurn

logger = logging.getLogger(__name__)


def _scripted_user_turns(scenario: Any) -> list[str] | None:
    """Return a scenario's scripted user messages, if it declares them."""
    if not isinstance(scenario, dict):
        return None
    turns = scenario.get("user_turns")
    if not isinstance(turns, (list, tuple)) or not turns:
        return None
    return [str(turn) for turn in turns]


def _opening_user_message(scenario: Any) -> str | None:
    """The message the customer opens with: ``user_turns[0]`` or ``user_query``."""
    scripted = _scripted_user_turns(scenario)
    if scripted is not None:
        return scripted[0]
    if isinstance(scenario, dict):
        query = scenario.get("user_query")
        if isinstance(query, str) and query.strip():
            return query
    return None


#: Phrases that end an unscripted episode when ``end_on_farewell`` is on.
FAREWELL_PHRASES: tuple[str, ...] = ("goodbye", "bye", "see you", "talk to you later")
_FAREWELL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(p) for p in FAREWELL_PHRASES) + r")\b",
    re.IGNORECASE,
)


class ConversationEnvironment(Environment):
    """Environment for open-ended conversations with a convenience step API.

    System prompt: the episode opens with a system turn only when one is
    explicitly provided — the ``system_prompt`` argument, a scenario's
    ``"system_prompt"``, or a scenario's ``"context"`` (rendered as
    ``"Context: ..."``). ``persona`` is appended to that turn when one is
    emitted. Otherwise :meth:`get_initial_prompt` returns ``""`` and the
    agent's own ``AgentConfig.system_prompt`` applies; when both exist, the
    environment's system turn takes precedence (``MultiTurnAgent`` does not
    add its own system prompt to a history that already has one).

    Episode end: ``max_turns``, the end of a scenario's scripted
    ``user_turns``, or — for unscripted scenarios with ``end_on_farewell``
    (default ``True``) — an assistant reply containing a farewell
    (:data:`FAREWELL_PHRASES`, matched as whole words). Scripted scenarios
    never end early on a farewell: the script defines the episode length.
    """

    scores_steps = True
    supports_concurrent_episodes = True

    def __init__(
        self,
        scenarios: list[dict[str, Any]],
        max_turns: int = 10,
        reward_fn: RewardFunction | None = None,
        persona: str | None = None,
        system_prompt: str | None = None,
        end_on_farewell: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_turns, reward_fn, **kwargs)
        self.scenarios = scenarios
        self.persona = persona
        self.system_prompt = system_prompt
        self.end_on_farewell = bool(end_on_farewell)
        self.current_scenario: dict[str, Any] | None = None
        self._last_state: EnvironmentState | None = None

    async def reset(
        self,
        scenario: dict[str, Any] | None = None,
        scenario_id: str | None = None,
    ) -> EnvironmentState:
        """Start a new episode and return initial state."""
        if scenario is None and scenario_id is not None:
            scenario = next(
                (s for s in self.scenarios if s.get("id") == scenario_id),
                None,
            )
            if scenario is None:
                raise ValueError(f"Unknown scenario_id: {scenario_id}")

        if scenario is None:
            scenario = random.choice(self.scenarios) if self.scenarios else {}
        self.current_scenario = scenario
        episode_id = str(uuid.uuid4())
        scenario_conv_id = None
        if isinstance(scenario, dict):
            scenario_conv_id = scenario.get("conversation_id")
        conversation_id = (
            str(scenario_conv_id)
            if scenario_conv_id is not None and str(scenario_conv_id)
            else episode_id
        )
        state = EnvironmentState(
            episode_id=episode_id,
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={
                "conversation_id": conversation_id,
                "scenario": scenario,
                "persona": self.persona,
                "conversation_topic": scenario.get("topic"),
                "user_goal": scenario.get("user_goal"),
                "history": [],
            },
        )
        opening = _opening_user_message(scenario)
        if opening is not None:
            state.context["history"] = [{"role": "user", "content": opening}]
        if isinstance(scenario, dict):
            if scenario.get("id") is not None:
                state.context["scenario_id"] = scenario.get("id")
            if scenario.get("task_id") is not None:
                state.context["task_id"] = scenario.get("task_id")
        self.active_episodes[episode_id] = state
        self._last_state = state
        return state

    async def step(
        self,
        *args: Any,
    ) -> Any:
        """Advance the environment by one turn.

        Supports both the explicit ``step(state, action)`` API used by training
        loops and a convenience ``step(action)`` mode backed by the last known
        environment state.
        """
        if len(args) == 1:
            return await self.step_stateful(cast(str | ConversationTurn, args[0]))
        if len(args) == 2:
            return await self._step_impl(
                cast(EnvironmentState, args[0]),
                cast(str | ConversationTurn, args[1]),
            )
        raise TypeError("step() expects either (action) or (state, action)")

    async def step_stateful(
        self,
        action: str | ConversationTurn,
    ) -> dict[str, Any]:
        """Advance the environment using internal state tracking."""
        if self._last_state is None:
            raise ValueError("Call reset() before step_stateful()")
        new_state, reward, done, info = await self._step_impl(self._last_state, action)
        payload: dict[str, Any] = new_state.as_dict()
        payload.update(
            {
                "state": new_state,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )
        return payload

    async def _step_impl(
        self,
        state: EnvironmentState,
        action: str | ConversationTurn,
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
        agent_turn = (
            action
            if isinstance(action, ConversationTurn)
            else ConversationTurn(role="assistant", content=str(action))
        )

        if state.status != EpisodeStatus.ONGOING:
            return state.copy(), 0.0, True, {"error": "Episode already completed"}

        new_state = state.copy()
        new_state.turn_count += 1

        prior_history = list(new_state.context.get("history", []))
        history = prior_history + [{"role": "assistant", "content": agent_turn.content}]
        new_state.context["history"] = history

        # Score the reply against the conversation it answers, before the
        # simulated user's next message exists.
        step_reward = 0.0
        if self.reward_fn is not None and self._should_compute_step_reward():
            reward_turns = [
                ConversationTurn(role=str(m["role"]), content=str(m["content"]))
                for m in prior_history
            ] + [agent_turn]
            try:
                reward_result = await self.reward_fn.compute_reward(
                    reward_turns,
                    new_state.context,
                )
                if hasattr(reward_result, "score"):
                    step_reward = float(reward_result.score)
                elif isinstance(reward_result, dict) and "score" in reward_result:
                    step_reward = float(reward_result["score"])
                elif isinstance(reward_result, (int, float)):
                    step_reward = float(reward_result)
            except ENVIRONMENT_EXCEPTIONS as reward_err:
                if getattr(self.reward_fn, "fail_on_error", False):
                    raise
                logger.debug("Reward computation failed: %s", reward_err)
                step_reward = 0.0
        elif self.reward_fn is None:
            step_reward = float(await self._calculate_step_reward(agent_turn, state))

        scripted = _scripted_user_turns(new_state.context.get("scenario"))
        user_response: ConversationTurn | None
        if scripted is None:
            user_response = await self._generate_user_response(agent_turn, state)
        elif new_state.turn_count < len(scripted):
            user_response = ConversationTurn(
                role="user",
                content=scripted[new_state.turn_count],
                metadata={"scripted": True, "turn_number": new_state.turn_count},
            )
        else:
            user_response = None
        if user_response is not None:
            history.append({"role": "user", "content": user_response.content})

        done = (
            new_state.turn_count >= self.max_turns
            or user_response is None
            or await self._should_end_conversation(agent_turn, state)
        )
        if done:
            new_state.status = EpisodeStatus.COMPLETED
        self._last_state = new_state

        info: dict[str, Any] = {
            "env_response": user_response,
            "assistant_turn": agent_turn,
            "scenario": new_state.context.get("scenario"),
        }
        return new_state, step_reward, bool(done), info

    async def get_initial_turns(
        self, state: EnvironmentState
    ) -> list[ConversationTurn]:
        """Open with the scenario's first user message when it has one."""
        return [
            ConversationTurn(role=str(m["role"]), content=str(m["content"]))
            for m in state.context.get("history", [])
        ]

    async def get_initial_prompt(self, scenario: dict[str, Any] | None = None) -> str:
        """The episode's system turn, or ``""`` when none is explicitly set.

        Returning ``""`` leaves the agent's own ``AgentConfig.system_prompt``
        in charge; see the class docstring for the precedence rules.
        """
        scenario = scenario if scenario is not None else self.current_scenario
        if not isinstance(scenario, dict):
            scenario = {}
        parts: list[str] = []
        explicit = self.system_prompt or scenario.get("system_prompt")
        if isinstance(explicit, str) and explicit.strip():
            parts.append(explicit.strip())
        if scenario.get("context"):
            parts.append(f"Context: {scenario['context']}")
        if not parts:
            return ""
        if self.persona:
            parts.insert(1, str(self.persona))
        return " ".join(parts)

    async def _generate_user_response(
        self, agent_turn: ConversationTurn, state: EnvironmentState
    ) -> ConversationTurn:
        scenario = state.context.get("scenario", {})
        user_responses = scenario.get(
            "user_responses",
            [
                "That's interesting, tell me more.",
                "Can you explain that differently?",
                "I see, what about other options?",
                "Thank you, that's helpful.",
            ],
        )
        response_content = random.choice(user_responses)
        return ConversationTurn(
            role="user",
            content=response_content,
            metadata={"generated": True, "turn_number": state.turn_count},
        )

    async def _calculate_step_reward(
        self, agent_turn: ConversationTurn, state: EnvironmentState
    ) -> float:
        base_reward = 0.1
        if len(agent_turn.content or "") > 50:
            base_reward += 0.1
        if len(agent_turn.content or "") < 10:
            base_reward -= 0.1
        return base_reward

    async def _should_end_conversation(
        self, agent_turn: ConversationTurn, state: EnvironmentState
    ) -> bool:
        """End early on an assistant farewell (unscripted scenarios only)."""
        if not self.end_on_farewell:
            return False
        if _scripted_user_turns(state.context.get("scenario")) is not None:
            return False
        return bool(_FAREWELL_RE.search(agent_turn.content or ""))

    def clone(self) -> ConversationEnvironment:
        """Create a new ConversationEnvironment with the same configuration."""
        return ConversationEnvironment(
            scenarios=list(self.scenarios),
            max_turns=self.max_turns,
            reward_fn=self.reward_fn,
            persona=self.persona,
            system_prompt=self.system_prompt,
            end_on_farewell=self.end_on_farewell,
            timeout_seconds=self.timeout_seconds,
        )
