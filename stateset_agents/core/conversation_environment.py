"""
Conversation environment implementation.
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import Any, overload

from .environment_base import (
    ENVIRONMENT_EXCEPTIONS,
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from .reward import RewardFunction
from .trajectory import ConversationTurn

logger = logging.getLogger(__name__)


class ConversationEnvironment(Environment):
    """Environment for open-ended conversations with a convenience step API."""

    def __init__(
        self,
        scenarios: list[dict[str, Any]],
        max_turns: int = 10,
        reward_fn: RewardFunction | None = None,
        persona: str | None = None,
        **kwargs,
    ):
        super().__init__(max_turns, reward_fn, **kwargs)
        self.scenarios = scenarios
        self.persona = persona
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
        if isinstance(scenario, dict):
            if scenario.get("id") is not None:
                state.context["scenario_id"] = scenario.get("id")
            if scenario.get("task_id") is not None:
                state.context["task_id"] = scenario.get("task_id")
        self.active_episodes[episode_id] = state
        self._last_state = state
        return state

    @overload
    async def step(
        self,
        state: EnvironmentState,
        action: str | ConversationTurn,
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
        ...

    @overload
    async def step(self, action: str | ConversationTurn) -> dict[str, Any]:
        ...

    async def step(
        self,
        state: EnvironmentState | str | ConversationTurn,
        action: str | ConversationTurn | None = None,
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]] | dict[str, Any]:
        """Advance the environment by one turn."""
        if action is None:
            if isinstance(state, EnvironmentState):
                raise TypeError(
                    "ConversationEnvironment.step() missing required argument: 'action'"
                )
            return await self.step_stateful(state)

        if not isinstance(state, EnvironmentState):
            raise TypeError(
                "ConversationEnvironment.step(state, action) requires state to be an EnvironmentState"
            )

        return await self._step_impl(state, action)

    async def step_stateful(
        self,
        action: str | ConversationTurn,
    ) -> dict[str, Any]:
        """Advance the environment using internal state tracking."""
        if self._last_state is None:
            raise ValueError("Call reset() before step_stateful()")
        new_state, reward, done, info = await self._step_impl(self._last_state, action)
        payload = new_state.as_dict()
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

        history = list(new_state.context.get("history", []))
        history.append({"role": "assistant", "content": agent_turn.content})

        user_response = await self._generate_user_response(agent_turn, state)
        history.append({"role": "user", "content": user_response.content})
        new_state.context["history"] = history

        step_reward = 0.0
        if self.reward_fn is not None and self._should_compute_step_reward():
            try:
                reward_result = await self.reward_fn.compute_reward(  # type: ignore[arg-type]
                    [agent_turn, user_response],
                    new_state.context,
                )
                if hasattr(reward_result, "score"):
                    step_reward = float(reward_result.score)
                elif isinstance(reward_result, dict) and "score" in reward_result:
                    step_reward = float(reward_result["score"])
                elif isinstance(reward_result, (int, float)):
                    step_reward = float(reward_result)
            except ENVIRONMENT_EXCEPTIONS as reward_err:
                logger.debug("Reward computation failed: %s", reward_err)
                step_reward = 0.0
        elif self.reward_fn is None:
            step_reward = float(await self._calculate_step_reward(agent_turn, state))

        done = (
            new_state.turn_count >= self.max_turns
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

    async def get_initial_prompt(
        self, scenario: dict[str, Any] | None = None
    ) -> str:
        scenario = scenario or self.current_scenario
        base_prompt = (
            "You are a helpful AI assistant. Engage in natural conversation with the user."
        )
        if self.persona:
            base_prompt += f" {self.persona}"
        if scenario and "context" in scenario:
            base_prompt += f" Context: {scenario['context']}"
        return base_prompt

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
        goodbye_phrases = ["goodbye", "bye", "see you", "talk to you later"]
        return any(
            phrase in (agent_turn.content or "").lower() for phrase in goodbye_phrases
        )

    def clone(self) -> ConversationEnvironment:
        """Create a new ConversationEnvironment with the same configuration."""
        return ConversationEnvironment(
            scenarios=list(self.scenarios),
            max_turns=self.max_turns,
            reward_fn=self.reward_fn,
            persona=self.persona,
            timeout_seconds=self.timeout_seconds,
        )
