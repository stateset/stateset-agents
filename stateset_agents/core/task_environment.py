"""
Task-oriented environment implementation.
"""

from __future__ import annotations

import random
import uuid
from typing import Any
from collections.abc import Callable

from .environment_base import Environment, EnvironmentState, EpisodeStatus
from .reward import RewardFunction
from .trajectory import ConversationTurn


class TaskEnvironment(Environment):
    """
    Environment for task-oriented interactions
    """

    def __init__(
        self,
        tasks: list[dict[str, Any]],
        success_criteria: Callable[[list[ConversationTurn], dict[str, Any]], bool],
        max_turns: int = 20,
        reward_fn: RewardFunction | None = None,
        **kwargs,
    ):
        super().__init__(max_turns, reward_fn, **kwargs)
        self.tasks = tasks
        self.success_criteria = success_criteria
        self.current_task: dict[str, Any] | None = None
        self._last_state: EnvironmentState | None = None

    async def reset(
        self, scenario: dict[str, Any] | None = None
    ) -> EnvironmentState:
        """Reset for a new task"""
        selected_scenario = scenario if scenario is not None else random.choice(self.tasks)

        self.current_task = selected_scenario
        episode_id = str(uuid.uuid4())
        scenario_conv_id = None
        scenario_conv_id = selected_scenario.get("conversation_id")
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
                "task": selected_scenario,
                "task_goal": selected_scenario.get("goal"),
                "task_type": selected_scenario.get("type"),
                "required_actions": selected_scenario.get("required_actions", []),
                "completed_actions": [],
                "task_progress": 0.0,
                "turns": [],
            },
        )
        task_id = selected_scenario.get("task_id", selected_scenario.get("id"))
        if task_id is not None:
            state.context["task_id"] = task_id

        self.active_episodes[episode_id] = state
        self._last_state = state
        return state

    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> Any:
        """Process agent action and update task state"""
        new_state = state.copy()
        new_state.turn_count += 1

        turns = list(new_state.context.get("turns", []))
        turns.append(action)

        await self._update_task_progress(action, new_state)

        env_response = await self._generate_task_response(action, new_state)
        turns.append(env_response)
        new_state.context["turns"] = turns

        step_reward = await self._calculate_task_reward(action, new_state)

        task_complete = await self._check_task_completion(new_state)
        done = task_complete or new_state.turn_count >= self.max_turns

        if done:
            new_state.status = (
                EpisodeStatus.COMPLETED if task_complete else EpisodeStatus.TIMEOUT
            )

        return new_state, env_response, step_reward, done

    async def get_initial_prompt(
        self, scenario: dict[str, Any] | None = None
    ) -> str:
        """Get initial task description"""
        active_scenario = scenario or self.current_task or {}

        task_prompt = (
            f"Task: {active_scenario.get('description', 'Complete the given task.')}"
        )

        if active_scenario.get("instructions"):
            task_prompt += f"\n\nInstructions: {active_scenario['instructions']}"

        return task_prompt

    async def _update_task_progress(
        self, action: ConversationTurn, state: EnvironmentState
    ) -> None:
        """Update task progress based on agent action"""
        required_actions = state.context.get("required_actions", [])
        completed_actions = state.context.get("completed_actions", [])

        action_content = (action.content or "").lower()

        for req_action in required_actions:
            if req_action not in completed_actions:
                if any(
                    keyword in action_content
                    for keyword in req_action.get("keywords", [])
                ):
                    completed_actions.append(req_action)

        state.context["completed_actions"] = completed_actions
        state.context["task_progress"] = len(completed_actions) / max(
            1, len(required_actions)
        )

    async def _generate_task_response(
        self, action: ConversationTurn, state: EnvironmentState
    ) -> ConversationTurn:
        """Generate environment response for task"""
        progress = state.context.get("task_progress", 0.0)

        if progress == 1.0:
            response = "Task completed successfully!"
        elif progress > 0.5:
            response = "Good progress! Continue with the remaining steps."
        else:
            response = "Please proceed with the task requirements."

        return ConversationTurn(
            role="system", content=response, metadata={"task_progress": progress}
        )

    async def _calculate_task_reward(
        self, action: ConversationTurn, state: EnvironmentState
    ) -> float:
        """Calculate reward based on task progress"""
        progress = state.context.get("task_progress", 0.0)
        previous_progress = state.context.get("previous_progress", 0.0)
        progress_delta = progress - previous_progress

        state.context["previous_progress"] = progress

        reward = progress_delta * 10.0
        if progress == 1.0:
            reward += 5.0

        return float(reward)

    async def _check_task_completion(self, state: EnvironmentState) -> bool:
        """Check if task is completed using success_criteria"""
        turns = state.context.get("turns", [])
        return self.success_criteria(turns, state.context)

    def clone(self) -> TaskEnvironment:
        """Create a new TaskEnvironment with the same configuration."""
        return TaskEnvironment(
            tasks=list(self.tasks),
            success_criteria=self.success_criteria,
            max_turns=self.max_turns,
            reward_fn=self.reward_fn,
            timeout_seconds=self.timeout_seconds,
        )
