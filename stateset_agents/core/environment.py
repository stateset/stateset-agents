"""
Environment classes for multi-turn agent training

This module defines the environments where agents interact and learn.
Environments handle state management, turn progression, and episode termination.
"""

import asyncio
import logging
import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

from .reward import RewardFunction
from .trajectory import ConversationTurn, MultiTurnTrajectory, TrajectoryGroup

logger = logging.getLogger(__name__)


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
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_done(self) -> bool:
        """Compatibility helper used by some tests."""
        return self.status != EpisodeStatus.ONGOING

    def copy(self) -> "EnvironmentState":
        return EnvironmentState(
            episode_id=self.episode_id,
            turn_count=self.turn_count,
            status=self.status,
            context=self.context.copy(),
            metadata=self.metadata.copy(),
        )

    # Dict-like helpers used by tests
    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "turn_count": self.turn_count,
            "step": self.turn_count,
            "status": self.status.value,
            "context": self.context,
            "metadata": self.metadata,
            # Convenience access used by some tests
            "scenario": self.context.get("scenario"),
        }

    def __contains__(self, key: str) -> bool:  # type: ignore[override]
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
        reward_fn: Optional[RewardFunction] = None,
        reward_function: Optional[RewardFunction] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.max_turns = max_turns
        self.reward_fn = reward_fn if reward_fn is not None else reward_function
        self.timeout_seconds = timeout_seconds
        self.active_episodes: Dict[str, EnvironmentState] = {}

    @property
    def reward_function(self) -> Optional[RewardFunction]:
        return self.reward_fn

    @reward_function.setter
    def reward_function(self, value: Optional[RewardFunction]) -> None:
        self.reward_fn = value

    @abstractmethod
    async def reset(self, scenario: Optional[Dict[str, Any]] = None) -> EnvironmentState:
        """Reset environment and return initial state"""
        pass

    @abstractmethod
    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Returns:
            (next_state, reward, done, info)
        """
        pass

    async def get_initial_prompt(self, scenario: Optional[Dict[str, Any]] = None) -> str:
        """Get initial prompt/context (override in subclasses)"""
        return ""

    async def get_reward(self, trajectory: Any) -> float:
        """Compute reward for a trajectory.

        This method provides a unified interface for getting rewards from the environment.
        It uses the reward_fn if configured, otherwise returns a neutral reward.

        Args:
            trajectory: A Trajectory object or similar with a `turns` attribute

        Returns:
            float: The computed reward score
        """
        if self.reward_fn is not None:
            try:
                # Extract turns from trajectory
                turns = getattr(trajectory, 'turns', [])
                if not turns and hasattr(trajectory, '__iter__'):
                    turns = list(trajectory)

                result = await self.reward_fn.compute_reward(turns)
                if hasattr(result, 'score'):
                    return float(result.score)
                elif isinstance(result, dict) and 'score' in result:
                    return float(result['score'])
                elif isinstance(result, (int, float)):
                    return float(result)
                else:
                    logger.warning(f"Unexpected reward result type: {type(result)}")
                    return 0.5
            except Exception as e:
                logger.warning(f"Failed to compute reward from reward_fn: {e}")
                return 0.5

        # No reward function configured, return neutral reward
        return 0.5

    async def run_episode(
        self,
        agent_fn: Callable,
        scenario: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
    ) -> MultiTurnTrajectory:
        """Run a complete episode using the standard step signature."""
        max_turns = max_turns or self.max_turns
        state = await self.reset(scenario)
        turns: List[ConversationTurn] = []
        turn_rewards: List[float] = []
        total_reward = 0.0

        initial_prompt = ""
        try:
            initial_prompt = await self.get_initial_prompt(scenario)
        except Exception:
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

            step_result = await self.step(state, agent_turn)
            if (
                isinstance(step_result, tuple)
                and len(step_result) == 4
                and isinstance(step_result[1], ConversationTurn)
            ):
                # Legacy (next_state, env_response, reward, done)
                new_state, env_response, step_reward, done = step_result  # type: ignore[misc]
                info: Dict[str, Any] = {}
            else:
                new_state, step_reward, done, info = step_result  # type: ignore[misc]
                env_response = info.get("env_response")

            total_reward += float(step_reward)
            turn_rewards.append(float(step_reward))
            if isinstance(env_response, ConversationTurn):
                turns.append(env_response)
            state = new_state
            if done or state.status != EpisodeStatus.ONGOING:
                break

        if self.reward_fn:
            try:
                final_reward = await self.reward_fn.compute_reward(turns, state.context)
                if hasattr(final_reward, "score"):
                    total_reward += final_reward.score
                elif isinstance(final_reward, dict) and "score" in final_reward:
                    total_reward += float(final_reward["score"])  # type: ignore[arg-type]
            except Exception:
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


class ConversationEnvironment(Environment):
    """Environment for open-ended conversations with a convenience step API."""

    def __init__(
        self,
        scenarios: List[Dict[str, Any]],
        max_turns: int = 10,
        reward_fn: Optional[RewardFunction] = None,
        persona: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(max_turns, reward_fn, **kwargs)
        self.scenarios = scenarios
        self.persona = persona
        self.current_scenario: Optional[Dict[str, Any]] = None
        self._last_state: Optional[EnvironmentState] = None

    async def reset(
        self,
        scenario: Optional[Dict[str, Any]] = None,
        scenario_id: Optional[str] = None,
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
        state = EnvironmentState(
            episode_id=episode_id,
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={
                "scenario": scenario,
                "persona": self.persona,
                "conversation_topic": scenario.get("topic"),
                "user_goal": scenario.get("user_goal"),
                "history": [],
            },
        )
        self.active_episodes[episode_id] = state
        self._last_state = state
        return state

    async def step(
        self,
        state: Union[EnvironmentState, str, ConversationTurn],
        action: Optional[Union[str, ConversationTurn]] = None,
    ) -> Union[Tuple[EnvironmentState, float, bool, Dict[str, Any]], Dict[str, Any]]:
        """Advance the environment by one turn.

        Supports two calling conventions:
        - `await env.step(state, action)` returns `(next_state, reward, done, info)`
        - `await env.step(action)` uses the most recent state from `reset()` and
          returns a dict with common keys (`step`, `reward`, `done`, `state`, `info`).
        """

        if action is None:
            if self._last_state is None:
                raise ValueError("Call reset() before step() without explicit state")
            new_state, reward, done, info = await self._step_impl(self._last_state, state)
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

        if not isinstance(state, EnvironmentState):
            raise TypeError("step(state, action) requires an EnvironmentState as the first argument")

        return await self._step_impl(state, action)

    async def _step_impl(
        self,
        state: EnvironmentState,
        action: Union[str, ConversationTurn],
    ) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
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
        if self.reward_fn is not None:
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
            except Exception as reward_err:
                logger.debug("Reward computation failed: %s", reward_err)
                step_reward = 0.0
        else:
            step_reward = float(await self._calculate_step_reward(agent_turn, state))

        done = (
            new_state.turn_count >= self.max_turns
            or await self._should_end_conversation(agent_turn, state)
        )
        if done:
            new_state.status = EpisodeStatus.COMPLETED
        self._last_state = new_state

        info: Dict[str, Any] = {
            "env_response": user_response,
            "assistant_turn": agent_turn,
            "scenario": new_state.context.get("scenario"),
        }
        return new_state, step_reward, bool(done), info

    async def get_initial_prompt(self, scenario: Optional[Dict[str, Any]] = None) -> str:
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


class TaskEnvironment(Environment):
    """
    Environment for task-oriented interactions
    """

    def __init__(
        self,
        tasks: List[Dict[str, Any]],
        success_criteria: Callable[[List[ConversationTurn], Dict[str, Any]], bool],
        max_turns: int = 20,
        reward_fn: Optional[RewardFunction] = None,
        **kwargs,
    ):
        super().__init__(max_turns, reward_fn, **kwargs)
        self.tasks = tasks
        self.success_criteria = success_criteria
        self.current_task = None

    async def reset(
        self, scenario: Optional[Dict[str, Any]] = None
    ) -> EnvironmentState:
        """Reset for a new task"""
        if scenario is None:
            scenario = random.choice(self.tasks)

        self.current_task = scenario
        episode_id = str(uuid.uuid4())

        state = EnvironmentState(
            episode_id=episode_id,
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={
                "task": scenario,
                "task_goal": scenario.get("goal"),
                "task_type": scenario.get("type"),
                "required_actions": scenario.get("required_actions", []),
                "completed_actions": [],
                "task_progress": 0.0,
            },
        )

        self.active_episodes[episode_id] = state
        self._last_state = state
        return state

    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> Tuple[EnvironmentState, ConversationTurn, float, bool]:
        """Process agent action and update task state"""
        new_state = state.copy()
        new_state.turn_count += 1

        # Update task progress based on action
        await self._update_task_progress(action, new_state)

        # Generate environment response
        env_response = await self._generate_task_response(action, new_state)

        # Calculate reward
        step_reward = await self._calculate_task_reward(action, new_state)

        # Check if task is complete
        task_complete = await self._check_task_completion(new_state)
        done = task_complete or new_state.turn_count >= self.max_turns

        if done:
            new_state.status = (
                EpisodeStatus.COMPLETED if task_complete else EpisodeStatus.TIMEOUT
            )

        return new_state, env_response, step_reward, done

    async def get_initial_prompt(
        self, scenario: Optional[Dict[str, Any]] = None
    ) -> str:
        """Get initial task description"""
        if not scenario:
            scenario = self.current_task

        task_prompt = f"Task: {scenario.get('description', 'Complete the given task.')}"

        if scenario.get("instructions"):
            task_prompt += f"\n\nInstructions: {scenario['instructions']}"

        return task_prompt

    async def _update_task_progress(
        self, action: ConversationTurn, state: EnvironmentState
    ):
        """Update task progress based on agent action"""
        # Simple progress tracking based on keywords/actions
        required_actions = state.context.get("required_actions", [])
        completed_actions = state.context.get("completed_actions", [])

        action_content = action.content.lower()

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

        # Reward based on progress made this turn
        previous_progress = state.context.get("previous_progress", 0.0)
        progress_delta = progress - previous_progress

        state.context["previous_progress"] = progress

        # Base reward for progress
        reward = progress_delta * 10.0

        # Bonus for task completion
        if progress == 1.0:
            reward += 5.0

        return reward

    async def _check_task_completion(self, state: EnvironmentState) -> bool:
        """Check if task is completed"""
        return state.context.get("task_progress", 0.0) >= 1.0


# Utility function for creating environments
def create_environment(env_type: str, config: Dict[str, Any]) -> Environment:
    """Factory function for creating environments"""
    if env_type == "conversation":
        return ConversationEnvironment(**config)
    elif env_type == "task":
        return TaskEnvironment(**config)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


# Pre-defined environment configurations
CONVERSATION_CONFIGS = {
    "customer_service": {
        "scenarios": [
            {
                "topic": "product_inquiry",
                "user_goal": "Learn about product features",
                "context": "User is interested in purchasing a product",
            },
            {
                "topic": "technical_support",
                "user_goal": "Resolve technical issue",
                "context": "User is experiencing a problem with their device",
            },
        ],
        "persona": "You are a professional customer service representative.",
        "max_turns": 15,
    },
    "technical_support": {
        "scenarios": [
            {
                "topic": "app_crash",
                "user_goal": "Fix an application crashing on launch",
                "context": "User reports the app crashes immediately after opening.",
            },
            {
                "topic": "network_issue",
                "user_goal": "Restore connectivity",
                "context": "User cannot connect to Wi-Fi after a recent update.",
            },
            {
                "topic": "performance",
                "user_goal": "Improve slow system performance",
                "context": "User notices the computer is slow and wants troubleshooting steps.",
            },
        ],
        "persona": "You are a user seeking technical support. Provide symptoms and answer questions.",
        "max_turns": 15,
    },
    "sales": {
        "scenarios": [
            {
                "topic": "plan_selection",
                "user_goal": "Choose the right plan",
                "context": "User is comparing tiers and wants a recommendation.",
            },
            {
                "topic": "pricing",
                "user_goal": "Understand pricing and discounts",
                "context": "User asks about pricing, billing, and whether there are promotions.",
            },
            {
                "topic": "objection_handling",
                "user_goal": "Address concerns and decide",
                "context": "User is interested but worried about cost and onboarding effort.",
            },
        ],
        "persona": "You are a potential customer exploring products. Ask questions and raise objections.",
        "max_turns": 15,
    },
    "tutoring": {
        "scenarios": [
            {
                "topic": "math_help",
                "user_goal": "Understand a math concept",
                "context": "Student needs help with algebra",
            },
            {
                "topic": "essay_writing",
                "user_goal": "Improve writing skills",
                "context": "Student working on an essay",
            },
        ],
        "persona": "You are a patient and encouraging tutor.",
        "max_turns": 20,
    },
}

TASK_CONFIGS = {
    "data_analysis": {
        "tasks": [
            {
                "description": "Analyze the provided dataset and generate insights",
                "goal": "Complete data analysis",
                "type": "analysis",
                "required_actions": [
                    {"name": "load_data", "keywords": ["load", "read", "import"]},
                    {
                        "name": "explore_data",
                        "keywords": ["explore", "summary", "describe"],
                    },
                    {"name": "visualize", "keywords": ["plot", "chart", "graph"]},
                    {
                        "name": "insights",
                        "keywords": ["insight", "conclusion", "finding"],
                    },
                ],
            }
        ]
    }
}
