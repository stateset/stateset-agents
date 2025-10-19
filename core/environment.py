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
            "step": self.turn_count,
            "status": self.status.value,
            "context": self.context,
            "metadata": self.metadata,
        }

    def __contains__(self, key: str) -> bool:  # type: ignore[override]
        return key in self.as_dict()

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


class Environment(ABC):
    """Abstract base class for all environments"""

    def __init__(
        self,
        max_turns: int = 10,
        reward_fn: Optional[RewardFunction] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.max_turns = max_turns
        self.reward_fn = reward_fn
        self.timeout_seconds = timeout_seconds
        self.active_episodes: Dict[str, EnvironmentState] = {}

    @abstractmethod
    async def reset(self, scenario: Optional[Dict[str, Any]] = None) -> EnvironmentState:
        """Reset environment and return initial state"""
        pass

    @abstractmethod
    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> Tuple[EnvironmentState, ConversationTurn, float, bool]:
        """Execute one step in the environment"""
        pass

    async def get_initial_prompt(self, scenario: Optional[Dict[str, Any]] = None) -> str:
        """Get initial prompt/context (override in subclasses)"""
        return ""

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

            new_state, env_response, step_reward, done = await self.step(state, agent_turn)
            total_reward += step_reward
            turn_rewards.append(step_reward)
            if env_response:
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

    async def reset(self, scenario: Optional[Dict[str, Any]] = None) -> EnvironmentState:
        if scenario is None:
            scenario = random.choice(self.scenarios)
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
            },
        )
        self.active_episodes[episode_id] = state
        self._last_state = state
        return state

    async def step(
        self, *args, **kwargs
    ) -> Union[Tuple[EnvironmentState, ConversationTurn, float, bool], Dict[str, Any]]:
        if len(args) == 1 and not kwargs:
            action_input = args[0]
            state = self._last_state or await self.reset()
            agent_turn = (
                action_input
                if isinstance(action_input, ConversationTurn)
                else ConversationTurn(role="assistant", content=str(action_input))
            )
            new_state = state.copy()
            new_state.turn_count += 1
            _ = await self._generate_user_response(agent_turn, state)
            step_reward = await self._calculate_step_reward(agent_turn, state)
            done = (
                new_state.turn_count >= self.max_turns
                or await self._should_end_conversation(agent_turn, state)
            )
            if done:
                new_state.status = EpisodeStatus.COMPLETED
            self._last_state = new_state
            return {"step": new_state.turn_count, "reward": step_reward, "done": done}

        # Standard form
        state, action = args[0], args[1]
        agent_turn = (
            action
            if isinstance(action, ConversationTurn)
            else ConversationTurn(role="assistant", content=str(action))
        )
        new_state = state.copy()
        new_state.turn_count += 1
        user_response = await self._generate_user_response(agent_turn, state)
        step_reward = await self._calculate_step_reward(agent_turn, state)
        done = (
            new_state.turn_count >= self.max_turns
            or await self._should_end_conversation(agent_turn, state)
        )
        if done:
            new_state.status = EpisodeStatus.COMPLETED
        self._last_state = new_state
        return new_state, user_response, step_reward, done

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
