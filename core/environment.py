"""
Environment classes for multi-turn agent training

This module defines the environments where agents interact and learn.
Environments handle state management, turn progression, and episode termination.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator, Union, Callable
from dataclasses import dataclass, field
import asyncio
import logging
from enum import Enum
import random
import uuid

from .trajectory import ConversationTurn, MultiTurnTrajectory, TrajectoryGroup
from .reward import RewardFunction

logger = logging.getLogger(__name__)


class EpisodeStatus(Enum):
    """Status of an episode"""
    ONGOING = "ongoing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class EnvironmentState:
    """
    Represents the state of an environment at a given time
    """
    episode_id: str
    turn_count: int
    status: EpisodeStatus
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def copy(self) -> 'EnvironmentState':
        """Create a copy of the state"""
        return EnvironmentState(
            episode_id=self.episode_id,
            turn_count=self.turn_count,
            status=self.status,
            context=self.context.copy(),
            metadata=self.metadata.copy()
        )


class Environment(ABC):
    """
    Abstract base class for all environments
    """
    
    def __init__(
        self,
        max_turns: int = 10,
        reward_fn: Optional[RewardFunction] = None,
        timeout_seconds: Optional[float] = None
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
        self,
        state: EnvironmentState,
        action: ConversationTurn
    ) -> Tuple[EnvironmentState, ConversationTurn, float, bool]:
        """
        Execute one step in the environment
        
        Returns:
            new_state: Updated environment state
            response: Environment's response to the action
            reward: Immediate reward for this step
            done: Whether episode is complete
        """
        pass
    
    @abstractmethod
    async def get_initial_prompt(self, scenario: Optional[Dict[str, Any]] = None) -> str:
        """Get the initial prompt/context for the agent"""
        pass
    
    async def run_episode(
        self,
        agent_fn: Callable,
        scenario: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None
    ) -> MultiTurnTrajectory:
        """
        Run a complete episode with an agent
        """
        max_turns = max_turns or self.max_turns
        state = await self.reset(scenario)
        turns = []
        turn_rewards = []
        total_reward = 0.0
        
        # Get initial prompt
        initial_prompt = await self.get_initial_prompt(scenario)
        if initial_prompt:
            system_turn = ConversationTurn(
                role="system",
                content=initial_prompt,
                metadata={"scenario": scenario}
            )
            turns.append(system_turn)
        
        try:
            for turn_num in range(max_turns):
                # Agent's turn
                conversation_history = [
                    {"role": turn.role, "content": turn.content} for turn in turns
                ]
                
                agent_response = await agent_fn(conversation_history, state.context)
                
                if isinstance(agent_response, str):
                    agent_turn = ConversationTurn(
                        role="assistant",
                        content=agent_response
                    )
                else:
                    agent_turn = agent_response
                
                turns.append(agent_turn)
                
                # Environment step
                new_state, env_response, step_reward, done = await self.step(state, agent_turn)
                
                total_reward += step_reward
                turn_rewards.append(step_reward)
                
                if env_response:
                    turns.append(env_response)
                
                state = new_state
                
                if done or state.status != EpisodeStatus.ONGOING:
                    break
            
            # Final reward calculation
            if self.reward_fn:
                final_reward = await self.reward_fn.compute_reward(turns, state.context)
                total_reward += final_reward
            
        except Exception as e:
            logger.error(f"Episode failed: {e}")
            state.status = EpisodeStatus.FAILED
        
        trajectory = MultiTurnTrajectory(
            turns=turns,
            total_reward=total_reward,
            turn_rewards=turn_rewards,
            initial_state={"scenario": scenario} if scenario else None,
            final_state=state.context,
            metadata={
                "episode_id": state.episode_id,
                "status": state.status.value,
                "scenario": scenario
            }
        )
        
        return trajectory
    
    async def generate_trajectory_group(
        self,
        agent_fn: Callable,
        scenario: Dict[str, Any],
        num_trajectories: int = 4
    ) -> TrajectoryGroup:
        """
        Generate a group of trajectories for the same scenario (for GRPO)
        """
        trajectories = []
        
        tasks = [
            self.run_episode(agent_fn, scenario)
            for _ in range(num_trajectories)
        ]
        
        trajectories = await asyncio.gather(*tasks)
        
        return TrajectoryGroup(
            scenario_id=scenario.get("id", str(uuid.uuid4())),
            trajectories=trajectories,
            scenario_metadata=scenario
        )


class ConversationEnvironment(Environment):
    """
    Environment for open-ended conversations
    """
    
    def __init__(
        self,
        scenarios: List[Dict[str, Any]],
        max_turns: int = 10,
        reward_fn: Optional[RewardFunction] = None,
        persona: Optional[str] = None,
        **kwargs
    ):
        super().__init__(max_turns, reward_fn, **kwargs)
        self.scenarios = scenarios
        self.persona = persona
        self.current_scenario = None
    
    async def reset(self, scenario: Optional[Dict[str, Any]] = None) -> EnvironmentState:
        """Reset for a new conversation"""
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
            }
        )
        
        self.active_episodes[episode_id] = state
        return state
    
    async def step(
        self,
        state: EnvironmentState,
        action: ConversationTurn
    ) -> Tuple[EnvironmentState, ConversationTurn, float, bool]:
        """
        Process agent's response and generate user's next turn
        """
        new_state = state.copy()
        new_state.turn_count += 1
        
        # Simulate user response based on scenario
        user_response = await self._generate_user_response(action, state)
        
        # Calculate reward for this turn
        step_reward = await self._calculate_step_reward(action, state)
        
        # Check if conversation should end
        done = (
            new_state.turn_count >= self.max_turns or
            await self._should_end_conversation(action, state)
        )
        
        if done:
            new_state.status = EpisodeStatus.COMPLETED
        
        return new_state, user_response, step_reward, done
    
    async def get_initial_prompt(self, scenario: Optional[Dict[str, Any]] = None) -> str:
        """Get system prompt for the conversation"""
        if not scenario:
            scenario = self.current_scenario
        
        base_prompt = "You are a helpful AI assistant. Engage in natural conversation with the user."
        
        if self.persona:
            base_prompt += f" {self.persona}"
        
        if scenario and "context" in scenario:
            base_prompt += f" Context: {scenario['context']}"
        
        return base_prompt
    
    async def _generate_user_response(
        self,
        agent_turn: ConversationTurn,
        state: EnvironmentState
    ) -> ConversationTurn:
        """Generate user's response to agent"""
        scenario = state.context.get("scenario", {})
        
        # Simple rule-based user simulation
        # In practice, this could be another LLM or human-in-the-loop
        user_responses = scenario.get("user_responses", [
            "That's interesting, tell me more.",
            "Can you explain that differently?",
            "I see, what about other options?",
            "Thank you, that's helpful."
        ])
        
        response_content = random.choice(user_responses)
        
        return ConversationTurn(
            role="user",
            content=response_content,
            metadata={"generated": True, "turn_number": state.turn_count}
        )
    
    async def _calculate_step_reward(
        self,
        agent_turn: ConversationTurn,
        state: EnvironmentState
    ) -> float:
        """Calculate reward for agent's turn"""
        # Basic reward calculation
        # In practice, this would be more sophisticated
        base_reward = 0.1  # Small positive reward for engagement
        
        # Bonus for longer, more detailed responses
        if len(agent_turn.content) > 50:
            base_reward += 0.1
        
        # Penalty for very short responses
        if len(agent_turn.content) < 10:
            base_reward -= 0.1
        
        return base_reward
    
    async def _should_end_conversation(
        self,
        agent_turn: ConversationTurn,
        state: EnvironmentState
    ) -> bool:
        """Determine if conversation should end"""
        # End if agent says goodbye or conversation reaches natural conclusion
        goodbye_phrases = ["goodbye", "bye", "see you", "talk to you later"]
        content_lower = agent_turn.content.lower()
        
        return any(phrase in content_lower for phrase in goodbye_phrases)


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
        **kwargs
    ):
        super().__init__(max_turns, reward_fn, **kwargs)
        self.tasks = tasks
        self.success_criteria = success_criteria
        self.current_task = None
    
    async def reset(self, scenario: Optional[Dict[str, Any]] = None) -> EnvironmentState:
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
                "task_progress": 0.0
            }
        )
        
        self.active_episodes[episode_id] = state
        return state
    
    async def step(
        self,
        state: EnvironmentState,
        action: ConversationTurn
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
            new_state.status = EpisodeStatus.COMPLETED if task_complete else EpisodeStatus.TIMEOUT
        
        return new_state, env_response, step_reward, done
    
    async def get_initial_prompt(self, scenario: Optional[Dict[str, Any]] = None) -> str:
        """Get initial task description"""
        if not scenario:
            scenario = self.current_task
        
        task_prompt = f"Task: {scenario.get('description', 'Complete the given task.')}"
        
        if scenario.get("instructions"):
            task_prompt += f"\n\nInstructions: {scenario['instructions']}"
        
        return task_prompt
    
    async def _update_task_progress(
        self,
        action: ConversationTurn,
        state: EnvironmentState
    ):
        """Update task progress based on agent action"""
        # Simple progress tracking based on keywords/actions
        required_actions = state.context.get("required_actions", [])
        completed_actions = state.context.get("completed_actions", [])
        
        action_content = action.content.lower()
        
        for req_action in required_actions:
            if req_action not in completed_actions:
                if any(keyword in action_content for keyword in req_action.get("keywords", [])):
                    completed_actions.append(req_action)
        
        state.context["completed_actions"] = completed_actions
        state.context["task_progress"] = len(completed_actions) / max(1, len(required_actions))
    
    async def _generate_task_response(
        self,
        action: ConversationTurn,
        state: EnvironmentState
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
            role="system",
            content=response,
            metadata={"task_progress": progress}
        )
    
    async def _calculate_task_reward(
        self,
        action: ConversationTurn,
        state: EnvironmentState
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
def create_environment(
    env_type: str,
    config: Dict[str, Any]
) -> Environment:
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
                "context": "User is interested in purchasing a product"
            },
            {
                "topic": "technical_support", 
                "user_goal": "Resolve technical issue",
                "context": "User is experiencing a problem with their device"
            }
        ],
        "persona": "You are a professional customer service representative.",
        "max_turns": 15
    },
    
    "tutoring": {
        "scenarios": [
            {
                "topic": "math_help",
                "user_goal": "Understand a math concept",
                "context": "Student needs help with algebra"
            },
            {
                "topic": "essay_writing",
                "user_goal": "Improve writing skills", 
                "context": "Student working on an essay"
            }
        ],
        "persona": "You are a patient and encouraging tutor.",
        "max_turns": 20
    }
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
                    {"name": "explore_data", "keywords": ["explore", "summary", "describe"]},
                    {"name": "visualize", "keywords": ["plot", "chart", "graph"]},
                    {"name": "insights", "keywords": ["insight", "conclusion", "finding"]}
                ]
            }
        ]
    }
}