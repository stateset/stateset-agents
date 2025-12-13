"""
Gym/Gymnasium Environment Adapter

Wraps Gym/Gymnasium environments to work with the stateset-agents framework.
Maps gym's (obs, reward, done, info) API to Environment's (state, turn, reward, done) API.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional, Tuple

from ..environment import Environment, EnvironmentState, EpisodeStatus
from ..trajectory import ConversationTurn
from .processors import ObservationProcessor, create_observation_processor
from .mappers import ActionMapper, create_action_mapper


logger = logging.getLogger(__name__)


class GymEnvironmentAdapter(Environment):
    """
    Adapter that wraps Gym/Gymnasium environments for use with GRPO agents.

    This adapter converts between gym's (obs, reward, done, info) format and
    the framework's (state, turn, reward, done) format. Each gym step becomes
    a "turn" in a multi-turn episode.

    Args:
        gym_env: The gym/gymnasium environment instance
        observation_processor: Processor to convert observations to text/structured format
        action_mapper: Mapper to convert agent responses to gym actions
        max_steps: Maximum steps per episode (overrides gym's default)
        negative_reward_on_invalid: Reward penalty for invalid actions
        auto_create_processors: Auto-create processor/mapper if not provided

    Example:
        >>> import gymnasium as gym
        >>> from core.gym.adapter import GymEnvironmentAdapter
        >>>
        >>> env = gym.make("CartPole-v1")
        >>> adapter = GymEnvironmentAdapter(env, auto_create_processors=True)
        >>>
        >>> state = await adapter.reset()
        >>> # Now use with GRPO trainer as a normal Environment
    """

    def __init__(
        self,
        gym_env: Any,
        observation_processor: Optional[ObservationProcessor] = None,
        action_mapper: Optional[ActionMapper] = None,
        max_steps: Optional[int] = None,
        negative_reward_on_invalid: float = -1.0,
        auto_create_processors: bool = True,
        **kwargs
    ):
        # Determine max steps
        if max_steps is None:
            # Try to get from gym env spec
            spec = getattr(gym_env, "spec", None)
            raw_max_steps = getattr(spec, "max_episode_steps", None)
            if isinstance(raw_max_steps, int) and raw_max_steps > 0:
                max_steps = raw_max_steps
            else:
                max_steps = 500  # Reasonable default

        super().__init__(max_turns=max_steps, **kwargs)

        self.gym_env = gym_env
        self.negative_reward_on_invalid = negative_reward_on_invalid
        self.env_id = gym_env.spec.id if hasattr(gym_env, 'spec') else "Unknown"

        # Auto-create processors if requested and not provided
        if auto_create_processors:
            if observation_processor is None:
                observation_processor = create_observation_processor(self.env_id)
            if action_mapper is None:
                action_mapper = create_action_mapper(gym_env)

        if observation_processor is None:
            raise ValueError(
                "observation_processor is required. Set auto_create_processors=True "
                "or provide an ObservationProcessor instance."
            )
        if action_mapper is None:
            raise ValueError(
                "action_mapper is required. Set auto_create_processors=True "
                "or provide an ActionMapper instance."
            )

        self.observation_processor = observation_processor
        self.action_mapper = action_mapper

        # Track current gym state for each episode
        self._gym_states: Dict[str, Any] = {}
        self._episode_rewards: Dict[str, float] = {}

    async def reset(self, scenario: Optional[Dict[str, Any]] = None) -> EnvironmentState:
        """
        Reset the gym environment and return initial state.

        Args:
            scenario: Optional dict (unused for gym, kept for API compatibility)

        Returns:
            Initial EnvironmentState with observation in context
        """
        # Reset gym environment
        try:
            # Try new gymnasium API first (returns obs, info)
            result = self.gym_env.reset()
            if isinstance(result, tuple):
                obs, info = result
            else:
                obs = result
                info = {}
        except Exception as e:
            logger.error(f"Error resetting gym environment: {e}")
            raise

        # Create new episode
        episode_id = str(uuid.uuid4())

        # Process observation
        obs_text = self.observation_processor.process(obs)

        # Create initial state
        state = EnvironmentState(
            episode_id=episode_id,
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={
                "observation": obs,
                "observation_text": obs_text,
                "cumulative_reward": 0.0,
                "env_id": self.env_id,
            },
            metadata={
                "gym_info": info,
            }
        )

        # Track state
        self.active_episodes[episode_id] = state
        self._gym_states[episode_id] = obs
        self._episode_rewards[episode_id] = 0.0

        logger.debug(f"Reset gym environment {self.env_id}, episode {episode_id}")

        return state

    async def step(
        self,
        state: EnvironmentState,
        action: ConversationTurn
    ) -> Tuple[EnvironmentState, ConversationTurn, float, bool]:
        """
        Execute one step in the gym environment.

        Args:
            state: Current environment state
            action: Agent's action (as ConversationTurn with text content)

        Returns:
            Tuple of (new_state, observation_turn, reward, done)
        """
        episode_id = state.episode_id

        # Parse action from agent's text response
        try:
            gym_action = self.action_mapper.parse_action(
                action.content,
                context={"state": state}
            )
        except Exception as e:
            logger.error(f"Error parsing action from '{action.content}': {e}")
            # Use fallback action (random)
            gym_action = self.gym_env.action_space.sample()
            reward = self.negative_reward_on_invalid
            done = False
            obs = self._gym_states.get(episode_id)
            info = {"error": "invalid_action"}

            # Create error observation turn
            obs_text = f"Invalid action. Using fallback. Current observation: {self.observation_processor.process(obs)}"
            obs_turn = ConversationTurn(
                role="system",
                content=obs_text,
                metadata={"gym_info": info, "error": "invalid_action"}
            )

            # Update state
            new_state = state.copy()
            new_state.turn_count += 1
            new_state.context["observation_text"] = obs_text
            new_state.context["cumulative_reward"] = state.context.get("cumulative_reward", 0.0) + reward

            return new_state, obs_turn, reward, done

        # Execute action in gym environment
        try:
            result = self.gym_env.step(gym_action)

            # Handle both old gym (obs, reward, done, info) and new gymnasium (obs, reward, terminated, truncated, info)
            if len(result) == 4:
                obs, reward, done, info = result
            elif len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                raise ValueError(f"Unexpected gym step result length: {len(result)}")

        except Exception as e:
            logger.error(f"Error executing gym step with action {gym_action}: {e}")
            # Fallback: treat as failed episode
            obs = self._gym_states.get(episode_id)
            reward = self.negative_reward_on_invalid
            done = True
            info = {"error": "gym_step_failed"}

        # Update tracking
        self._gym_states[episode_id] = obs
        self._episode_rewards[episode_id] = self._episode_rewards.get(episode_id, 0.0) + reward

        # Process observation
        obs_text = self.observation_processor.process(obs, context=state.context)

        # Create observation turn
        obs_turn = ConversationTurn(
            role="system",
            content=obs_text,
            reward=reward,
            metadata={
                "gym_observation": obs,
                "gym_action": gym_action,
                "gym_info": info,
            }
        )

        # Update state
        new_state = state.copy()
        new_state.turn_count += 1
        new_state.context["observation"] = obs
        new_state.context["observation_text"] = obs_text
        new_state.context["cumulative_reward"] = self._episode_rewards[episode_id]

        # Check termination
        if done:
            new_state.status = EpisodeStatus.COMPLETED
            # Cleanup tracking
            self.active_episodes.pop(episode_id, None)
            self._gym_states.pop(episode_id, None)
            self._episode_rewards.pop(episode_id, None)
        elif new_state.turn_count >= self.max_turns:
            new_state.status = EpisodeStatus.TIMEOUT
            done = True
            # Cleanup tracking
            self.active_episodes.pop(episode_id, None)
            self._gym_states.pop(episode_id, None)
            self._episode_rewards.pop(episode_id, None)

        logger.debug(
            f"Gym step: action={gym_action}, reward={reward:.3f}, "
            f"done={done}, turn={new_state.turn_count}/{self.max_turns}"
        )

        return new_state, obs_turn, reward, done

    async def get_initial_prompt(self, scenario: Optional[Dict[str, Any]] = None) -> str:
        """
        Get initial task description for the agent.

        Returns:
            System prompt describing the gym task
        """
        return self.observation_processor.get_system_prompt(self.gym_env)

    def close(self):
        """Close the gym environment and cleanup resources."""
        if hasattr(self.gym_env, 'close'):
            self.gym_env.close()
        self.active_episodes.clear()
        self._gym_states.clear()
        self._episode_rewards.clear()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"GymEnvironmentAdapter(env_id='{self.env_id}', "
            f"max_steps={self.max_turns}, "
            f"processor={self.observation_processor.__class__.__name__}, "
            f"mapper={self.action_mapper.__class__.__name__})"
        )
