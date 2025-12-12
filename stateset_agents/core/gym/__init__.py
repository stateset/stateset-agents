"""
Gym/Gymnasium Integration for StateSet-Agents

This module provides adapters and utilities for using Gym/Gymnasium environments
with the stateset-agents RL framework. It enables training agents on classic RL
tasks like CartPole, MountainCar, Atari games, and MuJoCo robotics.

Main Components:
- GymEnvironmentAdapter: Wraps gym environments for framework compatibility
- ObservationProcessor: Converts gym observations to agent-compatible formats
- ActionMapper: Parses agent outputs to gym actions
- GymAgent: Specialized agent optimized for gym tasks

Example:
    >>> import gymnasium as gym
    >>> from core.gym import GymEnvironmentAdapter, create_gym_agent
    >>>
    >>> # Create gym environment
    >>> env = gym.make("CartPole-v1")
    >>>
    >>> # Wrap it for the framework
    >>> adapter = GymEnvironmentAdapter(env, auto_create_processors=True)
    >>>
    >>> # Create agent
    >>> agent = create_gym_agent(model_name="gpt2")
    >>> await agent.initialize()
    >>>
    >>> # Now use with GRPO trainer as normal!
"""

from .adapter import GymEnvironmentAdapter
from .agents import GymAgent, create_gym_agent
from .mappers import (
    ActionMapper,
    DiscreteActionMapper,
    ContinuousActionMapper,
    create_action_mapper,
)
from .processors import (
    ObservationProcessor,
    VectorObservationProcessor,
    CartPoleObservationProcessor,
    MountainCarObservationProcessor,
    create_observation_processor,
)


__all__ = [
    # Main adapter
    "GymEnvironmentAdapter",
    # Agents
    "GymAgent",
    "create_gym_agent",
    # Observation processors
    "ObservationProcessor",
    "VectorObservationProcessor",
    "CartPoleObservationProcessor",
    "MountainCarObservationProcessor",
    "create_observation_processor",
    # Action mappers
    "ActionMapper",
    "DiscreteActionMapper",
    "ContinuousActionMapper",
    "create_action_mapper",
]
