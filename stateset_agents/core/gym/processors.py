"""
Observation Processors for Gym/Gymnasium Environments

Converts gym observations (numeric vectors, images, etc.) into formats
compatible with the stateset-agents framework (text descriptions, structured data).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class ObservationProcessor(ABC):
    """
    Base class for processing gym observations into agent-compatible formats.

    Observation processors convert raw gym observations (vectors, images, etc.)
    into text descriptions or structured formats that agents can process.
    """

    @abstractmethod
    def process(self, observation: Any, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Convert a gym observation to a text description.

        Args:
            observation: Raw observation from gym environment
            context: Optional context dict with additional information

        Returns:
            Text description of the observation
        """
        pass

    @abstractmethod
    def get_system_prompt(self, gym_env: Any) -> str:
        """
        Generate system prompt describing the environment and task.

        Args:
            gym_env: The gym environment instance

        Returns:
            System prompt text for the agent
        """
        pass


class VectorObservationProcessor(ObservationProcessor):
    """
    Process Box observation spaces (continuous numeric vectors).

    Useful for classic control tasks like CartPole, MountainCar, Pendulum.
    Converts numeric state vectors to human-readable text descriptions.

    Args:
        feature_names: Optional list of names for each feature in the vector
        precision: Number of decimal places for formatting (default: 3)
    """

    def __init__(self, feature_names: Optional[list[str]] = None, precision: int = 3):
        self.feature_names = feature_names
        self.precision = precision

    def process(self, observation: Any, context: Optional[Dict[str, Any]] = None) -> str:
        """Convert numeric vector to text description."""
        if not isinstance(observation, (np.ndarray, list, tuple)):
            observation = np.array(observation)

        if isinstance(observation, (list, tuple)):
            observation = np.array(observation)

        # Flatten if needed
        if len(observation.shape) > 1:
            observation = observation.flatten()

        # Format with or without feature names
        if self.feature_names and len(self.feature_names) == len(observation):
            parts = [
                f"{name}: {val:.{self.precision}f}"
                for name, val in zip(self.feature_names, observation)
            ]
            return "Observation: " + ", ".join(parts)
        else:
            # Generic format
            formatted_vals = [f"{val:.{self.precision}f}" for val in observation]
            return f"Observation: [{', '.join(formatted_vals)}]"

    def get_system_prompt(self, gym_env: Any) -> str:
        """Generate generic system prompt for vector observations."""
        return (
            f"You are an RL agent controlling a {gym_env.spec.id if hasattr(gym_env, 'spec') else 'gym'} environment. "
            f"You will receive numeric observations and must output action numbers. "
            f"Your goal is to maximize reward."
        )


class CartPoleObservationProcessor(VectorObservationProcessor):
    """
    Specialized processor for CartPole-v1 environment.

    CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    Provides domain-specific feature names and task description.
    """

    def __init__(self, precision: int = 3):
        super().__init__(
            feature_names=[
                "cart_position",
                "cart_velocity",
                "pole_angle",
                "pole_angular_velocity"
            ],
            precision=precision
        )

    def process(self, observation: Any, context: Optional[Dict[str, Any]] = None) -> str:
        """Convert CartPole observation to detailed text description."""
        if not isinstance(observation, (np.ndarray, list, tuple)):
            observation = np.array(observation)

        if isinstance(observation, (list, tuple)):
            observation = np.array(observation)

        # Flatten if needed
        if len(observation.shape) > 1:
            observation = observation.flatten()

        if len(observation) != 4:
            # Fallback to generic processing
            return super().process(observation, context)

        cart_pos, cart_vel, pole_angle, pole_vel = observation

        # Create rich description
        cart_dir = "right" if cart_vel > 0 else "left" if cart_vel < 0 else "stationary"
        pole_dir = "clockwise" if pole_vel > 0 else "counterclockwise" if pole_vel < 0 else "stable"
        pole_tilt = "right" if pole_angle > 0 else "left" if pole_angle < 0 else "upright"

        description = (
            f"Cart at position {cart_pos:.{self.precision}f}, "
            f"moving {cart_dir} at {abs(cart_vel):.{self.precision}f} m/s. "
            f"Pole tilted {pole_tilt} by {abs(pole_angle):.{self.precision}f} radians, "
            f"rotating {pole_dir} at {abs(pole_vel):.{self.precision}f} rad/s."
        )

        return description

    def get_system_prompt(self, gym_env: Any) -> str:
        """Generate CartPole-specific system prompt."""
        return (
            "You are an RL agent controlling a CartPole environment. "
            "Your goal is to balance a pole on top of a moving cart. "
            "\n\n"
            "You will receive observations describing:\n"
            "- Cart position and velocity\n"
            "- Pole angle and angular velocity\n"
            "\n"
            "You must output ONE action:\n"
            "- 0: Push cart to the LEFT\n"
            "- 1: Push cart to the RIGHT\n"
            "\n"
            "Choose wisely to keep the pole balanced. Respond with ONLY the action number (0 or 1)."
        )


class MountainCarObservationProcessor(VectorObservationProcessor):
    """
    Specialized processor for MountainCar-v0 environment.

    MountainCar state: [position, velocity]
    """

    def __init__(self, precision: int = 3):
        super().__init__(
            feature_names=["position", "velocity"],
            precision=precision
        )

    def process(self, observation: Any, context: Optional[Dict[str, Any]] = None) -> str:
        """Convert MountainCar observation to text."""
        if not isinstance(observation, (np.ndarray, list, tuple)):
            observation = np.array(observation)

        if isinstance(observation, (list, tuple)):
            observation = np.array(observation)

        if len(observation.shape) > 1:
            observation = observation.flatten()

        if len(observation) != 2:
            return super().process(observation, context)

        position, velocity = observation

        # Determine location and movement
        if position < -0.5:
            location = "left valley"
        elif position > 0.3:
            location = "right goal area"
        else:
            location = "middle slope"

        movement = "moving right" if velocity > 0 else "moving left" if velocity < 0 else "stationary"

        description = (
            f"Car at {location} (position: {position:.{self.precision}f}), "
            f"{movement} (velocity: {velocity:.{self.precision}f})."
        )

        return description

    def get_system_prompt(self, gym_env: Any) -> str:
        """Generate MountainCar-specific system prompt."""
        return (
            "You are an RL agent controlling a car in the MountainCar environment. "
            "Your goal is to drive up the right mountain to reach the goal. "
            "\n\n"
            "The car doesn't have enough power to drive straight up. "
            "You need to build momentum by rocking back and forth. "
            "\n\n"
            "Actions:\n"
            "- 0: Push LEFT\n"
            "- 1: Do NOTHING (coast)\n"
            "- 2: Push RIGHT\n"
            "\n"
            "Respond with ONLY the action number (0, 1, or 2)."
        )


# Factory function for easy processor creation
def create_observation_processor(
    env_id: str,
    **kwargs
) -> ObservationProcessor:
    """
    Factory function to create the appropriate observation processor for a gym environment.

    Args:
        env_id: Gym environment ID (e.g., "CartPole-v1", "MountainCar-v0")
        **kwargs: Additional arguments passed to processor constructor

    Returns:
        ObservationProcessor instance

    Examples:
        >>> processor = create_observation_processor("CartPole-v1")
        >>> processor = create_observation_processor("MountainCar-v0", precision=2)
    """
    env_id_lower = env_id.lower()

    if "cartpole" in env_id_lower:
        return CartPoleObservationProcessor(**kwargs)
    elif "mountaincar" in env_id_lower:
        return MountainCarObservationProcessor(**kwargs)
    else:
        # Default to generic vector processor
        return VectorObservationProcessor(**kwargs)
