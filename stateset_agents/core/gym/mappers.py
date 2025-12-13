"""
Action Mappers for Gym/Gymnasium Environments

Converts agent outputs (text, numeric) into gym-compatible actions (integers, vectors).
Provides robust parsing with multiple fallback strategies for error handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging
import re
import random
import numpy as np


logger = logging.getLogger(__name__)


class ActionMapper(ABC):
    """
    Base class for converting agent outputs to gym actions.

    Action mappers parse agent responses (typically text) and extract
    valid gym actions. They handle malformed responses gracefully with
    fallback strategies.
    """

    @abstractmethod
    def parse_action(
        self,
        agent_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Parse agent response to extract gym action.

        Args:
            agent_response: Text response from agent
            context: Optional context dict (may contain action_space, etc.)

        Returns:
            Valid gym action (type depends on action space)

        Raises:
            ValueError: If action cannot be parsed and no fallback available
        """
        pass

    @abstractmethod
    def get_action_space_size(self, gym_env: Any) -> int:
        """
        Get the size/dimension of the action space.

        Args:
            gym_env: The gym environment instance

        Returns:
            Number of discrete actions or dimension of continuous space
        """
        pass


class DiscreteActionMapper(ActionMapper):
    """
    Map agent text output to discrete actions (integers).

    Handles various response formats:
    - Simple integers: "0", "1", "2"
    - Labeled: "Action: 1", "I choose 2"
    - Named actions: "LEFT", "RIGHT" (with action_names mapping)

    Args:
        n_actions: Number of discrete actions
        action_names: Optional list of action names (e.g., ["LEFT", "RIGHT"])
        default_action: Action to use if parsing fails (default: random)
    """

    def __init__(
        self,
        n_actions: int,
        action_names: Optional[List[str]] = None,
        default_action: Optional[int] = None
    ):
        self.n_actions = n_actions
        self.action_names = action_names or []
        self.default_action = default_action

        # Create reverse mapping from names to indices
        self.name_to_action = {}
        if self.action_names:
            for idx, name in enumerate(self.action_names):
                if name:
                    self.name_to_action[name.upper()] = idx

    def parse_action(
        self,
        agent_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Parse agent response to extract discrete action.

        Tries multiple strategies in order:
        1. Direct integer in response
        2. Action name matching
        3. Regex extraction
        4. Default or random action
        """
        if not agent_response or not isinstance(agent_response, str):
            logger.warning(f"Invalid agent response: {agent_response}. Using fallback action.")
            return self._get_fallback_action()

        response = agent_response.strip()

        # Strategy 1: Check if response is a simple integer
        if response.isdigit():
            action = int(response)
            if 0 <= action < self.n_actions:
                return action
            else:
                logger.warning(
                    f"Action {action} out of bounds [0, {self.n_actions}). Using fallback."
                )
                return self._get_fallback_action()

        # Strategy 2: Check for action names (case-insensitive)
        response_upper = response.upper()
        for name, action_idx in self.name_to_action.items():
            if name in response_upper:
                return action_idx

        # Strategy 3: Extract number using regex (handles "Action: 1", "I choose 2", etc.)
        number_match = re.search(r'\b(\d+)\b', response)
        if number_match:
            action = int(number_match.group(1))
            if 0 <= action < self.n_actions:
                return action
            else:
                logger.warning(
                    f"Extracted action {action} out of bounds. Using fallback."
                )
                return self._get_fallback_action()

        # Strategy 4: No valid action found - use fallback
        logger.warning(
            f"Could not parse action from response: '{response[:50]}...'. Using fallback."
        )
        return self._get_fallback_action()

    def _get_fallback_action(self) -> int:
        """Get fallback action when parsing fails."""
        if self.default_action is not None:
            return self.default_action
        else:
            # Random exploration as last resort
            action = random.randint(0, self.n_actions - 1)
            logger.debug(f"Using random fallback action: {action}")
            return action

    def get_action_space_size(self, gym_env: Any) -> int:
        """Get number of discrete actions."""
        return self.n_actions


class ContinuousActionMapper(ActionMapper):
    """
    Map agent text output to continuous actions (vectors).

    Handles formats like:
    - "[0.5, -0.2, 0.1]"
    - "0.5 -0.2 0.1"
    - "Action: [0.5, -0.2]"

    Args:
        action_dim: Dimension of continuous action space
        action_low: Lower bounds for actions (default: -1.0)
        action_high: Upper bounds for actions (default: 1.0)
        default_action: Action to use if parsing fails (default: zeros)
    """

    def __init__(
        self,
        action_dim: int,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
        default_action: Optional[np.ndarray] = None
    ):
        self.action_dim = action_dim

        # Set bounds
        if action_low is None:
            self.action_low = np.full(action_dim, -1.0)
        else:
            self.action_low = np.asarray(action_low)

        if action_high is None:
            self.action_high = np.full(action_dim, 1.0)
        else:
            self.action_high = np.asarray(action_high)

        # Set default action
        if default_action is None:
            self.default_action = np.zeros(action_dim)
        else:
            self.default_action = np.asarray(default_action)

    def parse_action(
        self,
        agent_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Parse agent response to extract continuous action vector.

        Tries multiple parsing strategies and clips to valid range.
        """
        if not agent_response or not isinstance(agent_response, str):
            logger.warning(f"Invalid agent response. Using default action.")
            return self.default_action.copy()

        response = agent_response.strip()

        # Strategy 1: Extract numbers from brackets [...]
        bracket_match = re.search(r'\[([\d\s.,e+-]+)\]', response)
        if bracket_match:
            try:
                numbers_str = bracket_match.group(1)
                numbers = [float(x.strip()) for x in re.split(r'[,\s]+', numbers_str) if x.strip()]
                if len(numbers) == self.action_dim:
                    action = np.array(numbers)
                    return self._clip_action(action)
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse bracketed numbers: {e}")

        # Strategy 2: Extract space/comma-separated numbers
        try:
            numbers = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', response)]
            if len(numbers) == self.action_dim:
                action = np.array(numbers)
                return self._clip_action(action)
            elif len(numbers) > self.action_dim:
                # Take first action_dim numbers
                action = np.array(numbers[:self.action_dim])
                logger.warning(f"Found {len(numbers)} numbers, using first {self.action_dim}.")
                return self._clip_action(action)
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse numbers: {e}")

        # Strategy 3: Use default action
        logger.warning(
            f"Could not parse continuous action from: '{response[:50]}...'. Using default."
        )
        return self.default_action.copy()

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip action to valid range."""
        clipped = np.clip(action, self.action_low, self.action_high)
        if not np.allclose(action, clipped):
            logger.debug(f"Clipped action {action} to bounds.")
        return clipped

    def get_action_space_size(self, gym_env: Any) -> int:
        """Get dimension of continuous action space."""
        return self.action_dim


# Factory function for easy mapper creation
def create_action_mapper(gym_env: Any, **kwargs) -> ActionMapper:
    """
    Factory function to create the appropriate action mapper for a gym environment.

    Args:
        gym_env: Gym environment instance
        **kwargs: Additional arguments passed to mapper constructor

    Returns:
        ActionMapper instance

    Examples:
        >>> env = gym.make("CartPole-v1")
        >>> mapper = create_action_mapper(env)
        >>> mapper = create_action_mapper(env, action_names=["LEFT", "RIGHT"])
    """
    gym = None
    try:
        import gymnasium as gym  # type: ignore[assignment]
    except Exception:
        try:
            import gym  # type: ignore[assignment]
        except Exception:
            gym = None

    action_space = getattr(gym_env, "action_space", None)
    if action_space is None:
        raise ValueError("gym_env.action_space is required to create an action mapper")

    # If gym/gymnasium is installed, prefer the canonical type checks.
    if gym is not None:
        if isinstance(action_space, gym.spaces.Discrete):
            return DiscreteActionMapper(n_actions=action_space.n, **kwargs)

        if isinstance(action_space, gym.spaces.Box):
            action_dim = action_space.shape[0] if len(action_space.shape) > 0 else 1
            return ContinuousActionMapper(
                action_dim=action_dim,
                action_low=action_space.low,
                action_high=action_space.high,
                **kwargs,
            )

        if isinstance(action_space, gym.spaces.MultiDiscrete):
            n_actions = int(np.prod(action_space.nvec))
            logger.warning(
                "MultiDiscrete space with %s dims. Treating as single discrete with %s actions.",
                action_space.nvec,
                n_actions,
            )
            return DiscreteActionMapper(n_actions=n_actions, **kwargs)

        raise ValueError(
            f"Unsupported action space type: {type(action_space)}. "
            "Supported types: Discrete, Box, MultiDiscrete"
        )

    # Fallback when gym isn't installed (unit tests / optional dependency).
    n_actions = getattr(action_space, "n", None)
    if isinstance(n_actions, int) and n_actions > 0:
        return DiscreteActionMapper(n_actions=n_actions, **kwargs)

    nvec = getattr(action_space, "nvec", None)
    if nvec is not None:
        n_actions = int(np.prod(nvec))
        return DiscreteActionMapper(n_actions=n_actions, **kwargs)

    shape = getattr(action_space, "shape", None)
    if shape is not None:
        action_dim = int(shape[0]) if hasattr(shape, "__len__") and len(shape) > 0 else 1
        return ContinuousActionMapper(
            action_dim=action_dim,
            action_low=getattr(action_space, "low", None),
            action_high=getattr(action_space, "high", None),
            **kwargs,
        )

    raise ValueError(
        f"Unsupported action space type: {type(action_space)} and gym is not installed."
    )
