"""
Test matchers for StateSet Agents.

Provides assertion helpers for common test patterns.
"""

from typing import Any, Dict, List, Optional, Union


class RewardMatcher:
    """Matchers for reward-related assertions."""

    @staticmethod
    def is_within_range(
        value: float,
        min_val: float = -10.0,
        max_val: float = 10.0,
    ) -> bool:
        """Check if reward is within expected range."""
        return min_val <= value <= max_val

    @staticmethod
    def is_normalized(
        value: float,
        tolerance: float = 0.01,
    ) -> bool:
        """Check if reward is normalized to [0, 1]."""
        return -tolerance <= value <= 1.0 + tolerance

    @staticmethod
    def has_reasonable_variance(
        rewards: List[float],
        max_variance: float = 10.0,
    ) -> bool:
        """Check if reward variance is reasonable."""
        if len(rewards) < 2:
            return True
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        return variance <= max_variance


class TrajectoryMatcher:
    """Matchers for trajectory-related assertions."""

    @staticmethod
    def has_consistent_turns(trajectory: Any) -> bool:
        """Check if trajectory has consistent turn structure."""
        if not hasattr(trajectory, "turns"):
            return False
        turns = trajectory.turns
        if not turns:
            return True

        for turn in turns:
            if not hasattr(turn, "role") or not hasattr(turn, "content"):
                return False
            if turn.role not in ["user", "assistant", "system"]:
                return False
        return True

    @staticmethod
    def has_matching_rewards(
        trajectory: Any,
        expected_count: Optional[int] = None,
    ) -> bool:
        """Check if rewards match trajectory turns."""
        if not hasattr(trajectory, "turns") or not hasattr(trajectory, "rewards"):
            return False

        if expected_count is not None:
            return len(trajectory.rewards) == expected_count

        return len(trajectory.rewards) == len(trajectory.turns)

    @staticmethod
    def respects_max_turns(trajectory: Any, max_turns: int) -> bool:
        """Check if trajectory respects max turns constraint."""
        if not hasattr(trajectory, "turns"):
            return True
        return len(trajectory.turns) <= max_turns


class ConfigMatcher:
    """Matchers for configuration-related assertions."""

    @staticmethod
    def has_required_keys(config: Dict[str, Any], required: List[str]) -> bool:
        """Check if config has all required keys."""
        return all(key in config for key in required)

    @staticmethod
    def has_valid_types(config: Dict[str, Any], types: Dict[str, type]) -> bool:
        """Check if config values have valid types."""
        for key, expected_type in types.items():
            if key in config and not isinstance(config[key], expected_type):
                return False
        return True

    @staticmethod
    def has_valid_ranges(config: Dict[str, Any], ranges: Dict[str, tuple]) -> bool:
        """Check if config values are within valid ranges."""
        for key, (min_val, max_val) in ranges.items():
            if key in config:
                value = config[key]
                if not (min_val <= value <= max_val):
                    return False
        return True


class ModelMatcher:
    """Matchers for model-related assertions."""

    @staticmethod
    def is_valid_model_name(name: str) -> bool:
        """Check if model name follows expected format."""
        # Basic check - should be improved based on actual requirements
        return isinstance(name, str) and len(name) > 0

    @staticmethod
    def has_valid_generation_config(config: Dict[str, Any]) -> bool:
        """Check if generation config is valid."""
        if "temperature" in config:
            if not 0 <= config["temperature"] <= 2.0:
                return False
        if "top_p" in config:
            if not 0 < config["top_p"] <= 1.0:
                return False
        if "max_new_tokens" in config:
            if config["max_new_tokens"] < 1:
                return False
        return True
