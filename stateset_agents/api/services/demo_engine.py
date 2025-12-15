"""
Lightweight Demo Engine Service

A lightweight demonstration engine for API testing and development
without requiring heavy compute resources.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class DemoEngineMetrics:
    """Metrics tracked by the demo engine."""
    total_trajectories: int = 0
    total_reward: float = 0.0
    scale_factor: float = 1.0

    @property
    def average_reward(self) -> float:
        """Calculate average reward."""
        if self.total_trajectories == 0:
            return 0.0
        return self.total_reward / self.total_trajectories


class LightweightDemoEngine:
    """Lightweight demo engine to keep the API responsive in constrained environments.

    This engine simulates training operations without requiring GPU or heavy
    compute resources, making it suitable for:
    - API development and testing
    - CI/CD pipelines
    - Demonstration purposes
    - Resource-constrained environments
    """

    def __init__(self, scale_factor: float = 1.0) -> None:
        """Initialize the demo engine.

        Args:
            scale_factor: Initial computation scale factor
        """
        self._metrics = DemoEngineMetrics(scale_factor=scale_factor)
        logger.info("LightweightDemoEngine initialized with scale_factor=%.2f", scale_factor)

    async def train_iteration(
        self,
        prompts: List[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Simulate a training iteration without heavy compute.

        Args:
            prompts: List of training prompts
            **kwargs: Additional training parameters (ignored in demo mode)

        Returns:
            Dict containing simulated training metrics
        """
        trajectories = len(prompts)
        average_reward = 0.5  # Simulated neutral reward

        # Update cumulative metrics
        self._metrics.total_trajectories += trajectories
        self._metrics.total_reward += average_reward * trajectories

        computation_used = trajectories * self._metrics.scale_factor

        return {
            "trajectories_generated": trajectories,
            "average_reward": average_reward,
            "total_computation_used": computation_used,
            "scale_factor": self._metrics.scale_factor,
            "is_demo_mode": True,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return current engine metrics.

        Returns:
            Dict containing engine metrics
        """
        return {
            "total_trajectories": self._metrics.total_trajectories,
            "average_reward": self._metrics.average_reward,
            "scale_factor": self._metrics.scale_factor,
            "is_demo_mode": True,
        }

    def scale_computation(self, scale_factor: float) -> Dict[str, Any]:
        """Adjust the simulated computation scale.

        Args:
            scale_factor: New scale factor (must be positive)

        Returns:
            Dict containing updated scale factor
        """
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")

        old_factor = self._metrics.scale_factor
        self._metrics.scale_factor = scale_factor

        logger.info(
            "DemoEngine scale_factor changed from %.2f to %.2f",
            old_factor,
            scale_factor,
        )

        return {"scale_factor": scale_factor, "previous_scale_factor": old_factor}

    def reset_metrics(self) -> None:
        """Reset all tracked metrics."""
        old_metrics = self.get_metrics()
        self._metrics.total_trajectories = 0
        self._metrics.total_reward = 0.0
        logger.info("DemoEngine metrics reset. Previous: %s", old_metrics)

    def cleanup(self) -> None:
        """Cleanup hook for parity with real engines."""
        logger.info("LightweightDemoEngine cleanup called (no-op)")


def create_demo_engine(scale_factor: float = 1.0) -> LightweightDemoEngine:
    """Factory function to create a demo engine.

    Args:
        scale_factor: Initial computation scale factor

    Returns:
        Configured LightweightDemoEngine instance
    """
    return LightweightDemoEngine(scale_factor=scale_factor)
