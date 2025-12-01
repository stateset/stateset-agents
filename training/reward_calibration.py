"""
Reward Calibration and Normalization System

Ensures reward functions produce consistent, well-scaled scores across different
domains and training stages.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stateset_agents.core.reward import RewardFunction, RewardResult
from stateset_agents.core.trajectory import ConversationTurn

logger = logging.getLogger(__name__)


@dataclass
class RewardStatistics:
    """Statistics for reward calibration"""

    mean: float = 0.0
    std: float = 1.0
    min: float = 0.0
    max: float = 1.0
    count: int = 0
    percentiles: Dict[int, float] = field(default_factory=dict)

    def update(self, values: List[float]):
        """Update statistics with new values"""
        if not values:
            return

        values_array = np.array(values)
        self.mean = float(np.mean(values_array))
        self.std = float(np.std(values_array))
        self.min = float(np.min(values_array))
        self.max = float(np.max(values_array))
        self.count = len(values)

        # Compute percentiles
        for p in [5, 25, 50, 75, 95]:
            self.percentiles[p] = float(np.percentile(values_array, p))


class RewardNormalizer:
    """
    Normalizes rewards to a standard range using running statistics
    """

    def __init__(
        self,
        method: str = "z_score",
        target_mean: float = 0.0,
        target_std: float = 1.0,
        clip_range: Optional[Tuple[float, float]] = None,
        buffer_size: int = 10000,
    ):
        """
        Args:
            method: Normalization method ('z_score', 'min_max', 'percentile')
            target_mean: Target mean for z-score normalization
            target_std: Target std for z-score normalization
            clip_range: Optional range to clip normalized values
            buffer_size: Size of running statistics buffer
        """
        self.method = method
        self.target_mean = target_mean
        self.target_std = target_std
        self.clip_range = clip_range
        self.buffer_size = buffer_size

        # Running statistics
        self.reward_buffer = deque(maxlen=buffer_size)
        self.stats = RewardStatistics()

    def add_reward(self, reward: float):
        """Add a reward value to the buffer"""
        self.reward_buffer.append(reward)

        # Update statistics periodically
        if len(self.reward_buffer) % 100 == 0:
            self._update_statistics()

    def _update_statistics(self):
        """Update statistics from buffer"""
        if not self.reward_buffer:
            return

        values = list(self.reward_buffer)
        self.stats.update(values)

    def normalize(self, reward: float) -> float:
        """Normalize a reward value"""
        if self.stats.count == 0:
            return reward

        if self.method == "z_score":
            # Z-score normalization
            if self.stats.std > 0:
                normalized = (
                    (reward - self.stats.mean) / self.stats.std
                ) * self.target_std + self.target_mean
            else:
                normalized = reward

        elif self.method == "min_max":
            # Min-max normalization to [0, 1]
            if self.stats.max > self.stats.min:
                normalized = (reward - self.stats.min) / (self.stats.max - self.stats.min)
            else:
                normalized = 0.5

        elif self.method == "percentile":
            # Percentile-based normalization (robust to outliers)
            p5 = self.stats.percentiles.get(5, self.stats.min)
            p95 = self.stats.percentiles.get(95, self.stats.max)
            if p95 > p5:
                normalized = (reward - p5) / (p95 - p5)
            else:
                normalized = 0.5

        else:
            normalized = reward

        # Clip if specified
        if self.clip_range:
            normalized = np.clip(normalized, self.clip_range[0], self.clip_range[1])

        return float(normalized)

    def get_statistics(self) -> RewardStatistics:
        """Get current statistics"""
        self._update_statistics()
        return self.stats


class CalibratedRewardFunction(RewardFunction):
    """
    Wrapper that calibrates and normalizes rewards from any reward function
    """

    def __init__(
        self,
        base_reward_fn: RewardFunction,
        normalizer: Optional[RewardNormalizer] = None,
        auto_calibrate: bool = True,
        calibration_interval: int = 100,
    ):
        """
        Args:
            base_reward_fn: Base reward function to calibrate
            normalizer: Reward normalizer (created if None)
            auto_calibrate: Automatically update calibration
            calibration_interval: Update interval for auto-calibration
        """
        super().__init__(
            weight=base_reward_fn.weight,
            reward_type=base_reward_fn.reward_type,
            name=f"Calibrated_{base_reward_fn.name}",
        )

        self.base_reward_fn = base_reward_fn
        self.normalizer = normalizer or RewardNormalizer(
            method="z_score", clip_range=(-3.0, 3.0)
        )
        self.auto_calibrate = auto_calibrate
        self.calibration_interval = calibration_interval

        self.call_count = 0

    async def compute_reward(
        self,
        turns: List[ConversationTurn],
        context: Optional[Dict[str, Any]] = None,
    ) -> RewardResult:
        """Compute calibrated reward"""
        # Get base reward
        base_result = await self.base_reward_fn.compute_reward(turns, context)

        # Add to calibration buffer
        if self.auto_calibrate:
            self.normalizer.add_reward(base_result.score)

        # Normalize reward
        calibrated_score = self.normalizer.normalize(base_result.score)

        self.call_count += 1

        # Update breakdown
        calibrated_breakdown = base_result.breakdown.copy()
        calibrated_breakdown["base_score"] = base_result.score
        calibrated_breakdown["calibrated_score"] = calibrated_score
        calibrated_breakdown["normalization_method"] = self.normalizer.method

        return RewardResult(
            score=calibrated_score,
            breakdown=calibrated_breakdown,
            metadata={
                **base_result.metadata,
                "calibration_stats": {
                    "mean": self.normalizer.stats.mean,
                    "std": self.normalizer.stats.std,
                    "count": self.normalizer.stats.count,
                },
            },
            explanation=base_result.explanation,
        )


class MultiRewardCalibrator:
    """
    Calibrates multiple reward functions to have consistent scales
    """

    def __init__(self, reward_functions: List[RewardFunction]):
        """
        Args:
            reward_functions: List of reward functions to calibrate
        """
        self.reward_functions = reward_functions
        self.normalizers = {
            rf.name: RewardNormalizer(method="z_score") for rf in reward_functions
        }

    async def calibrate(
        self,
        episodes: List[List[ConversationTurn]],
        contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, RewardStatistics]:
        """
        Calibrate reward functions on a set of episodes

        Args:
            episodes: List of episode turn sequences
            contexts: Optional contexts for each episode

        Returns:
            Dictionary mapping reward function names to their statistics
        """
        if contexts is None:
            contexts = [None] * len(episodes)

        logger.info(f"Calibrating {len(self.reward_functions)} reward functions...")

        # Collect rewards for each function
        for rf in self.reward_functions:
            logger.info(f"  Processing {rf.name}...")

            for turns, context in zip(episodes, contexts):
                result = await rf.compute_reward(turns, context)
                self.normalizers[rf.name].add_reward(result.score)

        # Get final statistics
        statistics = {}
        for rf in self.reward_functions:
            normalizer = self.normalizers[rf.name]
            normalizer._update_statistics()
            statistics[rf.name] = normalizer.stats

            logger.info(
                f"  {rf.name}: mean={normalizer.stats.mean:.3f}, "
                f"std={normalizer.stats.std:.3f}, "
                f"range=[{normalizer.stats.min:.3f}, {normalizer.stats.max:.3f}]"
            )

        logger.info("✓ Calibration complete")
        return statistics

    def get_calibrated_functions(self) -> List[CalibratedRewardFunction]:
        """Get calibrated versions of all reward functions"""
        calibrated = []
        for rf in self.reward_functions:
            calibrated.append(
                CalibratedRewardFunction(
                    base_reward_fn=rf,
                    normalizer=self.normalizers[rf.name],
                    auto_calibrate=True,
                )
            )
        return calibrated


class AdaptiveRewardScaler:
    """
    Adaptively scales rewards based on training progress
    """

    def __init__(
        self,
        initial_scale: float = 1.0,
        min_scale: float = 0.1,
        max_scale: float = 10.0,
        adaptation_rate: float = 0.01,
    ):
        """
        Args:
            initial_scale: Starting scale factor
            min_scale: Minimum allowed scale
            max_scale: Maximum allowed scale
            adaptation_rate: Rate of scale adaptation
        """
        self.scale = initial_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.adaptation_rate = adaptation_rate

        # Track reward history for adaptation
        self.reward_history = deque(maxlen=1000)

    def scale_reward(self, reward: float) -> float:
        """Scale a reward value"""
        self.reward_history.append(reward)
        return reward * self.scale

    def adapt_scale(self, target_mean: float = 0.5, target_std: float = 0.2):
        """
        Adapt scale based on reward distribution

        Args:
            target_mean: Target mean for rewards
            target_std: Target standard deviation
        """
        if len(self.reward_history) < 100:
            return

        rewards = np.array(list(self.reward_history))
        current_mean = np.mean(rewards)
        current_std = np.std(rewards)

        # Adjust scale to move towards target distribution
        if current_mean > 0:
            mean_adjustment = target_mean / current_mean
            self.scale *= 1.0 + (mean_adjustment - 1.0) * self.adaptation_rate

        if current_std > 0:
            std_adjustment = target_std / current_std
            self.scale *= 1.0 + (std_adjustment - 1.0) * self.adaptation_rate

        # Clip scale to valid range
        self.scale = np.clip(self.scale, self.min_scale, self.max_scale)

    def get_statistics(self) -> Dict[str, float]:
        """Get current scaling statistics"""
        if not self.reward_history:
            return {}

        rewards = np.array(list(self.reward_history))
        return {
            "scale": self.scale,
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
        }


# Convenience functions


def calibrate_reward_functions(
    reward_functions: List[RewardFunction],
    calibration_episodes: List[List[ConversationTurn]],
    contexts: Optional[List[Dict[str, Any]]] = None,
) -> List[CalibratedRewardFunction]:
    """
    Calibrate a list of reward functions

    Args:
        reward_functions: Reward functions to calibrate
        calibration_episodes: Episodes to use for calibration
        contexts: Optional contexts for episodes

    Returns:
        List of calibrated reward functions
    """
    calibrator = MultiRewardCalibrator(reward_functions)

    # Run calibration (blocking)
    import asyncio

    asyncio.run(calibrator.calibrate(calibration_episodes, contexts))

    # Return calibrated functions
    return calibrator.get_calibrated_functions()
