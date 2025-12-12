"""
Curriculum Learning System for Progressive Task Difficulty

This module implements curriculum learning strategies that progressively increase
task difficulty during training, leading to better convergence and performance.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .environment import Environment, EnvironmentState
from .trajectory import MultiTurnTrajectory

logger = logging.getLogger(__name__)


class DifficultyMetric(Enum):
    """Metrics for measuring task difficulty"""

    CONVERSATION_LENGTH = "conversation_length"
    VOCABULARY_COMPLEXITY = "vocabulary_complexity"
    CONTEXT_WINDOW_SIZE = "context_window_size"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    AMBIGUITY_LEVEL = "ambiguity_level"
    CONSTRAINT_COMPLEXITY = "constraint_complexity"
    DOMAIN_SPECIFICITY = "domain_specificity"


class ProgressionStrategy(Enum):
    """Strategy for progressing through curriculum"""

    LINEAR = "linear"  # Fixed progression schedule
    PERFORMANCE_BASED = "performance_based"  # Progress based on agent performance
    ADAPTIVE = "adaptive"  # Dynamically adjust based on learning curves
    MIXED = "mixed"  # Combination of strategies


@dataclass
class CurriculumStage:
    """Represents a stage in the curriculum"""

    stage_id: str
    difficulty_level: float  # 0.0 (easiest) to 1.0 (hardest)
    task_config: Dict[str, Any]
    success_threshold: float = 0.7  # Required success rate to advance
    min_episodes: int = 100  # Minimum episodes before considering advancement
    max_episodes: int = 1000  # Maximum episodes before forced advancement
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CurriculumProgress:
    """Tracks progress through curriculum"""

    current_stage_idx: int = 0
    episodes_in_stage: int = 0
    stage_rewards: List[float] = field(default_factory=list)
    stage_success_rate: float = 0.0
    total_episodes: int = 0
    advancement_history: List[Tuple[int, int, float]] = field(default_factory=list)  # (stage, episode, performance)


class CurriculumScheduler(ABC):
    """Base class for curriculum scheduling strategies"""

    @abstractmethod
    def should_advance(
        self,
        current_stage: CurriculumStage,
        progress: CurriculumProgress,
        recent_trajectories: List[MultiTurnTrajectory],
    ) -> bool:
        """Determine if agent should advance to next stage"""
        pass

    @abstractmethod
    def get_difficulty_adjustment(
        self,
        current_stage: CurriculumStage,
        progress: CurriculumProgress,
    ) -> float:
        """Get fine-grained difficulty adjustment within a stage (-1.0 to 1.0)"""
        pass


class PerformanceBasedScheduler(CurriculumScheduler):
    """Progress based on performance thresholds"""

    def __init__(
        self,
        window_size: int = 50,
        success_threshold: float = 0.7,
        min_episodes: int = 100,
    ):
        self.window_size = window_size
        self.success_threshold = success_threshold
        self.min_episodes = min_episodes

    def should_advance(
        self,
        current_stage: CurriculumStage,
        progress: CurriculumProgress,
        recent_trajectories: List[MultiTurnTrajectory],
    ) -> bool:
        """Advance if performance exceeds threshold"""
        # Must meet minimum episode requirement
        if progress.episodes_in_stage < max(self.min_episodes, current_stage.min_episodes):
            return False

        # Force advancement if max episodes reached
        if progress.episodes_in_stage >= current_stage.max_episodes:
            logger.info(f"Forcing advancement after {progress.episodes_in_stage} episodes")
            return True

        # Check recent performance
        if len(progress.stage_rewards) < self.window_size:
            return False

        recent_rewards = progress.stage_rewards[-self.window_size :]
        avg_reward = np.mean(recent_rewards)
        success_rate = np.mean([r > 0 for r in recent_rewards])

        threshold = max(self.success_threshold, current_stage.success_threshold)

        should_advance = success_rate >= threshold
        if should_advance:
            logger.info(
                f"Advancing from stage {current_stage.stage_id}: "
                f"success_rate={success_rate:.3f} >= {threshold:.3f}, "
                f"avg_reward={avg_reward:.3f}"
            )

        return should_advance

    def get_difficulty_adjustment(
        self,
        current_stage: CurriculumStage,
        progress: CurriculumProgress,
    ) -> float:
        """Adjust difficulty within stage based on performance"""
        if len(progress.stage_rewards) < 10:
            return 0.0

        recent_rewards = progress.stage_rewards[-20:]
        avg_reward = np.mean(recent_rewards)
        reward_std = np.std(recent_rewards)

        # If performance is very stable and high, slightly increase difficulty
        if reward_std < 0.1 and avg_reward > 0.8:
            return 0.2
        # If performance is struggling, slightly decrease difficulty
        elif avg_reward < 0.3:
            return -0.2

        return 0.0


class AdaptiveScheduler(CurriculumScheduler):
    """Adaptive scheduling based on learning curves"""

    def __init__(
        self,
        learning_rate_threshold: float = 0.01,
        performance_threshold: float = 0.65,
        lookback_window: int = 100,
    ):
        self.learning_rate_threshold = learning_rate_threshold
        self.performance_threshold = performance_threshold
        self.lookback_window = lookback_window

    def should_advance(
        self,
        current_stage: CurriculumStage,
        progress: CurriculumProgress,
        recent_trajectories: List[MultiTurnTrajectory],
    ) -> bool:
        """Advance when learning plateaus and performance is adequate"""
        if progress.episodes_in_stage < current_stage.min_episodes:
            return False

        if progress.episodes_in_stage >= current_stage.max_episodes:
            return True

        if len(progress.stage_rewards) < self.lookback_window:
            return False

        rewards = progress.stage_rewards[-self.lookback_window :]

        # Calculate learning rate (slope of recent rewards)
        x = np.arange(len(rewards))
        slope, _ = np.polyfit(x, rewards, 1)

        # Calculate current performance
        recent_performance = np.mean(rewards[-20:])

        # Advance if learning has plateaued and performance is good enough
        learning_plateaued = abs(slope) < self.learning_rate_threshold
        performance_adequate = recent_performance >= self.performance_threshold

        if learning_plateaued and performance_adequate:
            logger.info(
                f"Advancing from stage {current_stage.stage_id}: "
                f"learning_rate={slope:.6f}, performance={recent_performance:.3f}"
            )
            return True

        return False

    def get_difficulty_adjustment(
        self,
        current_stage: CurriculumStage,
        progress: CurriculumProgress,
    ) -> float:
        """Adjust based on learning velocity"""
        if len(progress.stage_rewards) < 20:
            return 0.0

        rewards = progress.stage_rewards[-50:]
        x = np.arange(len(rewards))
        slope, _ = np.polyfit(x, rewards, 1)

        # If learning quickly, can increase difficulty
        if slope > 0.01:
            return 0.3
        # If learning is negative, decrease difficulty
        elif slope < -0.01:
            return -0.3

        return 0.0


class CurriculumLearning:
    """
    Main curriculum learning coordinator.

    Manages progression through stages of increasing difficulty,
    tracks performance, and adapts to agent learning.
    """

    def __init__(
        self,
        stages: List[CurriculumStage],
        scheduler: Optional[CurriculumScheduler] = None,
        environment_factory: Optional[Callable[[Dict[str, Any]], Environment]] = None,
        auto_generate_stages: bool = False,
        num_stages: int = 5,
    ):
        """
        Initialize curriculum learning system.

        Args:
            stages: List of curriculum stages (ordered by difficulty)
            scheduler: Strategy for advancing through curriculum
            environment_factory: Factory function to create environments with config
            auto_generate_stages: If True, automatically generate stages
            num_stages: Number of stages to auto-generate
        """
        if auto_generate_stages and not stages:
            stages = self._generate_default_stages(num_stages)

        if not stages:
            raise ValueError("Must provide stages or set auto_generate_stages=True")

        # Sort stages by difficulty
        self.stages = sorted(stages, key=lambda s: s.difficulty_level)
        self.scheduler = scheduler or PerformanceBasedScheduler()
        self.environment_factory = environment_factory
        self.progress = CurriculumProgress()
        self.recent_trajectories: List[MultiTurnTrajectory] = []
        self.trajectory_window_size = 100

    def _generate_default_stages(self, num_stages: int) -> List[CurriculumStage]:
        """Generate default curriculum stages"""
        stages = []
        for i in range(num_stages):
            difficulty = (i + 1) / num_stages
            stages.append(
                CurriculumStage(
                    stage_id=f"stage_{i}",
                    difficulty_level=difficulty,
                    task_config={
                        "max_turns": int(5 + difficulty * 15),  # 5 to 20 turns
                        "context_complexity": difficulty,
                        "vocabulary_level": difficulty,
                        "reasoning_depth": int(1 + difficulty * 4),  # 1 to 5 steps
                    },
                    success_threshold=0.6 + difficulty * 0.2,  # 0.6 to 0.8
                    min_episodes=max(50, 200 - i * 30),
                    max_episodes=500 + i * 200,
                    description=f"Stage {i}: Difficulty {difficulty:.2f}",
                )
            )
        return stages

    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage"""
        return self.stages[self.progress.current_stage_idx]

    def get_current_config(self) -> Dict[str, Any]:
        """Get current task configuration with difficulty adjustments"""
        stage = self.get_current_stage()
        config = stage.task_config.copy()

        # Apply fine-grained difficulty adjustment
        adjustment = self.scheduler.get_difficulty_adjustment(stage, self.progress)
        config["difficulty_adjustment"] = adjustment

        return config

    def record_episode(self, trajectory: MultiTurnTrajectory) -> None:
        """Record completed episode and update progress"""
        self.progress.episodes_in_stage += 1
        self.progress.total_episodes += 1
        self.progress.stage_rewards.append(trajectory.total_reward)

        # Update success rate
        recent_window = min(50, len(self.progress.stage_rewards))
        recent_rewards = self.progress.stage_rewards[-recent_window:]
        self.progress.stage_success_rate = np.mean([r > 0 for r in recent_rewards])

        # Keep trajectory history for scheduler
        self.recent_trajectories.append(trajectory)
        if len(self.recent_trajectories) > self.trajectory_window_size:
            self.recent_trajectories.pop(0)

        # Check if should advance
        current_stage = self.get_current_stage()
        if self.scheduler.should_advance(current_stage, self.progress, self.recent_trajectories):
            self._advance_stage()

    def _advance_stage(self) -> None:
        """Advance to next curriculum stage"""
        old_stage_idx = self.progress.current_stage_idx

        # Can't advance beyond final stage
        if old_stage_idx >= len(self.stages) - 1:
            logger.info("Already at final curriculum stage")
            return

        # Record advancement
        self.progress.advancement_history.append(
            (old_stage_idx, self.progress.total_episodes, self.progress.stage_success_rate)
        )

        # Advance
        self.progress.current_stage_idx += 1
        self.progress.episodes_in_stage = 0
        self.progress.stage_rewards = []
        self.progress.stage_success_rate = 0.0

        new_stage = self.get_current_stage()
        logger.info(
            f"Advanced to stage {new_stage.stage_id} "
            f"(difficulty={new_stage.difficulty_level:.2f}) "
            f"after {self.progress.total_episodes} total episodes"
        )

    def is_curriculum_complete(self) -> bool:
        """Check if curriculum is complete"""
        return (
            self.progress.current_stage_idx >= len(self.stages) - 1
            and self.progress.episodes_in_stage >= self.get_current_stage().min_episodes
        )

    def get_environment(self) -> Optional[Environment]:
        """Get environment configured for current stage"""
        if self.environment_factory is None:
            return None

        config = self.get_current_config()
        return self.environment_factory(config)

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of curriculum progress"""
        stage = self.get_current_stage()
        return {
            "current_stage": self.progress.current_stage_idx,
            "total_stages": len(self.stages),
            "stage_id": stage.stage_id,
            "difficulty": stage.difficulty_level,
            "episodes_in_stage": self.progress.episodes_in_stage,
            "total_episodes": self.progress.total_episodes,
            "stage_success_rate": self.progress.stage_success_rate,
            "stage_avg_reward": np.mean(self.progress.stage_rewards[-50:])
            if self.progress.stage_rewards
            else 0.0,
            "advancement_count": len(self.progress.advancement_history),
            "curriculum_complete": self.is_curriculum_complete(),
        }

    def reset_stage(self) -> None:
        """Reset progress in current stage (useful for retraining)"""
        self.progress.episodes_in_stage = 0
        self.progress.stage_rewards = []
        self.progress.stage_success_rate = 0.0
        logger.info(f"Reset progress in stage {self.get_current_stage().stage_id}")

    def set_stage(self, stage_idx: int) -> None:
        """Manually set curriculum stage"""
        if stage_idx < 0 or stage_idx >= len(self.stages):
            raise ValueError(f"Invalid stage index: {stage_idx}")

        self.progress.current_stage_idx = stage_idx
        self.reset_stage()
        logger.info(f"Manually set to stage {self.get_current_stage().stage_id}")


class TaskDifficultyController:
    """
    Dynamic difficulty adjustment within a task.

    Complements curriculum learning by providing fine-grained
    difficulty control within each curriculum stage.
    """

    def __init__(
        self,
        base_difficulty: float = 0.5,
        adjustment_rate: float = 0.1,
        min_difficulty: float = 0.0,
        max_difficulty: float = 1.0,
    ):
        self.base_difficulty = base_difficulty
        self.adjustment_rate = adjustment_rate
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.current_difficulty = base_difficulty

    def update(self, success: bool, performance_score: float) -> float:
        """
        Update difficulty based on performance.

        Args:
            success: Whether task was successful
            performance_score: Normalized score (0-1)

        Returns:
            New difficulty level
        """
        if success and performance_score > 0.8:
            # Increase difficulty if doing well
            adjustment = self.adjustment_rate
        elif not success and performance_score < 0.3:
            # Decrease difficulty if struggling
            adjustment = -self.adjustment_rate
        else:
            # Minor adjustments based on score
            adjustment = (performance_score - 0.5) * self.adjustment_rate * 0.5

        self.current_difficulty += adjustment
        self.current_difficulty = np.clip(self.current_difficulty, self.min_difficulty, self.max_difficulty)

        return self.current_difficulty

    def get_difficulty(self) -> float:
        """Get current difficulty level"""
        return self.current_difficulty

    def reset(self, difficulty: Optional[float] = None) -> None:
        """Reset to base or specified difficulty"""
        self.current_difficulty = difficulty if difficulty is not None else self.base_difficulty
