"""
Adaptive Learning Controller for GRPO Agent Framework

This module provides intelligent adaptive learning capabilities including:
- Dynamic curriculum learning with difficulty progression
- Self-modifying hyperparameter optimization
- Intelligent exploration vs exploitation strategies
- Meta-learning for fast adaptation to new tasks
"""

import asyncio
import json
import logging
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .advanced_monitoring import get_monitoring_service, monitor_async_function
from .error_handling import ErrorHandler, RetryConfig, retry_async
from .performance_optimizer import PerformanceOptimizer

logger = logging.getLogger(__name__)


class CurriculumStrategy(Enum):
    """Curriculum learning strategies"""

    LINEAR_PROGRESSION = "linear_progression"
    EXPONENTIAL_PROGRESSION = "exponential_progression"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    PERFORMANCE_BASED = "performance_based"
    DIVERSITY_DRIVEN = "diversity_driven"
    META_LEARNING = "meta_learning"


class ExplorationStrategy(Enum):
    """Exploration strategies"""

    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"  # Upper Confidence Bound
    THOMPSON_SAMPLING = "thompson_sampling"
    CURIOSITY_DRIVEN = "curiosity_driven"
    INFORMATION_GAIN = "information_gain"
    NOVELTY_SEARCH = "novelty_search"


@dataclass
class LearningProgress:
    """Tracks learning progress metrics"""

    task_id: str
    difficulty_level: float
    success_rate: float
    average_reward: float
    learning_velocity: float  # Rate of improvement
    confidence_score: float
    exploration_ratio: float
    episodes_completed: int
    total_steps: int
    last_updated: datetime = field(default_factory=datetime.now)

    def update_metrics(self, reward: float, success: bool, steps: int):
        """Update progress metrics"""
        self.episodes_completed += 1
        self.total_steps += steps

        # Update success rate with exponential moving average
        alpha = 0.1
        self.success_rate = (
            alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        )

        # Update average reward
        self.average_reward = alpha * reward + (1 - alpha) * self.average_reward

        self.last_updated = datetime.now()


class CurriculumController:
    """Controls curriculum learning progression"""

    def __init__(
        self,
        strategy: CurriculumStrategy = CurriculumStrategy.PERFORMANCE_BASED,
        initial_difficulty: float = 0.1,
        max_difficulty: float = 1.0,
        progression_threshold: float = 0.8,
        regression_threshold: float = 0.4,
    ):
        self.strategy = strategy
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.progression_threshold = progression_threshold
        self.regression_threshold = regression_threshold

        self.progress_history: Dict[str, LearningProgress] = {}
        self.difficulty_history: deque = deque(maxlen=100)

    async def should_progress_difficulty(self, task_id: str) -> bool:
        """Determine if difficulty should be increased"""
        if task_id not in self.progress_history:
            return False

        progress = self.progress_history[task_id]

        if self.strategy == CurriculumStrategy.PERFORMANCE_BASED:
            return (
                progress.success_rate >= self.progression_threshold
                and progress.episodes_completed >= 10
            )
        elif self.strategy == CurriculumStrategy.ADAPTIVE_THRESHOLD:
            # Adaptive threshold based on learning velocity
            threshold = max(
                0.6, self.progression_threshold - progress.learning_velocity * 0.1
            )
            return progress.success_rate >= threshold
        elif self.strategy == CurriculumStrategy.META_LEARNING:
            # Use meta-learning to predict optimal progression point
            return await self._meta_learning_progression_decision(task_id)
        else:
            return progress.success_rate >= self.progression_threshold

    async def should_regress_difficulty(self, task_id: str) -> bool:
        """Determine if difficulty should be decreased"""
        if task_id not in self.progress_history:
            return False

        progress = self.progress_history[task_id]
        return progress.success_rate < self.regression_threshold

    async def update_difficulty(self, task_id: str) -> float:
        """Update and return new difficulty level"""
        if await self.should_progress_difficulty(task_id):
            self.current_difficulty = min(
                self.max_difficulty,
                self.current_difficulty + self._get_progression_step(),
            )
            logger.info(
                f"Progressed difficulty to {self.current_difficulty:.3f} for task {task_id}"
            )

        elif await self.should_regress_difficulty(task_id):
            self.current_difficulty = max(
                0.1, self.current_difficulty - self._get_regression_step()
            )
            logger.info(
                f"Regressed difficulty to {self.current_difficulty:.3f} for task {task_id}"
            )

        self.difficulty_history.append(self.current_difficulty)
        return self.current_difficulty

    def _get_progression_step(self) -> float:
        """Calculate difficulty progression step size"""
        if self.strategy == CurriculumStrategy.LINEAR_PROGRESSION:
            return 0.1
        elif self.strategy == CurriculumStrategy.EXPONENTIAL_PROGRESSION:
            return self.current_difficulty * 0.2
        else:
            # Adaptive step size based on recent performance
            if len(self.difficulty_history) > 5:
                variance = np.var(list(self.difficulty_history)[-5:])
                return max(0.05, min(0.2, 0.1 / (1 + variance)))
            return 0.1

    def _get_regression_step(self) -> float:
        """Calculate difficulty regression step size"""
        return self._get_progression_step() * 0.5

    async def _meta_learning_progression_decision(self, task_id: str) -> bool:
        """Use meta-learning to decide on progression"""
        # Simplified meta-learning approach
        # In a full implementation, this would use a separate meta-learner model
        progress = self.progress_history[task_id]

        # Consider multiple factors for progression decision
        factors = [
            progress.success_rate >= 0.7,
            progress.learning_velocity > 0.01,
            progress.confidence_score > 0.8,
            progress.episodes_completed >= 20,
        ]

        # Require at least 3 out of 4 factors to be true
        return sum(factors) >= 3


class ExplorationController:
    """Controls exploration vs exploitation balance"""

    def __init__(
        self,
        strategy: ExplorationStrategy = ExplorationStrategy.CURIOSITY_DRIVEN,
        initial_epsilon: float = 0.9,
        min_epsilon: float = 0.05,
        decay_rate: float = 0.995,
    ):
        self.strategy = strategy
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

        self.action_counts: Dict[str, int] = defaultdict(int)
        self.action_rewards: Dict[str, List[float]] = defaultdict(list)
        self.curiosity_scores: Dict[str, float] = defaultdict(float)

    async def should_explore(self, state: Any, available_actions: List[str]) -> bool:
        """Determine whether to explore or exploit"""
        if self.strategy == ExplorationStrategy.EPSILON_GREEDY:
            return random.random() < self.epsilon

        elif self.strategy == ExplorationStrategy.UCB:
            return await self._ucb_exploration_decision(available_actions)

        elif self.strategy == ExplorationStrategy.CURIOSITY_DRIVEN:
            return await self._curiosity_driven_decision(state, available_actions)

        elif self.strategy == ExplorationStrategy.INFORMATION_GAIN:
            return await self._information_gain_decision(state, available_actions)

        else:
            return random.random() < self.epsilon

    async def select_exploration_action(self, available_actions: List[str]) -> str:
        """Select action for exploration"""
        if self.strategy == ExplorationStrategy.UCB:
            return await self._ucb_action_selection(available_actions)
        elif self.strategy == ExplorationStrategy.CURIOSITY_DRIVEN:
            return await self._curiosity_action_selection(available_actions)
        else:
            return random.choice(available_actions)

    async def update_exploration_state(self, action: str, reward: float, state: Any):
        """Update exploration-related state"""
        self.action_counts[action] += 1
        self.action_rewards[action].append(reward)

        # Update curiosity scores based on prediction error
        if self.strategy == ExplorationStrategy.CURIOSITY_DRIVEN:
            await self._update_curiosity_scores(action, reward, state)

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    async def _ucb_exploration_decision(self, available_actions: List[str]) -> bool:
        """UCB-based exploration decision"""
        total_counts = sum(self.action_counts.values())
        if total_counts < len(available_actions):
            return True  # Explore if haven't tried all actions

        # Calculate UCB values for each action
        ucb_values = []
        for action in available_actions:
            if self.action_counts[action] == 0:
                ucb_values.append(float("inf"))
            else:
                mean_reward = np.mean(self.action_rewards[action])
                confidence = math.sqrt(
                    2 * math.log(total_counts) / self.action_counts[action]
                )
                ucb_values.append(mean_reward + confidence)

        # Check if highest UCB action is under-explored
        max_ucb_idx = np.argmax(ucb_values)
        best_action = available_actions[max_ucb_idx]
        return self.action_counts[best_action] < total_counts * 0.1

    async def _curiosity_driven_decision(
        self, state: Any, available_actions: List[str]
    ) -> bool:
        """Curiosity-driven exploration decision"""
        # Calculate state novelty (simplified)
        state_hash = hash(str(state))
        novelty_score = 1.0 / (1.0 + self.curiosity_scores.get(str(state_hash), 0))

        # Explore more in novel states
        exploration_probability = min(0.9, novelty_score * 0.5 + 0.1)
        return random.random() < exploration_probability

    async def _information_gain_decision(
        self, state: Any, available_actions: List[str]
    ) -> bool:
        """Information gain-based exploration decision"""
        # Simplified information gain calculation
        # In practice, this would use model uncertainty or ensemble disagreement
        total_actions = sum(self.action_counts.values())
        if total_actions == 0:
            return True

        # Calculate action entropy
        action_probs = [
            self.action_counts[action] / total_actions for action in available_actions
        ]
        entropy = -sum(p * math.log(p + 1e-8) for p in action_probs if p > 0)

        # Explore more when entropy is high (uniform distribution)
        max_entropy = math.log(len(available_actions))
        exploration_probability = entropy / max_entropy
        return random.random() < exploration_probability

    async def _ucb_action_selection(self, available_actions: List[str]) -> str:
        """Select action using UCB"""
        total_counts = sum(self.action_counts.values())
        if total_counts == 0:
            return random.choice(available_actions)

        ucb_values = []
        for action in available_actions:
            if self.action_counts[action] == 0:
                return action  # Select unvisited action
            else:
                mean_reward = np.mean(self.action_rewards[action])
                confidence = math.sqrt(
                    2 * math.log(total_counts) / self.action_counts[action]
                )
                ucb_values.append(mean_reward + confidence)

        return available_actions[np.argmax(ucb_values)]

    async def _curiosity_action_selection(self, available_actions: List[str]) -> str:
        """Select action based on curiosity"""
        # Select action with highest curiosity potential
        curiosity_scores = [
            self.curiosity_scores.get(action, 1.0) for action in available_actions
        ]
        return available_actions[np.argmax(curiosity_scores)]

    async def _update_curiosity_scores(self, action: str, reward: float, state: Any):
        """Update curiosity scores based on prediction error"""
        # Simplified curiosity update - in practice, use intrinsic motivation models
        if action in self.action_rewards and len(self.action_rewards[action]) > 1:
            expected_reward = np.mean(self.action_rewards[action][:-1])
            prediction_error = abs(reward - expected_reward)
            self.curiosity_scores[action] = (
                0.9 * self.curiosity_scores[action] + 0.1 * prediction_error
            )


class HyperparameterOptimizer:
    """Adaptive hyperparameter optimization"""

    def __init__(self):
        self.parameter_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.performance_history: deque = deque(maxlen=50)
        self.optimization_step = 0

    async def optimize_parameters(
        self, current_params: Dict[str, float], recent_performance: float
    ) -> Dict[str, float]:
        """Optimize hyperparameters based on recent performance"""
        self.performance_history.append(recent_performance)
        self.optimization_step += 1

        if len(self.performance_history) < 10:
            return current_params

        optimized_params = current_params.copy()

        # Simple adaptive optimization using performance trends
        performance_trend = self._calculate_performance_trend()

        for param_name, param_value in current_params.items():
            new_value = await self._optimize_single_parameter(
                param_name, param_value, performance_trend
            )
            optimized_params[param_name] = new_value
            self.parameter_history[param_name].append(new_value)

        return optimized_params

    def _calculate_performance_trend(self) -> float:
        """Calculate recent performance trend"""
        if len(self.performance_history) < 5:
            return 0.0

        recent = list(self.performance_history)[-5:]
        earlier = (
            list(self.performance_history)[-10:-5]
            if len(self.performance_history) >= 10
            else recent
        )

        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)

        return (recent_avg - earlier_avg) / (earlier_avg + 1e-8)

    async def _optimize_single_parameter(
        self, param_name: str, current_value: float, performance_trend: float
    ) -> float:
        """Optimize a single hyperparameter"""
        # Adaptive parameter adjustment based on performance
        if performance_trend > 0.05:  # Performance improving
            # Continue in the same direction or stay stable
            adjustment_factor = 1.0 + min(0.1, performance_trend)
        elif performance_trend < -0.05:  # Performance declining
            # Reverse direction or reduce magnitude
            adjustment_factor = 1.0 / (1.0 + abs(performance_trend))
        else:
            # Performance stable, small random exploration
            adjustment_factor = 1.0 + (random.random() - 0.5) * 0.02

        # Parameter-specific constraints
        new_value = current_value * adjustment_factor

        if param_name == "learning_rate":
            new_value = max(1e-6, min(1e-2, new_value))
        elif param_name == "epsilon":
            new_value = max(0.01, min(0.9, new_value))
        elif param_name == "batch_size":
            new_value = max(1, min(512, int(new_value)))
        elif param_name == "temperature":
            new_value = max(0.1, min(2.0, new_value))

        return new_value


class AdaptiveLearningController:
    """Main adaptive learning controller orchestrating all components"""

    def __init__(
        self,
        curriculum_strategy: CurriculumStrategy = CurriculumStrategy.PERFORMANCE_BASED,
        exploration_strategy: ExplorationStrategy = ExplorationStrategy.CURIOSITY_DRIVEN,
    ):
        self.curriculum_controller = CurriculumController(curriculum_strategy)
        self.exploration_controller = ExplorationController(exploration_strategy)
        self.hyperparameter_optimizer = HyperparameterOptimizer()

        self.error_handler = ErrorHandler()
        self.monitoring = get_monitoring_service()

        self.learning_episodes = 0
        self.total_steps = 0
        self.best_performance = float("-inf")

    @monitor_async_function("adaptive_learning_step")
    async def step(
        self,
        task_id: str,
        state: Any,
        available_actions: List[str],
        reward: float,
        success: bool,
        current_hyperparams: Dict[str, float],
    ) -> Tuple[float, bool, str, Dict[str, float]]:
        """Perform one adaptive learning step"""
        try:
            # Update curriculum
            new_difficulty = await self.curriculum_controller.update_difficulty(task_id)

            # Update progress tracking
            if task_id in self.curriculum_controller.progress_history:
                progress = self.curriculum_controller.progress_history[task_id]
                progress.update_metrics(reward, success, 1)
            else:
                progress = LearningProgress(
                    task_id=task_id,
                    difficulty_level=new_difficulty,
                    success_rate=1.0 if success else 0.0,
                    average_reward=reward,
                    learning_velocity=0.0,
                    confidence_score=0.5,
                    exploration_ratio=self.exploration_controller.epsilon,
                    episodes_completed=1,
                    total_steps=1,
                )
                self.curriculum_controller.progress_history[task_id] = progress

            # Determine exploration vs exploitation
            should_explore = await self.exploration_controller.should_explore(
                state, available_actions
            )

            # Select action
            if should_explore:
                selected_action = (
                    await self.exploration_controller.select_exploration_action(
                        available_actions
                    )
                )
            else:
                # In practice, this would use the policy network's best action
                selected_action = available_actions[0]  # Simplified exploitation

            # Update exploration state
            await self.exploration_controller.update_exploration_state(
                selected_action, reward, state
            )

            # Optimize hyperparameters
            optimized_hyperparams = (
                await self.hyperparameter_optimizer.optimize_parameters(
                    current_hyperparams, reward
                )
            )

            # Update statistics
            self.learning_episodes += 1
            self.total_steps += 1
            if reward > self.best_performance:
                self.best_performance = reward

            # Log metrics
            await self._log_metrics(task_id, new_difficulty, should_explore, reward)

            return (
                new_difficulty,
                should_explore,
                selected_action,
                optimized_hyperparams,
            )

        except Exception as e:
            self.error_handler.handle_error(e, "adaptive_learning", "step")
            # Return safe defaults
            return 0.5, True, available_actions[0], current_hyperparams

    async def _log_metrics(
        self, task_id: str, difficulty: float, explored: bool, reward: float
    ):
        """Log metrics to monitoring system"""
        metrics = {
            "curriculum_difficulty": difficulty,
            "exploration_epsilon": self.exploration_controller.epsilon,
            "explored": 1.0 if explored else 0.0,
            "reward": reward,
            "learning_episodes": self.learning_episodes,
            "best_performance": self.best_performance,
        }

        for metric_name, value in metrics.items():
            await self.monitoring.record_metric(
                f"adaptive_learning.{metric_name}", value, {"task_id": task_id}
            )

    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning process"""
        insights = {
            "curriculum_insights": {
                "current_difficulty": self.curriculum_controller.current_difficulty,
                "difficulty_trend": list(self.curriculum_controller.difficulty_history)[
                    -10:
                ],
                "task_progress": {
                    task_id: {
                        "success_rate": progress.success_rate,
                        "average_reward": progress.average_reward,
                        "episodes": progress.episodes_completed,
                    }
                    for task_id, progress in self.curriculum_controller.progress_history.items()
                },
            },
            "exploration_insights": {
                "current_epsilon": self.exploration_controller.epsilon,
                "action_statistics": {
                    action: {
                        "count": count,
                        "average_reward": np.mean(
                            self.exploration_controller.action_rewards[action]
                        )
                        if self.exploration_controller.action_rewards[action]
                        else 0.0,
                    }
                    for action, count in self.exploration_controller.action_counts.items()
                },
            },
            "hyperparameter_insights": {
                "optimization_steps": self.hyperparameter_optimizer.optimization_step,
                "performance_trend": self.hyperparameter_optimizer._calculate_performance_trend()
                if len(self.hyperparameter_optimizer.performance_history) >= 5
                else 0.0,
            },
            "overall_stats": {
                "learning_episodes": self.learning_episodes,
                "total_steps": self.total_steps,
                "best_performance": self.best_performance,
            },
        }

        return insights


# Factory function for easy instantiation
def create_adaptive_learning_controller(
    curriculum_strategy: str = "performance_based",
    exploration_strategy: str = "curiosity_driven",
) -> AdaptiveLearningController:
    """Create an adaptive learning controller with specified strategies"""
    curr_strategy = CurriculumStrategy(curriculum_strategy)
    expl_strategy = ExplorationStrategy(exploration_strategy)

    return AdaptiveLearningController(curr_strategy, expl_strategy)
