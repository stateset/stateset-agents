"""
High-level training interface for GRPO Agent Framework
"""

import asyncio
import logging
from enum import Enum
from typing import Any, cast

from stateset_agents.core.agent import Agent, MultiTurnAgent
from stateset_agents.core.environment import Environment
from stateset_agents.core.reward import RewardFunction

from .config import TrainingConfig, TrainingProfile
from .diagnostics import DiagnosticsMonitor
from .trainer import MultiTurnGRPOTrainer

logger = logging.getLogger(__name__)

TRAIN_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    OSError,
    asyncio.TimeoutError,
)
UNCERTAINTY_WRAP_EXCEPTIONS = (ImportError,) + TRAIN_EXCEPTIONS
DATASET_LOAD_EXCEPTIONS = (ImportError, FileNotFoundError) + TRAIN_EXCEPTIONS


class TrainingMode(Enum):
    """Training modes"""

    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    AUTO = "auto"
    OFFLINE = "offline"
    HYBRID = "hybrid"


async def train(
    agent: Agent,
    environment: Environment,
    reward_fn: RewardFunction | None = None,
    num_episodes: int = 1000,
    profile: str = "balanced",
    training_mode: str = "auto",
    config_overrides: dict[str, Any] | None = None,
    save_path: str | None = None,
    resume_from_checkpoint: str | None = None,
    callbacks: list[Any] | None = None,
    dataset: Any | None = None,
    dataset_path: str | None = None,
    uncertainty_weighted: bool = False,
    **kwargs,
) -> Agent:
    """
    High-level training function for agents

    Args:
        agent: Agent to train
        environment: Environment for training
        reward_fn: Reward function (if None, uses environment's reward)
        num_episodes: Number of training episodes
        profile: Training profile ("conservative", "balanced", "aggressive")
        training_mode: Training mode ("single_turn", "multi_turn", "auto",
            "offline", "hybrid")
        config_overrides: Custom configuration overrides
        save_path: Path to save trained agent
        resume_from_checkpoint: Optional checkpoint path to resume training
        dataset: Pre-loaded ConversationDataset for offline/hybrid training
        dataset_path: Path to JSONL dataset (loaded automatically if dataset is None)
        uncertainty_weighted: Wrap reward_fn with Bayesian uncertainty estimates
            to down-weight low-confidence rewards during training
        **kwargs: Additional arguments

    Returns:
        Trained agent
    """

    logger.info(f"Starting training with profile: {profile}")

    # Determine training mode
    if training_mode == "auto":
        if dataset is not None or dataset_path:
            training_mode = "offline"
        else:
            training_mode = (
                "multi_turn" if isinstance(agent, MultiTurnAgent) else "single_turn"
            )

    # Create training configuration
    merged_overrides = dict(config_overrides or {})
    if resume_from_checkpoint is not None:
        merged_overrides["resume_from_checkpoint"] = resume_from_checkpoint

    config = TrainingConfig.from_profile(
        profile=TrainingProfile(profile),
        num_episodes=num_episodes,
        **merged_overrides,
    )

    # Set up reward function
    if reward_fn is None:
        reward_fn = environment.reward_fn

    # Optional: wrap reward with Bayesian uncertainty weighting
    if uncertainty_weighted and reward_fn is not None:
        reward_fn = _wrap_with_uncertainty(reward_fn)

    # Load dataset for offline/hybrid modes
    if training_mode in ("offline", "hybrid") and dataset is None and dataset_path:
        dataset = _load_dataset(dataset_path)

    # Create trainer based on mode
    if training_mode == "offline":
        return await _train_offline(
            agent, config, reward_fn, dataset, save_path, callbacks,
        )
    elif training_mode == "hybrid":
        return await _train_hybrid(
            agent, environment, config, reward_fn, dataset, save_path, callbacks,
        )
    elif training_mode == "multi_turn":
        trainer: Any = MultiTurnGRPOTrainer(
            agent=agent, environment=environment, reward_fn=reward_fn, config=config
        )
    else:
        from stateset_agents.training.trainer import SingleTurnGRPOTrainer

        trainer = SingleTurnGRPOTrainer(
            agent=agent, environment=environment, reward_fn=reward_fn, config=config
        )

    # Add diagnostics
    diagnostics = DiagnosticsMonitor(config)
    trainer.add_callback(diagnostics)

    # Add caller-supplied callbacks
    for cb in callbacks or []:
        trainer.add_callback(cb)

    # Initialize and train
    await trainer.initialize()
    trained_agent = cast(Agent, await trainer.train())

    # Save if requested
    if save_path:
        await trainer.save_checkpoint(checkpoint_name=save_path)
        logger.info(f"Trained agent saved to {save_path}")

    return trained_agent


def _wrap_with_uncertainty(reward_fn: RewardFunction) -> RewardFunction:
    """Wrap a reward function with Bayesian uncertainty weighting.

    When the Bayesian reward model is available, this wraps the base reward
    function to discount rewards that have high uncertainty.  Falls back to
    the original reward when Bayesian models aren't available.
    """
    try:
        import importlib

        importlib.import_module("stateset_agents.rewards.bayesian_reward_model")

        logger.info("Wrapping reward function with Bayesian uncertainty weighting")
        return UncertaintyWeightedReward(reward_fn)
    except UNCERTAINTY_WRAP_EXCEPTIONS as exc:
        logger.warning(
            "Bayesian uncertainty weighting unavailable: %s — using base reward", exc
        )
        return reward_fn


class UncertaintyWeightedReward(RewardFunction):
    """Wraps a reward function to discount high-uncertainty predictions.

    Uses the base reward's score but scales it by confidence:
    adjusted_score = score * (1.0 - uncertainty_discount * uncertainty)

    This prevents the training loss from being dominated by noisy reward
    signals on ambiguous samples.
    """

    def __init__(
        self,
        base_reward: RewardFunction,
        uncertainty_discount: float = 0.5,
    ):
        super().__init__(
            weight=base_reward.weight,
            reward_type=base_reward.reward_type,
            name=f"UncertaintyWeighted({base_reward.name})",
        )
        self.base_reward = base_reward
        self.uncertainty_discount = uncertainty_discount

    async def compute_reward(self, turns, context=None):
        from stateset_agents.core.reward_base import RewardResult

        result = await self.base_reward.compute_reward(turns, context)

        # Extract uncertainty from breakdown/metadata if the base reward
        # provides it (e.g. BayesianRewardFunction, LLMJudgeRewardWithFallback)
        uncertainty = 0.0
        if hasattr(result, "breakdown"):
            uncertainty = result.breakdown.get(
                "total_uncertainty",
                result.breakdown.get("epistemic_uncertainty", 0.0),
            )
        if hasattr(result, "metadata"):
            uncertainty = max(
                uncertainty, result.metadata.get("total_uncertainty", 0.0)
            )

        # Scale score by confidence
        confidence = max(0.0, 1.0 - self.uncertainty_discount * uncertainty)
        adjusted = result.score * confidence

        return RewardResult(
            score=adjusted,
            components={
                "base_score": result.score,
                "uncertainty": uncertainty,
                "confidence": confidence,
            },
            metadata={
                "uncertainty_weighted": True,
                "base_metadata": result.metadata,
            },
        )


def _load_dataset(dataset_path: str) -> Any:
    """Load a conversation dataset from path."""
    try:
        from stateset_agents.data.conversation_dataset import ConversationDataset

        if dataset_path.endswith(".jsonl"):
            return ConversationDataset.from_jsonl(dataset_path)
        elif dataset_path.endswith(".json"):
            return ConversationDataset.from_json(dataset_path)
        else:
            return ConversationDataset.from_jsonl(dataset_path)
    except DATASET_LOAD_EXCEPTIONS as exc:
        logger.error("Failed to load dataset from %s: %s", dataset_path, exc)
        raise


async def _train_offline(
    agent: Agent,
    config: Any,
    reward_fn: Any,
    dataset: Any,
    save_path: str | None,
    callbacks: list[Any] | None,
) -> Agent:
    """Train using offline RL from a dataset of logged conversations."""
    from .offline_grpo_trainer import OfflineGRPOConfig, OfflineGRPOTrainer

    if dataset is None:
        raise ValueError(
            "Offline training requires a dataset. "
            "Pass dataset= or dataset_path= to train()."
        )

    offline_config = OfflineGRPOConfig(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        num_episodes=config.num_episodes,
        offline_algorithm="iql",
    )

    trainer = OfflineGRPOTrainer(
        config=offline_config,
        model=agent.model,
        tokenizer=getattr(agent, "tokenizer", None),
        reward_fn=reward_fn,
        dataset=dataset,
    )

    logger.info("Starting offline training with IQL on %s samples", len(dataset))

    # Pre-train value functions
    pretrain_metrics = await trainer.pretrain_value_functions(dataset, num_steps=100)
    logger.info("Value pre-training: %s", pretrain_metrics)

    # Main training
    await trainer.train(
        num_epochs=config.num_epochs,
        dataset=dataset,
    )

    if save_path:
        logger.info("Saving offline-trained model to %s", save_path)

    return agent


async def _train_hybrid(
    agent: Agent,
    environment: Environment,
    config: Any,
    reward_fn: Any,
    dataset: Any,
    save_path: str | None,
    callbacks: list[Any] | None,
) -> Agent:
    """Hybrid training: offline pre-training then online fine-tuning."""

    # Phase 1: Offline pre-training (if dataset available)
    if dataset is not None:
        logger.info("Phase 1: Offline pre-training from dataset")
        await _train_offline(agent, config, reward_fn, dataset, None, callbacks)

    # Phase 2: Online fine-tuning
    logger.info("Phase 2: Online fine-tuning with environment")
    trainer = MultiTurnGRPOTrainer(
        agent=agent, environment=environment, reward_fn=reward_fn, config=config
    )

    diagnostics = DiagnosticsMonitor(config)
    trainer.add_callback(diagnostics)
    for cb in callbacks or []:
        trainer.add_callback(cb)

    await trainer.initialize()
    trained_agent = cast(Agent, await trainer.train())

    if save_path:
        await trainer.save_checkpoint(checkpoint_name=save_path)
        logger.info("Hybrid-trained agent saved to %s", save_path)

    return trained_agent


class AutoTrainer:
    """
    Automatic trainer that optimizes hyperparameters and handles training
    """

    def __init__(
        self, auto_adjust: bool = True, early_stopping: bool = True, patience: int = 50
    ):
        self.auto_adjust = auto_adjust
        self.early_stopping = early_stopping
        self.patience = patience

    async def train(
        self,
        agent: Agent,
        environment: Environment,
        reward_fn: RewardFunction | None = None,
        **kwargs,
    ) -> Agent:
        """
        Automatically train agent with optimization
        """

        # Analyze task difficulty
        logger.info("Analyzing task characteristics...")
        task_analysis = await self._analyze_task(agent, environment, reward_fn)

        # Select optimal profile based on analysis
        profile = self._select_profile(task_analysis)
        logger.info(f"Selected training profile: {profile}")

        # Configure auto-adjustments
        config_overrides = {
            "auto_adjust": self.auto_adjust,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
        }

        # Train with selected profile
        return await train(
            agent=agent,
            environment=environment,
            reward_fn=reward_fn,
            profile=profile,
            config_overrides=config_overrides,
            **kwargs,
        )

    async def _analyze_task(
        self,
        agent: Agent,
        environment: Environment,
        reward_fn: RewardFunction | None,
    ) -> dict[str, Any]:
        """Analyze task characteristics"""

        logger.info("Running baseline evaluation...")

        # Run a few episodes to assess baseline performance
        baseline_rewards = []
        reward_diversities = []

        for _ in range(5):  # Small sample for analysis
            try:
                # Generate trajectory
                async def agent_wrapper(history, context):
                    return await agent.generate_response(history, context)

                trajectory = await environment.run_episode(agent_wrapper)
                baseline_rewards.append(trajectory.total_reward)

                # Calculate reward diversity if we have turn rewards
                if trajectory.turn_rewards:
                    import numpy as np

                    reward_diversities.append(np.std(trajectory.turn_rewards))

            except TRAIN_EXCEPTIONS as e:
                logger.warning(f"Baseline evaluation episode failed: {e}")
                continue

        if not baseline_rewards:
            logger.warning("No successful baseline episodes, using default analysis")
            return {"performance": 0.0, "diversity": 0.0, "difficulty": "unknown"}

        import numpy as np

        avg_performance = np.mean(baseline_rewards)
        avg_diversity = np.mean(reward_diversities) if reward_diversities else 0.0

        # Determine difficulty
        if avg_performance < 0.1:
            difficulty = "very_hard"
        elif avg_performance < 0.3:
            difficulty = "hard"
        elif avg_performance < 0.7:
            difficulty = "moderate"
        else:
            difficulty = "easy"

        analysis = {
            "performance": avg_performance,
            "diversity": avg_diversity,
            "difficulty": difficulty,
            "num_episodes": len(baseline_rewards),
        }

        logger.info(f"Task analysis: {analysis}")
        return analysis

    def _select_profile(self, task_analysis: dict[str, Any]) -> str:
        """Select optimal training profile based on task analysis"""

        difficulty = task_analysis.get("difficulty", "moderate")
        performance = task_analysis.get("performance", 0.5)
        diversity = task_analysis.get("diversity", 0.1)

        # Profile selection logic
        if difficulty in ["very_hard", "hard"] or performance < 0.2:
            return "conservative"  # Prioritize stability
        elif difficulty == "easy" and performance > 0.8:
            return "aggressive"  # Can afford to be aggressive
        elif diversity < 0.05:
            return "conservative"  # Low diversity needs stability
        else:
            return "balanced"  # Default for moderate cases


from .preset_training import (
    train_customer_service_agent,
    train_task_agent,
    train_tutoring_agent,
)

__all__ = [
    "TrainingMode",
    "train",
    "AutoTrainer",
    "UncertaintyWeightedReward",
    "train_customer_service_agent",
    "train_tutoring_agent",
    "train_task_agent",
]
