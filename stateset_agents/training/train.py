"""
High-level training interface for GRPO Agent Framework
"""

import asyncio
import logging
import math
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Any, cast

from stateset_agents.core.agent import Agent, MultiTurnAgent
from stateset_agents.core.environment import Environment
from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.reward import RewardFunction
from stateset_agents.core.reward_base import RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn

from .._deprecation import warn_deprecated
from .configuration import resolve_training_config as _build_training_config
from .diagnostics import DiagnosticsMonitor
from .offline_backend import load_offline_trainer
from .trainer import MultiTurnGRPOTrainer

logger = logging.getLogger(__name__)

#: Failures a baseline evaluation episode may legitimately hit (a model,
#: environment or reward outage, bad data, a timeout). A TypeError or
#: AttributeError there is a bug and propagates instead of skewing the
#: profile choice.
TRAIN_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    OSError,
    asyncio.TimeoutError,
)
#: Logged, then re-raised.
DATASET_LOAD_EXCEPTIONS = (ImportError,) + TRAIN_EXCEPTIONS


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
    objective: str | None = None,
    objective_overrides: dict[str, Any] | None = None,
    **kwargs: Any,
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
        objective: Policy objective preset name (e.g. "dapo", "gspo", "cispo");
            see ``stateset_agents.training.objectives.OBJECTIVES``
        objective_overrides: Field overrides for the selected objective
        **kwargs: ``TrainingConfig`` fields (e.g. ``learning_rate=1e-5``),
            merged with ``config_overrides``. Unknown names raise ``TypeError``.

    Returns:
        Trained agent
    """

    valid_modes = {mode.value for mode in TrainingMode}
    if training_mode not in valid_modes:
        raise ConfigurationError(
            f"training_mode must be one of {sorted(valid_modes)}, "
            f"got {training_mode!r}"
        )

    extra_config = dict(kwargs)
    agent_config = getattr(agent, "config", None)
    agent_model = getattr(agent_config, "model_name", None)
    agent_identity: dict[str, str | None] = {}
    if isinstance(agent_model, str):
        agent_identity["model_name"] = agent_model
        revision = getattr(agent_config, "model_revision", None)
        if revision is None or isinstance(revision, str):
            agent_identity["model_revision"] = revision
        for name, value in agent_identity.items():
            if name not in (config_overrides or {}) and name not in extra_config:
                extra_config[name] = value
    config = _build_training_config(
        profile=profile,
        num_episodes=num_episodes,
        config_overrides=config_overrides,
        resume_from_checkpoint=resume_from_checkpoint,
        objective=objective,
        objective_overrides=objective_overrides,
        extra_config=extra_config,
    )
    for name, value in agent_identity.items():
        if getattr(config, name) != value:
            raise ConfigurationError(
                f"Training {name} differs from the supplied agent configuration. "
                "Set the model identity in AgentConfig before calling train()."
            )
    # Fields the caller set explicitly (not filled in by the profile).
    explicit_fields = set(config_overrides or {}) | set(kwargs)
    explicit_fields |= {
        name
        for name, value in (
            ("objective", objective),
            ("objective_overrides", objective_overrides),
            ("resume_from_checkpoint", resume_from_checkpoint),
        )
        if value is not None
    }

    logger.info(f"Starting training with profile: {profile}")

    # Determine training mode
    if training_mode == "auto":
        if dataset is not None or dataset_path:
            training_mode = "offline"
        else:
            training_mode = (
                "multi_turn" if isinstance(agent, MultiTurnAgent) else "single_turn"
            )

    # Set up reward function
    if training_mode == "offline" and config.objective is not None:
        raise ConfigurationError(
            "Offline training uses IQL-weighted logged-action extraction; "
            "objective selects an online policy objective and cannot be applied here"
        )
    environment_reward = getattr(environment, "reward_fn", None)
    if reward_fn is None:
        reward_fn = environment_reward

    # Optional: wrap reward with Bayesian uncertainty weighting
    restore_environment_reward = False
    if uncertainty_weighted and reward_fn is not None:
        wraps_environment_reward = reward_fn is environment_reward
        reward_fn = _wrap_with_uncertainty(reward_fn)
        if wraps_environment_reward and reward_fn is not environment_reward:
            # Score each episode once: the environment applies the same wrapped
            # reward the trainer would, so the trainer reuses its score instead
            # of calling the base reward a second time. Restored on exit.
            environment.reward_fn = reward_fn
            restore_environment_reward = True

    try:
        return await _run_training(
            agent,
            environment,
            config,
            reward_fn,
            dataset,
            dataset_path,
            training_mode,
            save_path,
            callbacks,
            explicit_fields,
        )
    finally:
        if restore_environment_reward:
            environment.reward_fn = environment_reward


async def _run_training(
    agent: Any,
    environment: Any,
    config: Any,
    reward_fn: Any,
    dataset: Any,
    dataset_path: str | None,
    training_mode: str,
    save_path: str | None,
    callbacks: Any,
    explicit_fields: Any,
) -> Agent:
    """Dispatch to the trainer for ``training_mode`` once rewards are wired."""
    # Load dataset for offline/hybrid modes
    if training_mode in ("offline", "hybrid") and dataset is None and dataset_path:
        dataset = _load_dataset(dataset_path)

    # Create trainer based on mode
    if training_mode == "offline":
        return await _train_offline(
            agent,
            config,
            reward_fn,
            dataset,
            save_path,
            callbacks,
            explicit_fields=explicit_fields,
        )
    elif training_mode == "hybrid":
        return await _train_hybrid(
            agent,
            environment,
            config,
            reward_fn,
            dataset,
            save_path,
            callbacks,
            explicit_fields=explicit_fields,
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
    """Wrap a reward function with uncertainty weighting.

    :class:`UncertaintyWeightedReward` discounts a score by the
    ``total_uncertainty`` / ``epistemic_uncertainty`` the base reward reports
    (for example the experimental Bayesian reward model) and leaves scores
    without an estimate unchanged. It needs nothing from the experimental
    layer, so it no longer probes for it (that probe was a hidden
    stable -> experimental import).
    """
    logger.info("Wrapping reward function with uncertainty weighting")
    return UncertaintyWeightedReward(reward_fn)


def _finite_nonnegative_uncertainty(value: Any, name: str) -> float:
    """Accept finite real estimates without coercing strings or booleans."""
    message = f"{name} must be a finite nonnegative uncertainty value"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(message)
    return numeric


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
    ) -> None:
        discount = _finite_nonnegative_uncertainty(
            uncertainty_discount, "uncertainty_discount"
        )
        super().__init__(
            weight=base_reward.weight,
            reward_type=base_reward.reward_type,
            name=f"UncertaintyWeighted({base_reward.name})",
        )
        self.base_reward = base_reward
        self._reward_type_override: RewardType | None = None
        self.uncertainty_discount = discount

    @property
    def reward_type(self) -> RewardType:
        """Use the wrapped reward's current step or episode timing."""
        if self._reward_type_override is not None:
            return self._reward_type_override
        return self.base_reward.reward_type

    @reward_type.setter
    def reward_type(self, value: RewardType) -> None:
        # RewardFunction also initializes this field before base_reward is
        # attached. The constructor clears that initial value to track the base.
        self._reward_type_override = value

    @property
    def fail_on_error(self) -> bool:
        """Keep the wrapped reward's critical failures visible to trainers."""
        return bool(getattr(self.base_reward, "fail_on_error", False))

    @property
    def requires_environment_context(self) -> bool:
        """Forward the wrapped reward's need for private evaluator state."""
        return bool(getattr(self.base_reward, "requires_environment_context", False))

    async def compute_reward(
        self, turns: list[ConversationTurn], context: dict[str, Any] | None = None
    ) -> RewardResult:
        """Discount valid uncertainty while retaining the underlying outcome evidence."""
        result = await self.base_reward.compute_reward(turns, context)
        if not math.isfinite(result.score):
            raise ValueError("Base reward score must be finite")
        # Validate every supplied estimate: NaN must not be silently replaced
        # by a max/min clamp, and negative uncertainty must not amplify scores.
        uncertainty = 0.0
        for evidence in (result.breakdown, result.metadata):
            for key in ("total_uncertainty", "epistemic_uncertainty"):
                if key not in evidence:
                    continue
                value = _finite_nonnegative_uncertainty(evidence[key], key)
                uncertainty = max(uncertainty, value)

        # Scale score by confidence
        confidence = max(0.0, 1.0 - self.uncertainty_discount * uncertainty)
        adjusted = result.score * confidence

        return RewardResult(
            score=adjusted,
            breakdown={
                **result.breakdown,
                "base_score": result.score,
                "uncertainty": uncertainty,
                "confidence": confidence,
            },
            components={
                **result.components,
                "base_score": result.score,
                "uncertainty": uncertainty,
                "confidence": confidence,
            },
            metadata={
                **result.metadata,
                "uncertainty_weighted": True,
                "base_metadata": dict(result.metadata),
                "base_breakdown": dict(result.breakdown),
            },
            explanation=result.explanation,
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


# Fields used by offline policy extraction and its checkpointed sampler.
_OFFLINE_MAPPED_FIELDS = (
    "model_name",
    "model_revision",
    "learning_rate",
    "num_epochs",
    "max_grad_norm",
    "seed",
    "max_prompt_length",
    "max_completion_length",
    "gradient_accumulation_steps",
    "adam_beta1",
    "adam_beta2",
    "weight_decay",
)
_OFFLINE_HONOURED_FIELDS = frozenset(_OFFLINE_MAPPED_FIELDS) | {
    "batch_size",
    "per_device_train_batch_size",
    "resume_from_checkpoint",
}


#: The implicit import below is the one stable-training -> experimental edge
#: (tests/unit/test_layering.py allowlist); it goes in this release.
_IMPLICIT_OFFLINE_REMOVAL = "0.72.0"


def _offline_trainer() -> tuple[type[Any], type[Any]]:
    """``(config_class, trainer_class)`` of the registered offline trainer.

    ``import stateset_agents.experimental.offline_rl`` registers it (see
    :mod:`stateset_agents.training.offline_backend`). Without that import the
    experimental component is loaded implicitly, which is deprecated and will
    be removed in v0.72.0.
    """
    loaded = load_offline_trainer()
    if loaded is None:
        warn_deprecated(
            "train(training_mode='offline'|'hybrid') loading the experimental "
            "offline-RL trainer implicitly",
            removal=_IMPLICIT_OFFLINE_REMOVAL,
            replacement="`import stateset_agents.experimental.offline_rl` first",
            stacklevel=3,
        )
        # Layering allowlist entry (dated): see the warning above. Importing
        # the package registers its trainer with training.offline_backend.
        import stateset_agents.experimental.offline_rl  # noqa: F401

        loaded = load_offline_trainer()
    if loaded is None:  # pragma: no cover - offline_rl always registers
        raise ConfigurationError("no offline trainer is registered")
    return loaded


def _offline_config(
    config: Any, explicit_fields: set[str] | frozenset[str], *, strict: bool
) -> Any:
    """Build the offline trainer's config from a ``TrainingConfig``.

    With ``strict`` (pure offline mode) an explicitly set field the offline
    trainer would ignore raises instead of being silently dropped; in hybrid
    mode such fields still apply to the online phase.
    """
    offline_config_cls, _ = _offline_trainer()

    ignored = sorted(set(explicit_fields) - _OFFLINE_HONOURED_FIELDS)
    if ignored and strict:
        raise ConfigurationError(
            "offline training does not use "
            + ", ".join(repr(name) for name in ignored)
            + "; it honours only "
            + ", ".join(sorted(_OFFLINE_HONOURED_FIELDS))
        )
    if ignored:
        logger.info(
            "Offline pre-training phase ignores %s (applied to the online phase)",
            ", ".join(ignored),
        )
    return offline_config_cls(
        **{name: getattr(config, name) for name in _OFFLINE_MAPPED_FIELDS},
        batch_size=config.batch_size or config.per_device_train_batch_size,
        offline_algorithm="iql",
    )


def _seed_offline(config: Any, explicit_fields: set[str] | frozenset[str]) -> None:
    if "seed" not in explicit_fields:
        return
    import random

    import numpy as np

    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:  # pragma: no cover - offline training needs torch
        return
    torch.manual_seed(seed)


async def _train_offline(
    agent: Agent,
    config: Any,
    reward_fn: Any,
    dataset: Any,
    save_path: str | None,
    callbacks: list[Any] | None,
    *,
    explicit_fields: set[str] | frozenset[str] = frozenset(),
    strict: bool = True,
) -> Agent:
    """Train using offline RL from a dataset of logged conversations."""
    from stateset_agents.data.conversation_dataset import (
        ConversationDataset,
        EmbeddingCache,
    )

    if dataset is None:
        raise ConfigurationError(
            "Offline training requires a dataset. "
            "Pass dataset= or dataset_path= to train()."
        )

    offline_config = _offline_config(config, explicit_fields, strict=strict)
    _seed_offline(config, explicit_fields)
    if getattr(agent, "model", None) is None:
        await agent.initialize()
    if isinstance(dataset, ConversationDataset):
        if dataset.embedding_cache is None:
            dataset.embedding_cache = EmbeddingCache(
                embedding_model=dataset.config.embedding_model
            )
        dimensions = dataset.embedding_cache.embedding_dim
        offline_config.state_dim = dimensions
        offline_config.action_dim = dimensions

    _, offline_trainer_cls = _offline_trainer()
    trainer = offline_trainer_cls(
        config=offline_config,
        model=agent.model,
        tokenizer=getattr(agent, "tokenizer", None),
        reward_fn=reward_fn,
        dataset=dataset,
    )
    trainer.callbacks = [DiagnosticsMonitor(config), *(callbacks or [])]

    logger.info("Starting offline training with IQL on %s samples", len(dataset))

    if config.resume_from_checkpoint:
        from .checkpoint_publication import validate_checkpoint_publication

        validate_checkpoint_publication(
            Path(config.resume_from_checkpoint), required=True
        )
        trainer.load(
            str(Path(config.resume_from_checkpoint) / "training_state.pt"),
            require_policy=True,
        )

    # Main training
    await trainer.train(
        num_epochs=config.num_epochs,
        dataset=dataset,
    )

    if save_path:
        trainer.save_checkpoint(save_path)
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
    *,
    explicit_fields: set[str] | frozenset[str] = frozenset(),
) -> Agent:
    """Hybrid training: offline pre-training then online fine-tuning."""

    # Phase 1: Offline pre-training (if dataset available)
    if dataset is not None:
        logger.info("Phase 1: Offline pre-training from dataset")
        await _train_offline(
            agent,
            config,
            reward_fn,
            dataset,
            None,
            callbacks,
            explicit_fields=explicit_fields,
            strict=False,
        )

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
