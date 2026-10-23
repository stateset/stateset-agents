"""
Multi-turn GRPO trainer implementation.

This module provides the MultiTurnGRPOTrainer class for training
multi-turn conversational agents using Group Relative Policy Optimization.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import math
from collections.abc import Iterator
from typing import Any, TypeAlias

import numpy as np

from stateset_agents.core import trajectory as core_trajectory
from stateset_agents.core.reward_context import RewardUnavailableError

from .callbacks import _maybe_await, notify_checkpoint_saved
from .config import config_value, estimate_grpo_optimizer_steps
from .continual_learning import ContinualLearningManager
from .ddp_population import run_token_local
from .evaluation import EvaluationConfig, evaluate_agent
from .loss_computation import (
    _compute_group_policy_loss,
    compute_enhanced_grpo_loss,
    compute_grpo_loss,
    compute_token_old_logprobs,
)
from .multi_turn_checkpointing import (
    load_multi_turn_checkpoint,
    save_multi_turn_checkpoint,
)
from .multi_turn_evaluation import (
    coerce_reward_result,
    format_trajectory_for_model,
    run_post_training_evaluation,
)
from .multi_turn_scenarios import (
    apply_task_schedule,
    build_eval_scenarios,
    build_training_scenarios,
    expand_scenarios,
    get_environment_scenarios,
    split_scenarios,
)
from .profiling import (
    TrainingProfiler,
    _observed_assistant_tokens,
    profile_learner_step,
)
from .runtime_state import validate_training_config
from .spine.grpo_update import GRPOUpdateMixin
from .spine.loop import SpineLoopMixin
from .spine.rollouts import record_rollout_sync
from .spine.scoring import EpisodeScorer
from .spine.update import NonFiniteGradientError
from .trainer_utils import (
    apply_dropout_setting,
    backward_training_loss,
    get_amp,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    require_torch,
    require_transformers,
    safe_optimizer_step,
)

Trajectory: TypeAlias = core_trajectory.Trajectory
MultiTurnTrajectory: TypeAlias = core_trajectory.MultiTurnTrajectory
TrajectoryGroup: TypeAlias = core_trajectory.TrajectoryGroup

logger = logging.getLogger(__name__)


def _loss_scalars(loss_dict: Any) -> dict[str, float]:
    """Scalar loss terms (finite or not), for diagnosing a bad update."""
    scalars: dict[str, float] = {}
    for key, value in dict(loss_dict).items():
        try:
            if hasattr(value, "detach") and value.numel() == 1:
                scalars[key] = float(value.detach().float().item())
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                scalars[key] = float(value)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
    return scalars


def rollout_sampling_transforms(agent: Any) -> list[str]:
    """Sampling settings that make rollouts differ from the policy's distribution.

    Policy-gradient objectives assume each reply was sampled from the policy
    itself (temperature 1, no top-p/top-k truncation, no repetition penalty).
    Returns a description of every setting on ``agent.config`` that changes
    that distribution; an empty list means rollouts are on-policy.
    """
    config = getattr(agent, "config", None)
    if config is None:
        return []
    found: list[str] = []
    if getattr(config, "do_sample", True) is False:
        found.append("greedy decoding")
        return found
    temperature = getattr(config, "temperature", 1.0)
    if isinstance(temperature, (int, float)) and float(temperature) != 1.0:
        found.append(f"temperature={temperature}")
    top_p = getattr(config, "top_p", 1.0)
    if isinstance(top_p, (int, float)) and float(top_p) < 1.0:
        found.append(f"top_p={top_p}")
    top_k = getattr(config, "top_k", None)
    if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0:
        found.append(f"top_k={top_k}")
    penalty = getattr(config, "repetition_penalty", 1.0)
    if isinstance(penalty, (int, float)) and float(penalty) != 1.0:
        found.append(f"repetition_penalty={penalty}")
    return found


def _warn_on_biased_rollout_sampling(agent: Any) -> None:
    transforms = rollout_sampling_transforms(agent)
    if transforms:
        logger.warning(
            "Rollouts are not sampled from the policy itself (%s), so the "
            "on-policy gradient is biased toward the transformed distribution. "
            "For unbiased RL set temperature=1.0, top_p=1.0, top_k=0, "
            "repetition_penalty=1.0, do_sample=True on the agent.",
            ", ".join(transforms),
        )


class AdapterDisabledReference:
    """Frozen-base reference for a PEFT policy, without a second weight copy.

    Calls run the policy with its adapters disabled, which is exactly the
    pre-training model when adapters start as a no-op (LoRA's zero-initialized
    ``B``). It owns no parameters, so optimizers and freezing loops skip it.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._model.disable_adapter():
            return self._model(*args, **kwargs)

    def eval(self) -> AdapterDisabledReference:
        return self

    def parameters(self) -> Any:
        return iter(())


def build_reference_model(model: Any) -> Any:
    """Return the frozen KL reference for ``model``.

    PEFT policies reuse their base weights through ``disable_adapter``; any
    other model is deep-copied, put in eval mode, and frozen.
    """
    if callable(getattr(model, "disable_adapter", None)):
        return AdapterDisabledReference(model)
    from copy import deepcopy

    reference = deepcopy(model).eval()
    for param in reference.parameters():
        param.requires_grad = False
    return reference


MULTI_TRAINER_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    OSError,
)


class MultiTurnGRPOTrainer(GRPOUpdateMixin, SpineLoopMixin):
    """
    GRPO trainer for multi-turn agents with HuggingFace and W&B integration

    This trainer implements Group Relative Policy Optimization specifically
    designed for multi-turn conversational agents, with full integration
    to HuggingFace ecosystem and Weights & Biases tracking.

    Driving it by hand (``initialize`` then ``generate_trajectories`` and
    ``training_step`` per batch) gets the same learning-rate schedule as
    :meth:`train`; ``initialize`` builds it from ``num_episodes``. Mind two
    ``TrainingConfig`` defaults in such loops: ``gradient_accumulation_steps``
    is 4 (an optimizer step every fourth ``training_step``) and
    ``entropy_coef`` is 0.01 (an entropy bonus is added). Inside
    ``training_step`` the per-token loss is backpropagated one row chunk
    (``generation_batch_size`` rows) at a time; see ``docs/OBJECTIVES.md``.
    """

    def __init__(
        self,
        agent: Any,
        environment: Any,
        reward_fn: Any | None = None,
        config: Any | None = None,
        wandb_logger: Any | None = None,
        callbacks: list | None = None,
    ):
        self.agent = agent
        self.environment = environment
        self.reward_fn = reward_fn
        validate_training_config(config)
        self.config = config
        self.wandb_logger = wandb_logger
        self.callbacks = callbacks or []
        self.profiler = TrainingProfiler()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self._next_episode = 0
        self.best_eval_metric = float("-inf")
        self.steps_without_improvement = 0
        self._grad_accum_step = 0
        self._grpo_accumulation: dict[str, Any] | None = None

        # HuggingFace components
        self.optimizer: Any | None = None
        self.lr_scheduler: Any | None = None
        self.scaler: Any | None = None

        # Training data
        self.train_dataset = None
        self.eval_dataset = None

        # GRPO enhancements
        self._global_reward_mean: float = 0.0
        self._global_reward_count: int = 0
        self._loss_reward_baseline: dict[str, Any] | None = None
        self.reference_model = None
        self.continual_manager: ContinualLearningManager | None = None
        self._current_task_id: str | None = None

        logger.info(f"GRPO Trainer initialized with {type(agent).__name__}")

    async def initialize(self):
        """Initialize trainer components"""
        logger.info("Initializing GRPO trainer...")

        torch = require_torch()
        require_transformers()
        amp = get_amp()

        # Initialize agent if not already done
        if not hasattr(self.agent, "model") or self.agent.model is None:
            await self.agent.initialize()

        apply_dropout_setting(self.config, getattr(self.agent, "model", None))
        _warn_on_biased_rollout_sampling(self.agent)
        self._apply_gradient_checkpointing()
        self._warn_ignored_config_fields()

        # Set seeds for reproducibility if provided
        try:
            if self._seed_everything(torch):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except MULTI_TRAINER_EXCEPTIONS as seed_err:
            logger.warning(f"Failed to fully set deterministic seeds: {seed_err}")

        # Initialize optimizer with HuggingFace best practices
        self._setup_optimizer()

        # Warmup/decay schedule for manual generate_trajectories + training_step
        # loops; train() rebuilds it for its own step budget.
        if self.lr_scheduler is None:
            self._setup_default_scheduler()

        # Loss scaling guards fp16's narrow exponent range only; bf16 shares
        # fp32's range, and a GradScaler there just adds skipped steps.
        if torch.cuda.is_available() and config_value(self.config, "fp16") is True:
            if amp is not None:
                self.scaler = amp.GradScaler("cuda")
            else:  # pragma: no cover - AMP not available
                logger.warning(
                    "Mixed precision requested but torch.amp is unavailable; "
                    "proceeding without GradScaler."
                )

        self._initialize_reference_model()

        # Initialize W&B if configured
        if (
            self.wandb_logger
            and hasattr(self.config, "report_to")
            and self.config.report_to == "wandb"
        ):
            await self._init_wandb()

        self._init_continual_learning()

        logger.info("GRPO trainer initialization complete")

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        torch = require_torch()

        # Get model parameters
        model_params = list(self.agent.model.named_parameters())

        # Apply weight decay to all parameters except biases and layer norms
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model_params if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": config_value(self.config, "weight_decay"),
            },
            {
                "params": [
                    p for n, p in model_params if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=config_value(self.config, "learning_rate"),
            betas=(
                config_value(self.config, "adam_beta1"),
                config_value(self.config, "adam_beta2"),
            ),
            eps=1e-8,
        )

        logger.info(
            f"Optimizer initialized: AdamW with lr={config_value(self.config, 'learning_rate')}"
        )

    def _initialize_reference_model(self) -> None:
        """Create a required frozen reference, propagating allocation failures.

        Stage the copy before assigning it: a failed clone/freeze cannot leave a
        partial reference or silently remove an explicitly requested KL policy.
        """
        if (
            float(config_value(self.config, "beta") or 0.0) > 0.0
            and not config_value(self.config, "use_reference_model")
            and self.reference_model is None
        ):
            logger.warning(
                "beta=%s is set but use_reference_model=False: the KL penalty "
                "is disabled. Set use_reference_model=True to apply it.",
                self.config.beta,
            )
        if (
            config_value(self.config, "use_reference_model")
            and self.reference_model is None
        ):
            from copy import deepcopy

            reference = deepcopy(self.agent.model).eval()
            for parameter in reference.parameters():
                parameter.requires_grad = False
            self.reference_model = reference
            logger.info("Reference model initialized for KL regularization")

    def _planned_optimizer_steps(self) -> int:
        """Optimizer-step budget implied by ``num_episodes`` (one rollout
        batch per episode), accumulation and inner updates."""
        num_episodes = int(config_value(self.config, "num_episodes"))
        return estimate_grpo_optimizer_steps(
            num_episodes,
            gradient_accumulation_steps=self._get_grad_accum_steps(),
            num_gradient_updates=config_value(self.config, "num_gradient_updates"),
        )

    def _setup_default_scheduler(self) -> None:
        """Build the configured schedule for :meth:`_planned_optimizer_steps`.

        Called from :meth:`initialize` so loops that drive
        :meth:`generate_trajectories` and :meth:`training_step` themselves get
        the same warmup and decay as :meth:`train`. A loop with a different
        step budget can call ``_setup_scheduler(n)`` afterwards.
        """
        if self.optimizer is None:
            return
        try:
            steps = self._planned_optimizer_steps()
        except (TypeError, ValueError) as err:
            logger.warning("Could not size the learning-rate schedule: %s", err)
            return
        self._setup_scheduler(steps)

    def _apply_gradient_checkpointing(self) -> bool:
        """Honor ``config.gradient_checkpointing`` on the policy model.

        Non-reentrant activation checkpointing (it composes with frozen base
        weights and ``torch.no_grad`` reference passes). A PEFT policy's
        embeddings are frozen, so their outputs are made to require grad;
        otherwise checkpointed blocks would see no grad-requiring input and
        the adapters would receive no gradient.
        """
        if config_value(self.config, "gradient_checkpointing") is not True:
            return False
        model = getattr(self.agent, "model", None)
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable):
            logger.warning(
                "gradient_checkpointing=True but %s does not support it; "
                "training without activation checkpointing",
                type(model).__name__,
            )
            return False
        try:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:  # older transformers: no kwargs
            enable()
        if getattr(model, "peft_config", None):
            require_grads = getattr(model, "enable_input_require_grads", None)
            if callable(require_grads):
                require_grads()
        logger.info("Gradient checkpointing enabled (non-reentrant)")
        return True

    # Fields of TrainingConfig this trainer cannot act on: the model, its LoRA
    # adapters, sampling and the rollout engine belong to the agent (configure
    # them on AgentConfig), and the rest describe other trainers' loops.
    _IGNORED_CONFIG_FIELDS: tuple[str, ...] = (
        "model_name",
        "model_revision",
        "num_epochs",
        "per_device_train_batch_size",
        "batch_size",
        "use_cpu",
        "use_lora",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "use_vllm",
        "vllm_gpu_memory_utilization",
        "vllm_tensor_parallel_size",
        "vllm_enable_prefix_caching",
        "vllm_max_model_len",
        "vllm_quantization",
        "vllm_enable_chunked_prefill",
        "gamma",
        "gae_lambda",
        "value_clip",
    )

    def _warn_ignored_config_fields(self) -> list[str]:
        """Warn once about non-default config fields this trainer ignores."""
        import dataclasses

        if getattr(self, "_ignored_fields_warned", False):
            return []
        config = self.config
        if config is None or not dataclasses.is_dataclass(config):
            return []
        defaults = {
            f.name: (
                f.default
                if f.default is not dataclasses.MISSING
                else (
                    f.default_factory()
                    if f.default_factory is not dataclasses.MISSING
                    else dataclasses.MISSING
                )
            )
            for f in dataclasses.fields(config)
        }
        changed = [
            name
            for name in self._IGNORED_CONFIG_FIELDS
            if name in defaults and getattr(config, name, None) != defaults[name]
        ]
        self._ignored_fields_warned = True
        if changed:
            logger.warning(
                "MultiTurnGRPOTrainer ignores TrainingConfig field(s) %s: the "
                "policy model, LoRA adapters and sampling come from the agent "
                "(AgentConfig), and the others belong to other trainers.",
                ", ".join(changed),
            )
        return changed

    def _setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler"""
        require_transformers()
        cosine_schedule = get_cosine_schedule_with_warmup
        linear_schedule = get_linear_schedule_with_warmup
        if cosine_schedule is None or linear_schedule is None:
            raise ImportError("transformers scheduler helpers are unavailable")

        num_warmup_steps = int(
            num_training_steps * config_value(self.config, "warmup_ratio")
        )

        lr_scheduler_type = config_value(self.config, "lr_scheduler_type")

        if lr_scheduler_type == "cosine":
            self.lr_scheduler = cosine_schedule(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif lr_scheduler_type == "linear":
            self.lr_scheduler = linear_schedule(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            # Constant learning rate
            self.lr_scheduler = None

        if self.lr_scheduler is not None:
            logger.info(
                f"Scheduler initialized: {lr_scheduler_type} with {num_warmup_steps} warmup steps"
            )

    def _get_grad_accum_steps(self) -> int:
        steps = int(config_value(self.config, "gradient_accumulation_steps"))
        return max(1, steps)

    def _sync_rollout_backend(self) -> None:
        """Keep an attached rollout engine on-policy after an optimizer step.

        ``self.last_rollout_sync`` is True/False for the sync outcome and
        ``None`` when nothing was attempted (``config.rollout_sync`` off or the
        agent has no ``sync_rollout_backend``).
        """
        self.last_rollout_sync = None
        if not bool(config_value(self.config, "rollout_sync")):
            return
        sync = getattr(self.agent, "sync_rollout_backend", None)
        if callable(sync):
            # Applies config.rollout_sync_failure to a failed sync.
            record_rollout_sync(self, bool(sync()), label="GRPO")

    def _apply_optimizer_step(self, torch, *, gradient_scale: float = 1.0) -> bool:
        skip_policy = config_value(self.config, "nonfinite_grad_policy") == "skip"
        # safe_optimizer_step discards gradients before raising, so record which
        # parameters were non-finite first (only when we will report and skip).
        self._nonfinite_params = (
            self._nonfinite_parameter_names() if skip_policy else []
        )
        try:
            applied = safe_optimizer_step(
                self,
                torch,
                max_grad_norm=config_value(self.config, "max_grad_norm"),
                scaler=self.scaler,
                gradient_scale=gradient_scale,
            )
        except NonFiniteGradientError as error:
            if not skip_policy:
                raise
            return self._skip_nonfinite_update(error)
        if applied:
            self._consecutive_nonfinite = 0
        return applied

    @property
    def nonfinite_skips(self) -> int:
        """Updates discarded because their gradients were non-finite."""
        return int(getattr(self, "_nonfinite_skips", 0))

    def _nonfinite_parameter_names(self) -> list[str]:
        model = getattr(self.agent, "model", None)
        bad: list[str] = []
        if model is not None and hasattr(model, "named_parameters"):
            for name, parameter in model.named_parameters():
                grad = parameter.grad
                if grad is not None and not bool(grad.isfinite().all()):
                    bad.append(name)
        return bad

    def _skip_nonfinite_update(self, error: Exception) -> bool:
        """Discard a non-finite update, logging where it came from."""
        bad = list(getattr(self, "_nonfinite_params", []))
        self._nonfinite_skips = self.nonfinite_skips + 1
        self._consecutive_nonfinite = (
            int(getattr(self, "_consecutive_nonfinite", 0)) + 1
        )
        logger.warning(
            "Non-finite gradients at step %s (skip %d, %d consecutive): %d "
            "parameter(s), first %s; loss terms %s",
            getattr(self, "global_step", "?"),
            self._nonfinite_skips,
            self._consecutive_nonfinite,
            len(bad),
            bad[:8],
            getattr(self, "_last_loss_scalars", {}),
        )
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        limit = int(config_value(self.config, "max_consecutive_nonfinite_skips"))
        if self._consecutive_nonfinite > limit:
            raise NonFiniteGradientError(
                f"{self._consecutive_nonfinite} consecutive non-finite updates "
                f"(limit {limit})"
            ) from error
        return False

    def _flush_accumulation(self, torch: Any) -> bool:
        """Close pending accumulated work before a different update regime."""
        steps = self._get_grad_accum_steps()
        pending = self._grad_accum_step % steps
        applied = False
        if pending:
            applied = self._apply_optimizer_step(torch, gradient_scale=steps / pending)
        self._grad_accum_step = 0
        return applied

    async def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        wandb_logger = self.wandb_logger
        if wandb_logger is None:
            return
        try:
            # Prepare configuration for W&B
            config_dict = {
                "framework": "stateset-agents",
                "agent_type": type(self.agent).__name__,
                "environment_type": type(self.environment).__name__,
            }

            if hasattr(self.config, "__dict__"):
                config_dict.update(self.config.__dict__)
            if hasattr(self.agent.config, "__dict__"):
                config_dict.update(self.agent.config.__dict__)

            wandb_logger.init_run(
                config=config_dict,
                name=config_value(self.config, "run_name"),
                tags=config_value(self.config, "wandb_tags"),
                notes=f"Training {type(self.agent).__name__} with GRPO",
            )

            logger.info("W&B tracking initialized")

        except MULTI_TRAINER_EXCEPTIONS as e:
            logger.error(f"Failed to initialize W&B: {e}")

    def _init_continual_learning(self) -> None:
        """Initialize continual learning manager if configured."""
        if self.config is None:
            return

        manager = ContinualLearningManager.from_training_config(self.config)
        if manager.enabled:
            self.continual_manager = manager
            logger.info(
                "Continual learning enabled: strategy=%s, replay_buffer=%s",
                manager.config.strategy,
                manager.config.replay_buffer_size,
            )

    async def generate_trajectories(
        self, scenarios: list[dict[str, Any]], num_generations: int | None = None
    ) -> list[TrajectoryGroup]:
        """Generate trajectory groups for training"""
        num_generations = int(
            config_value(self.config, "num_generations")
            if num_generations is None
            else num_generations
        )
        if num_generations < 1:
            raise ValueError("num_generations must be a positive integer")
        trajectory_groups: list[TrajectoryGroup] = []
        # Prefer generate_turn: it returns the assistant turn with the generated
        # token ids and log-probs, which unlocks the batched per-token GRPO
        # path (docs/OBJECTIVES.md). run_episode accepts either return type.
        generate_turn = getattr(self.agent, "generate_turn", None)
        if generate_turn is not None and not inspect.iscoroutinefunction(generate_turn):
            generate_turn = None  # mock agents expose arbitrary attributes

        async def agent_fn(history, context):
            if generate_turn is not None:
                turn = await generate_turn(history, context)
                if isinstance(turn, core_trajectory.ConversationTurn):
                    return turn
            return await self.agent.generate_response(history, context)

        batched = await self._generate_batched_trajectories(scenarios, num_generations)
        attempted = 0

        for index, scenario in enumerate(scenarios):
            # Generate multiple trajectories for the same scenario
            trajectories: list[MultiTurnTrajectory] = []

            if batched is not None:
                trajectories = batched[index]
                attempted += num_generations
            for _ in range(num_generations if batched is None else 0):
                trajectory = None
                accepted = False
                try:
                    await self._reset_agent()

                    # Generate trajectory
                    with self.profiler.measure("environment_rollout"):
                        trajectory = await self.environment.run_episode(
                            agent_fn, scenario
                        )

                    await self._score_trajectory(trajectory, scenario)
                    trajectories.append(trajectory)
                    accepted = True

                except RewardUnavailableError as e:
                    # Only a reward outage drops the episode; any other error
                    # (a bug, bad data, a None or NaN reward) propagates.
                    if getattr(self.reward_fn, "fail_on_error", False) or getattr(
                        self, "_stateset_native_rollout", False
                    ):
                        raise
                    logger.warning(
                        "Dropping an episode: its reward is unavailable: %s", e
                    )
                    continue
                finally:
                    attempted += 1
                    self._record_rollout(trajectory, accepted=accepted)

            if trajectories:
                group_trajectories: list[Trajectory | MultiTurnTrajectory] = list(
                    trajectories
                )
                group = TrajectoryGroup(
                    scenario_id=scenario.get(
                        "id", f"scenario_{len(trajectory_groups)}"
                    ),
                    trajectories=group_trajectories,
                    scenario_metadata=scenario,
                )
                trajectory_groups.append(group)

        if attempted and not trajectory_groups:
            # Training on nothing would silently stall: every attempted
            # episode lost its reward to an outage.
            raise RewardUnavailableError(
                f"No episode got a reward: all {attempted} attempted episodes "
                "were dropped because their reward was unavailable"
            )
        return trajectory_groups

    def _record_rollout(
        self, trajectory: MultiTurnTrajectory | None, *, accepted: bool
    ) -> None:
        """Record collection acceptance and only explicitly observed token IDs."""
        evidence = None
        tokens = 0
        if trajectory is not None:
            evidence = (trajectory.final_state or {}).get("task_success")
            if type(evidence) is not bool:
                evidence = None
            tokens = _observed_assistant_tokens(trajectory)
        self.profiler.record_rollout(accepted=accepted, success=evidence, tokens=tokens)

    async def _reset_agent(self) -> None:
        reset_fn = getattr(self.agent, "reset", None)
        if callable(reset_fn):
            await _maybe_await(reset_fn())

    async def _score_trajectory(
        self, trajectory: MultiTurnTrajectory, scenario: dict[str, Any]
    ) -> None:
        """Give a finished episode its one reward (spine ``EpisodeScorer``,
        i.e. ``reward_context.score_episode``).

        Training, periodic evaluation, post-training evaluation and the
        distributed trainer share that helper, so the same episode earns the
        same reward on every path. A failed reward raises, and the caller
        drops the episode.
        """
        if not self.reward_fn and (
            trajectory.metadata.get("reward_source") == "default_length_heuristic"
        ):
            self._warn_default_reward()
        with self.profiler.measure("reward_evaluation"):
            await EpisodeScorer(self.reward_fn, self.environment).score(
                trajectory, scenario
            )

    def _warn_default_reward(self) -> None:
        if getattr(self, "_warned_default_reward", False):
            return
        self._warned_default_reward = True
        logger.warning(
            "Training on the environment's built-in length heuristic: neither the "
            "trainer nor %s has a reward_fn, so rewards measure reply length "
            "only. Pass reward_fn=... for a meaningful training signal.",
            type(self.environment).__name__,
        )

    async def _generate_batched_trajectories(
        self, scenarios: list[dict[str, Any]], num_generations: int
    ) -> list[list[MultiTurnTrajectory]] | None:
        """Roll out every scenario x generation in lockstep, or return None.

        Each conversation turn of all still-running episodes is one call to
        ``agent.generate_turns_batch``; trajectories are identical to running
        the episodes one by one. Returns None (use the sequential path) when
        ``config.batched_rollouts`` is off, the agent or environment cannot
        batch, or the batched rollout fails.
        """
        batch_turns = getattr(self.agent, "generate_turns_batch", None)
        run_batched = getattr(self.environment, "run_episodes_batched", None)
        if (
            not config_value(self.config, "batched_rollouts")
            or not inspect.iscoroutinefunction(batch_turns)
            or not inspect.iscoroutinefunction(run_batched)
            or getattr(self.environment, "supports_concurrent_episodes", False)
            is not True
        ):
            return None
        flat = [s for s in scenarios for _ in range(num_generations)]
        strict = bool(
            getattr(self.reward_fn, "fail_on_error", False)
            or getattr(self, "_stateset_native_rollout", False)
        )
        try:
            await self._reset_agent()
            with self.profiler.measure("environment_rollout"):
                episodes: list[MultiTurnTrajectory] = await run_batched(
                    lambda histories, contexts: batch_turns(histories, contexts), flat
                )
        except BaseException as e:
            # Discarded work remains attempted even when the sequential
            # fallback succeeds.
            for _ in flat:
                self._record_rollout(None, accepted=False)
            if strict or not isinstance(e, MULTI_TRAINER_EXCEPTIONS):
                raise
            logger.warning(
                "Batched rollout failed (%s); rolling episodes out one at a time", e
            )
            return None

        # Score each episode on its own: one failed reward drops that episode
        # rather than the whole batch (which used to roll every episode out
        # again). A strict failure still rejects the batch as a whole.
        scored: list[bool] = []
        for index, (trajectory, scenario) in enumerate(
            zip(episodes, flat, strict=True)
        ):
            try:
                await self._score_trajectory(trajectory, scenario)
            except BaseException as e:
                # Only a reward outage drops the episode; any other error (a
                # bug, bad data, a None or NaN reward) rejects the batch.
                if strict or not isinstance(e, RewardUnavailableError):
                    for observed in episodes:
                        self._record_rollout(observed, accepted=False)
                    raise
                logger.warning(
                    "Dropping episode %d: its reward is unavailable: %s", index, e
                )
                scored.append(False)
            else:
                scored.append(True)
        groups: list[list[MultiTurnTrajectory]] = [[] for _ in scenarios]
        for index, (trajectory, accepted) in enumerate(
            zip(episodes, scored, strict=True)
        ):
            self._record_rollout(trajectory, accepted=accepted)
            if accepted:
                groups[index // num_generations].append(trajectory)
        return groups

    async def _coerce_reward_result(
        self, reward_result: Any
    ) -> tuple[float, dict[str, float]]:
        """Normalize reward outputs into (score, breakdown)."""
        normalized_result: tuple[float, dict[str, float]] = await coerce_reward_result(
            reward_result, MULTI_TRAINER_EXCEPTIONS
        )
        return normalized_result

    def compute_grpo_loss(
        self,
        trajectory_groups: list[TrajectoryGroup],
        old_logprobs: list[Any] | None = None,
        *,
        reward_stats: tuple[float, int] | None = None,
        update_stats: bool = True,
    ) -> dict[str, Any]:
        """Compute GRPO loss from trajectory groups.

        ``reward_stats`` pins the ``(mean, count)`` running-reward baseline and
        ``update_stats=False`` keeps this batch out of it, so repeated inner
        updates on one batch count it once and share one baseline.
        """
        baseline = self._reward_baseline_kwargs()
        if reward_stats is not None:
            baseline.update(
                global_reward_mean=reward_stats[0],
                global_reward_count=reward_stats[1],
                global_reward_sum=None,
            )
        if not update_stats:
            baseline["update_global_stats"] = None
        loss_dict: dict[str, Any] = compute_grpo_loss(
            trajectory_groups=trajectory_groups,
            config=self.config,
            agent=self.agent,
            **baseline,
            backward_fn=getattr(self, "_loss_backward_fn", None),
            old_logprobs=old_logprobs,
        )
        return loss_dict

    def _reward_baseline_kwargs(self) -> dict[str, Any]:
        """Return the rollout's frozen baseline or the current running state."""
        frozen = getattr(self, "_loss_reward_baseline", None)
        if frozen is not None:
            return dict(frozen)
        return {
            "global_reward_mean": self._global_reward_mean,
            "global_reward_count": self._global_reward_count,
            "global_reward_sum": getattr(self, "_global_reward_sum", None),
            "update_global_stats": self._update_global_stats,
        }

    @contextlib.contextmanager
    def _frozen_reward_baseline(
        self, snapshot: dict[str, Any], *, update: bool
    ) -> Iterator[None]:
        """Reuse one rollout baseline while observing its rewards only once."""
        previous = getattr(self, "_loss_reward_baseline", None)
        self._loss_reward_baseline = {
            **snapshot,
            "update_global_stats": self._update_global_stats if update else None,
        }
        try:
            yield
        finally:
            self._loss_reward_baseline = previous

    def _compute_group_policy_loss(
        self, group: TrajectoryGroup, advantages: Any
    ) -> Any:
        """Compute per-group policy loss (compatibility shim for tests)."""
        loss, _entropy = _compute_group_policy_loss(
            group=group,
            advantages=advantages,
            config=self.config,
            agent=self.agent,
        )
        return loss

    def compute_enhanced_grpo_loss(
        self,
        trajectory_groups: list[TrajectoryGroup],
        beta: float = 0.0,
        old_logprobs: list[Any] | None = None,
        *,
        reward_stats: tuple[float, int] | None = None,
        update_stats: bool = True,
    ) -> dict[str, Any]:
        """Enhanced GRPO loss computation with KL penalty (see
        :meth:`compute_grpo_loss` for ``reward_stats``/``update_stats``)."""
        baseline = self._reward_baseline_kwargs()
        if reward_stats is not None:
            baseline.update(
                global_reward_mean=reward_stats[0],
                global_reward_count=reward_stats[1],
                global_reward_sum=None,
            )
        if not update_stats:
            baseline["update_global_stats"] = None
        loss_dict: dict[str, Any] = compute_enhanced_grpo_loss(
            trajectory_groups=trajectory_groups,
            beta=beta,
            config=self.config,
            agent=self.agent,
            reference_model=self.reference_model,
            old_logprobs=old_logprobs,
            **baseline,
            backward_fn=getattr(self, "_loss_backward_fn", None),
        )
        return loss_dict

    def _resolve_task_id(self, scenario: Any) -> str | None:
        """Extract a task id from scenario metadata when available."""
        if not isinstance(scenario, dict) or self.config is None:
            return None
        task_key = config_value(self.config, "task_id_key")
        if task_key and task_key in scenario:
            value = scenario.get(task_key)
            return str(value) if value is not None else None
        return None

    def _maybe_handle_task_switch(self, task_id: str | None) -> None:
        """Handle task boundary transitions for continual learning."""
        if (
            self.continual_manager is None
            or task_id is None
            or not getattr(self.continual_manager, "enabled", True)
        ):
            return

        if self._current_task_id is None:
            self._current_task_id = task_id
            return

        if task_id != self._current_task_id:
            self._finish_current_task()
            self._current_task_id = task_id

    def _task_boundary_receipt(self) -> dict[str, Any]:
        """Identify completed memory at this policy update and replay population."""
        return {
            "task_id": self._current_task_id,
            "global_step": self.global_step,
            "buffer_sequence": getattr(
                getattr(self.continual_manager, "buffer", None), "_sequence", 0
            ),
        }

    def _finish_current_task(self) -> None:
        """Commit prior pending updates before capturing task memory once."""
        manager = self.continual_manager
        if self._grad_accum_step % self._get_grad_accum_steps():
            self._flush_accumulation(require_torch())
        self._grad_accum_step = 0
        if (
            manager is None
            or self._current_task_id is None
            or not getattr(manager, "enabled", True)
        ):
            return
        receipt = self._task_boundary_receipt()
        if (
            isinstance(manager, ContinualLearningManager)
            and manager._task_boundary == receipt
        ):
            return
        manager.on_task_end(agent=self.agent, task_id=self._current_task_id)
        reference_model = getattr(manager, "reference_model", None)
        if reference_model is not None:
            self.reference_model = reference_model
        if isinstance(manager, ContinualLearningManager):
            manager._task_boundary = receipt

    def _maybe_mix_replay(
        self, new_groups: list[TrajectoryGroup], task_id: str | None
    ) -> tuple[list[TrajectoryGroup], list[TrajectoryGroup]]:
        """Optionally sample replay groups and return combined list."""
        if self.continual_manager is None:
            return new_groups, []

        replay_groups = self.continual_manager.sample_replay_groups(len(new_groups))
        self.continual_manager.add_trajectory_groups(new_groups, task_id=task_id)

        if replay_groups:
            return new_groups + replay_groups, replay_groups
        return new_groups, []

    def _update_global_stats(self, batch_mean: float, batch_size: int):
        """Update global reward statistics.

        Uses Welford-style sum/count tracking to avoid floating-point drift
        that accumulates when using incremental mean over many batches.
        """
        if not hasattr(self, "_global_reward_sum"):
            self._global_reward_sum = (
                self._global_reward_mean * self._global_reward_count
            )

        self._global_reward_sum += batch_mean * batch_size
        self._global_reward_count += batch_size
        self._global_reward_mean = (
            self._global_reward_sum / self._global_reward_count
            if self._global_reward_count > 0
            else 0.0
        )

    def _get_environment_scenarios(self) -> list[dict[str, Any]]:
        scenarios: list[dict[str, Any]] = get_environment_scenarios(
            self.environment, self.config
        )
        return scenarios

    def _split_scenarios(
        self, scenarios: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        split_result: tuple[list[dict[str, Any]], list[dict[str, Any]]] = (
            split_scenarios(scenarios, self.config)
        )
        return split_result

    def _apply_task_schedule(self, scenario: dict[str, Any], index: int) -> None:
        apply_task_schedule(self.config, scenario, index)

    def _expand_scenarios(
        self,
        scenarios: list[dict[str, Any]],
        count: int,
        prefix: str,
    ) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = expand_scenarios(
            self.config, scenarios, count, prefix
        )
        return expanded

    def _format_trajectory_for_model(self, trajectory: MultiTurnTrajectory) -> str:
        """Format trajectory into text for model input."""
        formatted: str = format_trajectory_for_model(self.agent, trajectory)
        return formatted

    @profile_learner_step
    async def training_step(
        self, trajectory_groups: list[TrajectoryGroup]
    ) -> dict[str, Any]:
        """Execute a single training step (the spine's ``GRPOUpdateMixin``)."""
        self.agent.model.train()
        _loss_dict, metrics = self._grpo_update(trajectory_groups)
        return metrics

    # --- GRPOUpdateMixin hooks ------------------------------------------------

    def _update_modules(self) -> tuple[Any, Any]:
        return require_torch(), get_amp()

    def _update_old_logprobs(
        self, trajectory_groups: list[TrajectoryGroup]
    ) -> list[Any] | None:
        old: list[Any] | None = run_token_local(
            self.agent.model,
            "old-policy",
            lambda: compute_token_old_logprobs(
                trajectory_groups, self.config, self.agent
            ),
        )
        return old

    def _update_baseline(self) -> dict[str, Any]:
        return self._reward_baseline_kwargs()

    def _update_loss(
        self,
        trajectory_groups: list[Any],
        *,
        enhanced: bool,
        beta: float,
        old_logprobs: list[Any] | None,
        baseline: dict[str, Any] | None,
        update_stats: bool,
        backward_fn: Any,
    ) -> dict[str, Any]:
        frozen = (
            self._frozen_reward_baseline(baseline, update=update_stats)
            if baseline is not None
            else contextlib.nullcontext()
        )
        with frozen, self._loss_backward(backward_fn):
            # Call shapes as before the extraction (spies see the same calls).
            if old_logprobs is None:
                if enhanced:
                    return self.compute_enhanced_grpo_loss(trajectory_groups, beta=beta)
                return self.compute_grpo_loss(trajectory_groups)
            if enhanced:
                return self.compute_enhanced_grpo_loss(
                    trajectory_groups, beta=beta, old_logprobs=old_logprobs
                )
            return self.compute_grpo_loss(trajectory_groups, old_logprobs=old_logprobs)

    def _update_regularizer(self, loss_dict: dict[str, Any]) -> tuple[Any, Any]:
        return self._regularized_loss(loss_dict)

    def _update_commit(self, torch: Any) -> bool:
        return bool(self._apply_optimizer_step(torch))

    def _update_flush(self, torch: Any) -> bool:
        return self._flush_accumulation(torch)

    def _regularized_loss(self, loss_dict: dict[str, Any]) -> tuple[Any, Any]:
        """Coordinate local continual penalties before entering backward phases."""

        def local() -> tuple[Any, Any]:
            loss = loss_dict["total_loss"]
            self._last_loss_scalars = _loss_scalars(loss_dict)
            penalty = None
            if self.continual_manager is not None:
                penalty = self.continual_manager.compute_ewc_penalty(self.agent)
            if penalty is not None:
                loss = loss + penalty
                loss_dict["ewc_penalty"] = penalty
            return loss, penalty

        return run_token_local(self.agent.model, "regularizer", local)

    @contextlib.contextmanager
    def _loss_backward(self, backward_fn: Any) -> Iterator[Any]:
        """Install the backward closure ``compute_*_grpo_loss`` stream into."""
        previous = getattr(self, "_loss_backward_fn", None)
        self._loss_backward_fn: Any = backward_fn
        try:
            yield backward_fn
        finally:
            self._loss_backward_fn = previous

    @contextlib.contextmanager
    def _chunked_backward(
        self, torch: Any, amp: Any, device_type: str, divisor: int
    ) -> Any:
        """Stream chunks while preserving population and entropy gradient banks."""
        from .grpo_accumulation import NativeChunkedBackward

        with self._loss_backward(
            NativeChunkedBackward(self, torch, steps=divisor, scaler=self.scaler)
        ) as backward_fn:
            yield backward_fn

    def _backward_remaining(
        self,
        loss_dict: dict[str, Any],
        loss: Any,
        ewc_penalty: Any | None,
        torch: Any,
        *,
        divisor: int,
    ) -> None:
        """Backpropagate what the loss computation has not already."""
        if loss_dict.get("backward_done"):
            # The policy loss was backpropagated chunk by chunk; only the
            # continual-learning penalty (if any) is left.
            if ewc_penalty is not None:
                backward_training_loss(
                    self, ewc_penalty / divisor, torch, scaler=self.scaler
                )
            return
        backward_training_loss(self, loss / divisor, torch, scaler=self.scaler)

    def _warn_inner_updates_unavailable(self) -> None:
        if getattr(self, "_inner_updates_warned", False):
            return
        self._inner_updates_warned = True
        logger.warning(
            "num_gradient_updates > 1 needs per-token rollouts (agent.generate_turn); "
            "these trajectories carry no token metadata, so a single on-policy "
            "update is applied."
        )

    async def _evaluate_local(
        self, eval_scenarios: list[dict[str, Any]], num_eval_episodes: int
    ) -> dict[str, float]:
        """Evaluate local episodes without advancing training RNGs or modes."""
        from stateset_agents.evaluation.isolation import isolated_evaluation

        if type(num_eval_episodes) is not int or num_eval_episodes < 0:
            raise ValueError("num_eval_episodes must be a nonnegative integer")
        concurrency = int(config_value(self.config, "rollout_concurrency") or 1)
        num_generations = int(config_value(self.config, "eval_num_generations") or 4)
        seed = config_value(self.config, "seed")
        if seed is None:
            seed = 42
        with isolated_evaluation(seed=seed, model=self.agent.model):
            metrics = await evaluate_agent(
                agent=self.agent,
                environment=self.environment,
                scenarios=eval_scenarios[:num_eval_episodes],
                reward_fn=self.reward_fn,
                config=EvaluationConfig(
                    num_episodes=min(num_eval_episodes, len(eval_scenarios)),
                    num_generations=num_generations,
                    max_turns=config_value(self.config, "max_steps_per_episode"),
                    seed=seed,
                    concurrency=concurrency,
                ),
            )
        return metrics

    async def _update_best_checkpoint(self, metrics: dict[str, float]) -> None:
        """Select only complete evaluations and roll back a failed best save."""
        if (
            metrics.get("eval_complete") != 1
            or metrics.get("eval_num_episodes", 0) <= 0
        ):
            return
        current_metric = metrics["eval_reward"]
        if not math.isfinite(current_metric):
            raise ValueError("Best-checkpoint selection requires a finite reward")
        if current_metric > self.best_eval_metric:
            previous = self.best_eval_metric, self.steps_without_improvement
            self.best_eval_metric, self.steps_without_improvement = current_metric, 0
            try:
                await self.save_checkpoint(is_best=True)
            except BaseException:
                self.best_eval_metric, self.steps_without_improvement = previous
                raise
        else:
            self.steps_without_improvement += 1

    async def evaluate(
        self, eval_scenarios: list[dict[str, Any]], num_eval_episodes: int = 10
    ) -> dict[str, float]:
        """Evaluate without disturbing training and select a completed best save."""
        metrics = await self._evaluate_local(eval_scenarios, num_eval_episodes)
        await self._update_best_checkpoint(metrics)
        return metrics

    async def run_post_training_evaluation(
        self,
        eval_scenarios: list[dict[str, Any]],
        num_samples: int = 5,
        detailed: bool = True,
    ) -> dict[str, Any]:
        """Run comprehensive post-training evaluation."""
        evaluation: dict[str, Any] = await run_post_training_evaluation(
            self,
            eval_scenarios,
            num_samples,
            detailed,
            MULTI_TRAINER_EXCEPTIONS,
            self._coerce_reward_result,
        )
        return evaluation

    def _data_parallel_rank_world(self) -> tuple[int, int]:
        """This process's data-parallel ``(rank, world_size)``; ``(0, 1)``
        outside an initialized ``torch.distributed`` process group."""
        try:
            import torch.distributed as dist
        except ImportError:  # pragma: no cover - torch is required to train
            return 0, 1
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), int(dist.get_world_size())
        return 0, 1

    def _seed_everything(self, torch: Any, step: int = 0) -> bool:
        """Seed Python, NumPy and torch from ``config.seed``.

        Each data-parallel rank gets its own stream (``seed + rank``) so ranks
        sample different rollouts; a single process keeps ``config.seed``.
        ``step`` moves to a fresh stream per optimizer step when a resumed
        distributed run cannot restore every rank's saved generator state.
        """
        import random

        seed_value = config_value(self.config, "seed")
        if seed_value is None:
            return False
        rank, world = self._data_parallel_rank_world()
        value = int(seed_value) + rank + world * int(step)
        random.seed(value)
        np.random.seed(value % (2**32))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
        torch.manual_seed(value)
        return True

    def _restore_rng_on_resume(self) -> bool:
        """Checkpoints carry the main process's generator state, which only
        continues the right stream when there is a single process."""
        return self._data_parallel_rank_world()[1] == 1

    def _episode_scenario(
        self, training_scenarios: list[dict[str, Any]], episode: int
    ) -> dict[str, Any]:
        """The scenario this rank trains on in ``episode``.

        Ranks stride through the scenario list so data parallelism adds
        distinct samples instead of repeating rank 0's.
        """
        rank, world = self._data_parallel_rank_world()
        if world == 1:
            return training_scenarios[episode]
        return training_scenarios[(episode * world + rank) % len(training_scenarios)]

    def _get_training_scenarios(self) -> list[dict[str, Any]]:
        """Get training scenarios."""
        scenarios: list[dict[str, Any]] = build_training_scenarios(
            self.environment, self.config
        )
        return scenarios

    def _get_eval_scenarios(self) -> list[dict[str, Any]]:
        """Get evaluation scenarios."""
        scenarios: list[dict[str, Any]] = build_eval_scenarios(
            self.environment, self.config
        )
        return scenarios

    async def _log_training_metrics(
        self, metrics: dict[str, Any], trajectory_groups: list[TrajectoryGroup]
    ):
        """Log training metrics"""
        if self.wandb_logger:
            self.wandb_logger.log_training_step(
                losses=self._training_log_scalars(metrics),
                learning_rate=metrics["learning_rate"],
                step=self.global_step,
                trajectory_groups=trajectory_groups,
            )

    async def _log_eval_metrics(self, eval_metrics: dict[str, float]):
        """Log evaluation metrics"""
        if self.wandb_logger:
            self.wandb_logger.log_evaluation(
                eval_metrics=eval_metrics, step=self.global_step
            )

    async def save_checkpoint(
        self, is_best: bool = False, checkpoint_name: str | None = None
    ):
        """Save model checkpoint with HuggingFace format."""
        await save_multi_turn_checkpoint(
            self,
            is_best=is_best,
            checkpoint_name=checkpoint_name,
            require_torch_fn=require_torch,
            notify_checkpoint_saved_fn=notify_checkpoint_saved,
        )

    def add_callback(self, callback: Any) -> None:
        """Add training callback"""
        self.callbacks.append(callback)

    def load_checkpoint(self, checkpoint_path: Any, *, trusted: bool = False) -> bool:
        """Load model and training state from a checkpoint directory.

        Args:
            checkpoint_path: Directory written by :meth:`save_checkpoint`.
            trusted: Pass ``True`` only for checkpoints from a source you
                control; the default unpickles with ``weights_only=True``.

        A model-only or metadata-only import returns ``False``. Imported agent
        weights remain available for inference, but a mutated learner stays
        quarantined until full recovery. Construct a fresh trainer for a
        deliberate warm start. Full recovery includes rollout publication unless
        ``rollout_sync=False`` explicitly opts out.
        """
        loaded: bool = load_multi_turn_checkpoint(
            self,
            checkpoint_path,
            require_torch_fn=require_torch,
            trainer_exceptions=MULTI_TRAINER_EXCEPTIONS,
            trusted=trusted,
        )
        if loaded:
            if not self._restore_rng_on_resume():
                # Per-rank streams were not saved; continue on fresh,
                # rank-distinct streams instead of replaying step 0's.
                self._seed_everything(require_torch(), step=self.global_step)
            # The transactional loader already published the restored policy.
        return loaded


# Alias for backwards compatibility
GRPOTrainer = MultiTurnGRPOTrainer
