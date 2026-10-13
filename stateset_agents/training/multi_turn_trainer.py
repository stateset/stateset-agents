"""
Multi-turn GRPO trainer implementation.

This module provides the MultiTurnGRPOTrainer class for training
multi-turn conversational agents using Group Relative Policy Optimization.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np

from stateset_agents.core import trajectory as core_trajectory
from stateset_agents.utils.async_calls import run_sync_owned

from .callbacks import (
    _maybe_await,
    notify_checkpoint_saved,
    notify_episode_end,
    notify_evaluation_end,
    notify_training_end,
    notify_training_start,
)
from .config import estimate_grpo_optimizer_steps
from .continual_learning import ContinualLearningManager
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


class MultiTurnGRPOTrainer:
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
        self.config = config
        self.wandb_logger = wandb_logger
        self.callbacks = callbacks or []

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_metric = float("-inf")
        self.steps_without_improvement = 0
        self._grad_accum_step = 0

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
        if torch.cuda.is_available() and getattr(self.config, "fp16", False) is True:
            if amp is not None:
                self.scaler = amp.GradScaler("cuda")
            else:  # pragma: no cover - AMP not available
                logger.warning(
                    "Mixed precision requested but torch.amp is unavailable; "
                    "proceeding without GradScaler."
                )

        # Optional reference model for KL penalty
        try:
            if (
                getattr(self.config, "use_reference_model", False)
                and self.reference_model is None
            ):
                self.reference_model = build_reference_model(self.agent.model)
                logger.info("Reference model initialized for KL regularization")
        except MULTI_TRAINER_EXCEPTIONS as ref_err:
            logger.warning(f"Could not initialize reference model: {ref_err}")
            self.reference_model = None

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
                "weight_decay": getattr(self.config, "weight_decay", 0.01),
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
            lr=getattr(self.config, "learning_rate", 5e-6),
            betas=(
                getattr(self.config, "adam_beta1", 0.9),
                getattr(self.config, "adam_beta2", 0.99),
            ),
            eps=1e-8,
        )

        logger.info(
            f"Optimizer initialized: AdamW with lr={getattr(self.config, 'learning_rate', 5e-6)}"
        )

    def _planned_optimizer_steps(self) -> int:
        """Optimizer-step budget implied by ``num_episodes`` (one rollout
        batch per episode), accumulation and inner updates."""
        num_episodes = int(getattr(self.config, "num_episodes", 100) or 100)
        return estimate_grpo_optimizer_steps(
            num_episodes,
            gradient_accumulation_steps=self._get_grad_accum_steps(),
            num_gradient_updates=getattr(self.config, "num_gradient_updates", 1),
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
        if getattr(self.config, "gradient_checkpointing", False) is not True:
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
            num_training_steps * getattr(self.config, "warmup_ratio", 0.1)
        )

        lr_scheduler_type = getattr(self.config, "lr_scheduler_type", "cosine")

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
        steps = int(getattr(self.config, "gradient_accumulation_steps", 1) or 1)
        return max(1, steps)

    def _sync_rollout_backend(self) -> None:
        """Keep an attached rollout engine on-policy after an optimizer step.

        ``self.last_rollout_sync`` is True/False for the sync outcome and
        ``None`` when nothing was attempted (``config.rollout_sync`` off or the
        agent has no ``sync_rollout_backend``).
        """
        self.last_rollout_sync = None
        if not bool(getattr(self.config, "rollout_sync", True)):
            return
        sync = getattr(self.agent, "sync_rollout_backend", None)
        if callable(sync):
            self.last_rollout_sync = bool(sync())

    def _apply_optimizer_step(self, torch, *, gradient_scale: float = 1.0) -> bool:
        return safe_optimizer_step(
            self,
            torch,
            max_grad_norm=getattr(self.config, "max_grad_norm", 1.0),
            scaler=self.scaler,
            gradient_scale=gradient_scale,
        )

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

            self.wandb_logger.init_run(
                config=config_dict,
                name=getattr(self.config, "run_name", None),
                tags=getattr(self.config, "wandb_tags", ["grpo", "multi-turn"]),
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
            num_generations or getattr(self.config, "num_generations", 16) or 16
        )
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

        for index, scenario in enumerate(scenarios):
            # Generate multiple trajectories for the same scenario
            trajectories: list[MultiTurnTrajectory] = []

            if batched is not None:
                trajectories = batched[index]
            for _ in range(num_generations if batched is None else 0):
                try:
                    await self._reset_agent()

                    # Generate trajectory
                    trajectory = await self.environment.run_episode(agent_fn, scenario)
                    await self._score_trajectory(trajectory, scenario)
                    trajectories.append(trajectory)

                except MULTI_TRAINER_EXCEPTIONS as e:
                    logger.warning(f"Failed to generate trajectory: {e}")
                    continue

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

        return trajectory_groups

    async def _reset_agent(self) -> None:
        reset_fn = getattr(self.agent, "reset", None)
        if callable(reset_fn):
            await _maybe_await(reset_fn())

    async def _score_trajectory(
        self, trajectory: MultiTurnTrajectory, scenario: dict[str, Any]
    ) -> None:
        """Apply the trainer's reward function, if any, to a finished episode."""
        if self.reward_fn:
            reward_result = await self.reward_fn.compute_reward(
                trajectory.turns, scenario
            )
            score, breakdown = await self._coerce_reward_result(reward_result)
            trajectory.total_reward = score
            trajectory.metadata["reward_breakdown"] = breakdown

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
            not getattr(self.config, "batched_rollouts", True)
            or not inspect.iscoroutinefunction(batch_turns)
            or not inspect.iscoroutinefunction(run_batched)
            or getattr(self.environment, "supports_concurrent_episodes", False)
            is not True
        ):
            return None
        flat = [s for s in scenarios for _ in range(num_generations)]
        try:
            await self._reset_agent()
            episodes = await run_batched(
                lambda histories, contexts: batch_turns(histories, contexts), flat
            )
            for trajectory, scenario in zip(episodes, flat, strict=True):
                await self._score_trajectory(trajectory, scenario)
        except MULTI_TRAINER_EXCEPTIONS as e:
            logger.warning(
                "Batched rollout failed (%s); rolling episodes out one at a time", e
            )
            return None
        return [
            list(episodes[i * num_generations : (i + 1) * num_generations])
            for i in range(len(scenarios))
        ]

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
        mean, count = reward_stats or (
            self._global_reward_mean,
            self._global_reward_count,
        )
        loss_dict: dict[str, Any] = compute_grpo_loss(
            trajectory_groups=trajectory_groups,
            config=self.config,
            agent=self.agent,
            global_reward_mean=mean,
            global_reward_count=count,
            update_global_stats=self._update_global_stats if update_stats else None,
            old_logprobs=old_logprobs,
            backward_fn=getattr(self, "_loss_backward_fn", None),
        )
        return loss_dict

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
        mean, count = reward_stats or (
            self._global_reward_mean,
            self._global_reward_count,
        )
        loss_dict: dict[str, Any] = compute_enhanced_grpo_loss(
            trajectory_groups=trajectory_groups,
            beta=beta,
            config=self.config,
            agent=self.agent,
            reference_model=self.reference_model,
            old_logprobs=old_logprobs,
            global_reward_mean=mean,
            global_reward_count=count,
            update_global_stats=self._update_global_stats if update_stats else None,
            backward_fn=getattr(self, "_loss_backward_fn", None),
        )
        return loss_dict

    def _resolve_task_id(self, scenario: Any) -> str | None:
        """Extract a task id from scenario metadata when available."""
        if not isinstance(scenario, dict) or self.config is None:
            return None
        task_key = getattr(self.config, "task_id_key", "task_id")
        if task_key and task_key in scenario:
            value = scenario.get(task_key)
            return str(value) if value is not None else None
        return None

    def _maybe_handle_task_switch(self, task_id: str | None) -> None:
        """Handle task boundary transitions for continual learning."""
        if self.continual_manager is None or task_id is None:
            return

        if self._current_task_id is None:
            self._current_task_id = task_id
            return

        if task_id != self._current_task_id:
            self.continual_manager.on_task_end(
                agent=self.agent, task_id=self._current_task_id
            )
            reference_model = getattr(self.continual_manager, "reference_model", None)
            if reference_model is not None:
                self.reference_model = reference_model
            self._current_task_id = task_id

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

    async def training_step(
        self, trajectory_groups: list[TrajectoryGroup]
    ) -> dict[str, Any]:
        """Execute a single training step"""
        torch = require_torch()
        amp = get_amp()

        self.agent.model.train()

        # Compute GRPO loss
        use_amp = bool(
            getattr(self.config, "bf16", False) or getattr(self.config, "fp16", False)
        )
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        base_beta = float(getattr(self.config, "beta", 0.0))
        if self.continual_manager is not None:
            reference_model = getattr(self.continual_manager, "reference_model", None)
            if reference_model is not None:
                self.reference_model = reference_model
            base_beta = self.continual_manager.get_effective_beta(base_beta)

        use_enhanced = bool(
            base_beta > 0.0
            or getattr(self.config, "use_reference_model", False)
            or self.reference_model is not None
        )

        autocast_ctx = (
            amp.autocast(device_type=device_type, enabled=use_amp)
            if amp is not None
            else contextlib.nullcontext()
        )

        inner = max(1, int(getattr(self.config, "num_gradient_updates", 1) or 1))
        if inner > 1:
            # Same precision as the inner-update losses, so the first ratio is 1.
            with (
                amp.autocast(device_type=device_type, enabled=use_amp)
                if amp is not None
                else contextlib.nullcontext()
            ):
                old_logprobs = compute_token_old_logprobs(
                    trajectory_groups, self.config, self.agent
                )
            if old_logprobs is None:
                self._warn_inner_updates_unavailable()
            else:
                return self._inner_update_steps(
                    trajectory_groups,
                    inner,
                    old_logprobs,
                    use_enhanced=use_enhanced,
                    beta=base_beta,
                    autocast_ctx_factory=lambda: (
                        amp.autocast(device_type=device_type, enabled=use_amp)
                        if amp is not None
                        else contextlib.nullcontext()
                    ),
                    torch=torch,
                )

        grad_accum_steps = self._get_grad_accum_steps()
        with (
            autocast_ctx,
            self._chunked_backward(torch, amp, device_type, grad_accum_steps),
        ):
            if use_enhanced:
                loss_dict = self.compute_enhanced_grpo_loss(
                    trajectory_groups, beta=base_beta
                )
            else:
                loss_dict = self.compute_grpo_loss(trajectory_groups)
            loss = loss_dict["total_loss"]

            ewc_penalty = None
            if self.continual_manager is not None:
                ewc_penalty = self.continual_manager.compute_ewc_penalty(self.agent)
            if ewc_penalty is not None:
                loss = loss + ewc_penalty
                loss_dict["ewc_penalty"] = ewc_penalty

        self._backward_remaining(
            loss_dict, loss, ewc_penalty, torch, divisor=grad_accum_steps
        )

        self._grad_accum_step += 1
        optimizer_step = False
        if self._grad_accum_step % grad_accum_steps == 0:
            optimizer_step = self._apply_optimizer_step(torch)

        # Prepare metrics
        optimizer = self.optimizer
        metrics = {
            **{k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()},
            "learning_rate": (
                optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
            ),
            "global_step": self.global_step,
            "optimizer_step": optimizer_step,
            "optimizer_updates": int(optimizer_step),
            "grad_accum_step": self._grad_accum_step,
            "inner_updates": 1,
        }

        return metrics

    @contextlib.contextmanager
    def _chunked_backward(
        self, torch: Any, amp: Any, device_type: str, divisor: int
    ) -> Any:
        """Let the token-path loss backpropagate each row chunk as it goes.

        While active, :meth:`compute_grpo_loss` and
        :meth:`compute_enhanced_grpo_loss` hand the loss a ``backward_fn``
        that divides a chunk's share by ``divisor`` (``gradient_accumulation_
        steps`` on the accumulation path) and backpropagates it through the
        GradScaler, outside autocast. Peak activation memory is then one
        chunk's graph instead of the whole batch's. Losses that return
        ``backward_done`` must not be backpropagated again
        (:meth:`_backward_remaining`).
        """

        def backward_fn(share: Any) -> None:
            with (
                amp.autocast(device_type=device_type, enabled=False)
                if amp is not None
                else contextlib.nullcontext()
            ):
                backward_training_loss(self, share / divisor, torch, scaler=self.scaler)

        previous: Callable[[Any], None] | None = getattr(
            self, "_loss_backward_fn", None
        )
        self._loss_backward_fn: Callable[[Any], None] | None = backward_fn
        try:
            yield backward_fn
        finally:
            self._loss_backward_fn = previous

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

    def _inner_update_steps(
        self,
        trajectory_groups: list[TrajectoryGroup],
        inner: int,
        old_logprobs: list[Any],
        *,
        use_enhanced: bool,
        beta: float,
        autocast_ctx_factory: Any,
        torch: Any,
    ) -> dict[str, Any]:
        """PPO/DAPO-style mini-epochs: ``inner`` full optimizer steps against
        the frozen old-policy log-probs of this rollout batch."""
        ratio_means: list[float] = []
        # old_logprobs was frozen before this flush. The first inner loss must
        # therefore account for any policy change caused by the pending update.
        accumulation_flush_updates = int(self._flush_accumulation(torch))
        optimizer_updates = accumulation_flush_updates
        loss_dict: dict[str, Any] = {}
        # Every inner update scores the batch against the same baseline, and
        # the batch enters the running reward statistics once.
        reward_stats = (self._global_reward_mean, self._global_reward_count)
        amp = get_amp()
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        for step_idx in range(inner):
            with (
                autocast_ctx_factory(),
                self._chunked_backward(torch, amp, device_type, 1),
            ):
                if use_enhanced:
                    loss_dict = self.compute_enhanced_grpo_loss(
                        trajectory_groups,
                        beta=beta,
                        old_logprobs=old_logprobs,
                        reward_stats=reward_stats,
                        update_stats=step_idx == 0,
                    )
                else:
                    loss_dict = self.compute_grpo_loss(
                        trajectory_groups,
                        old_logprobs=old_logprobs,
                        reward_stats=reward_stats,
                        update_stats=step_idx == 0,
                    )
                loss = loss_dict["total_loss"]
                ewc_penalty = None
                if self.continual_manager is not None:
                    ewc_penalty = self.continual_manager.compute_ewc_penalty(self.agent)
                if ewc_penalty is not None:
                    loss = loss + ewc_penalty
                    loss_dict["ewc_penalty"] = ewc_penalty
            self._backward_remaining(loss_dict, loss, ewc_penalty, torch, divisor=1)
            optimizer_updates += int(self._apply_optimizer_step(torch))
            ratio_means.append(float(loss_dict.get("ratio_mean", 1.0)))

        optimizer = self.optimizer
        return {
            **{k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()},
            "learning_rate": (
                optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
            ),
            "global_step": self.global_step,
            "optimizer_step": optimizer_updates > 0,
            "optimizer_updates": optimizer_updates,
            "accumulation_flush_updates": accumulation_flush_updates,
            "grad_accum_step": self._grad_accum_step,
            "inner_updates": inner,
            "ratio_mean_last": ratio_means[-1] if ratio_means else 1.0,
        }

    async def evaluate(
        self, eval_scenarios: list[dict[str, Any]], num_eval_episodes: int = 10
    ) -> dict[str, float]:
        """Evaluate the agent"""
        self.agent.model.eval()

        concurrency = int(getattr(self.config, "rollout_concurrency", 1) or 1)
        num_generations = int(getattr(self.config, "eval_num_generations", 4) or 4)

        eval_metrics = await evaluate_agent(
            agent=self.agent,
            environment=self.environment,
            scenarios=eval_scenarios[:num_eval_episodes],
            reward_fn=self.reward_fn,
            config=EvaluationConfig(
                num_episodes=min(num_eval_episodes, len(eval_scenarios)),
                num_generations=num_generations,
                max_turns=getattr(self.config, "max_steps_per_episode", None),
                seed=getattr(self.config, "seed", None),
                concurrency=concurrency,
            ),
        )

        # Check for best model
        current_metric = eval_metrics["eval_reward"]
        if current_metric > self.best_eval_metric:
            self.best_eval_metric = current_metric
            self.steps_without_improvement = 0

            # Save best checkpoint
            await self.save_checkpoint(is_best=True)
        else:
            self.steps_without_improvement += 1

        typed_eval_metrics: dict[str, float] = eval_metrics
        return typed_eval_metrics

    async def train(self) -> Any:
        """Train, finalizing checkpoints only after a successful loop exit."""
        logger.info("Starting GRPO training")
        errored = False
        cancelled = False
        try:
            # Calculate total training steps
            num_episodes = int(getattr(self.config, "num_episodes", 100) or 100)
            grad_accum_steps = self._get_grad_accum_steps()
            total_steps = self._planned_optimizer_steps()

            # Setup learning rate scheduler
            self._setup_scheduler(total_steps)

            resume_from = getattr(self.config, "resume_from_checkpoint", None)
            resumed = False
            if resume_from is not None:
                if isinstance(resume_from, str) and not resume_from.strip():
                    raise ValueError("resume_from_checkpoint cannot be empty")
                resumed = self.load_checkpoint(resume_from)
                if resumed is not True:
                    raise ValueError(
                        "Requested training checkpoint could not be restored; "
                        "refusing to start a fresh run instead of resuming"
                    )

            # Training scenarios (this would come from your dataset)
            training_scenarios = self._get_training_scenarios()
            eval_scenarios = self._get_eval_scenarios()

            await notify_training_start(
                self.callbacks, trainer=self, config=self.config
            )

            start_episode = 0
            if resumed:
                start_episode = min(num_episodes, max(0, int(self.current_epoch) + 1))
            for episode in range(start_episode, num_episodes):
                self.current_epoch = episode
                scenario = self._episode_scenario(training_scenarios, episode)
                task_id = self._resolve_task_id(scenario)
                self._maybe_handle_task_switch(task_id)

                # Generate trajectory groups
                new_groups = await self.generate_trajectories([scenario])

                if not new_groups:
                    logger.warning(
                        f"No trajectory groups generated for episode {episode}"
                    )
                    continue

                training_groups, replay_groups = self._maybe_mix_replay(
                    new_groups, task_id
                )

                # Training step
                step_before = self.global_step
                metrics = await self.training_step(training_groups)
                if replay_groups:
                    metrics["replay_group_count"] = len(replay_groups)
                    if self.continual_manager is not None:
                        metrics["replay_buffer_size"] = (
                            self.continual_manager.buffer.size
                        )

                # Callback: episode end (include reward signal for generic monitors)
                episode_rewards = [
                    float(getattr(traj, "total_reward", 0.0))
                    for group in new_groups
                    for traj in group.trajectories
                ]
                avg_episode_reward = (
                    float(np.mean(episode_rewards)) if episode_rewards else 0.0
                )
                await notify_episode_end(
                    self.callbacks,
                    episode=episode,
                    metrics={
                        **metrics,
                        "total_reward": avg_episode_reward,
                        "episode_reward": avg_episode_reward,
                    },
                )

                # Periodic work fires once per interval boundary crossed by this
                # episode's optimizer steps: not again on accumulation-only
                # episodes (global_step unchanged), and not skipped when one
                # episode takes several steps past a multiple.
                def crossed(every: int, before: int = step_before) -> bool:
                    every = max(1, int(every))
                    return self.global_step // every > before // every

                logging_steps = getattr(self.config, "logging_steps", 10)
                if crossed(logging_steps):
                    await self._log_training_metrics(metrics, training_groups)

                # Evaluation (only after first optimizer step)
                eval_steps = getattr(self.config, "eval_steps", 50)
                if crossed(eval_steps):
                    eval_metrics = await self.evaluate(eval_scenarios)
                    await self._log_eval_metrics(eval_metrics)
                    await notify_evaluation_end(
                        self.callbacks, metrics=dict(eval_metrics)
                    )

                    logger.info(
                        f"Step {self.global_step}: "
                        f"Train Loss = {metrics['total_loss']:.4f}, "
                        f"Eval Reward = {eval_metrics['eval_reward']:.4f}"
                    )

                # Save checkpoint (only after first optimizer step)
                save_steps = getattr(self.config, "save_steps", 100)
                if crossed(save_steps):
                    await self.save_checkpoint()

                # Early stopping check
                early_stopping = getattr(self.config, "early_stopping", False)
                patience = getattr(self.config, "patience", 50)
                if early_stopping and self.steps_without_improvement >= patience:
                    logger.info(f"Early stopping at step {self.global_step}")
                    break

            # Finalization closes the accumulation window. Saved state must not
            # make a resumed, already-complete run flush an empty window again.
            if self._grad_accum_step % grad_accum_steps:
                self._flush_accumulation(require_torch())
            self._grad_accum_step = 0

            if self.continual_manager is not None and self._current_task_id is not None:
                self.continual_manager.on_task_end(
                    agent=self.agent, task_id=self._current_task_id
                )
                reference_model = getattr(
                    self.continual_manager, "reference_model", None
                )
                if reference_model is not None:
                    self.reference_model = reference_model

            # A failed final save is a failed run, not a successful completion
            # with a silently missing output artifact.
            await self.save_checkpoint()
            await notify_training_end(
                self.callbacks,
                metrics={
                    "final_step": self.global_step,
                    "best_eval_metric": self.best_eval_metric,
                },
            )
        except BaseException as error:
            # Cancellation and interrupts must not flush pending gradients or
            # enter success finalization. Already committed updates stay intact.
            errored = True
            cancelled = isinstance(error, (asyncio.CancelledError, KeyboardInterrupt))
            logger.info(
                "Training stopped without successful finalization: %s",
                type(error).__name__,
            )
            if self.optimizer is not None:
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except Exception as cleanup_error:
                    logger.warning(
                        "Could not discard pending gradients: %s", cleanup_error
                    )
            self._grad_accum_step = 0
            raise
        finally:
            if self.wandb_logger:
                try:
                    await run_sync_owned(
                        self.wandb_logger.finish_run,
                        {
                            "final_step": self.global_step,
                            "best_eval_metric": self.best_eval_metric,
                            "errored": errored,
                            "cancelled": cancelled,
                        },
                    )
                except Exception as cleanup_error:
                    if not errored:
                        raise
                    logger.warning(
                        "Tracking cleanup failed after training stopped: %s",
                        cleanup_error,
                    )

        logger.info("Training completed")
        return self.agent

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

        seed_value = getattr(self.config, "seed", None)
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
                losses={k: v for k, v in metrics.items() if "loss" in k},
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
            # An attached rollout engine still holds the pre-resume weights.
            self._sync_rollout_backend()
        return loaded


# Alias for backwards compatibility
GRPOTrainer = MultiTurnGRPOTrainer
