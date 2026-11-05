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
import math
from collections.abc import Iterator
from typing import Any, TypeAlias

import numpy as np

from stateset_agents.core import trajectory as core_trajectory
from stateset_agents.utils.async_calls import run_sync_owned

from .callbacks import (
    notify_checkpoint_saved,
    notify_episode_end,
    notify_evaluation_end,
    notify_optimizer_step,
    notify_training_end,
    notify_training_start,
)
from .config import estimate_grpo_optimizer_steps
from .continual_learning import ContinualLearningManager
from .ddp_population import run_token_local
from .evaluation import EvaluationConfig, evaluate_agent
from .grpo_accumulation import backward_accumulated_loss
from .loss_computation import (
    _compute_group_policy_loss,
    compute_enhanced_grpo_loss,
    compute_grpo_loss,
    compute_token_old_logprobs,
)
from .metric_reduction import carry_loss_metrics
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
from .profiling import TrainingProfiler, profile_learner_step
from .runtime_state import TrainingRuntime, validate_training_config
from .trainer_utils import (
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

MULTI_TRAINER_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    OSError,
)


class MultiTurnGRPOTrainer(TrainingRuntime):
    """
    GRPO trainer for multi-turn agents with HuggingFace and W&B integration

    This trainer implements Group Relative Policy Optimization specifically
    designed for multi-turn conversational agents, with full integration
    to HuggingFace ecosystem and Weights & Biases tracking.
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

        # Set seeds for reproducibility if provided
        try:
            import random

            seed_value = getattr(self.config, "seed", None)
            if seed_value is not None:
                random.seed(seed_value)
                np.random.seed(seed_value)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_value)
                torch.manual_seed(seed_value)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except MULTI_TRAINER_EXCEPTIONS as seed_err:
            logger.warning(f"Failed to fully set deterministic seeds: {seed_err}")

        # Initialize optimizer with HuggingFace best practices
        self._setup_optimizer()

        # Initialize mixed precision scaler (CUDA only)
        if torch.cuda.is_available() and (
            getattr(self.config, "bf16", False) or getattr(self.config, "fp16", False)
        ):
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

    def _initialize_reference_model(self) -> None:
        """Create a required frozen reference, propagating allocation failures.

        Stage the copy before assigning it: a failed clone/freeze cannot leave a
        partial reference or silently remove an explicitly requested KL policy.
        """
        if (
            getattr(self.config, "use_reference_model", False)
            and self.reference_model is None
        ):
            from copy import deepcopy

            reference = deepcopy(self.agent.model).eval()
            for parameter in reference.parameters():
                parameter.requires_grad = False
            self.reference_model = reference
            logger.info("Reference model initialized for KL regularization")

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

        for scenario in scenarios:
            # Generate multiple trajectories for the same scenario
            trajectories: list[MultiTurnTrajectory] = []

            for _ in range(num_generations):
                try:
                    reset_fn = getattr(self.agent, "reset", None)
                    if callable(reset_fn):
                        reset_result = reset_fn()
                        if asyncio.iscoroutine(reset_result):
                            await reset_result

                    # Generate trajectory
                    with self.profiler.measure("environment_rollout"):
                        trajectory = await self.environment.run_episode(
                            agent_fn, scenario
                        )

                    # Apply reward function if provided
                    if self.reward_fn:
                        reward_context = scenario
                        if getattr(
                            self.reward_fn, "requires_environment_context", False
                        ):
                            reward_context = {
                                **(trajectory.final_state or {}),
                                "scenario": scenario,
                            }
                        with self.profiler.measure("reward_evaluation"):
                            reward_result = await self.reward_fn.compute_reward(
                                trajectory.turns, reward_context
                            )
                        score, breakdown = await self._coerce_reward_result(
                            reward_result
                        )
                        trajectory.total_reward = score
                        trajectory.metadata["reward_breakdown"] = breakdown

                    evidence = (trajectory.final_state or {}).get("task_success")
                    if type(evidence) is not bool:
                        evidence = None
                    self.profiler.record_rollout(
                        accepted=True,
                        success=evidence,
                        tokens=sum(
                            len(turn.metadata.get("token_ids", []))
                            for turn in trajectory.turns
                            if turn.role == "assistant"
                        ),
                    )
                    trajectories.append(trajectory)

                except MULTI_TRAINER_EXCEPTIONS as e:
                    self.profiler.record_rollout(accepted=False, success=None)
                    if getattr(self.reward_fn, "fail_on_error", False) or getattr(
                        self, "_stateset_native_rollout", False
                    ):
                        raise
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
    ) -> dict[str, Any]:
        """Compute GRPO loss from trajectory groups"""
        loss_dict: dict[str, Any] = compute_grpo_loss(
            trajectory_groups=trajectory_groups,
            config=self.config,
            agent=self.agent,
            **self._reward_baseline_kwargs(),
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
    ) -> dict[str, Any]:
        """Enhanced GRPO loss computation with KL penalty"""
        loss_dict: dict[str, Any] = compute_enhanced_grpo_loss(
            trajectory_groups=trajectory_groups,
            beta=beta,
            config=self.config,
            agent=self.agent,
            reference_model=self.reference_model,
            old_logprobs=old_logprobs,
            **self._reward_baseline_kwargs(),
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
                old_logprobs = run_token_local(
                    self.agent.model,
                    "old-policy",
                    lambda: compute_token_old_logprobs(
                        trajectory_groups, self.config, self.agent
                    ),
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

        with autocast_ctx:
            if use_enhanced:
                loss_dict = self.compute_enhanced_grpo_loss(
                    trajectory_groups, beta=base_beta
                )
            else:
                loss_dict = self.compute_grpo_loss(trajectory_groups)
            loss, ewc_penalty = self._regularized_loss(loss_dict)

        grad_accum_steps = self._get_grad_accum_steps()
        backward_accumulated_loss(
            self, loss, loss_dict, torch, scaler=self.scaler, regularizer=ewc_penalty
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

        return carry_loss_metrics(metrics, loss_dict)

    def _regularized_loss(self, loss_dict: dict[str, Any]) -> tuple[Any, Any]:
        """Coordinate local continual penalties before entering backward phases."""

        def local() -> tuple[Any, Any]:
            loss = loss_dict["total_loss"]
            penalty = None
            if self.continual_manager is not None:
                penalty = self.continual_manager.compute_ewc_penalty(self.agent)
            if penalty is not None:
                loss = loss + penalty
                loss_dict["ewc_penalty"] = penalty
            return loss, penalty

        return run_token_local(self.agent.model, "regularizer", local)

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
        baseline = self._reward_baseline_kwargs()
        for step_index in range(inner):
            with (
                self._frozen_reward_baseline(baseline, update=step_index == 0),
                autocast_ctx_factory(),
            ):
                if use_enhanced:
                    loss_dict = self.compute_enhanced_grpo_loss(
                        trajectory_groups, beta=beta, old_logprobs=old_logprobs
                    )
                else:
                    loss_dict = self.compute_grpo_loss(
                        trajectory_groups, old_logprobs=old_logprobs
                    )
                loss, ewc_penalty = self._regularized_loss(loss_dict)
            backward_accumulated_loss(
                self,
                loss,
                loss_dict,
                torch,
                scaler=self.scaler,
                regularizer=ewc_penalty,
                accumulation_steps=1,
            )
            optimizer_updates += int(self._apply_optimizer_step(torch))
            ratio_means.append(float(loss_dict.get("ratio_mean", 1.0)))

        optimizer = self.optimizer
        metrics = {
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
        return carry_loss_metrics(metrics, loss_dict)

    async def _evaluate_local(
        self, eval_scenarios: list[dict[str, Any]], num_eval_episodes: int
    ) -> dict[str, float]:
        """Evaluate local episodes without advancing training RNGs or modes."""
        from stateset_agents.evaluation.isolation import isolated_evaluation

        if type(num_eval_episodes) is not int or num_eval_episodes < 0:
            raise ValueError("num_eval_episodes must be a nonnegative integer")
        concurrency = int(getattr(self.config, "rollout_concurrency", 1) or 1)
        num_generations = int(getattr(self.config, "eval_num_generations", 4) or 4)
        seed = getattr(self.config, "seed", 42)
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
                    max_turns=getattr(self.config, "max_steps_per_episode", None),
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

    def _prepare_training_loop(self) -> tuple[int, list[dict[str, Any]], int]:
        """Capture local scenarios once and budget the resolved task windows."""
        num_episodes = int(getattr(self.config, "num_episodes", 100) or 100)
        scenarios = self._get_training_scenarios()
        total_steps = estimate_grpo_optimizer_steps(
            num_episodes,
            gradient_accumulation_steps=self._get_grad_accum_steps(),
            num_gradient_updates=getattr(self.config, "num_gradient_updates", 1),
            task_ids=(
                [
                    self._resolve_task_id(scenario)
                    for scenario in scenarios[:num_episodes]
                ]
                if self.continual_manager is not None
                and getattr(self.continual_manager, "enabled", True)
                else None
            ),
        )
        if len(scenarios) < num_episodes:
            raise ValueError("Training scenarios do not cover the episode budget")
        return num_episodes, scenarios, total_steps

    async def _training_phase(
        self, name: str, function: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Run local work; distributed trainers coordinate its outcome."""
        result = function(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    def _resume_start(self, resumed: bool, num_episodes: int) -> int:
        return min(num_episodes, self._next_episode) if resumed else 0

    def _step_event_due(self, interval: int, metrics: dict[str, Any]) -> bool:
        """Fire once when committed updates cross a reporting boundary."""
        if not metrics.get("optimizer_step", True):
            return False
        updates = int(metrics.get("optimizer_updates", 1))
        previous = max(0, self.global_step - updates)
        return self.global_step // interval > previous // interval

    def _optimizer_update_receipt(
        self,
        previous_step: int,
        source: str,
        metrics: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Describe actual committed updates without inventing flush measurements."""
        if source not in {"batch", "task_boundary", "final_flush"}:
            raise ValueError("Unknown optimizer update source")
        if (
            type(previous_step) is not int
            or type(self.global_step) is not int
            or not 0 <= previous_step <= self.global_step <= 2**63 - 1
        ):
            raise ValueError("Invalid optimizer update interval")
        updates = self.global_step - previous_step
        if not updates:
            return None
        optimizer = self.optimizer
        result = {
            **(metrics or {}),
            "global_step": self.global_step,
            "optimizer_step": True,
            "optimizer_updates": updates,
            "optimizer_step_start": previous_step,
            "optimizer_step_end": self.global_step,
            "update_source": source,
            "update_task_id": task_id,
            "learning_rate": optimizer.param_groups[0]["lr"] if optimizer else 0.0,
            "grad_accum_step": self._grad_accum_step,
        }
        if source != "batch":
            result.update(
                accumulation_flush_updates=updates,
                trajectory_metrics={},
                trajectory_metric_scope="not_measured_flush",
            )
        return result

    async def _report_optimizer_update(
        self,
        receipt: dict[str, Any] | None,
        groups: list[TrajectoryGroup],
        eval_scenarios: list[dict[str, Any]],
    ) -> bool:
        """Run callback, logging, evaluation and save cadence for committed work."""
        if receipt is None:
            return False
        callback_metrics = dict(receipt)
        if isinstance(callback_metrics.get("trajectory_metrics"), dict):
            callback_metrics["trajectory_metrics"] = dict(
                callback_metrics["trajectory_metrics"]
            )
        await self._training_phase(
            "optimizer-step",
            notify_optimizer_step,
            self.callbacks,
            step=self.global_step,
            metrics=callback_metrics,
        )
        if self._step_event_due(getattr(self.config, "logging_steps", 10), receipt):
            await self._training_phase(
                "training-log", self._log_training_metrics, receipt, groups
            )
        if self._step_event_due(getattr(self.config, "eval_steps", 50), receipt):
            eval_metrics = await self.evaluate(eval_scenarios)
            await self._training_phase(
                "evaluation-log", self._log_eval_metrics, eval_metrics
            )
            await self._training_phase(
                "evaluation-end",
                notify_evaluation_end,
                self.callbacks,
                metrics=dict(eval_metrics),
            )
            logger.info(
                "Step %s (%s): Eval Reward = %.4f",
                self.global_step,
                receipt["update_source"],
                eval_metrics["eval_reward"],
            )
        if self._step_event_due(getattr(self.config, "save_steps", 100), receipt):
            await self.save_checkpoint()
            return True
        return False

    def _training_log_scalars(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Keep flush update counts distinct from measured objective losses."""
        if metrics.get("update_source") in {"task_boundary", "final_flush"}:
            updates = metrics["optimizer_updates"]
            return {
                "optimizer_updates": updates,
                "accumulation_flush_updates": updates,
                f"{metrics['update_source']}_updates": updates,
            }
        return {k: v for k, v in metrics.items() if "loss" in k}

    async def _generate_training_rollouts(
        self, scenarios: list[dict[str, Any]]
    ) -> list[TrajectoryGroup]:
        return await self.generate_trajectories(scenarios)

    def _rollouts_available(self, groups: list[TrajectoryGroup]) -> bool:
        return bool(groups)

    def _episode_metrics(
        self,
        metrics: dict[str, Any],
        new_groups: list[TrajectoryGroup],
        training_groups: list[TrajectoryGroup],
    ) -> dict[str, Any]:
        """Attach newly generated episode rewards, separate from replay samples."""
        rewards = [
            float(getattr(traj, "total_reward", 0.0))
            for group in new_groups
            for traj in group.trajectories
        ]
        reward = float(np.mean(rewards)) if rewards else 0.0
        return {**metrics, "total_reward": reward, "episode_reward": reward}

    def _should_stop_training(self) -> bool:
        return bool(
            getattr(self.config, "early_stopping", False)
            and self.steps_without_improvement >= getattr(self.config, "patience", 50)
        )

    async def _finish_tracking(self, summary: dict[str, Any]) -> None:
        if self.wandb_logger:
            await run_sync_owned(self.wandb_logger.finish_run, summary)

    async def train(self) -> Any:
        """Train, finalizing checkpoints only after a successful loop exit."""
        logger.info("Starting GRPO training")
        errored = False
        cancelled = False
        try:
            num_episodes, training_scenarios, total_steps = (
                self._prepare_training_loop()
            )
            await self._training_phase("scheduler", self._setup_scheduler, total_steps)

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
            eval_scenarios = await self._training_phase(
                "evaluation-scenarios", self._get_eval_scenarios
            )

            await self._training_phase(
                "training-start",
                notify_training_start,
                self.callbacks,
                trainer=self,
                config=self.config,
            )

            start_episode = self._resume_start(resumed, num_episodes)
            boundary_stopped = False
            for episode in range(start_episode, num_episodes):
                self.current_epoch = episode
                scenario = training_scenarios[episode]
                task_id = await self._training_phase(
                    "task-id", self._resolve_task_id, scenario
                )
                previous_step, previous_task = self.global_step, self._current_task_id
                self._maybe_handle_task_switch(task_id)
                boundary_receipt = self._optimizer_update_receipt(
                    previous_step, "task_boundary", task_id=previous_task
                )
                await self._report_optimizer_update(
                    boundary_receipt, [], eval_scenarios
                )
                if boundary_receipt is not None and self._should_stop_training():
                    # The outgoing task is complete; do not consolidate an
                    # incoming task that never generated a training population.
                    self._current_task_id = previous_task
                    boundary_stopped = True
                    break

                # Generate trajectory groups
                new_groups = await self._generate_training_rollouts([scenario])

                if not self._rollouts_available(new_groups):
                    logger.warning(
                        f"No trajectory groups generated for episode {episode}"
                    )
                    self._next_episode = episode + 1
                    continue

                training_groups, replay_groups = await self._training_phase(
                    "replay", self._maybe_mix_replay, new_groups, task_id
                )

                # Training step
                previous_step = self.global_step
                metrics = await self.training_step(training_groups)
                self._next_episode = episode + 1
                if replay_groups:
                    metrics["replay_group_count"] = len(replay_groups)
                    if self.continual_manager is not None:
                        metrics["replay_buffer_size"] = (
                            self.continual_manager.buffer.size
                        )

                # Native workers collectively report new/replay populations
                # before invoking rank-local callback or logger implementations.
                metrics = self._episode_metrics(metrics, new_groups, training_groups)
                callback_metrics = dict(metrics)
                if isinstance(callback_metrics.get("trajectory_metrics"), dict):
                    callback_metrics["trajectory_metrics"] = dict(
                        callback_metrics["trajectory_metrics"]
                    )
                await self._training_phase(
                    "episode-end",
                    notify_episode_end,
                    self.callbacks,
                    episode=episode,
                    metrics=callback_metrics,
                )

                receipt = self._optimizer_update_receipt(
                    previous_step, "batch", metrics, task_id=self._current_task_id
                )
                await self._report_optimizer_update(
                    receipt, training_groups, eval_scenarios
                )

                # Early stopping check
                if self._should_stop_training():
                    logger.info(f"Early stopping at step {self.global_step}")
                    break

            # Finalization closes the accumulation window. Saved state must not
            # make a resumed, already-complete run flush an empty window again.
            previous_step = self.global_step
            if not boundary_stopped:
                self._finish_current_task()
            final_receipt = self._optimizer_update_receipt(
                previous_step, "final_flush", task_id=self._current_task_id
            )
            saved = await self._report_optimizer_update(
                final_receipt, [], eval_scenarios
            )

            # A cadence save after final consolidation already captures this
            # exact learner state. Other successful exits still need a final save.
            if not saved:
                await self.save_checkpoint()
            await self._training_phase(
                "training-end",
                notify_training_end,
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
            self._grpo_accumulation = None
            raise
        finally:
            try:
                await self._training_phase(
                    "tracking-finish",
                    self._finish_tracking,
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
                    "Tracking cleanup failed after training stopped: %s", cleanup_error
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
        return loaded


# Alias for backwards compatibility
GRPOTrainer = MultiTurnGRPOTrainer
