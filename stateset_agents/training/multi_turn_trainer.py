"""
Multi-turn GRPO trainer implementation.

This module provides the MultiTurnGRPOTrainer class for training
multi-turn conversational agents using Group Relative Policy Optimization.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
from typing import Any

import numpy as np

from stateset_agents.core import trajectory as core_trajectory

from .callbacks import (
    notify_checkpoint_saved,
    notify_episode_end,
    notify_evaluation_end,
    notify_training_end,
    notify_training_start,
)
from .continual_learning import ContinualLearningManager
from .evaluation import EvaluationConfig, evaluate_agent
from .loss_computation import (
    _compute_group_policy_loss,
    compute_enhanced_grpo_loss,
    compute_grpo_loss,
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
    get_amp,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    require_torch,
    require_transformers,
)

MultiTurnTrajectory = core_trajectory.MultiTurnTrajectory
TrajectoryGroup = core_trajectory.TrajectoryGroup

logger = logging.getLogger(__name__)

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
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None

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

        # Optional reference model for KL penalty
        try:
            if (
                getattr(self.config, "use_reference_model", False)
                and self.reference_model is None
            ):
                from copy import deepcopy

                self.reference_model = deepcopy(self.agent.model).eval()
                for param in self.reference_model.parameters():
                    param.requires_grad = False
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

    def _setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler"""
        require_transformers()

        num_warmup_steps = int(
            num_training_steps * getattr(self.config, "warmup_ratio", 0.1)
        )

        lr_scheduler_type = getattr(self.config, "lr_scheduler_type", "cosine")

        if lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif lr_scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            # Constant learning rate
            self.lr_scheduler = None

        if self.lr_scheduler:
            logger.info(
                f"Scheduler initialized: {lr_scheduler_type} with {num_warmup_steps} warmup steps"
            )

    def _get_grad_accum_steps(self) -> int:
        steps = int(getattr(self.config, "gradient_accumulation_steps", 1) or 1)
        return max(1, steps)

    def _apply_optimizer_step(self, torch) -> None:
        if self.optimizer is None:
            return

        max_grad_norm = getattr(self.config, "max_grad_norm", 1.0)
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.agent.model.parameters(),
            max_grad_norm,
        )
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()

        self.optimizer.zero_grad()
        self.global_step += 1

    async def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        try:
            # Prepare configuration for W&B
            config_dict = {
                "framework": "grpo-agent-framework",
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
        num_generations = num_generations or getattr(self.config, "num_generations", 16)
        trajectory_groups = []

        for scenario in scenarios:
            # Generate multiple trajectories for the same scenario
            trajectories = []

            for _ in range(num_generations):
                try:
                    reset_fn = getattr(self.agent, "reset", None)
                    if callable(reset_fn):
                        reset_result = reset_fn()
                        if asyncio.iscoroutine(reset_result):
                            await reset_result

                    # Create agent function wrapper
                    async def agent_fn(history, context):
                        return await self.agent.generate_response(history, context)

                    # Generate trajectory
                    trajectory = await self.environment.run_episode(agent_fn, scenario)

                    # Apply reward function if provided
                    if self.reward_fn:
                        reward_result = await self.reward_fn.compute_reward(
                            trajectory.turns, scenario
                        )
                        score, breakdown = await self._coerce_reward_result(
                            reward_result
                        )
                        trajectory.total_reward = score
                        trajectory.metadata["reward_breakdown"] = breakdown

                    trajectories.append(trajectory)

                except MULTI_TRAINER_EXCEPTIONS as e:
                    logger.warning(f"Failed to generate trajectory: {e}")
                    continue

            if trajectories:
                group = TrajectoryGroup(
                    scenario_id=scenario.get(
                        "id", f"scenario_{len(trajectory_groups)}"
                    ),
                    trajectories=trajectories,
                    scenario_metadata=scenario,
                )
                trajectory_groups.append(group)

        return trajectory_groups

    async def _coerce_reward_result(
        self, reward_result: Any
    ) -> tuple[float, dict[str, float]]:
        """Normalize reward outputs into (score, breakdown)."""
        return await coerce_reward_result(reward_result, MULTI_TRAINER_EXCEPTIONS)

    def compute_grpo_loss(
        self, trajectory_groups: list[TrajectoryGroup]
    ) -> dict[str, Any]:
        """Compute GRPO loss from trajectory groups"""
        return compute_grpo_loss(
            trajectory_groups=trajectory_groups,
            config=self.config,
            agent=self.agent,
            global_reward_mean=self._global_reward_mean,
            global_reward_count=self._global_reward_count,
            update_global_stats=self._update_global_stats,
        )

    def _compute_group_policy_loss(
        self, group: TrajectoryGroup, advantages: Any
    ) -> Any:
        """Compute per-group policy loss (compatibility shim for tests)."""
        return _compute_group_policy_loss(
            group=group,
            advantages=advantages,
            config=self.config,
            agent=self.agent,
        )

    def compute_enhanced_grpo_loss(
        self, trajectory_groups: list[TrajectoryGroup], beta: float = 0.0
    ) -> dict[str, Any]:
        """Enhanced GRPO loss computation with KL penalty"""
        return compute_enhanced_grpo_loss(
            trajectory_groups=trajectory_groups,
            beta=beta,
            config=self.config,
            agent=self.agent,
            reference_model=self.reference_model,
        )

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
            if self.continual_manager.reference_model is not None:
                self.reference_model = self.continual_manager.reference_model
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
            self._global_reward_sum = self._global_reward_mean * self._global_reward_count

        self._global_reward_sum += batch_mean * batch_size
        self._global_reward_count += batch_size
        self._global_reward_mean = (
            self._global_reward_sum / self._global_reward_count
            if self._global_reward_count > 0
            else 0.0
        )

    def _get_environment_scenarios(self) -> list[dict[str, Any]]:
        return get_environment_scenarios(self.environment, self.config)

    def _split_scenarios(
        self, scenarios: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return split_scenarios(scenarios, self.config)

    def _apply_task_schedule(self, scenario: dict[str, Any], index: int) -> None:
        apply_task_schedule(self.config, scenario, index)

    def _expand_scenarios(
        self,
        scenarios: list[dict[str, Any]],
        count: int,
        prefix: str,
    ) -> list[dict[str, Any]]:
        return expand_scenarios(self.config, scenarios, count, prefix)

    def _format_trajectory_for_model(self, trajectory: MultiTurnTrajectory) -> str:
        """Format trajectory into text for model input."""
        return format_trajectory_for_model(self.agent, trajectory)

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
            if self.continual_manager.reference_model is not None:
                self.reference_model = self.continual_manager.reference_model
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

        with autocast_ctx:
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

        grad_accum_steps = self._get_grad_accum_steps()
        scaled_loss = loss / grad_accum_steps

        # Backward pass with gradient scaling
        if self.scaler:
            self.scaler.scale(scaled_loss).backward()

        else:
            scaled_loss.backward()

        self._grad_accum_step += 1
        optimizer_step = False
        if self._grad_accum_step % grad_accum_steps == 0:
            self._apply_optimizer_step(torch)
            optimizer_step = True

        # Prepare metrics
        metrics = {
            **{k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()},
            "learning_rate": self.optimizer.param_groups[0]["lr"]
            if self.optimizer
            else 0.0,
            "global_step": self.global_step,
            "optimizer_step": optimizer_step,
            "grad_accum_step": self._grad_accum_step,
        }

        return metrics

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

        return eval_metrics

    async def train(self) -> Any:
        """Main training loop"""
        logger.info("Starting GRPO training")

        # Calculate total training steps
        num_episodes = getattr(self.config, "num_episodes", 100)
        grad_accum_steps = self._get_grad_accum_steps()
        total_steps = math.ceil(num_episodes / grad_accum_steps)

        # Setup learning rate scheduler
        self._setup_scheduler(total_steps)

        resume_from = getattr(self.config, "resume_from_checkpoint", None)
        resumed = False
        if resume_from:
            resumed = self.load_checkpoint(resume_from)

        # Training scenarios (this would come from your dataset)
        training_scenarios = self._get_training_scenarios()
        eval_scenarios = self._get_eval_scenarios()

        await notify_training_start(self.callbacks, trainer=self, config=self.config)

        errored = False
        try:
            start_episode = 0
            if resumed:
                start_episode = min(num_episodes, max(0, int(self.current_epoch) + 1))
            for episode in range(start_episode, num_episodes):
                self.current_epoch = episode
                scenario = training_scenarios[episode]
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
                metrics = await self.training_step(training_groups)
                if replay_groups:
                    metrics["replay_group_count"] = len(replay_groups)
                    if self.continual_manager is not None:
                        metrics[
                            "replay_buffer_size"
                        ] = self.continual_manager.buffer.size

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

                # Log training metrics (only after first optimizer step to avoid
                # triggering every episode at global_step=0 during gradient accumulation)
                logging_steps = getattr(self.config, "logging_steps", 10)
                if self.global_step > 0 and self.global_step % logging_steps == 0:
                    await self._log_training_metrics(metrics, training_groups)

                # Evaluation (only after first optimizer step)
                eval_steps = getattr(self.config, "eval_steps", 50)
                if self.global_step > 0 and self.global_step % eval_steps == 0:
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
                if self.global_step > 0 and self.global_step % save_steps == 0:
                    await self.save_checkpoint()

                # Early stopping check
                early_stopping = getattr(self.config, "early_stopping", False)
                patience = getattr(self.config, "patience", 50)
                if early_stopping and self.steps_without_improvement >= patience:
                    logger.info(f"Early stopping at step {self.global_step}")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except MULTI_TRAINER_EXCEPTIONS as e:
            logger.error(f"Training failed: {e}")
            errored = True
            raise

        finally:
            if not errored and self._grad_accum_step % grad_accum_steps != 0:
                torch = require_torch()
                self._apply_optimizer_step(torch)

            if self.continual_manager is not None and self._current_task_id is not None:
                self.continual_manager.on_task_end(
                    agent=self.agent, task_id=self._current_task_id
                )
                if self.continual_manager.reference_model is not None:
                    self.reference_model = self.continual_manager.reference_model

            # Final checkpoint
            await self.save_checkpoint()

            await notify_training_end(
                self.callbacks,
                metrics={
                    "final_step": self.global_step,
                    "best_eval_metric": self.best_eval_metric,
                },
            )

            # Finish W&B run
            if self.wandb_logger:
                self.wandb_logger.finish_run(
                    {
                        "final_step": self.global_step,
                        "best_eval_metric": self.best_eval_metric,
                    }
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
        return await run_post_training_evaluation(
            self,
            eval_scenarios,
            num_samples,
            detailed,
            MULTI_TRAINER_EXCEPTIONS,
            self._coerce_reward_result,
        )

    def _get_training_scenarios(self) -> list[dict[str, Any]]:
        """Get training scenarios."""
        return build_training_scenarios(self.environment, self.config)

    def _get_eval_scenarios(self) -> list[dict[str, Any]]:
        """Get evaluation scenarios."""
        return build_eval_scenarios(self.environment, self.config)

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

    def add_callback(self, callback):
        """Add training callback"""
        self.callbacks.append(callback)

    def load_checkpoint(self, checkpoint_path: Any) -> bool:
        """Load model and training state from a checkpoint directory."""
        return load_multi_turn_checkpoint(
            self,
            checkpoint_path,
            require_torch_fn=require_torch,
            trainer_exceptions=MULTI_TRAINER_EXCEPTIONS,
        )


# Alias for backwards compatibility
GRPOTrainer = MultiTurnGRPOTrainer
