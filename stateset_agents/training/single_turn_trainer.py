"""
Single-turn GRPO trainer implementation.

This module provides the SingleTurnGRPOTrainer class for training
agents on single-turn interactions (one prompt, one response).
"""

from __future__ import annotations

import contextlib
import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from stateset_agents.core.trajectory import (
    ConversationTurn,
    MultiTurnTrajectory,
    TrajectoryGroup,
)

from .callbacks import (
    notify_checkpoint_saved,
    notify_episode_end,
    notify_training_end,
    notify_training_start,
)
from .single_turn_checkpointing import (
    load_checkpoint_artifacts,
    resolve_checkpoint_path,
    save_checkpoint_artifacts,
)
from .single_turn_state import (
    apply_task_schedule,
    extract_context,
    extract_prompt,
    get_environment_scenarios,
    get_episode_scenario,
    merge_scenario_into_state,
    resolve_task_id,
)
from .continual_learning import ContinualLearningManager
from .loss_computation import compute_enhanced_grpo_loss, compute_grpo_loss
from .trainer_utils import (
    get_amp,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_torch,
    require_torch,
    require_transformers,
)

logger = logging.getLogger(__name__)

SINGLE_TRAINER_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    OSError,
)


class SingleTurnGRPOTrainer:
    """
    GRPO trainer for single-turn agents with HuggingFace and W&B integration

    This trainer implements Group Relative Policy Optimization for simple
    single-turn interactions (one prompt, one response), making it ideal
    for supervised fine-tuning and basic RL scenarios.
    """

    def __init__(
        self,
        agent: Any,  # Base Agent, not MultiTurnAgent
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

        # GRPO enhancements
        self._global_reward_mean: float = 0.0
        self._global_reward_count: int = 0
        self.reference_model = None
        self.continual_manager: ContinualLearningManager | None = None
        self._current_task_id: str | None = None

        logger.info(f"Single-Turn GRPO Trainer initialized with {type(agent).__name__}")

    async def initialize(self):
        """Initialize trainer components"""
        logger.info("Initializing Single-Turn GRPO trainer...")

        torch = self._get_torch_module()
        require_transformers()
        amp = get_amp()

        # Initialize agent if not already done
        if not hasattr(self.agent, "model") or self.agent.model is None:
            await self.agent.initialize()

        # Set seeds for reproducibility
        try:
            import random

            seed_value = getattr(self.config, "seed", None)
            if seed_value is not None and torch is not None:
                random.seed(seed_value)
                np.random.seed(seed_value)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_value)
                torch.manual_seed(seed_value)
        except SINGLE_TRAINER_EXCEPTIONS as seed_err:
            logger.warning(f"Failed to set seeds: {seed_err}")

        # Initialize optimizer
        self._setup_optimizer()

        # Initialize mixed precision scaler (CUDA only)
        if torch is not None and amp is not None and torch.cuda.is_available():
            bf16_enabled = getattr(self.config, "bf16", False)
            fp16_enabled = getattr(self.config, "fp16", False)
            if bf16_enabled or fp16_enabled:
                self.scaler = amp.GradScaler("cuda")

        # Optional reference model for KL penalty
        try:
            if (
                getattr(self.config, "use_reference_model", False)
                and self.reference_model is None
            ):
                self.reference_model = deepcopy(self.agent.model).eval()
                for param in self.reference_model.parameters():
                    param.requires_grad = False
                logger.info("Reference model initialized for KL regularization")
        except SINGLE_TRAINER_EXCEPTIONS as ref_err:
            logger.warning(f"Could not initialize reference model: {ref_err}")
            self.reference_model = None

        self._init_continual_learning()

        logger.info("Single-Turn GRPO trainer initialized successfully")

    def _get_torch_module(self) -> Any:
        """Return the torch module (or None if unavailable)."""
        return get_torch()

    def _setup_optimizer(self):
        """Set up optimizer with HuggingFace best practices"""
        torch = self._get_torch_module()
        if self.agent.model is None or torch is None:
            return

        learning_rate = getattr(self.config, "learning_rate", 5e-5)
        weight_decay = getattr(self.config, "weight_decay", 0.01)

        # Separate parameters that should not have weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.agent.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.agent.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=learning_rate
        )
        logger.info(f"Optimizer initialized with lr={learning_rate}")

    def _init_continual_learning(self) -> None:
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

    def _get_environment_scenarios(self) -> list[dict[str, Any]]:
        return get_environment_scenarios(self.environment, self.config)

    def _apply_task_schedule(self, scenario: dict[str, Any], episode: int) -> None:
        apply_task_schedule(self.config, scenario, episode)

    def _get_episode_scenario(self, episode: int) -> dict[str, Any] | None:
        return get_episode_scenario(self.environment, self.config, episode)

    def _resolve_task_id(self, context: dict[str, Any] | None) -> str | None:
        return resolve_task_id(self.config, context)

    def _maybe_handle_task_switch(self, task_id: str | None) -> None:
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

    def _get_grad_accum_steps(self) -> int:
        steps = int(getattr(self.config, "gradient_accumulation_steps", 1) or 1)
        return max(1, steps)

    def _apply_optimizer_step(self, torch, use_amp: bool, max_grad_norm: float) -> None:
        if self.optimizer is None:
            return
        if self.scaler is not None and use_amp:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), max_grad_norm)
        if self.scaler is not None and use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def _setup_scheduler(self, num_training_steps: int) -> None:
        """Set up learning rate scheduler."""
        # require_transformers already called in initialize
        if self.optimizer is None:
            self.lr_scheduler = None
            return

        num_warmup_steps = int(
            num_training_steps * getattr(self.config, "warmup_ratio", 0.1)
        )
        lr_scheduler_type = getattr(self.config, "lr_scheduler_type", "cosine")

        try:
            if lr_scheduler_type == "cosine" and get_cosine_schedule_with_warmup:
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                )
            elif lr_scheduler_type == "linear" and get_linear_schedule_with_warmup:
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                )
            else:
                self.lr_scheduler = None
        except SINGLE_TRAINER_EXCEPTIONS as sched_err:
            logger.debug("Scheduler initialization skipped: %s", sched_err)
            self.lr_scheduler = None

    def _update_global_stats(self, batch_mean: float, batch_size: int) -> None:
        """Update running reward statistics for global baselines."""
        self._global_reward_mean = (
            self._global_reward_mean * self._global_reward_count
            + batch_mean * batch_size
        ) / max(1, self._global_reward_count + batch_size)
        self._global_reward_count += batch_size

    async def _compute_reward_for_turns(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> float:
        """Compute reward using reward_fn, environment, or default."""
        if self.reward_fn is not None:
            try:
                result = await self.reward_fn.compute_reward(turns, context)
                if hasattr(result, "score"):
                    return float(result.score)
                if isinstance(result, dict) and "score" in result:
                    return float(result["score"])
                if isinstance(result, (int, float)):
                    return float(result)
            except SINGLE_TRAINER_EXCEPTIONS as e:
                logger.warning(f"Reward function failed: {e}")

        # Fallback to environment reward interface when available
        get_reward = getattr(self.environment, "get_reward", None)
        if callable(get_reward):
            try:
                trajectory = MultiTurnTrajectory(turns=turns, total_reward=0.0)
                env_reward = await get_reward(trajectory)  # type: ignore[misc]
                return float(env_reward)
            except SINGLE_TRAINER_EXCEPTIONS:
                pass

        return 0.5

    async def _generate_trajectory_group(
        self,
        prompt: str,
        num_generations: int,
        context: dict[str, Any] | None = None,
    ) -> tuple[TrajectoryGroup, str]:
        """Generate a GRPO trajectory group for a single prompt."""
        trajectories: list[MultiTurnTrajectory] = []
        best_response = ""
        best_reward = float("-inf")

        for gen_idx in range(num_generations):
            response = await self.agent.generate_response(prompt)
            turns = [
                ConversationTurn(role="user", content=str(prompt)),
                ConversationTurn(role="assistant", content=str(response)),
            ]
            reward = await self._compute_reward_for_turns(turns, context)

            trajectory = MultiTurnTrajectory(
                turns=turns,
                total_reward=reward,
                metadata={
                    "prompt": prompt,
                    "generation_index": gen_idx,
                },
            )
            trajectories.append(trajectory)

            if reward > best_reward:
                best_reward = reward
                best_response = str(response)

        group = TrajectoryGroup(
            scenario_id=f"prompt_{abs(hash(prompt))}",
            trajectories=trajectories,
            scenario_metadata={"prompt": prompt},
        )
        return group, best_response

    def _extract_context(self, state: Any) -> dict[str, Any] | None:
        return extract_context(state)

    def _extract_prompt(self, state: Any, context: dict[str, Any] | None) -> str:
        return extract_prompt(state, context)

    def _merge_scenario_into_state(
        self,
        state: Any,
        scenario: dict[str, Any] | None,
    ) -> Any:
        """Preserve selected scenario context across dict-based environments."""
        return merge_scenario_into_state(state, scenario, self.config)

    async def train(self) -> Any:
        """Run single-turn GRPO training loop with group-relative updates."""
        logger.info("Starting single-turn GRPO training...")

        torch = self._get_torch_module()
        amp = get_amp()
        use_amp = bool(
            getattr(self.config, "bf16", False) or getattr(self.config, "fp16", False)
        )

        await notify_training_start(self.callbacks, trainer=self, config=self.config)

        num_episodes = getattr(self.config, "num_episodes", 10)
        max_steps = getattr(self.config, "max_steps_per_episode", 1) or 1
        num_generations = getattr(self.config, "num_generations", 8)
        max_grad_norm = getattr(self.config, "max_grad_norm", 1.0)

        grad_accum_steps = self._get_grad_accum_steps()
        total_steps = num_episodes * max_steps
        total_updates = math.ceil(total_steps / grad_accum_steps)
        self._setup_scheduler(total_updates)

        resume_from = getattr(self.config, "resume_from_checkpoint", None)
        resumed = False
        if resume_from:
            resumed = self.load_checkpoint(resume_from)

        start_episode = 0
        if resumed:
            start_episode = min(num_episodes, max(0, int(self.current_epoch) + 1))

        for episode in range(start_episode, num_episodes):
            self.current_epoch = episode
            scenario = self._get_episode_scenario(episode)

            try:
                state = await self.environment.reset(scenario)
            except TypeError:
                state = await self.environment.reset()
            state = self._merge_scenario_into_state(state, scenario)
            episode_rewards: list[float] = []
            replay_group_count = 0

            for _ in range(max_steps):
                context = self._extract_context(state)
                prompt = self._extract_prompt(state, context)
                task_id = self._resolve_task_id(context)
                self._maybe_handle_task_switch(task_id)

                group, best_response = await self._generate_trajectory_group(
                    prompt=str(prompt),
                    num_generations=num_generations,
                    context=context,
                )

                training_groups = [group]
                replay_groups: list[TrajectoryGroup] = []
                if self.continual_manager is not None:
                    replay_groups = self.continual_manager.sample_replay_groups(
                        len(training_groups)
                    )
                    self.continual_manager.add_trajectory_groups(
                        training_groups, task_id=task_id
                    )
                    if replay_groups:
                        replay_group_count += len(replay_groups)
                        training_groups = training_groups + replay_groups

                # Compute GRPO loss
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

                try:
                    with autocast_ctx:
                        if use_enhanced:
                            loss_dict = compute_enhanced_grpo_loss(
                                trajectory_groups=training_groups,
                                beta=base_beta,
                                config=self.config,
                                agent=self.agent,
                                reference_model=self.reference_model,
                            )
                        else:
                            loss_dict = compute_grpo_loss(
                                trajectory_groups=training_groups,
                                config=self.config,
                                agent=self.agent,
                                global_reward_mean=self._global_reward_mean,
                                global_reward_count=self._global_reward_count,
                                update_global_stats=self._update_global_stats,
                            )

                        ewc_penalty = None
                        if self.continual_manager is not None:
                            ewc_penalty = self.continual_manager.compute_ewc_penalty(
                                self.agent
                            )
                        if ewc_penalty is not None:
                            loss_dict["ewc_penalty"] = ewc_penalty
                            if loss_dict.get("total_loss") is not None:
                                loss_dict["total_loss"] = (
                                    loss_dict["total_loss"] + ewc_penalty
                                )
                except SINGLE_TRAINER_EXCEPTIONS as loss_err:
                    logger.warning(
                        "Falling back to heuristic single-turn update: %s",
                        loss_err,
                    )
                    loss_dict = {
                        "policy_loss": None,
                        "total_loss": None,
                        "mean_advantage": float(np.mean(group.rewards))
                        if group.rewards
                        else 0.0,
                    }

                policy_loss = loss_dict.get("total_loss") or loss_dict.get(
                    "policy_loss"
                )
                update_applied = False

                # Backprop/update
                if self.optimizer is not None and policy_loss is not None:
                    scaled_loss = policy_loss / grad_accum_steps
                    if self.scaler is not None and use_amp:
                        self.scaler.scale(scaled_loss).backward()  # type: ignore[arg-type]
                    else:
                        scaled_loss.backward()  # type: ignore[union-attr]

                    self._grad_accum_step += 1
                    if self._grad_accum_step % grad_accum_steps == 0:
                        self._apply_optimizer_step(torch, use_amp, max_grad_norm)
                        update_applied = True

                # Progress environment once using best response
                action_turn = ConversationTurn(
                    role="assistant",
                    content=str(best_response),
                )
                done = False
                try:
                    try:
                        step_result = await self.environment.step(state, action_turn)
                    except TypeError:
                        step_result = await self.environment.step(best_response)

                    step_reward = (
                        float(np.mean(group.rewards)) if group.rewards else 0.0
                    )

                    if isinstance(step_result, tuple) and len(step_result) == 4:
                        if isinstance(step_result[1], ConversationTurn):
                            next_state, _, step_reward, done = step_result
                        else:
                            next_state, step_reward, done, _info = step_result
                        state = self._merge_scenario_into_state(next_state, scenario)
                    elif isinstance(step_result, dict):
                        next_state = step_result.get("state", state)
                        state = self._merge_scenario_into_state(next_state, scenario)
                        step_reward = step_result.get(
                            "reward",
                            float(loss_dict.get("mean_advantage", 0.0)),
                        )
                        done = bool(step_result.get("done", False))
                    else:
                        step_reward = (
                            float(np.mean(group.rewards)) if group.rewards else 0.0
                        )
                except SINGLE_TRAINER_EXCEPTIONS:
                    step_reward = (
                        float(np.mean(group.rewards)) if group.rewards else 0.0
                    )

                episode_rewards.append(float(step_reward))

                if self.optimizer is None or policy_loss is None:
                    if not update_applied:
                        self.global_step += 1

                if done:
                    break

            avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            logger.info(
                "Episode %s/%s: Steps=%s AvgReward=%.3f",
                episode + 1,
                num_episodes,
                len(episode_rewards),
                avg_reward,
            )

            if self.wandb_logger is not None:
                self.wandb_logger.log(
                    {
                        "episode": episode,
                        "episode_reward": avg_reward,
                        "episode_steps": len(episode_rewards),
                        "policy_loss": float(
                            getattr(policy_loss, "item", lambda: 0.0)()
                        ),
                    }
                )

            await notify_episode_end(
                self.callbacks,
                episode=episode,
                metrics={
                    "avg_reward": avg_reward,
                    "total_reward": avg_reward,
                    "global_step": int(self.global_step),
                    **(
                        {
                            "replay_group_count": replay_group_count,
                            "replay_buffer_size": self.continual_manager.buffer.size,
                        }
                        if self.continual_manager is not None
                        else {}
                    ),
                },
            )

        logger.info("Single-turn GRPO training completed")
        if self.optimizer is not None and self._grad_accum_step % grad_accum_steps != 0:
            self._apply_optimizer_step(torch, use_amp, max_grad_norm)

        if self.continual_manager is not None and self._current_task_id is not None:
            self.continual_manager.on_task_end(
                agent=self.agent, task_id=self._current_task_id
            )
            if self.continual_manager.reference_model is not None:
                self.reference_model = self.continual_manager.reference_model

        await notify_training_end(
            self.callbacks,
            metrics={"final_step": int(self.global_step)},
        )
        return self.agent

    async def save_checkpoint(
        self, is_best: bool = False, checkpoint_name: str | None = None
    ):
        """Save training checkpoint with HuggingFace format

        Args:
            is_best: Whether this is the best checkpoint so far
            checkpoint_name: Custom checkpoint name/path. If None, uses global_step.
        """
        checkpoint_path = resolve_checkpoint_path(
            self.config,
            self.global_step,
            is_best=is_best,
            checkpoint_name=checkpoint_name,
        )
        save_checkpoint_artifacts(
            self,
            checkpoint_path,
            SINGLE_TRAINER_EXCEPTIONS,
            logger,
        )

        logger.info(f"Checkpoint saved to {checkpoint_path}")
        await notify_checkpoint_saved(
            self.callbacks,
            path=str(checkpoint_path),
            step=int(self.global_step),
            is_best=bool(is_best),
        )

    def add_callback(self, callback: Any):
        """Add training callback"""
        self.callbacks.append(callback)

    def load_checkpoint(self, checkpoint_path: str | Path) -> bool:
        """Load model and training state from a checkpoint directory."""
        return load_checkpoint_artifacts(
            self,
            checkpoint_path,
            require_torch,
            SINGLE_TRAINER_EXCEPTIONS,
            logger,
        )
