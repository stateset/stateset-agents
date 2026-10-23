"""
Single-turn GRPO trainer implementation.

This module provides the SingleTurnGRPOTrainer class for training
agents on single-turn interactions (one prompt, one response).
"""

from __future__ import annotations

import inspect
import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from stateset_agents.core.reward_context import (
    RewardUnavailableError,
    normalize_turns,
)
from stateset_agents.core.trajectory import (
    ConversationTurn,
    MultiTurnTrajectory,
    Trajectory,
    TrajectoryGroup,
)
from stateset_agents.utils.async_calls import run_sync_owned

from .callbacks import (
    notify_checkpoint_saved,
    notify_episode_end,
    notify_optimizer_step,
    notify_training_end,
)
from .config import config_value, estimate_grpo_optimizer_steps
from .continual_learning import ContinualLearningManager
from .loss_computation import (
    compute_enhanced_grpo_loss,
    compute_grpo_loss,
    compute_token_old_logprobs,
)
from .runtime_state import validate_training_config
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
from .spine.grpo_update import GRPOUpdateMixin
from .spine.loop import SpineLoopMixin
from .spine.rollouts import record_rollout_sync
from .spine.scoring import ResponseScorer
from .trainer_utils import (
    apply_dropout_setting,
    get_amp,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_torch,
    require_torch,
    require_transformers,
    safe_optimizer_step,
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


def _usable_generate_turn(agent: Any):
    """Return ``agent.generate_turn`` only when it is a real coroutine function.

    Mock agents in tests expose arbitrary attributes, so ``callable`` alone
    is not enough; the caller also checks the result is a ConversationTurn.
    """
    fn = getattr(agent, "generate_turn", None)
    return fn if fn is not None and inspect.iscoroutinefunction(fn) else None


async def _call_environment(method: Any, *argument_forms: tuple[Any, ...]) -> Any:
    """Bind legacy call shapes before invocation; never retry a method error."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return await method(*argument_forms[0])
    for arguments in argument_forms:
        try:
            signature.bind(*arguments)
        except TypeError:
            continue
        return await method(*arguments)
    raise TypeError("Environment method does not accept a supported argument form")


class SingleTurnGRPOTrainer(GRPOUpdateMixin, SpineLoopMixin):
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
        validate_training_config(config)
        self.config = config
        self.wandb_logger = wandb_logger
        self.callbacks = callbacks or []

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_metric = float("-inf")
        self.steps_without_improvement = 0
        self._grad_accum_step = 0
        self._grpo_accumulation: dict[str, Any] | None = None

        # HuggingFace components
        self.optimizer: Any | None = None
        self.lr_scheduler: Any | None = None
        self.scaler: Any | None = None

        # GRPO enhancements
        self._global_reward_mean: float = 0.0
        self._global_reward_count: int = 0
        self.reference_model: Any | None = None
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

        apply_dropout_setting(self.config, getattr(self.agent, "model", None))

        # Set seeds for reproducibility
        try:
            import random

            seed_value = config_value(self.config, "seed")
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
            fp16_enabled = config_value(self.config, "fp16")
            if bf16_enabled or fp16_enabled:
                self.scaler = amp.GradScaler("cuda")

        # Optional reference model for KL penalty
        try:
            if (
                config_value(self.config, "use_reference_model")
                and self.reference_model is None
            ):
                reference_model = deepcopy(self.agent.model).eval()
                for param in reference_model.parameters():
                    param.requires_grad = False
                self.reference_model = reference_model
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

        learning_rate = config_value(self.config, "learning_rate")
        weight_decay = config_value(self.config, "weight_decay")

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

        betas = (
            float(config_value(self.config, "adam_beta1")),
            float(config_value(self.config, "adam_beta2")),
        )
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=learning_rate, betas=betas
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
        scenarios: list[dict[str, Any]] = get_environment_scenarios(
            self.environment, self.config
        )
        return scenarios

    def _apply_task_schedule(self, scenario: dict[str, Any], episode: int) -> None:
        apply_task_schedule(self.config, scenario, episode)

    def _get_episode_scenario(self, episode: int) -> dict[str, Any] | None:
        scenario: dict[str, Any] | None = get_episode_scenario(
            self.environment, self.config, episode
        )
        return scenario

    def _resolve_task_id(self, context: dict[str, Any] | None) -> str | None:
        task_id: str | None = resolve_task_id(self.config, context)
        return task_id

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
            reference_model = getattr(self.continual_manager, "reference_model", None)
            if reference_model is not None:
                self.reference_model = reference_model
            self._current_task_id = task_id

    def _get_grad_accum_steps(self) -> int:
        steps = int(config_value(self.config, "gradient_accumulation_steps"))
        return max(1, steps)

    def _apply_optimizer_step(
        self, torch, use_amp: bool, max_grad_norm: float, *, gradient_scale: float = 1.0
    ) -> bool:
        return safe_optimizer_step(
            self,
            torch,
            max_grad_norm=max_grad_norm,
            scaler=self.scaler if use_amp else None,
            gradient_scale=gradient_scale,
        )

    def _sync_rollout_backend(self) -> None:
        """Keep an attached rollout engine on-policy after an optimizer step
        (see ``MultiTurnGRPOTrainer._sync_rollout_backend``)."""
        self.last_rollout_sync = None
        if not bool(config_value(self.config, "rollout_sync")):
            return
        sync = getattr(self.agent, "sync_rollout_backend", None)
        if callable(sync):
            # Applies config.rollout_sync_failure to a failed sync.
            record_rollout_sync(self, bool(sync()), label="GRPO")

    def _flush_accumulation(
        self, torch: Any, use_amp: bool, max_grad_norm: float
    ) -> bool:
        """Close pending accumulated work before a different update regime."""
        steps = self._get_grad_accum_steps()
        pending = self._grad_accum_step % steps
        applied = False
        if self.optimizer is not None and pending:
            applied = self._apply_optimizer_step(
                torch, use_amp, max_grad_norm, gradient_scale=steps / pending
            )
        self._grad_accum_step = 0
        return applied

    # --- GRPOUpdateMixin hooks ------------------------------------------------

    def _use_amp(self) -> bool:
        return bool(
            getattr(self.config, "bf16", False) or config_value(self.config, "fp16")
        )

    def _update_modules(self) -> tuple[Any, Any]:
        return self._get_torch_module(), get_amp()

    def _update_scaler(self, use_amp: bool) -> Any:
        return self.scaler if use_amp else None

    def _update_old_logprobs(self, trajectory_groups: list[Any]) -> list[Any] | None:
        old: list[Any] | None = compute_token_old_logprobs(
            trajectory_groups, self.config, self.agent
        )
        return old

    def _update_baseline(self) -> dict[str, Any]:
        return {
            "global_reward_mean": self._global_reward_mean,
            "global_reward_count": self._global_reward_count,
            "global_reward_sum": getattr(self, "_global_reward_sum", None),
        }

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
        kwargs: dict[str, Any] = {
            **(self._update_baseline() if baseline is None else baseline),
            "update_global_stats": self._update_global_stats if update_stats else None,
            "backward_fn": backward_fn,
            "old_logprobs": old_logprobs,
        }
        if enhanced:
            enhanced_loss: dict[str, Any] = compute_enhanced_grpo_loss(
                trajectory_groups=trajectory_groups,
                beta=beta,
                config=self.config,
                agent=self.agent,
                reference_model=self.reference_model,
                **kwargs,
            )
            return enhanced_loss
        loss: dict[str, Any] = compute_grpo_loss(
            trajectory_groups=trajectory_groups,
            config=self.config,
            agent=self.agent,
            **kwargs,
        )
        return loss

    def _update_regularizer(self, loss_dict: dict[str, Any]) -> tuple[Any, Any]:
        penalty = None
        if self.continual_manager is not None:
            penalty = self.continual_manager.compute_ewc_penalty(self.agent)
        if penalty is not None:
            loss_dict["ewc_penalty"] = penalty
            if loss_dict.get("total_loss") is not None:
                loss_dict["total_loss"] = loss_dict["total_loss"] + penalty
        loss = loss_dict.get("total_loss")
        if loss is None:
            loss = loss_dict.get("policy_loss")
        return loss, penalty

    def _update_commit(self, torch: Any) -> bool:
        max_grad_norm = config_value(self.config, "max_grad_norm")
        return bool(self._apply_optimizer_step(torch, self._use_amp(), max_grad_norm))

    def _update_flush(self, torch: Any) -> bool:
        max_grad_norm = config_value(self.config, "max_grad_norm")
        return self._flush_accumulation(torch, self._use_amp(), max_grad_norm)

    def _warn_inner_updates_unavailable(self) -> None:
        logger.warning(
            "Inner updates require exact sampled tokens; applying one compatibility update"
        )

    def _setup_scheduler(self, num_training_steps: int) -> None:
        """Set up learning rate scheduler."""
        require_transformers()
        if self.optimizer is None:
            self.lr_scheduler = None
            return
        cosine_schedule = get_cosine_schedule_with_warmup
        linear_schedule = get_linear_schedule_with_warmup
        if cosine_schedule is None or linear_schedule is None:
            raise ImportError("transformers scheduler helpers are unavailable")

        num_warmup_steps = int(
            num_training_steps * config_value(self.config, "warmup_ratio")
        )
        lr_scheduler_type = config_value(self.config, "lr_scheduler_type")

        # A failed build propagates, as in MultiTurn: training at a constant
        # rate the config did not ask for is worse than stopping.
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
            self.lr_scheduler = None

    def _update_global_stats(self, batch_mean: float, batch_size: int) -> None:
        """Update running reward statistics for global baselines."""
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

    async def _compute_reward_for_turns(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> float:
        """Score one generation with ``reward_fn`` or, without one, the environment.

        There is no stand-in score: a reward that fails, is not finite, or is
        unavailable (no ``reward_fn`` and no environment reward) raises. Only
        an outage (``RewardUnavailableError``) drops the generation from its
        group; every other error propagates.
        """
        if self.reward_fn is not None:
            # The shared single-response path (spine ResponseScorer, i.e.
            # reward_context.score_response): [user, assistant] turns and the
            # standard context, so the reply scores exactly as it would in a
            # one-turn episode.
            *prompt_turns, reply = normalize_turns(turns)
            score = await ResponseScorer(self.reward_fn).score(
                prompt_turns, reply, context
            )
        else:
            get_reward = getattr(self.environment, "get_reward", None)
            if not callable(get_reward):
                raise ValueError(
                    "No reward_fn and the environment has no get_reward(); "
                    "pass reward_fn=..."
                )
            trajectory = MultiTurnTrajectory(turns=turns, total_reward=0.0)
            score = float(await get_reward(trajectory))
        if not math.isfinite(score):
            raise ValueError(f"Reward must be finite, got {score}")
        return score

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

        generate_turn = _usable_generate_turn(self.agent)
        for gen_idx in range(num_generations):
            assistant_turn = None
            if generate_turn is not None:
                # Keeps the generated token ids / log-probs on the turn so the
                # batched per-token GRPO path can be used.
                candidate = await generate_turn(prompt)
                if isinstance(candidate, ConversationTurn):
                    assistant_turn = candidate
                    response = str(candidate.content)
            if assistant_turn is None:
                response = await self.agent.generate_response(prompt)
                assistant_turn = ConversationTurn(
                    role="assistant", content=str(response)
                )
            turns = [
                ConversationTurn(role="user", content=str(prompt)),
                assistant_turn,
            ]
            try:
                reward = await self._compute_reward_for_turns(turns, context)
            except RewardUnavailableError as error:
                # Only a reward outage drops the generation; any other error
                # (a bug, bad data, a None or NaN reward) propagates.
                if getattr(self.reward_fn, "fail_on_error", False):
                    raise
                logger.warning(
                    "Dropping generation %d: its reward is unavailable: %s",
                    gen_idx,
                    error,
                )
                continue

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

        if not trajectories:
            raise RewardUnavailableError(
                f"Every one of {num_generations} generations failed to get a reward"
            )
        group_trajectories: list[Trajectory | MultiTurnTrajectory] = list(trajectories)
        group = TrajectoryGroup(
            scenario_id=f"prompt_{abs(hash(prompt))}",
            trajectories=group_trajectories,
            scenario_metadata={"prompt": prompt},
        )
        return group, best_response

    def _extract_context(self, state: Any) -> dict[str, Any] | None:
        context: dict[str, Any] | None = extract_context(state)
        return context

    def _extract_prompt(self, state: Any, context: dict[str, Any] | None) -> str:
        prompt: str = extract_prompt(state, context)
        return prompt

    def _merge_scenario_into_state(
        self,
        state: Any,
        scenario: dict[str, Any] | None,
    ) -> Any:
        """Preserve selected scenario context across dict-based environments."""
        return merge_scenario_into_state(state, scenario, self.config)

    # --- the spine episode loop (SpineLoopMixin hooks) -------------------
    # ``train()`` is the spine's: resume, the training start/end callbacks,
    # discarding pending gradients and closing tracking on every exit. A
    # single-turn episode is several environment steps, each one sampled
    # group and one GRPO update, its commits reported as spine receipts
    # (``optimizer-step`` callbacks). It keeps its own reporting: per-episode
    # W&B metrics, no evaluation and no periodic or final checkpoint.

    _training_start_message = "Starting single-turn GRPO training..."
    _training_done_message = "Single-turn GRPO training completed"

    def _max_steps_per_episode(self) -> int:
        """``max_steps_per_episode``; unset (``None``) means one turn."""
        steps = config_value(self.config, "max_steps_per_episode")
        return 1 if steps is None else int(steps)

    def _prepare_training_loop(self) -> tuple[int, list[dict[str, Any]], int]:
        """The episode count and optimizer-step budget; scenarios resolve per
        episode (:meth:`_scenario_for_episode`)."""
        num_episodes = int(config_value(self.config, "num_episodes"))
        total_updates = estimate_grpo_optimizer_steps(
            num_episodes * self._max_steps_per_episode(),
            gradient_accumulation_steps=self._get_grad_accum_steps(),
            num_gradient_updates=config_value(self.config, "num_gradient_updates"),
        )
        return num_episodes, [], total_updates

    def _scenario_for_episode(
        self, episode: int, scenarios: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        return self._get_episode_scenario(episode)

    def _get_eval_scenarios(self) -> list[dict[str, Any]]:
        return []

    def _resume_start(self, resumed: bool, num_episodes: int) -> int:
        """A checkpoint records the episode it was saved in; resume after it."""
        if not resumed:
            return 0
        return min(num_episodes, max(0, int(self.current_epoch) + 1))

    async def _finish_tracking(self, summary: dict[str, Any]) -> None:
        finish = getattr(self.wandb_logger, "finish_run", None)
        if callable(finish):
            await run_sync_owned(finish, summary)

    async def _report_committed_updates(
        self, previous_step: int, source: str, metrics: dict[str, Any] | None = None
    ) -> None:
        """Report the optimizer updates committed since ``previous_step`` as a
        spine receipt (``optimizer-step`` callbacks)."""
        receipt = self._optimizer_update_receipt(
            previous_step, source, metrics, task_id=self._current_task_id
        )
        if receipt is not None:
            await self._training_phase(
                "optimizer-step",
                notify_optimizer_step,
                self.callbacks,
                strict=self._strict_callbacks,
                step=self.global_step,
                metrics=dict(receipt),
            )

    def _record_diagnostics(self, loss_dict: dict[str, Any]) -> None:
        torch = self._get_torch_module()
        from .diagnostics import DiagnosticsMonitor

        if getattr(self, "diagnostics", None) is None:
            self.diagnostics = DiagnosticsMonitor(self.config)
        self._diagnostic_observations = getattr(self, "_diagnostic_observations", 0) + 1
        self.diagnostics(
            "step_end",
            {
                **{
                    name: (
                        value.detach().item()
                        if torch.is_tensor(value) and value.numel() == 1
                        else value
                    )
                    for name, value in loss_dict.items()
                },
                "step": self._diagnostic_observations,
            },
        )

    async def _train_turn(
        self, state: Any, scenario: dict[str, Any] | None
    ) -> tuple[Any, float, bool, int, Any]:
        """One environment step: sample a group, update, step with the best reply.

        Returns ``(next_state, step_reward, done, replay_groups, policy_loss)``.
        """
        context = self._extract_context(state)
        prompt = self._extract_prompt(state, context)
        task_id = self._resolve_task_id(context)
        self._maybe_handle_task_switch(task_id)

        group, best_response = await self._generate_trajectory_group(
            prompt=str(prompt),
            num_generations=config_value(self.config, "num_generations"),
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
                training_groups = training_groups + replay_groups

        # The GRPO update (the spine's GRPOUpdateMixin, shared with
        # MultiTurnGRPOTrainer.training_step): inner passes or one
        # accumulated backward, EWC, and the optimizer commit.
        previous_step = self.global_step
        loss_dict, metrics = self._grpo_update(
            training_groups, backward=self.optimizer is not None
        )
        policy_loss = loss_dict.get("total_loss")
        if policy_loss is None:
            policy_loss = loss_dict.get("policy_loss")
        self._record_diagnostics(loss_dict)
        await self._report_committed_updates(previous_step, "batch", metrics)

        # Progress environment once using best response
        action_turn = ConversationTurn(
            role="assistant",
            content=str(best_response),
        )
        done = False
        step_result = await _call_environment(
            self.environment.step, (state, action_turn), (best_response,)
        )

        step_reward = float(np.mean(group.rewards)) if group.rewards else 0.0

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
        return state, float(step_reward), done, len(replay_groups), policy_loss

    async def _train_episode(
        self,
        episode: int,
        scenario: dict[str, Any] | None,
        eval_scenarios: list[dict[str, Any]],
    ) -> str | None:
        """One episode: reset, up to ``max_steps_per_episode`` turns, report."""
        self.current_epoch = episode
        state = await _call_environment(self.environment.reset, (scenario,), ())
        state = self._merge_scenario_into_state(state, scenario)
        episode_rewards: list[float] = []
        replay_group_count = 0
        policy_loss: Any = None

        for _ in range(self._max_steps_per_episode()):
            state, step_reward, done, replayed, policy_loss = await self._train_turn(
                state, scenario
            )
            replay_group_count += replayed
            episode_rewards.append(step_reward)
            if done:
                break

        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        logger.info(
            "Episode %s/%s: Steps=%s AvgReward=%.3f",
            episode + 1,
            int(config_value(self.config, "num_episodes")),
            len(episode_rewards),
            avg_reward,
        )

        if self.wandb_logger is not None:
            metrics = {
                "episode": episode,
                "episode_reward": avg_reward,
                "episode_steps": len(episode_rewards),
                "policy_loss": float(getattr(policy_loss, "item", lambda: 0.0)()),
            }
            log_metrics = getattr(self.wandb_logger, "log_metrics", None)
            if callable(log_metrics):
                log_metrics(metrics, step=int(self.global_step))
            else:
                self.wandb_logger.log(metrics)

        await self._training_phase(
            "episode-end",
            notify_episode_end,
            self.callbacks,
            strict=self._strict_callbacks,
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
        return None

    async def _finalize_training(
        self, eval_scenarios: list[dict[str, Any]], boundary_stopped: bool
    ) -> None:
        """Flush the pending window, finish the current task, report the end."""
        torch = self._get_torch_module()
        previous_step = self.global_step
        self._flush_accumulation(
            torch, self._use_amp(), config_value(self.config, "max_grad_norm")
        )
        await self._report_committed_updates(previous_step, "final_flush")

        if self.continual_manager is not None and self._current_task_id is not None:
            self.continual_manager.on_task_end(
                agent=self.agent, task_id=self._current_task_id
            )
            reference_model = getattr(self.continual_manager, "reference_model", None)
            if reference_model is not None:
                self.reference_model = reference_model

        await self._training_phase(
            "training-end",
            notify_training_end,
            self.callbacks,
            strict=self._strict_callbacks,
            metrics={"final_step": int(self.global_step)},
        )

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
            strict=self._strict_callbacks,
        )

    def add_callback(self, callback: Any):
        """Add training callback"""
        self.callbacks.append(callback)

    def load_checkpoint(
        self, checkpoint_path: str | Path, *, trusted: bool = False
    ) -> bool:
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
        loaded: bool = load_checkpoint_artifacts(
            self,
            checkpoint_path,
            require_torch,
            SINGLE_TRAINER_EXCEPTIONS,
            logger,
            trusted=trusted,
        )
        # The transactional loader publishes restored weights once, after
        # committing all policy/optimizer state, and quarantines sync failures.
        return loaded
