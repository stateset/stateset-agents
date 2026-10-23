"""The episode/trajectory outer loop shared by the native GRPO trainers.

:class:`SpineLoopMixin` is ``MultiTurnGRPOTrainer``'s ``train()`` with its
update receipts, reporting cadence (callbacks, logging, evaluation, saves),
resume, task boundaries and success-only finalization -- moved here verbatim
(unified RL spine, slice S7). ``MultiTurnGRPOTrainer`` and through it
``DistributedTrainer`` inherit it unchanged.

``DistributedTrainer`` coordinates ranks by overriding the hook methods
(``_training_phase``, ``_prepare_training_loop``, ``_resume_start``,
``_generate_training_rollouts``, ``_rollouts_available``,
``_optimizer_update_receipt``, ``_episode_metrics``, ``_should_stop_training``
...). The phase names passed to ``_training_phase`` and the order in which
``train()`` reaches them are a cross-rank protocol: keep both unchanged.

The host provides the algorithm (``training_step``,
``generate_trajectories``, ``evaluate``, scheduler, checkpoints, task and
replay handling); see the declarations under ``TYPE_CHECKING``.

:func:`run_dedicated_loop` is the outer loop of the dedicated prompt/response
entry points (``train_with_gspo`` and GSPO-token, ``train_with_dapo``,
``train_with_gepo``, ``train_with_vapo``; slice S9): a fixed number of
iterations of batch -> ``train_step`` -> report -> periodic save -> optional
callbacks, then one final save. ``config.seed`` seeds Python, NumPy and torch
(:func:`seed_dedicated_run`) and the NumPy ``Generator`` the batches are drawn
from (:func:`sample_prompts`).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from stateset_agents.core.trajectory import TrajectoryGroup
from stateset_agents.utils.async_calls import run_sync_owned

from ..callbacks import (
    notify_episode_end,
    notify_evaluation_end,
    notify_optimizer_step,
    notify_training_end,
    notify_training_start,
)
from ..config import estimate_grpo_optimizer_steps
from ..runtime_state import TrainingRuntime

logger = logging.getLogger(__name__)


class SpineLoopMixin(TrainingRuntime):
    """``train()``: rollouts, ``training_step``, receipts, cadence, finalization.

    Each episode resolves its task (a task switch may flush pending
    accumulation, reported as a ``task_boundary`` update), generates rollouts,
    mixes continual-learning replay, takes a ``training_step`` and reports the
    updates it committed (``batch``). After the last episode the current task
    is finished and its pending window flushed (``final_flush``); only a
    successful exit saves a final checkpoint. ``global_step`` counts committed
    optimizer updates.
    """

    agent: Any
    callbacks: list[Any]
    wandb_logger: Any
    optimizer: Any
    continual_manager: Any
    global_step: int
    current_epoch: int
    best_eval_metric: float
    steps_without_improvement: int
    _next_episode: int
    _current_task_id: str | None
    _grad_accum_step: int
    _grpo_accumulation: dict[str, Any] | None

    if TYPE_CHECKING:

        async def training_step(
            self, trajectory_groups: list[TrajectoryGroup]
        ) -> dict[str, Any]: ...

        async def generate_trajectories(
            self, scenarios: list[dict[str, Any]], num_generations: int | None = None
        ) -> list[TrajectoryGroup]: ...

        async def evaluate(
            self, eval_scenarios: list[dict[str, Any]], num_eval_episodes: int = 10
        ) -> dict[str, float]: ...

        async def save_checkpoint(
            self, is_best: bool = False, checkpoint_name: str | None = None
        ) -> Any: ...

        def load_checkpoint(
            self, checkpoint_path: Any, *, trusted: bool = False
        ) -> bool: ...

        def _setup_scheduler(self, num_training_steps: int) -> Any: ...

        def _get_grad_accum_steps(self) -> int: ...

        def _get_training_scenarios(self) -> list[dict[str, Any]]: ...

        def _get_eval_scenarios(self) -> list[dict[str, Any]]: ...

        def _resolve_task_id(self, scenario: Any) -> str | None: ...

        def _maybe_handle_task_switch(self, task_id: str | None) -> None: ...

        def _maybe_mix_replay(
            self, new_groups: list[TrajectoryGroup], task_id: str | None
        ) -> tuple[list[TrajectoryGroup], list[TrajectoryGroup]]: ...

        def _finish_current_task(self) -> None: ...

        async def _log_training_metrics(
            self, metrics: dict[str, Any], trajectory_groups: list[TrajectoryGroup]
        ) -> None: ...

        async def _log_eval_metrics(self, eval_metrics: dict[str, float]) -> None: ...

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

    async def _train_episode(
        self,
        episode: int,
        scenario: dict[str, Any],
        eval_scenarios: list[dict[str, Any]],
    ) -> str | None:
        """One episode of :meth:`train`: task boundary, rollouts, step, report.

        Returns ``"boundary_stop"`` when early stopping fired at a task
        boundary, ``"stop"`` when it fired after the step, else ``None``.
        """
        self.current_epoch = episode
        task_id = await self._training_phase("task-id", self._resolve_task_id, scenario)
        previous_step, previous_task = self.global_step, self._current_task_id
        self._maybe_handle_task_switch(task_id)
        boundary_receipt = self._optimizer_update_receipt(
            previous_step, "task_boundary", task_id=previous_task
        )
        await self._report_optimizer_update(boundary_receipt, [], eval_scenarios)
        if boundary_receipt is not None and self._should_stop_training():
            # The outgoing task is complete; do not consolidate an
            # incoming task that never generated a training population.
            self._current_task_id = previous_task
            return "boundary_stop"

        # Generate trajectory groups
        new_groups = await self._generate_training_rollouts([scenario])

        if not self._rollouts_available(new_groups):
            logger.warning(f"No trajectory groups generated for episode {episode}")
            self._next_episode = episode + 1
            return None

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
                metrics["replay_buffer_size"] = self.continual_manager.buffer.size

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
        await self._report_optimizer_update(receipt, training_groups, eval_scenarios)

        # Early stopping check
        if self._should_stop_training():
            logger.info(f"Early stopping at step {self.global_step}")
            return "stop"
        return None

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
                outcome = await self._train_episode(
                    episode, training_scenarios[episode], eval_scenarios
                )
                if outcome == "boundary_stop":
                    boundary_stopped = True
                if outcome is not None:
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


@dataclass
class DedicatedRun:
    """What :func:`run_dedicated_loop` did."""

    #: Iterations that completed a ``train_step``.
    iterations: int = 0
    #: True when ``after_save`` asked to stop early.
    aborted: bool = False
    #: The last iteration's metrics (``None`` before the first).
    metrics: dict[str, Any] | None = None


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


_T = TypeVar("_T")


def seed_dedicated_run(seed: int | None) -> np.random.Generator:
    """Seed Python, NumPy and torch from ``seed``; return a batch generator.

    The dedicated entry points call this before loading the model (so LoRA
    and value-head initialisation repeat) and :func:`run_dedicated_loop` calls
    it again when the loop starts, so the batches and sampling do not depend
    on how much randomness model loading consumed. ``None`` seeds nothing and
    returns an unseeded generator.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        try:
            import torch
        except ImportError:  # pragma: no cover - torch-free install
            pass
        else:
            torch.manual_seed(seed)  # also seeds every CUDA device
    return np.random.default_rng(seed)


def sample_prompts(
    prompts: Sequence[_T], batch_size: int
) -> Callable[[int, np.random.Generator], list[_T] | None]:
    """A :func:`run_dedicated_loop` batch hook: ``batch_size`` distinct
    prompts per iteration (all of them when there are fewer), drawn from the
    loop's generator. No prompts ends training."""

    def batch(iteration: int, rng: np.random.Generator) -> list[_T] | None:
        size = min(batch_size, len(prompts))
        if size <= 0:
            return None
        return [prompts[int(i)] for i in rng.choice(len(prompts), size, replace=False)]

    return batch


async def run_dedicated_loop(
    *,
    iterations: int,
    batch: Callable[[int, np.random.Generator], Any],
    step: Callable[[Any], Awaitable[dict[str, Any]]],
    save: Callable[[str], Any],
    output_dir: str,
    save_steps: int,
    final_name: str,
    report: Callable[[int, dict[str, Any]], Any] | None = None,
    after_save: Callable[[int, dict[str, Any]], Any] | None = None,
    seed: int | None = None,
) -> DedicatedRun:
    """The dedicated entry points' outer loop.

    The loop first seeds the run (:func:`seed_dedicated_run`) with ``seed``.
    Each iteration ``i``: ``batch(i, rng)`` gives the step's prompts, drawing
    any randomness from the seeded generator ``rng`` (``None`` ends training
    early), ``step(prompts)`` runs one ``train_step``,
    ``report(i, metrics)`` logs it, every ``save_steps`` iterations
    ``save(output_dir/checkpoint-<i+1>)`` publishes a checkpoint, and
    ``after_save(i, metrics)`` (callbacks) may return True to abort. After the
    loop -- also after an abort -- ``save(output_dir/<final_name>)`` publishes
    the final model. An exception propagates without a final save, like
    :class:`SpineLoopMixin`'s success-only finalization. ``report`` and
    ``after_save`` may be sync or async.
    """
    run = DedicatedRun()
    rng = seed_dedicated_run(seed)
    for iteration in range(iterations):
        prompts = batch(iteration, rng)
        if prompts is None:
            break
        metrics = await step(prompts)
        run.iterations, run.metrics = iteration + 1, metrics
        if report is not None:
            await _maybe_await(report(iteration, metrics))
        if (iteration + 1) % save_steps == 0:
            save(os.path.join(output_dir, f"checkpoint-{iteration + 1}"))
        if after_save is not None and await _maybe_await(
            after_save(iteration, metrics)
        ):
            run.aborted = True
            break
    save(os.path.join(output_dir, final_name))
    return run


__all__ = [
    "DedicatedRun",
    "SpineLoopMixin",
    "run_dedicated_loop",
    "sample_prompts",
    "seed_dedicated_run",
]
