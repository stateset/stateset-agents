"""
Training callback utilities.

The codebase currently supports two callback styles:

1) Callable callbacks: ``cb(event: str, data: dict | None)`` (e.g. DiagnosticsMonitor)
2) Method callbacks: ``cb.on_episode_end(...)`` (used by some trainers/tests)

This module provides a small, consistent dispatch layer so trainers can emit
events without duplicating per-callback boilerplate.

A callback that raises one of :data:`CALLBACK_EXCEPTIONS` is logged at
WARNING with its traceback and training continues, unless the dispatch is
strict (``TrainingConfig.strict_callbacks``, passed as ``strict=True``) or
the callback sets ``fail_on_error = True``; then the error propagates.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

logger = logging.getLogger(__name__)

CALLBACK_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    OSError,
)

TRAINING_START_EVENT = "training_start"
TRAINING_END_EVENT = "training_end"
EPISODE_END_EVENT = "episode_end"
STEP_END_EVENT = "step_end"
OPTIMIZER_STEP_EVENT = "optimizer_step"
EVAL_END_EVENT = "evaluation_end"
CHECKPOINT_SAVED_EVENT = "checkpoint_saved"


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


async def _call(func: Any, args: tuple[Any, ...]) -> Any:
    return await _maybe_await(func(*args))


async def _try_call_variants(func: Any, variants: Iterable[tuple[Any, ...]]) -> None:
    """Bind arguments before invoking a callback, never retrying its body.

    A TypeError raised inside a callback is a callback failure, not evidence
    that another argument variant is safe to execute. Opaque callables without
    an inspectable signature receive the canonical (first) variant once.
    """
    variants_list = list(variants)
    if not variants_list:
        return
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        await _call(func, variants_list[0])
        return
    for args in variants_list:
        try:
            signature.bind(*args)
        except TypeError:
            continue
        await _call(func, args)
        return
    # Surface a useful signature error without executing an incompatible body.
    signature.bind(*variants_list[0])


def _fails_loudly(callback: Any, strict: bool) -> bool:
    """Whether a failure of ``callback`` stops training: ``strict`` dispatch
    (``TrainingConfig.strict_callbacks``) or the callback's own
    ``fail_on_error``."""
    return strict or getattr(callback, "fail_on_error", False) is True


async def _guarded(
    callback: Any,
    hook: str,
    func: Any,
    variants: Iterable[tuple[Any, ...]],
    *,
    strict: bool,
) -> bool:
    """Call ``func`` with the first fitting argument variant.

    A failure (one of :data:`CALLBACK_EXCEPTIONS`) is re-raised when it must
    stop training (:func:`_fails_loudly`); otherwise it is logged at WARNING
    with its traceback and dispatch continues. Returns False after a
    tolerated failure.
    """
    try:
        await _try_call_variants(func, variants)
    except CALLBACK_EXCEPTIONS:
        if _fails_loudly(callback, strict):
            raise
        logger.warning(
            "Training callback %r failed in %s; training continues "
            "(set strict_callbacks=True to stop on callback errors)",
            callback,
            hook,
            exc_info=True,
        )
        return False
    return True


async def _dispatch_callable(
    callback: Any, event: str, data: dict[str, Any], *, strict: bool = False
) -> None:
    if not callable(callback):
        return
    await _guarded(
        callback,
        event,
        callback,
        ((event, data), (event,), (data,), ()),
        strict=strict,
    )


async def _notify(
    callbacks: Iterable[Any],
    methods: tuple[str, ...],
    variants: tuple[tuple[Any, ...], ...],
    event: str,
    payload: dict[str, Any],
    *,
    strict: bool,
) -> None:
    """Dispatch one event: each callback's first defined ``methods`` hook,
    then the callback itself when it is callable (unless the hook failed)."""
    for callback in callbacks:
        hook, method = methods[0], None
        for hook in methods:
            method = getattr(callback, hook, None)
            if method:
                break
        if callable(method) and not await _guarded(
            callback, hook, method, variants, strict=strict
        ):
            continue
        await _dispatch_callable(callback, event, payload, strict=strict)


async def notify_training_start(
    callbacks: Iterable[Any],
    *,
    trainer: Any,
    config: Any,
    strict: bool = False,
) -> None:
    """Emit a training start signal to callbacks."""
    await _notify(
        callbacks,
        ("on_train_start", "on_training_start"),
        ((trainer, config), (config,), ()),
        TRAINING_START_EVENT,
        {"trainer": trainer, "config": config},
        strict=strict,
    )


async def notify_episode_end(
    callbacks: Iterable[Any],
    *,
    episode: int,
    metrics: dict[str, Any],
    strict: bool = False,
) -> None:
    """Emit an episode end signal to callbacks."""
    await _notify(
        callbacks,
        ("on_episode_end",),
        ((episode, metrics), (metrics,), ()),
        EPISODE_END_EVENT,
        {"episode": episode, **metrics},
        strict=strict,
    )


async def notify_step_end(
    callbacks: Iterable[Any],
    *,
    step: int,
    metrics: dict[str, Any],
    strict: bool = False,
) -> None:
    """Emit a training step end signal to callbacks."""
    await _notify(
        callbacks,
        ("on_step_end",),
        ((step, metrics), (metrics,), ()),
        STEP_END_EVENT,
        {"step": step, **metrics},
        strict=strict,
    )


async def notify_optimizer_step(
    callbacks: Iterable[Any],
    *,
    step: int,
    metrics: dict[str, Any],
    strict: bool = False,
) -> None:
    """Report one receipt for committed updates, including accumulation flushes.

    A receipt may cover several inner updates. It is not a new rollout or
    policy-signal observation; diagnostic monitors ignore this event.
    """
    await _notify(
        callbacks,
        ("on_optimizer_step",),
        ((step, metrics), (metrics,), ()),
        OPTIMIZER_STEP_EVENT,
        {"step": step, **metrics},
        strict=strict,
    )


async def notify_evaluation_end(
    callbacks: Iterable[Any],
    *,
    metrics: dict[str, Any],
    strict: bool = False,
) -> None:
    """Emit an evaluation end signal to callbacks."""
    await _notify(
        callbacks,
        ("on_evaluation_end", "on_eval_end"),
        ((metrics,), ()),
        EVAL_END_EVENT,
        dict(metrics),
        strict=strict,
    )


async def notify_training_end(
    callbacks: Iterable[Any],
    *,
    metrics: dict[str, Any],
    strict: bool = False,
) -> None:
    """Emit a training end signal to callbacks."""
    await _notify(
        callbacks,
        ("on_train_end", "on_training_end"),
        ((metrics,), ()),
        TRAINING_END_EVENT,
        dict(metrics),
        strict=strict,
    )


async def notify_checkpoint_saved(
    callbacks: Iterable[Any],
    *,
    path: str,
    step: int,
    is_best: bool,
    strict: bool = False,
) -> None:
    """Emit a checkpoint saved signal to callbacks."""
    payload = {"path": path, "step": step, "is_best": is_best}
    await _notify(
        callbacks,
        ("on_checkpoint_saved",),
        ((path, step, is_best), (payload,), ()),
        CHECKPOINT_SAVED_EVENT,
        payload,
        strict=strict,
    )


def has_no_policy_signal(metrics: Mapping[str, Any]) -> bool:
    """Identify measured zero advantages, without treating missing data as zero.

    Native GSPO reports the fraction of computed advantages that are nonzero.
    Prefer that evidence over pooled reward statistics, which can hide constant
    rewards within each group. Legacy callers must supply both a zero reward
    mean and zero standard deviation. Unknown or invalid evidence returns False.
    This describes the policy advantage term, not KL or optimizer-state effects.
    """

    def finite_number(value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, Real):
            return False
        try:
            return math.isfinite(value)
        except (OverflowError, ValueError):
            return False

    if "nonzero_advantage_fraction" in metrics:
        fraction = metrics["nonzero_advantage_fraction"]
        return finite_number(fraction) and fraction == 0.0

    mean = metrics.get("average_reward", metrics.get("mean_reward"))
    std = metrics.get("reward_std")
    return finite_number(mean) and finite_number(std) and mean == 0.0 and std == 0.0


class ZeroSignalGuard:
    """Request abort after consecutive steps with no policy advantage signal.

    Trainers that honor ``should_abort`` (native GSPO) stop and retain the reason.
    Missing or invalid evidence breaks the streak; an abort remains latched.
    """

    def __init__(self, max_zero_steps: int = 5) -> None:
        if type(max_zero_steps) is not int or max_zero_steps < 1:
            raise ValueError("max_zero_steps must be a positive integer")
        self.max_zero_steps = max_zero_steps
        self.zero_steps = 0
        self.should_abort = False
        self.abort_reason: str | None = None

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        if has_no_policy_signal(metrics):
            self.zero_steps += 1
        else:
            self.zero_steps = 0
        if self.zero_steps >= self.max_zero_steps and not self.should_abort:
            self.should_abort = True
            self.abort_reason = (
                f"policy advantages identically zero for {self.zero_steps} "
                f"consecutive steps (through step {step}): no learning signal "
                "from policy advantages; check prompts, reward context, rollout "
                "diversity and task difficulty"
            )
