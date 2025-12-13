"""
Training callback utilities.

The codebase currently supports two callback styles:

1) Callable callbacks: ``cb(event: str, data: dict | None)`` (e.g. DiagnosticsMonitor)
2) Method callbacks: ``cb.on_episode_end(...)`` (used by some trainers/tests)

This module provides a small, consistent dispatch layer so trainers can emit
events without duplicating per-callback boilerplate.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

TRAINING_START_EVENT = "training_start"
TRAINING_END_EVENT = "training_end"
EPISODE_END_EVENT = "episode_end"
STEP_END_EVENT = "step_end"
EVAL_END_EVENT = "evaluation_end"
CHECKPOINT_SAVED_EVENT = "checkpoint_saved"


async def _maybe_await(result: Any) -> Any:
    if asyncio.iscoroutine(result):
        return await result
    return result


async def _call(func: Any, args: Tuple[Any, ...]) -> Any:
    if asyncio.iscoroutinefunction(func):
        return await func(*args)
    return await _maybe_await(func(*args))


async def _try_call_variants(func: Any, variants: Iterable[Tuple[Any, ...]]) -> None:
    variants_list = list(variants)
    for args in variants_list:
        try:
            await _call(func, args)
            return
        except TypeError:
            continue

    # Best-effort final attempt with the first variant to surface a useful error
    if variants_list:
        await _call(func, variants_list[0])


async def _dispatch_callable(callback: Any, event: str, data: Dict[str, Any]) -> None:
    if not callable(callback):
        return

    try:
        await _try_call_variants(
            callback,
            variants=(
                (event, data),
                (event,),
                (data,),
                tuple(),
            ),
        )
    except Exception as exc:  # pragma: no cover - callbacks are best-effort
        logger.debug("Callback %r failed for event %s: %s", callback, event, exc)


async def notify_training_start(
    callbacks: Iterable[Any],
    *,
    trainer: Any,
    config: Any,
) -> None:
    """Emit a training start signal to callbacks."""
    payload = {"trainer": trainer, "config": config}
    for callback in callbacks:
        method = getattr(callback, "on_train_start", None) or getattr(
            callback, "on_training_start", None
        )
        if callable(method):
            try:
                await _try_call_variants(method, variants=((trainer, config), (config,), tuple()))
            except Exception as exc:  # pragma: no cover
                logger.debug("Callback %r on_train_start failed: %s", callback, exc)
                continue
        await _dispatch_callable(callback, TRAINING_START_EVENT, payload)


async def notify_episode_end(
    callbacks: Iterable[Any],
    *,
    episode: int,
    metrics: Dict[str, Any],
) -> None:
    """Emit an episode end signal to callbacks."""
    payload = {"episode": episode, **metrics}
    for callback in callbacks:
        method = getattr(callback, "on_episode_end", None)
        if callable(method):
            try:
                await _try_call_variants(method, variants=((episode, metrics), (metrics,), tuple()))
            except Exception as exc:  # pragma: no cover
                logger.debug("Callback %r on_episode_end failed: %s", callback, exc)
                continue
        await _dispatch_callable(callback, EPISODE_END_EVENT, payload)


async def notify_step_end(
    callbacks: Iterable[Any],
    *,
    step: int,
    metrics: Dict[str, Any],
) -> None:
    """Emit a training step end signal to callbacks."""
    payload = {"step": step, **metrics}
    for callback in callbacks:
        method = getattr(callback, "on_step_end", None)
        if callable(method):
            try:
                await _try_call_variants(method, variants=((step, metrics), (metrics,), tuple()))
            except Exception as exc:  # pragma: no cover
                logger.debug("Callback %r on_step_end failed: %s", callback, exc)
                continue
        await _dispatch_callable(callback, STEP_END_EVENT, payload)


async def notify_evaluation_end(
    callbacks: Iterable[Any],
    *,
    metrics: Dict[str, Any],
) -> None:
    """Emit an evaluation end signal to callbacks."""
    payload = dict(metrics)
    for callback in callbacks:
        method = getattr(callback, "on_evaluation_end", None) or getattr(
            callback, "on_eval_end", None
        )
        if callable(method):
            try:
                await _try_call_variants(method, variants=((metrics,), tuple()))
            except Exception as exc:  # pragma: no cover
                logger.debug("Callback %r on_evaluation_end failed: %s", callback, exc)
                continue
        await _dispatch_callable(callback, EVAL_END_EVENT, payload)


async def notify_training_end(
    callbacks: Iterable[Any],
    *,
    metrics: Dict[str, Any],
) -> None:
    """Emit a training end signal to callbacks."""
    payload = dict(metrics)
    for callback in callbacks:
        method = getattr(callback, "on_train_end", None) or getattr(
            callback, "on_training_end", None
        )
        if callable(method):
            try:
                await _try_call_variants(method, variants=((metrics,), tuple()))
            except Exception as exc:  # pragma: no cover
                logger.debug("Callback %r on_train_end failed: %s", callback, exc)
                continue
        await _dispatch_callable(callback, TRAINING_END_EVENT, payload)


async def notify_checkpoint_saved(
    callbacks: Iterable[Any],
    *,
    path: str,
    step: int,
    is_best: bool,
) -> None:
    """Emit a checkpoint saved signal to callbacks."""
    payload = {"path": path, "step": step, "is_best": is_best}
    for callback in callbacks:
        method = getattr(callback, "on_checkpoint_saved", None)
        if callable(method):
            try:
                await _try_call_variants(
                    method,
                    variants=((path, step, is_best), (payload,), tuple()),
                )
            except Exception as exc:  # pragma: no cover
                logger.debug("Callback %r on_checkpoint_saved failed: %s", callback, exc)
                continue
        await _dispatch_callable(callback, CHECKPOINT_SAVED_EVENT, payload)
