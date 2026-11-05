"""The learning-rate schedule of the dedicated trainers, over the steps they take.

Every spine commit (:class:`~stateset_agents.training.spine.update.OptimizerStep`)
steps the LR schedulers once, so a run of ``iterations`` outer iterations
with ``passes`` update passes each takes ``iterations * passes`` optimizer
steps. :func:`optimizer_step_budget` is that budget, computed once here for
every :class:`~stateset_agents.training.spine.algorithm.AlgorithmTrainer`
(GSPO, GSPO-token, DAPO, GEPO) and for VAPO. Sizing the schedule over the
outer iterations alone (as GEPO and VAPO did) made Hugging Face's cosine
reach zero halfway through a two-pass run and climb back to the full rate.

:func:`schedule_factor` is the multiplier: linear warmup, then a cosine or
linear decay to zero (``"constant"`` stays at 1 with no warmup), *clamped*
at the end so a step past the budget holds the final value instead of
following Hugging Face's periodic cosine back up. :func:`build_lr_scheduler`
wraps it in a ``LambdaLR``.

The GRPO trainers keep their own Hugging Face schedules over
``config.estimate_grpo_optimizer_steps`` (an upper bound on their commits).

This module never imports ``torch``; the caller passes the module.
"""

from __future__ import annotations

import functools
import math
from typing import Any

#: ``lr_scheduler_type`` values the dedicated trainers implement.
LR_SCHEDULE_KINDS: tuple[str, ...] = ("cosine", "linear", "constant")


def optimizer_step_budget(iterations: int, passes: int) -> int:
    """Optimizer steps of a run: ``iterations`` outer iterations times
    ``passes`` committed update passes each (at least one step).

    ``passes`` must be a positive integer; a non-positive ``iterations``
    still yields a one-iteration schedule so the scheduler is never empty.
    """
    if type(passes) is not int or passes < 1:
        raise ValueError(f"update passes must be a positive integer (got {passes!r})")
    return max(1, int(iterations)) * passes


def schedule_factor(
    step: int, *, kind: str, warmup_steps: int, total_steps: int
) -> float:
    """The LR multiplier at ``step`` of a ``total_steps`` schedule.

    Linear warmup over ``warmup_steps``, then a half-cosine (``"cosine"``)
    or straight-line (``"linear"``) decay to 0 at ``total_steps``, held at 0
    afterwards. ``"constant"`` is 1 throughout (no warmup, like Hugging
    Face's ``get_constant_schedule``). In range the values are bit-identical
    to Hugging Face's ``get_cosine_schedule_with_warmup`` and
    ``get_linear_schedule_with_warmup``.
    """
    if kind == "constant":
        return 1.0
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, progress)
    if kind == "linear":
        return max(0.0, 1.0 - progress)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_lr_scheduler(
    torch: Any,
    optimizer: Any,
    *,
    kind: str,
    total_steps: int,
    warmup_ratio: float,
) -> Any:
    """A ``LambdaLR`` over ``total_steps`` optimizer steps (:func:`schedule_factor`).

    ``kind`` is a config's ``lr_scheduler_type``; the warmup is
    ``int(total_steps * warmup_ratio)`` steps.
    """
    if kind not in LR_SCHEDULE_KINDS:
        raise ValueError(
            f"lr_scheduler_type must be one of {LR_SCHEDULE_KINDS} (got {kind!r})"
        )
    total = int(total_steps)
    if total < 1:
        raise ValueError(f"total_steps must be positive (got {total_steps!r})")
    warmup = int(total * float(warmup_ratio))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=functools.partial(
            schedule_factor, kind=kind, warmup_steps=warmup, total_steps=total
        ),
    )


__all__ = [
    "LR_SCHEDULE_KINDS",
    "build_lr_scheduler",
    "optimizer_step_budget",
    "schedule_factor",
]
