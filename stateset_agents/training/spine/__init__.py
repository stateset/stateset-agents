"""Unified RL training spine (provisional).

Shared building blocks the native trainers are migrating onto, one reviewed
slice at a time (see ``docs/superpowers/plans/2026-10-22-unified-rl-spine.md``).
The public trainer classes keep their names, modules and signatures; the spine
is what they delegate to.

Currently provided:

* :class:`OptimizerStep` -- validate, clip and commit one optimizer update
  (one or several optimizers, LR schedulers, an optional GradScaler), with an
  explicit step-counting rule and non-finite-gradient policy.

Importing this package does not import ``torch``.
"""

from __future__ import annotations

from .update import (
    NONFINITE_GRAD_POLICIES,
    STEP_COUNTERS,
    NonfiniteGradPolicy,
    OptimizerStep,
    StepCounter,
)

__all__ = [
    "NONFINITE_GRAD_POLICIES",
    "STEP_COUNTERS",
    "NonfiniteGradPolicy",
    "OptimizerStep",
    "StepCounter",
]
