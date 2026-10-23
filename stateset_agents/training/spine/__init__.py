"""Unified RL training spine (provisional).

Shared building blocks the native trainers are migrating onto, one reviewed
slice at a time (see ``docs/superpowers/plans/2026-10-22-unified-rl-spine.md``).
The public trainer classes keep their names, modules and signatures; the spine
is what they delegate to.

Currently provided:

* :class:`OptimizerStep` -- validate, clip and commit one optimizer update
  (one or several optimizers, LR schedulers, an optional GradScaler), with an
  explicit step-counting rule and non-finite-gradient policy.
* :class:`UpdateLoop` -- the K update passes over one rollout batch with
  frozen old-policy log-probs (chunked or single-graph), each committed
  through an :class:`OptimizerStep`; :func:`batch_shares` / :func:`share_size`
  give the token- or row-share weights.
* :class:`PromptGroupRollouts` -- the prompt/response trainers' rollout phase
  (one sampled group per prompt, built into update items);
  :class:`VLLMEngineMixin` / :func:`vllm_engine_config` -- the one vLLM
  rollout-engine setup.

Importing this package does not import ``torch``.
"""

from __future__ import annotations

from .rollouts import (
    PromptGroupRollouts,
    VLLMEngineMixin,
    vllm_dtype,
    vllm_engine_config,
)
from .update import (
    NONFINITE_GRAD_POLICIES,
    STEP_COUNTERS,
    UPDATE_MODES,
    NonfiniteGradPolicy,
    OptimizerStep,
    StepCounter,
    UpdateLoop,
    UpdateMode,
    UpdatePasses,
    batch_shares,
    nonfinite_grad_settings,
    share_size,
)

__all__ = [
    "PromptGroupRollouts",
    "VLLMEngineMixin",
    "vllm_dtype",
    "vllm_engine_config",
    "NONFINITE_GRAD_POLICIES",
    "STEP_COUNTERS",
    "UPDATE_MODES",
    "NonfiniteGradPolicy",
    "OptimizerStep",
    "StepCounter",
    "UpdateLoop",
    "UpdateMode",
    "UpdatePasses",
    "batch_shares",
    "nonfinite_grad_settings",
    "share_size",
]
