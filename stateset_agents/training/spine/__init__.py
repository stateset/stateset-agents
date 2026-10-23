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
* :class:`GRPOUpdateMixin` -- the GRPO update half (inner passes or an
  accumulated backward, EWC, the optimizer commit) shared by
  ``MultiTurnGRPOTrainer.training_step`` and ``SingleTurnGRPOTrainer``.
* :class:`SpineLoopMixin` -- the GRPO trainers' outer loop (``train()``,
  update receipts, reporting cadence, resume, finalization), inherited by
  ``MultiTurnGRPOTrainer`` and ``DistributedTrainer``.
* :class:`EpisodeScorer`, :class:`ResponseScorer`, :class:`CallableScorer` --
  the reward adapters every trainer's rollouts score through, built on
  ``core.reward_context`` (``score_episode`` / ``score_response``).
* :class:`PromptGroupRollouts` -- the prompt/response trainers' rollout phase
  (one sampled group per prompt, built into update items);
  :class:`VLLMEngineMixin` / :func:`vllm_engine_config` -- the one vLLM
  rollout-engine setup.

Importing this package does not import ``torch``.
"""

from __future__ import annotations

from .grpo_update import GRPOUpdateMixin
from .loop import SpineLoopMixin
from .rollouts import (
    PromptGroupRollouts,
    VLLMEngineMixin,
    vllm_dtype,
    vllm_engine_config,
)
from .scoring import CallableScorer, EpisodeScorer, ResponseScorer
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
    "GRPOUpdateMixin",
    "CallableScorer",
    "EpisodeScorer",
    "ResponseScorer",
    "SpineLoopMixin",
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
