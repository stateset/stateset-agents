"""Rollout sources shared by the prompt/response trainers.

:class:`PromptGroupRollouts` is the rollout phase of DAPO, GEPO and VAPO: one
sampled group per prompt, turned by the trainer's ``build`` callback (scoring,
advantages, frozen old-policy log-probs) into the items an
:class:`~stateset_agents.training.spine.update.UpdateLoop` runs over.
Generation itself stays the trainer's overridable ``generate_group_responses``
(HF: ``trainer_runtime.hf_generate_group``).

:class:`VLLMEngineMixin` and :func:`vllm_engine_config` are the one vLLM
rollout-engine setup used by DAPO and the GSPO/base trajectory generators.

This module never imports ``torch``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, TypeVar

from stateset_agents.core.errors import TrainingError

from ..config import config_value

logger = logging.getLogger(__name__)

_Item = TypeVar("_Item")

#: Engine failures a weight sync survives (the engine is marked stale and
#: :func:`record_rollout_sync` applies ``config.rollout_sync_failure``): the
#: engine's runtime and I/O errors (CUDA, RPC, timeouts). TypeError,
#: KeyError, AttributeError and ValueError are bugs in the sync path (a
#: wrong call, a weight-name or layout mismatch) and propagate.
_WEIGHT_SYNC_EXCEPTIONS: tuple[type[BaseException], ...] = (RuntimeError, OSError)


class RolloutSyncError(TrainingError, RuntimeError):
    """The rollout engine could not be kept on-policy: weight syncs failed
    ``max_consecutive_rollout_sync_failures`` times in a row under
    ``rollout_sync_failure="raise"``. A :class:`TrainingError` that is still a
    ``RuntimeError`` and prints its bare message."""

    message_only_str = True


def record_rollout_sync(trainer: Any, outcome: bool | None, *, label: str) -> None:
    """Record one weight-sync outcome on ``trainer`` and apply its policy.

    ``outcome`` is True after a successful sync, False after a failed one
    (the engine keeps serving the previous policy) and ``None`` when nothing
    was attempted. It becomes ``trainer.last_rollout_sync``; a success resets
    the consecutive-failure count, a failure increments it, and under
    ``config.rollout_sync_failure="raise"`` (the default) the
    ``max_consecutive_rollout_sync_failures``-th consecutive failure raises
    :class:`RolloutSyncError`. ``"warn"`` keeps training on stale rollouts.
    """
    trainer.last_rollout_sync = outcome
    if outcome is None:
        return
    if outcome:
        trainer._rollout_sync_failures = 0
        return
    failures = int(getattr(trainer, "_rollout_sync_failures", 0)) + 1
    trainer._rollout_sync_failures = failures
    policy = config_value(trainer.config, "rollout_sync_failure")
    limit = int(config_value(trainer.config, "max_consecutive_rollout_sync_failures"))
    if policy == "raise" and failures >= limit:
        raise RolloutSyncError(
            f"{label}: rollout-engine weight sync failed {failures} consecutive "
            f"times (limit {limit}), so rollouts would keep sampling a stale "
            "policy. Fix the engine, or set rollout_sync_failure='warn' to "
            "keep training on stale rollouts."
        )
    logger.warning(
        "%s: rollout-engine weights are stale (%d consecutive sync failure(s))",
        label,
        failures,
    )


def rollout_sync_metrics(trainer: Any) -> dict[str, Any]:
    """The step metrics for the rollout engine's weights: whether they are
    stale (the last sync failed) and the consecutive-failure count. Empty
    until a sync was attempted, so trainers without an engine report
    nothing new."""
    outcome = getattr(trainer, "last_rollout_sync", None)
    failures = int(getattr(trainer, "_rollout_sync_failures", 0))
    if outcome is None and not failures:
        return {}
    return {
        "rollout_weights_stale": outcome is False,
        "rollout_sync_failures": failures,
    }


class RewardFallbacks:
    """Count the replies (or episodes) left out of a step for a reward
    fallback (``ScoredReply.fallback``): a flagged score is never trained on.

    :meth:`keep` filters one group; :meth:`require_trainable` raises
    ``RewardUnavailableError`` when every scored reply of the step was
    flagged (training on nothing would silently stall); :meth:`metrics` is
    the step's ``reward_fallback_responses`` count, reported only when
    nonzero.
    """

    def __init__(self) -> None:
        self.scored = 0
        self.flagged = 0

    def keep(
        self, items: Sequence[_Item], replies: Sequence[Any]
    ) -> tuple[list[_Item], list[float]]:
        """The items whose reply score is genuine, with those scores."""
        kept: list[_Item] = []
        scores: list[float] = []
        for item, reply in zip(items, replies, strict=True):
            self.scored += 1
            if reply.fallback is not None:
                self.flagged += 1
                logger.warning(
                    "Leaving a reply out of training: its reward is a fallback " "(%s)",
                    reply.fallback,
                )
                continue
            kept.append(item)
            scores.append(float(reply.score))
        return kept, scores

    def require_trainable(self, label: str) -> None:
        """Raise when every scored reply of the step was flagged."""
        if self.scored and self.flagged == self.scored:
            from stateset_agents.core.reward_context import RewardUnavailableError

            raise RewardUnavailableError(
                f"{label}: all {self.scored} scored replies carry a reward "
                "fallback flag, so nothing in this step can be trained on"
            )

    def metrics(self) -> dict[str, Any]:
        return {"reward_fallback_responses": self.flagged} if self.flagged else {}


def exclude_flagged_groups(groups: Sequence[Any]) -> tuple[list[Any], int]:
    """Trajectory groups without their reward-fallback episodes, and how many
    episodes were left out.

    A flagged episode (``spine.scoring.trajectory_reward_fallback``) is
    dropped from its group; the group's advantages are then relative to the
    genuine rewards only. A group left empty is dropped. The input groups are
    not modified (a group losing episodes is copied).
    """
    import copy
    import dataclasses

    from .scoring import trajectory_reward_fallback

    kept: list[Any] = []
    excluded = 0
    for group in groups:
        trajectories = getattr(group, "trajectories", None)
        if not trajectories:
            kept.append(group)
            continue
        clean = [t for t in trajectories if trajectory_reward_fallback(t) is None]
        if len(clean) == len(trajectories):
            kept.append(group)
            continue
        excluded += len(trajectories) - len(clean)
        if not clean:
            continue
        if dataclasses.is_dataclass(group) and not isinstance(group, type):
            kept.append(dataclasses.replace(group, trajectories=clean))
        else:
            copied = copy.copy(group)
            copied.trajectories = clean
            kept.append(copied)
    return kept, excluded


class PromptGroupRollouts:
    """Sample one response group per prompt and build an update item from it.

    ``generate(prompt)`` returns the group (the trainer's own
    ``generate_group_responses``, so tests and subclasses can replace it).
    Groups smaller than ``min_group_size`` are skipped with a warning (a
    group-relative advantage needs at least two samples). ``build(prompt,
    responses)`` scores the group and returns its item, or ``None`` to drop it
    (e.g. DAPO's dynamic sampling). :meth:`collect` stops once ``limit`` items
    were built. ``build`` scores through :meth:`score_group`, which leaves
    reward-fallback replies out (:attr:`fallbacks`); a step whose every
    scored reply was flagged raises ``RewardUnavailableError``.
    """

    def __init__(
        self,
        generate: Callable[[str], Awaitable[Sequence[Any]]],
        *,
        min_group_size: int = 1,
        label: str = "rollout",
    ) -> None:
        self.generate = generate
        self.min_group_size = int(min_group_size)
        self.label = label
        #: Prompts sampled by the last :meth:`collect`.
        self.prompts_processed = 0
        #: Groups the last :meth:`collect` skipped (too small or dropped).
        self.groups_dropped = 0
        #: Replies the last :meth:`collect` left out for a reward fallback.
        self.fallbacks = RewardFallbacks()

    async def score_group(
        self,
        responses: Sequence[_Item],
        score: Callable[[_Item], Awaitable[Any]],
    ) -> tuple[list[_Item], list[float]]:
        """Score each reply (``score`` returns a ``ScoredReply``) in order and
        keep the genuine ones: ``(kept replies, their scores)``."""
        replies = [await score(response) for response in responses]
        return self.fallbacks.keep(responses, replies)

    async def collect(
        self,
        prompts: Iterable[str],
        build: Callable[[str, list[Any]], Awaitable[_Item | None]],
        *,
        limit: int | None = None,
    ) -> list[_Item]:
        """Return the built items, in prompt order."""
        self.prompts_processed = 0
        self.groups_dropped = 0
        self.fallbacks = RewardFallbacks()
        items: list[_Item] = []
        for prompt in prompts:
            if limit is not None and len(items) >= limit:
                break
            responses = list(await self.generate(prompt))
            self.prompts_processed += 1
            if len(responses) < self.min_group_size:
                logger.warning(
                    "%s: %d response(s) for a prompt, need %d; skipping it",
                    self.label,
                    len(responses),
                    self.min_group_size,
                )
                self.groups_dropped += 1
                continue
            item = await build(prompt, responses)
            if item is None:
                self.groups_dropped += 1
            else:
                items.append(item)
        self.fallbacks.require_trainable(self.label)
        return items


def vllm_dtype(config: Any) -> str:
    """vLLM engine dtype for a training config: fp16, bf16, else checkpoint."""
    if config_value(config, "fp16"):
        return "float16"
    if getattr(config, "bf16", False):
        return "bfloat16"
    return "auto"


def vllm_engine_config(config: Any, config_cls: Any) -> Any:
    """Build a ``VLLMConfig`` (``config_cls``) from a trainer config.

    Every ``vllm_*`` training field is honoured; absent fields fall back to
    the ``TrainingConfig`` defaults.
    """
    return config_cls(
        model_name=config.model_name,
        revision=config_value(config, "model_revision"),
        gpu_memory_utilization=config_value(config, "vllm_gpu_memory_utilization"),
        tensor_parallel_size=config_value(config, "vllm_tensor_parallel_size"),
        enable_prefix_caching=config_value(config, "vllm_enable_prefix_caching"),
        max_model_len=config_value(config, "vllm_max_model_len"),
        quantization=config_value(config, "vllm_quantization"),
        enable_chunked_prefill=config_value(config, "vllm_enable_chunked_prefill"),
        max_tokens=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        dtype=vllm_dtype(config),
    )


class VLLMEngineMixin:
    """vLLM rollout-engine lifecycle: create, initialize, report readiness.

    The host sets ``config``, calls :meth:`_reset_vllm_state` in ``__init__``
    and, after loading its (patchable) backend symbols, calls
    :meth:`_install_vllm_generator`. ``_vllm_label`` names the host in logs;
    ``_vllm_init_exceptions`` are the initialization failures that fall back
    to HuggingFace generation.
    """

    config: Any
    vllm_generator: Any | None
    _vllm_initialized: bool
    #: True after a failed weight sync, until one succeeds.
    vllm_weights_stale: bool = False
    _vllm_label: str = "vLLM rollouts"
    _vllm_init_exceptions: tuple[type[BaseException], ...] = (
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        OSError,
    )

    def _reset_vllm_state(self) -> None:
        self.vllm_generator = None
        self._vllm_initialized = False

    def _install_vllm_generator(self, config_cls: Any, generator_cls: Any) -> None:
        """Create the engine from the loaded backend classes (lazy start)."""
        if config_cls is None or generator_cls is None:
            logger.warning(
                "%s: vLLM backend unavailable, using HuggingFace generation",
                self._vllm_label,
            )
            return
        self.vllm_generator = generator_cls(vllm_engine_config(self.config, config_cls))

    async def initialize_vllm(self) -> bool:
        """Start the vLLM engine; False (HuggingFace fallback) on failure."""
        generator = self.vllm_generator
        if generator is None:
            return False
        if self._vllm_initialized:
            return True
        try:
            success = bool(await generator.initialize())
        except self._vllm_init_exceptions as error:
            logger.warning(
                "%s: failed to initialize vLLM (%s); using HuggingFace generation",
                self._vllm_label,
                error,
            )
            self._vllm_initialized = False
            return False
        self._vllm_initialized = success
        if success:
            logger.info("%s: vLLM engine initialized", self._vllm_label)
        return success

    @property
    def using_vllm(self) -> bool:
        """Whether generation goes through an initialized vLLM engine."""
        return bool(self._vllm_initialized and self.vllm_generator is not None)

    def sync_vllm_weights(self, model: Any) -> bool | None:
        """Push ``model``'s weights into the engine after a committed update.

        Returns ``None`` when no engine is in use, True after a successful
        sync, and False when the engine has no ``sync_weights`` or the sync
        failed with an engine runtime/I-O error: the engine is then stale
        (``vllm_weights_stale``) and keeps serving the previous policy; the
        caller's :func:`record_rollout_sync` decides whether training goes
        on. Any other exception is a bug in the sync path and propagates.
        """
        generator = self.vllm_generator
        if not self.using_vllm or generator is None:
            return None
        sync = getattr(generator, "sync_weights", None)
        if not callable(sync):
            self.vllm_weights_stale = True
            logger.warning(
                "%s: %s has no sync_weights(model); rollouts lag the policy",
                self._vllm_label,
                type(generator).__name__,
            )
            return False
        try:
            sync(model)
        except _WEIGHT_SYNC_EXCEPTIONS as error:
            self.vllm_weights_stale = True
            logger.warning(
                "%s: vLLM weight sync failed; rollouts lag the policy: %s",
                self._vllm_label,
                error,
            )
            return False
        self.vllm_weights_stale = False
        return True


__all__ = [
    "PromptGroupRollouts",
    "RewardFallbacks",
    "RolloutSyncError",
    "VLLMEngineMixin",
    "vllm_dtype",
    "record_rollout_sync",
    "rollout_sync_metrics",
    "vllm_engine_config",
]
