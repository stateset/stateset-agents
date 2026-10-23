"""Validated optimizer commits shared by every native RL trainer.

:class:`OptimizerStep` generalises the GRPO trainers' ``safe_optimizer_step``
(``trainer_utils.safe_optimizer_step`` now delegates here) so the dedicated
GSPO/DAPO/GEPO/VAPO trainers commit updates under the same rules:

* explicit optimizers (one or several), LR schedulers, GradScaler and clip
  groups instead of reading fixed trainer attributes;
* an explicit step-counting rule (:data:`StepCounter`), because
  ``global_step`` means different things across trainers and that difference
  is preserved on purpose: ``"commits"`` counts committed optimizer updates
  (MultiTurn/SingleTurn GRPO), ``"train_steps"`` counts ``train_step`` calls
  (DAPO/GEPO/VAPO; see :meth:`OptimizerStep.end_train_step`) and ``"none"``
  keeps no counter (GSPO);
* a non-finite-gradient policy (:data:`NonfiniteGradPolicy`): ``"raise"``
  fails fast, ``"skip"`` logs which parameters were non-finite, discards that
  update and aborts after ``max_consecutive_nonfinite_skips`` consecutive
  skips. Either way a non-finite gradient never reaches ``optimizer.step()``.

Gradient clipping goes through ``torch.nn.utils.clip_grad_norm_`` looked up
at call time with ``error_if_nonfinite=True``; its return value is never used
(callers and tests may substitute a no-op clip).

This module never imports ``torch`` itself; the caller passes the module.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, get_args

logger = logging.getLogger(__name__)

StepCounter = Literal["commits", "train_steps", "none"]
NonfiniteGradPolicy = Literal["raise", "skip"]

#: Accepted :data:`StepCounter` values.
STEP_COUNTERS: tuple[str, ...] = get_args(StepCounter)
#: Accepted :data:`NonfiniteGradPolicy` values.
NONFINITE_GRAD_POLICIES: tuple[str, ...] = get_args(NonfiniteGradPolicy)


def nonfinite_grad_settings(config: Any) -> dict[str, Any]:
    """:class:`OptimizerStep` keyword arguments for a trainer config's policy.

    Reads ``nonfinite_grad_policy`` and ``max_consecutive_nonfinite_skips``
    (``TrainingConfig`` fields; their defaults when a config lacks them).
    """
    policy = getattr(config, "nonfinite_grad_policy", "raise")
    if policy not in NONFINITE_GRAD_POLICIES:
        raise ValueError(
            "nonfinite_grad_policy must be one of "
            f"{NONFINITE_GRAD_POLICIES}, got {policy!r}"
        )
    return {
        "nonfinite_grad_policy": policy,
        "max_consecutive_nonfinite_skips": int(
            getattr(config, "max_consecutive_nonfinite_skips", 3)
        ),
    }


class OptimizerStep:
    """Validate, clip and commit one update across one or more optimizers.

    One call is one *pass*. In order, it:

    1. validates ``gradient_scale`` (finite, positive) and ``max_grad_norm``
       (finite, nonnegative);
    2. runs ``finalize`` (e.g. merging normalised accumulation banks); a
       True result means the gradients are already exact means, so
       ``gradient_scale`` no longer applies;
    3. discards and returns False when no optimizer parameter has a
       gradient (nothing is stepped, counted or scheduled);
    4. unscales every optimizer through an enabled GradScaler;
    5. checks every gradient is finite. With an enabled GradScaler a
       non-finite gradient is an overflow: every optimizer of the pass is
       skipped together and the scaler lowers its scale. Otherwise
       ``nonfinite_grad_policy`` decides: ``"raise"`` raises
       ``ValueError("<label> gradients must be finite")``; ``"skip"``
       discards the update and returns False;
    6. multiplies gradients by ``gradient_scale`` (partial accumulation
       windows), after unscaling and before clipping;
    7. clips each clip group to ``max_grad_norm`` (a total norm that
       overflows raises ``RuntimeError``);
    8. calls ``step()`` exactly once on every optimizer (through the scaler
       when enabled; a reduced scale afterwards is an overflow skip);
    9. commits: counts it (``counter="commits"``), clears gradients when
       ``zero_grad_after_commit``, steps every scheduler once and finally runs
       ``on_commit``. Returns True.

    Any exception discards the gradients (``discard``, default: zero every
    optimizer's gradients) and resets a GradScaler that unscaled but did not
    update, then propagates. Work already applied by ``optimizer.step`` cannot
    be rolled back.
    """

    def __init__(
        self,
        owner: Any,
        torch_module: Any,
        *,
        optimizers: Sequence[Any],
        max_grad_norm: float,
        schedulers: Sequence[Any] = (),
        scaler: Any = None,
        clip_groups: Sequence[Sequence[Any]] | None = None,
        counter: StepCounter = "none",
        nonfinite_grad_policy: NonfiniteGradPolicy = "raise",
        max_consecutive_nonfinite_skips: int = 3,
        zero_grad_after_commit: bool = False,
        finalize: Callable[[], bool] | None = None,
        discard: Callable[[], None] | None = None,
        on_commit: Callable[[], None] | None = None,
        named_modules: Sequence[Any] = (),
        label: str = "Policy",
    ) -> None:
        """
        Args:
            owner: The trainer; holds ``global_step`` and skip bookkeeping.
            torch_module: The ``torch`` module.
            optimizers: Optimizers stepped together (``None`` entries ignored).
            max_grad_norm: Clip threshold applied to each clip group.
            schedulers: LR schedulers stepped once per commit (``None``
                entries ignored).
            scaler: Optional ``GradScaler``; used only when enabled.
            clip_groups: Parameter groups clipped independently. Default: one
                group holding every optimizer parameter.
            counter: How ``owner.global_step`` advances (:data:`StepCounter`).
            nonfinite_grad_policy: ``"raise"`` or ``"skip"``.
            max_consecutive_nonfinite_skips: Consecutive skips tolerated
                before a skip raises instead.
            zero_grad_after_commit: Clear gradients after a committed step.
            finalize: Called before gradients are read; returns True when it
                produced exact gradients (``gradient_scale`` then ignored).
            discard: Drops pending gradients (and any accumulation state).
            on_commit: Called last after a committed step.
            named_modules: Modules whose ``named_parameters`` name non-finite
                gradients in the skip warning.
            label: Prefix for error messages (``"GRPO"``, ``"DAPO"``, ...).
        """
        if counter not in STEP_COUNTERS:
            raise ValueError(f"counter must be one of {STEP_COUNTERS}, got {counter!r}")
        if nonfinite_grad_policy not in NONFINITE_GRAD_POLICIES:
            raise ValueError(
                "nonfinite_grad_policy must be one of "
                f"{NONFINITE_GRAD_POLICIES}, got {nonfinite_grad_policy!r}"
            )
        self.owner = owner
        self.torch = torch_module
        self.optimizers = [
            optimizer for optimizer in optimizers if optimizer is not None
        ]
        self.schedulers = [
            scheduler for scheduler in schedulers if scheduler is not None
        ]
        self.scaler = scaler
        self.clip_groups = (
            None if clip_groups is None else [list(group) for group in clip_groups]
        )
        self.max_grad_norm = max_grad_norm
        self.counter: StepCounter = counter
        self.nonfinite_grad_policy: NonfiniteGradPolicy = nonfinite_grad_policy
        self.max_consecutive_nonfinite_skips = int(max_consecutive_nonfinite_skips)
        self.zero_grad_after_commit = bool(zero_grad_after_commit)
        self.finalize = finalize
        self.discard = discard
        self.on_commit = on_commit
        self.named_modules = list(named_modules)
        self.label = label

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def parameters(self) -> list[Any]:
        """Every optimizer parameter once, in optimizer/param-group order."""
        return list(
            {
                id(parameter): parameter
                for optimizer in self.optimizers
                for group in optimizer.param_groups
                for parameter in group["params"]
            }.values()
        )

    def __call__(self, *, gradient_scale: float = 1.0) -> bool:
        """Run one pass; return True only when the update was committed."""
        if not self.optimizers:
            return False
        torch = self.torch
        active_scaler = None
        unscaled = False
        scaler_updated = False
        previous_scale = 1.0
        try:
            correction = float(gradient_scale)
            if not math.isfinite(correction) or correction <= 0:
                raise ValueError("gradient_scale must be finite and positive")
            max_norm = float(self.max_grad_norm)
            if not math.isfinite(max_norm) or max_norm < 0:
                raise ValueError("max_grad_norm must be finite and nonnegative")
            if self.finalize is not None and self.finalize():
                correction = 1.0
            parameters = self.parameters()
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not gradients:
                self._discard()
                return False
            scaler = self.scaler
            active_scaler = (
                scaler if scaler is not None and scaler.is_enabled() else None
            )
            if active_scaler is not None:
                previous_scale = float(active_scaler.get_scale())
                if not math.isfinite(previous_scale) or previous_scale <= 0:
                    raise ValueError("AMP scale must be finite and positive")
                for optimizer in self.optimizers:
                    active_scaler.unscale_(optimizer)
                unscaled = True
                gradients = [p.grad for p in parameters if p.grad is not None]
            if not self._all_finite(gradients):
                if active_scaler is not None:
                    # unscale_ recorded these non-finite gradients, so update()
                    # lowers the scale from them. GradScaler.step skips only
                    # optimizers whose own gradients overflowed; several
                    # optimizers are skipped together by not stepping any.
                    if len(self.optimizers) == 1:
                        active_scaler.step(self.optimizers[0])
                    active_scaler.update()
                    scaler_updated = True
                    self._discard()
                    return False
                if self.nonfinite_grad_policy == "skip":
                    return self._skip_nonfinite()
                raise ValueError(f"{self.label} gradients must be finite")
            # Partial windows were accumulated as sum(loss / configured_steps).
            # Correct their mean after AMP unscaling, before clipping. Any
            # overflow introduced here must fail clipping, not masquerade as
            # an AMP skip.
            if correction != 1.0:
                with torch.no_grad():
                    for gradient in gradients:
                        gradient.mul_(correction)
            groups = [parameters] if self.clip_groups is None else self.clip_groups
            for group in groups:
                if group:
                    torch.nn.utils.clip_grad_norm_(
                        group, max_norm, error_if_nonfinite=True
                    )
            if active_scaler is not None:
                for optimizer in self.optimizers:
                    active_scaler.step(optimizer)
                active_scaler.update()
                scaler_updated = True
                if float(active_scaler.get_scale()) < previous_scale:
                    self._discard()
                    return False
            else:
                for optimizer in self.optimizers:
                    optimizer.step()
            # The model/optimizer commit occurred even if later scheduling fails.
            if self.counter == "commits":
                self.owner.global_step += 1
            if self.nonfinite_grad_policy == "skip":
                self.owner._consecutive_nonfinite = 0
            if self.zero_grad_after_commit:
                for optimizer in self.optimizers:
                    optimizer.zero_grad(set_to_none=True)
            for scheduler in self.schedulers:
                scheduler.step()
            if self.on_commit is not None:
                self.on_commit()
            return True
        except BaseException:
            self._discard()
            if active_scaler is not None and unscaled and not scaler_updated:
                try:
                    # Reset its per-optimizer phase without advancing the
                    # scale's growth tracker after a rejected clip/backward.
                    active_scaler.update(new_scale=previous_scale)
                except Exception as error:
                    logger.warning("Could not reset AMP state after failure: %s", error)
            raise

    def end_train_step(self) -> None:
        """Close one ``train_step`` call: ``counter="train_steps"`` counts it."""
        if self.counter == "train_steps":
            self.owner.global_step += 1

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _all_finite(self, gradients: Iterable[Any]) -> bool:
        # Aggregate on each device before converting to bool, avoiding one
        # host/device synchronization for every parameter tensor.
        torch = self.torch
        checks_by_device: dict[Any, list[Any]] = {}
        for gradient in gradients:
            checks_by_device.setdefault(gradient.device, []).append(
                torch.isfinite(gradient).all()
            )
        return all(
            bool(torch.stack(checks).all()) for checks in checks_by_device.values()
        )

    def _discard(self) -> None:
        if self.discard is not None:
            self.discard()
            return
        for optimizer in self.optimizers:
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception as error:
                logger.warning("Could not discard failed gradients: %s", error)

    def _nonfinite_parameter_names(self) -> list[str]:
        bad: list[str] = []
        for module in self.named_modules:
            named = getattr(module, "named_parameters", None)
            if not callable(named):
                continue
            for name, parameter in named():
                grad = parameter.grad
                if grad is not None and not bool(self.torch.isfinite(grad).all()):
                    bad.append(name)
        return bad

    def _skip_nonfinite(self) -> bool:
        """Discard a non-finite update, logging where it came from."""
        owner = self.owner
        bad = self._nonfinite_parameter_names()
        owner._nonfinite_skips = int(getattr(owner, "_nonfinite_skips", 0)) + 1
        owner._consecutive_nonfinite = (
            int(getattr(owner, "_consecutive_nonfinite", 0)) + 1
        )
        logger.warning(
            "%s: non-finite gradients at step %s (skip %d, %d consecutive): "
            "%d parameter(s), first %s",
            self.label,
            getattr(owner, "global_step", "?"),
            owner._nonfinite_skips,
            owner._consecutive_nonfinite,
            len(bad),
            bad[:8],
        )
        self._discard()
        limit = self.max_consecutive_nonfinite_skips
        if owner._consecutive_nonfinite > limit:
            raise ValueError(
                f"{owner._consecutive_nonfinite} consecutive non-finite updates "
                f"(limit {limit})"
            )
        return False


__all__ = [
    "NONFINITE_GRAD_POLICIES",
    "STEP_COUNTERS",
    "NonfiniteGradPolicy",
    "OptimizerStep",
    "StepCounter",
    "nonfinite_grad_settings",
]
