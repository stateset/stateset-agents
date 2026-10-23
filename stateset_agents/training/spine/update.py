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

:class:`UpdateLoop` runs the K update passes (``num_gradient_updates`` /
``num_iterations``) a trainer takes over one rollout batch whose old-policy
log-probs are frozen, committing each pass through an :class:`OptimizerStep`:
``"chunked"`` backpropagates each group's share-weighted loss before the next
group's forward (bounded memory), ``"single_graph"`` sums the groups into one
loss first. :func:`batch_shares` and :func:`share_size` give the token- or
row-share weights that make chunked gradients equal the batch objective's.

This module never imports ``torch`` itself; the caller passes the module.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, get_args

from ..config import config_value

logger = logging.getLogger(__name__)

StepCounter = Literal["commits", "train_steps", "none"]
NonfiniteGradPolicy = Literal["raise", "skip"]
UpdateMode = Literal["chunked", "single_graph"]

#: Accepted :data:`StepCounter` values.
STEP_COUNTERS: tuple[str, ...] = get_args(StepCounter)
#: Accepted :data:`NonfiniteGradPolicy` values.
NONFINITE_GRAD_POLICIES: tuple[str, ...] = get_args(NonfiniteGradPolicy)
#: Accepted :data:`UpdateMode` values.
UPDATE_MODES: tuple[str, ...] = get_args(UpdateMode)

_Item = TypeVar("_Item")


def nonfinite_grad_settings(config: Any) -> dict[str, Any]:
    """:class:`OptimizerStep` keyword arguments for a trainer config's policy.

    Reads ``nonfinite_grad_policy`` and ``max_consecutive_nonfinite_skips``
    (``TrainingConfig`` fields; their defaults when a config lacks them).
    """
    policy = config_value(config, "nonfinite_grad_policy")
    if policy not in NONFINITE_GRAD_POLICIES:
        raise ValueError(
            "nonfinite_grad_policy must be one of "
            f"{NONFINITE_GRAD_POLICIES}, got {policy!r}"
        )
    return {
        "nonfinite_grad_policy": policy,
        "max_consecutive_nonfinite_skips": int(
            config_value(config, "max_consecutive_nonfinite_skips")
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

    def zero_grad(self) -> None:
        """Clear every optimizer's gradients (``set_to_none``)."""
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def discard_gradients(self) -> None:
        """Drop pending gradients the way a failed pass does (``discard``)."""
        self._discard()

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


def batch_shares(sizes: Sequence[float]) -> list[float]:
    """Each entry's fraction of the total (all zeros when the total is 0)."""
    total = sum(sizes)
    return [size / total if total > 0 else 0.0 for size in sizes]


def share_size(response_mask: Any, aggregate: str) -> float:
    """One group's size in a batch-level objective.

    ``"token_mean"`` counts its action tokens (the response mask shifted by
    one for next-token prediction); any other aggregation counts its rows.
    Weighting each group's own mean by ``size / batch_total``
    (:func:`batch_shares`) makes the accumulated gradient exactly the
    batch-level objective's.
    """
    if aggregate == "token_mean":
        return float(response_mask[:, 1:].sum())
    return float(response_mask.shape[0])


@dataclass
class UpdatePasses:
    """What :meth:`UpdateLoop.run` did: one loss per pass and the commits."""

    #: Each pass's loss: the sum of the backpropagated (share-weighted) group
    #: losses in chunked mode, the reduced loss in single-graph mode.
    losses: list[float] = field(default_factory=list)
    #: Passes whose update was committed by the optimizer step.
    committed: int = 0


class UpdateLoop:
    """K update passes over one rollout batch with frozen old-policy log-probs.

    Every pass zeroes the gradients, evaluates each item's loss through
    ``loss_fn(pass_index, item_index, item)`` (the trainer freezes or reuses
    its old-policy log-probs there; ``pass_index == 0`` is the first pass) and
    commits exactly one update. After the last pass
    :meth:`OptimizerStep.end_train_step` applies the step-counting rule.

    ``mode="chunked"``: each item's loss, multiplied by its share when
    ``shares`` are given (items whose share is not positive are skipped), is
    backpropagated immediately, releasing that item's graph before the next
    item's forward. The pass then commits through ``optimizer_step``. A
    ``loss_fn`` may return ``None`` to contribute nothing.

    ``mode="single_graph"``: the item losses are added to ``initial_loss()``
    in one graph, ``reduce`` maps the sum to the pass loss (e.g. the mean over
    groups) and ``commit(loss)`` backpropagates and steps (default: zero
    gradients, ``loss.backward()``, ``optimizer_step()``). Use it when the
    trainer validates or normalises the whole batch loss before backward.

    An empty batch zeroes, backpropagates and commits nothing (stale
    gradients are never stepped). Any exception discards pending gradients
    and propagates.
    """

    def __init__(
        self,
        optimizer_step: OptimizerStep,
        *,
        passes: int,
        mode: UpdateMode = "chunked",
    ) -> None:
        if type(passes) is not int or passes < 1:
            raise ValueError(f"passes must be a positive integer, got {passes!r}")
        if mode not in UPDATE_MODES:
            raise ValueError(f"mode must be one of {UPDATE_MODES}, got {mode!r}")
        self.optimizer_step = optimizer_step
        self.passes = passes
        self.mode: UpdateMode = mode

    def run(
        self,
        items: Sequence[_Item],
        loss_fn: Callable[[int, int, _Item], Any],
        *,
        shares: Sequence[float] | None = None,
        on_pass_start: Callable[[int], None] | None = None,
        initial_loss: Callable[[], Any] | None = None,
        reduce: Callable[[Any], Any] | None = None,
        commit: Callable[[Any], bool] | None = None,
    ) -> UpdatePasses:
        """Run every pass; return the per-pass losses and the commit count."""
        if shares is not None and len(shares) != len(items):
            raise ValueError("shares must give one weight per item")
        if self.mode == "single_graph" and initial_loss is None:
            raise ValueError("single_graph mode needs initial_loss")
        result = UpdatePasses()
        for pass_index in range(self.passes):
            if on_pass_start is not None:
                on_pass_start(pass_index)
            if not items:
                result.losses.append(0.0)
                continue
            try:
                if self.mode == "chunked":
                    loss_value, committed = self._chunked_pass(
                        pass_index, items, loss_fn, shares
                    )
                else:
                    assert initial_loss is not None
                    loss_value, committed = self._single_graph_pass(
                        pass_index, items, loss_fn, initial_loss, reduce, commit
                    )
            except BaseException:
                self.optimizer_step.discard_gradients()
                raise
            result.losses.append(loss_value)
            result.committed += int(committed)
        self.optimizer_step.end_train_step()
        return result

    def _chunked_pass(
        self,
        pass_index: int,
        items: Sequence[_Item],
        loss_fn: Callable[[int, int, _Item], Any],
        shares: Sequence[float] | None,
    ) -> tuple[float, bool]:
        self.optimizer_step.zero_grad()
        pass_loss = 0.0
        for item_index, item in enumerate(items):
            share = None if shares is None else shares[item_index]
            if share is not None and share <= 0:
                continue
            loss = loss_fn(pass_index, item_index, item)
            if loss is None:
                continue
            weighted = loss if share is None else loss * share
            weighted.backward()
            pass_loss += weighted.item()
        return pass_loss, self.optimizer_step()

    def _single_graph_pass(
        self,
        pass_index: int,
        items: Sequence[_Item],
        loss_fn: Callable[[int, int, _Item], Any],
        initial_loss: Callable[[], Any],
        reduce: Callable[[Any], Any] | None,
        commit: Callable[[Any], bool] | None,
    ) -> tuple[float, bool]:
        total = initial_loss()
        for item_index, item in enumerate(items):
            total = total + loss_fn(pass_index, item_index, item)
        loss = total if reduce is None else reduce(total)
        committed = (
            self._backward_and_step(loss) if commit is None else bool(commit(loss))
        )
        return float(loss.detach().item()), committed

    def _backward_and_step(self, loss: Any) -> bool:
        self.optimizer_step.zero_grad()
        loss.backward()
        return self.optimizer_step()


__all__ = [
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
