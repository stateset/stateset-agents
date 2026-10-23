"""The GRPO update half shared by the multi-turn and single-turn trainers.

:class:`GRPOUpdateMixin` is ``MultiTurnGRPOTrainer.training_step``'s update
(unified RL spine, slice S8): continual-learning reference and KL beta,
autocast, the token-path inner passes over frozen old-policy log-probs
(``num_gradient_updates``) or one accumulated backward
(``gradient_accumulation_steps``), the streamed ``NativeChunkedBackward``,
the EWC regularizer, the optimizer commit and the step's metrics.
``SingleTurnGRPOTrainer`` runs the same update for each of its steps.

The trainers differ only where their hooks say:

* ``_update_modules`` -- the ``(torch, amp)`` modules (each trainer resolves
  them through its own module, which tests patch);
* ``_update_scaler`` -- the GradScaler for a step (MultiTurn: its scaler;
  SingleTurn: only when bf16/fp16 is on);
* ``_update_old_logprobs`` -- the frozen old-policy log-probs (MultiTurn
  inside its token-bank ``run_token_local`` phase);
* ``_update_baseline`` / ``_update_loss`` -- the reward-baseline snapshot and
  the loss with a backward closure. MultiTurn freezes the baseline with
  ``_frozen_reward_baseline`` and installs the closure as
  ``_loss_backward_fn`` for its ``compute_*_grpo_loss`` methods; SingleTurn
  passes both as keyword arguments (``update_global_stats=None`` after the
  first inner pass);
* ``_update_regularizer`` -- the EWC penalty (and the loss it is added to);
* ``_update_commit`` / ``_update_flush`` -- the trainer's optimizer step and
  accumulation flush;
* ``_warn_inner_updates_unavailable`` -- when trajectories carry no token
  metadata for inner passes.

This module never imports ``torch`` itself.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from ..grpo_accumulation import NativeChunkedBackward, backward_accumulated_loss
from ..metric_reduction import carry_loss_metrics


class GRPOUpdateMixin:
    """One GRPO update for a batch of trajectory groups (see module docs)."""

    config: Any
    agent: Any
    optimizer: Any
    continual_manager: Any
    reference_model: Any
    global_step: int
    _grad_accum_step: int

    # ------------------------------------------------------------------
    # host hooks
    # ------------------------------------------------------------------

    def _update_modules(self) -> tuple[Any, Any]:
        """``(torch, amp)`` as the host module resolves them."""
        raise NotImplementedError

    def _update_scaler(self, use_amp: bool) -> Any:
        """The GradScaler for this step (``None`` when unused)."""
        return getattr(self, "scaler", None)

    def _update_old_logprobs(self, trajectory_groups: list[Any]) -> list[Any] | None:
        """Frozen old-policy per-token log-probs, or ``None`` (no token data)."""
        raise NotImplementedError

    def _update_baseline(self) -> dict[str, Any]:
        """A snapshot of the running reward baseline for repeated passes."""
        raise NotImplementedError

    def _update_loss(
        self,
        trajectory_groups: list[Any],
        *,
        enhanced: bool,
        beta: float,
        old_logprobs: list[Any] | None,
        baseline: dict[str, Any] | None,
        update_stats: bool,
        backward_fn: Any,
    ) -> dict[str, Any]:
        """The loss dict; ``baseline`` ``None`` means the live baseline."""
        raise NotImplementedError

    def _update_regularizer(self, loss_dict: dict[str, Any]) -> tuple[Any, Any]:
        """``(loss to backpropagate, regularizer or None)``."""
        raise NotImplementedError

    def _update_commit(self, torch: Any) -> bool:
        """The trainer's validated optimizer step; True when committed."""
        raise NotImplementedError

    def _update_flush(self, torch: Any) -> bool:
        """Close a pending accumulation window; True when it committed."""
        raise NotImplementedError

    def _warn_inner_updates_unavailable(self) -> None:
        raise NotImplementedError

    def _get_grad_accum_steps(self) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # the update
    # ------------------------------------------------------------------

    def _grpo_update(
        self, trajectory_groups: list[Any], *, backward: bool = True
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute, backpropagate and (when due) commit one batch's update.

        Returns ``(loss_dict, metrics)``. ``backward=False`` only computes the
        loss (a trainer without an optimizer).
        """
        torch, amp = self._update_modules()

        use_amp = bool(
            getattr(self.config, "bf16", False) or getattr(self.config, "fp16", False)
        )
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        base_beta = float(getattr(self.config, "beta", 0.0))
        if self.continual_manager is not None:
            reference_model = getattr(self.continual_manager, "reference_model", None)
            if reference_model is not None:
                self.reference_model = reference_model
            base_beta = self.continual_manager.get_effective_beta(base_beta)

        use_enhanced = bool(
            base_beta > 0.0
            or getattr(self.config, "use_reference_model", False)
            or self.reference_model is not None
        )

        def autocast_ctx() -> Any:
            return (
                amp.autocast(device_type=device_type, enabled=use_amp)
                if amp is not None
                else contextlib.nullcontext()
            )

        inner = max(1, int(getattr(self.config, "num_gradient_updates", 1) or 1))
        old_logprobs = None
        if inner > 1:
            # Same precision as the inner-update losses, so the first ratio is 1.
            with autocast_ctx():
                old_logprobs = self._update_old_logprobs(trajectory_groups)
            if old_logprobs is None:
                self._warn_inner_updates_unavailable()
            elif backward:
                return self._inner_update_steps(
                    trajectory_groups,
                    inner,
                    old_logprobs,
                    use_enhanced=use_enhanced,
                    beta=base_beta,
                    autocast_ctx_factory=autocast_ctx,
                    torch=torch,
                )
            else:
                # Nothing to update: evaluate against the frozen policy once.
                self._update_flush(torch)

        grad_accum_steps = self._get_grad_accum_steps()
        scaler = self._update_scaler(use_amp)
        backward_fn = (
            NativeChunkedBackward(self, torch, steps=grad_accum_steps, scaler=scaler)
            if backward
            else None
        )
        with autocast_ctx():
            loss_dict = self._update_loss(
                trajectory_groups,
                enhanced=use_enhanced,
                beta=base_beta,
                old_logprobs=old_logprobs,
                baseline=None,
                update_stats=True,
                backward_fn=backward_fn,
            )
            loss, regularizer = self._update_regularizer(loss_dict)

        optimizer_step = False
        if backward and loss is not None:
            backward_accumulated_loss(
                self, loss, loss_dict, torch, scaler=scaler, regularizer=regularizer
            )
            self._grad_accum_step += 1
            if self._grad_accum_step % grad_accum_steps == 0:
                optimizer_step = self._update_commit(torch)

        return loss_dict, self._update_metrics(
            torch,
            loss_dict,
            optimizer_step=optimizer_step,
            optimizer_updates=int(optimizer_step),
            grad_accum_step=self._grad_accum_step,
            inner_updates=1,
        )

    def _inner_update_steps(
        self,
        trajectory_groups: list[Any],
        inner: int,
        old_logprobs: list[Any],
        *,
        use_enhanced: bool,
        beta: float,
        autocast_ctx_factory: Callable[[], Any],
        torch: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """PPO/DAPO-style mini-epochs: ``inner`` full optimizer steps against
        the frozen old-policy log-probs of this rollout batch."""
        ratio_means: list[float] = []
        # old_logprobs was frozen before this flush. The first inner loss must
        # therefore account for any policy change caused by the pending update.
        accumulation_flush_updates = int(self._update_flush(torch))
        optimizer_updates = accumulation_flush_updates
        loss_dict: dict[str, Any] = {}
        baseline = self._update_baseline()
        use_amp = bool(
            getattr(self.config, "bf16", False) or getattr(self.config, "fp16", False)
        )
        scaler = self._update_scaler(use_amp)
        for step_index in range(inner):
            with autocast_ctx_factory():
                loss_dict = self._update_loss(
                    trajectory_groups,
                    enhanced=use_enhanced,
                    beta=beta,
                    old_logprobs=old_logprobs,
                    baseline=baseline,
                    update_stats=step_index == 0,
                    backward_fn=NativeChunkedBackward(
                        self, torch, steps=1, scaler=scaler
                    ),
                )
                loss, regularizer = self._update_regularizer(loss_dict)
            backward_accumulated_loss(
                self,
                loss,
                loss_dict,
                torch,
                scaler=scaler,
                regularizer=regularizer,
                accumulation_steps=1,
            )
            optimizer_updates += int(self._update_commit(torch))
            ratio_means.append(float(loss_dict.get("ratio_mean", 1.0)))

        return loss_dict, self._update_metrics(
            torch,
            loss_dict,
            optimizer_step=optimizer_updates > 0,
            optimizer_updates=optimizer_updates,
            accumulation_flush_updates=accumulation_flush_updates,
            grad_accum_step=self._grad_accum_step,
            inner_updates=inner,
            ratio_mean_last=ratio_means[-1] if ratio_means else 1.0,
        )

    def _update_metrics(
        self, torch: Any, loss_dict: dict[str, Any], **step: Any
    ) -> dict[str, Any]:
        """The step's loss scalars plus its optimizer bookkeeping."""
        optimizer = self.optimizer
        metrics = {
            **{k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()},
            "learning_rate": (
                optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
            ),
            "global_step": self.global_step,
            **step,
        }
        result: dict[str, Any] = carry_loss_metrics(metrics, loss_dict)
        return result


__all__ = ["GRPOUpdateMixin"]
