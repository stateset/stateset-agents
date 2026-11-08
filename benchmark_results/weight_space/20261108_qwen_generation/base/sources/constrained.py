"""Minimum-change linear edits with explicit activation preservation constraints.

For input rows K, desired output rows T, and protected input rows P, solve
``min ||D||_F² + locality_weight * ||H D^T||_F² / len(H)`` subject to
``K (W + D)^T + b = T`` and ``P D^T = 0``. The optional locality rows H
penalize behavioral drift beyond the hard protected span.
The solve uses float64 CPU SVDs; the returned factors are float32 on W's device.
Validation checks the rounded weight and actual linear evaluations before an
edit is returned. This is a local linear-layer contract, not a guarantee about
unseen prompts or nonlinear downstream model behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...core.errors import ConfigurationError
from .deltas import LowRankDelta

if TYPE_CHECKING:
    import torch

__all__ = ["ConstrainedEdit", "EditDiagnostics", "constrained_edit"]


@dataclass(frozen=True)
class EditDiagnostics:
    """Numerical evidence for an edit on the supplied weight and activations."""

    protected_rank: int
    available_dimensions: int
    edit_rank: int
    condition_number: float | None
    update_norm: float
    target_max_error: float
    protected_max_error: float
    protected_span_error: float
    locality_cases: int = 0
    locality_weight: float = 0.0
    locality_rms_drift: float | None = None
    locality_max_drift: float | None = None
    objective_value: float = 0.0


@dataclass(frozen=True)
class ConstrainedEdit:
    """A proposed low-rank update and its checks; no input tensors are changed."""

    delta: LowRankDelta
    diagnostics: EditDiagnostics


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or not value.is_floating_point()
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not bool(torch.isfinite(value).all())
    ):
        raise ConfigurationError(f"{name} must be a finite floating-point matrix")
    return value.detach().to(device="cpu", dtype=torch.float64).clone()


def _max_abs(value: torch.Tensor) -> float:
    return float(value.abs().max()) if value.numel() else 0.0


@dataclass(frozen=True)
class _LocalityMetric:
    """Thin representation of (I + strength * H.T H / n)^(-1/2)."""

    rows: torch.Tensor
    inverse_scales: torch.Tensor

    def whiten(self, value: torch.Tensor) -> torch.Tensor:
        coordinates = value @ self.rows.T
        # Split orthogonal and parallel parts instead of subtracting nearly one
        # from the inverse scales when the locality penalty is strong.
        return (value - coordinates @ self.rows) + (
            coordinates * self.inverse_scales
        ) @ self.rows


def _locality_metric(
    locality: torch.Tensor,
    protected_basis: torch.Tensor,
    strength: float,
    rcond: float,
) -> _LocalityMetric | None:
    import torch

    # D annihilates the protected span, so only locality outside it contributes
    # to the objective. Whitening this projected metric leaves protection intact.
    h = locality - (locality @ protected_basis.T) @ protected_basis
    h = h - (h @ protected_basis.T) @ protected_basis
    _, singular, rows = torch.linalg.svd(h, full_matrices=False)
    eigenvalues = 1 + (strength / len(locality)) * singular.square()
    inverse_scales = eigenvalues.rsqrt()
    if (
        not bool(torch.isfinite(eigenvalues).all())
        or float(inverse_scales.min()) <= rcond
    ):
        raise ConfigurationError("locality metric is numerically singular at rcond")
    # Omit exactly zero-energy rows; this also keeps an all-zero locality set
    # byte-equivalent to the unweighted solver.
    active = eigenvalues != 1
    if not bool(active.any()):
        return None
    return _LocalityMetric(rows[active], inverse_scales[active])


def constrained_edit(
    weight: torch.Tensor,
    keys: torch.Tensor,
    targets: torch.Tensor,
    *,
    protect: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    locality: torch.Tensor | None = None,
    locality_weight: float = 1.0,
    rcond: float = 1e-10,
    target_atol: float = 1e-5,
    target_rtol: float = 1e-5,
    preservation_atol: float = 1e-5,
    max_update_norm: float | None = None,
    max_locality_rms: float | None = None,
) -> ConstrainedEdit:
    """Design a batched edit that preserves a declared reference activation span.

    ``weight`` is ``[out, in]``, ``keys`` is ``[edits, in]``, ``targets`` is
    ``[edits, out]``, and optional ``protect`` is ``[references, in]``. Targets
    include the unchanged bias when supplied. Keys and references must have the
    weight's dtype; fp16, bfloat16 and float32 weights are supported. No model
    forwards, optimizer steps or random numbers are used.

    Row normalization makes rank selection independent of individual row scale.
    ``rcond`` sets the numerical rank cutoff, relative to the unprojected keys
    (so projection roundoff cannot invent capacity in a fully protected span).
    Conflicting or numerically unrealizable requests raise ConfigurationError.

    ``preservation_atol`` bounds the Frobenius norm of the realized update on
    an orthonormal basis of the retained protected span. This also bounds its
    operator norm: linear drift is at most this value times input norm within
    that span. Actual reference forward differences are checked separately,
    with tolerance ``preservation_atol * max(1, ||reference||_2)``. Neither
    guarantee extends to directions discarded by numerical rank selection.

    Optional ``locality`` rows ``[cases, in]`` add a soft drift penalty without
    weakening these hard constraints. The objective is squared weight norm plus
    ``locality_weight * mean(||locality_row @ D.T||_2²)``. Locality rows are not
    normalized: their magnitude and frequency determine their importance.
    ``locality_weight=0`` retains the original minimum-weight-norm solve.
    ``max_locality_rms`` optionally rejects the actual rounded update if its
    per-coordinate RMS drift on these rows exceeds the cap. Use a separate
    held-out set to assess generalization; this cap covers only supplied rows.
    """
    import torch
    from torch.nn import functional as F

    for name, value in (
        ("rcond", rcond),
        ("target_atol", target_atol),
        ("target_rtol", target_rtol),
        ("preservation_atol", preservation_atol),
        ("locality_weight", locality_weight),
    ):
        if not math.isfinite(value) or value < 0:
            raise ConfigurationError(f"{name} must be finite and nonnegative")
    if not 0 < rcond < 1:
        raise ConfigurationError("rcond must be strictly between zero and one")
    if max_update_norm is not None and (
        not math.isfinite(max_update_norm) or max_update_norm < 0
    ):
        raise ConfigurationError("max_update_norm must be finite and nonnegative")
    if max_locality_rms is not None and (
        not math.isfinite(max_locality_rms) or max_locality_rms < 0
    ):
        raise ConfigurationError("max_locality_rms must be finite and nonnegative")
    if max_locality_rms is not None and locality is None:
        raise ConfigurationError("max_locality_rms requires locality activations")
    w, k, t = (
        _matrix(weight, "weight"),
        _matrix(keys, "keys"),
        _matrix(targets, "targets"),
    )
    out_dim, in_dim = w.shape
    if not out_dim or not in_dim or not len(k):
        raise ConfigurationError("weight and keys must have nonempty dimensions")
    if k.shape[1] != in_dim or t.shape != (len(k), out_dim):
        raise ConfigurationError("keys/targets dimensions do not match weight")
    if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ConfigurationError("weight must be float32, float16 or bfloat16")
    if keys.dtype != weight.dtype:
        raise ConfigurationError("keys must have the same dtype as weight")
    p = w.new_empty((0, in_dim)) if protect is None else _matrix(protect, "protect")
    if p.shape[1] != in_dim or (protect is not None and protect.dtype != weight.dtype):
        raise ConfigurationError("protect must match weight's input width and dtype")
    h = None if locality is None else _matrix(locality, "locality")
    if h is not None and (
        not len(h)
        or h.shape[1] != in_dim
        or (locality is not None and locality.dtype != weight.dtype)
    ):
        raise ConfigurationError(
            "locality must be nonempty and match weight's input width and dtype"
        )
    b = w.new_zeros(out_dim)
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != out_dim or bias.dtype != weight.dtype:
            raise ConfigurationError("bias must match weight's output width and dtype")
        b = _matrix(bias[None, :], "bias")[0]

    # Work with thin row bases: no [in_dim, in_dim] projection is materialized.
    pn = p / p.norm(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
    if len(p):
        _, ps, pv = torch.linalg.svd(pn, full_matrices=False)
        protected_rank = int((ps > rcond * ps[0]).sum())
        q = pv[:protected_rank]
    else:
        protected_rank, q = 0, p
    metric = (
        _locality_metric(h, q, locality_weight, rcond)
        if h is not None and locality_weight > 0
        else None
    )
    scale = k.norm(dim=1, keepdim=True)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    kn = k / scale
    residual = (t - (k @ w.T + b)) / scale
    projected = kn - (kn @ q.T) @ q
    # Reorthogonalize to avoid magnifying residual projection error in a solve.
    projected = projected - (projected @ q.T) @ q
    if metric is not None:
        projected = metric.whiten(projected)
        projected = projected - (projected @ q.T) @ q
    u, s, v = torch.linalg.svd(projected, full_matrices=False)
    cutoff = rcond * float(
        torch.linalg.matrix_norm(kn if metric is None else metric.whiten(kn), ord=2)
    )
    rank = int((s > cutoff).sum())
    if rank:
        a = v[:rank] - (v[:rank] @ q.T) @ q
        factor_b = (residual.T @ u[:, :rank]) / s[:rank]
        if metric is not None:
            a = metric.whiten(a)
            a = a - (a @ q.T) @ q
    else:
        # A zero update is still a valid rank-one LoRA representation.
        a, factor_b = w.new_zeros((1, in_dim)), w.new_zeros((out_dim, 1))
    ideal = factor_b @ a
    if not bool(torch.isfinite(ideal).all()) or not torch.allclose(
        k @ (w + ideal).T + b, t, atol=target_atol, rtol=target_rtol
    ):
        raise ConfigurationError(
            "edit constraints are infeasible at the selected numerical rank; "
            "targets may conflict with each other or the protected span"
        )
    delta = LowRankDelta(
        B=factor_b.to(device=weight.device, dtype=torch.float32),
        A=a.to(device=weight.device, dtype=torch.float32),
    )
    with torch.no_grad(), torch.autocast(device_type=weight.device.type, enabled=False):
        edited = weight.detach() + delta.dense().to(dtype=weight.dtype)
        if not bool(torch.isfinite(edited).all()):
            raise ConfigurationError("rounded edit produces nonfinite weights")
        realized = edited.to(device="cpu", dtype=torch.float64) - w
        update_norm = float(realized.norm())
        if max_update_norm is not None and update_norm > max_update_norm:
            raise ConfigurationError(
                f"update norm {update_norm:.6g} exceeds max_update_norm "
                f"{max_update_norm:.6g}"
            )
        span_error = float((realized @ q.T).norm())
        if span_error > preservation_atol:
            raise ConfigurationError(
                "rounded edit exceeds protected span tolerance; "
                "use a higher precision weight or a better conditioned request"
            )
        device_bias = None if bias is None else bias.detach().to(weight.device)
        actual = (
            F.linear(keys.detach().to(weight.device), edited, device_bias)
            .double()
            .cpu()
        )
        before_p = F.linear(p.to(weight), weight.detach(), device_bias).double().cpu()
        after_p = F.linear(p.to(weight), edited, device_bias).double().cpu()
        locality_rms, locality_max = None, None
        objective_value = update_norm**2
        if h is not None:
            drift = (
                F.linear(h.to(weight), edited, device_bias).double()
                - F.linear(h.to(weight), weight.detach(), device_bias).double()
            ).cpu()
            if not bool(torch.isfinite(drift).all()):
                raise ConfigurationError(
                    "locality execution produces nonfinite outputs"
                )
            locality_rms = float(drift.square().mean().sqrt())
            locality_max = _max_abs(drift)
            objective_value += locality_weight * float(
                (h @ realized.T).square().sum() / len(h)
            )
            if not math.isfinite(objective_value):
                raise ConfigurationError("locality objective is nonfinite")
            if max_locality_rms is not None and locality_rms > max_locality_rms:
                raise ConfigurationError(
                    f"locality RMS drift {locality_rms:.6g} exceeds max_locality_rms "
                    f"{max_locality_rms:.6g}"
                )
    target_error = _max_abs(actual - t)
    protected_error = _max_abs(after_p - before_p)
    if not bool(torch.isfinite(actual).all()) or not torch.allclose(
        actual, t, atol=target_atol, rtol=target_rtol
    ):
        raise ConfigurationError(
            "rounded edit fails target tolerance in linear execution"
        )
    if not bool(
        torch.isfinite(before_p).all() and torch.isfinite(after_p).all()
    ) or bool(
        (
            (after_p - before_p).abs()
            > preservation_atol * p.norm(dim=1, keepdim=True).clamp_min(1)
        ).any()
    ):
        raise ConfigurationError(
            "rounded edit fails reference preservation in linear execution"
        )
    return ConstrainedEdit(
        delta=delta,
        diagnostics=EditDiagnostics(
            protected_rank=protected_rank,
            available_dimensions=in_dim - protected_rank,
            edit_rank=rank,
            condition_number=float(s[0] / s[rank - 1]) if rank else None,
            update_norm=update_norm,
            target_max_error=target_error,
            protected_max_error=protected_error,
            protected_span_error=span_error,
            locality_cases=0 if h is None else len(h),
            locality_weight=0.0 if h is None else locality_weight,
            locality_rms_drift=locality_rms,
            locality_max_drift=locality_max,
            objective_value=objective_value,
        ),
    )
