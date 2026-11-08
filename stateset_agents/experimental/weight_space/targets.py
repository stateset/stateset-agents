"""Certified minimum-change targets for a bias-free linear vocabulary readout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...core.errors import ConfigurationError

if TYPE_CHECKING:
    import torch

__all__ = ["MarginTarget", "MarginDiagnostics", "minimum_margin_target"]


@dataclass(frozen=True)
class MarginDiagnostics:
    """Primal/dual checks for the float64 target, before model deployment."""

    update_norm: float
    minimum_margin: float
    max_constraint_violation: float
    primal_objective: float
    dual_objective: float
    duality_gap: float
    vocabulary_scans: int
    coordinate_sweeps: int


@dataclass(frozen=True)
class MarginTarget:
    """Owned CPU float64 tensors and a dual witness for the target calculation.

    Multipliers correspond to unit-normalized contrast rows for ``token_ids``.
    The result describes a readout-space projection, not an accepted weight edit.
    """

    target: torch.Tensor
    delta: torch.Tensor
    token_ids: tuple[int, ...]
    multipliers: torch.Tensor
    diagnostics: MarginDiagnostics


def minimum_margin_target(
    readout: torch.Tensor,
    residual: torch.Tensor,
    target_token_id: int,
    *,
    output_scale: torch.Tensor | None = None,
    margin: float = 1.0,
    feasibility_atol: float = 1e-8,
    duality_atol: float = 1e-6,
    chunk_size: int = 2048,
    max_constraints: int = 128,
    max_sweeps: int = 20000,
) -> MarginTarget:
    """Find the smallest residual change that promotes a token over every rival.

    Solve ``min ||d||²/2`` subject to
    ``((readout[t] - readout[j]) * output_scale) @ (residual + d) >= margin``
    for every ``j != t``. Readout is ``[vocabulary, hidden]``; residual and the
    optional diagonal scale are ``[hidden]``. All calculations use CPU float64.
    Chunked vocabulary scans avoid a vocabulary-sized float64 matrix allocation.

    For a bias-free head after RMSNorm, supplying the norm's learned scale
    preserves token ranking: its positive, common denominator cancels. The
    requested margin is BEFORE that denominator, not a model-logit margin.
    This equivalence does not hold for LayerNorm or a biased output head.

    A constraint-generation solve uses nonnegative dual coordinate updates;
    success requires a full vocabulary feasibility scan and an absolute
    primal/dual gap check. Tolerances apply to unit-normalized constraints and
    objective values respectively. Hitting a resource limit is not proof of
    infeasibility. Inputs are not changed; no model forwards or gradients occur.
    Rounded deployment and downstream behavior require separate validation.
    """
    import torch

    for name, value in (
        ("margin", margin),
        ("feasibility_atol", feasibility_atol),
        ("duality_atol", duality_atol),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ConfigurationError(f"{name} must be finite and positive")
    for name, value in (
        ("chunk_size", chunk_size),
        ("max_constraints", max_constraints),
        ("max_sweeps", max_sweeps),
    ):
        if type(value) is not int or value < 1:
            raise ConfigurationError(f"{name} must be a positive integer")

    def validate(tensor: torch.Tensor, ndim: int, name: str) -> None:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != ndim
            or not tensor.is_floating_point()
            or tensor.layout != torch.strided
            or tensor.device.type == "meta"
        ):
            raise ConfigurationError(f"{name} must be a floating-point {ndim}D tensor")

    validate(readout, 2, "readout")
    validate(residual, 1, "residual")
    vocab, width = readout.shape
    if vocab < 2 or width < 1 or residual.shape != (width,):
        raise ConfigurationError("readout/residual dimensions do not match")
    if type(target_token_id) is not int or not 0 <= target_token_id < vocab:
        raise ConfigurationError("target_token_id must index the vocabulary")
    x = residual.detach().to(device="cpu", dtype=torch.float64).clone()
    scale = torch.ones_like(x)
    if output_scale is not None:
        validate(output_scale, 1, "output_scale")
        if output_scale.shape != (width,):
            raise ConfigurationError("output_scale must match residual width")
        scale = output_scale.detach().to(device="cpu", dtype=torch.float64).clone()
    if not bool(torch.isfinite(x).all() and torch.isfinite(scale).all()):
        raise ConfigurationError("residual and output_scale must be finite")
    chosen = readout[target_token_id].detach().to(device="cpu", dtype=torch.float64)

    def contrast(start: int, stop: int) -> torch.Tensor:
        rows = readout[start:stop].detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(rows).all()):
            raise ConfigurationError("readout must be finite")
        return (chosen - rows) * scale

    normals = torch.empty((0, width), dtype=torch.float64)
    rhs = torch.empty(0, dtype=torch.float64)
    multipliers = torch.empty(0, dtype=torch.float64)
    delta = torch.zeros_like(x)
    selected: list[int] = []
    sweeps = 0
    for scan in range(1, max_constraints + 2):
        worst, worst_id, minimum_margin = -math.inf, -1, math.inf
        # Every scan checks every row, including rows not in the working set.
        for start in range(0, vocab, chunk_size):
            stop = min(start + chunk_size, vocab)
            c = contrast(start, stop)
            lengths = c.norm(dim=1)
            values = c @ (x + delta)
            keep = torch.arange(start, stop) != target_token_id
            if not bool(torch.isfinite(lengths).all() and torch.isfinite(values).all()):
                raise ConfigurationError("contrast calculation is nonfinite")
            zero = (lengths == 0) & keep
            if bool(zero.any()) and bool((c[zero] != 0).any()):
                raise ConfigurationError(
                    "contrast norm underflow; infeasibility is not established"
                )
            if bool(zero.any()):
                raise ConfigurationError(
                    "a duplicate readout row makes positive margin infeasible"
                )
            violations = (margin - values) / lengths.clamp_min(
                torch.finfo(torch.float64).tiny
            )
            if not bool(torch.isfinite(violations[keep]).all()):
                raise ConfigurationError("contrast calculation is nonfinite")
            violations[~keep] = -math.inf
            if bool(keep.any()):
                minimum_margin = min(minimum_margin, float(values[keep].min()))
            value, index = violations.max(dim=0)
            if float(value) > worst:
                worst, worst_id = float(value), start + int(index)
        primal = float(delta.square().sum() / 2)
        dual = float(rhs @ multipliers) - primal
        gap = primal - dual
        if (
            worst <= feasibility_atol
            and abs(gap) <= duality_atol
            and minimum_margin > 0
        ):
            return MarginTarget(
                (x + delta).clone(),
                delta.clone(),
                tuple(selected),
                multipliers.clone(),
                MarginDiagnostics(
                    float(delta.norm()),
                    minimum_margin,
                    max(0.0, worst),
                    primal,
                    dual,
                    gap,
                    scan,
                    sweeps,
                ),
            )
        if worst_id in selected or len(selected) >= max_constraints:
            raise ConfigurationError(
                "margin solve reached its constraint/tolerance limit; infeasibility is not established"
            )
        row = contrast(worst_id, worst_id + 1)[0]
        length = row.norm()
        normal = row / length
        normals = torch.cat((normals, normal[None]))
        rhs = torch.cat((rhs, (margin / length - normal @ x).reshape(1)))
        multipliers = torch.cat((multipliers, torch.zeros(1, dtype=torch.float64)))
        selected.append(worst_id)
        gram = normals @ normals.T
        converged = False
        while sweeps < max_sweeps:
            sweeps += 1
            violations = rhs - gram @ multipliers
            for i in range(len(selected)):
                step = max(-float(multipliers[i]), float(violations[i] / gram[i, i]))
                multipliers[i] += step
                violations -= step * gram[:, i]
            # Recompute to avoid accumulating incremental gradient roundoff.
            violations = rhs - gram @ multipliers
            kkt = torch.where(
                multipliers > 0, violations.abs(), violations.clamp_min(0)
            )
            local_gap = float((multipliers * violations).sum())
            if (
                float(kkt.max()) <= feasibility_atol * 0.1
                and abs(local_gap) <= duality_atol * 0.1
            ):
                converged = True
                break
        if not converged:
            raise ConfigurationError(
                "margin solve reached max_sweeps; infeasibility is not established"
            )
        delta = normals.T @ multipliers
    raise AssertionError("unreachable margin solver state")
