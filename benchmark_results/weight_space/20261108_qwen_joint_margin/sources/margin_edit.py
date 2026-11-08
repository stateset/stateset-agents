"""Joint vocabulary-margin edits with hard and soft activation preservation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...core.errors import ConfigurationError
from .constrained import _locality_metric, _matrix
from .deltas import LowRankDelta
from .targets import _solve_dual

if TYPE_CHECKING:
    import torch

__all__ = ["JointMarginEdit", "JointMarginDiagnostics", "joint_margin_edit"]


@dataclass(frozen=True)
class JointMarginDiagnostics:
    """Numerical evidence; ideal objectives and rounded deployment are distinct."""

    protected_rank: int
    primal_objective: float
    dual_objective: float
    duality_gap: float
    max_constraint_violation: float
    ideal_minimum_margins: tuple[float, ...]
    deployed_minimum_margins: tuple[float, ...]
    update_norm: float
    protected_span_error: float
    protected_max_error: float
    locality_rms_drift: float | None
    vocabulary_scans: int
    coordinate_sweeps: int


@dataclass(frozen=True)
class JointMarginEdit:
    """Float32 deployable factors and owned CPU float64 certificate tensors.

    ``ideal_B @ ideal_A`` is the pre-rounding optimum. Dual multipliers use
    unit-normalized contrast/key outer products, indexed by context/token pairs.
    Neither returned factors nor certificates mutate the supplied weight.
    """

    delta: LowRankDelta
    diagnostics: JointMarginDiagnostics
    ideal_B: torch.Tensor
    ideal_A: torch.Tensor
    context_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    multipliers: torch.Tensor


def joint_margin_edit(
    weight: torch.Tensor,
    keys: torch.Tensor,
    residuals: torch.Tensor,
    readout: torch.Tensor,
    target_token_ids: Sequence[int],
    *,
    output_scale: torch.Tensor | None = None,
    protect: torch.Tensor | None = None,
    locality: torch.Tensor | None = None,
    locality_weight: float = 1.0,
    margin: float = 1.0,
    rcond: float = 1e-10,
    feasibility_atol: float = 1e-9,
    duality_atol: float = 1e-8,
    margin_atol: float = 1e-4,
    preservation_atol: float = 1e-5,
    max_update_norm: float | None = None,
    chunk_size: int = 2048,
    max_constraints: int = 256,
    max_sweeps: int = 40000,
) -> JointMarginEdit:
    """Minimize a shared weight update subject to every context's token margins.

    For ``weight[out, in]``, ``keys[cases, in]`` and captured pre-normalization
    ``residuals[cases, out]``, minimize half of
    ``||D||² + locality_weight * ||H D.T||² / len(H)`` subject to ``P D.T = 0``
    and ``c[i,j] @ (residuals[i] + D @ keys[i]) >= margin`` for every competing
    vocabulary token j, where ``c[i,j]=(readout[target[i]]-readout[j])*scale``.

    The linear model is exact only when the edited layer writes additively into
    these residuals, followed immediately by a bias-free linear head (optionally
    after RMSNorm with its learned ``output_scale``). Callers must establish this
    architecture and capture inputs on the same unmodified model. Margins before
    RMS normalization are not final-logit margins. This is not an earlier-layer
    approximation or a guarantee of unseen-prompt generalization.

    Thin protected/locality bases and a factored dual Gram avoid a vocabulary-
    sized float64 matrix and a square input-width metric. No model forwards,
    gradients or random numbers occur. CPU float64 ideal factors receive a full
    vocabulary feasibility/duality check. Returned float32 factors additionally
    pass deployment-dtype linear execution, margin and protection checks.
    Resource/rank failures do not establish mathematical infeasibility.
    """
    import torch
    from torch.nn import functional as F

    for name, value in (
        ("margin", margin),
        ("feasibility_atol", feasibility_atol),
        ("duality_atol", duality_atol),
    ):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ConfigurationError(f"{name} must be finite and positive")
    for name, value in (
        ("locality_weight", locality_weight),
        ("margin_atol", margin_atol),
        ("preservation_atol", preservation_atol),
    ):
        if not math.isfinite(value) or value < 0:
            raise ConfigurationError(f"{name} must be finite and nonnegative")
    if not math.isfinite(rcond) or not 0 < rcond < 1:
        raise ConfigurationError("rcond must be between zero and one")
    for name, value in (
        ("chunk_size", chunk_size),
        ("max_constraints", max_constraints),
        ("max_sweeps", max_sweeps),
    ):
        if type(value) is not int or value < 1:
            raise ConfigurationError(f"{name} must be a positive integer")
    if max_update_norm is not None and (
        not math.isfinite(max_update_norm) or max_update_norm < 0
    ):
        raise ConfigurationError("max_update_norm must be finite and nonnegative")
    w, k, r = (
        _matrix(weight, "weight"),
        _matrix(keys, "keys"),
        _matrix(residuals, "residuals"),
    )
    out_dim, in_dim = w.shape
    n = len(k)
    if (
        not n
        or not out_dim
        or not in_dim
        or k.shape[1] != in_dim
        or r.shape != (n, out_dim)
    ):
        raise ConfigurationError("weight, keys and residuals dimensions do not match")
    if (
        weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or keys.dtype != weight.dtype
    ):
        raise ConfigurationError("keys must match a supported deployment weight dtype")
    if (
        not isinstance(readout, torch.Tensor)
        or readout.ndim != 2
        or readout.shape[1] != out_dim
        or readout.shape[0] < 2
        or not readout.is_floating_point()
        or readout.layout != torch.strided
        or readout.device.type == "meta"
    ):
        raise ConfigurationError("readout must be a floating-point vocabulary matrix")
    ids = tuple(target_token_ids)
    if len(ids) != n or any(
        type(i) is not int or not 0 <= i < len(readout) for i in ids
    ):
        raise ConfigurationError("one valid target token id is required per context")
    scale = (
        torch.ones(out_dim, dtype=torch.float64)
        if output_scale is None
        else _matrix(output_scale[None], "output_scale")[0]
    )
    if scale.shape != (out_dim,):
        raise ConfigurationError("output_scale must match output width")
    p = w.new_empty((0, in_dim)) if protect is None else _matrix(protect, "protect")
    h = None if locality is None else _matrix(locality, "locality")
    if p.shape[1] != in_dim or (protect is not None and protect.dtype != weight.dtype):
        raise ConfigurationError("protect must match input width and dtype")
    if h is not None and (
        not len(h)
        or h.shape[1] != in_dim
        or (locality is not None and locality.dtype != weight.dtype)
    ):
        raise ConfigurationError(
            "locality must be nonempty and match input width/dtype"
        )
    q = p
    if len(p):
        pn = p / p.norm(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float64).tiny)
        _, singular, vectors = torch.linalg.svd(pn, full_matrices=False)
        q = vectors[singular > rcond * singular[0]]
    metric = (
        _locality_metric(h, q, locality_weight, rcond)
        if h is not None and locality_weight > 0
        else None
    )

    def project(rows):
        rows = rows - (rows @ q.T) @ q
        return rows - (rows @ q.T) @ q

    projected = project(k)
    whitened = projected if metric is None else project(metric.whiten(projected))
    lengths = whitened.norm(dim=1)
    inactive = lengths <= rcond * k.norm(dim=1)
    lengths[inactive] = 0
    units = whitened / lengths.clamp_min(torch.finfo(torch.float64).tiny)[:, None]
    units[inactive] = 0
    a = project(units if metric is None else metric.whiten(units))
    key_gram = units @ units.T
    chosen = readout[list(ids)].detach().to(device="cpu", dtype=torch.float64)

    def scan(points):
        worst, competitors = [-math.inf] * n, [-1] * n
        margins = [math.inf] * n
        for start in range(0, len(readout), chunk_size):
            stop = min(start + chunk_size, len(readout))
            rows = readout[start:stop].detach().to(device="cpu", dtype=torch.float64)
            if not bool(torch.isfinite(rows).all()):
                raise ConfigurationError("readout must be finite")
            for i in range(n):
                c = (chosen[i] - rows) * scale
                norms = c.norm(dim=1)
                values = c @ points[i]
                keep = torch.arange(start, stop) != ids[i]
                if not bool(
                    torch.isfinite(norms).all() and torch.isfinite(values).all()
                ) or bool(((norms == 0) & keep).any()):
                    raise ConfigurationError(
                        "nonfinite or zero contrast norm; infeasibility is not established"
                    )
                violation = (
                    (margin - values)
                    / norms.clamp_min(torch.finfo(torch.float64).tiny)
                    / max(float(lengths[i]), torch.finfo(torch.float64).tiny)
                )
                violation[~keep] = -math.inf
                if bool(keep.any()):
                    margins[i] = min(margins[i], float(values[keep].min()))
                value, index = violation.max(dim=0)
                if float(value) > worst[i]:
                    worst[i], competitors[i] = float(value), start + int(index)
        return worst, competitors, margins

    context_ids: list[int] = []
    token_ids: list[int] = []
    normals = w.new_empty((0, out_dim))
    rhs, multipliers = w.new_empty(0), w.new_empty(0)
    b = w.new_zeros((out_dim, n))
    sweeps, scans = 0, 0
    while True:
        scans += 1
        worst, competitors, margins = scan(r + (k @ a.T) @ b.T)
        primal = float(((b.T @ b) * key_gram).sum() / 2)
        dual = float(rhs @ multipliers) - primal
        if (
            max(worst) <= feasibility_atol
            and abs(primal - dual) <= duality_atol
            and min(margins) > 0
        ):
            break
        new = [i for i in range(n) if worst[i] > feasibility_atol or margins[i] <= 0]
        if not new or len(context_ids) + len(new) > max_constraints:
            raise ConfigurationError(
                "joint margin constraint/tolerance limit; infeasibility is not established"
            )
        for i in new:
            if lengths[i] == 0 or (i, competitors[i]) in zip(
                context_ids, token_ids, strict=True
            ):
                raise ConfigurationError(
                    "context has insufficient retained key capacity or precision; infeasibility is not established"
                )
            c = (
                chosen[i]
                - readout[competitors[i]].detach().to(device="cpu", dtype=torch.float64)
            ) * scale
            normal = c / c.norm()
            normals = torch.cat((normals, normal[None]))
            rhs = torch.cat(
                (rhs, ((margin / c.norm() - normal @ r[i]) / lengths[i]).reshape(1))
            )
            multipliers = torch.cat((multipliers, w.new_zeros(1)))
            context_ids.append(i)
            token_ids.append(competitors[i])
        gram = (normals @ normals.T) * key_gram[context_ids][:, context_ids]
        multipliers, used = _solve_dual(
            gram,
            rhs,
            multipliers,
            feasibility_atol=feasibility_atol,
            duality_atol=duality_atol,
            max_sweeps=max_sweeps - sweeps,
        )
        sweeps += used
        b.zero_()
        for j, i in enumerate(context_ids):
            b[:, i] += multipliers[j] * normals[j]

    ideal = b @ a
    actual_primal = float(ideal.square().sum() / 2)
    if h is not None:
        actual_primal += (
            locality_weight * float((h @ ideal.T).square().sum()) / (2 * len(h))
        )
    if not math.isfinite(actual_primal) or abs(actual_primal - dual) > duality_atol:
        raise ConfigurationError("joint margin numerical objective certificate failed")
    delta = LowRankDelta(
        b.to(device=weight.device, dtype=torch.float32),
        a.to(device=weight.device, dtype=torch.float32),
    )
    with torch.no_grad(), torch.autocast(device_type=weight.device.type, enabled=False):
        edited = weight.detach() + delta.dense().to(weight.dtype)
        realized = edited.double().cpu() - w
        update_norm = float(realized.norm())
        if not bool(torch.isfinite(edited).all()) or (
            max_update_norm is not None and update_norm > max_update_norm
        ):
            raise ConfigurationError(
                "rounded update is nonfinite or exceeds max_update_norm"
            )
        drift = (
            (
                F.linear(keys.to(weight.device), edited)
                - F.linear(keys.to(weight.device), weight.detach())
            )
            .double()
            .cpu()
        )
        _, _, deployed_margins = scan(r + drift)
        if min(deployed_margins) <= 0 or min(deployed_margins) < margin - margin_atol:
            raise ConfigurationError("rounded edit fails vocabulary margin tolerance")
        span_error = float((realized @ q.T).norm())
        pd = (
            (F.linear(p.to(weight), edited) - F.linear(p.to(weight), weight.detach()))
            .double()
            .cpu()
        )
        if (
            span_error > preservation_atol
            or not bool(torch.isfinite(pd).all())
            or bool(
                (
                    pd.abs()
                    > preservation_atol * p.norm(dim=1, keepdim=True).clamp_min(1)
                ).any()
            )
        ):
            raise ConfigurationError(
                "rounded edit fails protected span/activation tolerance"
            )
        locality_rms = None
        if h is not None:
            hd = (
                (
                    F.linear(h.to(weight), edited)
                    - F.linear(h.to(weight), weight.detach())
                )
                .double()
                .cpu()
            )
            locality_rms = float(hd.square().mean().sqrt())
            if not math.isfinite(locality_rms):
                raise ConfigurationError("locality execution is nonfinite")
    return JointMarginEdit(
        delta,
        JointMarginDiagnostics(
            len(q),
            actual_primal,
            dual,
            actual_primal - dual,
            max(0.0, max(worst)),
            tuple(margins),
            tuple(deployed_margins),
            update_norm,
            span_error,
            float(pd.abs().max()) if pd.numel() else 0.0,
            locality_rms,
            scans,
            sweeps,
        ),
        b.clone(),
        a.clone(),
        tuple(context_ids),
        tuple(token_ids),
        multipliers.clone(),
    )
