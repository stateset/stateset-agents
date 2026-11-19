"""Group expectation weights from frozen sampler scores and live response tokens."""

from __future__ import annotations

import math
from fractions import Fraction

import torch


def _centered_sum(row: list[float], mask: list[float], origin: float) -> float:
    """Round the weighted sum minus its origin once, even after cancellation."""
    if all(weight in (0.0, 1.0) for weight in mask):
        # Binary response masks need only a compensated sum. fsum can raise
        # if an intermediate partial sum is outside the double exponent range.
        terms = [value for value, keep in zip(row, mask, strict=True) if keep]
        terms.append(-origin)
        try:
            return math.fsum(terms)
        except OverflowError:
            pass
    total = -Fraction.from_float(origin)
    for value, weight in zip(row, mask, strict=True):
        if weight:
            total += Fraction.from_float(value) * Fraction.from_float(weight)
    try:
        return float(total)
    except OverflowError:
        return math.inf if total > 0 else -math.inf


def group_coefficients(
    current: torch.Tensor, mask: torch.Tensor, sampler: torch.Tensor, bound: float
) -> torch.Tensor:
    """Return one coefficient per row for a single empirical prompt group."""
    if (
        current.ndim != 2
        or mask.shape != current.shape
        or sampler.shape != (len(current),)
    ):
        raise ValueError("GEPO requires one sampler sequence sum per learner row")
    dtype = torch.promote_types(current.dtype, torch.float32)
    if not len(current):
        return current.sum(-1).to(dtype)
    device = "cpu" if current.device.type == "mps" else current.device
    frozen = sampler.detach().to(device=device, dtype=torch.float64)
    if not torch.isfinite(frozen).all():
        raise ValueError("GEPO sampler log probabilities must be finite")
    origin = frozen.max()
    shifted = frozen - origin
    # The common offset cancels between the learner and empirical denominator.
    correction = torch.logsumexp(2 * shifted, 0) - torch.logsumexp(shifted, 0)
    weights = mask.to(device=device, dtype=torch.float64)
    values = current.to(device=device, dtype=torch.float64).masked_fill(weights == 0, 0)
    tokens = values * weights
    gap = tokens.sum(-1) - origin
    if current.shape[1] > 1:
        # Only cancellation near the unsaturated ratio can change its value or
        # slope. Use exact dyadic arithmetic for those poorly conditioned rows.
        with torch.no_grad():
            magnitude = tokens.abs().sum(-1) + origin.abs()
            budget = magnitude * (
                (2 * current.shape[1] + 3) * torch.finfo(torch.float64).eps
            )
            uncertain = ((gap - correction).abs() <= bound + budget) | ~gap.isfinite()
            sensitive = (budget > 1e-9) & uncertain & values.isfinite().all(-1)
        if sensitive.any():
            anchor = origin.item()
            exact = [
                _centered_sum(r, m, anchor)
                for r, m in zip(
                    values[sensitive].detach().tolist(),
                    weights[sensitive].tolist(),
                    strict=True,
                )
            ]
            selected = values[sensitive]
            # Restore the linear derivative of the masked sum at its exact value.
            gap = gap.clone()
            gap[sensitive] = gap.new_tensor(exact) + (
                (selected - selected.detach()) * weights[sensitive]
            ).sum(-1)
    return (
        (gap - correction)
        .clamp(-bound, bound)
        .exp()
        .to(device=current.device, dtype=dtype)
    )
