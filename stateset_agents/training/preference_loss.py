"""Causal sequence likelihood ratios for direct preference optimization."""

from __future__ import annotations

import math
from fractions import Fraction

import torch
import torch.nn.functional as F

from .causal_likelihood import check_selected_support, selected_logprob_terms

_SCORE_CHUNK_ELEMENTS = 1 << 20


def completion_terms(
    logits: torch.Tensor,
    ids: torch.Tensor,
    start: int,
    cancel: int = 0,
    alternate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score the suffix and, at the first differing action, its raw logit contrast."""
    rows, labels = logits[0, start - 1 : -1], ids[0, start:, None]
    if cancel:
        shared = rows[:cancel].detach()
        check_selected_support(
            shared.amax(-1, keepdim=True), shared.gather(-1, labels[:cancel])
        )
    suffix, targets = rows[cancel:], labels[cancel:]
    terms = selected_logprob_terms(suffix, targets, _SCORE_CHUNK_ELEMENTS).reshape(-1)
    if alternate is not None:
        # Both actions are predicted at this same causal prefix. Their log
        # partitions cancel, so only the two selected logits carry a slope.
        fork = (
            rows[cancel - 1]
            .gather(0, torch.stack((labels[cancel - 1, 0], alternate)))
            .to(terms)
        )
        check_selected_support(fork[0], fork[1])
        terms = torch.cat((terms, fork[:1], -fork[1:]))
    return terms


def margin_loss(scaled: torch.Tensor) -> torch.Tensor:
    """Negative log probability of the observed preference at the scaled margin."""
    return -F.logsigmoid(scaled)


class _ScaledPreferenceTail(torch.autograd.Function):
    """Differentiate a saturated pair without first rounding its tiny weight."""

    @staticmethod
    def forward(ctx, terms, scaled, beta):
        ctx.save_for_backward(scaled)
        ctx.shape = terms.shape
        ctx.beta = beta
        return margin_loss(scaled)

    @staticmethod
    def backward(ctx, grad):
        (scaled,) = ctx.saved_tensors
        magnitude = (math.log(abs(ctx.beta)) - F.softplus(scaled)).exp()
        slope = -math.copysign(1.0, ctx.beta) * magnitude
        return (grad * slope).expand(ctx.shape), None, None


def preference_pair_loss(
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Evaluate the reference-adjusted pair loss with frozen reference evidence."""
    if not math.isfinite(beta):
        raise ValueError("DPO beta must be finite")
    terms = torch.cat(
        (chosen, -rejected, -ref_chosen.detach(), ref_rejected.detach())
    ).reshape(-1)
    scaled = terms.sum() * beta
    with torch.no_grad():
        error = (
            terms.abs().sum()
            * (terms.numel() * torch.finfo(terms.dtype).eps)
            * abs(beta)
        )
        uncertain = ~scaled.isfinite() | (error > 1e-12 * (1 + scaled.abs()))
    if uncertain:
        # Compensate cancellation across model scores and response paths.
        # Exact dyadic arithmetic also handles an intermediate overflow in fsum.
        values = terms.detach().tolist()
        try:
            rounded = math.fsum(values) * beta
        except OverflowError:
            exact = sum(
                map(Fraction.from_float, values), Fraction()
            ) * Fraction.from_float(float(beta))
            try:
                rounded = float(exact)
            except OverflowError:
                rounded = math.inf if exact > 0 else -math.inf
        scaled = terms.new_tensor(rounded) + (terms - terms.detach()).sum() * beta
    if abs(beta) > 1 and scaled.detach() > 700:
        return _ScaledPreferenceTail.apply(terms, scaled, beta)
    return margin_loss(scaled)
