"""Numerically stable selected-action scores for causal language models."""

import math
from typing import Any

import torch


def check_selected_support(peak: torch.Tensor, selected: torch.Tensor) -> None:
    """Require defined categorical distributions and supported observed actions."""
    if not (peak.isfinite() & selected.isfinite()).all():
        raise ValueError(
            "Scoring requires a finite normalizer and positive probability for each selected action"
        )


def _token_terms(rows: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    device = "cpu" if rows.device.type == "mps" else rows.device
    rows = rows.to(device=device, dtype=torch.float64)
    labels = labels.to(device)
    peak, pivot = rows.max(-1, keepdim=True)
    observed = rows.gather(-1, labels)
    check_selected_support(peak, observed)
    # log1p retains a small normalizer when one action dominates. Removing
    # the selected maximum algebraically also preserves its small gradient.
    tails = (rows - peak).scatter(-1, pivot, -math.inf).exp().sum(-1, keepdim=True)
    dominant = labels == pivot
    return torch.cat(
        (
            observed.masked_fill(dominant, 0),
            -peak.masked_fill(dominant, 0),
            -tails.log1p(),
        ),
        dim=1,
    )


def selected_logprob_terms(
    rows: torch.Tensor, targets: torch.Tensor, chunk_elements: int = 1 << 20
) -> torch.Tensor:
    """Return the numerator, offset and conditional log-normalizer per observed token."""
    if rows.numel() <= chunk_elements:
        return _token_terms(rows, targets)

    from torch.utils.checkpoint import checkpoint

    block = max(1, chunk_elements // rows.shape[-1])
    parts = []
    for chunk, actions in zip(rows.split(block), targets.split(block), strict=True):
        if chunk.requires_grad and torch.is_grad_enabled():
            part = checkpoint(
                _token_terms,
                chunk,
                actions,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            part = _token_terms(chunk, actions)
        parts.append(part)
    return torch.cat(parts)


def log_probs_from_offsets(offsets: torch.Tensor, pivot: torch.Tensor) -> torch.Tensor:
    """Normalize centered logits while retaining the winning action's tail and slope."""
    offsets = offsets.scatter(-1, pivot, 0.0)
    tail = offsets.scatter(-1, pivot, -math.inf).exp().sum(-1, keepdim=True)
    return offsets - tail.log1p()


def _categorical_logprobs(scores: torch.Tensor) -> torch.Tensor:
    device = "cpu" if scores.device.type == "mps" else scores.device
    scores = scores.to(device=device, dtype=torch.float64)
    peak, pivot = scores.max(-1, keepdim=True)
    if not peak.isfinite().all():
        raise ValueError("Categorical regularization requires finite normalizers")
    return log_probs_from_offsets(scores - peak, pivot)


def _small_kl_gap(d: torch.Tensor) -> torch.Tensor:
    """Degree-six expansion of exp(d) - 1 - d near zero."""
    return (d.square() / 2) * (1 + d / 3 * (1 + d / 4 * (1 + d / 5 * (1 + d / 6))))


def _categorical_terms(
    scores: torch.Tensor, reference: torch.Tensor | None, want_entropy: bool
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    logp = _categorical_logprobs(scores)
    absent = logp.isneginf()
    p = logp.exp()
    finite_logp = logp.masked_fill(absent, 0)
    entropy = -(p * finite_logp).sum(-1) if want_entropy else None
    if reference is None:
        return entropy, None
    logq = _categorical_logprobs(reference.detach())
    if (logq.isneginf() & ~absent).any():
        raise ValueError("Forward KL requires reference support for the policy")
    # Sum the nonnegative terms p log(p/q) - p + q. Extra reference mass
    # outside the learner support contributes q. Near equality the quadratic
    # series for exp(d)-1-d retains both the small value and its derivative.
    d = logq.masked_fill(absent, 0) - finite_logp
    tiny = d.abs() < 0.001
    u = d.masked_fill(~tiny, 0)
    taylor = _small_kl_gap(u)
    # Use the direct expression away from zero to avoid an overflowing q/p.
    near = d.abs() <= 0.5
    v = d.masked_fill(~near | tiny, 0)
    central = p * torch.where(tiny, taylor, v.expm1() - v)
    far = logq.exp() - p * (1 + d)
    terms = torch.where(absent, logq.exp(), torch.where(near, central, far))
    return entropy, terms.sum(-1)


def categorical_regularizers(
    logits: torch.Tensor,
    ref_logits: torch.Tensor | None,
    mask: torch.Tensor,
    *,
    want_entropy: bool,
    chunk_elements: int = 1 << 20,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute entropy and forward KL at active next-token prediction positions."""
    if ref_logits is not None and ref_logits.shape != logits.shape:
        raise ValueError("Policy and reference vocabularies and positions must align")
    row, pos = mask.nonzero(as_tuple=True)
    block = max(1, chunk_elements // logits.shape[-1])
    entropies, divergences = [], []
    for rows, positions in zip(row.split(block), pos.split(block), strict=True):
        scores = logits[rows, positions]
        reference = None if ref_logits is None else ref_logits[rows, positions]
        if row.numel() > block and scores.requires_grad and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            entropy, divergence = checkpoint(
                _categorical_terms,
                scores,
                reference,
                want_entropy,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            entropy, divergence = _categorical_terms(scores, reference, want_entropy)
        if entropy is not None:
            entropies.append(entropy)
        if divergence is not None:
            divergences.append(divergence)

    def scatter(parts: list[torch.Tensor]) -> torch.Tensor | None:
        if not parts:
            return None
        values = torch.cat(parts)
        return values.new_zeros(mask.shape).masked_scatter(
            mask.to(values.device).bool(), values
        )

    return scatter(entropies), scatter(divergences)


def regularized_policy_scores(
    model: Any,
    reference: Any,
    ids: torch.Tensor,
    attention: torch.Tensor,
    response_mask: torch.Tensor,
    objective: Any,
    logprob_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Score sampled actions and the configured entropy and frozen-reference terms."""
    from . import rl_losses
    from .trainer_utils import deterministic_scoring

    with deterministic_scoring(model):
        logits = model(input_ids=ids, attention_mask=attention).logits
    current, _ = rl_losses.gather_token_logprobs(
        logits, ids, response_mask, dtype=logprob_dtype
    )
    device = "cpu" if current.device.type == "mps" else current.device
    auxiliary = {}
    full_kl = objective.kl == "external" and objective.kl_coef > 0
    ref_logits = None
    if objective.kl_coef > 0:
        if reference is None:
            raise RuntimeError("KL regularization requires a reference policy")
        with torch.no_grad(), deterministic_scoring(reference):
            ref_logits = reference(input_ids=ids, attention_mask=attention).logits
            if not full_kl:
                ref_scores, _ = rl_losses.gather_token_logprobs(
                    ref_logits, ids, response_mask, dtype=logprob_dtype
                )
                auxiliary["logp_ref"] = ref_scores.to(device)
    if objective.entropy_coef > 0 or full_kl:
        entropy, divergence = categorical_regularizers(
            logits,
            ref_logits if full_kl else None,
            response_mask[:, 1:],
            want_entropy=objective.entropy_coef > 0,
        )
        if entropy is not None:
            auxiliary["entropy"] = entropy
        if divergence is not None:
            auxiliary["kl"] = divergence
    return current.to(device, torch.float64), auxiliary
