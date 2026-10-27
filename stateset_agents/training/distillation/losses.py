"""Distillation losses: pure tensor functions, no trainer state.

With ``S = softmax(z_student / τ)`` and ``T = softmax(z_teacher / τ)`` at one
response position, the per-token divergence is GKD's generalized
Jensen-Shannon divergence (Agarwal et al. 2024, eq. 1) with weight ``β``::

    β = 0       forward KL   KL(T ‖ S) = Σ_v T(v) (log T(v) - log S(v))
    β = 1       reverse KL   KL(S ‖ T) = Σ_v S(v) (log S(v) - log T(v))
    0 < β < 1   JSD_β = β KL(T ‖ M) + (1 - β) KL(S ‖ M),  M = β T + (1 - β) S

Forward KL is classic logit KD (mean-seeking); reverse KL is mode-seeking.
The endpoints are TRL's convention, so :func:`generalized_jsd_loss` equals
``trl``'s ``GKDTrainer.generalized_jsd_loss`` on identical tensors
(``tests/unit/test_distillation_trl_pin.py``). Like TRL, logits are divided by
``τ`` without a ``τ²`` gradient rescale.

On-policy samples can also be trained with :func:`sampled_reverse_kl`, the
sampled-token estimator (the teacher scores only the tokens the student
sampled). :func:`topk_divergence` evaluates any ``β`` against a cached top-k
teacher distribution plus one bucket for the remaining mass.

torch is fetched lazily, so importing this module never requires torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from stateset_agents.core.errors import ConfigurationError

from ..preference.losses import NonFiniteLossError, _log1mexp
from ..trainer_utils import get_torch, require_torch

__all__ = [
    "DistillationLoss",
    "NonFiniteLossError",
    "VocabularyMismatchError",
    "divergence_from_logprobs",
    "generalized_jsd_loss",
    "per_example_means",
    "sampled_reverse_kl",
    "token_divergence",
    "topk_divergence",
    "topk_logprobs",
    "validate_jsd_beta",
    "validate_temperature",
]


class VocabularyMismatchError(ConfigurationError):
    """Teacher and student distributions are over different vocabularies."""


@dataclass(frozen=True)
class DistillationLoss:
    """Per-example losses ``[B]`` (token means) and what reduces them.

    ``weights`` are ``example weight × response tokens``, so a window's
    weighted mean is the token mean. ``token_sums`` holds detached per-token
    sums of the diagnostics (``forward_kl``, ``reverse_kl``, ``divergence``
    and, for sampled on-policy batches, ``kl_estimate``) over ``tokens``.
    """

    losses: Any
    weights: Any
    token_sums: dict[str, float]
    tokens: int


def _t() -> Any:
    return get_torch() or require_torch()


def validate_jsd_beta(beta: float) -> float:
    """GKD's ``β`` must be a finite number in ``[0, 1]``."""
    if isinstance(beta, bool) or not isinstance(beta, int | float):
        raise ValueError("beta must be a number")
    value = float(beta)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta!r}")
    return value


def validate_temperature(temperature: float) -> float:
    """The distillation temperature must be finite and positive."""
    if isinstance(temperature, bool) or not isinstance(temperature, int | float):
        raise ValueError("temperature must be a number")
    value = float(temperature)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"temperature must be finite and positive, got {temperature!r}"
        )
    return value


def _upcast(tensor: Any) -> Any:
    torch = _t()
    if tensor.dtype in (torch.float32, torch.float64):
        return tensor
    return tensor.to(torch.float32)


def _kl(log_p: Any, log_q: Any) -> Any:
    """``Σ_v p (log p - log q)`` over the last dimension (``0 log 0 = 0``)."""
    torch = _t()
    probabilities = log_p.exp()
    # Zero the log-ratio (not the product) where p = 0, so neither the value
    # nor the gradient picks up 0 · inf.
    difference = log_p - log_q
    difference = torch.where(
        probabilities > 0, difference, torch.zeros_like(difference)
    )
    return (probabilities * difference).sum(-1)


def divergence_from_logprobs(
    student_logps: Any, teacher_logps: Any, *, beta: float
) -> Any:
    """Generalized JSD between normalized log-distributions ``[..., V]``,
    reduced over the last dimension (see the module docstring)."""
    torch = _t()
    beta = validate_jsd_beta(beta)
    if student_logps.shape != teacher_logps.shape:
        raise VocabularyMismatchError(
            f"student and teacher distributions differ in shape: "
            f"{tuple(student_logps.shape)} vs {tuple(teacher_logps.shape)}"
        )
    if beta == 0.0:
        return _kl(teacher_logps, student_logps)
    if beta == 1.0:
        return _kl(student_logps, teacher_logps)
    weight = torch.tensor(beta, dtype=student_logps.dtype, device=student_logps.device)
    mixture = torch.logsumexp(
        torch.stack(
            [student_logps + torch.log1p(-weight), teacher_logps + torch.log(weight)]
        ),
        dim=0,
    )
    return beta * _kl(teacher_logps, mixture) + (1.0 - beta) * _kl(
        student_logps, mixture
    )


def _logprobs(logits: Any, temperature: float) -> Any:
    return _t().log_softmax(_upcast(logits) / temperature, dim=-1)


def token_divergence(
    student_logits: Any, teacher_logits: Any, *, beta: float, temperature: float = 1.0
) -> Any:
    """Per-position divergence ``[...]`` between logits ``[..., V]``.

    Raises :class:`VocabularyMismatchError` when the vocabularies differ in
    size (never truncates one to the other).
    """
    temperature = validate_temperature(temperature)
    if student_logits.shape[-1] != teacher_logits.shape[-1]:
        raise VocabularyMismatchError(
            f"student logits have {student_logits.shape[-1]} entries, the "
            f"teacher's {teacher_logits.shape[-1]}; use vocab_alignment='shared'"
        )
    return divergence_from_logprobs(
        _logprobs(student_logits, temperature),
        _logprobs(teacher_logits, temperature),
        beta=beta,
    )


def generalized_jsd_loss(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any = None,
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "batchmean",
) -> Any:
    """TRL-compatible GKD loss over already shift-aligned logits ``[B, T, V]``.

    ``labels`` ``[B, T]`` mark ignored positions with ``-100``.
    ``batchmean`` is the mean over unmasked positions (TRL's semantics),
    ``sum`` their sum and ``none`` the per-position divergence ``[B, T]``
    (masked positions zeroed).
    """
    torch = _t()
    divergence = token_divergence(
        student_logits, teacher_logits, beta=beta, temperature=temperature
    )
    if labels is None:
        mask = torch.ones_like(divergence, dtype=torch.bool)
    else:
        mask = labels != -100
    divergence = torch.where(mask, divergence, torch.zeros_like(divergence))
    if reduction == "none":
        return divergence
    if reduction == "sum":
        return divergence.sum()
    if reduction == "batchmean":
        return divergence.sum() / mask.sum().clamp_min(1)
    raise ValueError(f"reduction must be batchmean, sum or none, got {reduction!r}")


def topk_logprobs(teacher_logits: Any, *, k: int, temperature: float = 1.0) -> Any:
    """The teacher's top-``k`` ``(ids, log-probs)`` ``[..., k]`` at ``temperature``
    (what a teacher cache stores)."""
    temperature = validate_temperature(temperature)
    logps = _logprobs(teacher_logits, temperature)
    values, ids = logps.topk(min(int(k), logps.shape[-1]), dim=-1)
    return ids, values


def _with_tail(logps: Any) -> Any:
    """Append the log of the remaining mass ``log(1 - Σ exp(logps))``."""
    torch = _t()
    tail = _log1mexp(torch.logsumexp(logps, dim=-1))
    return torch.cat([logps, tail.unsqueeze(-1)], dim=-1)


def topk_divergence(
    student_logps: Any, teacher_ids: Any, teacher_logps: Any, *, beta: float
) -> Any:
    """Divergence against a top-k teacher: both distributions coarse-grained
    to the teacher's ``k`` tokens plus one bucket holding everything else.

    ``student_logps`` are the student's full normalized log-probs ``[N, V]``;
    ``teacher_ids``/``teacher_logps`` ``[N, k]`` come from
    :func:`topk_logprobs`. By the data-processing inequality this lower-bounds
    the full divergence and equals it when the teacher's mass outside the top
    ``k`` is negligible. Bucket masses below ``1e-7`` are floored there.
    """
    if teacher_ids.shape != teacher_logps.shape:
        raise ValueError("teacher ids and log-probs must have the same shape")
    if int(teacher_ids.max()) >= student_logps.shape[-1] or int(teacher_ids.min()) < 0:
        raise VocabularyMismatchError(
            "cached teacher token ids fall outside the student's vocabulary"
        )
    student = student_logps.gather(-1, teacher_ids.long())
    return divergence_from_logprobs(
        _with_tail(student), _with_tail(_upcast(teacher_logps)), beta=beta
    )


def sampled_reverse_kl(
    student_token_logps: Any, teacher_token_logps: Any
) -> tuple[Any, Any]:
    """The sampled-token on-policy distillation objective.

    For tokens ``x_t`` the student sampled, returns ``(surrogate, estimate)``
    with ``estimate = log S(x_t) - log T(x_t)`` (detached; an unbiased
    estimate of the per-token reverse KL) and ``surrogate = estimate ·
    log S(x_t)``, whose gradient in expectation over ``x_t ~ S`` is
    ``∇ KL(S ‖ T)`` at that position. Each position is credited on its own
    (no credit to earlier tokens for later positions' divergence), as in
    GKD's on-policy reverse KL. Only the sampled tokens' teacher log-probs
    are needed.
    """
    if student_token_logps.shape != teacher_token_logps.shape:
        raise ValueError("student and teacher token log-probs must align")
    estimate = (student_token_logps - teacher_token_logps).detach()
    return estimate * student_token_logps, estimate


def per_example_means(values: Any, rows: Any, counts: Any) -> Any:
    """Mean of token ``values`` ``[N]`` per example (``rows`` ``[N]`` gives
    each token's example, ``counts`` ``[B]`` each example's token count)."""
    torch = _t()
    sums = torch.zeros(counts.shape[0], dtype=values.dtype, device=values.device)
    sums = sums.index_add(0, rows, values)
    return sums / counts.clamp_min(1).to(values.dtype)


def require_finite(tensor: Any, name: str) -> None:
    """Raise :class:`NonFiniteLossError` for NaN or infinite entries."""
    if tensor.numel() and not bool(_t().isfinite(tensor.detach()).all()):
        raise NonFiniteLossError(f"{name} contains non-finite values")
