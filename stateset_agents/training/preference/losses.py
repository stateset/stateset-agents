"""Preference-optimization losses: pure tensor functions, no trainer state.

Every loss returns *per-example* losses (reduce them with
:func:`reduce_losses`, which applies pair weights) and the detached implicit
rewards that preference metrics are reported from.

Log-probabilities are response-token log-probs gathered with the shared
shift-by-one helper (:func:`stateset_agents.training.rl_losses.gather_token_logprobs`):
``sequence_logprobs`` returns their per-sequence **sum** and the response token
count. DPO, cDPO and KTO use summed log-probs (the sequence likelihood); IPO,
SimPO and ORPO length-normalize (mean per response token), matching TRL 1.12.

torch is fetched lazily, so importing this module never requires torch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..rl_losses import gather_token_logprobs
from ..trainer_utils import get_torch, require_torch

__all__ = [
    "NonFiniteLossError",
    "PreferenceLoss",
    "bradley_terry_loss",
    "dpo_loss",
    "ipo_loss",
    "kto_kl_estimate",
    "kto_loss",
    "mean_logprobs",
    "orpo_loss",
    "pairwise_accuracy",
    "pairwise_auc",
    "reduce_losses",
    "sequence_logprobs",
    "simpo_loss",
    "validate_beta",
    "validate_label_smoothing",
]


class NonFiniteLossError(ValueError):
    """A loss input or result is NaN or infinite."""


@dataclass(frozen=True)
class PreferenceLoss:
    """Per-example losses ``[N]`` and detached implicit rewards.

    For paired methods ``chosen_rewards``/``rejected_rewards`` are ``[N]``
    (one per pair). For KTO they hold the desirable and undesirable examples'
    rewards respectively.
    """

    losses: Any
    chosen_rewards: Any
    rejected_rewards: Any


def _t() -> Any:
    return get_torch() or require_torch()


def _functional() -> Any:
    return _t().nn.functional


def validate_beta(beta: float, name: str = "beta") -> float:
    """``beta`` must be a finite positive number."""
    if isinstance(beta, bool) or not isinstance(beta, int | float):
        raise ValueError(f"{name} must be a number")
    value = float(beta)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {beta!r}")
    return value


def validate_label_smoothing(label_smoothing: float) -> float:
    """Conservative-DPO label smoothing must lie in ``[0, 0.5)``."""
    if isinstance(label_smoothing, bool) or not isinstance(
        label_smoothing, int | float
    ):
        raise ValueError("label_smoothing must be a number")
    value = float(label_smoothing)
    if not math.isfinite(value) or not 0.0 <= value < 0.5:
        raise ValueError(
            f"label_smoothing must be in [0, 0.5), got {label_smoothing!r}"
        )
    return value


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    return number


def _require_finite(tensor: Any, name: str) -> None:
    if tensor.numel() and not bool(_t().isfinite(tensor).all()):
        raise NonFiniteLossError(f"{name} contains non-finite values")


def _vectors(**tensors: Any) -> None:
    shapes = set()
    for name, tensor in tensors.items():
        if tensor.ndim != 1:
            raise ValueError(
                f"{name} must be a 1-D tensor, got shape {tuple(tensor.shape)}"
            )
        _require_finite(tensor, name)
        shapes.add(tuple(tensor.shape))
    if len(shapes) > 1:
        raise ValueError(f"tensor shapes differ: {sorted(shapes)}")


def _lengths(lengths: Any, name: str) -> Any:
    if bool((lengths <= 0).any()):
        raise ValueError(f"{name} must be positive response token counts")
    return lengths.to(_t().float32)


def sequence_logprobs(
    logits: Any, input_ids: Any, response_mask: Any, *, dtype: Any = None
) -> tuple[Any, Any]:
    """Summed response-token log-probs ``[B]`` and response token counts ``[B]``.

    Position ``t`` of ``logits`` predicts token ``t + 1``; only tokens whose
    ``response_mask`` is 1 count, so prompt and padding are excluded.
    """
    token_logps, mask = gather_token_logprobs(
        logits, input_ids, response_mask, dtype=dtype
    )
    return token_logps.sum(-1), mask.sum(-1)


def mean_logprobs(summed_logps: Any, lengths: Any) -> Any:
    """Length-normalized (per response token) log-probs."""
    return summed_logps / _lengths(lengths, "lengths")


def dpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    reference_chosen_logps: Any,
    reference_rejected_logps: Any,
    *,
    beta: float,
    label_smoothing: float = 0.0,
) -> PreferenceLoss:
    """DPO (sigmoid) loss; ``label_smoothing > 0`` gives conservative DPO.

    ``-(1-ε) log σ(β h) - ε log σ(-β h)`` with
    ``h = (log π(y_w) - log π_ref(y_w)) - (log π(y_l) - log π_ref(y_l))`` on
    summed response log-probs (Rafailov et al. 2023; cDPO: Mitchell 2023).
    """
    beta = validate_beta(beta)
    smoothing = validate_label_smoothing(label_smoothing)
    _vectors(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=reference_chosen_logps,
        reference_rejected_logps=reference_rejected_logps,
    )
    F = _functional()
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps
    logits = chosen_logratios - rejected_logratios
    losses = -F.logsigmoid(beta * logits) * (1.0 - smoothing)
    if smoothing:
        losses = losses - F.logsigmoid(-beta * logits) * smoothing
    return PreferenceLoss(
        losses,
        beta * chosen_logratios.detach(),
        beta * rejected_logratios.detach(),
    )


def ipo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    reference_chosen_logps: Any,
    reference_rejected_logps: Any,
    chosen_lengths: Any,
    rejected_lengths: Any,
    *,
    beta: float,
) -> PreferenceLoss:
    """IPO loss ``(h̄ - 1/(2β))²`` (Azar et al. 2023, eq. 17).

    ``h̄`` uses log-ratios averaged per response token, as TRL does (confirmed
    with the IPO authors), so ``beta`` is comparable across lengths.
    """
    beta = validate_beta(beta)
    _vectors(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=reference_chosen_logps,
        reference_rejected_logps=reference_rejected_logps,
    )
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps
    delta = chosen_logratios / _lengths(
        chosen_lengths, "chosen_lengths"
    ) - rejected_logratios / _lengths(rejected_lengths, "rejected_lengths")
    losses = (delta - 1.0 / (2.0 * beta)) ** 2
    return PreferenceLoss(
        losses,
        beta * chosen_logratios.detach(),
        beta * rejected_logratios.detach(),
    )


def simpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    chosen_lengths: Any,
    rejected_lengths: Any,
    *,
    beta: float,
    gamma: float,
    label_smoothing: float = 0.0,
) -> PreferenceLoss:
    """Reference-free SimPO loss (Meng et al. 2024).

    ``-log σ(β/|y_w| log π(y_w) - β/|y_l| log π(y_l) - γ)``: length-normalized
    implicit rewards with a target margin ``gamma``.
    """
    beta = validate_beta(beta)
    gamma = _nonnegative(gamma, "gamma")
    smoothing = validate_label_smoothing(label_smoothing)
    _vectors(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
    )
    F = _functional()
    chosen = policy_chosen_logps / _lengths(chosen_lengths, "chosen_lengths")
    rejected = policy_rejected_logps / _lengths(rejected_lengths, "rejected_lengths")
    logits = (chosen - rejected) - gamma / beta
    losses = -F.logsigmoid(beta * logits) * (1.0 - smoothing)
    if smoothing:
        losses = losses - F.logsigmoid(-beta * logits) * smoothing
    return PreferenceLoss(losses, beta * chosen.detach(), beta * rejected.detach())


def _log1mexp(x: Any) -> Any:
    """``log(1 - exp(x))`` for ``x < 0`` (branch at ``-ln 2``)."""
    torch = _t()
    x = torch.clamp(x, max=-1e-7)
    return torch.where(
        x < -math.log(2.0), torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x))
    )


def orpo_loss(
    policy_chosen_logps: Any,
    policy_rejected_logps: Any,
    chosen_lengths: Any,
    rejected_lengths: Any,
    *,
    beta: float,
) -> PreferenceLoss:
    """Reference-free ORPO loss (Hong et al. 2024).

    ``NLL(y_w) - β log σ(log odds(y_w) - log odds(y_l))`` with mean
    per-token log-probs; the NLL is the chosen response's mean token NLL.
    """
    beta = validate_beta(beta)
    _vectors(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
    )
    F = _functional()
    chosen = policy_chosen_logps / _lengths(chosen_lengths, "chosen_lengths")
    rejected = policy_rejected_logps / _lengths(rejected_lengths, "rejected_lengths")
    log_odds = (chosen - rejected) - (_log1mexp(chosen) - _log1mexp(rejected))
    losses = -chosen - beta * F.logsigmoid(log_odds)
    return PreferenceLoss(losses, beta * chosen.detach(), beta * rejected.detach())


def kto_kl_estimate(policy_kl_logps: Any, reference_kl_logps: Any) -> Any:
    """KTO's reference point: the batch mean of ``log π - log π_ref`` on
    *mismatched* prompt/completion pairs, clamped at 0 and detached."""
    _vectors(policy_kl_logps=policy_kl_logps, reference_kl_logps=reference_kl_logps)
    if policy_kl_logps.numel() == 0:
        return _t().zeros((), dtype=_t().float32)
    return (policy_kl_logps - reference_kl_logps).mean().detach().clamp(min=0.0)


def kto_loss(
    policy_logps: Any,
    reference_logps: Any,
    labels: Any,
    kl: Any,
    *,
    beta: float,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> PreferenceLoss:
    """KTO loss on unpaired examples (Ethayarajh et al. 2024, eq. 7).

    Desirable: ``λ_D (1 - σ(β(r - z)))``; undesirable:
    ``λ_U (1 - σ(β(z - r)))`` with ``r = log π - log π_ref`` (summed) and the
    reference point ``z`` from :func:`kto_kl_estimate`. Losses keep input
    order; rewards are split into desirable and undesirable.
    """
    beta = validate_beta(beta)
    desirable_weight = validate_beta(desirable_weight, "desirable_weight")
    undesirable_weight = validate_beta(undesirable_weight, "undesirable_weight")
    _vectors(policy_logps=policy_logps, reference_logps=reference_logps)
    torch = _t()
    if labels.shape != policy_logps.shape:
        raise ValueError("labels must match the log-prob shape")
    _require_finite(torch.as_tensor(kl, dtype=torch.float32), "kl")
    desirable = labels.to(torch.bool)
    logratios = policy_logps - reference_logps
    chosen_losses = desirable_weight * (1.0 - torch.sigmoid(beta * (logratios - kl)))
    rejected_losses = undesirable_weight * (
        1.0 - torch.sigmoid(beta * (kl - logratios))
    )
    losses = torch.where(desirable, chosen_losses, rejected_losses)
    rewards = beta * logratios.detach()
    return PreferenceLoss(losses, rewards[desirable], rewards[~desirable])


def bradley_terry_loss(
    chosen_rewards: Any,
    rejected_rewards: Any,
    *,
    margin: Any = None,
    center_coefficient: float | None = None,
) -> PreferenceLoss:
    """Bradley-Terry reward-model loss ``-log σ(r_w - r_l - m)``.

    ``margin`` (per pair) asks for a larger gap on clearer preferences
    (Llama 2); ``center_coefficient`` adds ``c (r_w + r_l)²`` to keep rewards
    centred at zero (TRL's ``center_rewards_coefficient``).
    """
    _vectors(chosen_rewards=chosen_rewards, rejected_rewards=rejected_rewards)
    F = _functional()
    difference = chosen_rewards - rejected_rewards
    if margin is not None:
        if margin.shape != difference.shape:
            raise ValueError("margin must have one value per pair")
        _require_finite(margin, "margin")
        difference = difference - margin
    losses = -F.logsigmoid(difference)
    if center_coefficient:
        coefficient = _nonnegative(center_coefficient, "center_coefficient")
        losses = losses + coefficient * (chosen_rewards + rejected_rewards) ** 2
    return PreferenceLoss(losses, chosen_rewards.detach(), rejected_rewards.detach())


def reduce_losses(
    losses: Any, weights: Any = None, *, normalizer: float | None = None
) -> Any:
    """``sum(w_i l_i) / normalizer`` (default ``sum(w_i)``), finite-checked.

    Pass the total weight of an accumulation window as ``normalizer`` so that
    micro-batches of a window add up to the window's weighted mean.
    """
    torch = _t()
    if weights is None:
        weights = torch.ones_like(losses)
    if weights.shape != losses.shape:
        raise ValueError("weights must match the losses")
    total = (losses * weights.to(losses.dtype)).sum()
    denominator = float(weights.sum()) if normalizer is None else float(normalizer)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("loss normalizer must be finite and positive")
    reduced = total / denominator
    _require_finite(reduced.detach().reshape(1), "loss")
    return reduced


def pairwise_accuracy(chosen: Sequence[float], rejected: Sequence[float]) -> float:
    """Fraction of pairs whose chosen score is strictly higher."""
    if len(chosen) != len(rejected):
        raise ValueError("chosen and rejected scores must pair up")
    if not chosen:
        return 0.0
    return sum(c > r for c, r in zip(chosen, rejected, strict=True)) / len(chosen)


def pairwise_auc(chosen: Sequence[float], rejected: Sequence[float]) -> float:
    """ROC AUC of chosen scores against rejected scores (ties count half).

    The probability that a random chosen response outscores a random rejected
    one, from the Mann-Whitney rank sum.
    """
    if not chosen or not rejected:
        return 0.0
    scored = sorted(
        [(float(value), 1) for value in chosen]
        + [(float(value), 0) for value in rejected]
    )
    rank_sum = 0.0
    index = 0
    while index < len(scored):
        end = index
        while end + 1 < len(scored) and scored[end + 1][0] == scored[index][0]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        rank_sum += average_rank * sum(label for _, label in scored[index : end + 1])
        index = end + 1
    n_chosen, n_rejected = len(chosen), len(rejected)
    return (rank_sum - n_chosen * (n_chosen + 1) / 2.0) / (n_chosen * n_rejected)
