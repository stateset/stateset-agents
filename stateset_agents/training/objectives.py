"""Declarative policy-optimisation objectives shared by every native trainer.

A :class:`PolicyObjective` names one point in the space
``advantage × ratio × clip × aggregate × kl``; :func:`compute_advantages`
and :func:`policy_loss` evaluate it on batched tensors. Presets in
:data:`OBJECTIVES` reproduce GRPO, Dr. GRPO, BNPO, DAPO, GSPO, GSPO-token,
GEPO, RLOO, REINFORCE++-baseline, CISPO, and PPO.

Pure tensor code: no trainer state, no model calls, torch fetched lazily so
importing this module never requires torch. Log-prob gathering stays with
the trainer (``rl_losses.gather_token_logprobs``); nothing here allocates a
full-vocabulary tensor.

Formulas and citations: ``docs/OBJECTIVES.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from . import rl_losses
from .trainer_utils import get_torch, require_torch

ADVANTAGE_KINDS = (
    "group_norm",
    "group_mean",
    "leave_one_out",
    "batch_norm",
    "external",
)
RATIO_KINDS = ("token", "sequence", "sequence_token", "group_expectation")
CLIP_KINDS = ("clipped", "cispo", "none")
AGGREGATE_KINDS = ("seq_mean", "token_mean", "seq_sum_const")
KL_KINDS = ("none", "k3_token", "k3_sequence", "external")


def _t() -> Any:
    return get_torch() or require_torch()


@dataclass(frozen=True)
class PolicyObjective:
    """One named policy-optimisation objective. See ``docs/OBJECTIVES.md``."""

    name: str
    advantage: str = "group_norm"
    advantage_eps: float = 1e-8
    ratio: str = "token"
    ratio_clamp: float = 20.0
    clip: str = "clipped"
    clip_low: float = 0.2
    clip_high: float = 0.2
    delta: float | None = None
    is_cap: float = 5.0
    aggregate: str = "seq_mean"
    max_completion_length: int | None = None
    kl: str = "none"
    kl_coef: float = 0.0
    kl_bias_correction: bool = False
    entropy_coef: float = 0.0

    def __post_init__(self) -> None:
        for fname, allowed in (
            ("advantage", ADVANTAGE_KINDS),
            ("ratio", RATIO_KINDS),
            ("clip", CLIP_KINDS),
            ("aggregate", AGGREGATE_KINDS),
            ("kl", KL_KINDS),
        ):
            value = getattr(self, fname)
            if value not in allowed:
                raise ValueError(f"{fname}={value!r}; expected one of {allowed}")
        for fname in (
            "clip_low",
            "clip_high",
            "advantage_eps",
            "kl_coef",
            "entropy_coef",
        ):
            if float(getattr(self, fname)) < 0:
                raise ValueError(f"{fname} must be >= 0")
        for fname in ("is_cap", "ratio_clamp"):
            if float(getattr(self, fname)) <= 0:
                raise ValueError(f"{fname} must be > 0")
        if self.delta is not None and float(self.delta) <= 0:
            raise ValueError("delta must be > 0 or None")
        if (
            self.max_completion_length is not None
            and int(self.max_completion_length) <= 0
        ):
            raise ValueError("max_completion_length must be > 0 or None")
        if self.kl == "none" and self.kl_coef != 0:
            raise ValueError("kl_coef must be 0 when kl='none'")
        if self.kl_bias_correction and self.kl != "k3_token":
            raise ValueError("kl_bias_correction only applies to kl='k3_token'")

    def with_(self, **changes: Any) -> PolicyObjective:
        """Return a copy with ``changes`` applied (re-validated)."""
        return replace(self, **changes)


@dataclass(frozen=True)
class PolicyLossResult:
    """Scalar differentiable ``loss``, detached ``ratio``, float ``metrics``."""

    loss: Any
    ratio: Any
    metrics: dict[str, float] = field(default_factory=dict)


# --- advantages ------------------------------------------------------------


def compute_advantages(rewards: Any, group_ids: Any, objective: PolicyObjective) -> Any:
    """Advantages ``[N]`` from scalar rewards ``[N]`` and integer ``group_ids``.

    Groups of size 1, constant rewards, and non-finite statistics give 0,
    never NaN. Always fp32.
    """
    torch = _t()
    if objective.advantage == "external":
        raise ValueError(
            f"objective {objective.name!r} uses external advantages; "
            "pass them to policy_loss"
        )
    r = torch.as_tensor(rewards).float().reshape(-1)
    g = torch.as_tensor(group_ids).long().reshape(-1)
    if r.shape != g.shape:
        raise ValueError(
            f"rewards {tuple(r.shape)} and group_ids {tuple(g.shape)} differ"
        )
    adv = torch.zeros_like(r)
    if r.numel() == 0:
        return adv
    kind = objective.advantage
    eps = float(objective.advantage_eps)
    for gid in torch.unique(g):
        idx = (g == gid).nonzero(as_tuple=True)[0]
        rg = r[idx]
        if rg.numel() <= 1:
            continue
        if kind == "leave_one_out":
            adv[idx] = rg - (rg.sum() - rg) / (rg.numel() - 1)
        elif kind == "group_norm":
            adv[idx] = rl_losses.group_advantages(rg, normalize=True, eps=eps)
        else:  # group_mean, batch_norm: centre now, batch_norm scales below
            adv[idx] = rl_losses.group_advantages(rg, normalize=False)
    if kind == "batch_norm":
        if r.numel() <= 1:
            return torch.zeros_like(r)
        std = r.std(correction=0)
        if not torch.isfinite(std) or std <= eps:
            return torch.zeros_like(r)
        adv = adv / (std + eps)
    return adv


# --- ratio -----------------------------------------------------------------


def _sequence_log_ratio(logp_cur: Any, logp_old: Any, mask: Any) -> Any:
    torch = _t()
    lengths = torch.clamp(mask.sum(-1), min=1.0)
    if logp_old.dim() == 2:
        return ((logp_cur - logp_old) * mask).sum(-1) / lengths
    return ((logp_cur * mask).sum(-1) - logp_old) / lengths


def _compute_ratio(
    objective: PolicyObjective,
    logp_cur: Any,
    logp_old: Any,
    mask: Any,
    group_ids: Any | None,
) -> Any:
    torch = _t()
    kind = objective.ratio
    clamp = float(objective.ratio_clamp)
    if kind == "token":
        if logp_old.dim() != 2:
            raise ValueError(
                "ratio='token' requires per-token logp_old of shape [N, T]"
            )
        return rl_losses.safe_exp_ratio(logp_cur - logp_old, clamp=clamp)
    if kind == "sequence":
        return rl_losses.safe_exp_ratio(
            _sequence_log_ratio(logp_cur, logp_old, mask), clamp=clamp
        ).unsqueeze(-1)
    if kind == "sequence_token":
        seq = rl_losses.safe_exp_ratio(
            _sequence_log_ratio(logp_cur, logp_old, mask), clamp=clamp
        ).detach()
        # Numerically equal to the sequence ratio; gradient flows per token.
        return seq.unsqueeze(-1) * torch.exp(logp_cur - logp_cur.detach())
    # group_expectation (GEPO)
    if logp_old.dim() != 1:
        raise ValueError(
            "ratio='group_expectation' requires sampler sequence sums logp_old [N]"
        )
    if group_ids is None:
        raise ValueError("ratio='group_expectation' requires group_ids")
    seq_cur = (logp_cur * mask).sum(-1)
    q = logp_old.detach()
    g = torch.as_tensor(group_ids).long().reshape(-1)
    out = torch.zeros_like(seq_cur)
    for gid in torch.unique(g):
        idx = (g == gid).nonzero(as_tuple=True)[0]
        log_e = torch.logsumexp(2 * q[idx], dim=0) - torch.logsumexp(q[idx], dim=0)
        out[idx] = rl_losses.safe_exp_ratio(seq_cur[idx] - log_e, clamp=clamp)
    return out.unsqueeze(-1)


# --- surrogate and aggregation ------------------------------------------------


def surrogate(objective: PolicyObjective, ratio: Any, adv: Any, logp_cur: Any) -> Any:
    """Per-element policy loss (not reduced) for ``objective.clip``."""
    torch = _t()
    if objective.clip == "clipped":
        if objective.delta is None:
            return rl_losses.clipped_surrogate(
                ratio, adv, clip_low=objective.clip_low, clip_high=objective.clip_high
            )
        capped = torch.clamp(ratio, max=float(objective.delta))
        clipped = torch.clamp(
            ratio, 1.0 - objective.clip_low, 1.0 + objective.clip_high
        )
        return -torch.min(capped * adv, clipped * adv)
    if objective.clip == "cispo":
        weight = torch.clamp(ratio, max=float(objective.is_cap)).detach()
        return -weight * adv * logp_cur
    return -ratio * adv


def aggregate(objective: PolicyObjective, per_token: Any, mask: Any) -> Any:
    """Reduce a per-token loss ``[N, T]`` (or ``[N, 1]``) to a scalar."""
    torch = _t()
    per_token = per_token * torch.ones_like(mask)  # broadcast [N,1] -> [N,T]
    if objective.aggregate == "seq_mean":
        return rl_losses.masked_mean(per_token, mask, mode="seq")
    if objective.aggregate == "token_mean":
        return rl_losses.masked_mean(per_token, mask, mode="token")
    if objective.max_completion_length is None:
        raise ValueError(
            f"objective {objective.name!r} aggregates with seq_sum_const and needs "
            "max_completion_length; use objective.with_(max_completion_length=...)"
        )
    n = per_token.shape[0]
    return (per_token * mask).sum() / float(n * int(objective.max_completion_length))


# --- policy loss -------------------------------------------------------------


def policy_loss(
    *,
    logp_cur: Any,
    mask: Any,
    advantages: Any,
    objective: PolicyObjective,
    logp_old: Any | None = None,
    logp_ref: Any | None = None,
    group_ids: Any | None = None,
    kl: Any | None = None,
    entropy: Any | None = None,
) -> PolicyLossResult:
    """Evaluate ``objective`` on one batch.

    ``logp_cur``/``mask`` are ``[N, T]``; ``logp_old`` is ``[N, T]`` per-token
    or ``[N]`` sequence sums (``None`` means ``logp_cur.detach()``);
    ``advantages`` is ``[N]`` or ``[N, T]``; ``logp_ref`` is ``[N, T]`` or
    ``[N]`` sums (only read when ``kl_coef > 0``); ``kl`` is the external
    per-token KL ``[N, T]`` for ``kl='external'``; ``entropy`` is ``[N, T]``.
    """
    torch = _t()
    if logp_cur.dim() != 2:
        raise ValueError("logp_cur must be [N, T]")
    mask = mask.to(logp_cur.dtype)
    if mask.shape != logp_cur.shape:
        raise ValueError("mask must match logp_cur shape")
    old = (
        logp_cur.detach() if logp_old is None else logp_old.detach().to(logp_cur.dtype)
    )
    adv = torch.as_tensor(advantages).to(logp_cur.dtype).detach()
    if adv.dim() == 1:
        adv = adv.unsqueeze(-1)
    elif adv.shape != logp_cur.shape:
        raise ValueError("advantages must be [N] or [N, T]")

    ratio = _compute_ratio(objective, logp_cur, old, mask, group_ids)
    per_token = surrogate(objective, ratio, adv, logp_cur)

    kl_value = 0.0
    kl_seq_term = None
    if objective.kl_coef > 0:
        if objective.kl == "k3_token":
            if logp_ref is None or logp_ref.dim() != 2:
                raise ValueError("kl='k3_token' requires per-token logp_ref [N, T]")
            d = logp_ref.detach().to(logp_cur.dtype) - logp_cur
            k3 = torch.exp(d) - d - 1.0
            if objective.kl_bias_correction:
                k3 = k3 * ratio
            per_token = per_token + objective.kl_coef * k3
            kl_value = float(
                rl_losses.masked_mean(k3.detach(), mask, mode="token").item()
            )
        elif objective.kl == "external":
            if kl is None:
                raise ValueError("kl='external' requires the kl tensor [N, T]")
            per_token = per_token + objective.kl_coef * kl.to(logp_cur.dtype)
            kl_value = float(
                rl_losses.masked_mean(kl.detach(), mask, mode="token").item()
            )
        elif objective.kl == "k3_sequence":
            if logp_ref is None:
                raise ValueError("kl='k3_sequence' requires logp_ref")
            lengths = torch.clamp(mask.sum(-1), min=1.0)
            cur_seq = (logp_cur * mask).sum(-1) / lengths
            ref = logp_ref.detach().to(logp_cur.dtype)
            ref_seq = ((ref * mask).sum(-1) if ref.dim() == 2 else ref) / lengths
            kl_seq_term = rl_losses.k3_kl(cur_seq, ref_seq)
            kl_value = float(kl_seq_term.detach().item())

    policy = aggregate(objective, per_token, mask)
    loss = policy
    if kl_seq_term is not None:
        loss = loss + objective.kl_coef * kl_seq_term

    entropy_value = 0.0
    if entropy is not None:
        ent = rl_losses.masked_mean(entropy.to(logp_cur.dtype), mask, mode="token")
        entropy_value = float(ent.detach().item())
        if objective.entropy_coef > 0:
            loss = loss - objective.entropy_coef * ent

    with torch.no_grad():
        r = ratio.detach() * torch.ones_like(mask)
        m = mask > 0
        if objective.clip == "clipped":
            out = (r < 1.0 - objective.clip_low) | (r > 1.0 + objective.clip_high)
        elif objective.clip == "cispo":
            out = r > objective.is_cap
        else:
            out = torch.zeros_like(m)
        n_tok = float(m.sum().item())
        clip_fraction = float(out[m].float().sum().item() / n_tok) if n_tok else 0.0
        ratio_mean = float(r[m].mean().item()) if n_tok else 1.0
        ratio_max = float(r[m].max().item()) if n_tok else 1.0
        adv_flat = adv.reshape(-1)
        metrics = {
            "policy_loss": float(policy.detach().item()),
            "kl": kl_value,
            "entropy": entropy_value,
            "clip_fraction": clip_fraction,
            "ratio_mean": ratio_mean,
            "ratio_max": ratio_max,
            "advantage_mean": (
                float(adv_flat.mean().item()) if adv_flat.numel() else 0.0
            ),
            "advantage_std": (
                float(adv_flat.std(correction=0).item())
                if adv_flat.numel() > 1
                else 0.0
            ),
        }
    return PolicyLossResult(loss=loss, ratio=ratio.detach(), metrics=metrics)


# --- presets -----------------------------------------------------------------


def _preset(name: str, **kwargs: Any) -> PolicyObjective:
    return PolicyObjective(name=name, **kwargs)


OBJECTIVES: Mapping[str, PolicyObjective] = MappingProxyType(
    {
        "grpo": _preset("grpo", kl="k3_token"),
        "dr_grpo": _preset(
            "dr_grpo", advantage="group_mean", aggregate="seq_sum_const"
        ),
        "bnpo": _preset("bnpo", aggregate="token_mean"),
        "dapo": _preset("dapo", clip_high=0.28, aggregate="token_mean"),
        "gspo": _preset(
            "gspo", ratio="sequence", clip_low=3e-4, clip_high=4e-4, kl="k3_sequence"
        ),
        "gspo_token": _preset(
            "gspo_token",
            ratio="sequence_token",
            clip_low=3e-4,
            clip_high=4e-4,
            kl="k3_sequence",
        ),
        "gepo": _preset("gepo", ratio="group_expectation", ratio_clamp=30.0),
        "rloo": _preset("rloo", advantage="leave_one_out"),
        "reinforce_pp_baseline": _preset(
            "reinforce_pp_baseline", advantage="batch_norm", aggregate="token_mean"
        ),
        "cispo": _preset("cispo", clip="cispo", aggregate="token_mean"),
        "ppo": _preset(
            "ppo", advantage="external", aggregate="token_mean", kl="k3_token"
        ),
    }
)

__all__ = [
    "ADVANTAGE_KINDS",
    "AGGREGATE_KINDS",
    "CLIP_KINDS",
    "KL_KINDS",
    "OBJECTIVES",
    "RATIO_KINDS",
    "PolicyLossResult",
    "PolicyObjective",
    "aggregate",
    "compute_advantages",
    "policy_loss",
    "surrogate",
]
