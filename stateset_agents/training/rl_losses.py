"""Shared, stateless RL loss primitives used by every trainer.

Pure tensor functions. No trainer state, no model calls. torch is fetched
lazily so importing this module never requires torch.
"""

from __future__ import annotations

from typing import Any

from .trainer_utils import get_torch, require_torch


def _t() -> Any:
    return get_torch() or require_torch()


def resolve_logprob_dtype(name: str | None) -> Any:
    """Parse a ``logprob_dtype`` config string into a torch dtype.

    ``None`` (the default) means fp32, the most accurate choice; the other
    names trade numerics for peak memory on large-vocab models. Parse once
    at trainer construction, not per forward pass.
    """
    if name is None:
        return None
    torch = _t()
    key = name.strip().lower()
    dtypes = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
    }
    if key not in dtypes:
        raise ValueError(
            f"unknown logprob_dtype {name!r}; expected one of {sorted(dtypes)}"
        )
    return dtypes[key]


def gather_token_logprobs(
    logits: Any, input_ids: Any, response_mask: Any, *, dtype: Any = None
) -> tuple[Any, Any]:
    """Shift-by-one gather of per-token log-probs, masked to response tokens.

    Returns ``(token_logprobs, shifted_mask)`` both of shape ``[B, T-1]``.
    ``token_logprobs`` is already multiplied by ``shifted_mask``.

    ``dtype`` selects the precision of the log-softmax; ``None`` means fp32,
    which is the most accurate but allocates a full-vocab fp32 tensor.  Pass
    e.g. ``torch.bfloat16`` to trade numerics for peak memory on large-vocab
    models.
    """
    torch = _t()
    shift_logits = logits[..., :-1, :]
    shift_labels = input_ids[..., 1:]
    # The mask is returned to callers that sum it for exact token counts, so
    # it stays fp32 rather than following a possibly low-precision logits
    # dtype (bf16 cannot represent integers past 256 exactly).
    shifted_mask = response_mask[..., 1:].to(torch.float32)
    softmax_dtype = torch.float32 if dtype is None else dtype
    log_probs = torch.log_softmax(shift_logits.to(softmax_dtype), dim=-1)
    token_logprobs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    # Cast the mask to the log-prob dtype for the multiply so the result keeps
    # the requested precision (an fp32 mask would silently promote a bf16
    # gather back to fp32); the returned mask stays fp32 for exact counts.
    return token_logprobs * shifted_mask.to(token_logprobs.dtype), shifted_mask


def safe_exp_ratio(log_ratio: Any, *, clamp: float = 20.0) -> Any:
    """``exp(log_ratio)`` with the exponent clamped to ``[-clamp, +clamp]``.

    A raw ``exp`` of a log-ratio overflows to ``inf`` past ~88 in fp32, and an
    ``inf`` ratio makes the surrogate loss (and every gradient in the batch)
    non-finite. Early in training, or after a bad update, individual token
    log-ratios genuinely do reach those magnitudes. Clamping first keeps the
    ratio finite; the clamped region has zero gradient, which is the right
    behaviour anyway since such a sample is far outside any trust region.
    ``exp(±20) ~ 5e8 / 2e-9`` is already several orders of magnitude beyond
    every clip boundary, so in-region samples are untouched.
    """
    torch = _t()
    return torch.exp(torch.clamp(log_ratio, min=-clamp, max=clamp))


def masked_mean(x: Any, mask: Any, *, mode: str = "token") -> Any:
    """Mean of ``x`` over positions where ``mask`` is 1.

    ``mode="token"``: one global mean over all masked tokens (DAPO style).
    ``mode="seq"``: per-row mean, then mean over rows (GRPO/GSPO style).
    An all-zero mask yields 0, never NaN.
    """
    torch = _t()
    mask = mask.to(x.dtype)
    if mode == "token":
        return (x * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    if mode == "seq":
        per_row = (x * mask).sum(-1) / torch.clamp(mask.sum(-1), min=1.0)
        return per_row.mean()
    raise ValueError(f"unknown mode {mode!r}; expected 'token' or 'seq'")


def group_advantages(rewards: Any, *, normalize: bool = True, eps: float = 1e-8) -> Any:
    """Group-relative advantages for one group of rewards ``[G]``.

    Groups of size 1, constant rewards, or non-finite statistics yield zeros
    rather than NaN (a NaN advantage silently poisons the whole batch).
    """
    torch = _t()
    rewards = rewards.float()
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    adv = rewards - rewards.mean()
    if not normalize:
        return adv
    std = adv.std(correction=0)
    if not torch.isfinite(std) or std <= eps:
        return torch.zeros_like(rewards)
    return adv / (std + eps)


def clipped_surrogate(
    ratio: Any, advantages: Any, *, clip_low: float, clip_high: float
) -> Any:
    """PPO/GSPO/DAPO clipped surrogate *loss* (elementwise, not reduced).

    ``-min(r·A, clip(r)·A)``. When the ratio leaves the trust region on the
    side the advantage would push it, the clipped branch is selected and,
    because ``clamp`` has zero gradient there, the sample contributes no
    gradient — that is the mechanism that bounds the policy step.
    """
    torch = _t()
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    return -torch.min(ratio * advantages, clipped * advantages)


def sequence_ratio(logp_cur: Any, logp_old: Any, mask: Any) -> Any:
    """GSPO length-normalised sequence importance ratio, one value per row."""
    torch = _t()
    mask = mask.to(logp_cur.dtype)
    log_ratio = ((logp_cur - logp_old) * mask).sum(-1) / torch.clamp(
        mask.sum(-1), min=1.0
    )
    return torch.exp(log_ratio)


def clip_fraction(ratio: Any, *, clip_low: float, clip_high: float) -> float:
    """Fraction of ratios outside the trust region (for logging)."""
    out = (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
    return float(out.float().mean().item()) if ratio.numel() else 0.0


def k3_kl(logp_cur: Any, logp_ref: Any, mask: Any | None = None) -> Any:
    """Schulman's k3 estimator of KL(π_cur ‖ π_ref) from sampled log-probs.

    ``k3 = exp(r) − r − 1`` with ``r = log π_ref − log π_cur``. It is
    non-negative, unbiased, and — unlike the naive ``log π_cur − log π_ref``
    — has a gradient whose expectation is the true KL gradient, so the
    penalty actually pulls the policy toward the reference.
    """
    torch = _t()
    r = logp_ref.detach() - logp_cur
    k3 = torch.exp(r) - r - 1.0
    if mask is None:
        return k3.mean()
    return masked_mean(k3, mask, mode="seq")
