"""Shared, stateless RL loss primitives used by every trainer.

Pure tensor functions. No trainer state, no model calls. torch is fetched
lazily so importing this module never requires torch.
"""

from __future__ import annotations

import math
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

    Positions with zero weight contribute zero score and zero logit gradient,
    including when their logits or target indices are undefined.
    The log-softmax defaults to float32. Set ``dtype`` to float64 for a
    double-precision model or bfloat16 to reduce peak vocabulary memory.
    """
    torch = _t()
    shift_logits = logits[..., :-1, :]
    shift_labels = input_ids[..., 1:]
    # The mask is returned to callers that sum it for exact token counts, so
    # it stays fp32 rather than following a possibly low-precision logits
    # dtype (bf16 cannot represent integers past 256 exactly).
    shifted_mask = response_mask[..., 1:].to(torch.float32)
    softmax_dtype = torch.float32 if dtype is None else dtype
    # Center before the reduction so a large common logit cannot round away
    # the log partition. Gather just the selected actions. The offset cancels
    # from log-softmax, so it requires no gradient.
    scores = shift_logits.to(softmax_dtype)
    scores = scores - scores.amax(-1, keepdim=True).detach()
    # An unused row can contain undefined logits, for example from padded
    # attention. Replace it before logsumexp so its backward is finite too.
    # The centered tensor above is fresh and can be masked in place.
    inactive = shifted_mask == 0
    scores.masked_fill_(inactive.unsqueeze(-1), 0)
    labels = shift_labels.masked_fill(inactive, 0)
    selected = scores.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = selected - torch.logsumexp(scores, dim=-1)
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
    if log_ratio.dtype in (torch.float16, torch.bfloat16):
        log_ratio = log_ratio.float()
    return torch.exp(torch.clamp(log_ratio, min=-clamp, max=clamp))


def masked_mean(x: Any, mask: Any, *, mode: str = "token") -> Any:
    """Mean of ``x`` over positions where ``mask`` is 1.

    ``mode="token"``: one global mean over all masked tokens (DAPO style).
    ``mode="seq"``: per-row mean, then mean over rows (GRPO/GSPO style).
    An all-zero mask yields 0, never NaN.
    """
    torch = _t()
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    mask = mask.to(x.dtype)
    if mode == "token":
        return (x * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    if mode == "seq":
        per_row = (x * mask).sum(-1) / torch.clamp(mask.sum(-1), min=1.0)
        return per_row.mean() if per_row.numel() else per_row.sum()
    raise ValueError(f"unknown mode {mode!r}; expected 'token' or 'seq'")


def _bellman_columns(rewards: Any, dones: Any, n: int) -> tuple[Any, Any]:
    """Validate one immediate reward and one terminal flag per transition."""
    torch = _t()
    for tensor in (rewards, dones):
        if tensor.shape not in ((n,), (n, 1)):
            raise ValueError("Bellman inputs must contain one scalar per transition")
    if dones.dtype != torch.bool and not ((dones == 0) | (dones == 1)).all():
        raise ValueError("Bellman terminal flags must be boolean or binary")
    return rewards.reshape(n, 1), dones.reshape(n, 1).bool()


def _narrow_backup(total: Any, reward: Any, future: Any, gamma: Any, dtype: Any) -> Any:
    """Resolve narrow-format midpoint decisions before the final critic cast."""
    torch = _t()
    # Treat overflow as rounding to the next power of two. Both endpoints of
    # every finite rounding interval (and their midpoint) fit exactly in fp64.
    limit = 2.0 ** math.frexp(torch.finfo(dtype).max)[1]

    def endpoint(x: Any) -> Any:
        return x.to(dtype).double().clamp(-limit, limit)

    d = total.detach()
    candidate = endpoint(d)
    # Near a boundary, reflecting around d finds the competing endpoint even
    # when the native half/bfloat16 conversion goes through float32.
    other = endpoint(d + (d - candidate))
    midpoint = (candidate + other) * 0.5
    r, v = reward.detach(), future.detach()
    diff = r - midpoint
    # FastTwoSum, with the larger magnitude first. diff + tail = r - midpoint.
    tail = torch.where(
        r.abs() >= midpoint.abs(), (r - diff) - midpoint, r - (midpoint + diff)
    )
    residual = torch.addcmul(diff, v, gamma) + tail
    change = (other != candidate) & torch.where(
        other > candidate, residual > 0, residual < 0
    )
    # Move halfway toward the selected endpoint, inside its rounding interval.
    # A detached correction preserves the Bellman derivative. Subtracting +0
    # also preserves a negative zero when no correction is needed.
    correction = torch.where(change, (d - other) * 0.5, 0.0)
    return (total - correction).to(dtype)


def _discounted_backup(
    reward: Any, future: Any, discount: float, terminal: Any = None
) -> Any:
    """Add the discounted continuation while retaining cancellation in the TD target."""
    torch = _t()
    # The public scalar-mask operation historically promoted an integral future
    # to the default float. Preserve its output dtype without rounding its input.
    future_dtype = future.dtype
    if terminal is not None:
        if future.is_floating_point() or future.is_complex():
            future = torch.where(terminal, 0.0, future)
            terminal = None
        else:
            future_dtype = torch.get_default_dtype()
    dtype = torch.get_default_dtype()
    if not (reward.is_complex() or future.is_complex()):
        if reward.is_floating_point() or future_dtype.is_floating_point:
            dtype = torch.promote_types(reward.dtype, future_dtype)
        if not (reward.is_floating_point() and future.is_floating_point()):
            from .integer_bellman import IntegerBellman

            return IntegerBellman.apply(
                reward, future, float(discount), dtype, terminal
            )
    device = reward.device
    work = "cpu" if device.type == "mps" else device
    # The configured discount is binary64, including when the critic uses float32.
    gamma = torch.scalar_tensor(discount, dtype=torch.float64, device=work)
    r, v = reward.to(work, torch.float64), future.to(work, torch.float64)
    backed_up = torch.addcmul(r, v, gamma)
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        backed_up = _narrow_backup(backed_up, r, v, gamma, dtype)
    return backed_up.to(device, dtype)


def bellman_target(rewards: Any, dones: Any, next_values: Any, discount: float) -> Any:
    """One TD target per transition, returned as a column to match scalar critics."""
    torch = _t()
    if next_values.ndim == 0:
        raise ValueError("Bellman inputs must contain one scalar per transition")
    n = next_values.shape[0]
    if next_values.shape not in ((n,), (n, 1)):
        raise ValueError("Bellman inputs must contain one scalar per transition")
    reward, terminal = _bellman_columns(rewards, dones, n)
    if discount == 0:
        terminal = torch.ones_like(terminal)
    return _discounted_backup(reward, next_values.reshape(n, 1), discount, terminal)


def td_regression(predicted: Any, target: Any) -> Any:
    """Average the squared TD errors with one detached scalar target per critic row."""
    torch = _t()
    if predicted.shape != target.shape or not predicted.numel():
        raise ValueError("TD predictions and targets must have the same nonempty shape")
    device = predicted.device
    work = "cpu" if device.type == "mps" else device
    residual = predicted.to(work, torch.float64) - target.detach().to(
        work, torch.float64
    )
    loss = ((residual / residual.numel()) * residual).sum()
    return loss.to(device, torch.float32) if device.type == "mps" else loss


def clipped_value_error(
    values: Any, old_values: Any, returns: Any, clip: float | None, *, scale: Any = 1
) -> Any:
    """Pessimistic squared critic error at a frozen rollout value and return."""
    torch = _t()
    old, target = old_values.detach(), returns.detach()
    raw = ((values - target) * scale).square()
    if clip is None:
        return raw
    clipped = values.clamp(min=old - clip, max=old + clip)
    return torch.maximum(raw, ((clipped - target) * scale).square())


def critic_value_loss(
    values: Any,
    old_values: Any,
    returns: Any,
    mask: Any,
    clip: float | None,
    *,
    half_square: bool = False,
    square_mask: bool = False,
) -> Any:
    """Reduce scaled residuals so a finite mean does not require a finite raw square."""
    torch = _t()
    dtype = torch.promote_types(
        torch.float32, torch.promote_types(values.dtype, returns.dtype)
    )
    work = dtype if values.device.type == "mps" else torch.float64
    factor = 0.5 if half_square else 1.0
    if mask is None:
        if values.numel() == 0:
            return values.to(dtype).mean()
        scale = math.sqrt(factor / values.numel())
    else:
        inactive = mask == 0
        values = values.masked_fill(inactive, 0)
        old_values = old_values.masked_fill(inactive, 0)
        returns = returns.masked_fill(inactive, 0)
        weights = mask.to(work)
        unit = (factor / weights.sum().clamp_min(1)).sqrt()
        scale = unit * (weights if square_mask else weights.sqrt())
    error = clipped_value_error(
        values.to(work), old_values.to(work), returns.to(work), clip, scale=scale
    )
    return error.sum().to(dtype)


def group_advantages(rewards: Any, *, normalize: bool = True, eps: float = 1e-8) -> Any:
    """Group-relative advantages for one group of rewards ``[G]``.

    Groups of size 1, constant rewards, or non-finite statistics yield zeros
    rather than NaN (a NaN advantage silently poisons the whole batch).
    """
    from .reward_advantages import grouped_advantages

    torch = _t()
    ids = torch.zeros(rewards.numel(), dtype=torch.long)
    kind = "group_norm" if normalize else "group_mean"
    return grouped_advantages(rewards, ids, kind, eps).reshape(rewards.shape)


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
    if logp_cur.dtype in (torch.float16, torch.bfloat16):
        logp_cur = logp_cur.float()
    mask = mask.to(logp_cur.dtype)
    log_ratio = ((logp_cur - logp_old) * mask).sum(-1) / torch.clamp(
        mask.sum(-1), min=1.0
    )
    return safe_exp_ratio(log_ratio)


def sequence_token_ratio(sequence_weight: Any, logp_cur: Any) -> Any:
    """Give each action the frozen sequence ratio as its local log-prob gradient."""
    weight = sequence_weight.detach()
    if weight.ndim + 1 == logp_cur.ndim:
        weight = weight.unsqueeze(-1)
    return weight * _t().exp(logp_cur - logp_cur.detach())


def clip_fraction(ratio: Any, *, clip_low: float, clip_high: float) -> float:
    """Fraction of ratios outside the trust region (for logging)."""
    out = (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
    return float(out.float().mean().item()) if ratio.numel() else 0.0


def k3_term(log_ratio: Any) -> Any:
    """Evaluate exp(d) - 1 - d and its small nonzero slope near equality."""
    from .causal_likelihood import _small_kl_gap

    torch = _t()
    if log_ratio.dtype in (torch.float16, torch.bfloat16):
        log_ratio = log_ratio.float()
    small = log_ratio.abs() < 0.001
    near = torch.where(small, log_ratio, 0.0)
    far = torch.where(small, 0.0, log_ratio)
    return torch.where(small, _small_kl_gap(near), torch.expm1(far) - far)


def _scaled_k3_tail(gap: Any, weight: Any) -> Any:
    """Put the weight inside the exponential before evaluating a large gap."""
    magnitude = weight.abs().masked_fill(weight == 0, 1)
    return weight.sign() * (gap + magnitude.log()).exp() - weight * (1 + gap)


def _mean_k3(gap: Any, mask: Any | None = None) -> Any:
    """Reduce FP64 k3 terms with the sample weight inside large exponentials."""
    torch = _t()
    if mask is not None:
        gap, mask = torch.broadcast_tensors(gap, mask.to(gap))
        gap = gap.masked_fill(mask == 0, 0)
    if not gap.numel():
        return gap.sum()
    if mask is None:
        weight = gap.new_tensor(1 / gap.numel())
    else:
        rows = math.prod(gap.shape[:-1])
        weight = (mask / mask.sum(-1, keepdim=True).clamp_min(1)) / rows
    # A token's exponential can exceed FP64 even when its weighted
    # contribution and derivative both fit. Only evaluate the large branch
    # at active large gaps, so an unused infinity cannot enter backward.
    large = gap > 700
    regular = weight * k3_term(gap.masked_fill(large, 0))
    far = torch.where(large, gap, 0.0)
    terms = torch.where(large, _scaled_k3_tail(far, weight), regular)
    total = terms.sum()
    if total.detach().isposinf() and (weight >= 0).all():
        # When every observation fits, a nonnegative mean is bounded by the
        # largest one. Rounded reciprocal weights can push the sum above
        # FP64 maximum (e.g. 22 copies). Retain the weighted term gradients.
        with torch.no_grad():
            observations = k3_term(gap)
            peak = observations.max()
            if not peak.isfinite():
                return total
            bounded = ((observations / peak) * weight).sum().clamp_max(1) * peak
        return bounded + (terms - terms.detach()).sum()
    return total


def k3_kl(logp_cur: Any, logp_ref: Any, mask: Any | None = None) -> Any:
    """Schulman's k3 estimator of KL(π_cur ‖ π_ref) from sampled log-probs.

    ``k3 = exp(r) − r − 1`` with ``r = log π_ref − log π_cur``. It is
    nonnegative. For common support and on-policy samples its expectation
    equals the forward KL value. Differentiating fixed samples holds the
    sampling probabilities constant.
    """
    torch = _t()
    device = "cpu" if logp_cur.device.type == "mps" else logp_cur.device
    current = logp_cur.to(device=device, dtype=torch.float64)
    reference = logp_ref.detach().to(current)
    dtype = torch.promote_types(
        torch.float32, torch.promote_types(logp_cur.dtype, logp_ref.dtype)
    )
    return _mean_k3(reference - current, mask).to(device=logp_cur.device, dtype=dtype)
