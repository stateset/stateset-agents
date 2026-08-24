"""Shared, stateless RL loss primitives used by every trainer.

Pure tensor functions. No trainer state, no model calls. torch is fetched
lazily so importing this module never requires torch.
"""
from __future__ import annotations

from typing import Any

from .trainer_utils import get_torch, require_torch


def _t() -> Any:
    return get_torch() or require_torch()


def gather_token_logprobs(logits: Any, input_ids: Any, response_mask: Any) -> tuple[Any, Any]:
    """Shift-by-one gather of per-token log-probs, masked to response tokens.

    Returns ``(token_logprobs, shifted_mask)`` both of shape ``[B, T-1]``.
    ``token_logprobs`` is already multiplied by ``shifted_mask``.
    """
    torch = _t()
    shift_logits = logits[..., :-1, :]
    shift_labels = input_ids[..., 1:]
    shifted_mask = response_mask[..., 1:].to(shift_logits.dtype)
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_logprobs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_logprobs * shifted_mask, shifted_mask


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
