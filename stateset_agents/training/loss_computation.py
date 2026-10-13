"""
GRPO loss computation functions.

This module provides the core loss computation functions for GRPO training,
including standard GRPO loss and enhanced GRPO loss with KL penalty.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import re
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..core.chat_masking import assistant_char_spans
from ..exceptions import LOSS_EXCEPTIONS
from . import rl_losses  # noqa: F401  (re-exported; tests spy on it)
from . import objectives
from .trainer_utils import get_amp, get_functional, get_torch, require_torch

logger = logging.getLogger(__name__)

__all__ = [
    "LOSS_EXCEPTIONS",
    "compute_grpo_loss",
    "compute_enhanced_grpo_loss",
    "compute_ppo_ratio",
    "compute_entropy_bonus",
]

_warned_missing_log_probs = False
_warned_missing_assistant_mask = False


def _resolve_model_device(agent: Any, torch_mod: Any) -> Any:
    """Resolve the device for model tensors with safe CPU fallback."""
    device = getattr(agent.model, "device", None)
    if device is None and hasattr(agent.model, "parameters"):
        try:
            first_param = next(agent.model.parameters())
            device = getattr(first_param, "device", None)
        except StopIteration:
            device = None
    if device is None:
        device = torch_mod.device("cpu")
    return device


def _grpo_objective(config: Any) -> Any:
    """GRPO trainers score a trajectory as summed log-probs, so the ratio is
    sequence-level and clipped at ``seq_clip_ratio`` (GSPO scale)."""
    seq_clip = float(getattr(config, "seq_clip_ratio", 3e-4))
    return objectives.resolve_objective(
        config,
        "gspo",
        max_completion_length=int(getattr(config, "max_completion_length", 0) or 0)
        or None,
        supported_ratios=("sequence", "sequence_token"),
        name="grpo_sequence",
        clip_low=seq_clip,
        clip_high=seq_clip,
        kl="none",
    )


def _clipped_trajectory_loss(
    nll: Any, token_count: int, advantage: Any, old_log_prob: Any, objective: Any
) -> Any:
    """Clipped sequence-ratio loss for one trajectory via ``objectives.policy_loss``.

    ``nll`` is the model's mean per-token NLL (differentiable); ``old_log_prob``
    the rollout-time summed log-prob. The trajectory is presented as
    ``token_count`` identical tokens so the sequence ratio normalises by the
    true length: ``exp((-nll * T - old) / T)``, exactly the previous
    ``compute_ppo_ratio`` quantity, and ``seq_mean`` over identical tokens is
    the row value.
    """
    torch = require_torch()
    count = max(int(token_count), 1)
    logp_cur = (-nll).reshape(1, 1).expand(1, count)
    mask = torch.ones(1, count, dtype=logp_cur.dtype, device=logp_cur.device)
    adv = (
        advantage.reshape(1)
        if torch.is_tensor(advantage)
        else torch.tensor([float(advantage)], device=logp_cur.device)
    )
    result = objectives.policy_loss(
        logp_cur=logp_cur,
        mask=mask,
        advantages=adv.to(logp_cur.dtype),
        objective=objective,
        logp_old=old_log_prob.reshape(1).to(logp_cur.dtype),
    )
    return result.loss


# --- batched per-token path ---------------------------------------------------


def _token_rows_for_group(group: Any) -> list[tuple[list[int], list[int], int]] | None:
    """Rows ``(prompt_ids, response_ids, trajectory_index)`` for every assistant
    turn in ``group``; ``None`` when any trajectory lacks token metadata (the
    caller then takes the sequence-level path)."""
    rows: list[tuple[list[int], list[int], int]] = []
    for ti, trajectory in enumerate(getattr(group, "trajectories", [])):
        found = False
        for turn in getattr(trajectory, "turns", []):
            role = (
                turn.get("role")
                if isinstance(turn, dict)
                else getattr(turn, "role", None)
            )
            if role != "assistant":
                continue
            md = (
                turn.get("metadata")
                if isinstance(turn, dict)
                else getattr(turn, "metadata", None)
            )
            md = md or {}
            prompt_ids, token_ids = md.get("prompt_token_ids"), md.get("token_ids")
            if not prompt_ids or not token_ids:
                return None
            rows.append(([int(x) for x in prompt_ids], [int(x) for x in token_ids], ti))
            found = True
        if not found:
            return None
    return rows or None


def _token_objective(config: Any) -> Any:
    """Native token-path objective: the ``grpo`` preset with ``clip_ratio`` as
    the symmetric trust region; ``config.objective`` selects any preset."""
    clip = float(getattr(config, "clip_ratio", getattr(config, "clip_epsilon", 0.2)))
    return objectives.resolve_objective(
        config,
        "grpo",
        max_completion_length=int(getattr(config, "max_completion_length", 0) or 0)
        or None,
        clip_low=clip,
        clip_high=clip,
        kl="none",
    )


def _token_kl_objective(objective: Any, config: Any, *, beta: float) -> Any:
    """Enable the KL penalty with coefficient ``beta`` on the token path.

    An objective chosen through ``config.objective`` keeps its own estimator
    (``gspo`` regularizes the length-normalized sequence log-ratio); the native
    objective, which has none, uses per-token k3.
    """
    kind = objective.kl
    if getattr(config, "objective", None) is None or kind == "none":
        kind = "k3_token"
    return objective.with_(kl=kind, kl_coef=float(beta))


def _legacy_group_advantages(
    rewards: Any,
    config: Any,
    *,
    global_reward_mean: float | None = None,
    update_global_stats: Callable[[float, int], None] | None = None,
    normalize: bool | None = None,
) -> Any:
    """``group_median`` / ``global_mean`` baselines, shared by both GRPO paths.

    ``global_mean`` subtracts the trainer's running mean (and reports this
    group's mean to it); without trainer statistics it uses the group mean.
    With ``advantage_normalization`` the centred advantages are scaled by their
    population std, as on the sequence path.
    """
    torch = require_torch()
    baseline_type = getattr(config, "baseline_type", "group_mean")
    if baseline_type == "global_mean" and update_global_stats is not None:
        update_global_stats(float(rewards.mean().item()), len(rewards))
    if baseline_type == "group_median":
        baseline = rewards.median()
    elif baseline_type == "global_mean" and global_reward_mean is not None:
        baseline = torch.tensor(
            global_reward_mean, dtype=rewards.dtype, device=rewards.device
        )
    else:
        baseline = rewards.mean()
    advantages = rewards - baseline
    if normalize is None:
        normalize = bool(getattr(config, "advantage_normalization", True))
    return _scale_by_group_std(advantages) if normalize else advantages


def _scale_by_group_std(advantages: Any) -> Any:
    """Divide by the group's population std without re-centring, so a
    median or running-mean baseline survives normalisation (the std of
    ``rewards - baseline`` equals the std of the rewards)."""
    torch = require_torch()
    if advantages.numel() <= 1:
        return advantages
    std = advantages.std(correction=0)
    if torch.isfinite(std) and std > 0:
        return advantages / (std + 1e-8)
    return advantages


def _sequence_group_advantages(
    rewards: Any,
    config: Any,
    *,
    global_reward_mean: float | None = None,
    update_global_stats: Callable[[float, int], None] | None = None,
) -> Any:
    """Group advantages for both sequence-level GRPO losses.

    Applies ``reward_clip``, then the ``objective`` preset's estimator when one
    is configured, else ``baseline_type`` with optional std scaling.
    """
    torch = require_torch()
    reward_clip = getattr(config, "reward_clip", None)
    if reward_clip is not None:
        rewards = torch.clamp(rewards, min=-float(reward_clip), max=float(reward_clip))
    zeros = torch.zeros_like(rewards, dtype=torch.long)
    if getattr(config, "objective", None) is not None:
        # The preset owns the estimator, including whether it normalises.
        return objectives.compute_advantages(
            rewards, zeros, _grpo_objective(config).with_(advantage_eps=1e-8)
        )
    baseline_type = getattr(config, "baseline_type", "group_mean")
    normalize = bool(getattr(config, "advantage_normalization", True))
    if baseline_type in ("group_median", "global_mean"):
        return _legacy_group_advantages(
            rewards,
            config,
            global_reward_mean=global_reward_mean,
            update_global_stats=update_global_stats,
            normalize=normalize,
        )
    if baseline_type == "leave_one_out":
        # RLOO: r_i minus the mean of the other rewards (0 for one sample).
        advantages = objectives.compute_advantages(
            rewards, zeros, objectives.OBJECTIVES["rloo"]
        )
    else:  # group_mean (default)
        advantages = rewards - rewards.mean()
    return _scale_by_group_std(advantages) if normalize else advantages


def _token_advantages(
    rewards: Any,
    config: Any,
    objective: Any,
    *,
    global_reward_mean: float | None = None,
    update_global_stats: Callable[[float, int], None] | None = None,
) -> Any:
    """Group advantages for the token path: the configured preset's estimator,
    or the legacy ``baseline_type``/``advantage_normalization`` switches."""
    torch = require_torch()
    zeros = torch.zeros_like(rewards, dtype=torch.long)
    if getattr(config, "objective", None) is not None:
        return objectives.compute_advantages(rewards, zeros, objective)
    baseline_type = getattr(config, "baseline_type", "group_mean")
    if baseline_type in ("group_median", "global_mean"):
        return _legacy_group_advantages(
            rewards,
            config,
            global_reward_mean=global_reward_mean,
            update_global_stats=update_global_stats,
        )
    normalize = bool(getattr(config, "advantage_normalization", True))
    if baseline_type == "leave_one_out":
        kind = "leave_one_out"
    elif normalize:
        kind = "group_norm"
    else:
        kind = "group_mean"
    return objectives.compute_advantages(
        rewards, zeros, objectives.OBJECTIVES["grpo"].with_(advantage=kind)
    )


def _response_token_logprobs(
    logits: Any, input_ids: Any, response_mask: Any, *, want_entropy: bool
) -> tuple[Any, Any, Any | None]:
    """Shift-by-one token log-probs computed at response positions only.

    Same contract as :func:`rl_losses.gather_token_logprobs` (``[B, T-1]``
    log-probs, zero off the mask, plus the fp32 shifted mask), but only the
    ``n`` response positions are upcast and normalised: the largest tensor is
    ``[n, V]`` in fp32 instead of a full ``[B, T-1, V]`` fp32 log-softmax.
    Without entropy the log-prob is ``logit[y] - logsumexp(logits)``; with it
    one ``log_softmax`` over the selected rows serves both. ``entropy`` is the
    per-token entropy at response positions (zero elsewhere) or ``None``.
    """
    torch = require_torch()
    shifted_mask = response_mask[..., 1:].to(torch.float32)
    selected = shifted_mask > 0
    dtype = torch.promote_types(logits.dtype, torch.float32)
    labels = input_ids[..., 1:][selected].unsqueeze(-1)
    picked = logits[..., :-1, :][selected].to(dtype)  # [n, V]
    if want_entropy:
        log_probs = torch.log_softmax(picked, dim=-1)
        token = log_probs.gather(-1, labels).squeeze(-1)
        ent = -(log_probs.exp() * log_probs).sum(-1)
    else:
        token = picked.gather(-1, labels).squeeze(-1) - torch.logsumexp(picked, dim=-1)
        ent = None
    zeros = torch.zeros(shifted_mask.shape, dtype=dtype, device=logits.device)
    token_logprobs = zeros.index_put((selected,), token)
    entropy = None if ent is None else zeros.index_put((selected,), ent)
    return token_logprobs, shifted_mask, entropy


def _forward_token_chunk(
    model: Any,
    chunk: list[tuple[list[int], list[int], int]],
    width: int,
    device: Any,
    *,
    grad: bool,
    want_entropy: bool = False,
) -> tuple[Any, Any, Any | None]:
    """Pad ``chunk`` to ``width`` and run one forward pass.

    Returns ``(token_logprobs, response_mask, entropy)``, each ``[len(chunk),
    width-1]``; see :func:`_response_token_logprobs`. The full-vocabulary
    logits are released on return; only the response-position graph is kept.
    """
    torch = require_torch()
    ids = torch.zeros(len(chunk), width, dtype=torch.long, device=device)
    attn = torch.zeros(len(chunk), width, dtype=torch.long, device=device)
    resp = torch.zeros(len(chunk), width, dtype=torch.float32, device=device)
    for i, (p, r, _) in enumerate(chunk):
        seq = torch.tensor(p + r, dtype=torch.long, device=device)
        ids[i, : seq.numel()] = seq
        attn[i, : seq.numel()] = 1
        resp[i, len(p) : seq.numel()] = 1.0
    with torch.set_grad_enabled(grad):
        logits = model(input_ids=ids, attention_mask=attn).logits
        return _response_token_logprobs(logits, ids, resp, want_entropy=want_entropy)


def _forward_token_rows(
    model: Any,
    rows: list[tuple[list[int], list[int], int]],
    device: Any,
    chunk_size: int,
    *,
    grad: bool,
    want_entropy: bool = False,
) -> tuple[Any, Any, Any | None]:
    """Pad ``rows`` and run ``model`` in chunks of ``chunk_size`` rows.

    Returns ``(token_logprobs [R, W-1], response_mask [R, W-1], entropy)``
    where ``entropy`` is the differentiable per-token entropy (or ``None``).
    Graphs are retained across chunks when ``grad`` is True; the training
    loss uses :func:`_forward_token_chunk` per chunk instead.
    """
    torch = require_torch()
    width = max(len(p) + len(r) for p, r, _ in rows)
    size = max(int(chunk_size), 1)
    lp_chunks, mask_chunks, ent_chunks = [], [], []
    for start in range(0, len(rows), size):
        lp, m, ent = _forward_token_chunk(
            model,
            rows[start : start + size],
            width,
            device,
            grad=grad,
            want_entropy=want_entropy,
        )
        lp_chunks.append(lp)
        mask_chunks.append(m)
        ent_chunks.append(ent)
    entropy = torch.cat(ent_chunks) if want_entropy else None
    return torch.cat(lp_chunks), torch.cat(mask_chunks), entropy


def compute_token_old_logprobs(
    trajectory_groups: list[Any], config: Any, agent: Any
) -> list[Any] | None:
    """Freeze the old policy's per-token log-probs for every group (one no-grad
    chunked forward on the stored token ids), for multiple inner updates.

    Returns one detached ``[rows, width-1]`` tensor per non-empty group, or
    ``None`` when any group lacks token metadata (the sequence fallback has no
    per-token old log-probs; inner updates are then not applied).
    """
    torch = require_torch()
    groups = [g for g in trajectory_groups if getattr(g, "trajectories", None)]
    rows = [_token_rows_for_group(g) for g in groups]
    if not groups or any(r is None for r in rows):
        return None
    device = _resolve_model_device(agent, torch)
    chunk = int(getattr(config, "generation_batch_size", 0) or 0) or max(
        len(r) for r in rows if r
    )
    snapshots = []
    for group_rows in rows:
        assert group_rows is not None
        lp, _, _ = _forward_token_rows(
            agent.model, group_rows, device, chunk, grad=False
        )
        snapshots.append(lp.detach())
    return snapshots


def sampler_old_logprobs(trajectory_groups: list[Any], device: Any) -> list[Any] | None:
    """The log-probs recorded at sampling time (``sampler_log_probs`` on each
    assistant turn), laid out like :func:`_forward_token_rows` output: one
    ``[rows, width-1]`` tensor per non-empty group with response token ``k``
    of a row at column ``len(prompt) - 1 + k`` and zeros elsewhere.

    Returns ``None`` when any row lacks sampler log-probs; raises when a row's
    log-probs disagree in length with its token ids (corrupt metadata, never
    silently misaligned).
    """
    torch = require_torch()
    groups = [g for g in trajectory_groups if getattr(g, "trajectories", None)]
    out: list[Any] = []
    for group in groups:
        rows = _token_rows_for_group(group)
        if rows is None:
            return None
        per_row: list[list[float]] = []
        for turn in _assistant_turns(group):
            md = turn.get("metadata") if isinstance(turn, dict) else turn.metadata
            lp = (md or {}).get("sampler_log_probs")
            if lp is None:
                return None
            per_row.append([float(x) for x in lp])
        if len(per_row) != len(rows):
            return None
        width = max(len(p) + len(r) for p, r, _ in rows)
        tensor = torch.zeros(len(rows), width - 1, dtype=torch.float32, device=device)
        for i, ((p, r, _), lp) in enumerate(zip(rows, per_row, strict=True)):
            if len(lp) != len(r):
                raise ValueError(
                    f"sampler_log_probs has {len(lp)} entries for {len(r)} response "
                    "tokens; rollout metadata is corrupt"
                )
            start = len(p) - 1
            tensor[i, start : start + len(r)] = torch.tensor(lp, dtype=torch.float32)
        out.append(tensor)
    return out or None


def _assistant_turns(group: Any) -> list[Any]:
    """Assistant turns of ``group`` in :func:`_token_rows_for_group` row order."""
    turns: list[Any] = []
    for trajectory in getattr(group, "trajectories", []):
        for turn in getattr(trajectory, "turns", []):
            role = (
                turn.get("role")
                if isinstance(turn, dict)
                else getattr(turn, "role", None)
            )
            if role == "assistant":
                turns.append(turn)
    return turns


class _WeightedMean:
    """Running weighted mean for per-chunk metrics."""

    __slots__ = ("total", "weight")

    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0.0

    def add(self, value: float, weight: float) -> None:
        self.total += float(value) * float(weight)
        self.weight += float(weight)

    def value(self, default: float) -> float:
        return self.total / self.weight if self.weight > 0 else default


def _compute_token_path_loss(
    trajectory_groups: list[Any],
    rows_per_group: list[list[tuple[list[int], list[int], int]]],
    config: Any,
    agent: Any,
    *,
    beta: float = 0.0,
    reference_model: Any | None = None,
    old_logprobs: list[Any] | None = None,
    global_reward_mean: float | None = None,
    update_global_stats: Callable[[float, int], None] | None = None,
    backward_fn: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Batched per-token GRPO over stored token ids for every group.

    The loss is the mean over groups of each group's ``policy_loss``. It is
    evaluated one row chunk (``generation_batch_size`` rows) at a time: each
    chunk's share is computed with the group's row/token counts as the
    aggregation denominators (``policy_loss(total_rows=, total_tokens=)``)
    and divided by the group count, so the shares sum exactly to the loss.

    With ``backward_fn`` every share is handed to it as soon as it is
    computed (the trainer scales and backpropagates it), so only one chunk's
    autograd graph -- its response-position log-softmax and activations --
    is alive at a time; the returned ``total_loss`` is then detached and
    ``backward_done`` is True. Without it the shares are summed into one
    differentiable ``total_loss`` (every chunk's graph retained).

    ``old_logprobs`` (from :func:`compute_token_old_logprobs`) makes the
    ratio off-policy for later inner updates; ``None`` means on-policy
    (``logp_old = logp_cur.detach()``).
    """
    torch = require_torch()
    device = _resolve_model_device(agent, torch)
    objective = _token_objective(config)
    entropy_coef = float(getattr(config, "entropy_coef", 0.0))
    use_kl = beta > 0 and reference_model is not None
    if use_kl:
        objective = _token_kl_objective(objective, config, beta=beta)
    if objective.ratio == "group_expectation":
        raise ValueError(
            "ratio='group_expectation' couples a group's rows and cannot be "
            "evaluated per chunk on the token path"
        )
    chunk = int(getattr(config, "generation_batch_size", 0) or 0) or len(
        max(rows_per_group, key=len)
    )

    if old_logprobs is not None and len(old_logprobs) != len(rows_per_group):
        raise ValueError(
            f"old_logprobs has {len(old_logprobs)} entries for "
            f"{len(rows_per_group)} groups; snapshot and loss must see the same groups"
        )
    old_source = "snapshot" if old_logprobs is not None else "recompute"
    if old_logprobs is None and (
        str(getattr(config, "old_logprobs_source", "recompute")) == "sampler"
    ):
        old_logprobs = sampler_old_logprobs(trajectory_groups, device)
        if old_logprobs is not None:
            old_source = "sampler"
    chunk_objective = objective.with_(entropy_coef=entropy_coef)
    n_groups = max(len(rows_per_group), 1)
    shares: list[Any] = []
    kls, ents, adv_log, n_rows = [], [], [], 0
    ratio_means, clip_fracs = [], []
    for gi, (group, rows) in enumerate(
        zip(trajectory_groups, rows_per_group, strict=True)
    ):
        rewards = torch.tensor(
            [float(t.total_reward) for t in group.trajectories],
            dtype=torch.float32,
            device=device,
        )
        reward_clip = getattr(config, "reward_clip", None)
        if reward_clip is not None:
            rewards = torch.clamp(
                rewards, min=-float(reward_clip), max=float(reward_clip)
            )
        advantages = _token_advantages(
            rewards,
            config,
            objective,
            global_reward_mean=global_reward_mean,
            update_global_stats=update_global_stats,
        )
        adv_log.extend(advantages.detach().cpu().tolist())
        row_adv = advantages[torch.tensor([ti for _, _, ti in rows], device=device)]

        width = max(len(p) + len(r) for p, r, _ in rows)
        group_rows = float(len(rows))
        group_tokens = float(sum(len(r) for _, r, _ in rows))
        kl_mean, ent_mean = _WeightedMean(), _WeightedMean()
        ratio_mean, clip_mean = _WeightedMean(), _WeightedMean()
        for start in range(0, len(rows), chunk):
            stop = start + chunk
            sub = rows[start:stop]
            logp_cur, mask, entropy = _forward_token_chunk(
                agent.model,
                sub,
                width,
                device,
                grad=True,
                want_entropy=entropy_coef > 0,
            )
            logp_ref = None
            if use_kl:
                logp_ref, _, _ = _forward_token_chunk(
                    reference_model, sub, width, device, grad=False
                )
            result = objectives.policy_loss(
                logp_cur=logp_cur,
                mask=mask,
                advantages=row_adv[start:stop],
                objective=chunk_objective,
                logp_old=None if old_logprobs is None else old_logprobs[gi][start:stop],
                logp_ref=logp_ref,
                entropy=entropy,
                total_rows=group_rows,
                total_tokens=group_tokens,
            )
            share = result.loss / n_groups
            if backward_fn is not None:
                backward_fn(share)
                share = share.detach()
            shares.append(share)
            chunk_tokens = float(mask.sum().item())
            kl_weight = len(sub) if objective.kl == "k3_sequence" else chunk_tokens
            kl_mean.add(result.metrics["kl"], kl_weight)
            ent_mean.add(result.metrics["entropy"], chunk_tokens)
            ratio_mean.add(result.metrics["ratio_mean"], chunk_tokens)
            clip_mean.add(result.metrics["clip_fraction"], chunk_tokens)
            del result, logp_cur, logp_ref, entropy, share
        kls.append(kl_mean.value(0.0))
        ents.append(ent_mean.value(0.0))
        ratio_means.append(ratio_mean.value(1.0))
        clip_fracs.append(clip_mean.value(0.0))
        n_rows += len(rows)

    if shares:
        total = torch.stack(shares).sum()
    else:
        total = torch.tensor(0.0, device=device, requires_grad=backward_fn is None)
    return {
        "total_loss": total,
        "policy_loss": total,
        "backward_done": backward_fn is not None,
        "kl_penalty": torch.tensor(float(np.mean(kls)) if kls else 0.0, device=device),
        "entropy": float(np.mean(ents)) if ents else 0.0,
        "mean_advantage": float(np.mean(adv_log)) if adv_log else 0.0,
        "advantage_std": float(np.std(adv_log)) if adv_log else 0.0,
        "num_trajectories": sum(len(g.trajectories) for g in trajectory_groups),
        "num_rows": n_rows,
        "path": "token",
        "objective": objective.name,
        "ratio_mean": float(np.mean(ratio_means)) if ratio_means else 1.0,
        "clip_fraction": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        "off_policy": old_logprobs is not None,
        "old_logprobs_source": old_source,
    }


def _try_token_path(
    trajectory_groups: list[Any], config: Any, agent: Any, **kwargs: Any
) -> dict[str, Any] | None:
    """Run the batched token path when every group carries token metadata."""
    groups = [g for g in trajectory_groups if getattr(g, "trajectories", None)]
    if not groups:
        return None
    rows = [_token_rows_for_group(g) for g in groups]
    if any(r is None for r in rows):
        return None
    return _compute_token_path_loss(
        groups, [r for r in rows if r], config, agent, **kwargs
    )


def compute_grpo_loss(
    trajectory_groups: list[Any],
    config: Any,
    agent: Any,
    global_reward_mean: float,
    global_reward_count: int,
    update_global_stats: Callable[[float, int], None] | None,
    old_logprobs: list[Any] | None = None,
    *,
    backward_fn: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """
    Compute GRPO loss from trajectory groups with configurable baseline and normalization.

    ``old_logprobs`` (see :func:`compute_token_old_logprobs`) is honoured on the
    batched token path only, for multiple inner updates per rollout batch.

    ``backward_fn`` (token path only) receives each row chunk's share of the
    loss as soon as it is computed, bounding peak activation memory by one
    chunk; the result then carries ``backward_done=True`` and a detached
    ``total_loss`` that must not be backpropagated again. The sequence path
    ignores it and returns a differentiable loss as before.

    Args:
        trajectory_groups: List of TrajectoryGroup objects
        config: Training configuration
        agent: The agent being trained
        global_reward_mean: Running mean of rewards
        global_reward_count: Count of rewards seen
        update_global_stats: Callback to update global reward statistics

    Returns:
        Dictionary containing loss tensors and metrics
    """
    # No running statistics yet: ``global_mean`` uses the group mean.
    effective_global_mean = global_reward_mean if global_reward_count > 0 else None
    token_result = _try_token_path(
        trajectory_groups,
        config,
        agent,
        old_logprobs=old_logprobs,
        global_reward_mean=effective_global_mean,
        update_global_stats=update_global_stats,
        backward_fn=backward_fn,
    )
    if token_result is not None:
        return token_result

    torch = require_torch()
    device = _resolve_model_device(agent, torch)

    policy_losses = []
    all_advantages_for_logging: list[float] = []
    entropy_values_for_logging: list[float] = []
    entropy_coef = float(getattr(config, "entropy_coef", 0.0))

    for group in trajectory_groups:
        if not group.trajectories:
            continue

        rewards = torch.tensor(
            [t.total_reward for t in group.trajectories],
            dtype=torch.float32,
            device=device,
        )
        advantages = _sequence_group_advantages(
            rewards,
            config,
            global_reward_mean=effective_global_mean,
            update_global_stats=update_global_stats,
        )

        all_advantages_for_logging.extend(advantages.detach().cpu().tolist())

        # Compute policy loss for this group
        group_loss, group_entropy = _compute_group_policy_loss(
            group, advantages, config, agent, entropy_coef=entropy_coef
        )
        policy_losses.append(group_loss)
        if group_entropy is not None:
            entropy_values_for_logging.append(group_entropy)

    # Aggregate losses
    if policy_losses:
        total_loss_tensor = torch.stack(policy_losses).mean()
    else:
        total_loss_tensor = torch.tensor(0.0, requires_grad=True)

    # Entropy bonus — encourages exploration by penalizing overly confident
    # distributions.  Computed inside each trajectory's grad-enabled forward
    # pass (see `_compute_group_policy_loss`) so it contributes a real
    # gradient rather than being estimated under `torch.no_grad()`.
    entropy_value = (
        float(np.mean(entropy_values_for_logging))
        if entropy_values_for_logging
        else 0.0
    )

    return {
        "path": "sequence",
        "policy_loss": total_loss_tensor,
        "total_loss": total_loss_tensor,
        "mean_advantage": (
            float(np.mean(all_advantages_for_logging))
            if all_advantages_for_logging
            else 0.0
        ),
        "advantage_std": (
            float(np.std(all_advantages_for_logging))
            if all_advantages_for_logging
            else 0.0
        ),
        "entropy": entropy_value,
    }


def compute_ppo_ratio(
    new_log_prob_sum: Any, old_log_prob_sum: Any, token_count: int
) -> Any:
    """Length-normalized PPO importance ratio.

    Ratios computed from raw summed log-probs blow up (overflow) or vanish
    (underflow) for long responses, since the sum scales with sequence
    length. Dividing by `token_count` first yields a bounded, O(1)
    sequence-mean ratio, matching standard PPO/GRPO implementations.
    """
    torch = get_torch() or require_torch()
    return torch.exp((new_log_prob_sum - old_log_prob_sum) / max(token_count, 1))


def compute_entropy_bonus(logits: Any, response_mask: Any) -> Any:
    """Differentiable masked-mean policy entropy from logits.

    Unlike a `torch.no_grad()` estimate, this stays in the autograd graph so
    the entropy bonus actually contributes gradient to the loss it's added
    to (encouraging exploration), rather than being a dead metric.
    """
    F = get_functional()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    # H = -sum(p * log p)
    token_entropy = -(probs * log_probs).sum(dim=-1)

    mask = response_mask
    if mask is None:
        return token_entropy.mean()
    mask = mask.to(dtype=token_entropy.dtype)
    denom = mask.sum()
    if denom.item() == 0:
        return token_entropy.mean()
    return (token_entropy * mask).sum() / denom


# NOT a drop-in replacement: `compute_entropy_bonus(logits, response_mask)`
# has a different signature than the old no-grad `_estimate_policy_entropy`
# estimator it replaces here. This alias exists only because the sole known
# external caller checks `callable(_estimate_policy_entropy)` rather than
# invoking it with the old signature — do not call this alias directly
# expecting the old API.
_estimate_policy_entropy = compute_entropy_bonus


def _compute_group_policy_loss(
    group: Any,
    advantages: Any,
    config: Any,
    agent: Any,
    entropy_coef: float = 0.0,
) -> tuple[Any, float | None]:
    """
    Compute policy loss for a single trajectory group with proper GRPO implementation.

    Args:
        group: TrajectoryGroup object
        advantages: Tensor of advantages for each trajectory
        config: Training configuration
        agent: The agent being trained
        entropy_coef: Coefficient for the (differentiable) entropy bonus

    Returns:
        Tuple of (policy loss tensor, mean entropy for logging or None)
    """
    torch = require_torch()

    device = _resolve_model_device(agent, torch)
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    num_trajectories = 0
    entropy_sum = 0.0
    entropy_count = 0

    # PPO-style clipping ratio (falls back to advantage clipping when old log probs are absent).
    # `clip_ratio` stays the on/off switch, but the clip magnitude comes from
    # `seq_clip_ratio`: the ratio here is a per-token-mean sequence ratio, so it
    # needs a GSPO-scale bound (+/-0.2 would never trigger).
    clip_ratio = getattr(config, "clip_ratio", getattr(config, "clip_epsilon", 0.2))
    objective = _grpo_objective(config)

    for traj_idx, (trajectory, advantage) in enumerate(
        zip(group.trajectories, advantages, strict=False)
    ):
        try:
            inputs, labels = _prepare_inputs_and_labels(trajectory, agent, config)
            if torch.is_tensor(labels) and not bool(labels.ne(-100).any()):
                continue  # nothing to supervise (e.g. reply truncated away)

            # Forward pass to get log probabilities
            with torch.set_grad_enabled(True):
                outputs = agent.model(**inputs, labels=labels)

                # Get the negative log likelihood (this is the loss from the model)
                nll = outputs.loss

                # GRPO policy gradient: -advantage * log_prob
                # Since outputs.loss is already negative log likelihood,
                # policy loss = advantage * nll (nll = -log_prob).
                policy_loss = advantage * nll

                old_log_prob = None
                token_count = 0
                if torch.is_tensor(labels):
                    loss_mask = labels.ne(-100)
                    token_count = int(loss_mask.sum().item())

                log_probs = getattr(trajectory, "log_probs", None)
                if log_probs is None and hasattr(trajectory, "metadata"):
                    log_probs = trajectory.metadata.get("log_probs")
                if log_probs is not None:
                    if torch.is_tensor(log_probs):
                        old_log_prob = log_probs.sum().detach()
                    elif isinstance(log_probs, (list, tuple)):
                        old_log_prob = torch.tensor(
                            float(sum(log_probs)), device=device
                        )
                    elif isinstance(log_probs, (int, float)):
                        old_log_prob = torch.tensor(float(log_probs), device=device)

                # Optional: PPO-style clipping when old log probs are available.
                if clip_ratio > 0 and old_log_prob is not None and token_count > 0:
                    policy_loss = _clipped_trajectory_loss(
                        outputs.loss, token_count, advantage, old_log_prob, objective
                    )
                elif clip_ratio > 0:
                    global _warned_missing_log_probs
                    if not _warned_missing_log_probs:
                        logger.warning(
                            "clip_ratio set but trajectories lack log_probs; "
                            "using unclipped REINFORCE loss."
                        )
                        _warned_missing_log_probs = True
                    # Without old log_probs, PPO-style clipping is impossible.
                    # clip_ratio operates in ratio space, not advantage magnitude.
                    policy_loss = advantage * nll

                # outputs.loss is already the per-token mean NLL, and the
                # clipped branch normalizes inside the ratio; no further
                # length division (dividing again gave a 1/L^2 bias).

                # Differentiable entropy bonus: computed from the same
                # grad-enabled forward pass's logits, so it actually
                # contributes gradient (encourages exploration) rather than
                # being a dead no-grad metric.
                if entropy_coef > 0 and torch.is_tensor(labels):
                    # Logits at position t predict token t+1: shift like the NLL.
                    response_mask = (
                        labels[:, 1:].ne(-100).to(dtype=outputs.logits.dtype)
                    )
                    entropy_bonus = compute_entropy_bonus(
                        outputs.logits[:, :-1], response_mask
                    )
                    policy_loss = policy_loss - entropy_coef * entropy_bonus
                    entropy_sum += float(entropy_bonus.detach().item())
                    entropy_count += 1

                total_loss = total_loss + policy_loss
                num_trajectories += 1

        except LOSS_EXCEPTIONS as e:
            logger.warning(
                f"Failed to compute policy loss for trajectory {traj_idx}: {e}"
            )
            continue

    mean_entropy = (entropy_sum / entropy_count) if entropy_count > 0 else None

    # Average over trajectories in the group
    if num_trajectories > 0:
        return total_loss / num_trajectories, mean_entropy
    else:
        return torch.tensor(0.0, device=device, requires_grad=True), mean_entropy


def compute_enhanced_grpo_loss(
    trajectory_groups: list[Any],
    beta: float,
    config: Any,
    agent: Any,
    reference_model: Any | None = None,
    old_logprobs: list[Any] | None = None,
    *,
    global_reward_mean: float | None = None,
    global_reward_count: int = 0,
    update_global_stats: Callable[[float, int], None] | None = None,
    backward_fn: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """
    Enhanced GRPO loss computation with KL penalty and proper advantages.

    Advantages, reward clipping and the entropy bonus match
    :func:`compute_grpo_loss`; only the reference-model KL term is added.
    ``backward_fn`` has the same chunked-backward contract as there.

    Args:
        trajectory_groups: List of TrajectoryGroup objects
        beta: KL penalty coefficient
        config: Training configuration
        agent: The agent being trained
        reference_model: Optional reference model for KL computation
        global_reward_mean: Running reward mean for ``baseline_type="global_mean"``
        global_reward_count: Count behind ``global_reward_mean`` (0 = none yet)
        update_global_stats: Callback to update global reward statistics

    Returns:
        Dictionary containing loss tensors and metrics
    """
    effective_global_mean = (
        global_reward_mean
        if global_reward_mean is not None and global_reward_count > 0
        else None
    )
    token_result = _try_token_path(
        trajectory_groups,
        config,
        agent,
        beta=beta,
        reference_model=reference_model,
        old_logprobs=old_logprobs,
        global_reward_mean=effective_global_mean,
        update_global_stats=update_global_stats,
        backward_fn=backward_fn,
    )
    if token_result is not None:
        return token_result

    torch = require_torch()
    device = _resolve_model_device(agent, torch)
    F = get_functional()
    amp = get_amp()

    if F is None:
        raise ImportError(
            "torch.nn.functional is required for enhanced GRPO loss computation."
        )

    all_losses = []
    all_advantages = []
    all_kl_divs = []
    entropy_values: list[float] = []
    entropy_coef = float(getattr(config, "entropy_coef", 0.0))
    objective = _grpo_objective(config)

    for group in trajectory_groups:
        if not group.trajectories:
            continue

        # Extract rewards for this group
        rewards = torch.tensor(
            [t.total_reward for t in group.trajectories],
            dtype=torch.float32,
            device=device,
        )

        advantages = _sequence_group_advantages(
            rewards,
            config,
            global_reward_mean=effective_global_mean,
            update_global_stats=update_global_stats,
        )

        all_advantages.extend(advantages.tolist())

        # Process each trajectory in the group
        for _traj_idx, (trajectory, advantage) in enumerate(
            zip(group.trajectories, advantages, strict=False)
        ):
            inputs, labels = _prepare_inputs_and_labels(trajectory, agent, config)
            if torch.is_tensor(labels) and not bool(labels.ne(-100).any()):
                continue  # nothing to supervise (e.g. reply truncated away)

            # Forward pass
            use_amp = getattr(config, "bf16", False) or getattr(config, "fp16", False)
            if amp is not None:
                autocast_ctx = amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    enabled=bool(use_amp),
                )
            else:  # pragma: no cover
                autocast_ctx = contextlib.nullcontext()

            with autocast_ctx:
                outputs = agent.model(**inputs, labels=labels)

                # Compute policy loss with advantage weighting.
                nll = outputs.loss
                policy_loss = advantage * nll

                # Optional PPO-style clipping, mirroring `_compute_group_policy_loss`.
                clip_ratio = getattr(
                    config, "clip_ratio", getattr(config, "clip_epsilon", 0.2)
                )
                old_log_prob = None
                log_probs_attr = getattr(trajectory, "log_probs", None)
                if log_probs_attr is None and hasattr(trajectory, "metadata"):
                    log_probs_attr = trajectory.metadata.get("log_probs")
                if log_probs_attr is not None:
                    if torch.is_tensor(log_probs_attr):
                        old_log_prob = log_probs_attr.sum().detach()
                    elif isinstance(log_probs_attr, (list, tuple)):
                        old_log_prob = torch.tensor(
                            float(sum(log_probs_attr)), device=device
                        )
                    elif isinstance(log_probs_attr, (int, float)):
                        old_log_prob = torch.tensor(
                            float(log_probs_attr), device=device
                        )

                token_count = 0
                if torch.is_tensor(labels):
                    token_count = int(labels.ne(-100).sum().item())

                if clip_ratio > 0 and old_log_prob is not None and token_count > 0:
                    policy_loss = _clipped_trajectory_loss(
                        nll, token_count, advantage, old_log_prob, objective
                    )

                if entropy_coef > 0 and torch.is_tensor(labels):
                    # Logits at position t predict token t+1: shift like the NLL.
                    entropy_bonus = compute_entropy_bonus(
                        outputs.logits[:, :-1],
                        labels[:, 1:].ne(-100).to(dtype=outputs.logits.dtype),
                    )
                    policy_loss = policy_loss - entropy_coef * entropy_bonus
                    entropy_values.append(float(entropy_bonus.detach().item()))

                all_losses.append(policy_loss)

                # Skip the full-vocab log_softmax entirely when there's no KL
                # penalty to compute — it's an expensive (batch, seq, vocab)
                # materialization that's otherwise dead weight.
                if beta > 0 and reference_model is not None:
                    log_probs = outputs.logits.log_softmax(dim=-1)
                    with torch.no_grad():
                        ref_outputs = reference_model(**inputs)
                        ref_log_probs = ref_outputs.logits.log_softmax(dim=-1)

                    # Token-wise KL(p || p_ref) = sum_v p(v) * (log p(v) - log p_ref(v))
                    kl_per_token = (log_probs.exp() * (log_probs - ref_log_probs)).sum(
                        dim=-1
                    )

                    # Mask to the tokens the LM loss scores. Logits at t
                    # predict token t+1, so shift like the NLL.
                    if torch.is_tensor(labels):
                        kl_per_token = kl_per_token[:, :-1]
                        loss_mask = labels[:, 1:].ne(-100)
                        if loss_mask.any():
                            kl_div = (kl_per_token * loss_mask).sum() / loss_mask.sum()
                        else:
                            kl_div = kl_per_token.mean()
                    else:  # pragma: no cover - non-tensor labels (stub tokenizer)
                        kl_div = kl_per_token.mean()
                    all_kl_divs.append(kl_div)

    # Aggregate losses
    if all_losses:
        policy_loss = torch.stack(all_losses).mean()

        # Add KL penalty if applicable
        if all_kl_divs and beta > 0:
            kl_penalty = torch.stack(all_kl_divs).mean()
            total_loss = policy_loss + beta * kl_penalty
        else:
            total_loss = policy_loss
            kl_penalty = torch.tensor(0.0, device=device)
    else:
        total_loss = torch.tensor(0.0, requires_grad=True, device=device)
        policy_loss = total_loss
        kl_penalty = torch.tensor(0.0, device=device)

    return {
        "path": "sequence",
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "kl_penalty": kl_penalty,
        "mean_advantage": np.mean(all_advantages) if all_advantages else 0.0,
        "advantage_std": np.std(all_advantages) if all_advantages else 0.0,
        "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "num_trajectories": sum(len(g.trajectories) for g in trajectory_groups),
    }


def _format_trajectory_for_model(trajectory: Any, agent: Any) -> str:
    """Format trajectory into text for model input."""
    messages = _trajectory_to_messages(trajectory)

    tokenizer = agent.tokenizer
    if hasattr(tokenizer, "apply_chat_template") and not (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is None
    ):
        # Use tokenizer's chat template
        rendered: object = agent.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return rendered if isinstance(rendered, str) else str(rendered)
    else:
        # Simple formatting
        parts = []
        for msg in messages:
            if msg["role"] == "user":
                parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"Assistant: {msg['content']}")

        return "\n".join(parts)


def _trajectory_to_messages(trajectory: Any) -> list[dict[str, str]]:
    """Convert trajectory turns into normalized chat messages."""
    messages: list[dict[str, str]] = []
    for turn in getattr(trajectory, "turns", []):
        if isinstance(turn, dict):
            role = str(turn.get("role", "user"))
            content = str(turn.get("content", ""))
        else:
            role = str(getattr(turn, "role", None) or "user")
            content = str(getattr(turn, "content", None) or "")
        messages.append({"role": role, "content": content})
    return messages


def _format_trajectory_with_spans(
    trajectory: Any,
) -> tuple[str, list[tuple[int, int]]]:
    """Format trajectory text and track assistant spans for masking."""
    messages = _trajectory_to_messages(trajectory)
    parts: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    cursor = 0

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            prefix = "Assistant: "
        elif role == "system":
            prefix = "System: "
        elif role == "tool":
            prefix = "Tool: "
        else:
            prefix = "User: "

        segment = f"{prefix}{content}\n"
        if role == "assistant":
            assistant_spans.append((cursor, cursor + len(segment)))
        parts.append(segment)
        cursor += len(segment)

    return "".join(parts), assistant_spans


_TEMPLATE_EXCEPTIONS: tuple[type[BaseException], ...] = (TypeError, *LOSS_EXCEPTIONS)
# Slow (non-Rust) tokenizers raise NotImplementedError for offset mappings.
_OFFSET_EXCEPTIONS: tuple[type[BaseException], ...] = (
    NotImplementedError,
    *_TEMPLATE_EXCEPTIONS,
)
_GENERATION_TAG = re.compile(r"\{%-?\s*generation\s*-?%\}")


def _as_bool_mask(value: Any, like: Any, allow_empty: bool = False) -> Any | None:
    """Coerce a tokenizer-produced mask to a bool tensor shaped like ``like``."""
    torch = require_torch()
    try:
        mask = torch.as_tensor(value, dtype=torch.bool)
    except (TypeError, ValueError, RuntimeError):
        return None
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.shape != like.shape or not (allow_empty or bool(mask.any())):
        return None
    return mask


def _chat_template_generation_mask(
    tokenizer: Any, messages: list[dict[str, str]], max_length: int
) -> tuple[Any, Any, Any] | None:
    """Use HF ``return_assistant_tokens_mask`` when the template marks generations."""
    try:
        params = inspect.signature(tokenizer.apply_chat_template).parameters
    except (TypeError, ValueError):
        return None
    if "return_dict" not in params or "return_assistant_tokens_mask" not in params:
        return None
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and not _GENERATION_TAG.search(template):
        return None  # HF would return an all-zero mask (and warn every call)
    try:
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=max_length,
            padding=True,
        )
    except _TEMPLATE_EXCEPTIONS:
        return None
    if not isinstance(out, Mapping) or "input_ids" not in out:
        return None
    raw_mask = out.get("assistant_masks")
    if raw_mask is None:
        raw_mask = out.get("assistant_tokens_mask")
    input_ids = out["input_ids"]
    if raw_mask is None or not hasattr(input_ids, "clone"):
        return None
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    # With generation markers in the template, an all-zero mask means the
    # replies were truncated away; without a readable template it is ambiguous.
    marked = isinstance(template, str) and bool(_GENERATION_TAG.search(template))
    mask = _as_bool_mask(raw_mask, input_ids, allow_empty=marked)
    if mask is None:
        return None
    attention_mask = out.get("attention_mask")
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    return input_ids, attention_mask, mask


def _render_chat(
    tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool
) -> str | None:
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except _TEMPLATE_EXCEPTIONS:
        return None
    return rendered if isinstance(rendered, str) else None


def _chat_template_prefix_mask(
    tokenizer: Any, messages: list[dict[str, str]], max_length: int
) -> tuple[Any, Any, Any] | None:
    """Assistant spans anchored inside the full render (see
    :func:`stateset_agents.core.chat_masking.assistant_char_spans`), mapped to
    tokens with offsets. ``None`` when the template rewrites history.
    """
    located = assistant_char_spans(
        lambda msgs, gen: _render_chat(tokenizer, msgs, add_generation_prompt=gen),
        messages,
    )
    if located is None:
        return None
    full, spans = located
    if not spans:
        return None
    try:
        encoded = tokenizer(
            full,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
            add_special_tokens=False,  # the template already rendered them
            return_offsets_mapping=True,
        )
    except _OFFSET_EXCEPTIONS:
        return None
    if not isinstance(encoded, Mapping):
        return None
    input_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if input_ids is None or offsets is None or not hasattr(input_ids, "clone"):
        return None
    torch = require_torch()
    offsets = torch.as_tensor(offsets)
    if offsets.dim() == 2:
        offsets = offsets.unsqueeze(0)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if offsets.shape[:2] != input_ids.shape:
        return None
    starts, ends = offsets[..., 0], offsets[..., 1]
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for span_start, span_end in spans:
        mask |= (starts < span_end) & (ends > span_start)
    mask &= ends > starts
    # An empty mask here means truncation removed every reply: supervise
    # nothing rather than falling back to the full conversation.
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    return input_ids, attention_mask, mask


def _prepare_inputs_and_labels(
    trajectory: Any, agent: Any, config: Any
) -> tuple[dict[str, Any], Any]:
    """Prepare model inputs and loss labels, masking to assistant tokens when possible."""
    torch = get_torch() or require_torch()
    max_length = int(getattr(config, "max_prompt_length", 512)) + int(
        getattr(config, "max_completion_length", 512)
    )

    tokenizer = agent.tokenizer
    device = agent.model.device

    # Assistant-only masking for chat-template tokenizers: prefer the template's
    # own ``{% generation %}`` mask, then derive spans by rendering prefixes.
    use_chat_template = hasattr(tokenizer, "apply_chat_template") and not (
        # HF tokenizers expose apply_chat_template even with no template set.
        hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is None
    )
    if use_chat_template:
        messages = _trajectory_to_messages(trajectory)
        encoded = _chat_template_generation_mask(tokenizer, messages, max_length)
        if encoded is None:
            encoded = _chat_template_prefix_mask(tokenizer, messages, max_length)
        if encoded is not None:
            input_ids, attention_mask, assistant_mask = encoded
            labels = input_ids.clone().masked_fill(~assistant_mask, -100)
            inputs: dict[str, Any] = {"input_ids": input_ids}
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask.eq(0), -100)
                inputs["attention_mask"] = attention_mask
            inputs = {k: v.to(device) for k, v in inputs.items()}
            return inputs, labels.to(device)

        global _warned_missing_assistant_mask
        if not _warned_missing_assistant_mask:
            logger.warning(
                "Could not derive assistant token masks from the chat template; "
                "loss will include non-assistant tokens."
            )
            _warned_missing_assistant_mask = True

    if not use_chat_template:
        # Try assistant-only masking with offsets for non-chat-template tokenizers.
        conversation_text, assistant_spans = _format_trajectory_with_spans(trajectory)
        if assistant_spans:
            try:
                inputs = tokenizer(
                    conversation_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                    return_offsets_mapping=True,
                )
                offsets = inputs.pop("offset_mapping", None)
                input_ids = inputs.get("input_ids")
                if offsets is not None and input_ids is not None:
                    if torch.is_tensor(offsets):
                        offsets_list = offsets[0].tolist()
                    elif isinstance(offsets, (list, tuple)):
                        offsets_list = offsets[0] if offsets else []
                    else:
                        offsets_list = []

                    if offsets_list and len(offsets_list) == input_ids.shape[1]:
                        assistant_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                        for idx, (start, end) in enumerate(offsets_list):
                            if start == end:
                                continue
                            if any(
                                start < span_end and end > span_start
                                for span_start, span_end in assistant_spans
                            ):
                                assistant_mask[0, idx] = True

                        labels = (
                            input_ids.clone()
                            if hasattr(input_ids, "clone")
                            else input_ids
                        )
                        labels = labels.masked_fill(~assistant_mask, -100)
                        if "attention_mask" in inputs and torch.is_tensor(labels):
                            labels = labels.masked_fill(
                                inputs["attention_mask"].eq(0), -100
                            )

                        inputs = {
                            k: v.to(device) if hasattr(v, "to") else v
                            for k, v in inputs.items()
                        }
                        if hasattr(labels, "to"):
                            labels = labels.to(device)
                        return inputs, labels
            except TypeError:
                pass
            except LOSS_EXCEPTIONS:
                pass

    # Fallback: compute loss over the full token stream.
    if use_chat_template:
        conversation_text = _format_trajectory_for_model(trajectory, agent)
    else:
        conversation_text, _assistant_spans = _format_trajectory_with_spans(trajectory)
    inputs = tokenizer(
        conversation_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    input_ids = inputs.get("input_ids")
    labels = (
        input_ids.clone()
        if input_ids is not None and hasattr(input_ids, "clone")
        else input_ids
    )
    if "attention_mask" in inputs and isinstance(labels, torch.Tensor):
        labels = labels.masked_fill(inputs["attention_mask"].eq(0), -100)

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if hasattr(labels, "to"):
        labels = labels.to(device)
    return inputs, labels
