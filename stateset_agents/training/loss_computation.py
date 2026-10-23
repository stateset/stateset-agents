"""
GRPO loss computation functions.

This module provides the core loss computation functions for GRPO training,
including standard GRPO loss and enhanced GRPO loss with KL penalty.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeGuard, cast

import numpy as np

from ..core.chat_masking import assistant_char_spans, has_observation_turns
from ..core.errors import TrainingError
from ..exceptions import LOSS_EXCEPTIONS
from . import rl_losses  # noqa: F401  (re-exported; tests spy on it)
from . import objectives
from .ddp_population import run_token_local, token_policy
from .distributed_advantages import prepare_reward_batch, validate_advantages
from .grpo_accumulation import NormalizedLoss
from .metric_reduction import LossMetricReduction
from .objectives import PolicyObjective
from .rollout_contract import (
    RolloutContractError,
    SampledTurn,
    require_explicit_fallback,
)
from .typing_protocols import (
    AgentLike,
    BackwardFn,
    CausalLMOutput,
    ChatTokenizerLike,
    DeviceLike,
    LossConfig,
    PolicyModel,
    ScoredTrajectory,
    TokenizerLike,
    TokenRow,
    TrajectoryGroupLike,
)

if TYPE_CHECKING:  # torch stays a lazy runtime dependency
    from collections.abc import Iterator
    from types import ModuleType

    from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "LOSS_EXCEPTIONS",
    "TrajectoryLossError",
    "compute_grpo_loss",
    "compute_enhanced_grpo_loss",
    "compute_ppo_ratio",
    "compute_entropy_bonus",
]

_warned_missing_log_probs = False
_warned_missing_assistant_mask = False
_warned_observation_fallback = False
_warned_unmasked_labels = False

#: Opt-in for supervising every token when assistant tokens cannot be located.
UNMASKED_LABELS_OPTION = "allow_unmasked_labels"


class TrajectoryLossError(TrainingError, RuntimeError):
    """One trajectory's sequence-path policy loss could not be computed.

    Skipping the row and averaging the rest would silently change the group
    objective, so the loss raises instead (the token path never skips rows
    either). The original exception is chained as ``__cause__``.
    """

    message_only_str: ClassVar[bool] = True

    def __init__(self, trajectory_index: int, group_size: int, cause: BaseException):
        message = (
            f"policy loss failed for trajectory {trajectory_index} of {group_size} "
            f"in the group: {type(cause).__name__}: {cause}"
        )
        super().__init__(
            message, trajectory_index=trajectory_index, group_size=group_size
        )
        self.trajectory_index = trajectory_index
        self.group_size = group_size
        self.cause = cause

    def __reduce__(
        self,
    ) -> tuple[type[TrajectoryLossError], tuple[int, int, BaseException]]:
        """Pickle by constructor arguments, so the error survives multiprocessing."""
        assert self.cause is not None
        return type(self), (self.trajectory_index, self.group_size, self.cause)


class _DeviceCarrier(Protocol):
    """A Hugging Face model, which exposes ``device``."""

    @property
    def device(self) -> DeviceLike: ...


class _ParameterCarrier(Protocol):
    """A torch module, which exposes ``parameters()``."""

    def parameters(self) -> Iterator[Tensor]: ...


def _policy(agent: AgentLike) -> PolicyModel:
    """``agent.model``, which a loss needs to be set."""
    model = agent.model
    if model is None:
        raise TypeError("agent.model is None; initialize the agent before the loss")
    return model


def _tokenizer(agent: AgentLike) -> TokenizerLike:
    """``agent.tokenizer``, which the sequence path needs to be set."""
    tokenizer = agent.tokenizer
    if tokenizer is None:
        raise TypeError("agent.tokenizer is None; initialize the agent first")
    return tokenizer


def _resolve_model_device(agent: AgentLike, torch_mod: ModuleType) -> DeviceLike:
    """Resolve the device for model tensors with safe CPU fallback."""
    device = getattr(agent.model, "device", None)
    if device is None and hasattr(agent.model, "parameters"):
        try:
            first_param = next(cast(_ParameterCarrier, agent.model).parameters())
            device = getattr(first_param, "device", None)
        except StopIteration:
            device = None
    if device is None:
        device = torch_mod.device("cpu")
    return cast("DeviceLike", device)


def _grpo_objective(config: LossConfig) -> PolicyObjective:
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
    nll: Tensor,
    token_count: int,
    advantage: Tensor | float,
    old_log_prob: Tensor,
    objective: PolicyObjective,
) -> Tensor:
    """Clipped sequence-ratio loss for one trajectory via ``objectives.policy_loss``.

    ``nll`` is the model's mean per-token NLL (differentiable); ``old_log_prob``
    the rollout-time summed log-prob. The trajectory is presented as
    ``token_count`` identical tokens so the sequence ratio normalises by the
    true length: ``exp((-nll * T - old) / T)``, exactly the previous
    ``compute_ppo_ratio`` quantity, and ``seq_mean`` over identical tokens is
    the row value.
    """
    import torch

    count = max(int(token_count), 1)
    logp_cur = (-nll).reshape(1, 1).expand(1, count)
    mask = torch.ones(1, count, dtype=logp_cur.dtype, device=logp_cur.device)
    adv = (
        advantage.reshape(1)
        if isinstance(advantage, torch.Tensor)
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


def _token_rows_for_group(group: object) -> list[TokenRow] | None:
    """Rows ``(prompt_ids, response_ids, trajectory_index)`` for every assistant
    turn in ``group``; ``None`` when any trajectory lacks token metadata (the
    caller then takes the sequence-level path)."""
    rows: list[TokenRow] = []
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
            sampled = SampledTurn.from_metadata(md)
            rows.append((list(sampled.prompt_token_ids), list(sampled.token_ids), ti))
            found = True
        if not found:
            return None
    return rows or None


def _token_objective(config: LossConfig) -> PolicyObjective:
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


def _token_kl_objective(
    objective: PolicyObjective, config: LossConfig, *, beta: float
) -> PolicyObjective:
    """Enable the KL penalty with coefficient ``beta`` on the token path.

    An objective chosen through ``config.objective`` keeps its own estimator
    (``gspo`` regularizes the length-normalized sequence log-ratio); the native
    objective, which has none, uses per-token k3.
    """
    kind = objective.kl
    if getattr(config, "objective", None) is None or kind == "none":
        kind = "k3_token"
    return objective.with_(kl=kind, kl_coef=float(beta))


def _scale_by_group_std(advantages: Tensor) -> Tensor:
    """Divide by the group's population std without re-centring, so a
    median or running-mean baseline survives normalisation (the std of
    ``rewards - baseline`` equals the std of the rewards)."""
    import torch

    if advantages.numel() <= 1:
        return advantages
    std = advantages.std(correction=0)
    if torch.isfinite(std) and std > 0:
        return advantages / (std + 1e-8)
    return advantages


def _sequence_group_advantages(
    rewards: Tensor,
    config: LossConfig,
    *,
    global_reward_mean: float | None = None,
    update_global_stats: Callable[[float, int], None] | None = None,
) -> Tensor:
    """Group advantages for both sequence-level GRPO losses.

    Applies ``reward_clip``, then the ``objective`` preset's estimator when one
    is configured, else ``baseline_type`` with optional std scaling.
    """
    import torch

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


def _response_token_logprobs(
    logits: Tensor, input_ids: Tensor, response_mask: Tensor, *, want_entropy: bool
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Shift-by-one token log-probs computed at response positions only.

    Same contract as :func:`rl_losses.gather_token_logprobs` (``[B, T-1]``
    log-probs, zero off the mask, plus the fp32 shifted mask), but only the
    ``n`` response positions are upcast and normalised: the largest tensor is
    ``[n, V]`` in fp32 instead of a full ``[B, T-1, V]`` fp32 log-softmax.
    Without entropy the log-prob is ``logit[y] - logsumexp(logits)``; with it
    one ``log_softmax`` over the selected rows serves both. ``entropy`` is the
    per-token entropy at response positions (zero elsewhere) or ``None``.
    """
    import torch

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
    model: PolicyModel,
    chunk: Sequence[TokenRow],
    width: int,
    device: DeviceLike,
    *,
    grad: bool,
    want_entropy: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Pad ``chunk`` to ``width`` and run one forward pass.

    Returns ``(token_logprobs, response_mask, entropy)``, each ``[len(chunk),
    width-1]``; see :func:`_response_token_logprobs`. The full-vocabulary
    logits are released on return; only the response-position graph is kept.
    """
    import torch

    ids = torch.zeros(len(chunk), width, dtype=torch.long, device=device)
    attn = torch.zeros(len(chunk), width, dtype=torch.long, device=device)
    resp = torch.zeros(len(chunk), width, dtype=torch.float32, device=device)
    for i, (p, r, _) in enumerate(chunk):
        seq = torch.tensor(p + r, dtype=torch.long, device=device)
        ids[i, : seq.numel()] = seq
        attn[i, : seq.numel()] = 1
        resp[i, len(p) : seq.numel()] = 1.0
    with torch.set_grad_enabled(grad):
        policy = cast(PolicyModel, token_policy(model))
        logits = policy(input_ids=ids, attention_mask=attn).logits
        return _response_token_logprobs(logits, ids, resp, want_entropy=want_entropy)


def _legacy_group_advantages(
    rewards: Tensor,
    config: LossConfig,
    *,
    global_reward_mean: float | None = None,
    update_global_stats: Callable[[float, int], None] | None = None,
    normalize: bool | None = None,
) -> Tensor:
    """``group_median`` / ``global_mean`` baselines, shared by both GRPO paths.

    ``global_mean`` subtracts the trainer's running mean (and reports this
    group's mean to it); without trainer statistics it uses the group mean.
    With ``advantage_normalization`` the centred advantages are scaled by their
    population std, as on the sequence path.
    """
    import torch

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


def _token_advantages(
    rewards: Tensor,
    config: LossConfig,
    objective: PolicyObjective,
    *,
    global_reward_mean: float | None = None,
    update_global_stats: Callable[[float, int], None] | None = None,
) -> Tensor:
    """Group advantages for the token path: the configured preset's estimator,
    or the legacy ``baseline_type``/``advantage_normalization`` switches."""
    import torch

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


def _token_batch_advantages(
    trajectory_groups: list[TrajectoryGroupLike],
    config: LossConfig,
    objective: PolicyObjective,
    device: DeviceLike,
    *,
    model: PolicyModel | None = None,
    global_reward_mean: float | None = None,
    global_reward_count: int = 0,
    global_reward_sum: float | None = None,
    update_global_stats: Callable[[float, int], None] | None = None,
) -> list[Tensor]:
    """Preserve prompt baselines while applying batch statistics to the batch."""
    import torch

    if not trajectory_groups and model is None:
        return []
    rewards_by_group, count, mean, std = prepare_reward_batch(
        trajectory_groups,
        config,
        objective,
        device,
        model=model,
        global_reward_mean=global_reward_mean,
        global_reward_count=global_reward_count,
        global_reward_sum=global_reward_sum,
        updating=update_global_stats is not None,
    )

    def local_advantages() -> list[Tensor]:
        if not rewards_by_group:
            result: list[Tensor] = []
        elif getattr(config, "objective", None) is not None:
            group_ids = torch.cat(
                [
                    torch.full_like(rewards, index, dtype=torch.long)
                    for index, rewards in enumerate(rewards_by_group)
                ]
            )
            batch_norm = objective.advantage == "batch_norm"
            estimator = (
                objective.with_(advantage="group_mean") if batch_norm else objective
            )
            advantages = objectives.compute_advantages(
                torch.cat(rewards_by_group), group_ids, estimator
            )
            if batch_norm:
                scale = torch.tensor(std, dtype=torch.float32, device=device)
                advantages = (
                    torch.zeros_like(advantages)
                    if scale <= objective.advantage_eps
                    else advantages / (scale + objective.advantage_eps)
                )
            result = list(
                advantages.split([len(rewards) for rewards in rewards_by_group])
            )
        else:
            result = [
                _token_advantages(
                    rewards,
                    config,
                    objective,
                    global_reward_mean=(
                        global_reward_mean if global_reward_count > 0 else None
                    ),
                )
                for rewards in rewards_by_group
            ]
        return result

    result = run_token_local(model, "advantages", local_advantages)
    validate_advantages(result, model, device)
    if (
        getattr(config, "objective", None) is None
        and getattr(config, "baseline_type", "group_mean") == "global_mean"
        and update_global_stats is not None
    ):
        record = update_global_stats
        run_token_local(model, "baseline", lambda: record(mean, count))
    return result


def _forward_token_rows(
    model: PolicyModel,
    rows: Sequence[TokenRow],
    device: DeviceLike,
    chunk_size: int,
    *,
    grad: bool,
    want_entropy: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Pad ``rows`` and run ``model`` in chunks of ``chunk_size`` rows.

    Returns ``(token_logprobs [R, W-1], response_mask [R, W-1], entropy)``
    where ``entropy`` is the differentiable per-token entropy (or ``None``).
    Graphs are retained across chunks when ``grad`` is True; the training
    loss uses :func:`_forward_token_chunk` per chunk instead.
    """
    import torch

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
    entropy = (
        torch.cat([cast("Tensor", ent) for ent in ent_chunks]) if want_entropy else None
    )
    return torch.cat(lp_chunks), torch.cat(mask_chunks), entropy


def compute_token_old_logprobs(
    trajectory_groups: Sequence[TrajectoryGroupLike],
    config: LossConfig,
    agent: AgentLike,
) -> list[Tensor] | None:
    """Freeze the old policy's per-token log-probs for every group (one no-grad
    chunked forward on the stored token ids), for multiple inner updates.

    Returns one detached ``[rows, width-1]`` tensor per non-empty group, or
    ``None`` when any group lacks token metadata (the sequence fallback has no
    per-token old log-probs; inner updates are then not applied).
    """
    import torch

    groups = [g for g in trajectory_groups if getattr(g, "trajectories", None)]
    rows = [_token_rows_for_group(g) for g in groups]
    if not groups and getattr(agent.model, "_stateset_token_forward", False) is True:
        return []
    if not groups or any(r is None for r in rows):
        return None
    device = _resolve_model_device(agent, torch)
    chunk = int(getattr(config, "generation_batch_size", 0) or 0) or max(
        len(r) for r in rows if r
    )
    snapshots = []
    # Score the old policy without dropout: a stochastic snapshot makes the
    # first inner-update ratio noise rather than 1, so the clip bites on it.
    model = _policy(agent)
    was_training = bool(getattr(model, "training", False))
    set_eval = getattr(model, "eval", None) if was_training else None
    if callable(set_eval):
        set_eval()
    try:
        for group_rows in rows:
            assert group_rows is not None
            lp, _, _ = _forward_token_rows(model, group_rows, device, chunk, grad=False)
            snapshots.append(lp.detach())
    finally:
        set_train = getattr(model, "train", None) if was_training else None
        if callable(set_train):
            set_train()
    return snapshots


def sampler_old_logprobs(
    trajectory_groups: Sequence[TrajectoryGroupLike], device: DeviceLike
) -> list[Tensor] | None:
    """The log-probs recorded at sampling time (``sampler_log_probs`` on each
    assistant turn), laid out like :func:`_forward_token_rows` output: one
    ``[rows, width-1]`` tensor per non-empty group with response token ``k``
    of a row at column ``len(prompt) - 1 + k`` and zeros elsewhere.

    Returns ``None`` when any row lacks sampler log-probs; raises when a row's
    log-probs disagree in length with its token ids (corrupt metadata, never
    silently misaligned).
    """
    import torch

    groups = [g for g in trajectory_groups if getattr(g, "trajectories", None)]
    out: list[Tensor] = []
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


def _assistant_turns(group: object) -> list[Any]:
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


@dataclass(frozen=True)
class _TokenPathOptions:
    """Optional inputs shared by both GRPO entry points' token path."""

    beta: float = 0.0
    reference_model: PolicyModel | None = None
    old_logprobs: Sequence[Tensor] | None = None
    global_reward_mean: float | None = None
    global_reward_count: int = 0
    global_reward_sum: float | None = None
    update_global_stats: Callable[[float, int], None] | None = None
    backward_fn: BackwardFn | None = None


@dataclass(frozen=True)
class _TokenBatch:
    """Everything the token-path forwards need, fixed before the first one."""

    groups: list[TrajectoryGroupLike]
    rows_per_group: list[list[TokenRow]]
    objective: PolicyObjective
    entropy_coef: float
    reference_model: PolicyModel | None
    chunk_size: int
    old_logprobs: Sequence[Tensor] | None
    old_source: str
    advantages: list[Tensor]
    device: DeviceLike
    n_trajectories: int
    n_tokens: int

    @property
    def sequence_measure(self) -> bool:
        """Whether the policy term is reduced per trajectory (else per token)."""
        return self.objective.aggregate in {"seq_mean", "seq_sum_const"}

    @property
    def measure(self) -> str:
        """``"trajectory"`` or ``"token"``: the policy denominator's unit."""
        return "trajectory" if self.sequence_measure else "token"

    @property
    def denominator(self) -> int:
        """Batch-level count the policy term is normalised by."""
        return self.n_trajectories if self.sequence_measure else self.n_tokens

    @property
    def kl_measure(self) -> str:
        """Unit the KL metric is averaged over."""
        return "trajectory" if self.objective.kl == "k3_sequence" else "token"


@dataclass
class _TokenLossTotals:
    """Running sums over chunks; tensors are detached once backward has run."""

    policy_shares: list[Tensor] = field(default_factory=list)
    entropy_shares: list[Tensor] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    rows: int = 0
    kl_sum: float = 0.0
    ratio_sum: float = 0.0
    clip_sum: float = 0.0
    entropy_sum: float = 0.0


def _token_objective_settings(config: LossConfig) -> tuple[PolicyObjective, float]:
    """The validated token objective (entropy split out) and ``entropy_coef``.

    Validates before forwards, reward statistics or streamed backward; the
    entropy bonus is then applied separately with its own token denominator.
    """
    entropy = float(getattr(config, "entropy_coef", 0.0))
    validated = _token_objective(config).with_(entropy_coef=entropy)
    return validated.with_(entropy_coef=0.0), entropy


def _resolve_old_logprobs(
    trajectory_groups: Sequence[TrajectoryGroupLike],
    rows_per_group: Sequence[Sequence[TokenRow]],
    config: LossConfig,
    agent: AgentLike,
    old_logprobs: Sequence[Tensor] | None,
    device: DeviceLike,
) -> tuple[Sequence[Tensor] | None, str]:
    """The old-policy log-probs to evaluate against and their source label."""
    if old_logprobs is not None and len(old_logprobs) != len(rows_per_group):
        raise ValueError("old_logprobs and loss must contain the same groups")
    source = "snapshot" if old_logprobs is not None else "recompute"
    if (
        old_logprobs is None
        and getattr(config, "old_logprobs_source", "recompute") == "sampler"
    ):
        old_logprobs = (
            []
            if not trajectory_groups
            and getattr(agent.model, "_stateset_rollout_source", None) == "sampler"
            else sampler_old_logprobs(trajectory_groups, device)
        )
        if old_logprobs is not None:
            source = "sampler"
        else:
            require_explicit_fallback(
                config,
                "allow_missing_sampler_logprobs",
                "Rollouts lack sampler log probabilities",
            )
            source = "recompute_fallback"
    return old_logprobs, source


def _build_token_batch(
    trajectory_groups: list[TrajectoryGroupLike],
    rows_per_group: list[list[TokenRow]],
    config: LossConfig,
    agent: AgentLike,
    options: _TokenPathOptions,
) -> _TokenBatch:
    """Fix the objective, old log-probs, advantages and counts of a batch.

    Runs the token-bank phases ``objective``, ``old-source`` and (inside
    :func:`_token_batch_advantages`) ``advantages``/``baseline`` in that
    order, and rejects objectives the chunked loss cannot evaluate before any
    forward pass.
    """
    import torch

    device = _resolve_model_device(agent, torch)
    objective, entropy_coef = run_token_local(
        agent.model, "objective", lambda: _token_objective_settings(config)
    )
    reference = options.reference_model if options.beta > 0 else None
    if reference is not None:
        objective = _token_kl_objective(objective, config, beta=options.beta)
    if objective.ratio == "group_expectation":
        raise ValueError(
            "group_expectation couples rows and cannot use chunked token loss"
        )
    sequence_measure = objective.aggregate in {"seq_mean", "seq_sum_const"}
    if reference is not None and objective.kl == "k3_sequence" and not sequence_measure:
        raise ValueError(
            "Sequence KL requires a trajectory reduction on the native token path"
        )
    size = max(
        1,
        int(getattr(config, "generation_batch_size", 0) or 0)
        or len(max(rows_per_group, key=len, default=[])),
    )
    old_logprobs, source = run_token_local(
        agent.model,
        "old-source",
        lambda: _resolve_old_logprobs(
            trajectory_groups,
            rows_per_group,
            config,
            agent,
            options.old_logprobs,
            device,
        ),
    )
    advantages = _token_batch_advantages(
        trajectory_groups,
        config,
        objective,
        device,
        model=agent.model,
        global_reward_mean=options.global_reward_mean,
        global_reward_count=options.global_reward_count,
        global_reward_sum=options.global_reward_sum,
        update_global_stats=options.update_global_stats,
    )
    return _TokenBatch(
        groups=trajectory_groups,
        rows_per_group=rows_per_group,
        objective=objective,
        entropy_coef=entropy_coef,
        reference_model=reference,
        chunk_size=size,
        old_logprobs=old_logprobs,
        old_source=source,
        advantages=advantages,
        device=device,
        n_trajectories=sum(len(group.trajectories) for group in trajectory_groups),
        n_tokens=sum(len(reply) for rows in rows_per_group for _, reply, _ in rows),
    )


def _prepare_streamed_backward(
    backward_fn: BackwardFn | None, batch: _TokenBatch
) -> None:
    """Tell a streaming ``backward_fn`` the batch denominators up front."""
    prepare = getattr(backward_fn, "prepare", None)
    if callable(prepare):
        prepare(
            policy_count=batch.denominator,
            token_count=batch.n_tokens,
            measure=batch.measure,
            separate_entropy=batch.sequence_measure and batch.entropy_coef > 0,
        )


def _backward_token_chunk(
    backward_fn: BackwardFn | None,
    batch: _TokenBatch,
    share: Tensor,
    policy: Tensor,
    entropy_loss: Tensor | None,
) -> None:
    """Backpropagate one chunk's additive share of the loss, if streaming."""
    normalized = getattr(backward_fn, "backward_normalized", None)
    if callable(normalized):
        normalized(
            NormalizedLoss(
                {"total_loss": share},
                policy_mean=policy if batch.sequence_measure else share,
                policy_count=batch.denominator,
                measure=batch.measure,
                entropy_loss=entropy_loss if batch.sequence_measure else None,
                token_count=batch.n_tokens,
            )
        )
    elif backward_fn is not None:
        backward_fn(share)


def _token_chunk_loss(
    batch: _TokenBatch,
    totals: _TokenLossTotals,
    model: PolicyModel,
    rows: Sequence[TokenRow],
    width: int,
    advantages: Tensor,
    old: Tensor | None,
    backward_fn: BackwardFn | None,
) -> None:
    """Forward one row chunk, evaluate its loss share and fold it into ``totals``.

    With ``backward_fn`` the share is backpropagated here and only detached
    values are kept, so the chunk's graph is released when this returns,
    before the next chunk's forward pass.
    """
    logp, mask, entropy = _forward_token_chunk(
        model,
        rows,
        width,
        batch.device,
        grad=True,
        want_entropy=batch.entropy_coef > 0,
    )
    reference = None
    if batch.reference_model is not None:
        reference, _, _ = _forward_token_chunk(
            batch.reference_model, rows, width, batch.device, grad=False
        )
    result = objectives.policy_loss(
        logp_cur=logp,
        mask=mask,
        advantages=advantages,
        objective=batch.objective,
        logp_old=old,
        logp_ref=reference,
        total_rows=max(batch.n_trajectories, 1),
        total_tokens=max(batch.n_tokens, 1),
    )
    policy = result.loss
    entropy_sum = (entropy * mask).sum() if entropy is not None else None
    entropy_loss = (
        -batch.entropy_coef * entropy_sum / max(batch.n_tokens, 1)
        if entropy_sum is not None
        else None
    )
    share = policy if entropy_loss is None else policy + entropy_loss
    _backward_token_chunk(backward_fn, batch, share, policy, entropy_loss)
    chunk_tokens = sum(len(reply) for _, reply, _ in rows)
    totals.ratio_sum += result.metrics["ratio_mean"] * chunk_tokens
    totals.clip_sum += result.metrics["clip_fraction"] * chunk_tokens
    totals.kl_sum += result.metrics["kl"] * (
        len(rows) if batch.objective.kl == "k3_sequence" else chunk_tokens
    )
    if entropy_sum is not None:
        totals.entropy_sum += float(entropy_sum.detach())
    streamed = backward_fn is not None
    totals.policy_shares.append(policy.detach() if streamed else policy)
    if entropy_loss is not None:
        totals.entropy_shares.append(
            entropy_loss.detach() if streamed else entropy_loss
        )
    totals.rows += len(rows)


def _token_forwards(
    batch: _TokenBatch, agent: AgentLike, backward_fn: BackwardFn | None
) -> NormalizedLoss:
    """Run every chunk of every group (the token bank's ``forwards`` phase)."""
    import torch

    totals = _TokenLossTotals()
    size = batch.chunk_size
    for gi, (_group, rows, group_adv) in enumerate(
        zip(batch.groups, batch.rows_per_group, batch.advantages, strict=True)
    ):
        totals.advantages.extend(group_adv.detach().cpu().tolist())
        row_adv = group_adv[
            torch.tensor([ti for _, _, ti in rows], device=batch.device)
        ]
        width = max(len(prompt) + len(reply) for prompt, reply, _ in rows)
        for start in range(0, len(rows), size):
            _token_chunk_loss(
                batch,
                totals,
                _policy(agent),
                rows[start : start + size],
                width,
                row_adv[start : start + size],
                (
                    None
                    if batch.old_logprobs is None
                    else batch.old_logprobs[gi][start : start + size]
                ),
                backward_fn,
            )
    return _finish_token_loss(batch, totals, streamed=backward_fn is not None)


def _finish_token_loss(
    batch: _TokenBatch, totals: _TokenLossTotals, *, streamed: bool
) -> NormalizedLoss:
    """Aggregate the chunk shares into the loss, its metrics and reduction."""
    import torch

    n_tokens, denominator = batch.n_tokens, batch.denominator
    zero = torch.tensor(0.0, device=batch.device, requires_grad=not streamed)
    policy = torch.stack(totals.policy_shares).sum() if totals.policy_shares else zero
    entropy_loss = (
        torch.stack(totals.entropy_shares).sum()
        if totals.entropy_shares
        else (zero if batch.entropy_coef > 0 else None)
    )
    total = policy if entropy_loss is None else policy + entropy_loss
    adv_log = totals.advantages
    values = np.asarray(adv_log, dtype=np.float64)
    mean = float(values.mean()) if adv_log else 0.0
    reduction = LossMetricReduction(
        measure=batch.measure,
        policy_count=denominator,
        token_count=n_tokens,
        trajectory_count=batch.n_trajectories,
        row_count=totals.rows,
        advantage_count=len(adv_log),
        nonzero_advantages=int(np.count_nonzero(values)),
        policy_sum=float(policy.detach()) * denominator,
        entropy_loss_sum=(
            float(entropy_loss.detach()) * n_tokens if entropy_loss is not None else 0.0
        ),
        entropy_sum=totals.entropy_sum,
        kl_sum=totals.kl_sum,
        ratio_sum=totals.ratio_sum,
        clip_sum=totals.clip_sum,
        advantage_mean=mean,
        advantage_m2=float(np.square(values - mean).sum()),
        kl_measure=batch.kl_measure,
    )
    kl_count = batch.n_trajectories if batch.kl_measure == "trajectory" else n_tokens
    metrics = {
        "total_loss": total,
        "policy_loss": total,
        "policy_objective_loss": policy,
        "backward_done": streamed,
        "kl_penalty": torch.tensor(
            totals.kl_sum / max(kl_count, 1), device=batch.device
        ),
        "entropy": totals.entropy_sum / max(n_tokens, 1),
        "entropy_measured": batch.entropy_coef > 0,
        "mean_advantage": mean,
        "advantage_std": float(values.std()) if adv_log else 0.0,
        "nonzero_advantage_fraction": (
            float(np.count_nonzero(values) / len(values)) if adv_log else 0.0
        ),
        "num_trajectories": batch.n_trajectories,
        "num_rows": totals.rows,
        "num_tokens": n_tokens,
        "policy_denominator": denominator,
        "path": "token",
        "objective": batch.objective.name,
        "ratio_mean": totals.ratio_sum / n_tokens if n_tokens else 1.0,
        "clip_fraction": totals.clip_sum / max(n_tokens, 1),
        "off_policy": batch.old_logprobs is not None,
        "old_logprobs_source": batch.old_source,
    }
    return NormalizedLoss(
        metrics,
        policy_mean=policy if batch.sequence_measure else total,
        policy_count=denominator,
        measure=batch.measure,
        entropy_loss=entropy_loss if batch.sequence_measure else None,
        token_count=n_tokens,
        metric_reduction=reduction,
    )


def _compute_token_path_loss(
    trajectory_groups: list[TrajectoryGroupLike],
    rows_per_group: list[list[TokenRow]],
    config: LossConfig,
    agent: AgentLike,
    options: _TokenPathOptions,
) -> NormalizedLoss:
    """Score exact sampled actions with population-weighted chunk contributions.

    Sequence reductions sum assistant-turn losses within each trajectory;
    token reductions use every sampled response token. Entropy always uses
    its own token denominator. A callback releases each chunk's graph before
    the next forward. Native callbacks preserve these separate gradient banks.
    """
    batch = _build_token_batch(
        trajectory_groups, rows_per_group, config, agent, options
    )
    _prepare_streamed_backward(options.backward_fn, batch)
    return run_token_local(
        agent.model,
        "forwards",
        lambda: _token_forwards(batch, agent, options.backward_fn),
    )


def _try_token_path(
    trajectory_groups: Sequence[TrajectoryGroupLike],
    config: LossConfig,
    agent: AgentLike,
    options: _TokenPathOptions,
) -> NormalizedLoss | None:
    """Run the batched token path when every group carries token metadata."""
    groups = [g for g in trajectory_groups if getattr(g, "trajectories", None)]
    if not groups:
        if getattr(agent.model, "_stateset_token_forward", False) is not True:
            return None
        return _compute_token_path_loss([], [], config, agent, options)
    rows = [_token_rows_for_group(g) for g in groups]
    if any(r is None for r in rows):
        require_explicit_fallback(
            config,
            "allow_sequence_fallback",
            "Rollouts lack exact sampled token metadata",
        )
        return None
    return _compute_token_path_loss(
        groups, [r for r in rows if r], config, agent, options
    )


def _group_rewards(group: TrajectoryGroupLike, device: DeviceLike) -> Tensor:
    """``total_reward`` of every trajectory in ``group`` as fp32 ``[G]``."""
    import torch

    return torch.tensor(
        [cast(ScoredTrajectory, t).total_reward for t in group.trajectories],
        dtype=torch.float32,
        device=device,
    )


def _float_moments(values: list[float]) -> dict[str, float]:
    """Mean, population std and nonzero share of logged advantages."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "nonzero": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "nonzero": float(np.count_nonzero(values) / len(values)),
    }


def compute_grpo_loss(
    trajectory_groups: Sequence[TrajectoryGroupLike],
    config: LossConfig,
    agent: AgentLike,
    global_reward_mean: float,
    global_reward_count: int,
    update_global_stats: Callable[[float, int], None] | None,
    old_logprobs: Sequence[Tensor] | None = None,
    *,
    global_reward_sum: float | None = None,
    backward_fn: BackwardFn | None = None,
) -> dict[str, Any]:
    """
    Compute GRPO loss from trajectory groups with configurable baseline and normalization.

    ``old_logprobs`` (see :func:`compute_token_old_logprobs`) is honoured on the
    batched token path only, for multiple inner updates per rollout batch.

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
    token_result = _try_token_path(
        trajectory_groups,
        config,
        agent,
        _TokenPathOptions(
            old_logprobs=old_logprobs,
            global_reward_mean=global_reward_mean,
            global_reward_count=global_reward_count,
            global_reward_sum=global_reward_sum,
            update_global_stats=update_global_stats,
            backward_fn=backward_fn,
        ),
    )
    if token_result is not None:
        return token_result

    import torch

    device = _resolve_model_device(agent, torch)

    policy_losses = []
    all_advantages_for_logging: list[float] = []
    entropy_values_for_logging: list[float] = []
    entropy_coef = float(getattr(config, "entropy_coef", 0.0))

    for group in trajectory_groups:
        if not group.trajectories:
            continue

        advantages = _sequence_group_advantages(
            _group_rewards(group, device),
            config,
            global_reward_mean=global_reward_mean if global_reward_count > 0 else None,
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
    moments = _float_moments(all_advantages_for_logging)
    return {
        "path": "sequence",
        "objective": _grpo_objective(config).name,
        "fallback": "explicit_sequence_compatibility",
        "nonzero_advantage_fraction": moments["nonzero"],
        "policy_loss": total_loss_tensor,
        "total_loss": total_loss_tensor,
        "mean_advantage": moments["mean"],
        "advantage_std": moments["std"],
        "entropy": entropy_value,
    }


def compute_ppo_ratio(
    new_log_prob_sum: Tensor, old_log_prob_sum: Tensor, token_count: int
) -> Tensor:
    """Length-normalized PPO importance ratio.

    Ratios computed from raw summed log-probs blow up (overflow) or vanish
    (underflow) for long responses, since the sum scales with sequence
    length. Dividing by `token_count` first yields a bounded, O(1)
    sequence-mean ratio, matching standard PPO/GRPO implementations.
    """
    import torch

    return torch.exp((new_log_prob_sum - old_log_prob_sum) / max(token_count, 1))


def compute_entropy_bonus(logits: Tensor, response_mask: Tensor | None) -> Tensor:
    """Differentiable masked-mean policy entropy from logits.

    Unlike a `torch.no_grad()` estimate, this stays in the autograd graph so
    the entropy bonus actually contributes gradient to the loss it's added
    to (encouraging exploration), rather than being a dead metric.
    """
    import torch.nn.functional as F

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


def _out_of_memory_types(torch_mod: object) -> tuple[type[BaseException], ...]:
    """torch's typed OOM errors: ``torch.OutOfMemoryError`` (2.5+) and
    ``torch.cuda.OutOfMemoryError`` (1.13+); empty on older torch."""
    found: list[type[BaseException]] = []
    for owner in (torch_mod, getattr(torch_mod, "cuda", None)):
        oom = getattr(owner, "OutOfMemoryError", None)
        if isinstance(oom, type) and issubclass(oom, BaseException):
            found.append(oom)
    return tuple(found)


def _is_fatal_loss_error(error: BaseException, torch: object) -> bool:
    """Errors a per-trajectory loss loop re-raises unchanged.

    Out-of-memory and rollout-contract violations are not properties of one
    trajectory, and callers act on their type (OOM recovery, configuration
    errors), so they are not wrapped in :class:`TrajectoryLossError`. OOM is
    detected by torch's exception types; the message is matched only on
    torch versions too old to have them.
    """
    if isinstance(error, RolloutContractError):
        return True
    oom_types = _out_of_memory_types(torch)
    if oom_types:
        return isinstance(error, oom_types)
    return "out of memory" in str(error).lower()


def _old_log_prob_sum(trajectory: object, device: DeviceLike) -> Tensor | None:
    """The rollout-time summed log-prob of ``trajectory``, if it carries one.

    Read from ``trajectory.log_probs`` or ``trajectory.metadata["log_probs"]``
    (a tensor, a list of per-token values or a scalar sum).
    """
    import torch

    log_probs = getattr(trajectory, "log_probs", None)
    if log_probs is None and hasattr(trajectory, "metadata"):
        log_probs = trajectory.metadata.get("log_probs")
    if log_probs is None:
        return None
    if isinstance(log_probs, torch.Tensor):
        return log_probs.sum().detach()
    if isinstance(log_probs, (list, tuple)):
        return torch.tensor(float(sum(log_probs)), device=device)
    if isinstance(log_probs, (int, float)):
        return torch.tensor(float(log_probs), device=device)
    return None


def _supervised_token_count(labels: Tensor | None) -> int:
    """Number of supervised (non ``-100``) label positions."""
    import torch

    if isinstance(labels, torch.Tensor):
        return int(labels.ne(-100).sum().item())
    return 0


def _nothing_supervised(labels: Tensor | None) -> bool:
    """Whether every label is masked (e.g. the reply was truncated away)."""
    import torch

    return isinstance(labels, torch.Tensor) and not bool(labels.ne(-100).any())


def _entropy_bonus(outputs: CausalLMOutput, labels: Tensor) -> Tensor:
    """Differentiable entropy over supervised positions of one forward pass.

    Logits at position ``t`` predict token ``t+1``: shift like the NLL.
    """
    response_mask = labels[:, 1:].ne(-100).to(dtype=outputs.logits.dtype)
    return compute_entropy_bonus(outputs.logits[:, :-1], response_mask)


def _unclipped_fallback(config: LossConfig) -> None:
    """Allow the unclipped REINFORCE loss only when explicitly requested."""
    require_explicit_fallback(
        config,
        "allow_sequence_fallback",
        "Rollouts lack old policy log probabilities",
    )
    global _warned_missing_log_probs
    if not _warned_missing_log_probs:
        logger.warning(
            "clip_ratio set but trajectories lack log_probs; "
            "using unclipped REINFORCE loss."
        )
        _warned_missing_log_probs = True


def _sequence_policy_term(
    trajectory: object,
    advantage: Tensor,
    config: LossConfig,
    agent: AgentLike,
    objective: PolicyObjective,
    *,
    clip_ratio: float,
    entropy_coef: float,
    device: DeviceLike,
) -> tuple[Tensor, float | None] | None:
    """One trajectory's policy loss and entropy (``None``: nothing supervised).

    GRPO policy gradient ``advantage * nll`` (``outputs.loss`` is the mean
    per-token NLL), or the clipped sequence-ratio loss when the rollout kept
    its old log-probs. No further length division: the NLL is already a
    per-token mean and the clipped branch normalises inside the ratio
    (dividing again gave a 1/L^2 bias).
    """
    import torch

    inputs, labels = _prepare_inputs_and_labels(trajectory, agent, config)
    if _nothing_supervised(labels):
        return None
    with torch.set_grad_enabled(True):
        outputs = _policy(agent)(**inputs, labels=labels)
        nll = outputs.loss
        policy_loss = advantage * nll
        token_count = _supervised_token_count(labels)
        old_log_prob = _old_log_prob_sum(trajectory, device)
        if clip_ratio > 0 and old_log_prob is not None and token_count > 0:
            policy_loss = _clipped_trajectory_loss(
                outputs.loss, token_count, advantage, old_log_prob, objective
            )
        elif clip_ratio > 0:
            # Without old log_probs, PPO-style clipping is impossible
            # (clip_ratio operates in ratio space, not advantage magnitude).
            _unclipped_fallback(config)
            policy_loss = advantage * nll
        entropy = None
        if entropy_coef > 0 and isinstance(labels, torch.Tensor):
            # Differentiable: from the same grad-enabled forward pass, so it
            # contributes gradient rather than being a dead no-grad metric.
            entropy_bonus = _entropy_bonus(outputs, labels)
            policy_loss = policy_loss - entropy_coef * entropy_bonus
            entropy = float(entropy_bonus.detach().item())
        return policy_loss, entropy


def _compute_group_policy_loss(
    group: TrajectoryGroupLike,
    advantages: Tensor,
    config: LossConfig,
    agent: AgentLike,
    entropy_coef: float = 0.0,
) -> tuple[Tensor, float | None]:
    """
    Compute policy loss for a single trajectory group with proper GRPO implementation.

    A trajectory whose loss raises fails the group with
    :class:`TrajectoryLossError` (out-of-memory and rollout-contract errors
    propagate unchanged); a trajectory with no supervised tokens is skipped.

    Args:
        group: TrajectoryGroup object
        advantages: Tensor of advantages for each trajectory
        config: Training configuration
        agent: The agent being trained
        entropy_coef: Coefficient for the (differentiable) entropy bonus

    Returns:
        Tuple of (policy loss tensor, mean entropy for logging or None)
    """
    import torch

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
    trajectories = group.trajectories

    for traj_idx, (trajectory, advantage) in enumerate(
        zip(trajectories, advantages, strict=False)
    ):
        try:
            term = _sequence_policy_term(
                trajectory,
                advantage,
                config,
                agent,
                objective,
                clip_ratio=clip_ratio,
                entropy_coef=entropy_coef,
                device=device,
            )
        except LOSS_EXCEPTIONS as e:
            if _is_fatal_loss_error(e, torch):
                raise
            raise TrajectoryLossError(traj_idx, len(trajectories), e) from e
        if term is None:
            continue  # nothing to supervise (e.g. reply truncated away)
        policy_loss, entropy = term
        if entropy is not None:
            entropy_sum += entropy
            entropy_count += 1
        total_loss = total_loss + policy_loss
        num_trajectories += 1

    mean_entropy = (entropy_sum / entropy_count) if entropy_count > 0 else None

    # Average over trajectories in the group
    if num_trajectories > 0:
        return total_loss / num_trajectories, mean_entropy
    else:
        return torch.tensor(0.0, device=device, requires_grad=True), mean_entropy


def _sequence_kl(
    outputs: CausalLMOutput,
    reference_model: PolicyModel,
    inputs: dict[str, Tensor],
    labels: Tensor | None,
) -> Tensor:
    """Masked mean token KL(p || p_ref) at supervised positions.

    ``KL = sum_v p(v) * (log p(v) - log p_ref(v))`` per token; logits at ``t``
    predict the scored token at ``t+1``, as in the NLL.
    """
    import torch

    log_probs = outputs.logits.log_softmax(dim=-1)
    with torch.no_grad():
        ref_outputs = reference_model(**inputs)
        ref_log_probs = ref_outputs.logits.log_softmax(dim=-1)
    kl_per_token = (log_probs.exp() * (log_probs - ref_log_probs)).sum(dim=-1)
    if isinstance(labels, torch.Tensor):
        kl_per_token = kl_per_token[:, :-1]
        loss_mask = labels[:, 1:].ne(-100)
        if loss_mask.any():
            return (kl_per_token * loss_mask).sum() / loss_mask.sum()
        return kl_per_token.mean()
    return kl_per_token.mean()  # pragma: no cover - non-tensor labels (stub)


@dataclass
class _EnhancedTerms:
    """Per-trajectory terms of the enhanced sequence loss, in batch order."""

    losses: list[Tensor] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    kl_divs: list[Tensor] = field(default_factory=list)
    entropy_values: list[float] = field(default_factory=list)


def _enhanced_trajectory_terms(
    terms: _EnhancedTerms,
    trajectory: object,
    advantage: Tensor,
    config: LossConfig,
    agent: AgentLike,
    *,
    beta: float,
    reference_model: PolicyModel | None,
    objective: PolicyObjective,
    entropy_coef: float,
    device: DeviceLike,
) -> None:
    """Append one trajectory's policy loss, entropy and KL to ``terms``."""
    import torch

    inputs, labels = _prepare_inputs_and_labels(trajectory, agent, config)
    if _nothing_supervised(labels):
        return  # nothing to supervise (e.g. reply truncated away)
    use_amp = getattr(config, "bf16", False) or getattr(config, "fp16", False)
    autocast_ctx = torch.amp.autocast(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        enabled=bool(use_amp),
    )
    with autocast_ctx:
        outputs = _policy(agent)(**inputs, labels=labels)
        # Policy loss with advantage weighting; optional PPO-style clipping
        # mirrors `_compute_group_policy_loss`.
        nll = outputs.loss
        policy_loss = advantage * nll
        clip_ratio = getattr(config, "clip_ratio", getattr(config, "clip_epsilon", 0.2))
        old_log_prob = _old_log_prob_sum(trajectory, device)
        token_count = _supervised_token_count(labels)
        if clip_ratio > 0 and old_log_prob is not None and token_count > 0:
            policy_loss = _clipped_trajectory_loss(
                nll, token_count, advantage, old_log_prob, objective
            )
        if entropy_coef > 0 and isinstance(labels, torch.Tensor):
            entropy_bonus = _entropy_bonus(outputs, labels)
            policy_loss = policy_loss - entropy_coef * entropy_bonus
            terms.entropy_values.append(float(entropy_bonus.detach().item()))
        terms.losses.append(policy_loss)
        # Skip the full-vocab log_softmax entirely when there's no KL
        # penalty to compute — it's an expensive (batch, seq, vocab)
        # materialization that's otherwise dead weight.
        if beta > 0 and reference_model is not None:
            terms.kl_divs.append(_sequence_kl(outputs, reference_model, inputs, labels))


def _enhanced_loss_result(
    terms: _EnhancedTerms,
    beta: float,
    device: DeviceLike,
    trajectory_groups: Sequence[TrajectoryGroupLike],
) -> dict[str, Any]:
    """Aggregate the enhanced loss terms into the loss and its metrics."""
    import torch

    all_advantages = terms.advantages
    if terms.losses:
        policy_loss = torch.stack(terms.losses).mean()
        # Add KL penalty if applicable
        if terms.kl_divs and beta > 0:
            kl_penalty = torch.stack(terms.kl_divs).mean()
            total_loss = policy_loss + beta * kl_penalty
        else:
            total_loss = policy_loss
            kl_penalty = torch.tensor(0.0, device=device)
    else:
        total_loss = torch.tensor(0.0, requires_grad=True, device=device)
        policy_loss = total_loss
        kl_penalty = torch.tensor(0.0, device=device)

    entropy_values = terms.entropy_values
    return {
        "path": "sequence",
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "kl_penalty": kl_penalty,
        "entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "mean_advantage": np.mean(all_advantages) if all_advantages else 0.0,
        "nonzero_advantage_fraction": (
            float(np.count_nonzero(all_advantages) / len(all_advantages))
            if all_advantages
            else 0.0
        ),
        "advantage_std": np.std(all_advantages) if all_advantages else 0.0,
        "num_trajectories": sum(len(g.trajectories) for g in trajectory_groups),
    }


def compute_enhanced_grpo_loss(
    trajectory_groups: Sequence[TrajectoryGroupLike],
    beta: float,
    config: LossConfig,
    agent: AgentLike,
    reference_model: PolicyModel | None = None,
    old_logprobs: Sequence[Tensor] | None = None,
    *,
    global_reward_mean: float | None = None,
    global_reward_count: int = 0,
    global_reward_sum: float | None = None,
    update_global_stats: Callable[[float, int], None] | None = None,
    backward_fn: BackwardFn | None = None,
) -> dict[str, Any]:
    """
    Enhanced GRPO loss computation with KL penalty and proper advantages.

    Args:
        trajectory_groups: List of TrajectoryGroup objects
        beta: KL penalty coefficient
        config: Training configuration
        agent: The agent being trained
        reference_model: Optional reference model for KL computation

    Returns:
        Dictionary containing loss tensors and metrics
    """
    token_result = _try_token_path(
        trajectory_groups,
        config,
        agent,
        _TokenPathOptions(
            beta=beta,
            reference_model=reference_model,
            old_logprobs=old_logprobs,
            global_reward_mean=global_reward_mean,
            global_reward_count=global_reward_count,
            global_reward_sum=global_reward_sum,
            update_global_stats=update_global_stats,
            backward_fn=backward_fn,
        ),
    )
    if token_result is not None:
        return token_result

    import torch

    device = _resolve_model_device(agent, torch)
    terms = _EnhancedTerms()
    entropy_coef = float(getattr(config, "entropy_coef", 0.0))
    objective = _grpo_objective(config)

    for group in trajectory_groups:
        if not group.trajectories:
            continue
        advantages = _sequence_group_advantages(
            _group_rewards(group, device),
            config,
            global_reward_mean=global_reward_mean if global_reward_count > 0 else None,
            update_global_stats=update_global_stats,
        )
        terms.advantages.extend(advantages.tolist())
        for trajectory, advantage in zip(group.trajectories, advantages, strict=False):
            _enhanced_trajectory_terms(
                terms,
                trajectory,
                advantage,
                config,
                agent,
                beta=beta,
                reference_model=reference_model,
                objective=objective,
                entropy_coef=entropy_coef,
                device=device,
            )
    return _enhanced_loss_result(terms, beta, device, trajectory_groups)


def _format_trajectory_for_model(trajectory: object, agent: AgentLike) -> str:
    """Format trajectory into text for model input."""
    messages = _trajectory_to_messages(trajectory)

    tokenizer = agent.tokenizer
    if _uses_chat_template(tokenizer):
        # Use tokenizer's chat template
        rendered: object = tokenizer.apply_chat_template(
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


def _trajectory_to_messages(trajectory: object) -> list[dict[str, str]]:
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
    trajectory: object,
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


def _as_bool_mask(
    value: object, like: Tensor, allow_empty: bool = False
) -> Tensor | None:
    """Coerce a tokenizer-produced mask to a bool tensor shaped like ``like``."""
    import torch

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
    tokenizer: ChatTokenizerLike, messages: list[dict[str, str]], max_length: int
) -> tuple[Tensor, Tensor | None, Tensor] | None:
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
    tokenizer: ChatTokenizerLike,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str | None:
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    except _TEMPLATE_EXCEPTIONS:
        return None
    return rendered if isinstance(rendered, str) else None


def _chat_template_prefix_mask(
    tokenizer: ChatTokenizerLike, messages: list[dict[str, str]], max_length: int
) -> tuple[Tensor, Tensor | None, Tensor] | None:
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
        # Typed ``object``: fakes and odd tokenizers may return non-mappings.
        encoded: object = tokenizer(
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
    import torch

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


def _unsupervised_inputs(
    tokenizer: TokenizerLike, trajectory: object, agent: AgentLike, max_length: int
) -> tuple[dict[str, Tensor], Tensor]:
    """Inputs for ``trajectory`` with every label masked (nothing supervised)."""
    global _warned_observation_fallback
    if not _warned_observation_fallback:
        logger.warning(
            "Could not locate assistant tokens in a conversation with tool "
            "observations; skipping it rather than training on tool output."
        )
        _warned_observation_fallback = True
    text, _spans = _format_trajectory_with_spans(trajectory)
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    input_ids = inputs["input_ids"]
    labels = input_ids.clone().fill_(-100)
    device = _model_device(agent)
    moved = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    return moved, labels.to(device)


def _model_device(agent: AgentLike) -> DeviceLike:
    """``agent.model.device`` (an ``AttributeError`` when the model has none)."""
    return cast(_DeviceCarrier, agent.model).device


def _uses_chat_template(tokenizer: object) -> TypeGuard[ChatTokenizerLike]:
    """HF tokenizers expose ``apply_chat_template`` even with no template set."""
    return hasattr(tokenizer, "apply_chat_template") and not (
        hasattr(tokenizer, "chat_template")
        and getattr(tokenizer, "chat_template", None) is None
    )


def _chat_template_inputs(
    tokenizer: ChatTokenizerLike,
    trajectory: object,
    max_length: int,
    device: DeviceLike,
) -> tuple[dict[str, Tensor], Tensor] | None:
    """Assistant-only labels from the chat template: the template's own
    ``{% generation %}`` mask, else spans found by rendering prefixes."""
    messages = _trajectory_to_messages(trajectory)
    encoded = _chat_template_generation_mask(tokenizer, messages, max_length)
    if encoded is None:
        encoded = _chat_template_prefix_mask(tokenizer, messages, max_length)
    if encoded is None:
        global _warned_missing_assistant_mask
        if not _warned_missing_assistant_mask:
            logger.warning(
                "Could not derive assistant token masks from the chat template; "
                "loss will include non-assistant tokens."
            )
            _warned_missing_assistant_mask = True
        return None
    input_ids, attention_mask, assistant_mask = encoded
    labels = input_ids.clone().masked_fill(~assistant_mask, -100)
    inputs: dict[str, Tensor] = {"input_ids": input_ids}
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask.eq(0), -100)
        inputs["attention_mask"] = attention_mask
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs, labels.to(device)


def _offsets_list(offsets: object) -> list[Sequence[int]]:
    """First row of a tokenizer's offset mapping as a list of ``(start, end)``."""
    import torch

    if isinstance(offsets, torch.Tensor):
        return list(offsets[0].tolist())
    if isinstance(offsets, (list, tuple)):
        return list(offsets[0]) if offsets else []
    return []


def _offset_assistant_mask(
    input_ids: Tensor,
    offsets_list: list[Sequence[int]],
    spans: list[tuple[int, int]],
) -> Tensor:
    """Bool mask of tokens overlapping any assistant character span."""
    import torch

    assistant_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for idx, (start, end) in enumerate(offsets_list):
        if start == end:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            assistant_mask[0, idx] = True
    return assistant_mask


def _offset_masked_inputs(
    tokenizer: TokenizerLike,
    conversation_text: str,
    assistant_spans: list[tuple[int, int]],
    max_length: int,
    device: DeviceLike,
) -> tuple[dict[str, Tensor], Tensor | None] | None:
    """Assistant-only labels via offset mappings (tokenizers without a chat
    template); ``None`` when the tokenizer returns no usable offsets."""
    import torch

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
    if offsets is None or input_ids is None:
        return None
    offsets_list = _offsets_list(offsets)
    if not offsets_list or len(offsets_list) != input_ids.shape[1]:
        return None
    assistant_mask = _offset_assistant_mask(input_ids, offsets_list, assistant_spans)
    labels = input_ids.clone() if hasattr(input_ids, "clone") else input_ids
    labels = labels.masked_fill(~assistant_mask, -100)
    if "attention_mask" in inputs and isinstance(labels, torch.Tensor):
        labels = labels.masked_fill(inputs["attention_mask"].eq(0), -100)
    moved = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if hasattr(labels, "to"):
        labels = labels.to(device)
    return moved, labels


def _offset_inputs_or_error(
    tokenizer: TokenizerLike, trajectory: object, max_length: int, device: DeviceLike
) -> tuple[tuple[dict[str, Tensor], Tensor | None] | None, BaseException | None]:
    """Offset-masked inputs, or the tokenizer error that prevented them
    (e.g. slow tokenizers have no offset mapping)."""
    conversation_text, assistant_spans = _format_trajectory_with_spans(trajectory)
    if not assistant_spans:
        return None, None
    try:
        masked = _offset_masked_inputs(
            tokenizer, conversation_text, assistant_spans, max_length, device
        )
    except _OFFSET_EXCEPTIONS as error:
        return None, error
    return masked, None


def _allow_unmasked_labels(config: LossConfig, error: BaseException) -> None:
    """Gate the every-token fallback taken when offsets are unavailable.

    Supervising user (and system) tokens changes the estimator, so it needs
    ``allow_unmasked_labels``; otherwise raise ``RolloutContractError``
    chained to the tokenizer's error.
    """
    try:
        require_explicit_fallback(
            config,
            UNMASKED_LABELS_OPTION,
            "The tokenizer cannot map assistant spans to tokens "
            f"({type(error).__name__}: {error}), so the loss would also train "
            "on user and system tokens",
        )
    except RolloutContractError as contract:
        raise contract from error
    global _warned_unmasked_labels
    if not _warned_unmasked_labels:
        logger.warning(
            "Tokenizer offsets unavailable (%s); %s=True supervises every token.",
            type(error).__name__,
            UNMASKED_LABELS_OPTION,
        )
        _warned_unmasked_labels = True


def _full_stream_inputs(
    tokenizer: TokenizerLike,
    trajectory: object,
    agent: AgentLike,
    use_chat_template: bool,
    max_length: int,
    device: DeviceLike,
) -> tuple[dict[str, Tensor], Tensor | None]:
    """Fallback: supervise the full token stream (padding excluded)."""
    import torch

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

    moved = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if labels is not None and hasattr(labels, "to"):
        labels = labels.to(device)
    return moved, labels


def _prepare_inputs_and_labels(
    trajectory: object, agent: AgentLike, config: LossConfig
) -> tuple[dict[str, Tensor], Tensor | None]:
    """Prepare model inputs and loss labels, masking to assistant tokens when possible.

    Order: the chat template's assistant mask; offset mappings for tokenizers
    without a template; nothing supervised for conversations with tool
    observations (never train on tool output); else the full token stream.
    """
    max_length = int(getattr(config, "max_prompt_length", 512)) + int(
        getattr(config, "max_completion_length", 512)
    )
    tokenizer = _tokenizer(agent)
    device = _model_device(agent)
    chat_tokenizer = tokenizer if _uses_chat_template(tokenizer) else None
    use_chat_template = chat_tokenizer is not None
    offsets_error: BaseException | None = None
    masked: tuple[dict[str, Tensor], Tensor | None] | None
    if chat_tokenizer is not None:
        masked = _chat_template_inputs(chat_tokenizer, trajectory, max_length, device)
    else:
        masked, offsets_error = _offset_inputs_or_error(
            tokenizer, trajectory, max_length, device
        )
    if masked is not None:
        return masked
    if has_observation_turns(getattr(trajectory, "turns", [])):
        # Tool observations must never be trained on: without faithful
        # assistant spans, supervise nothing (the caller skips the row).
        return _unsupervised_inputs(tokenizer, trajectory, agent, max_length)
    if offsets_error is not None:
        _allow_unmasked_labels(config, offsets_error)
    return _full_stream_inputs(
        tokenizer, trajectory, agent, use_chat_template, max_length, device
    )
