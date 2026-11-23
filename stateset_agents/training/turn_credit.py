"""Turn-Level Process Reward Credit Assignment for Multi-Turn Agentic RL.

In multi-turn tool-calling and coding episodes (3-12 turns), smearing a single
terminal scalar reward uniformly across every token forces early correct tool
calls to receive the same gradient sign as a late broken tool call.

This module computes **discounted turn-level returns** blending step-level
Process Reward Model (PRM) / tool-execution signals with terminal verifiable
rewards, performs group-relative advantage normalization across prompt groups,
and expands per-turn advantages cleanly onto token-level ``[B, T]`` masks.

Zero eager ``torch`` imports (works on pure Python lists and optional PyTorch tensors).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TurnCreditResult:
    """Computed per-turn returns and group-normalized advantages for one episode."""

    episode_index: int
    turn_returns: tuple[float, ...]
    turn_advantages: tuple[float, ...]
    terminal_reward: float
    mean_advantage: float


def compute_discounted_turn_returns(
    turn_process_rewards: list[float],
    terminal_reward: float,
    *,
    gamma: float = 0.98,
    prm_weight: float = 0.35,
) -> list[float]:
    """Blend per-turn PRM scores with discounted terminal verifiable reward.

    For turn ``t`` in ``0 .. T-1``:
        ``G_t = prm_weight * prm_t + (1 - prm_weight) * (gamma ** (T - 1 - t)) * R_terminal``
    """
    n = len(turn_process_rewards)
    if n == 0:
        return []
    w = max(0.0, min(1.0, float(prm_weight)))
    g = max(0.0, min(1.0, float(gamma)))
    term = float(terminal_reward)
    returns: list[float] = []
    for idx, prm in enumerate(turn_process_rewards):
        steps_to_end = (n - 1) - idx
        blended = w * float(prm) + (1.0 - w) * (g**steps_to_end) * term
        returns.append(round(blended, 6))
    return returns


def compute_turn_level_advantages(
    episodes: list[dict[str, Any]],
    *,
    gamma: float = 0.98,
    prm_weight: float = 0.35,
    min_std: float = 1e-4,
) -> list[TurnCreditResult]:
    """Compute group-relative turn-level advantages across a prompt group of episodes.

    Each episode dict expects:
    - ``turn_process_rewards``: ``list[float]`` (one per assistant turn)
    - ``terminal_reward``: ``float`` (verifiable final outcome reward)
    """
    raw_returns: list[list[float]] = []
    flat_values: list[float] = []
    for ep in episodes:
        prms = [float(x) for x in (ep.get("turn_process_rewards") or [])]
        term = float(ep.get("terminal_reward", 0.0))
        if not prms:
            prms = [term]
        ret = compute_discounted_turn_returns(
            prms, term, gamma=gamma, prm_weight=prm_weight
        )
        raw_returns.append(ret)
        flat_values.extend(ret)

    if not flat_values:
        return []

    mean = sum(flat_values) / len(flat_values)
    var = sum((v - mean) ** 2 for v in flat_values) / len(flat_values)
    std = math.sqrt(var)

    results: list[TurnCreditResult] = []
    for idx, (ep, ret) in enumerate(zip(episodes, raw_returns, strict=True)):
        if std < min_std:
            advs = [0.0 for _ in ret]
        else:
            advs = [round((v - mean) / std, 6) for v in ret]
        mean_adv = round(sum(advs) / len(advs), 6) if advs else 0.0
        results.append(
            TurnCreditResult(
                episode_index=idx,
                turn_returns=tuple(ret),
                turn_advantages=tuple(advs),
                terminal_reward=float(ep.get("terminal_reward", 0.0)),
                mean_advantage=mean_adv,
            )
        )
    return results


def expand_turn_advantages_to_tokens(
    turn_advantages: list[tuple[float, ...]],
    turn_token_spans: list[list[tuple[int, int]]],
    seq_len: int,
    *,
    as_torch: bool = False,
) -> Any:
    """Map per-turn advantages onto a ``[B, seq_len]`` token-level advantage matrix.

    ``turn_token_spans[b][t] = (start_idx, end_idx)`` gives the half-open
    response token span for assistant turn ``t`` in batch row ``b``.
    """
    bsz = len(turn_advantages)
    grid: list[list[float]] = [[0.0] * seq_len for _ in range(bsz)]
    for b in range(bsz):
        advs = turn_advantages[b]
        spans = turn_token_spans[b] if b < len(turn_token_spans) else []
        for adv, (s, e) in zip(advs, spans, strict=False):
            lo = max(0, min(seq_len, int(s)))
            hi = max(lo, min(seq_len, int(e)))
            for pos in range(lo, hi):
                grid[b][pos] = float(adv)

    if as_torch:
        from .trainer_utils import get_torch, require_torch

        torch = get_torch() or require_torch()
        return torch.tensor(grid, dtype=torch.float32)
    return grid


__all__ = [
    "TurnCreditResult",
    "compute_discounted_turn_returns",
    "compute_turn_level_advantages",
    "expand_turn_advantages_to_tokens",
]
