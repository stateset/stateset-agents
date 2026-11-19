"""
Rust-accelerated computation module for StateSet RL Framework.

This module provides Python bindings to high-performance Rust implementations
of critical RL operations. Falls back to pure Python implementations when
the Rust extension is not available.

Usage:
    from stateset_agents.core.rust_accelerator import (
        compute_group_advantages,
        compute_gae,
        normalize_rewards,
    )
"""

import logging
import math
import sys
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# Try to import Rust extension
_RUST_AVAILABLE = False
try:
    import stateset_rl_core as _rust_core

    _RUST_AVAILABLE = True
    logger.info("Rust acceleration enabled (stateset_rl_core)")
except ImportError:
    logger.info(
        "Rust acceleration not available. Install with: "
        "cd rust_core && maturin develop --release"
    )
    _rust_core = None


def is_rust_available() -> bool:
    """Check if Rust acceleration is available."""
    return _RUST_AVAILABLE


def compute_group_advantages(
    rewards: np.ndarray,
    baseline_type: str = "mean",
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute group-relative advantages for GRPO training.

    Uses Rust acceleration when available, falls back to NumPy.

    Args:
        rewards: 2D array of shape (num_groups, group_size) or 1D array
        baseline_type: "mean", "median", or "min"
        normalize: Whether to normalize advantages

    Returns:
        Array of advantages
    """
    rewards = np.asarray(rewards, dtype=np.float64)

    if _RUST_AVAILABLE and rewards.ndim == 2:
        return cast(
            np.ndarray,
            np.asarray(
                _rust_core.compute_group_advantages(rewards, baseline_type, normalize)
            ),
        )

    from .reward_moments import RewardMoments

    if rewards.ndim == 1:
        rewards = rewards.reshape(1, -1)
    all_advantages: list[float] = []
    for group in rewards:
        if not np.isfinite(group).all():
            all_advantages.extend([0.0] * len(group))
            continue
        pop = RewardMoments(group.tolist())
        if normalize and pop.above(1e-8):
            unit = pop.scale()
            p, q = unit.as_integer_ratio()
            denom = pop.deviation(unit) + 1e-8 / unit
            all_advantages.extend(
                (v * q / (pop.denominator * p)) / denom for v in pop.residuals
            )
        elif pop.count:
            residuals = pop.residuals
            denominator = pop.denominator
            if baseline_type == "min":
                minimum = min(residuals)
                residuals = [v - minimum for v in residuals]
            elif baseline_type == "median":
                ordered = sorted(residuals)
                middle = ordered[(pop.count - 1) // 2] + ordered[pop.count // 2]
                residuals = [2 * v - middle for v in residuals]
                denominator *= 2
            for numerator in residuals:
                try:
                    all_advantages.append(numerator / denominator)
                except OverflowError:
                    all_advantages.append(math.inf if numerator > 0 else -math.inf)
    return cast(np.ndarray, np.asarray(all_advantages, dtype=np.float64))


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: Per-step rewards
        values: Value estimates (length = len(rewards) + 1 for bootstrap)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        Array of advantage estimates
    """
    return compute_gae_targets(rewards, values, gamma, gae_lambda)[0]


def compute_gae_targets(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Return separate advantages and lambda returns, including an optional final bootstrap."""
    rewards = np.ascontiguousarray(rewards, dtype=np.float64)
    values = np.ascontiguousarray(values, dtype=np.float64)
    if _RUST_AVAILABLE and hasattr(_rust_core, "compute_gae_targets"):
        adv, targets = _rust_core.compute_gae_targets(
            rewards, values, gamma, gae_lambda
        )
        return np.asarray(adv), np.asarray(targets)
    n = len(rewards)
    advantages, targets = np.empty(n), np.empty(n)
    reward_part = 0.0
    weight = gamma * gae_lambda
    tiny_weight = abs(weight) < sys.float_info.min
    value_part = float(values[n]) if n < len(values) else 0.0
    for t in reversed(range(n)):
        current = float(values[t]) if t < len(values) else 0.0
        successor = float(values[t + 1]) if t + 1 < len(values) else 0.0
        reward = float(rewards[t])
        if gamma == 0 or gae_lambda == 0:
            reward_part = reward
            value_part = 0.0 if gamma == 0 else gamma * successor
        else:
            carry = (
                gamma * (gae_lambda * reward_part)
                if tiny_weight
                else weight * reward_part
            )
            reward_part = reward + carry
            if gae_lambda != 1:
                unit = (
                    2.0**-512
                    if max(abs(successor), abs(value_part)) < 2.0**-512
                    else 1.0
                )
                value_part = unit * (
                    (1 - gae_lambda) * (successor / unit)
                    + gae_lambda * (value_part / unit)
                )
            value_part *= gamma
        target = reward_part + value_part
        targets[t] = target
        large, small = (
            (reward_part, value_part)
            if abs(reward_part) >= abs(value_part)
            else (value_part, reward_part)
        )
        advantages[t] = (
            (target - current) + ((large - target) + small)
            if math.isfinite(target)
            else (large - current) + small
        )
    return advantages, targets


def batch_compute_gae(
    all_rewards: list[np.ndarray],
    all_values: list[np.ndarray],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> list[np.ndarray]:
    """
    Batch compute GAE for multiple trajectories in parallel.

    Args:
        all_rewards: List of reward arrays
        all_values: List of value arrays
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        List of advantage arrays
    """
    if len(all_rewards) != len(all_values):
        raise ValueError("GAE batch rows must align")
    all_rewards = [np.asarray(r, dtype=np.float64) for r in all_rewards]
    all_values = [np.asarray(v, dtype=np.float64) for v in all_values]

    if _RUST_AVAILABLE and hasattr(_rust_core, "compute_gae_targets"):
        return [
            cast(np.ndarray, np.asarray(a))
            for a in _rust_core.batch_compute_gae(
                all_rewards, all_values, gamma, gae_lambda
            )
        ]

    # Sequential fallback
    return [
        compute_gae(r, v, gamma, gae_lambda)
        for r, v in zip(all_rewards, all_values, strict=False)
    ]


def normalize_rewards(
    rewards: np.ndarray,
    running_mean: float = 0.0,
    running_var: float = 1.0,
    count: int = 0,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, float, float, int]:
    """
    Normalize rewards using running statistics.

    Args:
        rewards: Rewards to normalize
        running_mean: Current running mean
        running_var: Current running variance
        count: Current sample count
        epsilon: Numerical stability constant

    Returns:
        Tuple of (normalized_rewards, new_mean, new_var, new_count)
    """
    rewards = np.asarray(rewards, dtype=np.float64)

    if _RUST_AVAILABLE:
        raw_normalized, raw_mean, raw_var, raw_count = _rust_core.normalize_rewards(
            rewards, running_mean, running_var, count, epsilon
        )
        return (
            cast(np.ndarray, np.asarray(raw_normalized)),
            float(raw_mean),
            float(raw_var),
            int(raw_count),
        )

    # Pure Python Welford's algorithm
    mean = running_mean
    m2 = running_var * count
    n = count

    for reward in rewards:
        n += 1
        delta = reward - mean
        mean += delta / n
        delta2 = reward - mean
        m2 += delta * delta2

    new_var = m2 / n if n > 1 else 0.0
    std = np.sqrt(new_var + epsilon)
    normalized = (rewards - mean) / std

    return normalized, mean, new_var, n


def clip_rewards(
    rewards: np.ndarray,
    min_val: float,
    max_val: float,
) -> np.ndarray:
    """Clip rewards to a specified range."""
    rewards = np.asarray(rewards, dtype=np.float64)

    if _RUST_AVAILABLE:
        return cast(
            np.ndarray, np.asarray(_rust_core.clip_rewards(rewards, min_val, max_val))
        )

    return cast(np.ndarray, np.clip(rewards, min_val, max_val))


def compute_gspo_importance_ratios(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    sequence_lengths: np.ndarray,
) -> np.ndarray:
    """
    Compute GSPO sequence-level importance ratios.

    Implements: s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)

    Args:
        log_probs_new: Sum of log probs under new policy per sequence
        log_probs_old: Sum of log probs under old policy per sequence
        sequence_lengths: Length of each sequence

    Returns:
        Array of importance ratios
    """
    log_probs_new = np.asarray(log_probs_new, dtype=np.float64)
    log_probs_old = np.asarray(log_probs_old, dtype=np.float64)
    sequence_lengths = np.asarray(sequence_lengths, dtype=np.int64)

    if not (
        log_probs_new.ndim == 1
        and log_probs_new.shape == log_probs_old.shape == sequence_lengths.shape
    ):
        raise ValueError("GSPO requires aligned vectors of sequence scores and lengths")

    if _RUST_AVAILABLE:
        return cast(
            np.ndarray,
            np.asarray(
                _rust_core.compute_gspo_importance_ratios(
                    log_probs_new, log_probs_old, sequence_lengths
                )
            ),
        )

    # Pure Python fallback
    log_ratios = log_probs_new - log_probs_old
    lengths = np.maximum(sequence_lengths, 1)  # Avoid division by zero
    normalized_log_ratios = np.where(sequence_lengths > 0, log_ratios / lengths, 0.0)
    return cast(np.ndarray, np.exp(np.clip(normalized_log_ratios, -20.0, 20.0)))


def apply_gspo_clipping(
    ratios: np.ndarray,
    advantages: np.ndarray,
    clip_left: float = 3e-4,
    clip_right: float = 4e-4,
) -> np.ndarray:
    """
    Apply GSPO clipping to importance ratios.

    Args:
        ratios: Importance ratios
        advantages: Advantage values
        clip_left: Left clipping bound
        clip_right: Right clipping bound

    Returns:
        Clipped surrogate objectives
    """
    ratios = np.asarray(ratios, dtype=np.float64)
    advantages = np.asarray(advantages, dtype=np.float64)

    if _RUST_AVAILABLE:
        return cast(
            np.ndarray,
            np.asarray(
                _rust_core.apply_gspo_clipping(
                    ratios, advantages, clip_left, clip_right
                )
            ),
        )

    # Pure Python fallback
    unclipped = ratios * advantages
    clipped_ratios = np.where(
        advantages >= 0,
        np.minimum(ratios, 1.0 + clip_right),
        np.maximum(ratios, 1.0 - clip_left),
    )
    clipped = clipped_ratios * advantages
    return cast(np.ndarray, np.minimum(unclipped, clipped))


def compute_ppo_surrogate(
    ratios: np.ndarray,
    advantages: np.ndarray,
    clip_epsilon: float = 0.2,
) -> np.ndarray:
    """
    Compute PPO clipped surrogate objective.

    Args:
        ratios: Importance ratios (π_new / π_old)
        advantages: Advantage estimates
        clip_epsilon: PPO clipping parameter

    Returns:
        Clipped surrogate objectives
    """
    ratios = np.asarray(ratios, dtype=np.float64)
    advantages = np.asarray(advantages, dtype=np.float64)

    if _RUST_AVAILABLE:
        return cast(
            np.ndarray,
            np.asarray(
                _rust_core.compute_ppo_surrogate(ratios, advantages, clip_epsilon)
            ),
        )

    unclipped = ratios * advantages
    clipped = np.clip(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    return cast(np.ndarray, np.minimum(unclipped, clipped))


def compute_reward_statistics(rewards: list[float]) -> dict[str, float]:
    """Compute comprehensive reward statistics."""
    if _RUST_AVAILABLE:
        raw_stats = cast(dict[str, Any], _rust_core.compute_reward_statistics(rewards))
        return {
            key: float(value)
            for key, value in raw_stats.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    if not rewards:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "count": 0.0,
        }

    arr = np.array(rewards)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "count": float(len(arr)),
    }


# Convenience class for stateful normalization
class RunningRewardNormalizer:
    """Maintains running statistics for reward normalization."""

    def __init__(self, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.epsilon = epsilon

    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards and update statistics."""
        normalized, self.mean, self.var, self.count = normalize_rewards(
            rewards, self.mean, self.var, self.count, self.epsilon
        )
        return normalized

    def reset(self):
        """Reset running statistics."""
        self.mean = 0.0
        self.var = 1.0
        self.count = 0


__all__ = [
    "is_rust_available",
    "compute_group_advantages",
    "compute_gae",
    "batch_compute_gae",
    "normalize_rewards",
    "clip_rewards",
    "compute_gspo_importance_ratios",
    "apply_gspo_clipping",
    "compute_ppo_surrogate",
    "compute_reward_statistics",
    "RunningRewardNormalizer",
]
