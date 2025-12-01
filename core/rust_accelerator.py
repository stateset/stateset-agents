"""
Rust-accelerated computation module for StateSet RL Framework.

This module provides Python bindings to high-performance Rust implementations
of critical RL operations. Falls back to pure Python implementations when
the Rust extension is not available.

Usage:
    from core.rust_accelerator import (
        compute_group_advantages,
        compute_gae,
        normalize_rewards,
    )
"""

import logging
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# Try to import Rust extension
_RUST_AVAILABLE = False
try:
    import stateset_rl_core as _rust_core
    _RUST_AVAILABLE = True
    logger.info("Rust acceleration enabled (stateset_rl_core)")
except ImportError:
    logger.warning(
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
        return np.asarray(_rust_core.compute_group_advantages(
            rewards, baseline_type, normalize
        ))

    # Pure Python fallback
    if rewards.ndim == 1:
        rewards = rewards.reshape(1, -1)

    all_advantages = []
    for group_rewards in rewards:
        if baseline_type == "median":
            baseline = np.median(group_rewards)
        elif baseline_type == "min":
            baseline = np.min(group_rewards)
        else:
            baseline = np.mean(group_rewards)

        advantages = group_rewards - baseline

        if normalize and len(advantages) > 1:
            std = np.std(advantages)
            if std > 1e-8:
                advantages = (advantages - np.mean(advantages)) / (std + 1e-8)

        all_advantages.extend(advantages)

    return np.array(all_advantages)


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
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if _RUST_AVAILABLE:
        return np.asarray(_rust_core.compute_gae(rewards, values, gamma, gae_lambda))

    # Pure Python fallback
    n = len(rewards)
    advantages = np.zeros(n)
    gae = 0.0

    for t in reversed(range(n)):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        current_value = values[t] if t < len(values) else 0.0

        delta = rewards[t] + gamma * next_value - current_value
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    return advantages


def batch_compute_gae(
    all_rewards: List[np.ndarray],
    all_values: List[np.ndarray],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> List[np.ndarray]:
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
    all_rewards = [np.asarray(r, dtype=np.float64) for r in all_rewards]
    all_values = [np.asarray(v, dtype=np.float64) for v in all_values]

    if _RUST_AVAILABLE:
        return [np.asarray(a) for a in _rust_core.batch_compute_gae(
            all_rewards, all_values, gamma, gae_lambda
        )]

    # Sequential fallback
    return [compute_gae(r, v, gamma, gae_lambda)
            for r, v in zip(all_rewards, all_values)]


def normalize_rewards(
    rewards: np.ndarray,
    running_mean: float = 0.0,
    running_var: float = 1.0,
    count: int = 0,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, float, float, int]:
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
        return _rust_core.normalize_rewards(
            rewards, running_mean, running_var, count, epsilon
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
        return np.asarray(_rust_core.clip_rewards(rewards, min_val, max_val))

    return np.clip(rewards, min_val, max_val)


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

    if _RUST_AVAILABLE:
        return np.asarray(_rust_core.compute_gspo_importance_ratios(
            log_probs_new, log_probs_old, sequence_lengths
        ))

    # Pure Python fallback
    log_ratios = log_probs_new - log_probs_old
    lengths = np.maximum(sequence_lengths, 1)  # Avoid division by zero
    normalized_log_ratios = log_ratios / lengths
    return np.exp(normalized_log_ratios)


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
        return np.asarray(_rust_core.apply_gspo_clipping(
            ratios, advantages, clip_left, clip_right
        ))

    # Pure Python fallback
    unclipped = ratios * advantages
    clipped_ratios = np.where(
        advantages >= 0,
        np.minimum(ratios, 1.0 + clip_right),
        np.maximum(ratios, 1.0 - clip_left)
    )
    clipped = clipped_ratios * advantages
    return np.minimum(unclipped, clipped)


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
        return np.asarray(_rust_core.compute_ppo_surrogate(
            ratios, advantages, clip_epsilon
        ))

    unclipped = ratios * advantages
    clipped = np.clip(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    return np.minimum(unclipped, clipped)


def compute_reward_statistics(rewards: List[float]) -> Dict[str, float]:
    """Compute comprehensive reward statistics."""
    if _RUST_AVAILABLE:
        return _rust_core.compute_reward_statistics(rewards)

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
