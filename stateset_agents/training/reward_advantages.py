"""Prompt-group reward baselines and population normalization for policy objectives."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from stateset_agents.core.reward_moments import PopulationMoments, RewardMoments


def reward_tensor(
    rewards: object, device: torch.device | str | None = None
) -> torch.Tensor:
    """Preserve the precision of scalar rollout rewards before computing a baseline."""
    if device is None and isinstance(rewards, torch.Tensor):
        device = rewards.device
    # MPS supports fp32 learner tensors. Compute reward statistics on the CPU.
    work = (
        "cpu" if device is not None and torch.device(device).type == "mps" else device
    )
    # NumPy preserves Python float precision while distinguishing integral sequences.
    source = np.asarray(rewards) if isinstance(rewards, (list, tuple)) else rewards
    values = torch.as_tensor(source, device=work)
    return (
        values.double() if values.is_floating_point() or values.is_complex() else values
    )


def _centered(
    pop: RewardMoments, values: torch.Tensor, scale: float, loo: bool = False
) -> torch.Tensor:
    """Convert exactly centered rewards to the requested scale, retaining autograd."""
    p, q = scale.as_integer_ratio()
    divisor = pop.denominator * p
    if loo:
        q *= pop.count
        divisor *= pop.count - 1
    centered = values.new_tensor([v * q / divisor for v in pop.residuals])
    if values.requires_grad:
        # The exact residuals are the value at this point. The centering map
        # is affine, so reattaching its linear part preserves all derivatives.
        delta = (values - values.detach()) / scale
        delta = delta - delta.mean()
        centered = centered + (delta * (pop.count / (pop.count - 1)) if loo else delta)
    return centered


def _normalizer(
    pop: RewardMoments, values: torch.Tensor, epsilon: float, correction: int = 0
) -> tuple[float, torch.Tensor]:
    scale = pop.scale()
    deviation = values.new_tensor(pop.deviation(scale))
    if values.requires_grad:
        live = torch.linalg.vector_norm(_centered(pop, values, scale)) / math.sqrt(
            pop.count
        )
        deviation = deviation + (live - live.detach())
    if correction:
        deviation = deviation * math.sqrt(pop.count / (pop.count - correction))
    return scale, deviation + epsilon / scale


def standardized_rewards(
    rewards: torch.Tensor, epsilon: float, correction: int = 0
) -> torch.Tensor:
    """Standardize a single reward group whenever its deviation is positive."""
    values = rewards.detach().reshape(-1).tolist()
    if any(not math.isfinite(v) for v in values):
        return torch.zeros_like(rewards, dtype=torch.float32)
    group = RewardMoments(values)
    if group.count <= correction or not group.variance_numerator:
        return torch.zeros_like(rewards, dtype=torch.float32)
    work = rewards.to(device="cpu", dtype=torch.float64)
    scale, denominator = _normalizer(group, work, epsilon, correction)
    return (_centered(group, work, scale) / denominator).to(
        device=rewards.device, dtype=torch.float32
    )


def grouped_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    kind: str,
    epsilon: float,
    *,
    population: PopulationMoments | None = None,
) -> torch.Tensor:
    """Compute group means, leave-one-out or standardized advantages, returning fp32."""
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Reward normalization epsilon must be finite and nonnegative")
    device = rewards.device
    original = rewards.to("cpu").reshape(-1)
    values = original.double()
    ids = group_ids.reshape(-1).tolist()
    if len(ids) != len(values):
        raise ValueError("rewards and group_ids must have the same shape")
    rows: dict[int, list[int]] = defaultdict(list)
    for i, gid in enumerate(ids):
        rows[gid].append(i)
    data = original.detach().tolist()
    result = torch.zeros_like(values, dtype=torch.float32)
    if values.requires_grad:
        # A discarded group contributes a differentiable zero, including
        # when one of its rewards is nonfinite.
        result = result + values[:0].sum().float()
    batch = None
    if kind == "batch_norm":
        if not data or not all(map(math.isfinite, data)):
            return result.to(device)
        if population is None:
            local = RewardMoments(data)
            if not local.above(epsilon):
                return result.to(device)
            batch = _normalizer(local, values, epsilon)
        else:
            if not population.above(epsilon):
                return result.to(device)
            scale = population.scale()
            batch = scale, values.new_tensor(
                population.deviation(scale) + epsilon / scale
            )

    for indices in rows.values():
        group = [data[i] for i in indices]
        if len(group) < 2 or not all(map(math.isfinite, group)):
            continue
        pop = RewardMoments(group)
        if not pop.variance_numerator:
            continue
        v = values[indices]
        if kind == "group_norm":
            if not pop.above(epsilon):
                continue
            scale, denominator = _normalizer(pop, v, epsilon)
        elif batch is not None:
            scale, denominator = batch
        else:
            # The public output is fp32. Preserve the existing whole-group
            # fallback if an unnormalized result exceeds its finite range.
            loo = kind == "leave_one_out"
            numerator = max(map(abs, pop.residuals)) * (pop.count if loo else 1)
            divisor = pop.denominator * (pop.count - 1 if loo else 1)
            if numerator > int(torch.finfo(torch.float32).max) * divisor:
                continue
            result[indices] = _centered(pop, v, 1.0, loo).float()
            continue
        result[indices] = (_centered(pop, v, scale) / denominator).float()
    return result.to(device)
