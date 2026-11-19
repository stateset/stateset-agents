"""Batched actor advantages and critic targets for finite trajectories."""

from __future__ import annotations

import torch


def gae_dtype(rewards: torch.Tensor, values: torch.Tensor) -> torch.dtype:
    """Use a real floating dtype for returns and their residuals."""
    dtype = torch.promote_types(rewards.dtype, values.dtype)
    if dtype.is_complex:
        raise ValueError("GAE requires real rewards and values")
    return dtype if dtype.is_floating_point else torch.get_default_dtype()


def terminal_rewards(
    reward: float | torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    """Place each episode reward at its final retained action."""
    rows, width = action_mask.shape
    if isinstance(reward, torch.Tensor):
        scalar = reward.to(
            device=action_mask.device, dtype=gae_dtype(reward, action_mask)
        )
    else:
        dtype = (
            action_mask.dtype
            if action_mask.is_floating_point()
            else torch.get_default_dtype()
        )
        scalar = torch.tensor(reward, device=action_mask.device, dtype=dtype)
    if scalar.ndim == 0:
        scalar = scalar.expand(rows)
    if scalar.shape != (rows,):
        raise ValueError("Expected one episode reward per response row")
    result = scalar.new_zeros((rows, width))
    if width:
        columns = torch.arange(width, device=action_mask.device)
        last = torch.where(action_mask > 0, columns, -1).amax(dim=1)
        active = last >= 0
        result[active, last[active]] = scalar[active]
    return result


def normalize_advantages(
    raw: torch.Tensor, active: torch.Tensor | None = None
) -> torch.Tensor:
    """Center selected advantages and divide by their sample deviation plus 1e-8."""
    valid = raw.flatten() if active is None else raw[active]
    if valid.numel() < 2:
        return raw if active is None else raw.masked_fill(~active, 0)
    device = "cpu" if raw.device.type == "mps" else raw.device
    values = valid.to(device=device, dtype=torch.float64)
    # An exact power of two keeps the variance finite at either end of the
    # exponent range. Subtract a nearby origin before computing the mean so
    # close, large advantages retain their differences.
    peak = values.detach().abs().amax().clamp_min(1)
    _, exponent = torch.frexp(peak)
    unit = torch.ldexp(torch.ones_like(peak), exponent - 1)
    scaled = values / unit
    offsets = scaled - scaled[0].detach()
    normalized = (offsets - offsets.mean()) / (offsets.std() + 1e-8 / unit)
    normalized = normalized.to(device=raw.device, dtype=raw.dtype)
    if active is None:
        return normalized.reshape_as(raw)
    return torch.zeros_like(raw).masked_scatter(active, normalized)


def gae_targets(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    trace_decay: float,
    dones: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan reward and bootstrap contributions separately with a zero final bootstrap."""
    if rewards.ndim != 2 or values.shape != rewards.shape:
        raise ValueError("GAE requires aligned [batch, time] rewards and values")
    if dones is not None and dones.shape != rewards.shape:
        raise ValueError("GAE terminal flags must align with rewards")
    dtype = torch.float32 if rewards.device.type == "mps" else torch.float64
    advantages = torch.empty_like(rewards, dtype=gae_dtype(rewards, values))
    r, v = rewards.to(dtype), values.to(dtype)
    returns = torch.empty_like(advantages)
    reward_part = r.new_zeros(r.shape[0])
    value_part = torch.zeros_like(reward_part)
    successor = torch.zeros_like(reward_part)
    weight = gamma * trace_decay
    tiny_weight = abs(weight) < torch.finfo(dtype).tiny
    for t in reversed(range(r.shape[1])):
        # A terminal step cuts both traces before inspecting its successor.
        if dones is not None:
            terminal = dones[:, t].bool()
            reward_part = reward_part.masked_fill(terminal, 0)
            value_part = value_part.masked_fill(terminal, 0)
            successor = successor.masked_fill(terminal, 0)
        if gamma == 0:
            reward_part = r[:, t]
            value_part = torch.zeros_like(successor)
        else:
            if trace_decay == 0:
                reward_part = r[:, t]
                value_part = gamma * successor
            else:
                # Apply the factors to the carry separately if their product
                # would lose precision before a large continuation is read.
                carry = (
                    gamma * (trace_decay * reward_part)
                    if tiny_weight
                    else weight * reward_part
                )
                reward_part = r[:, t] + carry
                if trace_decay != 1:
                    if dtype == values.dtype:
                        # Scale tiny pairs by an exact power of two so the two
                        # weighted terms do not underflow before their sum.
                        scale = 2.0 ** (-512 if dtype == torch.float64 else -64)
                        unit = torch.where(
                            torch.maximum(successor.abs(), value_part.abs()) < scale,
                            torch.full_like(successor, scale),
                            1.0,
                        )
                        mixed = (1 - trace_decay) * (successor / unit) + trace_decay * (
                            value_part / unit
                        )
                        value_part = gamma * (mixed * unit)
                    else:
                        value_part = gamma * (
                            (1 - trace_decay) * successor + trace_decay * value_part
                        )
                # With unit decay and zero final bootstrap this trace stays zero.
        target = reward_part + value_part
        # FastTwoSum retains a small reward hidden by a much larger value.
        large = reward_part.abs() >= value_part.abs()
        residual = torch.where(
            large,
            (reward_part - target) + value_part,
            (value_part - target) + reward_part,
        )
        alt = torch.where(
            large,
            (reward_part - v[:, t]) + value_part,
            (value_part - v[:, t]) + reward_part,
        )
        advantages[:, t] = torch.where(
            torch.isfinite(target), (target - v[:, t]) + residual, alt
        )
        returns[:, t] = target
        successor = v[:, t]
    return advantages, returns
