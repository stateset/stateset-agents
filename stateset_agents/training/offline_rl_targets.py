"""Target policy and critic operations shared by the sampled offline backups."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Integral
from typing import Any

from .rl_losses import _bellman_columns, _discounted_backup


def validate_target_parameters(
    discount: float,
    tau: float,
    frequency: object = 1,
    proposals: object = 1,
    min_weight: float = 1.0,
) -> None:
    """Validate the domains used by the target interpolation and sampled maximum."""
    for name, value in (
        ("discount_factor", discount),
        ("soft_target_tau", tau),
        ("lmbda", min_weight),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one")
    for name, count in (
        ("target_update_frequency", frequency),
        ("proposal count", proposals),
    ):
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError(f"{name} must be a positive integer")


def critic_network(linear_layers: Iterable[Any], activation: str) -> Any:
    """Assemble a scalar critic while preserving the optimizer's linear parameters."""
    from torch import nn

    make_activation = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}.get(
        activation
    )
    layers: list[nn.Module] = []
    for layer in linear_layers:
        if layers and make_activation:
            layers.append(make_activation())
        layers.append(layer)
    return nn.Sequential(*layers)


def restore_optimizer(optimizer: Any, state: dict | None, learning_rate: float) -> None:
    """Restore Adam, starting with fresh moments for a legacy weights-only save."""
    if state is not None:
        optimizer.load_state_dict(state)
    else:
        optimizer.state.clear()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate


def _mix(old: Any, new: Any, weight: float) -> Any:
    """Interpolate finite target coordinates while preserving the endpoints."""
    import torch

    if weight == 0:
        return old
    if weight == 1:
        return new
    dtype = torch.float32 if old.device.type == "mps" else torch.float64
    a, b = old.to(dtype), new.to(dtype)
    if old.dtype == dtype:
        # Rescale tiny pairs so weighting a subnormal does not round too early.
        mag = torch.maximum(a.abs(), b.abs())
        unit = torch.ones_like(mag)
        scale = 2.0 ** (-512 if dtype == torch.float64 else -64)
        unit.masked_fill_(mag < scale, scale)
        a, b = a / unit, b / unit
    # Opposite signs make a+b safe, even when b-a overflows. Factoring
    # around the midpoint preserves a small residual when a is close to -b.
    if weight <= 0.5:
        opposite = (1 - 2 * weight) * a + weight * (a + b)
    else:
        opposite = (2 * weight - 1) * b + (1 - weight) * (a + b)
    result = torch.where(
        torch.signbit(a) != torch.signbit(b), opposite, torch.lerp(a, b, weight)
    )
    if old.dtype == dtype:
        result = result * unit
    return result.clamp(min=torch.minimum(old, new), max=torch.maximum(old, new)).to(
        old.dtype
    )


def update_target_networks(pairs: Iterable[tuple[Any, Any]], tau: float) -> None:
    """Advance each (target, live) network from the same completed learner step."""
    import torch

    with torch.no_grad():
        for target, live in pairs:
            for old, new in zip(target.parameters(), live.parameters(), strict=True):
                old.copy_(_mix(old, new, tau))


def successor_target(
    rewards: Any,
    dones: Any,
    next_states: Any,
    discount: float,
    evaluate: Callable[..., Any],
    *next_features: Any,
) -> Any:
    """Evaluate a frozen critic only on rows with a live Bellman continuation."""
    import numpy as np
    import torch

    with torch.no_grad():
        count = next_states.shape[0]
        reward, terminal = _bellman_columns(rewards, dones, count)
        if any(
            feature.ndim < 1 or feature.shape[0] != count for feature in next_features
        ):
            raise ValueError("Successor features must align with the transition rows")
        active = ~terminal.flatten()
        # Offline integer label columns can carry information beyond binary64.
        # Widen the published target on devices that support double precision.
        mixed = isinstance(reward, np.ndarray) and reward.dtype == object
        wide_labels = (
            not mixed
            and reward.dtype in (torch.int32, torch.uint32, torch.int64, torch.uint64)
            and reward.device.type != "mps"
        )

        def backup(future: Any, gamma: float) -> Any:
            if mixed or wide_labels:
                from .integer_bellman import exact_backup

                device = next_states.device
                dtype = torch.float32 if device.type == "mps" else torch.float64
                return exact_backup(reward, future, gamma, dtype, device, terminal)
            return _discounted_backup(reward, future, gamma)

        if discount == 0 or not active.any():
            if mixed or wide_labels:
                return backup(None, 0.0)
            if not reward.is_floating_point() and next_states.is_floating_point():
                return backup(next_states.new_zeros(reward.shape), 0.0)
            return reward.to(
                torch.promote_types(reward.dtype, next_states.dtype)
            ).detach()
        states = next_states[active]
        values = evaluate(states, *(feature[active] for feature in next_features))
        if values.shape != (states.shape[0], 1):
            raise ValueError("Expected one scalar value per continuing successor")
        # Scatter unsigned values through equal-width signed storage. Some
        # PyTorch backends lack masked assignment for unsigned integer types.
        signed = {
            torch.uint16: torch.int16,
            torch.uint32: torch.int32,
            torch.uint64: torch.int64,
        }
        storage = (
            values.view(signed[values.dtype]) if values.dtype in signed else values
        )
        future = storage.new_zeros(reward.shape)
        future[active] = storage
        return backup(future.view(values.dtype), discount)


def sampled_twin_target(
    rewards: Any,
    dones: Any,
    next_states: Any,
    discount: float,
    sample: Callable[[Any], Any],
    q1: Any,
    q2: Any,
    *,
    min_weight: float = 1.0,
    perturb: Callable[[Any, Any], Any] | None = None,
) -> Any:
    """Score each successor's own proposals and bootstrap only continuing rows."""
    import torch

    def evaluate(states: Any) -> Any:
        actions = sample(states)
        if (
            actions.ndim != 3
            or actions.shape[0] != states.shape[0]
            or not actions.shape[1]
        ):
            raise ValueError(
                "Expected [successors, positive proposals, action features]"
            )
        count = actions.shape[1]
        flat_states = states.repeat_interleave(count, dim=0)
        flat_actions = actions.flatten(0, 1)
        if perturb is not None:
            flat_actions = perturb(flat_states, flat_actions)
        first, second = q1(flat_states, flat_actions), q2(flat_states, flat_actions)
        if first.shape != (flat_states.shape[0], 1) or second.shape != first.shape:
            raise ValueError("Each target critic must return one scalar per proposal")
        lower = torch.minimum(first, second)
        scores = (
            lower
            if min_weight == 1
            else _mix(torch.maximum(first, second), lower, min_weight)
        )
        return scores.reshape(states.shape[0], count).amax(dim=1, keepdim=True)

    return successor_target(rewards, dones, next_states, discount, evaluate)
