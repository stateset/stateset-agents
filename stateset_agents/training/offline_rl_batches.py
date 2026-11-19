"""Shared transition batching for the offline critic trainers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.reward_columns import finite_rewards, reward_array
from .rl_losses import _bellman_columns

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CriticObservation:
    """One sampled transition and the auxiliary value critic's pre-update outputs."""

    state: tuple[float, ...]
    action: tuple[float, ...]
    successor: tuple[float, ...] | None
    reward: int | float
    prediction: float
    frozen_prediction: float
    target: float


@dataclass(frozen=True)
class CriticBatchSnapshot:
    """Immutable CPU scalars from a single auxiliary critic update."""

    discount: float
    rows: tuple[CriticObservation, ...]

    @classmethod
    def capture(
        cls,
        discount: float,
        states: Any,
        actions: Any,
        next_states: Any,
        dones: Any,
        rewards: Any,
        predictions: Any,
        frozen: Any,
        targets: Any,
    ) -> CriticBatchSnapshot:
        """Copy the values used by regression before either auxiliary optimizer moves."""
        return cls(
            discount,
            tuple(
                CriticObservation(
                    tuple(s), tuple(a), None if done else tuple(j), r, u, v, y
                )
                for s, a, j, done, r, u, v, y in zip(
                    states.tolist(),
                    actions.tolist(),
                    next_states.tolist(),
                    dones.reshape(-1).tolist(),
                    rewards.reshape(-1).tolist(),
                    predictions.reshape(-1).tolist(),
                    frozen.reshape(-1).tolist(),
                    targets.reshape(-1).tolist(),
                    strict=True,
                )
            ),
        )


def transition_terminals(dataset: Mapping[str, Any], count: int, device: str) -> Any:
    """Read environment termination before converting dataset features to trainer precision."""
    import torch

    def flags(key: str) -> Any:
        raw = dataset[key]
        value = raw if torch.is_tensor(raw) else torch.as_tensor(np.asarray(raw))
        if value.shape not in ((count,), (count, 1)):
            raise ValueError(f"Expected one {key} scalar per transition")
        if value.is_complex() or not ((value == 0) | (value == 1)).all():
            raise ValueError("Bellman terminal flags must be boolean or binary")
        return value.reshape(count, 1).to(device=device, dtype=torch.bool)

    if "terminated" not in dataset:
        if "dones" not in dataset:
            raise ValueError("Offline data requires terminated or dones")
        return flags("dones")
    terminal = flags("terminated")
    cutoff = (
        flags("truncated") if "truncated" in dataset else torch.zeros_like(terminal)
    )
    if "dones" in dataset:
        done = flags("dones")
        if ((terminal & ~done) | (done & ~(terminal | cutoff))).any():
            raise ValueError(
                "Offline dones disagree with terminated and truncated flags"
            )
    return terminal


def transition_batch(
    states: Any,
    actions: Any,
    rewards: Any,
    next_states: Any,
    dones: Any,
    *,
    state_dim: int,
    action_dim: int,
    discount: float,
    next_actions: Any = None,
) -> tuple[Any, Any]:
    """Check a whole batch before an optimizer can update on an unpaired TD target."""
    if states.ndim != 2 or not len(states) or states.shape[1] != state_dim:
        raise ValueError(
            "Offline RL requires a nonempty matrix of states with the configured width"
        )
    n = len(states)
    if actions.shape != (n, action_dim) or next_states.shape != states.shape:
        raise ValueError("States, actions and successors must align by transition")
    if next_actions is not None and next_actions.shape != actions.shape:
        raise ValueError("Logged next actions must align by transition")
    reward, terminal = _bellman_columns(rewards, dones, n)
    valid_rewards = (
        finite_rewards(reward)
        if isinstance(reward, np.ndarray)
        else reward.isfinite().all()
    )
    finite = states.isfinite().all() & actions.isfinite().all() & valid_rewards
    if discount != 0:
        successors = next_states.isfinite().all(dim=1, keepdim=True)
        if next_actions is not None:
            successors = successors & next_actions.isfinite().all(dim=1, keepdim=True)
        finite = finite & (terminal | successors).all()
    if not finite:
        raise ValueError("Offline features, rewards and live successors must be finite")
    return reward, terminal


def _reward_tensor(value: Any, device: str) -> Any:
    """Store the original integer when a floating label would lose source bits."""
    import torch

    target = torch.device(device)
    narrow = value.to(torch.float32)
    if value.dtype in (torch.int32, torch.uint32, torch.int64, torch.uint64):
        # Check the exclusive upper endpoint before casting back. An out-of-range
        # conversion may saturate to the original maximum on some backends.
        upper = float(torch.iinfo(value.dtype).max + 1)
        if (narrow >= upper).any() or not torch.equal(narrow.to(value.dtype), value):
            return value.to(target)
    elif value.dtype == torch.float64 and target.type != "mps":
        if not torch.equal(narrow.double(), value):
            return value.to(target)
    return narrow.to(target)


def transition_tensors(
    dataset: Mapping[str, Any],
    *,
    algorithm: str,
    state_dim: int,
    action_dim: int,
    device: str,
    discount: float,
) -> dict[str, Any]:
    """Read the logged rows shared by standalone and language-model critic pretraining."""
    import torch

    fields = ["states", "actions", "rewards", "next_states"]
    if algorithm == "cql" and "next_actions" in dataset:
        fields.append("next_actions")
    if any(key not in dataset for key in fields):
        raise ValueError(
            "Offline data requires states, actions, rewards and next_states"
        )
    data = {}
    for key in fields:
        raw = dataset[key]
        if key == "rewards" and not torch.is_tensor(raw):
            raw = reward_array(raw)
            if raw.dtype == object:
                data[key] = raw
                continue
        try:
            value = raw if torch.is_tensor(raw) else torch.as_tensor(np.asarray(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Offline {key} must contain real features") from exc
        if value.is_complex():
            raise ValueError(f"Offline {key} must contain real features")
        data[key] = (
            _reward_tensor(value, device)
            if key == "rewards"
            else value.to(device, torch.float32)
        )
    if data["states"].ndim != 2:
        raise ValueError("Offline RL requires a matrix of states")
    data["dones"] = transition_terminals(dataset, len(data["states"]), device)
    data["rewards"], terminal = transition_batch(
        **data, state_dim=state_dim, action_dim=action_dim, discount=discount
    )
    data["dones"] = terminal.float()
    return data


def decision_sequences(
    raw: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    discount: float,
) -> list[dict[str, np.ndarray]]:
    """Partition ordered decision rows into episodes and assign observed return labels."""
    from stateset_agents.data.discounted_returns import discounted_returns

    rewards, terminal = data["rewards"], data["dones"]
    n = len(rewards)
    ends = terminal.astype(bool).copy()
    if "truncated" in raw:
        cutoff = transition_terminals({"dones": raw["truncated"]}, n, "cpu")
        ends |= cutoff.flatten().numpy()
    ends[-1] = True  # a recording may end with an incomplete episode
    recorded = None
    if "returns_to_go" in raw:
        recorded = np.asarray(raw["returns_to_go"])
        if recorded.shape not in ((n,), (n, 1)) or recorded.dtype.kind not in "fiu":
            raise ValueError("Expected one real return-to-go per recorded decision")
        recorded = recorded.reshape(n)
        if not np.isfinite(recorded).all():
            raise ValueError("Recorded returns-to-go must be finite")
    trajectories = []
    start = 0
    for last in np.flatnonzero(ends):
        stop = int(last) + 1
        if discount and not np.array_equal(
            data["next_states"][start:last], data["states"][start + 1 : stop]
        ):
            raise ValueError(
                "Decision Transformer rows must be ordered within each recorded episode"
            )
        if recorded is None:
            if not terminal[last] and discount:
                raise ValueError(
                    "A continuing recording cutoff needs observed returns_to_go for Decision Transformer conditioning"
                )
            rtg = np.asarray(discounted_returns(rewards[start:stop].tolist(), discount))
        else:
            rtg = recorded[start:stop]
        trajectories.append(
            {
                "states": data["states"][start:stop],
                "actions": data["actions"][start:stop],
                "returns_to_go": rtg,
                "timesteps": np.arange(stop - start),
            }
        )
        start = stop
    return trajectories


def transition_epochs(
    dataset: dict[str, np.ndarray],
    train_step: Callable[..., dict[str, float]],
    *,
    algorithm: str,
    state_dim: int,
    action_dim: int,
    device: str,
    num_epochs: int,
    batch_size: int,
    discount: float = 1.0,
) -> Iterator[dict[str, float]]:
    """Visit every transition once per epoch and combine per-transition metrics."""
    import torch

    if batch_size < 1:
        raise ValueError("Offline RL batch_size must be positive")
    data = transition_tensors(
        dataset,
        algorithm=algorithm,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        discount=discount,
    )
    dataset_size = len(data["states"])

    logger.info(
        "Training %s on %d samples for %d epochs",
        algorithm.upper(),
        dataset_size,
        num_epochs,
    )
    for epoch in range(num_epochs):
        indices = torch.randperm(dataset_size)
        averages: dict[str, float] = {}
        for batch_indices in indices.split(batch_size):
            batch = {
                key: (
                    tensor[batch_indices.cpu().numpy()]
                    if isinstance(tensor, np.ndarray)
                    else tensor[batch_indices]
                )
                for key, tensor in data.items()
            }
            metrics = train_step(**batch)
            weight = batch_indices.numel() / dataset_size
            for key, value in metrics.items():
                if key == "training_step":
                    averages[key] = value
                else:
                    averages[key] = averages.get(key, 0.0) + weight * value
        averages["epoch"] = float(epoch)
        if epoch % 10 == 0:
            logger.info(
                "Epoch %d/%d: %s",
                epoch,
                num_epochs,
                ", ".join(
                    f"{k}={v:.4f}"
                    for k, v in averages.items()
                    if k not in ("epoch", "training_step")
                ),
            )
        yield averages
