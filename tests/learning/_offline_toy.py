"""Logged two-action bandit shared by the offline-RL learning tests.

Every logged state appears twice: once with the GOOD action (reward 1) and
once with the BAD action (reward 0). The behaviour policy is uniform, so any
preference a learner ends up with must come from the rewards. ``sign=-1``
negates the rewards; the learned preference must then flip.
"""

from __future__ import annotations

import numpy as np
import torch

STATE_DIM = 4
ACTION_DIM = 2
GOOD_ACTION = np.array([0.5, 0.5], dtype=np.float32)
BAD_ACTION = np.array([-0.5, -0.5], dtype=np.float32)


def bandit_dataset(
    seed: int, *, num_states: int = 64, sign: float = 1.0
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(num_states, STATE_DIM)).astype(np.float32)
    all_states = np.concatenate([states, states])
    actions = np.concatenate(
        [np.tile(GOOD_ACTION, (num_states, 1)), np.tile(BAD_ACTION, (num_states, 1))]
    )
    rewards = sign * np.concatenate([np.ones(num_states), np.zeros(num_states)])
    return {
        "states": all_states,
        "actions": actions.astype(np.float32),
        "rewards": rewards.astype(np.float32),
        "next_states": rng.normal(size=all_states.shape).astype(np.float32),
        "dones": np.ones(2 * num_states, dtype=np.float32),
    }


def heldout_states(seed: int, n: int = 32) -> torch.Tensor:
    rng = np.random.default_rng(10_000 + seed)
    return torch.as_tensor(rng.normal(size=(n, STATE_DIM)).astype(np.float32))


def mean_gap(score, states: torch.Tensor) -> float:
    """Mean of score(s, GOOD) - score(s, BAD) over held-out states."""
    good = torch.as_tensor(GOOD_ACTION)
    bad = torch.as_tensor(BAD_ACTION)
    gaps = [float(score(s, good)) - float(score(s, bad)) for s in states]
    return sum(gaps) / len(gaps)
