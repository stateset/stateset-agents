"""BCQ learns to value, and to select, the rewarded action of a logged bandit."""

from __future__ import annotations

import pytest
import torch

from tests.learning._offline_toy import (
    ACTION_DIM,
    BAD_ACTION,
    GOOD_ACTION,
    STATE_DIM,
    bandit_dataset,
    heldout_states,
    mean_gap,
)
from tests.learning._toy import seeds

pytestmark = pytest.mark.learning


def _train(seed: int, sign: float):
    from stateset_agents.training.offline_rl_bcq import BCQConfig, BCQTrainer

    torch.manual_seed(seed)
    config = BCQConfig(
        hidden_size=64,
        num_layers=2,
        latent_dim=4,
        vae_hidden_size=64,
        learning_rate=1e-3,
        vae_learning_rate=1e-3,
    )
    trainer = BCQTrainer(STATE_DIM, ACTION_DIM, config, device="cpu")
    trainer.train(bandit_dataset(seed, sign=sign), num_epochs=60, batch_size=32)
    return trainer.bcq


def _selection_preference(bcq, states: torch.Tensor) -> float:
    """Mean of (distance to BAD - distance to GOOD) for selected actions."""
    torch.manual_seed(0)
    chosen = bcq.select_action(states)
    good = torch.as_tensor(GOOD_ACTION)
    bad = torch.as_tensor(BAD_ACTION)
    return float(
        ((chosen - bad).norm(dim=-1) - (chosen - good).norm(dim=-1)).mean().item()
    )


@pytest.mark.parametrize("seed", seeds())
def test_bcq_prefers_rewarded_action(seed: int) -> None:
    bcq = _train(seed, sign=1.0)
    states = heldout_states(seed)
    gap = mean_gap(bcq.get_q_value, states)
    preference = _selection_preference(bcq, states)
    assert gap > 0.5, gap
    # |GOOD - BAD| = 1.41; observed preference 1.06-1.41 (Q gap ~1.0).
    assert preference > 0.5, preference


@pytest.mark.parametrize("seed", seeds())
def test_bcq_negated_reward_flips_preference(seed: int) -> None:
    bcq = _train(seed, sign=-1.0)
    states = heldout_states(seed)
    assert mean_gap(bcq.get_q_value, states) < -0.5
    assert _selection_preference(bcq, states) < -0.5
