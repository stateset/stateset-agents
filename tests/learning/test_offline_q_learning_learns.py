"""CQL and IQL learn to value the rewarded action of a logged bandit."""

from __future__ import annotations

import pytest
import torch

from tests.learning._offline_toy import (
    ACTION_DIM,
    STATE_DIM,
    bandit_dataset,
    heldout_states,
    mean_gap,
)
from tests.learning._toy import seeds

pytestmark = pytest.mark.learning


def _trainer(algorithm: str, seed: int):
    from stateset_agents.experimental.offline_rl.algorithms import (
        CQLConfig,
        IQLConfig,
        OfflineRLTrainer,
    )

    torch.manual_seed(seed)
    common = {"hidden_size": 64, "num_layers": 2, "learning_rate": 1e-3}
    config = CQLConfig(**common) if algorithm == "cql" else IQLConfig(**common)
    return OfflineRLTrainer(algorithm, STATE_DIM, ACTION_DIM, config, device="cpu")


def _train(algorithm: str, seed: int, sign: float):
    trainer = _trainer(algorithm, seed)
    torch.manual_seed(seed)
    trainer.train(bandit_dataset(seed, sign=sign), num_epochs=60, batch_size=32)
    learner = trainer.learner
    score = learner.get_q_value if algorithm == "cql" else learner.get_advantage
    return mean_gap(score, heldout_states(seed))


@pytest.mark.parametrize("algorithm", ["cql", "iql"])
@pytest.mark.parametrize("seed", seeds())
def test_offline_q_learning_prefers_rewarded_action(algorithm: str, seed: int):
    # Ideal gap is 1.0 (Q for CQL, advantage for IQL); observed 0.72-1.01.
    gap = _train(algorithm, seed, sign=1.0)
    assert gap > 0.5, gap


@pytest.mark.parametrize("algorithm", ["cql", "iql"])
@pytest.mark.parametrize("seed", seeds())
def test_offline_q_learning_negated_reward_flips_preference(algorithm: str, seed: int):
    gap = _train(algorithm, seed, sign=-1.0)
    assert gap < -0.5, gap
