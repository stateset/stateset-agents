"""Active-learning label updates must actually train the reward model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stateset_agents.rewards.bayesian_reward_model import (  # noqa: E402
    ActiveLearningSelector,
    BayesianRewardConfig,
    BayesianRewardFunction,
)


def _params(reward_fn):
    model = reward_fn.model
    modules = getattr(model, "models", [model])
    return [p.detach().clone() for m in modules for p in m.parameters()]


@pytest.mark.parametrize("use_ensemble", [True, False])
def test_update_with_labels_changes_weights(use_ensemble):
    torch.manual_seed(0)
    config = BayesianRewardConfig(
        use_ensemble=use_ensemble, hidden_size=8, num_layers=2, num_ensemble=2
    )
    reward_fn = BayesianRewardFunction(input_dim=4, config=config, device="cpu")
    before = _params(reward_fn)
    selector = ActiveLearningSelector()
    metrics = selector.update_with_labels(
        reward_fn,
        [np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)],
        [1.0, 0.0],
        num_update_steps=3,
        learning_rate=1e-2,
    )
    after = _params(reward_fn)
    assert metrics["update_loss"] != 0.0
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
