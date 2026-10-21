"""OfflineGRPOTrainer must bootstrap Q from V(s') and train on the logged data."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stateset_agents.experimental.offline_rl.offline_grpo import (  # noqa: E402
    OfflineGRPOConfig,
    OfflineGRPOTrainer,
)


def _trainer(**overrides):
    config = OfflineGRPOConfig(
        state_dim=2,
        action_dim=2,
        value_hidden_size=4,
        value_num_layers=2,
        pretrain_value_batch_size=4,
        pretrain_value_epochs=1,
        batch_size=2,
        **overrides,
    )
    return OfflineGRPOTrainer(config, device="cpu")


class _Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, states, actions=None):
        return torch.full((states.shape[0], 1), self.value)


def test_q_target_bootstraps_from_next_state_value():
    trainer = _trainer()
    trainer.value_target = _Constant(2.0)
    # A Q network that would return a large value if (wrongly) queried at s'.
    trainer.q_net = torch.nn.Module()
    seen = []

    def q_forward(states, actions):
        seen.append(states.clone())
        return torch.zeros(states.shape[0], 1, requires_grad=True)

    trainer.q_net.forward = q_forward
    trainer.q_net.parameters = lambda: iter([torch.nn.Parameter(torch.zeros(1))])
    trainer.q_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0)

    states = torch.ones(3, 2)
    next_states = torch.full((3, 2), 7.0)
    trainer._train_value_step(
        states,
        torch.ones(3, 2),
        torch.ones(3, 1),
        next_states,
        torch.zeros(3, 1),
    )
    assert all(not torch.equal(s, next_states) for s in seen)


@pytest.mark.asyncio
async def test_train_uses_logged_transitions_not_random_features():
    from tests.unit.test_offline_policy_training import trainer as policy_trainer

    trainer = policy_trainer()
    states = np.arange(10, dtype=np.float32).reshape(5, 2)
    data = {
        "states": states,
        "actions": states + 100,
        "next_states": states,
        "rewards": np.arange(5, dtype=np.float32),
        "dones": np.ones(5, dtype=np.float32),
    }
    dataset = SimpleNamespace(
        to_offline_rl_format=lambda: data,
        to_policy_training_format=lambda *args: [
            {"input_ids": [1, 2, 3], "labels": [-100, -100, 3]} for _ in states
        ],
    )
    seen = []

    original = trainer.train_step

    async def record(batch):
        seen.append(batch)
        return await original(batch)

    trainer.train_step = record
    await trainer.train(num_epochs=1, dataset=dataset)

    rows = np.concatenate([np.asarray(b["states"]) for b in seen])
    assert sorted(map(tuple, rows.tolist())) == sorted(map(tuple, states.tolist()))
    for batch in seen:
        np.testing.assert_array_equal(
            np.asarray(batch["actions"]), np.asarray(batch["states"]) + 100
        )
    assert trainer.optimizer_updates > 0
