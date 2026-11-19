"""Compare the sampled CPU critic update and checkpoint with scalar references."""

import math
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from formal.critic_certificate import audit_critic_snapshot
from stateset_agents.training import offline_grpo_trainer as offline
from stateset_agents.training.offline_rl_batches import (
    CriticBatchSnapshot,
    CriticObservation,
)

STATES = [(0.0,), (1.0,)]
DATA = SimpleNamespace(
    to_offline_rl_format=lambda: {
        "states": [[0], [0], [1], [1]],
        "actions": [[-1], [0.25], [0.5], [1]],
        "rewards": [2, -1, -3, 2],
        "next_states": [[1], [0], [0], [1]],
        "dones": [0, 1, 0, 1],
    }
)


def make_trainer(**overrides):
    config = {
        "state_dim": 1,
        "action_dim": 1,
        "value_num_layers": 1,
        "value_hidden_size": 3,
        "value_learning_rate": 0.125,
        "max_grad_norm": 0.25,
        "gamma": 0.5,
        "pretrain_value_batch_size": 4,
        "seed": 193,
    }
    return offline.OfflineGRPOTrainer(
        offline.OfflineGRPOConfig(**(config | overrides)), device="cpu"
    )


def params(network):
    return [x for p in network.parameters() for x in p.detach().flatten().tolist()]


def linear(weights, features):
    return sum(w * x for w, x in zip(weights, (*features, 1), strict=True))


def check_head(network, optimizer, before, rows, jacobians, predicted, history):
    # Differentiate the mean squared error for every weight and the bias
    # directly. Each head is clipped with its own norm before AdamW sees it.
    raw = [
        sum(
            2 * (p - row.target) * j[i]
            for j, p, row in zip(jacobians, predicted, rows, strict=True)
        )
        / len(rows)
        for i in range(len(before))
    ]
    norm = math.hypot(*raw)
    cap = 0.25
    assert norm > cap
    clipped = [g * cap / (norm + 1e-6) for g in raw]
    actual_grads = [x for p in network.parameters() for x in p.grad.flatten().tolist()]
    assert actual_grads == pytest.approx(clipped, abs=3e-8)
    history.append(clipped)
    group = optimizer.param_groups[0]
    b1, b2 = group["betas"]
    step = len(history)
    slots = [optimizer.state[p] for p in network.parameters()]
    assert all(slot["step"].item() == step for slot in slots)
    first = [x for slot in slots for x in slot["exp_avg"].flatten().tolist()]
    second = [x for slot in slots for x in slot["exp_avg_sq"].flatten().tolist()]
    for i, (old, new) in enumerate(zip(before, params(network), strict=True)):
        m = (1 - b1) * sum(b1**t * g[i] for t, g in enumerate(reversed(history)))
        v = (1 - b2) * sum(b2**t * g[i] ** 2 for t, g in enumerate(reversed(history)))
        assert first[i] == pytest.approx(m, rel=4e-6, abs=3e-9)
        assert second[i] == pytest.approx(v, rel=4e-6, abs=1e-12)
        expected = (1 - group["lr"] * group["weight_decay"]) * old - group["lr"] * (
            m / (1 - b1**step)
        ) / (math.sqrt(v / (1 - b2**step)) + group["eps"])
        assert new == pytest.approx(expected, abs=2e-7)


@pytest.mark.asyncio
async def test_captured_auxiliary_updates_match_fixed_point_gradients_and_adamw(
    monkeypatch, tmp_path
):
    original = make_trainer()
    with torch.no_grad():
        for net, weights in (
            (original.value_net, [0.75, -0.5]),
            (original.value_target, [-1.5, 1.0]),
            (original.q_net, [0.5, -0.75, 1.0]),
        ):
            weight, bias = net.parameters()
            weight.copy_(torch.tensor(weights[:-1]).reshape_as(weight))
            bias.fill_(weights[-1])
    inputs = []
    regression = offline.td_regression

    def record(prediction, target):
        inputs.append(
            (prediction.detach().flatten().tolist(), target.flatten().tolist())
        )
        return regression(prediction, target)

    monkeypatch.setattr(offline, "td_regression", record)
    value_history, q_history, batches = [], [], []
    path = str(tmp_path / "after-step-one.pt")
    for step in range(2):
        pre, qpre, frozen = map(
            params, (original.value_net, original.q_net, original.value_target)
        )
        result = await original.pretrain_value_functions(
            DATA, 1, on_critic_batch=batches.append
        )
        snapshot = batches[-1]
        rows = snapshot.rows
        (vp, vy), (qp, qy) = inputs[-2:]
        assert vp == [r.prediction for r in rows] and vy == qy == [
            r.target for r in rows
        ]
        assert sorted(
            (r.state[0], r.action[0], r.reward, r.successor) for r in rows
        ) == [
            (0, -1, 2, (1,)),
            (0, 0.25, -1, None),
            (1, 0.5, -3, (0,)),
            (1, 1, 2, None),
        ]
        for row, actual_q in zip(rows, qp, strict=True):
            assert row.prediction == pytest.approx(linear(pre, row.state), abs=1e-7)
            assert actual_q == pytest.approx(
                linear(qpre, row.state + row.action), abs=1e-7
            )
            assert row.frozen_prediction == float(np.float32(linear(frozen, row.state)))
            future = (
                0
                if row.successor is None
                else float(np.float32(linear(frozen, row.successor)))
            )
            expected = Fraction(row.reward) + Fraction(snapshot.discount) * Fraction(
                future
            )
            assert row.target == float(np.float32(float(expected)))
        if step:
            stale = {(r.state, r.action): r.target for r in batches[0].rows}
            assert any(
                r.target != stale[r.state, r.action]
                for r in rows
                if r.successor is not None
            )
        audit = audit_critic_snapshot(snapshot, STATES)
        assert result["final_loss"] == pytest.approx(float(audit.loss), rel=2e-15)
        # V0 = 1/2 + V1/4 and V1 = -1/2 + V0/4 for these four recorded rows.
        truth = (Fraction(2, 5), Fraction(-2, 5))
        error = max(abs(Fraction(r.prediction) - truth[int(r.state[0])]) for r in rows)
        budget = next(
            Fraction(i, 100) for i in range(2000) if audit.certifies(Fraction(i, 100))
        )
        assert error <= budget
        check_head(
            original.value_net,
            original.value_optimizer,
            pre,
            rows,
            [(*r.state, 1) for r in rows],
            vp,
            value_history,
        )
        check_head(
            original.q_net,
            original.q_optimizer,
            qpre,
            rows,
            [(*r.state, *r.action, 1) for r in rows],
            qp,
            q_history,
        )
        tau = Fraction(0.005)
        expected_target = [
            float(np.float32(float((1 - tau) * Fraction(old) + tau * Fraction(new))))
            for old, new in zip(frozen, params(original.value_net), strict=True)
        ]
        assert params(original.value_target) == expected_target
        new = params(original.value_net)
        movement = max(
            sum(
                abs(Fraction(a) - Fraction(b)) * abs(Fraction(x))
                for a, b, x in zip(new, pre, (*s, 1), strict=True)
            )
            for s in STATES
        )
        after = max(
            abs(linear(list(map(Fraction, new)), tuple(map(Fraction, s))) - v)
            for s, v in zip(STATES, truth, strict=True)
        )
        assert after <= budget + movement
        if step == 0:
            assert error == Fraction(9, 10) and budget == Fraction(185, 100)
            original.save(path)
    resumed = make_trainer(gamma=0.75, value_num_layers=2, value_activation="tanh")
    resumed.load(path)
    replay = []
    assert resumed._offline_learner.training_step == 1
    assert (
        await resumed.pretrain_value_functions(DATA, 1, on_critic_batch=replay.append)
        == result
    )
    assert replay == [batches[1]]
    for name in (
        "value_net",
        "q_net",
        "value_target",
        "value_optimizer",
        "q_optimizer",
    ):
        torch.testing.assert_close(
            getattr(resumed, name).state_dict(),
            getattr(original, name).state_dict(),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        resumed._iql_state(), original._iql_state(), rtol=0, atol=0
    )
    assert resumed._rng.bit_generator.state == original._rng.bit_generator.state
    assert (
        resumed._offline_learner.training_step
        == original._offline_learner.training_step
        == 2
    )


@pytest.mark.asyncio
async def test_two_layer_relu_critics_match_scalar_jacobians_and_resume(tmp_path):
    trainer = make_trainer(value_num_layers=2, value_hidden_size=2)
    with torch.no_grad():
        for net, first in (
            (trainer.value_net, [[0.5], [-1.0]]),
            (trainer.q_net, [[0.75, -0.5], [-0.25, 0.875]]),
        ):
            a, b = net.network[0], net.network[-1]
            a.weight.copy_(torch.tensor(first))
            a.bias.copy_(torch.tensor([0.625, 0.75]))
            b.weight.copy_(torch.tensor([[-0.5, 0.75]]))
            b.bias.fill_(-0.25)
        trainer.value_target.load_state_dict(trainer.value_net.state_dict())
        trainer.value_target.network[-1].bias.fill_(0.875)
    path = str(tmp_path / "relu.pt")
    histories = ([], [])
    crossed = False
    for step in range(2):
        heads = (trainer.value_net, trainer.q_net)
        previous = [params(net) for net in heads]
        batches = []
        result = await trainer.pretrain_value_functions(
            DATA, 1, on_critic_batch=batches.append
        )
        rows = batches[0].rows
        for head, (net, old, opt) in enumerate(
            zip(
                heads,
                previous,
                (trainer.value_optimizer, trainer.q_optimizer),
                strict=True,
            )
        ):
            width = 1 + head
            new = params(net)
            weights = [old[:width], old[width : 2 * width]]
            biases, readout = old[2 * width : 2 * width + 2], old[-3:-1]
            jacobians, predicted = [], []
            for row in rows:
                x = row.state if head == 0 else row.state + row.action
                hidden = [
                    linear([*w, b], x) for w, b in zip(weights, biases, strict=True)
                ]
                active = [max(z, 0) for z in hidden]
                slopes = [
                    a if z > 0 else 0 for a, z in zip(readout, hidden, strict=True)
                ]
                jacobians.append(
                    [*(g * t for g in slopes for t in x), *slopes, *active, 1]
                )
                predicted.append(linear([*readout, old[-1]], active))
                new_hidden = [
                    linear([*new[i * width : (i + 1) * width], new[2 * width + i]], x)
                    for i in range(2)
                ]
                new_active = [max(z, 0) for z in new_hidden]
                bound = abs(new[-1] - old[-1]) + sum(
                    abs(w2 - w1) * max(z2, 0) + abs(w1) * abs(z2 - z1)
                    for w2, w1, z2, z1 in zip(
                        new[-3:-1], readout, new_hidden, hidden, strict=True
                    )
                )
                crossed |= any(
                    (a > 0) != (b > 0) for a, b in zip(hidden, new_hidden, strict=True)
                )
                assert (
                    abs(linear(new[-3:], new_active) - predicted[-1]) <= bound + 1e-14
                )
            check_head(net, opt, old, rows, jacobians, predicted, histories[head])
        if step == 0:
            trainer.save(path)
    assert crossed
    restored = make_trainer(
        value_num_layers=2, value_hidden_size=2, value_activation="tanh"
    )
    restored.load(path)
    replay = []
    assert (
        await restored.pretrain_value_functions(DATA, 1, on_critic_batch=replay.append)
        == result
    )
    assert replay == batches
    for name in (
        "value_net",
        "q_net",
        "value_optimizer",
        "q_optimizer",
        "value_target",
    ):
        torch.testing.assert_close(
            getattr(trainer, name).state_dict(),
            getattr(restored, name).state_dict(),
            rtol=0,
            atol=0,
        )


def test_capture_rejects_incomplete_or_inconsistent_state_tables():
    row = CriticObservation((0,), (0,), None, 0, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="Every declared"):
        audit_critic_snapshot(CriticBatchSnapshot(0.5, (row,)), STATES)
    with pytest.raises(ValueError, match="inconsistent"):
        audit_critic_snapshot(
            CriticBatchSnapshot(0.5, (row, replace(row, frozen_prediction=2))), [(0.0,)]
        )


def test_snapshot_preserves_mixed_reward_bits_and_arrives_after_publication():
    trainer = make_trainer(gamma=1)
    with torch.no_grad():
        trainer.value_target.network[0].weight.zero_()
        trainer.value_target.network[0].bias.fill_(-(2**63))
    before = params(trainer.value_target)
    observed = []

    def observe(snapshot):
        observed.append((snapshot, params(trainer.value_target)))

    states = torch.tensor([[0.0], [1.0]])
    trainer._train_value_step(
        states,
        states,
        np.array([2**63 + 1, -0.5], dtype=object),
        torch.tensor([[1.0], [math.nan]]),
        torch.tensor([False, True]),
        on_critic_batch=observe,
    )
    ((snapshot, published),) = observed
    assert published == params(trainer.value_target) and published != before
    assert [r.reward for r in snapshot.rows] == [2**63 + 1, -0.5]
    assert [r.target for r in snapshot.rows] == [1, -0.5]
    assert [r.successor for r in snapshot.rows] == [(1.0,), None]
    assert audit_critic_snapshot(snapshot, STATES).mean_target_bias_bound == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("with_policy", [True, False])
async def test_pretraining_checkpoint_without_policy_optimizer_recovers_legacy_clock(
    tmp_path, with_policy
):
    model = make_trainer()
    if with_policy:
        model.model = torch.nn.Linear(2, 1)
    await model.pretrain_value_functions(DATA, 2)
    path = tmp_path / "old.pt"
    model.save(str(path))
    saved = torch.load(path, weights_only=True)
    del saved["iql_training_step"]
    torch.save(saved, path)
    reference = await model.pretrain_value_functions(DATA, 1)
    restored = make_trainer(gamma=0.75)
    if with_policy:
        restored.model = torch.nn.Linear(2, 1)
    restored.load(str(path))
    assert (
        restored._offline_learner.training_step == 2
        and restored.policy_optimizer is None
    )
    assert await restored.pretrain_value_functions(DATA, 1) == reference
    torch.testing.assert_close(
        restored._iql_state(), model._iql_state(), rtol=0, atol=0
    )
    if with_policy:
        torch.testing.assert_close(
            restored.model.state_dict(), model.model.state_dict(), rtol=0, atol=0
        )
