"""Independent scalar references for sampled, committed CPU critic updates."""

import math
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from formal.critic_certificate import (
    BlockCertificate,
    CriticRow,
    audit_critic_snapshot,
    audit_logged_critic,
)
from stateset_agents.training import offline_rl_algorithms as offline
from stateset_agents.training.offline_grpo_trainer import (
    OfflineGRPOConfig,
    OfflineGRPOTrainer,
)
from stateset_agents.training.offline_rl_batches import (
    CriticBatchSnapshot,
    CriticObservation,
)
from stateset_agents.training.state_fingerprint import state_fingerprint


def test_unit_discount_chain_branch_and_closed_class():
    chain = [CriticRow(0, 1, 2, 2), CriticRow(1, 2, 3, 3), CriticRow(2, None, 1, 1)]
    zeros = [0] * 3
    early = audit_logged_critic(zeros, zeros, chain, 1, horizon=2)
    assert early.block == BlockCertificate(2, 1, 5)
    assert not early.certifies(100)
    exact = audit_logged_critic(zeros, zeros, chain, 1)
    assert exact.contraction_factor == 1 and exact.block == BlockCertificate(3, 0, 6)
    assert exact.certifies(6) and not exact.certifies(Fraction(599, 100))
    rounded = BlockCertificate(1, Fraction(9, 10), Fraction(0.1))
    assert not rounded.certifies(1.0)

    # The exact branching solution is (6,5). Both two-step survival rates are 1/2.
    branch = [CriticRow(0, 1, 1, 1), CriticRow(1, None, 2, 2), CriticRow(1, 0, 2, 2)]
    observed = CriticBatchSnapshot(
        1,
        tuple(
            CriticObservation(
                (r.state,),
                (0,),
                None if r.successor is None else (r.successor,),
                r.reward,
                0,
                0,
                r.target,
            )
            for r in branch
        ),
    )
    actual = audit_critic_snapshot(observed, [(0,), (1,)], horizon=2)
    assert actual.block == BlockCertificate(2, Fraction(1, 2), 3)
    assert actual.certifies(6) and not actual.certifies(Fraction(599, 100))
    assert audit_logged_critic([6, 5], [6, 5], branch, 1).certifies(0)

    closed = [CriticRow(0, 0, 0, 0), CriticRow(1, None, 0, 0)]
    for horizon in (None, 1, 5):
        outcome = audit_logged_critic([0, 0], [0, 0], closed, 1, horizon=horizon)
        assert outcome.block == BlockCertificate(horizon or 1, 1, 0)
        assert not outcome.certifies(100)
    for horizon in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            audit_logged_critic(zeros, zeros, chain, 1, horizon=horizon)


STATES = [0.0, 1.0, 2.0]
ACTIONS = [0.0, 1.0, -1.0]
REWARDS = [2.0, -1.0, 1.0]
SUCCESSORS = [1.0, 0.0, 2.0]
DONES = [0.0, 1.0, 0.0]
Q_FEATURES = list(zip(STATES, ACTIONS, [1.0] * 3, strict=True))
V_FEATURES = [(s, 1.0) for s in STATES]


def batch():
    return tuple(
        torch.tensor(x).reshape(-1, 1)
        for x in (STATES, ACTIONS, REWARDS, SUCCESSORS, DONES)
    )


def flatten(tensors):
    return [x for tensor in tensors for x in tensor.detach().flatten().tolist()]


def weights(network):
    return flatten(network.parameters())


def set_weights(network, scalars):
    with torch.no_grad():
        weight, bias = network.parameters()
        weight.copy_(weight.new_tensor(scalars[:-1]).reshape_as(weight))
        bias.fill_(scalars[-1])


def linear(coefficients, features):
    return sum(w * x for w, x in zip(coefficients, features, strict=True))


def gradient(features, slopes):
    return [
        linear(column, slopes) / len(slopes) for column in zip(*features, strict=True)
    ]


def mean(values):
    return sum(values) / len(values)


def squared_loss(predictions, labels):
    return mean([(p - y) ** 2 for p, y in zip(predictions, labels, strict=True)])


def assert_adam(optimizer, before, raw, history, cap=math.inf):
    """Compare the actual clipped gradients, Adam slots, counters and every weight."""
    scale = min(1, cap / (math.hypot(*raw) + 1e-6))
    clipped = [scale * g for g in raw]
    history.append(clipped)
    step = len(history)
    (group,) = optimizer.param_groups
    parameters = group["params"]
    assert flatten(p.grad for p in parameters) == pytest.approx(clipped, abs=2e-7)
    slots = [optimizer.state[p] for p in parameters]
    assert all(s["step"].item() == step for s in slots)
    b1, b2 = group["betas"]
    moments = [
        sum((1 - b1) * b1**k * g for k, g in enumerate(reversed(gs)))
        for gs in zip(*history, strict=True)
    ]
    squares = [
        sum((1 - b2) * b2**k * g * g for k, g in enumerate(reversed(gs)))
        for gs in zip(*history, strict=True)
    ]
    assert flatten(s["exp_avg"] for s in slots) == pytest.approx(
        moments, rel=2e-6, abs=2e-8
    )
    assert flatten(s["exp_avg_sq"] for s in slots) == pytest.approx(
        squares, rel=5e-6, abs=1e-11
    )
    expected = [
        old
        - group["lr"]
        * (m / (1 - b1**step))
        / (math.sqrt(v / (1 - b2**step)) + group["eps"])
        for old, m, v in zip(before, moments, squares, strict=True)
    ]
    assert flatten(parameters) == pytest.approx(expected, rel=2e-6, abs=1.5e-7)


def record_targets(monkeypatch):
    recorded = []
    original = offline.td_regression

    def record(predicted, target):
        assert not target.requires_grad
        recorded.append(target.flatten().tolist())
        return original(predicted, target)

    monkeypatch.setattr(offline, "td_regression", record)
    return recorded


def test_iql_complete_expectile_and_twin_q_update(monkeypatch):
    config = offline.IQLConfig(
        num_layers=1,
        discount_factor=0.5,
        expectile=0.75,
        learning_rate=0.0625,
        soft_target_tau=0.25,
        max_grad_norm=0.125,
    )
    learner = offline.ImplicitQLearning(1, 1, config, "cpu")
    set_weights(learner.q1, [1, 1, 0])
    set_weights(learner.q2, [-1, 1, 1])
    set_weights(learner.value_net, [0, 0])
    set_weights(learner.value_target, [0.75, -0.25])
    recorded = record_targets(monkeypatch)
    value_history, q_history = [], []
    for step in (1, 2):
        q1, q2, v, frozen = map(
            weights, (learner.q1, learner.q2, learner.value_net, learner.value_target)
        )
        pred1, pred2 = ([linear(q, f) for f in Q_FEATURES] for q in (q1, q2))
        values = [linear(v, f) for f in V_FEATURES]
        diffs = [
            min(p, q) - value for p, q, value in zip(pred1, pred2, values, strict=True)
        ]
        if step == 1:
            assert diffs == [0, 1, -2]
        scales = [config.expectile if d > 0 else 1 - config.expectile for d in diffs]
        value_gradient = gradient(
            V_FEATURES, [-2 * a * d for a, d in zip(scales, diffs, strict=True)]
        )
        labels = [
            r if done else r + config.discount_factor * linear(frozen, (s, 1))
            for r, s, done in zip(REWARDS, SUCCESSORS, DONES, strict=True)
        ]
        q_gradient = [
            g
            for preds in (pred1, pred2)
            for g in gradient(
                Q_FEATURES, [2 * (p - y) for p, y in zip(preds, labels, strict=True)]
            )
        ]
        metrics = learner.train_step(*batch())
        assert recorded[-1] == recorded[-2] == pytest.approx(labels, abs=2e-7)
        assert metrics["value_loss"] == pytest.approx(
            mean([a * d * d for a, d in zip(scales, diffs, strict=True)])
        )
        assert metrics["q_loss"] == pytest.approx(
            squared_loss(pred1, labels) + squared_loss(pred2, labels)
        )
        assert_adam(
            learner.value_optimizer,
            v,
            value_gradient,
            value_history,
            config.max_grad_norm,
        )
        assert_adam(
            learner.q_optimizer, q1 + q2, q_gradient, q_history, config.max_grad_norm
        )
        mixed = [
            (1 - config.soft_target_tau) * old + config.soft_target_tau * new
            for old, new in zip(frozen, weights(learner.value_net), strict=True)
        ]
        assert weights(learner.value_target) == pytest.approx(mixed, abs=4e-8)
        assert all(p.grad is None for p in learner.value_target.parameters())
        assert learner.training_step == step


@contextmanager
def capture_calls(learner):
    """Observe the actual (state, action) inputs used by the four critic networks."""
    seen = defaultdict(list)

    def observe(name):
        def record(module, inputs):
            states, actions = inputs
            seen[name].append((states.flatten().tolist(), actions.flatten().tolist()))

        return record

    with ExitStack() as stack:
        for name in ("q1", "q2", "q1_target", "q2_target"):
            handle = getattr(learner, name).register_forward_pre_hook(observe(name))
            stack.callback(handle.remove)
        yield seen


def soft_partition(values, temperature):
    top = max(values)
    exponentials = [math.exp((q - top) / temperature) for q in values]
    total = sum(exponentials)
    return top + temperature * math.log(total), [e / total for e in exponentials]


def make_cql(with_lagrange):
    config = offline.CQLConfig(
        num_layers=1,
        num_random_actions=4,
        discount_factor=0.5,
        learning_rate=0.03125,
        soft_target_tau=0.25,
        temperature=0.625,
        target_update_frequency=2,
        cql_alpha=0.75,
        lagrange_threshold=0.5,
        with_lagrange=with_lagrange,
        max_grad_norm=0.25,
    )
    trainer = offline.OfflineRLTrainer("cql", 1, 1, config, "cpu")
    for name, values in (
        ("q1", [0.5, -0.75, 0.25]),
        ("q2", [-0.5, 1.0, -0.125]),
        ("q1_target", [0.25, 0.5, 0.125]),
        ("q2_target", [-0.25, -0.75, 0.5]),
    ):
        set_weights(getattr(trainer.learner, name), values)
    return trainer


def proposal_rows(call, state_ids, count):
    states, actions = call
    assert states == [s for s in state_ids for _ in range(count)]
    assert len(actions) == len(states) and all(-1 <= a <= 1 for a in actions)
    return [actions[i : i + count] for i in range(0, len(actions), count)]


@pytest.mark.parametrize("with_lagrange", [False, True])
@pytest.mark.parametrize("logged_next", [False, True])
def test_cql_sampled_penalty_gradient_and_committed_step(
    monkeypatch, with_lagrange, logged_next
):
    torch.manual_seed(372)
    learner = make_cql(with_lagrange).learner
    config = learner.config
    n, tau = config.num_random_actions, config.temperature
    recorded = record_targets(monkeypatch)
    penalties = []
    original = learner._compute_cql_loss

    def observed_loss(*args):
        losses = original(*args)
        penalties.append(
            [
                (
                    loss.item(),
                    flatten(
                        torch.autograd.grad(
                            loss, list(net.parameters()), retain_graph=True
                        )
                    ),
                )
                for loss, net in zip(losses, (learner.q1, learner.q2), strict=True)
            ]
        )
        return losses

    monkeypatch.setattr(learner, "_compute_cql_loss", observed_loss)
    histories = defaultdict(list)
    for step in (1, 2, 3):
        live = [weights(net) for net in (learner.q1, learner.q2)]
        frozen = [weights(net) for net in (learner.q1_target, learner.q2_target)]
        old_log = learner.log_alpha.item() if with_lagrange else None
        alpha = math.exp(old_log) if with_lagrange else config.cql_alpha
        next_actions = torch.tensor([[-0.5], [0.75], [0.25]]) if logged_next else None
        stream = torch.Generator().set_state(torch.get_rng_state())
        expected_successors = (
            [[-0.5], [0.25]]
            if logged_next
            else torch.empty(2, n).uniform_(-1, 1, generator=stream).tolist()
        )
        expected_sources = (
            torch.empty(len(STATES), n).uniform_(-1, 1, generator=stream).tolist()
        )
        with capture_calls(learner) as seen:
            metrics = learner.train_step(*batch(), next_actions=next_actions)
        assert seen["q1"] == seen["q2"] and len(seen["q1"]) == 2
        assert seen["q1"][0] == (STATES, ACTIONS)
        bank = proposal_rows(seen["q1"][1], STATES, n)
        assert seen["q1_target"] == seen["q2_target"] and len(seen["q1_target"]) == 1
        next_bank = proposal_rows(
            seen["q1_target"][0], [1.0, 2.0], 1 if logged_next else n
        )
        assert bank == expected_sources and next_bank == expected_successors
        assert torch.equal(torch.get_rng_state(), stream.get_state())
        labels = REWARDS.copy()
        for i, proposals in zip((0, 2), next_bank, strict=True):
            qs = [
                min(linear(q, (SUCCESSORS[i], a, 1)) for q in frozen) for a in proposals
            ]
            soft, _ = soft_partition(qs, tau)
            labels[i] += config.discount_factor * (soft - tau * math.log(len(qs)))
        assert recorded[-1] == recorded[-2] == pytest.approx(labels, abs=3e-7)

        gaps, td_losses = [], []
        for h, q in enumerate(live):
            predictions = [linear(q, f) for f in Q_FEATURES]
            row_gaps, expected_actions = [], []
            for s, proposals, logged in zip(STATES, bank, predictions, strict=True):
                partition, probabilities = soft_partition(
                    [linear(q, (s, a, 1)) for a in proposals], tau
                )
                row_gaps.append(partition - logged)
                expected_actions.append(linear(probabilities, proposals))
            gap = mean(row_gaps)
            # State and bias sensitivities cancel between proposals and the logged action.
            penalty_grad = [
                0.0,
                mean([v - a for v, a in zip(expected_actions, ACTIONS, strict=True)]),
                0.0,
            ]
            loss, actual_gradient = penalties[-1][h]
            assert loss == pytest.approx(gap, abs=2e-7)
            assert actual_gradient == pytest.approx(
                penalty_grad, abs=3 * torch.finfo(torch.float32).eps
            )
            td_grad = gradient(
                Q_FEATURES,
                [2 * (p - y) for p, y in zip(predictions, labels, strict=True)],
            )
            full_grad = [
                g + alpha * d for g, d in zip(td_grad, penalty_grad, strict=True)
            ]
            optimizer = (learner.q1_optimizer, learner.q2_optimizer)[h]
            assert_adam(optimizer, q, full_grad, histories[h], config.max_grad_norm)
            gaps.append(gap)
            td_losses.append(squared_loss(predictions, labels))
            net = (learner.q1, learner.q2)[h]
            target = (learner.q1_target, learner.q2_target)[h]
            mix = (
                config.soft_target_tau
                if (step - 1) % config.target_update_frequency == 0
                else 0.0
            )
            expected = [
                (1 - mix) * a + mix * b
                for a, b in zip(frozen[h], weights(net), strict=True)
            ]
            assert weights(target) == pytest.approx(expected, abs=5e-8)
            assert all(p.grad is None for p in target.parameters())
        assert metrics["cql_loss"] == pytest.approx(sum(gaps))
        assert metrics["bellman_loss"] == pytest.approx(mean(td_losses))
        assert metrics["total_loss"] == pytest.approx(
            mean(td_losses) + alpha * mean(gaps)
        )
        if with_lagrange:
            assert_adam(
                learner.alpha_optimizer,
                [old_log],
                [-alpha * (sum(gaps) - config.lagrange_threshold)],
                histories["alpha"],
            )
        assert learner.training_step == step


def dataset():
    return {
        k: v.numpy()
        for k, v in zip(
            ("states", "actions", "rewards", "next_states", "dones"),
            batch(),
            strict=True,
        )
    }


def assert_same_learner(first, second):
    assert first.training_step == second.training_step
    assert first.config == second.config
    for name, component in vars(first).items():
        if isinstance(component, (torch.nn.Module, torch.optim.Optimizer)):
            assert state_fingerprint(component.state_dict()) == state_fingerprint(
                getattr(second, name).state_dict()
            )
    if getattr(first.config, "with_lagrange", False):
        assert state_fingerprint(first.log_alpha) == state_fingerprint(second.log_alpha)
        assert second.alpha_optimizer.param_groups[0]["params"][0] is second.log_alpha


@pytest.mark.parametrize(
    "device",
    ["cpu"]
    + (["cuda"] if torch.cuda.is_available() else [])
    + (["mps"] if torch.backends.mps.is_available() else []),
)
def test_fixed_cql_replays_both_proposal_streams_on_available_devices(tmp_path, device):
    config = offline.CQLConfig(num_layers=1, num_random_actions=3)
    trainer = offline.OfflineRLTrainer("cql", 1, 1, config, device)
    tensors = [t.to(device) for t in batch()]
    trainer.learner.train_step(*tensors)
    path = str(tmp_path / "fixed-cql.pt")
    trainer.save(path)
    with capture_calls(trainer.learner) as expected:
        trainer.learner.train_step(*tensors)
    restored = offline.OfflineRLTrainer("cql", 1, 1, config, device)
    restored.load(path)
    with capture_calls(restored.learner) as replay:
        restored.learner.train_step(*tensors)
    assert replay == expected
    assert_same_learner(trainer.learner, restored.learner)


@pytest.mark.parametrize("algorithm", ["cql", "iql"])
def test_standalone_checkpoint_replays_shuffled_epoch_and_parameters(
    tmp_path, algorithm
):
    torch.manual_seed(287)
    trainer = (
        make_cql(True)
        if algorithm == "cql"
        else offline.OfflineRLTrainer(
            "iql", 1, 1, offline.IQLConfig(num_layers=1, expectile=0.875), "cpu"
        )
    )
    data = dataset()
    trainer.train(data, num_epochs=1, batch_size=3)
    path = str(tmp_path / "critic.pt")
    trainer.save(path)

    # Two minibatches cross both a skipped and a scheduled CQL target update.
    def advance(run):
        with ExitStack() as stack:
            inputs = (
                stack.enter_context(capture_calls(run.learner))
                if algorithm == "cql"
                else None
            )
            metrics = run.train(data, num_epochs=1, batch_size=2)
            return inputs, metrics

    expected_calls, expected_metrics = advance(trainer)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(1007)
    other = offline.OfflineRLTrainer(
        algorithm,
        1,
        1,
        (
            offline.CQLConfig(num_layers=2, hidden_size=3)
            if algorithm == "cql"
            else offline.IQLConfig(num_layers=2, hidden_size=3)
        ),
        "cpu",
    )
    other.load(path)
    assert other.learner.training_step == 1
    calls, metrics = advance(other)
    if algorithm == "cql":
        assert calls == expected_calls and len(calls["q1"]) == 4
    assert metrics == expected_metrics and len(metrics) == 2
    assert_same_learner(trainer.learner, other.learner)
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.asyncio
async def test_outer_cql_replays_sampler_critics_and_target_schedule(tmp_path):
    torch.manual_seed(148)

    def outer(seed):
        return OfflineGRPOTrainer(
            OfflineGRPOConfig(
                offline_algorithm="cql",
                seed=seed,
                state_dim=1,
                action_dim=1,
                value_num_layers=1,
                gamma=0.5,
                pretrain_value_batch_size=2,
            ),
            device="cpu",
        )

    trainer = outer(137)
    data = SimpleNamespace(to_offline_rl_format=dataset)
    await trainer.pretrain_value_functions(data, 1)
    path = str(tmp_path / "pretrained.pt")
    trainer.save(path)

    async def advance(run):
        snapshots = []
        with capture_calls(run._offline_learner) as calls:
            result = await run.pretrain_value_functions(
                data, 2, on_critic_batch=snapshots.append
            )
        return result, snapshots, calls

    expected = await advance(trainer)
    other = outer(53)
    other.load(path)
    assert other.value_pretrained and other._offline_learner.training_step == 1
    assert await advance(other) == expected
    assert_same_learner(trainer._offline_learner, other._offline_learner)
    assert other._rng.bit_generator.state == trainer._rng.bit_generator.state
    for name in (
        "value_net",
        "value_target",
        "q_net",
        "value_optimizer",
        "q_optimizer",
    ):
        torch.testing.assert_close(
            getattr(other, name).state_dict(),
            getattr(trainer, name).state_dict(),
            rtol=0,
            atol=0,
        )
    saved = torch.load(path, weights_only=True)
    del saved["cql_state"]
    torch.save(saved, path)
    other.load(path)
    assert other._offline_learner is None and not other.value_pretrained
    await other.pretrain_value_functions(data, 1)
    assert other._offline_learner.training_step == 1


def test_legacy_standalone_weights_start_fresh_and_rejected_load_preserves_learner(
    tmp_path,
):
    original = make_cql(True)
    original.learner.train_step(*batch())
    path = str(tmp_path / "old.pt")
    original.save(path)
    saved = torch.load(path, weights_only=True)
    legacy = {
        key: saved[key] for key in ("algorithm", "learner_state", "training_metrics")
    }
    restored = make_cql(True)
    restored.learner.train_step(*batch())
    old = restored.learner
    del saved["optimizers"]["alpha_optimizer"]
    torch.save(saved, path)
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match="optimizers"):
        restored.load(path)
    assert restored.learner is old and torch.equal(torch.get_rng_state(), rng)
    torch.save(legacy, path)
    restored.load(path)
    assert torch.equal(torch.get_rng_state(), rng)
    assert (
        restored.learner.training_step == 0 and restored.learner.log_alpha.item() == 0
    )
    for name in ("q1", "q2", "q1_target", "q2_target"):
        assert weights(getattr(restored.learner, name)) == weights(
            getattr(original.learner, name)
        )
    for name in ("q1_optimizer", "q2_optimizer", "alpha_optimizer"):
        assert not getattr(restored.learner, name).state
