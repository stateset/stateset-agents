"""Independent numerical checks for the real and finite-state RL contracts."""

from __future__ import annotations

import itertools
import math
import sys
from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import objectives, rl_losses  # noqa: E402
from stateset_agents.training.offline_rl_algorithms import (  # noqa: E402
    ImplicitQLearning,
    IQLConfig,
)
from stateset_agents.training.offline_rl_bear import (  # noqa: E402
    BEARConfig,
    ConversationalBEAR,
)
from stateset_agents.training.ppo_trainer import compute_gae  # noqa: E402
from stateset_agents.training.vapo_trainer import LengthAdaptiveGAE  # noqa: E402


@pytest.mark.parametrize(
    "gamma,lam", list(itertools.product((0.0, 0.5, 1.0), repeat=2))
)
def test_advantage_scan_matches_lambda_returns(gamma, lam, monkeypatch):
    """Check the TLC input grid against an independent lambda-return recurrence."""
    import numpy as np

    from stateset_agents.core import rust_accelerator
    from stateset_agents.data.conversation_dataset import TrajectoryData
    from stateset_agents.training import vapo_trainer

    grid = list(itertools.product((-1.0, 1.0), repeat=5))
    flags = list(itertools.product((False, True), repeat=2))
    rows = list(itertools.product(grid, flags))
    rewards = torch.tensor([x[:2] for x, _ in rows])
    values = torch.tensor([x[2:4] for x, _ in rows])
    dones = torch.tensor([d for _, d in rows])

    def targets(r, v, terminal, bootstrap):
        result = []
        future = next_v = bootstrap
        for reward, value, done in reversed(list(zip(r, v, terminal, strict=True))):
            future = (
                reward if done else reward + gamma * ((1 - lam) * next_v + lam * future)
            )
            result.append(future)
            next_v = value
        return result[::-1]

    expected = torch.tensor([targets(x[:2], x[2:4], d, 0) for x, d in rows])
    gae = LengthAdaptiveGAE(gamma)
    monkeypatch.setattr(vapo_trainer, "_rust_gae_available", lambda: False)
    adv, ret = gae.compute_gae(rewards, values, dones, lam)
    torch.testing.assert_close(ret, expected)
    torch.testing.assert_close(adv, expected - values)

    # Exercise the split-at-terminal adapter even on CPU installations using
    # the NumPy kernel fallback. The compiled Rust kernel has its own test.
    monkeypatch.setattr(vapo_trainer, "_rust_gae_available", lambda: True)
    fast, _ = gae.compute_gae(rewards, values, dones, lam)
    torch.testing.assert_close(fast, adv)

    ppo_adv, ppo_ret = compute_gae(rewards, values, gamma, lam)
    unbroken = torch.tensor(
        [targets(x[:2], x[2:4], (False, False), 0) for x, _ in rows]
    )
    torch.testing.assert_close(ppo_ret, unbroken)
    torch.testing.assert_close(ppo_adv, unbroken - values)
    for x in grid:
        target = targets(x[:2], x[2:4], (False, False), x[4])
        actual = rust_accelerator.compute_gae(
            np.array(x[:2]), np.array(x[2:]), gamma, lam
        )
        np.testing.assert_allclose(actual + x[2:4], target)
        if lam == 1:
            traj = TrajectoryData("test", [], sum(x[:2]), list(x[:2]))
            assert traj.compute_returns_to_go(gamma) == targets(
                x[:2], x[2:4], (False, False), 0
            )


@pytest.mark.parametrize("violation", [-0.125, 0.0, 0.125])
def test_bear_dual_gradient_and_first_adam_step(violation):
    config = BEARConfig(
        warmup_steps=0, lagrange_threshold=0.25, hidden_size=4, actor_hidden_size=4
    )
    trainer = ConversationalBEAR(2, 1, config, device="cpu")
    trainer._update_alpha(torch.tensor(0.25 + violation))
    assert trainer.log_alpha.grad.item() == pytest.approx(-violation)
    assert trainer.log_alpha.sign().item() == (
        1 if violation > 0 else -1 if violation < 0 else 0
    )


def test_iql_capped_weight_for_large_finite_advantage(monkeypatch):
    trainer = ImplicitQLearning(1, 1, IQLConfig(hidden_size=4), device="cpu")
    monkeypatch.setattr(trainer, "get_advantage", lambda *_: 10000.0)
    assert trainer.get_policy_weight(torch.zeros(1), torch.zeros(1)) == 100


def test_cql_soft_backup_stays_finite_under_large_common_offset():
    from stateset_agents.training.offline_rl_algorithms import (
        ConservativeQLearning,
        CQLConfig,
    )

    learner = ConservativeQLearning.__new__(ConservativeQLearning)
    learner.config = CQLConfig(temperature=0.0001)
    q = torch.full((2, 4, 1), 1e36, requires_grad=True)
    value = learner._soft_max(q, normalize=True)
    torch.testing.assert_close(value, q[:, 0])
    value.sum().backward()
    torch.testing.assert_close(q.grad, torch.full_like(q, 0.25))


@pytest.mark.parametrize("temperature", [1e-50, 0.8, 1.0, 1e16, 1e42])
def test_cql_normalized_backup_across_temperatures(temperature):
    import decimal

    from stateset_agents.training.offline_rl_algorithms import (
        ConservativeQLearning,
        CQLConfig,
    )

    learner = ConservativeQLearning.__new__(ConservativeQLearning)
    learner.config = CQLConfig(temperature=temperature)
    q = torch.tensor([[[1.0], [2.0], [1.0]]], requires_grad=True)
    value = learner._soft_max(q, normalize=True)
    (grad,) = torch.autograd.grad(value.sum(), q)
    # Decimal retains the difference from log N at high temperature.
    with decimal.localcontext() as ctx:
        ctx.prec = 90
        tau = decimal.Decimal.from_float(temperature)
        e = (-1 / tau).exp()
        z = 1 + 2 * e
        expected = 2 + tau * (z / 3).ln()
    assert value.item() == pytest.approx(float(expected), rel=2e-6)
    assert grad.flatten().tolist() == pytest.approx(
        [float(e / z), float(1 / z), float(e / z)]
    )


@pytest.mark.parametrize("normalize", [False, True])
def test_cql_subnormal_temperature_preserves_tied_critic_slopes(normalize):
    from stateset_agents.training.offline_rl_algorithms import (
        ConservativeQLearning,
        CQLConfig,
    )

    learner = ConservativeQLearning.__new__(ConservativeQLearning)
    learner.config = CQLConfig(temperature=math.ulp(0.0))
    q = torch.tensor([[[1.0], [1.0], [-1.0]]], dtype=torch.float64, requires_grad=True)
    tangent = torch.tensor([[[2.0], [4.0], [1.0]]], dtype=q.dtype)

    def evaluate(x):
        return learner._soft_max(x, normalize=normalize).sum()

    value, directional = torch.func.jvp(evaluate, (q,), (tangent,))
    (gradient,) = torch.autograd.grad(evaluate(q), q)
    assert value.item() == 1.0
    assert directional.item() == 3.0
    assert gradient.flatten().tolist() == [0.5, 0.5, 0.0]


@pytest.mark.parametrize("threshold,direction", [(0.8, 1), (1.6, -1)])
def test_cql_large_common_q_preserves_penalty_and_dual_direction(threshold, direction):
    from stateset_agents.training.offline_rl_algorithms import (
        ConservativeQLearning,
        CQLConfig,
    )

    config = CQLConfig(
        hidden_size=4,
        temperature=0.5,
        num_random_actions=4,
        with_lagrange=True,
        lagrange_threshold=threshold,
    )
    trainer = ConservativeQLearning(1, 1, config, device="cpu")
    for critic in (trainer.q1, trainer.q2):
        with torch.no_grad():
            for parameter in critic.parameters():
                parameter.zero_()
            list(critic.parameters())[-1].fill_(1e8)
    states = torch.zeros(2, 1)
    rewards = torch.full_like(states, 1e8)
    metrics = trainer.train_step(
        states, states, rewards, states, torch.ones_like(states)
    )
    # Each critic contributes temperature * log N. The threshold is shared.
    both = 2 * 0.5 * math.log(4)
    assert metrics["cql_loss"] == pytest.approx(both)
    assert trainer.log_alpha.grad.item() == pytest.approx(threshold - both)
    assert trainer.log_alpha.sign().item() == direction


def test_cql_per_state_penalty_gradients_match_softmax_and_logged_action():
    from stateset_agents.training.offline_rl_algorithms import (
        ConservativeQLearning,
        CQLConfig,
    )

    temp = 0.25
    profiles = [[[0, 1, -1], [2, 2, 2]], [[-1, 1, 1], [1, -1, 0]]]
    trainer = ConservativeQLearning(
        1,
        1,
        CQLConfig(hidden_size=4, temperature=temp, num_random_actions=3),
        device="cpu",
    )
    states = torch.tensor([[0.0], [1.0]])
    proposals = [
        torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in profiles
    ]
    logged = [
        torch.tensor([[3.0], [-2.0]], dtype=torch.float64, requires_grad=True)
        for _ in profiles
    ]

    def critic(table):
        def evaluate(s, actions):
            assert s[:, 0].tolist() == [0, 0, 0, 1, 1, 1]
            return table.reshape(-1, 1)

        return evaluate

    trainer.q1, trainer.q2 = map(critic, proposals)
    losses = trainer._compute_cql_loss(states, states, *logged)
    for loss, p, d, rows in zip(losses, proposals, logged, profiles, strict=True):
        dq, dd = torch.autograd.grad(loss, (p, d))
        expected_value = 0
        for index, row in enumerate(rows):
            exponentials = [math.exp(score / temp) for score in row]
            z = sum(exponentials)
            expected_value += (temp * math.log(z) - d[index].item()) / 2
            assert dq[index].tolist() == pytest.approx(
                [e / (2 * z) for e in exponentials]
            )
        assert loss.item() == pytest.approx(expected_value)
        assert dd[:, 0].tolist() == [-0.5, -0.5]


@pytest.mark.parametrize("tau", [0.0, 0.25, 0.7, 1.0])
def test_iql_expectile_gradient_at_and_across_target(tau):
    target = torch.tensor([-3.0, 1.0, 2.0], dtype=torch.float64)
    value = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float64, requires_grad=True)
    loss = ImplicitQLearning._expectile_loss(None, value, target, tau)
    (grad,) = torch.autograd.grad(loss, value)
    assert loss.item() == pytest.approx(((1 - tau) * 9 + tau * 16) / 3, rel=1e-13)
    assert grad.tolist() == pytest.approx([2 * (1 - tau), 0, -8 * tau / 3], rel=1e-13)


@pytest.mark.parametrize("targets", [[3.0], [1.0, 1.0], [-3.0, -1.0, 0.0, 5.0]])
@pytest.mark.parametrize("tau", [0.25, 0.5, 0.75])
def test_iql_empirical_expectile_fit(targets, tau):
    # Fix each possible partition of the ordered targets and solve the
    # resulting weighted-mean equation. Keep a self-consistent partition.
    for index in range(len(targets) + 1):
        weights = [1 - tau] * index + [tau] * (len(targets) - index)
        fit = math.fsum(y * w for y, w in zip(targets, weights, strict=True)) / sum(
            weights
        )
        if all(y <= fit for y in targets[:index]) and all(
            y >= fit for y in targets[index:]
        ):
            break
    else:
        pytest.fail("no self-consistent expectile partition")
    y = torch.tensor(targets, dtype=torch.float64)
    v = torch.tensor(fit, dtype=y.dtype, requires_grad=True)

    def loss(x):
        return ImplicitQLearning._expectile_loss(None, x.expand_as(y), y, tau)

    best = loss(v)
    (grad,) = torch.autograd.grad(best, v)
    assert grad.item() == pytest.approx(0.0, abs=1e-14)
    for delta in (-10.0, -0.25, 0.25, 10.0):
        gap = (loss(v.detach() + delta) - best.detach()).item()
        assert gap >= min(tau, 1 - tau) * delta**2 - 1e-13


@pytest.mark.parametrize(
    "dtype,tau,error",
    [
        (torch.float16, 0.5, 500),
        (torch.float64, 1e-48, 1),
        (torch.float32, 0.01, 1e20),
        (torch.float32, 0.5, 2e19),
        (torch.float32, 0.0, 1e20),
    ],
)
def test_iql_representable_weighted_mean_and_gradient(dtype, tau, error):
    values = torch.zeros(4, dtype=dtype, requires_grad=True)
    targets = torch.full_like(values, error)
    loss = ImplicitQLearning._expectile_loss(None, values, targets, tau)
    (grad,) = torch.autograd.grad(loss, values)
    residual = targets[0].item()
    assert loss.item() == pytest.approx(tau * residual**2, rel=2e-6, abs=0)
    assert grad.tolist() == pytest.approx([-tau * residual / 2] * 4, rel=2e-6, abs=0)


def test_iql_empty_expectile_batch_has_no_mean():
    with pytest.raises(ValueError, match="nonempty"):
        ImplicitQLearning._expectile_loss(None, torch.empty(0), torch.empty(0), 0.5)


@pytest.mark.parametrize("algorithm", ["cql", "iql", "bcq", "bear"])
@pytest.mark.parametrize(
    "size,capacity,column", [(3, 2, False), (1, 4, True), (4, 2, True)]
)
def test_offline_epoch_visits_every_transition_and_weights_observed_metrics(
    algorithm, size, capacity, column, monkeypatch
):
    import numpy as np

    from stateset_agents.training.offline_rl_algorithms import (
        CQLConfig,
        OfflineRLTrainer,
    )
    from stateset_agents.training.offline_rl_bcq import BCQConfig, BCQTrainer
    from stateset_agents.training.offline_rl_bear import BEARTrainer

    if algorithm == "bcq":
        trainer = BCQTrainer(
            1, 1, BCQConfig(hidden_size=4, vae_hidden_size=4), device="cpu"
        )
        learner = trainer.bcq
    elif algorithm == "bear":
        trainer = BEARTrainer(
            1, 1, BEARConfig(hidden_size=4, actor_hidden_size=4), device="cpu"
        )
        learner = trainer.bear
    else:
        config = (
            CQLConfig(hidden_size=4) if algorithm == "cql" else IQLConfig(hidden_size=4)
        )
        trainer = OfflineRLTrainer(algorithm, 1, 1, config, device="cpu")
        learner = trainer.learner
    ids = np.arange(size, dtype=np.float32).reshape(-1, 1)
    rewards = ids if column else ids[:, 0]
    data = {
        "states": ids,
        "actions": ids + 1,
        "next_states": ids + 2,
        "next_actions": ids + 3,
        "rewards": rewards,
        "dones": np.zeros_like(rewards),
    }
    visited = []
    observations = []
    step = learner.train_step

    def record(**batch):
        current = batch["states"]
        torch.testing.assert_close(batch["actions"], current + 1)
        torch.testing.assert_close(batch["next_states"], current + 2)
        torch.testing.assert_close(batch["rewards"], current)
        assert batch["dones"].shape == current.shape
        if algorithm == "cql":
            torch.testing.assert_close(batch["next_actions"], current + 3)
        visited.extend(current.flatten().tolist())
        metrics = step(**batch)
        observations.extend([metrics] * len(current))
        return metrics

    monkeypatch.setattr(learner, "train_step", record)
    (epoch,) = trainer.train(data, num_epochs=1, batch_size=capacity)
    assert sorted(visited) == list(range(size))
    assert learner.training_step == math.ceil(size / capacity)
    for key in observations[0]:
        if key == "training_step":
            assert epoch[key] == learner.training_step
            continue
        assert epoch[key] == pytest.approx(
            math.fsum(m[key] for m in observations) / size
        )


@pytest.mark.parametrize("error", ["empty", "batch_size", "reward_shape", "terminal"])
def test_offline_epoch_rejects_invalid_data_before_an_update(error):
    import numpy as np

    from stateset_agents.training.offline_rl_algorithms import OfflineRLTrainer

    trainer = OfflineRLTrainer("iql", 1, 1, IQLConfig(hidden_size=4), device="cpu")
    n = 0 if error == "empty" else 3
    data = {
        k: np.zeros((n, 1), np.float32)
        for k in ("states", "actions", "rewards", "next_states", "dones")
    }
    if error == "reward_shape":
        data["rewards"] = np.zeros((n, 2))
    if error == "terminal":
        data["dones"][-1] = -1
    with pytest.raises(ValueError):
        trainer.train(data, num_epochs=1, batch_size=0 if error == "batch_size" else 2)
    assert trainer.learner.training_step == 0


def test_offline_critic_configuration_checks_math_domains():
    from stateset_agents.training.offline_rl_algorithms import CQLConfig

    for kwargs in (
        {"temperature": 0.0},
        {"num_random_actions": 0},
        {"target_update_frequency": 0},
    ):
        with pytest.raises(ValueError):
            CQLConfig(**kwargs)
    for kwargs in (
        {"temperature": -1.0},
        {"expectile": 1.5},
        {"clip_score": float("nan")},
    ):
        with pytest.raises(ValueError):
            IQLConfig(**kwargs)


def test_k3_small_gap_and_padding_gradient():
    cur = torch.tensor([[-2.0001, -1000.0]], requires_grad=True)
    ref = torch.tensor([[-2.0, -0.01]])
    mask = torch.tensor([[1.0, 0.0]])
    gap = float((ref[0, 0] - cur[0, 0]).detach())
    result = objectives.policy_loss(
        objective=objectives.OBJECTIVES["grpo"].with_(kl_coef=1.0),
        mask=mask,
        logp_cur=cur,
        logp_ref=ref,
        advantages=torch.zeros(1),
    )
    assert result.loss.item() == pytest.approx(math.expm1(gap) - gap, rel=0.002)
    result.loss.backward()
    assert cur.grad[0, 1].item() == 0.0
    assert cur.grad[0, 0].item() == pytest.approx(-math.expm1(gap), rel=0.002)


def test_finite_low_precision_ratio_and_masked_mean():
    ratio = rl_losses.safe_exp_ratio(torch.tensor(40, dtype=torch.float16))
    assert ratio.item() == pytest.approx(math.exp(20))
    x = torch.ones(1, 70000, dtype=torch.float16)
    assert rl_losses.masked_mean(x, x) == 1.0


@pytest.mark.asyncio
async def test_dpo_step_scores_completion_tokens_and_updates_policy_only():
    from stateset_agents.core.enhanced.advanced_rl_algorithms import (
        DPOConfig,
        DPOTrainer,
    )

    class Tokenizer:
        bos_token_id = 0

        @staticmethod
        def encode(text, add_special_tokens):
            return [int(token) for token in text.split()]

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.zeros(5, 5))

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.table[input_ids])

    agent = SimpleNamespace(model=Policy(), tokenizer=Tokenizer())
    trainer = DPOTrainer(agent, DPOConfig(learning_rate=0.1, beta=0.5), device="cpu")
    pair = {"prompt": "1", "chosen": "2 2", "rejected": "3"}
    first = await trainer.train_step([pair])
    assert first["dpo_loss"] == pytest.approx(math.log(2))
    assert agent.model.table[1, 2] > agent.model.table[1, 3]
    assert agent.model.table[2, 2] > 0
    assert torch.count_nonzero(agent.model.table[[0, 3, 4]]) == 0
    assert torch.count_nonzero(trainer.reference_model.table) == 0
    assert trainer.reference_model.table.grad is None
    assert (await trainer.train_step([pair]))["dpo_loss"] < first["dpo_loss"]


def test_bellman_backups_keep_transitions_separate():
    from stateset_agents.training.offline_rl_algorithms import (
        ConservativeQLearning,
        CQLConfig,
    )

    class Critic(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, state, action):
            return state + self.offset

    config = CQLConfig(hidden_size=4, discount_factor=0.5)
    learner = ConservativeQLearning(1, 1, config, device="cpu")
    learner.q1_target, learner.q2_target = Critic(0), Critic(1)
    rewards = torch.tensor([1.0, -1.0])
    dones = torch.tensor([False, True])
    states = torch.tensor([[4.0], [100.0]])
    target = learner._bellman_target(rewards, states, dones, torch.zeros_like(states))
    torch.testing.assert_close(target, torch.tensor([[3.0], [-1.0]]))
    torch.testing.assert_close(target, learner._bellman_target(rewards, states, dones))


def test_gspo_accelerator_matches_length_and_ratio_bounds(monkeypatch):
    import numpy as np

    from stateset_agents.core import rust_accelerator

    new = np.array([-1.0, -1000.0, -2.0, -4.0])
    old = np.array([-1000.0, -1.0, -4.0, -999.0])
    lengths = np.array([1, 1, 2, 0])
    expected = np.exp([20.0, -20.0, 1.0, 0.0])
    for native in (rust_accelerator.is_rust_available(), False):
        monkeypatch.setattr(rust_accelerator, "_RUST_AVAILABLE", native)
        np.testing.assert_allclose(
            rust_accelerator.compute_gspo_importance_ratios(new, old, lengths), expected
        )


@pytest.mark.parametrize(
    "field", ["logp_old", "logp_ref", "advantages", "entropy", "group_ids"]
)
def test_objective_rejects_cross_row_broadcasts(field):
    kwargs = {
        "logp_cur": torch.zeros(2, 3),
        "mask": torch.ones(2, 3),
        "advantages": torch.ones(2),
        "objective": objectives.OBJECTIVES["grpo"].with_(kl_coef=0.1),
        "logp_ref": torch.zeros(2, 3),
    }
    kwargs[field] = (
        torch.zeros(1) if field in ("advantages", "group_ids") else torch.zeros(1, 3)
    )
    with pytest.raises(ValueError, match=field):
        objectives.policy_loss(**kwargs)


@pytest.mark.parametrize("aggregate", ["seq_mean", "seq_sum_const", "token_mean"])
def test_policy_reduction_boundary_cases(aggregate):
    """TLC's signed clipping branches and all 9 length pairs on a tensor batch."""
    obj = objectives.OBJECTIVES["grpo"].with_(
        clip_low=0.5, clip_high=0.5, aggregate=aggregate, max_completion_length=2
    )
    ratio = torch.tensor([[0.01, 2.0], [2.0, 0.01]], dtype=torch.float64)
    advantages = torch.tensor([-1.0, 1.0])
    for lengths in itertools.product(range(3), repeat=2):
        mask = torch.arange(2).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
        current = ratio.log().requires_grad_()
        result = objectives.policy_loss(
            objective=obj,
            logp_cur=current,
            logp_old=torch.zeros_like(current),
            advantages=advantages,
            mask=mask,
        )
        reference = 0.0
        denom = {"seq_mean": 2, "seq_sum_const": 4, "token_mean": max(1, sum(lengths))}[
            aggregate
        ]
        for row, n in enumerate(lengths):
            for t in range(n):
                r, adv = ratio[row, t].item(), advantages[row].item()
                effective = min(r, 1.5) if adv > 0 else max(r, 0.5)
                reference -= (
                    adv * effective / (n if aggregate == "seq_mean" else 1) / denom
                )
        assert result.loss.item() == pytest.approx(reference)
        result.loss.backward()
        assert torch.count_nonzero(current.grad[~mask]) == 0


def test_decision_transformer_actions_respect_causal_indices_and_key_mask():
    from stateset_agents.training.decision_transformer import (
        DecisionTransformer,
        DecisionTransformerConfig,
    )

    torch.manual_seed(17)
    config = DecisionTransformerConfig(
        state_dim=2,
        action_dim=2,
        n_head=2,
        n_embd=8,
        n_layer=2,
        max_context_length=4,
        dropout=0.0,
    )
    model = DecisionTransformer(config).eval()
    states, actions, returns = (
        torch.randn(1, 4, 2),
        torch.randn(1, 4, 2),
        torch.randn(1, 4, 1),
    )
    steps = torch.arange(4).unsqueeze(0)
    mask = torch.tensor([[0, 1, 1, 1]])
    baseline = model(states, actions, returns, steps, mask)[0][:, 2]
    # At t=2, a_2, a_3, future states/returns, and all masked keys are unavailable.
    actions[:, [0, 2, 3]] += 100
    states[:, [0, 3]] -= 100
    returns[:, [0, 3]] += 50
    changed = model(states, actions, returns, steps, mask)[0][:, 2]
    torch.testing.assert_close(changed, baseline)


@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm", ["ppo", "a2c"])
async def test_actor_critic_separates_truncated_rollout_from_reset(
    algorithm, monkeypatch
):
    import numpy as np

    from stateset_agents.core.enhanced import advanced_rl_algorithms as enhanced

    class Env:
        def __init__(self, initial, reward, done):
            self.initial, self.reward, self.done = initial, reward, done

        async def reset(self):
            return self.initial

        def observation_context(self, state):
            return state

        async def step(self, state, turn):
            return state + 1, self.reward, self.done, {}

    class Tokenizer:
        def __len__(self):
            return 2

        def decode(self, tokens):
            return "action"

    agent = SimpleNamespace(
        tokenizer=Tokenizer(),
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=1)),
    )
    gamma = 0.5
    if algorithm == "ppo":
        trainer = enhanced.PPOTrainer(
            agent, enhanced.PPOConfig(gamma=gamma, ppo_epochs=1), "cpu"
        )
        collect = trainer.collect_trajectory
    else:
        trainer = enhanced.A2CTrainer(agent, enhanced.A2CConfig(gamma=gamma), "cpu")
        collect = trainer.collect_experience

    # The critic returns the visible positive state as V(s).
    with torch.no_grad():
        for param in trainer.actor_critic.parameters():
            param.zero_()
        trainer.actor_critic.shared_layers[0].weight[0, 0] = 1
        trainer.actor_critic.shared_layers[2].weight[0, 0] = 1
        trainer.actor_critic.critic.weight[0, 0] = 1

    async def embed(messages):
        return np.array([float(messages[-1]["content"])], dtype=np.float32)

    monkeypatch.setattr(trainer, "_get_state_embedding", embed)
    await collect(Env(2, 1, False), 1)  # Time limit, bootstrap V(3).
    await collect(Env(10, -1, True), 1)  # Terminal, no bootstrap.
    assert trainer.bootstraps == {0: 3.0}
    observed = []
    original = enhanced.F.mse_loss

    def capture(values, targets):
        observed.append(targets.tolist())
        return original(values, targets)

    monkeypatch.setattr(enhanced.F, "mse_loss", capture)
    metrics = await trainer.train_step()
    assert all(math.isfinite(value) for value in metrics.values())
    assert observed == [[2.5, -1.0]]
    assert trainer.bootstraps == {}


def test_trajectory_kl_uses_the_logits_that_predict_the_label(monkeypatch):
    from stateset_agents.training import loss_computation as lc

    class Logits(torch.nn.Module):
        device = torch.device("cpu")

        def __init__(self, scores):
            super().__init__()
            self.scores = torch.nn.Parameter(scores)

        def forward(self, **kwargs):
            return SimpleNamespace(logits=self.scores, loss=self.scores.sum() * 0)

    policy = Logits(torch.tensor([[[2.0, -2.0], [0.0, 0.0], [0.0, 0.0]]]))
    ref = Logits(torch.zeros_like(policy.scores))
    labels = torch.tensor([[-100, 1, -100]])
    monkeypatch.setattr(lc, "_prepare_inputs_and_labels", lambda *_: ({}, labels))
    result = lc.compute_enhanced_grpo_loss(
        [
            SimpleNamespace(
                trajectories=[SimpleNamespace(total_reward=0.0, metadata={})]
            )
        ],
        1.0,
        SimpleNamespace(allow_sequence_fallback=True, clip_ratio=0.0),
        SimpleNamespace(model=policy),
        ref,
    )
    p = torch.softmax(policy.scores[0, 0].detach(), -1)
    expected = (p * (p.log() + math.log(2))).sum()
    torch.testing.assert_close(result["kl_penalty"], expected)
    result["total_loss"].backward()
    assert torch.count_nonzero(policy.scores.grad[0, 0]) == 2
    assert torch.count_nonzero(policy.scores.grad[0, 1:]) == 0


def test_gspo_saturated_log_ratio_keeps_zero_derivative():
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    trainer = object.__new__(GSPOTrainer)
    trainer.config = SimpleNamespace(beta=0)
    trainer._objective = objectives.OBJECTIVES["gspo"]
    current, old = torch.tensor([-1.0], requires_grad=True), torch.tensor([-1001.0])
    length = torch.ones(1)
    ratio = trainer.compute_sequence_importance_ratio(current, old, length)
    loss = trainer.compute_gspo_loss(ratio, -length, current, length, None, old)
    loss.backward()
    assert loss.detach().item() == pytest.approx(math.exp(20))
    assert current.grad.item() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("padding_side", ["left", "right"])
async def test_rlaif_scores_first_completion_action_and_ignores_padding(padding_side):
    from unittest.mock import AsyncMock

    from stateset_agents.rewards.llm_judge import JudgeConfig, LLMJudge
    from stateset_agents.training.rlaif_trainer import RLAIFTrainer
    from tests.unit.test_ppo_safety import CharacterTokenizer

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.zeros(32, 32))

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.table[input_ids])

    trainer = RLAIFTrainer.__new__(RLAIFTrainer)
    trainer.tokenizer = CharacterTokenizer()
    trainer.tokenizer.padding_side = padding_side
    trainer.config = SimpleNamespace(
        max_prompt_length=4,
        max_completion_length=2,
        learning_rate=0.25,
        max_grad_norm=10,
    )
    trainer.model, trainer.device = Policy(), torch.device("cpu")
    trainer.global_step, trainer.metrics_history = 0, {"policy_loss": [], "reward": []}
    trainer.cai = None
    trainer.generate_completion = lambda prompt: {"a": "x", "bbc": "y"}[prompt]
    trainer.judge = LLMJudge(JudgeConfig(provider="local"))
    trainer.judge.backend._call_api = AsyncMock(
        side_effect=['{"score": 1}', '{"score": 0.99999999}']
    )
    batch = await trainer.collect_trajectories(["a", "bbc"], num_generations=1)
    assert [row["reward"] for row in batch] == [1.0, 0.99999999]
    await trainer.train_step(batch)

    def token(char):
        return 1 + ord(char) % 31

    assert trainer.model.table[token("a"), token("x")] > 0
    assert trainer.model.table[token("c"), token("y")] < 0
    assert (
        torch.count_nonzero(
            trainer.model.table[[0, token("b"), token("x"), token("y")]]
        )
        == 0
    )


def test_all_masked_entropy_has_zero_value_and_gradient():
    from stateset_agents.training.loss_computation import compute_entropy_bonus

    logits = torch.zeros(1, 2, 3, requires_grad=True)
    entropy = compute_entropy_bonus(logits, torch.zeros(1, 2))
    entropy.backward()
    assert entropy.item() == 0 and torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_causal_scores_ignore_undefined_inactive_logits(dtype):
    logits = torch.tensor(
        [[[math.nan, 1.0], [-math.inf, -math.inf], [0.3, -0.4], [0.0, 0.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    ids = torch.tensor([[0, -100, 200, 1]])
    active = torch.tensor([[0, 0, 0, 1]])
    score, mask = rl_losses.gather_token_logprobs(logits, ids, active, dtype=dtype)
    expected = logits.detach()[0, 2].log_softmax(0)[1]
    torch.testing.assert_close(
        score.flatten(), torch.stack([expected * 0, expected * 0, expected])
    )
    torch.testing.assert_close(mask, active[:, 1:].float())
    score.sum().backward()
    gradient = torch.zeros_like(logits)
    gradient[0, 2] = -logits.detach()[0, 2].softmax(0)
    gradient[0, 2, 1] += 1
    torch.testing.assert_close(logits.grad, gradient)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rewards,responses",
    [
        ([1.0, 1.0 - 1e-8], ["xy", "z"]),
        ([1.0, math.nextafter(1.0, 0.0)], ["xy", "z"]),
        ([sys.float_info.max, -sys.float_info.max], ["xy", "z"]),
        ([sys.float_info.max, sys.float_info.max], ["xy", "z"]),
        ([0.125], ["z"]),
        ([0.125], ["xy"]),
    ],
)
async def test_rlaif_token_weights_match_scalar_sgd(rewards, responses):
    from stateset_agents.training.rlaif_trainer import RLAIFTrainer
    from tests.unit.test_ppo_safety import CharacterTokenizer

    class Table(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rows = torch.nn.Parameter(
                torch.arange(32, dtype=torch.float64).repeat(32, 1) / 32
            )

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.rows[input_ids])

    prompts = ["a", "bb"][: len(responses)]
    trainer = RLAIFTrainer.__new__(RLAIFTrainer)
    trainer.model, trainer.tokenizer = Table(), CharacterTokenizer()
    trainer.config = SimpleNamespace(
        max_prompt_length=4,
        max_completion_length=3,
        learning_rate=0.2,
        max_grad_norm=10,
    )
    trainer.device, trainer.global_step = torch.device("cpu"), 0
    trainer.metrics_history = {"policy_loss": [], "reward": []}

    def token(c):
        return 1 + ord(c) % 31

    original = trainer.model.rows.detach().clone()
    gradients = torch.zeros_like(original)
    terms = []
    scores = [
        Decimal.from_float(r)
        for r, text in zip(rewards, responses, strict=True)
        for _ in text
    ]
    with localcontext() as context:
        context.prec = 110
        if len(scores) > 1:
            shifted = [s - scores[0] for s in scores]
            mean = sum(shifted) / len(scores)
            centered = [s - mean for s in shifted]
            sd = (sum(s * s for s in centered) / (len(scores) - 1)).sqrt()
            weights = [float(s / (sd + Decimal("0.00000001"))) for s in centered]
        else:
            weights = [float(scores[0])]
    position = 0
    for prompt, response in zip(prompts, responses, strict=True):
        previous = token(prompt[-1])
        for char in response:
            action = token(char)
            weight = weights[position] / len(scores)
            terms.append(-weight * original[previous].log_softmax(0)[action].item())
            gradients[previous] += weight * original[previous].softmax(0)
            gradients[previous, action] -= weight
            previous, position = action, position + 1
    metrics = await trainer.train_step(
        [
            {"prompt": p, "completion": c, "reward": r}
            for p, c, r in zip(prompts, responses, rewards, strict=True)
        ]
    )
    torch.testing.assert_close(
        trainer.model.rows.grad, gradients, atol=1e-16, rtol=2e-12
    )
    torch.testing.assert_close(
        trainer.model.rows, original - 0.2 * gradients, atol=1e-16, rtol=2e-12
    )
    assert metrics["policy_loss"] == pytest.approx(
        math.fsum(terms), abs=2e-15, rel=2e-12
    )
    assert metrics["average_reward"] == math.fsum(r / len(rewards) for r in rewards)
    assert trainer.global_step == 1


@pytest.mark.parametrize("corrected", [False, True])
def test_sampled_k3_gradient_under_exact_categorical_enumeration(corrected):
    # One row per action with its behavior mass. This is an exact expectation.
    logits = torch.tensor([0.5, -0.5, 0.0], dtype=torch.float64, requires_grad=True)
    logp = logits.log_softmax(0)
    p = logp.detach().exp()
    q = torch.tensor([0.2, 0.3, 0.5], dtype=logits.dtype)
    behavior = torch.tensor([0.4, 0.3, 0.3], dtype=logits.dtype) if corrected else p
    obj = objectives.OBJECTIVES["grpo"].with_(kl_coef=1, kl_bias_correction=corrected)
    terms = [
        objectives.policy_loss(
            logp_cur=logp[i].reshape(1, 1),
            mask=logits.new_ones(1, 1),
            advantages=logits.new_zeros(1),
            objective=obj,
            logp_old=behavior[i].log().reshape(1, 1),
            logp_ref=q[i].log().reshape(1, 1),
        ).loss
        for i in range(3)
    ]
    expectation = behavior @ torch.stack(terms)
    kl = p @ (p.log() - q.log())
    torch.testing.assert_close(expectation.detach(), kl)
    expected_grad = p * (p.log() - q.log() - kl) if corrected else p - q
    torch.testing.assert_close(
        torch.autograd.grad(expectation, logits)[0], expected_grad
    )


@pytest.mark.parametrize("dtype,offset", [(torch.float32, 1e8), (torch.bfloat16, 1000)])
def test_logprob_gather_preserves_uniform_policy_after_common_shift(dtype, offset):
    ids = torch.zeros(1, 2, dtype=torch.long)
    logits = torch.full((1, 2, 4), offset, dtype=dtype, requires_grad=True)
    score, _ = rl_losses.gather_token_logprobs(
        logits, ids, torch.ones_like(ids), dtype=dtype
    )
    assert score.item() == pytest.approx(-math.log(4), abs=0.005)
    score.backward()
    torch.testing.assert_close(
        logits.grad[0, 0], logits.new_tensor([0.75, -0.25, -0.25, -0.25])
    )


@pytest.mark.parametrize("dones", [[0.5], [-1], [float("nan")]])
def test_bellman_rejects_nonbinary_terminal_flags(dones):
    with pytest.raises(ValueError, match="terminal flags"):
        rl_losses.bellman_target(torch.ones(1), torch.tensor(dones), torch.ones(1), 0.9)


@pytest.mark.parametrize(
    "kind", ["group_mean", "group_norm", "batch_norm", "leave_one_out"]
)
def test_invalid_reward_group_gives_finite_advantages(kind):
    result = objectives.compute_advantages(
        [1.0, float("nan"), -1.0, 1.0],
        [0, 0, 1, 1],
        objectives.OBJECTIVES["grpo"].with_(advantage=kind),
    )
    assert result.isfinite().all() and result[:2].count_nonzero() == 0
    if kind != "batch_norm":
        assert result[2] < 0 < result[3]


@pytest.mark.parametrize("kind", ["gaussian", "laplacian"])
def test_bear_kernel_matches_translated_unequal_sample_sets(kind):
    from stateset_agents.training.offline_rl_bear import MMDKernel

    kernel = MMDKernel(kind, sigma=2.0)
    x = torch.tensor([[[10000.0], [10001.0]]], requires_grad=True)
    y = torch.tensor([[[10002.0]]])
    values = getattr(kernel, f"{kind}_kernel")(x, x, 2.0)

    def k(distance):
        return math.exp(-(distance**2) / 8 if kind == "gaussian" else -distance / 2)

    torch.testing.assert_close(values, x.new_tensor([[[1, k(1)], [k(1), 1]]]))
    mmd = kernel.compute_mmd(x, y)
    assert mmd.item() == pytest.approx((1 + k(1)) / 2 + 1 - k(1) - k(2))
    torch.testing.assert_close(mmd, kernel.compute_mmd(x, y.expand(-1, 5, -1)))
    mmd.backward()
    assert torch.isfinite(x.grad).all() and x.grad.count_nonzero() == 2
    detached = x.detach().requires_grad_()
    at_match = kernel.compute_mmd(detached, detached.detach())
    at_match.backward()
    assert at_match.item() == 0
    torch.testing.assert_close(detached.grad, torch.zeros_like(detached))


def test_kl_schedules_at_endpoints_and_after_budget():
    from stateset_agents.training.kl_controllers import (
        HybridKLController,
        WarmupKLScheduler,
        create_kl_controller,
    )

    cosine = create_kl_controller(
        "cosine", init_kl_coef=1.0, min_kl_coef=0.2, total_steps=4
    )
    for step, expected in [(-1, 1.0), (0, 1.0), (2, 0.6), (4, 0.2), (5, 0.2)]:
        assert cosine.update(0.1, step) == pytest.approx(expected)
    hybrid = HybridKLController(
        init_kl_coef=1.0, min_kl_coef=0.2, target_kl=0.1, total_steps=4
    )
    hybrid.update(0.4, 1)
    assert hybrid.update(0.1, 2) == pytest.approx(0.6)
    warm = WarmupKLScheduler(
        init_kl_coef=0.2,
        peak_kl_coef=1,
        final_kl_coef=0.4,
        warmup_steps=2,
        total_steps=4,
    )
    assert [warm.update(0, step) for step in range(6)] == pytest.approx(
        [0.2, 0.6, 1.0, 0.7, 0.4, 0.4]
    )
    for warmup, expected in [(0, [1, 0.4, 0.4]), (4, [0.2, 1, 1])]:
        scheduler = WarmupKLScheduler(0.2, 1.0, 0.4, warmup, 4)
        assert [scheduler.update(0, t) for t in (0, 4, 5)] == pytest.approx(expected)


@pytest.mark.asyncio
async def test_exploration_entropy_conditions_on_available_actions(monkeypatch):
    from stateset_agents.experimental import adaptive_learning_controller as adaptive

    controller = adaptive.ExplorationController(
        strategy=adaptive.ExplorationStrategy.INFORMATION_GAIN
    )
    controller.action_counts.update({"a": 1, "b": 3, "old": 100})
    threshold = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75)) / math.log(2)
    for offset in (-1e-6, 1e-6):
        monkeypatch.setattr(
            adaptive.random, "random", lambda value=threshold + offset: value
        )
        assert await controller.should_explore(None, ["a", "b"]) == (offset < 0)
    assert not await controller.should_explore(None, ["a"])
    assert not await controller.should_explore(None, [])
    assert await controller.should_explore(None, ["untried", "new"])


@pytest.mark.parametrize("backend", ["neural", "transformer"])
def test_reward_regression_uses_examples_in_partial_batches(backend):
    from stateset_agents.training.neural_reward_trainer import NeuralRewardTrainer
    from stateset_agents.training.transformer_reward_model import (
        TransformerRewardTrainer,
    )

    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def forward(self, *args, **kwargs):
            batch = args[0] if args else kwargs["prompt_input_ids"]
            return self.bias.expand(batch.shape[0], 1)

    trainer_type = (
        NeuralRewardTrainer if backend == "neural" else TransformerRewardTrainer
    )
    trainer = trainer_type.__new__(trainer_type)
    trainer.device, trainer.model = torch.device("cpu"), Constant()
    trainer.criterion = torch.nn.MSELoss()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda _: 1.0
    )
    trainer.config = SimpleNamespace(max_grad_norm=10.0)
    batches = []
    for rewards in ([1.0, 1.0], [4.0]):
        x = torch.ones(len(rewards), 1)
        batches.append(
            {
                "prompt_embedding": x,
                "response_embedding": x,
                "prompt_input_ids": x,
                "response_input_ids": x,
                "prompt_attention_mask": x,
                "response_attention_mask": x,
                "reward": torch.tensor(rewards),
            }
        )
    assert trainer._train_epoch(batches) == pytest.approx(6.0)
    assert trainer._validate_epoch(batches) == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_gepo_uses_filtered_sampler_scores_in_its_group_denominator(monkeypatch):
    from tests.unit.test_gepo_trainer_behavioral import _tiny_gepo_trainer

    trainer = _tiny_gepo_trainer()

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, *args, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.ones(1, 2),
            }

        def decode(self, tokens, **kwargs):
            return "ok" if tokens[0].item() == 4 else "nope"

    trainer.tokenizer = Tokenizer()
    actions = iter((4, 5))

    def generate(**kwargs):
        assert kwargs["return_dict_in_generate"] and kwargs["output_scores"]
        scores = torch.full((1, 200), -torch.inf)
        scores[0, 4:6] = torch.tensor([0.25, 0.75]).log()
        return SimpleNamespace(
            sequences=torch.tensor([[1, 2, next(actions)]]), scores=(scores,)
        )

    monkeypatch.setattr(trainer.model, "generate", generate)
    from tests.unit.test_gepo_trainer_behavioral import _record_policy_scores

    observed = _record_policy_scores(monkeypatch)
    metrics = await trainer.train_step(["prompt"])
    assert math.isfinite(metrics["policy_loss"])
    sampler = observed[0][1]
    torch.testing.assert_close(sampler.exp(), sampler.new_tensor([0.25, 0.75]))


@pytest.mark.asyncio
async def test_gepo_sequence_override_weights_every_response_token(monkeypatch):
    from tests.unit.test_gepo_trainer_behavioral import _tiny_gepo_trainer

    trainer = _tiny_gepo_trainer()
    trainer._objective = objectives.OBJECTIVES["gspo"].with_(
        clip_low=0.5, clip_high=0.5, aggregate="token_mean"
    )

    async def generate(prompt, group_size):
        lengths = [1, 3] if prompt == "a" else [2, 2]
        rows = []
        for i, length in enumerate(lengths):
            ids = torch.tensor([1] + [2 + i] * length)
            attn = torch.ones_like(ids)
            with torch.no_grad():
                _, logp = trainer.compute_sequence_log_probs(ids[None], attn[None], 1)
            ratio = 1.0 + (0.1 if i == 0 else -0.1)
            rows.append(
                {
                    "input_ids": ids,
                    "attention_mask": attn,
                    "response_start_idx": 1,
                    "response": "ok" if i == 0 else "nope",
                    "sampler_log_prob": logp.item() - length * math.log(ratio),
                }
            )
        return rows

    monkeypatch.setattr(trainer, "generate_group_responses", generate)
    metrics = await trainer.train_step(["a", "b"])
    assert metrics["policy_loss"] == pytest.approx((-3 * 1.1 + 5 * 0.9) / 8, abs=1e-6)


@pytest.mark.asyncio
@pytest.mark.parametrize("estimator", ["external", "k3_token", "k3_sequence"])
async def test_gepo_regularization_matches_frozen_reference_distribution(
    estimator, monkeypatch
):
    from dataclasses import replace

    from stateset_agents.training.gepo_trainer import GEPOTrainer
    from tests.unit.test_gepo_trainer_behavioral import (
        _fixed_group_responses,
        _tiny_gepo_trainer,
    )

    base = _tiny_gepo_trainer()
    cfg = replace(
        base.config,
        beta=0.3,
        objective_overrides={"kl": estimator, "entropy_coef": 0.02},
    )
    trainer = GEPOTrainer(cfg, base.model, None, lambda *_: 1.0)
    assert trainer.ref_model is not trainer.model
    assert not any(p.requires_grad for p in trainer.ref_model.parameters())
    with torch.no_grad():
        for p in trainer.ref_model.parameters():
            p.zero_()
    rows = _fixed_group_responses()

    async def generate(*_):
        return rows

    monkeypatch.setattr(trainer, "generate_group_responses", generate)
    ids = torch.stack([row["input_ids"] for row in rows])
    with torch.no_grad():
        cur = trainer.model(input_ids=ids).logits[:, 3:-1].double().log_softmax(-1)
        log_q = -math.log(cur.shape[-1])
        entropy = -(cur.exp() * cur).sum(-1).mean()
        if estimator == "external":
            kl = (cur.exp() * (cur - log_q)).sum(-1).mean()
        else:
            gap = log_q - cur.gather(-1, ids[:, 4:, None]).squeeze(-1)
            if estimator == "k3_sequence":
                gap = gap.mean(-1)
            kl = (gap.exp() - gap - 1).mean()
    metrics = await trainer.train_step(["p"])
    assert metrics["policy_loss"] == pytest.approx(
        (0.3 * kl - 0.02 * entropy).item(), abs=1e-6
    )


@pytest.mark.parametrize("optimizer_kind", ["Adam", "AdamW"])
def test_adam_moments_bias_correction_and_parameter_updates(optimizer_kind):
    lr, decay, b1, b2, eps = 0.03, 0.1, 0.8, 0.6, 0.01
    expected = 1.25
    parameter = torch.nn.Parameter(torch.tensor(expected, dtype=torch.float64))
    opt = getattr(torch.optim, optimizer_kind)(
        [parameter], lr=lr, weight_decay=decay, betas=(b1, b2), eps=eps
    )
    history = []
    for t, gradient in enumerate([0.5, -0.25, 0.0, 1.0], start=1):
        # Direct geometric weights form an independent reference for both EMAs.
        history.append(
            gradient + decay * expected if optimizer_kind == "Adam" else gradient
        )
        m = (1 - b1) * sum(b1**i * g for i, g in enumerate(reversed(history)))
        v = (1 - b2) * sum(b2**i * g * g for i, g in enumerate(reversed(history)))
        if optimizer_kind == "AdamW":
            expected *= 1 - lr * decay
        expected -= lr * (m / (1 - b1**t)) / (math.sqrt(v / (1 - b2**t)) + eps)
        parameter.grad = parameter.new_tensor(gradient)
        opt.step()
        assert parameter.item() == pytest.approx(expected, rel=1e-12)
        assert opt.state[parameter]["exp_avg"].item() == pytest.approx(m, rel=1e-12)
        assert opt.state[parameter]["exp_avg_sq"].item() == pytest.approx(v, rel=1e-12)


@pytest.mark.parametrize("reduction", ["seq_mean", "seq_sum_const", "token_mean"])
def test_empty_policy_population_has_zero_loss_and_empty_gradient(reduction):
    obj = objectives.OBJECTIVES["gspo"].with_(
        kl_coef=0.1, aggregate=reduction, max_completion_length=2
    )
    scores = torch.empty(0, 2, requires_grad=True)
    result = objectives.policy_loss(
        objective=obj,
        logp_cur=scores,
        logp_ref=scores.detach(),
        advantages=torch.empty(0),
        mask=torch.zeros_like(scores),
    )
    assert result.loss.item() == 0
    result.loss.backward()
    assert scores.grad.shape == (0, 2)


@pytest.mark.parametrize("kind", ["group_mean", "leave_one_out"])
def test_identical_nonbinary_rewards_have_exactly_zero_advantages(kind):
    r = torch.full((13,), 0.1)
    result = objectives.compute_advantages(
        r, torch.zeros_like(r), objectives.OBJECTIVES["grpo"].with_(advantage=kind)
    )
    assert result.count_nonzero() == 0


@pytest.mark.parametrize("width", [1, 3, 100, 107])
def test_reward_calibrator_refreshes_the_retained_window(width):
    from statistics import fmean, pstdev

    from stateset_agents.training.reward_calibration import RewardNormalizer

    normalizer = RewardNormalizer(buffer_size=width)
    values = [float((i * 7) % 13 + i // 50) for i in range(403)]
    for seen, value in enumerate(values, start=1):
        normalizer.add_reward(value)
        if width <= 100 or seen <= 100 or seen % 100 == 0:
            window = values[max(seen - width, 0) : seen]
            assert normalizer.stats.count == len(window)
            assert normalizer.stats.mean == pytest.approx(fmean(window))
            assert normalizer.stats.std == pytest.approx(pstdev(window))
            assert (normalizer.stats.min, normalizer.stats.max) == (
                min(window),
                max(window),
            )
    with pytest.raises(ValueError, match="finite"):
        normalizer.add_reward(float("nan"))
    assert list(normalizer.reward_buffer) == values[-width:]
    assert normalizer.get_statistics().mean == pytest.approx(fmean(values[-width:]))


@pytest.mark.asyncio
async def test_curiosity_probability_tracks_state_visits_and_action_error(monkeypatch):
    from stateset_agents.experimental import adaptive_learning_controller as adaptive

    c = adaptive.ExplorationController()
    monkeypatch.setattr(adaptive.random, "random", lambda: 0.4)
    assert await c.should_explore("x", ["a", "b"])
    await c.update_exploration_state("a", 2.0, "x")
    assert not await c.should_explore("x", ["a", "b"])
    assert await c.should_explore("y", ["a", "b"])
    await c.update_exploration_state("a", -2.0, "y")
    assert c.curiosity_scores["a"] == pytest.approx(0.4)
    assert not await c.should_explore("y", ["a", "a"])
    with pytest.raises(ValueError, match="finite"):
        await c.update_exploration_state("a", float("inf"), "x")
    assert c.state_visits["x"] == 1
    assert c.action_counts["a"] == 2


def test_truncated_policy_k3_has_the_missing_reference_mass_term():
    p = torch.tensor([0.25, 0.75], dtype=torch.float64)
    q = torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64)
    gap = q[:2].log() - p.log()
    sampled = (p * rl_losses.k3_term(gap)).sum()
    forward_kl = (p * -gap).sum()
    torch.testing.assert_close(sampled, forward_kl - q[2])
    # Even exact importance weights cannot recover an unobserved reward.
    reward = torch.tensor([2.0, -1.0, 4.0], dtype=torch.float64)
    recovered = (p * (q[:2] / p) * reward[:2]).sum()
    torch.testing.assert_close(recovered + q[2] * reward[2], q @ reward)


def test_dapo_default_filter_matches_binary_reward_variance():
    from stateset_agents.training.dapo_trainer import DynamicSamplingBuffer

    buffer = DynamicSamplingBuffer()
    for n in range(1, 6):
        for bits in itertools.product((0.0, 1.0), repeat=n):
            rewards = torch.tensor(bits, dtype=torch.float64)
            accuracy = sum(bits) / n
            assert rewards.var(correction=0).item() == pytest.approx(
                accuracy * (1 - accuracy)
            )
            adv = rl_losses.group_advantages(rewards)
            assert buffer.should_include(accuracy) == bool(adv.count_nonzero())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalize", [False, True])
def test_cql_largest_temperature_values_and_both_derivatives(dtype, normalize):
    from types import SimpleNamespace

    from stateset_agents.training.offline_rl_algorithms import ConservativeQLearning

    m = torch.finfo(dtype).max
    trainer = SimpleNamespace(config=SimpleNamespace(temperature=m))
    q = torch.tensor([[[-m], [m]], [[-1.0], [1.0]], [[-m], [-m]]], dtype=dtype)
    logged = torch.tensor([[m], [0], [m]], dtype=dtype)
    slopes = torch.tensor(
        [[[3.0], [-1.0]], [[2.0], [4.0]], [[-2.0], [2.0]]], dtype=dtype
    )
    data_slope = torch.tensor([[1.0], [2.0], [4.0]], dtype=dtype)

    def evaluate(q, logged):
        return ConservativeQLearning._soft_max(
            trainer, q, normalize=normalize, baseline=None if normalize else logged
        )

    result, directional = torch.func.jvp(evaluate, (q, logged), (slopes, data_slope))
    high = 1 / (1 + math.exp(-2))
    p = torch.tensor([1 - high, high, 0.5, 0.5, 0.5, 0.5], dtype=dtype).reshape_as(q)
    expected_directional = (p * slopes).sum(1) - (0 if normalize else data_slope)
    torch.testing.assert_close(directional, expected_directional)
    grads = torch.func.grad(lambda x, b: evaluate(x, b).sum(), argnums=(0, 1))(
        q, logged
    )
    torch.testing.assert_close(grads[0], p)
    torch.testing.assert_close(
        grads[1], torch.zeros_like(logged) if normalize else -torch.ones_like(logged)
    )

    if normalize:
        assert result[0].item() / m == pytest.approx(math.log(math.cosh(1)), rel=2e-7)
        # tau*log(cosh(1/tau)) = 1/(2*tau) to the destination precision.
        assert result[1].item() == torch.tensor(0.5 / m, dtype=dtype).item()
        assert result[2].item() == -m
    else:
        assert result[0].item() / m == pytest.approx(math.log1p(math.exp(-2)), rel=2e-7)
        assert result[1].item() / m == pytest.approx(math.log(2), rel=2e-7)
        # Even after the log-partition correction, the true result is below -max.
        assert result[2].item() == -math.inf


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tau", [1.0, 1e20])
def test_cql_small_signed_scores_preserve_the_even_term(dtype, tau):
    from types import SimpleNamespace

    from stateset_agents.training.offline_rl_algorithms import ConservativeQLearning

    # The symmetric row's true value is x^2/(2*tau) to destination precision.
    x = 1e-18 if tau == 1 else 1.0
    trainer = SimpleNamespace(config=SimpleNamespace(temperature=tau))
    q = torch.tensor([[[-x], [x]], [[2.0], [2.0]]], dtype=dtype, requires_grad=True)
    values = ConservativeQLearning._soft_max(trainer, q, normalize=True)
    rounded_x = q[0, 1, 0].item()
    expected = torch.tensor([rounded_x**2 / (2 * tau), 2], dtype=dtype)
    torch.testing.assert_close(values.flatten(), expected, rtol=1e-6, atol=0)
    (g,) = torch.autograd.grad(values.sum(), q)
    torch.testing.assert_close(g, torch.full_like(q, 0.5))


def test_cql_extreme_temperature_through_sampled_bellman_backup(monkeypatch):
    from decimal import Inexact, localcontext

    from stateset_agents.training.offline_rl_algorithms import (
        ConservativeQLearning,
        CQLConfig,
    )

    scale = torch.finfo(torch.float32).max
    model = ConservativeQLearning(
        1,
        1,
        CQLConfig(
            hidden_size=4, num_random_actions=2, temperature=scale, discount_factor=0.5
        ),
        "cpu",
    )

    def evaluate(states, actions):
        # Each successor gets two opposite critic scores. The terminal state's
        # scores are undefined and must disappear from the Bellman result.
        return states * torch.tensor([[-scale], [scale]]).repeat(len(states) // 2, 1)

    monkeypatch.setattr(model.q1_target, "forward", evaluate)
    monkeypatch.setattr(model.q2_target, "forward", evaluate)
    s = torch.tensor([[1.0], [math.nan], [0.0]])
    rewards = torch.tensor([0.0, -3.0, 2.0])
    with localcontext() as caller:
        caller.traps[Inexact] = True
        target = model._bellman_target(rewards, s, torch.tensor([0, 1, 0]))
    assert target[0].item() / scale == pytest.approx(
        0.5 * math.log(math.cosh(1)), rel=2e-7
    )
    assert target[1:].flatten().tolist() == [-3.0, 2.0]
