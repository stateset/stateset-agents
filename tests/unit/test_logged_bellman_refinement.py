"""Public offline trainer paths and FP backups against tabular Bellman oracles."""

import itertools
import math
import random
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from formal.critic_certificate import CriticRow, audit_logged_critic  # noqa: E402
from stateset_agents.training.offline_grpo_trainer import (  # noqa: E402
    OfflineGRPOConfig,
    OfflineGRPOTrainer,
)
from stateset_agents.training.offline_rl_algorithms import (  # noqa: E402
    ConservativeQLearning,
    CQLConfig,
    ImplicitQLearning,
    IQLConfig,
)
from stateset_agents.training.offline_rl_batches import (  # noqa: E402
    transition_epochs,
    transition_terminals,
)
from stateset_agents.training.rl_losses import (  # noqa: E402
    bellman_target,
    td_regression,
)


def trainer(kind="iql", gamma=0.5):
    return OfflineGRPOTrainer(
        OfflineGRPOConfig(
            offline_algorithm=kind,
            gamma=gamma,
            state_dim=1,
            action_dim=1,
            value_hidden_size=2,
            value_num_layers=1,
            pretrain_value_batch_size=4,
        ),
        device="cpu",
    )


@pytest.mark.parametrize("kind", ["cql", "iql", "bcq", "bear"])
def test_configured_discount_reaches_the_actual_learner(kind):
    t = trainer(kind, 0.125)
    t._init_offline_learner()
    assert t._offline_learner.config.discount_factor == 0.125


@pytest.mark.parametrize("gamma", [0.0, 0.25, 1.0])
def test_outer_value_update_uses_configured_discount_and_only_live_rows(gamma):
    t = trainer(gamma=gamma)
    with torch.no_grad():
        t.value_net.network[0].weight.zero_()
        t.value_net.network[0].bias.zero_()
        t.value_target.network[0].weight.fill_(1)
        t.value_target.network[0].bias.zero_()
    calls = []
    t.value_target.register_forward_pre_hook(
        lambda _, args: calls.extend(args[0].flatten().tolist())
    )
    states = torch.zeros(3, 1)
    loss = t._train_value_step(
        states,
        states,
        torch.tensor([1.0, 2.0, -1.0]),
        torch.tensor([[4.0], [math.nan], [-2.0]]),
        torch.tensor([0, 1, 0]),
    )
    assert loss == pytest.approx(((1 + 4 * gamma) ** 2 + 4 + (-1 - 2 * gamma) ** 2) / 3)
    assert calls == ([4.0, -2.0] if gamma else [])


@pytest.mark.parametrize("gamma,done", [(0.0, False), (0.5, True)])
def test_cql_inactive_successors_consume_no_draws(gamma, done):
    q = ConservativeQLearning(
        1, 1, CQLConfig(hidden_size=2, discount_factor=gamma), "cpu"
    )

    def fail(*args):
        pytest.fail("inactive successor reached a target critic")

    q.q1_target.register_forward_pre_hook(fail)
    state = torch.random.get_rng_state()
    backed = q._bellman_target(
        torch.tensor([3.0]), torch.tensor([[math.nan]]), torch.tensor([done])
    )
    assert backed.tolist() == [[3.0]]
    assert torch.equal(state, torch.random.get_rng_state())


def test_cql_inserting_terminal_rows_preserves_live_proposals_and_rng():
    q = ConservativeQLearning(
        1, 1, CQLConfig(hidden_size=2, num_random_actions=5), "cpu"
    )
    x = torch.tensor([[2.0], [3.0]])
    start = torch.random.get_rng_state()
    expected = q._bellman_target(torch.zeros(2), x, torch.zeros(2))
    after = torch.random.get_rng_state()
    torch.random.set_rng_state(start)
    full = q._bellman_target(
        torch.tensor([0.0, 7.0, 0.0]),
        torch.tensor([[2.0], [math.nan], [3.0]]),
        torch.tensor([False, True, False]),
    )
    assert torch.equal(full[[0, 2]], expected)
    assert full[1].item() == 7
    assert torch.equal(after, torch.random.get_rng_state())


def test_iql_validates_backup_before_any_optimizer_update():
    q = ImplicitQLearning(1, 1, IQLConfig(hidden_size=2), "cpu")
    x = torch.ones(2, 1)
    before = [p.clone() for p in q.value_net.parameters()]
    with pytest.raises(ValueError, match="binary"):
        q.train_step(x, x, x, x, torch.tensor([0.2, 1.0]))
    assert not q.value_optimizer.state and not q.q_optimizer.state
    assert all(
        torch.equal(a, b) for a, b in zip(before, q.value_net.parameters(), strict=True)
    )


@pytest.mark.parametrize(
    "dtype,bits",
    [(torch.float16, 7), (torch.bfloat16, 5), (torch.float32, 14), (torch.float64, 27)],
)
def test_backup_survives_cancellation_against_exact_rational(dtype, bits):
    rewards = torch.tensor([-1.0, 1.0, 7.0], dtype=dtype, requires_grad=True)
    values = torch.tensor(
        [1 + 2.0**-bits, -(1 + 2.0**-bits), math.nan], dtype=dtype, requires_grad=True
    )
    gamma = 1 - 2.0**-bits
    target = bellman_target(rewards, torch.tensor([0, 0, 1]), values, gamma)
    expected = [
        float(Fraction(r) + Fraction(gamma) * Fraction(v))
        for r, v in zip(rewards.tolist()[:2], values.tolist()[:2], strict=True)
    ] + [7.0]
    assert target.dtype == dtype
    assert target.flatten().tolist() == expected
    target.sum().backward()
    assert rewards.grad.tolist() == [1.0, 1.0, 1.0]
    assert values.grad.tolist() == torch.tensor([gamma, gamma, 0], dtype=dtype).tolist()


@pytest.mark.parametrize("discount", [1 - 2.0**-30, 2.0**-175])
def test_discount_retains_binary64_precision_for_float32_critic(discount):
    reward, value = (-1.0, 1.0) if discount > 0.5 else (0.0, 2.0**100)
    actual = bellman_target(
        torch.tensor([reward]), torch.tensor([False]), torch.tensor([value]), discount
    )
    assert actual.item() == float(
        Fraction(reward) + Fraction(value) * Fraction(discount)
    )


def nearest_critic_value(exact, dtype):
    """Round a dyadic rational directly on the critic's grid, including overflow."""
    if not exact:
        return 0.0
    value = abs(exact)
    info = torch.finfo(dtype)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    spacing = max(Fraction(2) ** exponent, Fraction(info.tiny)) * Fraction(info.eps)
    rounded = round(value / spacing) * spacing
    result = math.inf if rounded > info.max else float(rounded)
    return -result if exact < 0 else result


def check_critic_rounding(rows, gamma, dtype, future_dtype=None, device="cpu"):
    r_values, v_values = zip(*rows, strict=True)
    rewards = torch.tensor(r_values, dtype=dtype, device=device)
    futures = torch.tensor(v_values, dtype=future_dtype or dtype, device=device)
    output_dtype = torch.promote_types(rewards.dtype, futures.dtype)
    result = bellman_target(
        rewards, torch.zeros_like(rewards, dtype=torch.bool), futures, gamma
    ).flatten()
    assert result.dtype == output_dtype and result.device == rewards.device
    for r, v, actual in zip(
        rewards.tolist(), futures.tolist(), result.tolist(), strict=True
    ):
        expected = nearest_critic_value(
            Fraction(r) + Fraction(gamma) * Fraction(v), output_dtype
        )
        assert actual == expected, (dtype, r, gamma, v, actual, expected)
        if actual == 0:
            assert math.copysign(1, actual) == math.copysign(1, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_backup_rounds_once_at_midpoints_in_every_binade(dtype):
    info = torch.finfo(dtype)
    smallest = info.tiny * info.eps
    rows = [(0, smallest), (smallest, -smallest), (smallest, smallest)]
    for exp in range(math.frexp(info.tiny)[1] - 1, math.frexp(info.max)[1]):
        base = 2.0**exp
        step = base * info.eps
        rows.extend([(base, step), (base + step, step), (base + step, -step)])
    rows.append((info.max, 2.0 ** (math.frexp(info.max)[1] - 1) * info.eps))
    rows += [(-r, -v) for r, v in rows]
    for gamma in (math.nextafter(0.5, 0), 0.5, math.nextafter(0.5, 1), 0.5 - 2.0**-22):
        check_critic_rounding(rows, gamma, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_tiny_reward_decides_successor_midpoint(dtype):
    info = torch.finfo(dtype)
    smallest = info.tiny * info.eps
    rows = list(itertools.product([-smallest, 0, smallest], [-2.0, 2.0]))
    for gamma in (0.5 + info.eps / 4, 0.5 + 3 * info.eps / 4):
        check_critic_rounding(rows, gamma, dtype)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_zero_td_loss_at_half_spacing_attains_the_value_error_bound(dtype):
    step = torch.finfo(dtype).eps
    value = torch.tensor([[1.0], [-1.0]], dtype=dtype)
    reward = torch.tensor([0.5 + step / 2, -0.5 - step / 2], dtype=dtype)
    target = bellman_target(reward, torch.tensor([False, False]), value, 0.5)
    assert torch.equal(target, value)
    assert td_regression(value, target).item() == 0.0
    for r, u in zip(reward.tolist(), value.flatten().tolist(), strict=True):
        true_value = Fraction(r) / (1 - Fraction(0.5))
        assert abs(Fraction(u) - true_value) == Fraction(step / 2) / (1 - Fraction(0.5))


@pytest.mark.parametrize(
    "dtype,infinity_bits", [(torch.float16, 0x7C00), (torch.bfloat16, 0x7F80)]
)
def test_both_sides_and_exact_tie_in_every_half_precision_cell(dtype, infinity_bits):
    # Positive encodings are ordered, and their low bit determines tie parity.
    bits = torch.arange(infinity_bits, dtype=torch.int32)
    lo = bits.short().view(dtype)
    hi = (bits + 1).short().view(dtype)
    virtual = 2.0 ** math.frexp(torch.finfo(dtype).max)[1]
    spacing = (hi.double().clamp_max(virtual) - lo.double()).to(dtype)
    rewards = torch.cat((lo, -lo))
    values = torch.cat((spacing, -spacing))
    done = torch.zeros_like(rewards, dtype=torch.bool)
    for gamma, expected in (
        (math.nextafter(0.5, 0), lo),
        (0.5, torch.where(bits % 2 == 0, lo, hi)),
        (math.nextafter(0.5, 1), hi),
    ):
        actual = bellman_target(rewards, done, values, gamma).flatten()
        assert torch.equal(
            actual.view(torch.int16), torch.cat((expected, -expected)).view(torch.int16)
        )


@pytest.mark.parametrize(
    "dtype,future_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
        (torch.float16, torch.bfloat16),
        (torch.float64, torch.float32),
    ],
)
def test_backup_on_independent_signed_dyadics(dtype, future_dtype):
    rng = random.Random(107)

    def sample(t):
        info = torch.finfo(t)
        exp = rng.randrange(math.frexp(info.tiny)[1] - 2, math.frexp(info.max)[1] - 1)
        return math.ldexp(rng.uniform(-1, 1), exp)

    rows = [(sample(dtype), sample(future_dtype)) for _ in range(257)]
    for gamma in (math.nextafter(0.0, 1), 2.0**-175, 0.99, math.nextafter(1.0, 0)):
        check_critic_rounding(rows, gamma, dtype, future_dtype)


@pytest.mark.parametrize(
    "device",
    [
        d
        for d, available in (
            ("cpu", True),
            ("cuda", torch.cuda.is_available()),
            ("mps", torch.backends.mps.is_available()),
        )
        if available
    ],
)
def test_corrected_backup_through_the_sampled_successor_entry_point(device):
    from stateset_agents.training.offline_rl_targets import successor_target

    eps = torch.finfo(torch.float32).eps
    r = torch.tensor([1 + eps, 8.0], device=device)
    states = torch.tensor([[eps / 2], [math.nan]], device=device)
    evaluated = []

    def evaluate(active):
        evaluated.extend(active.flatten().tolist())
        return active

    target = successor_target(
        r,
        torch.tensor([False, True], device=device),
        states,
        math.nextafter(1.0, 0),
        evaluate,
    )
    assert target.device == r.device and target.flatten().tolist() == r.tolist()
    assert evaluated == [eps / 2]


@pytest.mark.parametrize("learner", ["cql", "iql"])
def test_midpoint_backup_in_actual_offline_learner(learner):
    gamma = math.nextafter(1.0, 0)
    eps = torch.finfo(torch.float32).eps
    states = torch.zeros(1, 1)
    reward = torch.tensor([1 + eps])
    done = torch.tensor([False])

    def set_constant(net, value):
        with torch.no_grad():
            net.network[-1].weight.zero_()
            net.network[-1].bias.fill_(value)

    if learner == "cql":
        q = ConservativeQLearning(
            1, 1, CQLConfig(num_layers=1, discount_factor=gamma), "cpu"
        )
        for net in (q.q1_target, q.q2_target):
            set_constant(net, eps / 2)
        for actions in (states, None):
            assert (
                q._bellman_target(reward, states, done, actions).item() == reward.item()
            )
    else:
        q = ImplicitQLearning(
            1, 1, IQLConfig(num_layers=1, discount_factor=gamma), "cpu"
        )
        set_constant(q.value_target, eps / 2)
        for net in (q.q1, q.q2):
            set_constant(net, reward.item())
        metrics = q.train_step(states, states, reward, states, done)
        assert metrics["q_loss"] == 0.0
        assert all(net.network[-1].bias.item() == reward.item() for net in (q.q1, q.q2))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_midpoint_correction_preserves_bellman_autodiff_and_stopped_rows(dtype):
    eps = torch.finfo(dtype).eps
    gamma = math.nextafter(1.0, 0)
    reward = torch.tensor([1 + eps, -1 - eps, 7.0], dtype=dtype)
    future = torch.tensor([eps / 2, -eps / 2, math.nan], dtype=dtype)
    done = torch.tensor([False, False, True])

    def target(r, v):
        return bellman_target(r, done, v, gamma).flatten()

    values, pullback = torch.func.vjp(target, reward, future)
    assert values.tolist() == reward.tolist()
    dr, dv = pullback(torch.tensor([-2.0, 3.0, 4.0], dtype=dtype))
    assert dr.tolist() == [-2.0, 3.0, 4.0] and dv.tolist() == [-2.0, 3.0, 0.0]
    _, tangent = torch.func.jvp(
        target, (reward, future), (torch.ones_like(reward), torch.ones_like(future))
    )
    assert tangent.tolist() == [2.0, 2.0, 1.0]


@pytest.mark.asyncio
async def test_pretraining_uses_logged_actions_at_cutoffs_and_handles_true_terminals():
    t = trainer("cql")
    t._init_offline_learner()
    calls = []

    def evaluate(states, actions):
        calls.extend(
            zip(states.flatten().tolist(), actions.flatten().tolist(), strict=True)
        )
        return actions

    t._offline_learner.q1_target.forward = evaluate
    dataset = {
        "states": np.zeros((4, 1)),
        "actions": np.zeros((4, 1)),
        "rewards": np.ones((4, 1)),
        "next_states": np.arange(4).reshape(-1, 1),
        "next_actions": np.arange(8, 12).reshape(-1, 1),
        "terminated": [0, 0, 1, 1],
        "truncated": [0, 1, 0, 1],
        "dones": [0, 1, 1, 1],
    }
    await t.pretrain_value_functions(
        SimpleNamespace(to_offline_rl_format=lambda: dataset), num_steps=1
    )
    assert sorted(calls) == [(0.0, 8.0), (1.0, 9.0)]
    seen = []

    def collect(states, actions, rewards, next_states, dones, next_actions):
        seen.extend(
            zip(
                next_states.flatten().tolist(),
                dones.flatten().tolist(),
                next_actions.flatten().tolist(),
                strict=True,
            )
        )
        return {}

    list(
        transition_epochs(
            dataset,
            collect,
            algorithm="cql",
            state_dim=1,
            action_dim=1,
            device="cpu",
            num_epochs=1,
            batch_size=2,
        )
    )
    assert sorted(seen) == [
        (0.0, 0.0, 8.0),
        (1.0, 0.0, 9.0),
        (2.0, 1.0, 10.0),
        (3.0, 1.0, 11.0),
    ]


@pytest.mark.parametrize(
    "data",
    [
        {"dones": np.array([1 - 2.0**-35], dtype=np.float64)},
        {"dones": [0], "terminated": [1], "truncated": [1]},
        {"dones": [1], "terminated": [0], "truncated": [0]},
        {"terminated": [0], "truncated": [2]},
    ],
)
def test_invalid_terminal_evidence(data):
    with pytest.raises(ValueError):
        transition_terminals(data, 1, "cpu")


def test_tabular_residual_certifies_exact_value_with_censored_successors():
    # Independent solution by elimination, then repeated calls to the native TD target.
    r = (Fraction(-1), Fraction(3))
    for gamma, continue_a, continue_b in itertools.product(
        map(Fraction, [0, 0.5, 1]), (False, True), (False, True)
    ):
        a, b = gamma * continue_a / 2, gamma * continue_b / 2
        fixed = ((r[0] + a * r[1]) / (1 - a * b), (r[1] + b * r[0]) / (1 - a * b))
        u = [Fraction(5), Fraction(-3)]
        rates = [a, b]
        for _ in range(6):
            labels = torch.tensor([-1.0, -1.0, 3.0, 3.0], dtype=torch.float64)
            future = torch.tensor(
                [float(u[1]), math.nan, float(u[0]), math.nan], dtype=torch.float64
            )
            d = torch.tensor([not continue_a, True, not continue_b, True])
            v = list(
                map(
                    Fraction,
                    bellman_target(labels, d, future, float(gamma))
                    .reshape(2, 2)
                    .mean(1)
                    .tolist(),
                )
            )
            assert v == [r[i] + rates[i] * u[1 - i] for i in (0, 1)]
            err = max(abs(x - y) for x, y in zip(u, fixed, strict=True))
            residual = max(abs(x - y) for x, y in zip(u, v, strict=True))
            assert err * (1 - max(rates)) <= residual
            u = v


@pytest.mark.parametrize("kind", ["iql", "cql", "bcq", "bear"])
@pytest.mark.parametrize("invalid", ["broadcast", "active_nan"])
def test_invalid_direct_batch_rejected_before_any_learning(kind, invalid):
    t = trainer(kind)
    t._init_offline_learner()
    learner = t._offline_learner
    states = torch.ones(2, 1)
    if invalid == "broadcast":
        successors, rewards, dones = torch.ones(1, 1), torch.ones(1), torch.zeros(1)
    else:
        successors, rewards, dones = (
            torch.tensor([[0.0], [math.nan]]),
            torch.ones(2),
            torch.zeros(2),
        )
    with pytest.raises(ValueError):
        learner.train_step(states, states, rewards, successors, dones)
    assert learner.training_step == 0
    assert all(
        not obj.state
        for obj in vars(learner).values()
        if isinstance(obj, torch.optim.Optimizer)
    )


@pytest.mark.parametrize("gamma,flag", [(0.0, 0), (0.5, 1)])
def test_public_dataset_allows_unused_successor_placeholders(gamma, flag):
    t = trainer("cql", gamma)
    row = {
        "states": [[0.0]],
        "actions": [[0.0]],
        "rewards": [1.0],
        "next_states": [[math.nan]],
        "next_actions": [[math.inf]],
        "dones": [flag],
    }
    data = t._validate_offline_data(row)
    assert data["dones"].tolist() == [flag]
    if flag:
        with pytest.raises(ValueError, match="finite"):
            t._validate_offline_data(dict(row, terminated=[0], truncated=[1]))


def test_epoch_checks_all_live_rows_before_the_first_update():
    data = {
        "states": [[1.0], [2.0]],
        "actions": [[0.0], [1.0]],
        "rewards": [0.0, 1.0],
        "next_states": [[math.nan], [math.nan]],
        "terminated": [1, 0],
        "truncated": [0, 1],
    }

    def unexpected(**_):
        pytest.fail("an update ran before the invalid live successor was checked")

    with pytest.raises(ValueError, match="finite"):
        list(
            transition_epochs(
                data,
                unexpected,
                algorithm="iql",
                state_dim=1,
                action_dim=1,
                device="cpu",
                num_epochs=1,
                batch_size=1,
            )
        )


def test_mixed_precision_critic_regresses_to_the_full_detached_target():
    p = torch.tensor([[1.0], [2.0]], requires_grad=True)
    target = torch.tensor(
        [[1.0 + 2.0**-30], [3.0]], dtype=torch.float64, requires_grad=True
    )
    loss = td_regression(p, target)
    loss.backward()
    assert p.grad.flatten().tolist() == [-(2.0**-30), -1.0]
    assert target.grad is None
    assert loss.item() == float((Fraction(1, 2**60) + 1) / 2)


def test_wide_square_can_fit_after_averaging_even_when_the_row_square_overflows():
    p = torch.tensor([2.0**512, 0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    loss = td_regression(p, torch.zeros_like(p))
    assert loss.item() == 2.0**1022
    loss.backward()
    assert p.grad.tolist() == [2.0**511, 0.0, 0.0, 0.0]


def test_cql_critic_update_accepts_binary64_rewards_and_binary32_weights():
    cql = ConservativeQLearning(
        1, 1, CQLConfig(hidden_size=2, num_random_actions=2), "cpu"
    )
    x = torch.zeros(2, 1)
    stats = cql.train_step(
        x, x, torch.tensor([1.0, -1.0], dtype=torch.float64), x, torch.ones(2)
    )
    assert stats["bellman_loss"] == pytest.approx(1)
    assert cql.training_step == 1


def ordered_episodes():
    return {
        "states": np.array([[1.0], [2.0], [50.0], [60.0]]),
        "next_states": np.array([[2.0], [math.nan], [60.0], [65.0]]),
        "actions": np.array([[0.0], [1.0], [-1.0], [2.0]]),
        "rewards": [3.0, 4.0, -2.0, 1.0],
        "terminated": [0, 1, 0, 1],
        "truncated": [0, 0, 0, 1],
    }


@pytest.mark.asyncio
async def test_dt_pretraining_dispatches_whole_episodes_to_the_real_trainer(
    monkeypatch,
):
    import stateset_agents.training.decision_transformer as dt

    original_config = dt.DecisionTransformerConfig

    def tiny_config(**options):
        assert not options["use_conversation_embeddings"]
        return original_config(
            **options,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            warmup_steps=0,
            max_context_length=4,
        )

    monkeypatch.setattr(dt, "DecisionTransformerConfig", tiny_config)
    t = trainer("decision_transformer", 0.5)
    t._init_offline_learner()
    inner = t._offline_learner
    weight = inner.model.predict_action[0].weight
    initial = weight.detach().clone()
    seen = []
    train = inner.train_step

    def observe(episodes):
        seen.extend(
            (
                ep["states"].flatten().tolist(),
                ep["returns_to_go"].tolist(),
                ep["timesteps"].tolist(),
            )
            for ep in episodes
        )
        return train(episodes)

    monkeypatch.setattr(inner, "train_step", observe)
    result = await t.pretrain_value_functions(
        SimpleNamespace(to_offline_rl_format=ordered_episodes), num_steps=2
    )
    assert sorted(seen) == sorted(
        [([1.0, 2.0], [5.0, 4.0], [0, 1]), ([50.0, 60.0], [-1.5, 1.0], [0, 1])] * 2
    )
    assert inner.training_step == 2 and math.isfinite(result["offline_action_loss"])
    assert not torch.equal(initial, weight)


def test_dt_uses_supplied_returns_when_realized_continuation_is_unknown():
    from stateset_agents.training.offline_rl_batches import decision_sequences

    raw = ordered_episodes()
    raw["terminated"][-1] = 0
    t = trainer("decision_transformer", 0.5)
    data = t._validate_offline_data(raw)
    with pytest.raises(ValueError, match="cutoff"):
        decision_sequences(raw, data, 0.5)
    raw["returns_to_go"] = [5.0, 4.0, 0.0, 4.0]
    seq = decision_sequences(raw, data, 0.5)
    assert [ep["returns_to_go"].tolist() for ep in seq] == [[5.0, 4.0], [0.0, 4.0]]
    raw.pop("returns_to_go")
    seq = decision_sequences(raw, data, 0.0)
    assert [ep["returns_to_go"].tolist() for ep in seq] == [[3.0, 4.0], [-2.0, 1.0]]
    raw["next_states"][0, 0] = 50.0
    with pytest.raises(ValueError, match="ordered"):
        decision_sequences(raw, t._validate_offline_data(raw), 0.5)


def test_public_td_and_dt_preserve_sub_float32_cancellation_from_dataset():
    from stateset_agents.training.offline_rl_batches import (
        decision_sequences,
        transition_tensors,
    )

    small = 2.0**-30
    raw = {
        "states": [[0.0], [1.0]],
        "next_states": [[1.0], [2.0]],
        "actions": [[0.0], [0.0]],
        "terminated": [False, True],
        "rewards": [1.0 + small, -1.0],
    }
    t = trainer("decision_transformer", 1.0)
    arrays = t._validate_offline_data(raw)
    (episode,) = decision_sequences(raw, arrays, 1.0)
    assert episode["returns_to_go"].tolist() == [small, -1.0]
    batch = transition_tensors(
        raw, algorithm="iql", state_dim=1, action_dim=1, device="cpu", discount=1.0
    )
    target = bellman_target(
        batch["rewards"], batch["dones"], torch.tensor([-1.0, math.nan]), 1.0
    )
    assert target.flatten().tolist() == [small, -1.0]


@pytest.mark.asyncio
async def test_legacy_rounded_reward_binding_upgrades_on_real_policy_resume(tmp_path):
    from tests.unit.test_offline_policy_training import LoggedDataset
    from tests.unit.test_offline_policy_training import trainer as policy_trainer

    class Precise(LoggedDataset):
        def __init__(self, delta):
            self.delta = delta

        def to_offline_rl_format(self):
            data = super().to_offline_rl_format()
            data["rewards"][0] += self.delta
            return data

    model = policy_trainer()
    await model.train(dataset=LoggedDataset())
    path = tmp_path / "policy.pt"
    model.save(str(path))
    payload = torch.load(path, weights_only=True)
    payload.pop("dataset_hash_version")
    torch.save(payload, path)
    restored = policy_trainer()
    restored.load(str(path))
    old_digest = restored.dataset_sha256
    await restored.train(dataset=Precise(2.0**-30), num_epochs=2)
    assert restored.dataset_sha256 != old_digest
    restored.save(str(path))
    restored = policy_trainer()
    restored.load(str(path))
    with pytest.raises(ValueError, match="dataset differs"):
        await restored.train(dataset=Precise(2.0**-31), num_epochs=3)


def test_lagged_critic_certificate_is_attained_by_native_value_pretraining():
    # The continuing self-loop has fixed value 2. The fitted value is 6
    # and the target snapshot is 8, so the measured TD residual is just 1.
    t = trainer(gamma=0.5)
    with torch.no_grad():
        for net, bias in ((t.value_net, 6.0), (t.value_target, 8.0)):
            net.network[0].weight.zero_()
            net.network[0].bias.fill_(bias)
    state = torch.zeros(1, 1)
    fitted = t.value_net(state).item()
    lag = abs(t.value_target(state).item() - fitted)
    measured = t._train_value_step(state, state, torch.ones(1), state, torch.zeros(1))
    assert measured == 1.0
    assert (math.sqrt(measured) + 0.5 * lag) / (1 - 0.5) == abs(fitted - 2.0)


@pytest.mark.parametrize(
    "rt,vt,r,v,expected",
    [
        (torch.float64, torch.int32, -(2.0**24), 2**24 + 1, 1),
        (torch.int64, torch.float64, 2**53 + 1, -(2.0**53), 1),
        (torch.float64, torch.int64, -(2.0**63), 2**63 - 1, -1),
        (torch.uint64, torch.float64, 2**64 - 1, -(2.0**64), -1),
        (torch.int64, torch.uint64, -(2**63), 2**63 + 1, 1),
        (torch.uint64, torch.int64, 2**63 + 1, -(2**63), 1),
    ],
)
def test_integer_source_cancellation_and_terminal_rows(rt, vt, r, v, expected):
    reward = torch.tensor([r, 7], dtype=rt)
    future = torch.tensor([v, v], dtype=vt)
    output = bellman_target(reward, torch.tensor([False, True]), future, 1.0)
    assert output.flatten().tolist() == [expected, 7]
    legacy_future = vt if vt.is_floating_point else torch.get_default_dtype()
    assert output.dtype == torch.promote_types(rt, legacy_future)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_integer_reward_and_subnormal_product_decide_opposite_sides_of_a_tie(dtype):
    info = torch.finfo(dtype)
    # An integer midpoint (odd) cannot itself be stored in the critic dtype.
    # Even a nonzero continuation below the binary64 subnormal range decides it.
    midpoint = int(2 / info.eps) + 1
    smallest = info.tiny * info.eps
    gamma = math.nextafter(0.0, 1.0)
    r = torch.tensor([midpoint, midpoint, -midpoint, -midpoint])
    v = torch.tensor([-smallest, smallest, -smallest, smallest], dtype=dtype)
    out = bellman_target(r, torch.zeros(4, dtype=torch.bool), v, gamma)
    expected = [
        nearest_critic_value(Fraction(a) + Fraction(b) * Fraction(gamma), dtype)
        for a, b in zip(r.tolist(), v.tolist(), strict=True)
    ]
    assert out.flatten().tolist() == expected
    assert expected[0] != expected[1]


@pytest.mark.parametrize(
    "rt,vt",
    [
        (torch.int64, torch.float16),
        (torch.uint64, torch.float32),
        (torch.int32, torch.float64),
        (torch.float64, torch.int64),
        (torch.int64, torch.uint64),
    ],
)
def test_integer_backup_matches_direct_rational_grid(rt, vt):
    rng = random.Random(338)

    def values(dtype):
        if dtype.is_floating_point:
            info = torch.finfo(dtype)
            return [
                math.ldexp(
                    rng.uniform(-1, 1),
                    rng.randint(math.frexp(info.tiny)[1], math.frexp(info.max)[1] - 1),
                )
                for _ in range(97)
            ]
        lim = torch.iinfo(dtype)
        return [lim.min, lim.max] + [rng.randint(lim.min, lim.max) for _ in range(95)]

    r = torch.tensor(values(rt), dtype=rt)
    v = torch.tensor(values(vt), dtype=vt)
    for g in (
        0.0,
        math.nextafter(0.0, 1.0),
        0.125,
        0.99,
        math.nextafter(1.0, 0.0),
        1.0,
    ):
        ans = bellman_target(r, torch.zeros(len(r), dtype=torch.bool), v, g).flatten()
        assert ans.tolist() == [
            nearest_critic_value(Fraction(a) + Fraction(b) * Fraction(g), ans.dtype)
            for a, b in zip(r.tolist(), v.tolist(), strict=True)
        ]


def test_integer_routing_keeps_mixed_autodiff_vmap_and_terminal_mask():
    r = torch.tensor([2**53 + 1, 7], dtype=torch.int64)
    v = torch.tensor([-(2.0**54), math.nan], dtype=torch.float64)
    stop = torch.tensor([False, True])

    def f(v):
        return bellman_target(r, stop, v, 0.5).flatten()

    out, dv = torch.func.jvp(f, (v,), (torch.tensor([2.0, 3.0], dtype=torch.float64),))
    assert out.tolist() == [1.0, 7.0] and dv.tolist() == [1.0, 0.0]
    assert torch.func.grad(lambda x: f(x).sum())(v).tolist() == [0.5, 0.0]
    result = torch.vmap(lambda rr, vv: bellman_target(rr, stop, vv, 0.5))(
        r.expand(3, -1), v.expand(3, -1)
    )
    assert result.flatten().tolist() == [1.0, 7.0] * 3
    floating = torch.tensor([-(2.0**53), 4.0], dtype=torch.float64)

    def reward_target(x):
        return bellman_target(x, stop, r, 1.0).sum()

    assert torch.func.grad(reward_target)(floating).tolist() == [1.0, 1.0]


@pytest.mark.parametrize(
    "dtype,reward,future,terminal",
    [
        (np.int64, 2**53 + 1, -(2.0**53), 2**30 + 1),
        (np.uint64, 2**64 - 1, -(2.0**64), 2**30 + 1),
        (None, 2**63 + 1, -(2.0**63), 1),
        (None, 2**53 + 1, -(2.0**53), 0.0),
        (None, 2**63 + 1, -(2.0**63), -1),
        (None, 2**53 + 1, -(2.0**53), 0.5),
    ],
)
def test_loaded_integer_labels_reach_cql_iql_and_auxiliary_critics(
    dtype, reward, future, terminal
):
    from stateset_agents.training.offline_rl_batches import transition_tensors

    labels = [reward, terminal]
    raw = {
        "states": [[0], [0]],
        "actions": [[0], [0]],
        "next_states": [[0], [0]],
        "dones": [False, True],
        "rewards": np.array(labels, dtype=dtype) if dtype else labels,
    }
    batch = transition_tensors(
        raw, algorithm="iql", state_dim=1, action_dim=1, device="cpu", discount=1.0
    )
    assert batch["rewards"].flatten().tolist() == labels
    learner = ConservativeQLearning(
        1, 1, CQLConfig(num_layers=1, discount_factor=1.0), "cpu"
    )
    with torch.no_grad():
        for q in (learner.q1_target, learner.q2_target):
            q.network[-1].weight.zero_()
            q.network[-1].bias.fill_(future)
    target = learner._bellman_target(
        batch["rewards"], batch["next_states"], batch["dones"], batch["actions"]
    )
    live = reward + int(future)
    assert target.dtype == torch.float64 and target.flatten().tolist() == [
        live,
        terminal,
    ]
    tr = trainer(gamma=1.0)
    assert tr._validate_offline_data(raw)["rewards"].tolist() == labels
    iql = ImplicitQLearning(1, 1, IQLConfig(num_layers=1, discount_factor=1.0), "cpu")
    with torch.no_grad():
        iql.value_target.network[-1].bias.fill_(future)
        for q in (iql.q1, iql.q2):
            q.network[-1].weight.zero_()
            q.network[-1].bias.fill_(live)
    metrics = iql.train_step(**{k: v[:1] for k, v in batch.items()})
    assert metrics["q_loss"] == 0.0
    with torch.no_grad():
        for net, bias in ((tr.value_net, live), (tr.value_target, future)):
            net.network[0].weight.zero_()
            net.network[0].bias.fill_(bias)
    assert tr._train_value_step(**{k: v[:1] for k, v in batch.items()}) == 0.0


def test_integer_recorded_episodes_postpone_rounding_until_each_complete_suffix():
    from stateset_agents.training.offline_rl_batches import decision_sequences

    raw = {
        "states": [[0], [1]],
        "next_states": [[1], [2]],
        "actions": [[0], [0]],
        "dones": [False, True],
        "rewards": np.array([2**63 - 1, -(2**63)], dtype=np.int64),
    }
    arrays = trainer("decision_transformer", 1.0)._validate_offline_data(raw)
    (episode,) = decision_sequences(raw, arrays, 1.0)
    assert episode["returns_to_go"].tolist() == [-1.0, -(2.0**63)]


def test_heterogeneous_rewards_survive_export_batching_and_decision_returns():
    from stateset_agents.data.conversation_dataset import (
        ConversationDataset,
        ConversationDatasetConfig,
    )
    from stateset_agents.training.offline_rl_batches import decision_sequences
    from stateset_agents.training.offline_rl_targets import successor_target

    labels = [2**53 + 1, 0.5, -(2**53)]
    turns = [
        {"role": role, "content": str(i), "reward": label if role == "assistant" else 0}
        for i, label in enumerate(labels)
        for role in ("user", "assistant")
    ]
    data = ConversationDataset.from_dict(
        {"trajectories": [{"turns": turns}]},
        ConversationDatasetConfig(normalize_rewards=False, discount_factor=1.0),
    )
    data.embedding_cache = SimpleNamespace(get_or_compute=lambda *_: torch.zeros(1))
    raw = data.to_offline_rl_format()
    assert raw["rewards"].tolist() == labels
    raw["rewards"] = raw["rewards"].reshape(-1, 1)
    arrays = trainer("decision_transformer", 1.0)._validate_offline_data(raw)
    (episode,) = decision_sequences(raw, arrays, 1.0)
    assert episode["returns_to_go"][0] == 1.5
    seen = []

    def step(states, actions, rewards, next_states, dones):
        assert rewards.shape == (1, 1)
        seen.extend(rewards.flatten().tolist())

        def inactive(*_):
            pytest.fail("zero-discount batch evaluated a successor")

        result = successor_target(rewards, dones, next_states, 0, inactive)
        assert result.item() == float(rewards.item())
        return {"visited": 1.0}

    (metrics,) = transition_epochs(
        raw,
        step,
        algorithm="iql",
        state_dim=1,
        action_dim=1,
        device="cpu",
        num_epochs=1,
        batch_size=1,
    )
    assert metrics["visited"] == 1.0 and sorted(seen) == sorted(labels)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_heterogeneous_reward_validation_rejects_nonfinite_rows(invalid):
    raw = {
        "states": [[0], [0]],
        "actions": [[0], [0]],
        "next_states": [[0], [0]],
        "dones": [True, True],
        "rewards": [2**63 + 1, invalid],
    }
    with pytest.raises(ValueError, match="must be finite"):
        trainer()._validate_offline_data(raw)


def test_reward_columns_preserve_extended_scalars_or_report_unsupported_precision():
    from stateset_agents.data.reward_columns import reward_array

    high = 2**63 + 1
    assert reward_array([high, np.longdouble(0.5)]).tolist() == [high, 0.5]
    ordinary = reward_array(np.array([np.int16(1), np.float32(0.5)], dtype=object))
    assert ordinary.dtype == np.float64 and ordinary.tolist() == [1, 0.5]
    if np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant:
        extended = np.nextafter(np.longdouble(1), np.longdouble(2))
        with pytest.raises(ValueError, match="fit binary64 exactly"):
            reward_array(np.array([extended, high], dtype=object))


def test_heterogeneous_batches_score_live_successors_together_and_keep_terminal_zero():
    from stateset_agents.training.offline_rl_batches import transition_tensors
    from stateset_agents.training.offline_rl_targets import successor_target

    high = 2**63 + 1
    raw = {
        "states": [[0], [1], [2]],
        "actions": [[0], [0], [0]],
        "next_states": [[2], [math.nan], [4]],
        "dones": [False, True, False],
        "rewards": [high, -0.0, -0.5],
    }
    batch = transition_tensors(
        raw, algorithm="iql", state_dim=1, action_dim=1, device="cpu", discount=1.0
    )
    calls = []

    def score(states):
        calls.append(states.flatten().tolist())
        return torch.tensor([[-(2**63)], [1]], dtype=torch.int64)

    labels, dones, states = (batch[k] for k in ("rewards", "dones", "next_states"))
    target = successor_target(labels, dones, states, 1.0, score)
    assert calls == [[2.0, 4.0]] and target.flatten().tolist() == [1.0, 0.0, 0.5]
    assert torch.signbit(target[1, 0])
    terminal = successor_target(labels, torch.ones_like(dones), states, 1.0, score)
    assert torch.signbit(terminal[1, 0]) and calls == [[2.0, 4.0]]

    # A homogeneous unsigned column also keeps wide output when the critic is integral.
    raw["rewards"] = [high, 2**30 + 1, 1]
    batch = transition_tensors(
        raw, algorithm="iql", state_dim=1, action_dim=1, device="cpu", discount=1.0
    )
    target = successor_target(batch["rewards"], dones, states, 1.0, score)
    assert target.dtype == torch.float64 and target.flatten().tolist() == [
        1.0,
        2**30 + 1,
        2.0,
    ]


@pytest.mark.parametrize("method", ["standard", "minmax", "percentile"])
def test_integer_conversation_normalization_is_invariant_under_a_large_offset(method):
    from stateset_agents.data.conversation_dataset import (
        ConversationDataset,
        ConversationDatasetConfig,
    )

    config = ConversationDatasetConfig(
        reward_normalization_method=method, discount_factor=0.5
    )

    def normalized(offset):
        data = {
            "trajectories": [
                {
                    "turns": [
                        {"role": "assistant", "reward": offset + k}
                        for k in [0, 1, 2, 7, 11]
                    ]
                }
            ]
        }
        episode = ConversationDataset.from_dict(data, config)[0]
        return episode.turn_rewards, episode.returns_to_go

    assert normalized(2**59) == normalized(0)


def test_integer_json_decision_labels_reach_offline_export_and_exact_returns():
    from stateset_agents.data.conversation_dataset import (
        ConversationDataset,
        ConversationDatasetConfig,
    )

    rows = [
        {"role": role, "content": str(i), "reward": r}
        for i, (role, r) in enumerate(
            [
                ("user", 0),
                ("assistant", 2**63 - 1),
                ("user", 0),
                ("assistant", -(2**63)),
            ]
        )
    ]
    data = ConversationDataset.from_dict(
        {"trajectories": [{"turns": rows}]},
        ConversationDatasetConfig(normalize_rewards=False, discount_factor=1.0),
    )
    data.embedding_cache = SimpleNamespace(get_or_compute=lambda *_: torch.zeros(1))
    assert data[0].returns_to_go == [-1.0, -1.0, -(2.0**63), -(2.0**63)]
    assert data.to_offline_rl_format()["rewards"].tolist() == [2**63 - 1, -(2**63)]


@pytest.mark.asyncio
@pytest.mark.parametrize("old_version", [1, 2])
@pytest.mark.parametrize("mixed", [False, True])
async def test_pre_integer_checkpoint_migrates_to_an_exact_label_identity(
    tmp_path, old_version, mixed
):
    from tests.unit.test_offline_policy_training import LoggedDataset
    from tests.unit.test_offline_policy_training import trainer as policy_trainer

    class Labels(LoggedDataset):
        def __init__(self, first, dtype):
            self.first, self.dtype = first, dtype

        def to_offline_rl_format(self):
            data = super().to_offline_rl_format()
            labels = [self.first, 2**30 + 1, 0.5 if mixed else 0]
            data["rewards"] = (
                np.array(labels, dtype=self.dtype) if self.dtype else labels
            )
            return data

    dtype = np.float32 if old_version == 1 else np.float64
    source_dtype = None if mixed else np.int64
    t = policy_trainer()
    await t.train(dataset=Labels(2**53, dtype))
    path = tmp_path / "old.pt"
    t.save(str(path))
    payload = torch.load(path, weights_only=True)
    payload["dataset_hash_version"] = old_version
    torch.save(payload, path)
    tr = policy_trainer()
    tr.load(str(path))
    await tr.train(dataset=Labels(2**53 + 1, source_dtype), num_epochs=2)
    assert tr._dataset_hash_version == 3
    tr.save(str(path))
    resumed = policy_trainer()
    resumed.load(str(path))
    with pytest.raises(ValueError, match="dataset differs"):
        await resumed.train(dataset=Labels(2**53, source_dtype), num_epochs=3)


@pytest.mark.asyncio
async def test_heterogeneous_checkpoint_resumes_with_the_exact_source_labels(tmp_path):
    from tests.unit.test_offline_policy_training import LoggedDataset
    from tests.unit.test_offline_policy_training import trainer as policy_trainer

    class Labels(LoggedDataset):
        def __init__(self, labels):
            self.labels = labels

        def to_offline_rl_format(self):
            return {**super().to_offline_rl_format(), "rewards": self.labels}

    high = 2**63 + 1
    t = policy_trainer()
    await t.train(dataset=Labels([high, -0.5, 1]))
    path = tmp_path / "mixed.pt"
    t.save(str(path))
    restored = policy_trainer()
    restored.load(str(path))
    cells = np.array([np.uint64(high), np.float32(-0.5), np.int16(1)], dtype=object)
    await restored.train(dataset=Labels(cells), num_epochs=2)
    assert restored.current_epoch == 2
    with pytest.raises(ValueError, match="dataset differs"):
        await restored.train(dataset=Labels([high + 1, -0.5, 1]), num_epochs=3)


@pytest.mark.parametrize("visits", [2, 4, 16])
def test_pooled_critic_bound_is_attained_with_coverage_rounding_and_lag(visits):
    t = trainer(gamma=0.5)
    y = 2**23 + 2
    u, v, r = y + 1, y + 3, y // 2 - 2
    states = torch.zeros(visits, 1)
    states[0] = 1
    rewards = states.flatten() * r
    done = states.flatten() == 0
    with torch.no_grad():
        for net, weight in ((t.value_net, u), (t.value_target, v)):
            net.network[0].weight.fill_(weight)
            net.network[0].bias.zero_()
        published = bellman_target(rewards, done, t.value_target(states), 0.5)
        assert published[0].item() == y
    loss = t._train_value_step(states, states, rewards, states, done)
    coverage = 1 / visits
    bias = y - (Fraction(r) + Fraction(v, 2))
    lag = v - u
    exact_value = 2 * r
    assert loss == coverage and bias == 0.5
    assert (math.sqrt(loss / coverage) + bias + 0.5 * lag) / 0.5 == u - exact_value == 5
    report = audit_logged_critic(
        [0, u],
        [0, v],
        [
            CriticRow(int(s), None if stop else int(s), reward, target)
            for s, stop, reward, target in zip(
                states.flatten().tolist(),
                done.tolist(),
                rewards.tolist(),
                published.flatten().tolist(),
                strict=True,
            )
        ],
        0.5,
    )
    assert report.loss == loss and report.within_state_variance == 0
    assert report.mean_target_bias_bound == bias and report.target_lag == lag
    assert report.minimum_frequency == coverage and report.contraction_factor == 0.5
    assert report.certifies(5) and not report.certifies(
        Fraction(5) - Fraction(1, 2**100)
    )


def test_pooled_target_variance_accounts_for_all_loss_at_the_true_float32_value():
    t = trainer(gamma=0.5)
    u = 2**23 + 1
    with torch.no_grad():
        for net in (t.value_net, t.value_target):
            net.network[0].bias.fill_(u)
            net.network[0].weight.zero_()
    states = torch.zeros(2, 1)
    done = torch.zeros(2, dtype=torch.bool)
    rewards = torch.tensor([u // 2, u // 2 + 1], dtype=torch.float32)
    labels = (
        bellman_target(rewards, done, t.value_target(states), 0.5).detach().double()
    )
    assert labels.flatten().tolist() == [u - 1, u + 1]
    exact_targets = rewards.double() + u / 2
    assert (labels.flatten() - exact_targets).tolist() == [-0.5, 0.5]
    assert labels.mean().item() == u == rewards.double().mean().item() / (1 - 0.5)
    loss = t._train_value_step(states, states, rewards, states, done)
    assert labels.var(unbiased=False).item() == loss == 1.0
    audit = audit_logged_critic(
        [u],
        [u],
        (
            CriticRow(0, 0, r, y)
            for r, y in zip(rewards.tolist(), labels.flatten().tolist(), strict=True)
        ),
        0.5,
    )
    assert audit.loss == audit.within_state_variance == 1
    assert audit.mean_target_bias_bound == 0 and audit.certifies(0)


def test_reported_underflowed_td_loss_is_not_an_upper_bound_for_a_certificate():
    prediction = torch.tensor([[2.0**-600]], dtype=torch.float64)
    target = bellman_target(
        torch.tensor([0]),
        torch.tensor([True]),
        torch.tensor([math.nan], dtype=torch.float64),
        1.0,
    )
    reported = td_regression(prediction, target).item()
    exact_mse = (Fraction(prediction.item()) - Fraction(target.item())) ** 2
    assert reported == 0 and exact_mse == Fraction(1, 2**1200)
    # On a terminal self-loop, the certified value error is exactly sqrt(exact_mse).
    assert Fraction(prediction.item()) ** 2 == exact_mse
    report = audit_logged_critic(
        [prediction.item()], [0], [CriticRow(0, None, 0, target.item())], 1
    )
    assert report.loss == exact_mse and report.within_state_variance == 0
    assert report.certifies(prediction.item())
    assert not report.certifies(math.nextafter(prediction.item(), 0))
    assert not report.certifies(0)


def test_exact_loss_handles_overflow_and_large_integers_from_published_tensors():
    p = torch.tensor([torch.finfo(torch.float64).max], dtype=torch.float64)
    y = -p
    assert math.isinf(td_regression(p, y).item())
    maximum = Fraction(p.item())
    result = audit_logged_critic(
        [maximum], [0], [CriticRow(0, None, y.item(), y.item())], 1
    )
    assert result.loss == (2 * maximum) ** 2 and result.certifies(2 * maximum)
    assert not result.certifies(2 * maximum - 1)
    # Casting either input to double before subtracting loses the unit residual.
    large = 2**63 + 1
    result = audit_logged_critic(
        [large], [large], [CriticRow(0, None, large - 1, large - 1)], 1
    )
    assert result.loss == 1 and result.certifies(1) and not result.certifies(0)


def test_certification_uses_each_states_frequency_and_signed_rounding_bias():
    rows = [CriticRow(0, None, 1, 3), CriticRow(0, 1, -2, 1), CriticRow(1, 0, 2, -1)]
    report = audit_logged_critic([2, 1], [0, 4], rows, Fraction(1, 2))
    assert report == audit_logged_critic([2, 1], [0, 4], reversed(rows), 0.5)
    assert report.loss == 2 and report.within_state_variance == Fraction(2, 3)
    assert report.minimum_frequency == Fraction(1, 3)
    assert report.contraction_factor == Fraction(1, 2)
    assert report.mean_target_bias_bound == 3 and report.target_lag == 3
    assert report.certifies(13) and not report.certifies(Fraction(12999, 1000))
    assert not report.certifies(0)


def test_certificate_declines_a_unit_contraction_and_missing_state_coverage():
    row = CriticRow(0, 0, 0, 0)
    assert not audit_logged_critic([0], [0], [row], 1).certifies(10**100)
    with pytest.raises(ValueError, match="Every declared state"):
        audit_logged_critic([0, 2], [0, 2], [row], 0.5)
    with pytest.raises(ValueError, match="Successor state"):
        audit_logged_critic([0], [0], [CriticRow(0, -1, 0, 0)], 0.5)


def test_certified_tolerances_bound_independently_solved_empirical_values():
    rng = random.Random(416)
    accepted = rejected = 0
    for _ in range(96):
        gamma = rng.choice([0.0, 0.375, 0.625, 0.875, 1.0])
        fitted = [rng.randrange(-4, 5) for _ in range(2)]
        frozen = [rng.randrange(-4, 5) for _ in range(2)]
        sources = [s for s in range(2) for _ in range(rng.randint(1, 5))]
        successors = [rng.choice([None, 0, 1]) for _ in sources]
        rewards = [rng.randrange(-24, 25) / 8 for _ in sources]
        # Real float16 published targets include a reward close to a midpoint.
        rewards[0] += 2.0**-11
        labels = (
            bellman_target(
                torch.tensor(rewards, dtype=torch.float16),
                torch.tensor([j is None for j in successors]),
                torch.tensor(
                    [0 if j is None else frozen[j] for j in successors],
                    dtype=torch.float16,
                ),
                gamma,
            )
            .flatten()
            .tolist()
        )
        stored = torch.tensor(rewards, dtype=torch.float16).tolist()
        rows = [
            CriticRow(s, j, r, y)
            for s, j, r, y in zip(sources, successors, stored, labels, strict=True)
        ]
        report = audit_logged_critic(fitted, frozen, rows, gamma)
        # Solve (I - gamma*P)V = mean reward with exact 2x2 determinants.
        lhs, rhs = [], []
        for s in range(2):
            visits = [r for r in rows if r.state == s]
            rhs.append(sum(Fraction(r.reward) for r in visits) / len(visits))
            lhs.append(
                [
                    int(j == s)
                    - Fraction(gamma)
                    * sum(r.successor == j for r in visits)
                    / len(visits)
                    for j in range(2)
                ]
            )
        (a, b), (c, d) = lhs
        determinant = a * d - b * c
        for tolerance in (0, Fraction(1, 2), 1, 8, 256):
            if report.certifies(tolerance):
                accepted += 1
                assert determinant > 0
                values = [
                    (rhs[0] * d - b * rhs[1]) / determinant,
                    (a * rhs[1] - rhs[0] * c) / determinant,
                ]
                assert (
                    max(abs(t - p) for t, p in zip(values, fitted, strict=True))
                    <= tolerance
                )
            else:
                rejected += 1
    assert accepted > 96 and rejected > 96
