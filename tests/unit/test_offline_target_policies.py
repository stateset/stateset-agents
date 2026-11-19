"""Refinement checks for standalone BCQ and BEAR sampled Bellman updates."""

import math
from fractions import Fraction

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.offline_rl_bcq import (  # noqa: E402
    BatchConstrainedQLearning,
    BCQConfig,
)
from stateset_agents.training.offline_rl_bear import (  # noqa: E402
    BEARConfig,
    ConversationalBEAR,
)
from stateset_agents.training.offline_rl_targets import (  # noqa: E402
    _mix,
    sampled_twin_target,
)
from stateset_agents.training.rl_losses import bellman_target  # noqa: E402


def learner(kind, **kw):
    common = {
        "hidden_size": 4,
        "num_layers": 2,
        "soft_target_tau": 0.25,
        "discount_factor": 0.5,
    }
    if kind == "bcq":
        config = dict(**common, vae_hidden_size=4, latent_dim=2, num_action_samples=3)
        return BatchConstrainedQLearning(1, 1, BCQConfig(**(config | kw)), "cpu")
    config = dict(**common, actor_hidden_size=4, num_samples_for_mmd=3, warmup_steps=0)
    return ConversationalBEAR(1, 1, BEARConfig(**(config | kw)), "cpu")


def policy(model):
    return "perturbation" if isinstance(model, BatchConstrainedQLearning) else "actor"


def fail_if_called(*args, **kwargs):
    pytest.fail("Inactive successor or live policy was evaluated during the backup")


@pytest.mark.parametrize(
    "kind,weight", [("bcq", 0.0), ("bcq", 0.75), ("bcq", 1.0), ("bear", 1.0)]
)
def test_critic_gradient_recovers_per_transition_sampled_target(
    kind, weight, monkeypatch
):
    model = learner(kind, **({"lmbda": weight} if kind == "bcq" else {}))
    states = torch.arange(3.0).reshape(-1, 1)
    successors = torch.tensor([[3.0], [math.nan], [-4.0]])
    rewards = torch.tensor([-7.0, 13.0, -2.0])
    dones = torch.tensor([False, True, False])
    offsets = [-2.0, 0.0, 1.0]

    def propose(s, count):
        assert count == 3
        assert s.flatten().tolist() == [3.0, -4.0]
        return (s + torch.tensor(offsets)).unsqueeze(-1)

    if kind == "bcq":
        monkeypatch.setattr(model.vae, "sample", propose)
        monkeypatch.setattr(
            model.perturbation_target, "forward", lambda s, a: a + 0.25 * s
        )
        monkeypatch.setattr(model.perturbation, "forward", fail_if_called)
    else:
        monkeypatch.setattr(
            model.actor_target, "sample", lambda s, n: (propose(s, n), None)
        )
        monkeypatch.setattr(model.actor, "sample", fail_if_called)
    monkeypatch.setattr(model.q1_target, "forward", lambda s, a: s + 2 * a - 10)
    monkeypatch.setattr(model.q2_target, "forward", lambda s, a: -s - a - 8)
    # Treat each current prediction as a separate coordinate. The loss derivative
    # recovers every target, so a row swap cannot hide behind a correct batch mean.
    predictions = torch.zeros(3, 1, requires_grad=True)
    monkeypatch.setattr(
        model.q1, "forward", lambda s, _: predictions[s.flatten().long()]
    )
    loss, _ = model._compute_q_loss(states, states, rewards, successors, dones)
    (gradient,) = torch.autograd.grad(loss, predictions)

    expected = []
    for r, terminal, row in zip(
        rewards.tolist(), dones.tolist(), successors.tolist(), strict=True
    ):
        if terminal:
            expected.append(r)
            continue
        s = row[0]
        scores = []
        for offset in offsets:
            a = s + offset + (0.25 * s if kind == "bcq" else 0)
            lo, hi = sorted((s + 2 * a - 10, -s - a - 8))
            scores.append(weight * lo + (1 - weight) * hi)
        expected.append(r + 0.5 * max(scores))
    torch.testing.assert_close(-1.5 * gradient.flatten(), torch.tensor(expected))


@pytest.mark.parametrize("gamma,done", [(0.0, False), (0.5, True)])
def test_inactive_backup_skips_policy_critics_and_rng(gamma, done):
    reward = torch.tensor([3.0], requires_grad=True)
    rng = torch.random.get_rng_state()
    target = sampled_twin_target(
        reward,
        torch.tensor([done]),
        torch.tensor([[math.nan]]),
        gamma,
        fail_if_called,
        fail_if_called,
        fail_if_called,
    )
    assert not target.requires_grad
    assert target.tolist() == [[3.0]]
    assert torch.equal(rng, torch.random.get_rng_state())
    future = torch.tensor([math.nan], requires_grad=True)
    target = bellman_target(reward, torch.tensor([done]), future, gamma)
    assert target.item() == 3.0
    target.backward()
    assert future.grad.item() == 0 and reward.grad.item() == 1


def batch():
    s = torch.tensor([[-2.0], [0.5], [2.0]])
    return s, s / 3, torch.tensor([1.0, -2.0, 4.0]), s + 0.25, torch.tensor([0, 1, 0])


def assert_equal_state(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal_state(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        for x, y in zip(a, b, strict=True):
            assert_equal_state(x, y)
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    else:
        assert a == b


@pytest.mark.parametrize("kind", ["bcq", "bear"])
def test_target_schedule_and_resumed_adam_trajectory(kind, tmp_path):
    torch.manual_seed(33)
    model = learner(kind)
    names = [policy(model), "q1", "q2"]
    opt_names = [key for key in vars(model) if key.endswith("_optimizer")]
    inputs = batch()
    for step in range(3):
        prior = {
            n: [p.detach().clone() for p in getattr(model, n + "_target").parameters()]
            for n in names
        }
        model.train_step(*inputs)
        for name in names:
            target = getattr(model, name + "_target")
            for old, new, actual in zip(
                prior[name],
                getattr(model, name).parameters(),
                target.parameters(),
                strict=True,
            ):
                expected = (
                    0.75 * old.double() + 0.25 * new.detach().double()
                    if step % 2 == 0
                    else old
                )
                assert torch.equal(actual, expected.to(actual.dtype))
                assert not actual.requires_grad and actual.data_ptr() != new.data_ptr()
    file = tmp_path / "checkpoint.pt"
    model.save(str(file))
    other_config = (
        {"phi": 0.9}
        if kind == "bcq"
        else {"log_std_max": -12, "kernel_type": "gaussian"}
    )
    resumed = learner(kind, activation="gelu", **other_config)
    resumed.load(str(file))
    if kind == "bear":
        assert resumed.alpha_optimizer.param_groups[0]["params"][0] is resumed.log_alpha
    for seed in [27, 91]:
        torch.manual_seed(seed)
        metrics = model.train_step(*inputs)
        torch.manual_seed(seed)
        assert metrics == resumed.train_step(*inputs)
        model.save(str(file))
        original = torch.load(file, weights_only=True)
        resumed.save(str(file))
        replay = torch.load(file, weights_only=True)
        assert_equal_state(original, replay)
        assert all(original[n]["state"] for n in opt_names)


@pytest.mark.parametrize("kind", ["bcq", "bear"])
def test_legacy_load_resets_moments_and_initializes_target_policy(kind, tmp_path):
    original = learner(kind)
    original.train_step(*batch())
    path = tmp_path / "weights.pt"
    original.save(str(path))
    saved = torch.load(path, weights_only=True)
    del saved[policy(original) + "_target_state_dict"]
    for key in list(saved):
        if key.endswith("_optimizer"):
            del saved[key]
    torch.save(saved, path)
    restored = learner(kind, learning_rate=0.9)
    restored.train_step(*batch())
    restored.load(str(path))
    assert restored.q_optimizer.param_groups[0]["lr"] == original.config.learning_rate
    assert all(
        not opt.state for k, opt in vars(restored).items() if k.endswith("_optimizer")
    )
    assert_equal_state(
        getattr(restored, policy(original)).state_dict(),
        getattr(restored, policy(original) + "_target").state_dict(),
    )
    restored.train_step(*batch())
    if kind == "bear":
        assert (
            restored.alpha_optimizer.param_groups[0]["params"][0] is restored.log_alpha
        )
        assert restored.log_alpha.item() != original.log_alpha.item()


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_interpolation_extremes_against_exact_rationals(dtype, weight):
    big = float(torch.finfo(dtype).max)
    tiny = torch.nextafter(
        torch.tensor(0, dtype=dtype), torch.tensor(1, dtype=dtype)
    ).item()
    near = torch.nextafter(
        torch.tensor(big, dtype=dtype), torch.tensor(0, dtype=dtype)
    ).item()
    old = torch.tensor([big, -big, tiny, 2 * tiny, -tiny, big], dtype=dtype)
    new = torch.tensor([-big, near, tiny, tiny, tiny, big], dtype=dtype)
    w = Fraction(weight)
    expected = torch.tensor(
        [
            float((1 - w) * Fraction(a) + w * Fraction(b))
            for a, b in zip(old.tolist(), new.tolist(), strict=True)
        ],
        dtype=dtype,
    )
    assert torch.equal(_mix(old, new, weight), expected)


@pytest.mark.parametrize("kind", ["bcq", "bear"])
@pytest.mark.parametrize(
    "setting",
    [
        {"discount_factor": float("nan")},
        {"soft_target_tau": -0.1},
        {"target_update_frequency": 0},
        {"target_update_frequency": 1.5},
    ],
)
def test_invalid_backup_configuration(kind, setting):
    with pytest.raises(ValueError):
        learner(kind, **setting)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("scores", [(-4, -4, 4), (-4, 1, 4)])
def test_expected_finite_proposal_gap(k, scores):
    from itertools import product

    probabilities = (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4))
    draws = list(product(range(3), repeat=k))
    actions = (
        torch.tensor([[scores[a] for a in seq] for seq in draws]).float().unsqueeze(-1)
    )
    zeros = torch.zeros(len(draws), 1)
    result = sampled_twin_target(
        zeros,
        zeros.bool(),
        zeros,
        1.0,
        lambda _: actions,
        lambda s, a: a,
        lambda s, a: a + 1,
    )
    gap = sum(
        math.prod(probabilities[a] for a in seq) * (4 - Fraction(value))
        for seq, value in zip(draws, result.flatten().tolist(), strict=True)
    )
    bound = 8 * Fraction(3, 4) ** k
    assert 0 <= gap <= bound
    assert (gap == bound) == (scores[1] == -4)


@pytest.mark.parametrize("kind", ["cql", "iql", "bcq", "bear"])
def test_all_target_networks_preserve_equal_subnormal_parameters(kind):
    from stateset_agents.training.offline_rl_algorithms import (
        ConservativeQLearning,
        CQLConfig,
        ImplicitQLearning,
        IQLConfig,
    )

    if kind == "cql":
        model = ConservativeQLearning(
            1, 1, CQLConfig(hidden_size=4, soft_target_tau=0.5), "cpu"
        )
        names = ["q1", "q2"]
        update = model._soft_update_target_networks
    elif kind == "iql":
        model = ImplicitQLearning(
            1, 1, IQLConfig(hidden_size=4, soft_target_tau=0.5), "cpu"
        )
        pairs = [(model.value_target, model.value_net)]
        update = model._soft_update_target_network
    else:
        model = learner(kind, soft_target_tau=0.5)
        names = ["q1", "q2", policy(model)]
        update = model._soft_update_targets
    if kind != "iql":
        pairs = [
            (getattr(model, name + "_target"), getattr(model, name)) for name in names
        ]
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)).item()
    with torch.no_grad():
        for target, live in pairs:
            for net in (target, live):
                for p in net.parameters():
                    p.fill_(tiny)
    update()
    assert all((p == tiny).all() for target, _ in pairs for p in target.parameters())


@pytest.mark.parametrize(
    "bad", ["missing_axis", "wrong_batch", "empty", "critic_shape"]
)
def test_sampled_backup_rejects_misaligned_candidates(bad):
    s = torch.zeros(2, 1)
    proposals = {
        "missing_axis": (2, 3),
        "wrong_batch": (3, 2, 1),
        "empty": (2, 0, 1),
        "critic_shape": (2, 3, 1),
    }

    def sample(_):
        return torch.zeros(proposals[bad])

    def critic(s, a):
        return a.flatten() if bad == "critic_shape" else a

    with pytest.raises(ValueError, match="proposal"):
        sampled_twin_target(s, s, s, 0.5, sample, critic, critic)


@pytest.mark.parametrize("kind", ["bcq", "bear"])
def test_real_backups_use_lagged_policy_with_fixed_draws(kind):
    torch.manual_seed(112)
    model = learner(kind)
    model.q1_target.forward = model.q2_target.forward = lambda s, a: 2 * a
    inputs = batch()

    def loss():
        torch.manual_seed(41)
        return model._compute_q_loss(*inputs)[0].item()

    initial = loss()
    target_name = policy(model)
    with torch.no_grad():
        for p in getattr(model, target_name).parameters():
            p.add_(0.7)
    assert loss() == initial
    getattr(model, target_name + "_target").load_state_dict(
        getattr(model, target_name).state_dict()
    )
    assert loss() != initial


@pytest.mark.parametrize("step", [0, 1, 2])
def test_bear_warmup_actor_gradient(step):
    model = learner("bear", warmup_steps=1)
    model.training_step = step
    s, a, *_ = batch()
    torch.manual_seed(72)
    loss, mmd, _ = model._compute_actor_loss(s, a)
    (actual,) = torch.autograd.grad(loss.sum(), model.actor.mean_head.bias)
    torch.manual_seed(72)
    candidates, _ = model.actor.sample(s, num_samples=3)
    rows = []
    for state, actions, observed in zip(s, candidates, a, strict=True):
        scores = [
            torch.minimum(
                model.q1(state[None], action[None]), model.q2(state[None], action[None])
            )
            for action in actions
        ]
        row = -torch.stack(scores).mean()
        if step >= 1:
            row = row + model.log_alpha.detach().exp() * model.mmd_kernel.compute_mmd(
                actions[None], observed.view(1, 1, 1)
            )
        rows.append(row)
    (expected,) = torch.autograd.grad(
        torch.stack([r.reshape(()) for r in rows]).mean(), model.actor.mean_head.bias
    )
    torch.testing.assert_close(actual, expected)
    assert mmd.item() > 0


@pytest.mark.parametrize("kind", ["bcq", "bear"])
@pytest.mark.parametrize("samples", [1, 3, 7])
def test_gaussian_proposals_miss_threshold_with_predicted_frequency(kind, samples):
    count = 8192
    torch.manual_seed(130)
    setting = "num_action_samples" if kind == "bcq" else "num_samples_for_mmd"
    model = learner(kind, discount_factor=1.0, **{setting: samples})
    with torch.no_grad():
        if kind == "bear":
            for param in model.actor_target.parameters():
                param.zero_()  # Normal(0,1) at every successor.
        else:
            for net in (model.vae, model.perturbation_target):
                for param in net.parameters():
                    param.zero_()
            # VAE output = ReLU(z) - ReLU(-z) for its first normal coordinate.
            first, _, last = model.vae.decoder
            first.weight[0, 1] = 1
            first.weight[1, 1] = -1
            last.weight[0, :2] = torch.tensor([1.0, -1.0])
    model.q1_target.forward = model.q2_target.forward = lambda s, a: a
    predictions = torch.zeros(count, 1, requires_grad=True)
    model.q1.forward = lambda s, a: predictions
    zeros = torch.zeros_like(predictions)
    loss, _ = model._compute_q_loss(zeros, zeros, zeros, zeros, zeros.bool())
    (gradient,) = torch.autograd.grad(loss, predictions)
    draws = -gradient.flatten() * (count / 2)
    # Strict threshold handles ties like the Lean event. A continuous N(0,1)
    # score has CDF 1/2 at zero, so each row misses with probability 2^-K.
    miss = 0.5**samples
    observed = (draws < 0).sum().item()
    assert abs(observed - count * miss) <= 6 * math.sqrt(count * miss * (1 - miss))


@pytest.mark.parametrize("kind", ["cql", "iql", "bcq", "bear"])
def test_twin_critic_step_clips_the_configured_critic_group(kind):
    from stateset_agents.training.offline_rl_algorithms import (
        ConservativeQLearning,
        CQLConfig,
        ImplicitQLearning,
        IQLConfig,
    )

    torch.manual_seed(146)
    common = {"hidden_size": 4, "num_layers": 2, "max_grad_norm": 1.0}
    if kind == "cql":
        model = ConservativeQLearning(1, 1, CQLConfig(**common), "cpu")
    elif kind == "iql":
        model = ImplicitQLearning(1, 1, IQLConfig(**common), "cpu")
    else:
        model = learner(kind, max_grad_norm=1.0)

    q1, q2 = model.q1, model.q2
    if kind == "cql":
        model.q1_optimizer = torch.optim.SGD(q1.parameters(), lr=1)
        model.q2_optimizer = torch.optim.SGD(q2.parameters(), lr=1)
        expected = [1.0, -1.0]
    else:
        model.q_optimizer = torch.optim.SGD([*q1.parameters(), *q2.parameters()], lr=1)
        expected = [3 / 5, -4 / 5]

    bias1, bias2 = (list(q.parameters())[-1] for q in (q1, q2))
    old1, old2 = bias1.item(), bias2.item()
    h1 = bias1.register_hook(lambda g: torch.full_like(g, 3e20))
    h2 = bias2.register_hook(lambda g: torch.full_like(g, -4e20))
    s = torch.tensor([[0.0], [1.0]])
    try:
        model.train_step(s, s, torch.ones_like(s), s, torch.ones_like(s))
    finally:
        h1.remove()
        h2.remove()
    assert [old1 - bias1.item(), old2 - bias2.item()] == pytest.approx(expected)
    assert model.training_step == 1
