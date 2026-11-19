"""Independent references for PPO rollout reuse, stopping and critic targets."""

import json
import math
import statistics
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core import rust_accelerator  # noqa: E402
from stateset_agents.core.enhanced.advanced_rl_algorithms import (  # noqa: E402
    _rollout_targets,
)
from stateset_agents.core.value_function import ValueFunction  # noqa: E402
from stateset_agents.training import ppo_online  # noqa: E402
from stateset_agents.training.advantage_estimation import (  # noqa: E402
    normalize_advantages,
)
from stateset_agents.training.ppo_trainer import (  # noqa: E402
    _behavior_kl,
    _categorical_log_probs,
    compute_gae,
)
from stateset_agents.training.vapo_trainer import LengthAdaptiveGAE  # noqa: E402
from tests.unit.test_ppo_online import NonRoundTripTokenizer  # noqa: E402
from tests.unit.test_ppo_safety import trainer as trainer_fixture  # noqa: E402


def forward_targets(rewards, values, gamma, lam, dones, bootstrap=0):
    """Expand the forward TD-residual sum using exact binary rationals."""
    rewards = list(map(Fraction, rewards))
    v = [*map(Fraction, values), Fraction(bootstrap)]
    g, decay = Fraction(gamma), Fraction(lam)
    adv, targets = [], []
    for start in range(len(rewards)):
        a = Fraction(0)
        for i in range(start, len(rewards)):
            delta = rewards[i] - v[i] + (0 if dones[i] else g * v[i + 1])
            a += (g * decay) ** (i - start) * delta
            if dones[i]:
                break
        adv.append(float(a))
        targets.append(float(a + v[start]))
    return adv, targets


@pytest.mark.parametrize("gamma,lam", list(product((0.0, 0.5, 1.0), (0.0, 0.25, 1.0))))
def test_reward_and_value_traces_match_forward_rational_sums(gamma, lam, monkeypatch):
    # Reward remains visible in the target with a huge baseline. It remains
    # visible in the advantage too when a huge successor offsets that baseline.
    rewards = [1.0, -2.0, 0.5, 5.0]
    values = [2.0**85, 2.0**85, -(2.0**85), 2.0**86]
    r, v = (torch.tensor([x], dtype=torch.float64) for x in (rewards, values))
    adv, ret = compute_gae(r, v, gamma, lam)
    expected = forward_targets(rewards, values, gamma, lam, [False] * 4)
    assert (adv[0].tolist(), ret[0].tolist()) == expected
    for native in (rust_accelerator.is_rust_available(), False):
        monkeypatch.setattr(rust_accelerator, "_RUST_AVAILABLE", native)
        av, rv = rust_accelerator.compute_gae_targets(
            np.array(rewards), np.array([*values, 8.0]), gamma, lam
        )
        assert (av.tolist(), rv.tolist()) == forward_targets(
            rewards, values, gamma, lam, [False] * 4, 8.0
        )
        for dones in ([False, True, False, False], [True, False, False, True]):
            expected = forward_targets(rewards, values, gamma, lam, dones)
            at, rt = LengthAdaptiveGAE(gamma).compute_gae(
                r, v, torch.tensor([dones]), lam
            )
            assert (at[0].tolist(), rt[0].tolist()) == expected
    adv1, ret1 = _rollout_targets(
        rewards, values, [False, False, False, True], gamma, lam, {1: 4.0}
    )
    episodes = [
        forward_targets(rewards[:2], values[:2], gamma, lam, [False, False], 4.0),
        forward_targets(rewards[2:], values[2:], gamma, lam, [False, True]),
    ]
    assert (adv1, ret1) == tuple(x + y for x, y in zip(*episodes, strict=True))


def test_subnormal_value_trace_and_adaptive_lambda(monkeypatch):
    tiny = math.ulp(0.0)
    r = torch.zeros((1, 3), dtype=torch.float64)
    v = torch.tensor([[0, tiny, 2 * tiny]], dtype=torch.float64)
    assert compute_gae(r, v, 1.0, 0.5)[1][0, 0].item() == tiny
    for native in (rust_accelerator.is_rust_available(), False):
        monkeypatch.setattr(rust_accelerator, "_RUST_AVAILABLE", native)
        assert (
            rust_accelerator.compute_gae_targets(
                np.zeros(1), np.array([0.0, tiny]), 1.0, 0.5
            )[1].item()
            == tiny
        )
    for alpha, length in ((1e-30, 5), (0.25, 3), (1e308, 2)):
        scaled = Fraction(alpha) * length
        assert LengthAdaptiveGAE(lambda_policy_alpha=alpha).compute_lambda_policy(
            length
        ) == float(scaled / (1 + scaled))


def test_rust_batch_strides_and_row_counts():
    core = pytest.importorskip("stateset_rl_core")
    if not hasattr(core, "compute_gae_targets"):
        pytest.skip("Paired GAE bindings require the updated native extension")
    r = np.arange(8, dtype=np.float64)[::2]
    v = np.arange(8, dtype=np.float64)[1::2]
    assert (
        core.batch_compute_gae([r], [v], 1.0, 1.0)[0].tolist()
        == forward_targets(r, v, 1.0, 1.0, [False] * 4)[0]
    )
    for f in [core.batch_compute_gae, rust_accelerator.batch_compute_gae]:
        with pytest.raises(ValueError, match="rows must align"):
            f([r], [])


def test_undefined_successor_cannot_cross_terminal(monkeypatch):
    r = torch.tensor([[1.0, float("nan")]])
    v = torch.tensor([[2.0**40, float("nan")]])
    for native in (rust_accelerator.is_rust_available(), False):
        monkeypatch.setattr(rust_accelerator, "_RUST_AVAILABLE", native)
        adv, ret = LengthAdaptiveGAE().compute_gae(
            r, v, torch.tensor([[True, False]]), 0.75
        )
        assert adv[0, 0].item() == torch.tensor(1 - 2.0**40).item()
        assert ret[0, 0].item() == 1.0


def test_critic_mask_and_frozen_targets(monkeypatch):
    learner = trainer_fixture.__wrapped__(monkeypatch)
    v = torch.tensor([[3.0, math.nan, 3.0], [8.0, 4.0, math.nan]], requires_grad=True)
    old = torch.tensor([[3.0, math.nan, 2.0], [4.0, 4.0, math.nan]], requires_grad=True)
    target = torch.tensor(
        [[5.0, math.nan, 4.0], [3.0, 4.0, math.nan]], requires_grad=True
    )
    m = target.detach().isfinite().float()
    learner.config.value_clip_eps = 0.5
    loss = learner.value_loss(v, old, target, m)
    assert loss.item() == pytest.approx((4 + 2.25 + 25) / 8)
    loss.backward()
    torch.testing.assert_close(
        v.grad, torch.tensor([[-0.5, 0.0, 0.0], [1.25, 0.0, 0.0]])
    )
    assert old.grad is target.grad is None
    adv, ret = learner.compute_episode_advantages(
        torch.tensor([1.0]),
        torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([[math.nan, 1e8, math.nan]]),
    )
    assert ret.tolist() == [[0.0, 1.0, 0.0]]
    assert adv[0, [0, 2]].tolist() == [0.0, 0.0]
    # Half of the squared error is finite in float32 even though the square overflows.
    val = torch.tensor([[2e19]])
    out = learner.value_loss(val, val, torch.zeros_like(val))
    assert out.item() == pytest.approx(0.5 * val.item() ** 2, rel=2e-7)


@pytest.mark.parametrize(
    "deltas", [[1e-12, -2e-12], [0.0, 0.0], [-40.0, 3.0], [710.0, -0.5, 0.0, 0.0]]
)
def test_drift_has_behavior_orientation_and_action_denominator(deltas):
    gaps = torch.tensor([deltas + [math.nan]], dtype=torch.float64)
    old = -gaps.clamp_min(0)
    cur = gaps.clamp_max(0)
    with localcontext() as ctx:
        ctx.prec = 90
        exact = sum(x.exp() - 1 - x for x in map(Decimal.from_float, deltas)) / len(
            deltas
        )
    assert _behavior_kl(cur, old, torch.isfinite(gaps)) == pytest.approx(
        float(exact), rel=2e-13, abs=0
    )


def test_mean_drift_can_be_finite_when_its_sum_overflows():
    cur = torch.full((2,), -1e308, dtype=torch.float64)
    assert _behavior_kl(cur, torch.zeros_like(cur), torch.ones_like(cur)) == 1e308


@pytest.mark.parametrize("dtype,gap", [(torch.float32, 89.0), (torch.float64, 710.0)])
@pytest.mark.parametrize("masked", [False, True])
def test_large_reference_gap_can_have_finite_mean_and_gradient(
    monkeypatch, dtype, gap, masked
):
    learner = trainer_fixture.__wrapped__(monkeypatch)
    # One selected action carries half of the batch mass either way. The
    # masked variant also includes an entirely padded response row.
    values = [[-gap, math.nan], [math.nan, math.nan]] if masked else [[-gap, 0.0]]
    current = torch.tensor(values, dtype=dtype, requires_grad=True)
    mask = torch.isfinite(current) if masked else None
    loss = learner.compute_kl_divergence(current, torch.zeros_like(current), mask)
    with localcontext() as ctx:
        ctx.prec = 100
        exp = Decimal.from_float(gap).exp()
        expected = float((exp - 1 - Decimal.from_float(gap)) / 2)
        slope = float((1 - exp) / 2)
    assert loss.item() == pytest.approx(
        expected, rel=2e-7 if dtype == torch.float32 else 2e-13
    )
    grad = torch.autograd.grad(loss, current)[0]
    assert grad[0, 0].item() == pytest.approx(
        slope, rel=2e-7 if dtype == torch.float32 else 2e-13
    )
    assert grad.flatten()[1:].count_nonzero() == 0


def test_maximum_finite_drift_preserves_equal_weight_gradient(monkeypatch):
    learner = trainer_fixture.__wrapped__(monkeypatch)
    maximum = torch.finfo(torch.float64).max
    current = torch.full((22,), -maximum, dtype=torch.float64, requires_grad=True)
    assert (
        _behavior_kl(current, torch.zeros_like(current), torch.ones_like(current))
        == maximum
    )
    # Reference gap is negative. Its k3 slope with respect to current is +1.
    live = torch.zeros_like(current, requires_grad=True)
    loss = learner.compute_kl_divergence(live, current)
    assert loss.item() == maximum
    torch.testing.assert_close(
        torch.autograd.grad(loss, live)[0], torch.full_like(live, 1 / 22)
    )


def test_half_square_and_mean_precede_critic_overflow(monkeypatch):
    learner = trainer_fixture.__wrapped__(monkeypatch)
    v = torch.tensor([[1.5e154, 0.0]], dtype=torch.float64, requires_grad=True)
    loss = learner.value_loss(v, v, torch.zeros_like(v), torch.ones_like(v))
    expected = Fraction(v[0, 0].item()) ** 2 / 4
    assert loss.item() == pytest.approx(float(expected), rel=2e-15)
    loss.backward()
    torch.testing.assert_close(v.grad, v.detach() / 2)


def test_integer_rewards_and_masks_retain_fractional_targets(monkeypatch):
    r = torch.tensor([[0, 1]])
    v = torch.zeros_like(r)
    assert compute_gae(r, v, 0.5, 1.0)[0].tolist() == [[0.5, 1.0]]
    for native in (rust_accelerator.is_rust_available(), False):
        monkeypatch.setattr(rust_accelerator, "_RUST_AVAILABLE", native)
        actor, target = LengthAdaptiveGAE(0.5).compute_gae(
            r, v, torch.tensor([[False, True]]), 1.0
        )
        assert actor.tolist() == target.tolist() == [[0.5, 1.0]]
    learner = trainer_fixture.__wrapped__(monkeypatch)
    _, returns = learner.compute_episode_advantages(
        torch.tensor([1.25]), torch.tensor([[True]]), torch.zeros((1, 1))
    )
    assert returns.item() == 1.25


def test_small_trace_coefficient_preserves_a_large_continuation(monkeypatch):
    rewards = [0.0, 1e300]
    gamma = lam = 1e-200
    expected = float(Fraction(gamma) * Fraction(lam) * Fraction(rewards[1]))
    row = torch.tensor([rewards], dtype=torch.float64)
    targets = compute_gae(row, torch.zeros_like(row), gamma, lam)[1]
    assert targets[0, 0].item() == pytest.approx(expected, rel=1e-15, abs=0)
    for native in (rust_accelerator.is_rust_available(), False):
        monkeypatch.setattr(rust_accelerator, "_RUST_AVAILABLE", native)
        targets = rust_accelerator.compute_gae_targets(
            np.array(rewards), np.zeros(2), gamma, lam
        )[1]
        assert targets[0] == pytest.approx(expected, rel=1e-15, abs=0)


def test_value_function_honors_internal_terminals_and_keeps_critic_reward():
    estimator = SimpleNamespace(gamma=1.0, gae_lambda=0.75, normalize_advantages=False)
    v = torch.tensor([1e8, -1e8, float("nan")])
    advantages, returns = ValueFunction.compute_gae(
        estimator, [1.0, 2.0, 0.0], v, [True, True, False]
    )
    assert returns[:2].tolist() == [1.0, 2.0]
    assert advantages[:2].tolist() == [-1e8, 1e8]


@pytest.mark.parametrize(
    "values",
    [[1e308, -1e308, 0.0], [1e308, math.nextafter(1e308, 0.0), 1e308], [8.0, 8.0, 8.0]],
)
def test_standardization_matches_decimal_sample_deviation(values):
    with localcontext() as ctx:
        ctx.prec = 100
        dec = list(map(Decimal.from_float, values))
        mean = sum(dec) / len(dec)
        dev = (sum((x - mean) ** 2 for x in dec) / (len(dec) - 1)).sqrt()
        expected = [float((x - mean) / (dev + Decimal.from_float(1e-8))) for x in dec]
    x = torch.tensor(values + [math.nan], dtype=torch.float64)
    result = normalize_advantages(x, torch.isfinite(x))
    assert result.tolist() == pytest.approx(expected + [0.0], abs=2e-15)


def test_reference_penalty_uses_frozen_draws_after_policy_change(monkeypatch):
    learner = trainer_fixture.__wrapped__(monkeypatch)
    behavior = [0.125, 0.25, 0.625]
    current = [0.5, 0.375, 0.125]
    reference = [0.25, 0.25, 0.5]
    logits = torch.tensor(
        list(map(math.log, current)), dtype=torch.float64, requires_grad=True
    )
    scores = logits.log_softmax(0)
    reference_scores = torch.tensor(list(map(math.log, reference)), dtype=logits.dtype)
    # Enumerate draws from the frozen policy at this prefix.
    loss = sum(
        mu
        * learner.compute_kl_divergence(scores[i : i + 1], reference_scores[i : i + 1])
        for i, mu in enumerate(behavior)
    )
    expected = sum(
        mu * (q / p - 1 - math.log(q / p))
        for mu, p, q in zip(behavior, current, reference, strict=True)
    )
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    slopes = [
        sum(
            mu * (1 - q / p) * (float(i == j) - current[j])
            for i, (mu, p, q) in enumerate(
                zip(behavior, current, reference, strict=True)
            )
        )
        for j in range(3)
    ]
    torch.testing.assert_close(logits.grad, torch.tensor(slopes, dtype=logits.dtype))


@pytest.mark.parametrize("temperature", [0.5, 1e-60, 1e40])
def test_scores_invariant_to_common_large_logit(temperature):
    m = torch.finfo(torch.float32).max
    scores = _categorical_log_probs(torch.tensor([m, m]), temperature)
    torch.testing.assert_close(scores, torch.full_like(scores, -math.log(2)))
    # Wide, opposite logits can be subtracted safely after temperature scaling.
    scores = _categorical_log_probs(torch.tensor([m, -m]), max(temperature, 16.0))
    assert scores[1].item() - scores[0].item() == pytest.approx(
        -2 * m / max(temperature, 16.0), rel=1e-6
    )


@pytest.mark.parametrize(
    "dtype, distance", [(torch.float32, 20.0), (torch.float64, 50.0)]
)
def test_dominant_action_score_retains_both_logit_derivatives(dtype, distance):
    temperature = 0.5
    logits = torch.tensor([0, -distance * temperature], dtype=dtype, requires_grad=True)
    head = _categorical_log_probs(logits, temperature)[0]
    tail = math.exp(-distance)
    slope = tail / ((1 + tail) * temperature)
    assert head.item() == pytest.approx(
        -math.log1p(tail), rel=4 * torch.finfo(dtype).eps, abs=0
    )
    gradient = torch.autograd.grad(head, logits)[0]
    torch.testing.assert_close(
        gradient,
        torch.tensor([slope, -slope], dtype=dtype),
        rtol=4 * torch.finfo(dtype).eps,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("gap", [1e-9, -1e-9, 1e-16, -1e-16])
def test_nearly_equal_reference_penalty_and_gradient(monkeypatch, dtype, gap):
    learner = trainer_fixture.__wrapped__(monkeypatch)
    current = torch.tensor([[-2 * abs(gap)]], dtype=dtype, requires_grad=True)
    reference = current.detach() + gap
    loss = learner.compute_kl_divergence(current, reference, torch.ones_like(current))
    with localcontext() as ctx:
        ctx.prec = 100
        d = Decimal.from_float(reference.item()) - Decimal.from_float(current.item())
        expected = d.exp() - 1 - d
        slope = 1 - d.exp()
    tol = 4 * torch.finfo(dtype).eps
    assert loss.item() == pytest.approx(float(expected), rel=tol, abs=0)
    assert torch.autograd.grad(loss, current)[0].item() == pytest.approx(
        float(slope), rel=tol, abs=0
    )


@pytest.mark.parametrize("batch", [1, 2, 3])
@pytest.mark.parametrize("h", [1, 2, 3])
def test_stopped_path_drift_and_native_admission(monkeypatch, batch, h):
    learner = trainer_fixture.__wrapped__(monkeypatch)
    learner.config.target_kl = 0.2
    q = Fraction(h, 4)
    root = torch.tensor([math.log(1 - q), math.log(q)], dtype=torch.float64)
    child = torch.tensor([math.log(0.75), math.log(0.25)], dtype=torch.float64)

    def forward(input_ids, attention_mask, output_hidden_states=False):
        logits = torch.where((input_ids == 1).unsqueeze(-1), child, root)
        return SimpleNamespace(
            logits=logits + 0 * learner.model.projection.bias[0],
            hidden_states=(learner.model.embedding(input_ids),),
        )

    monkeypatch.setattr(learner.model, "forward", forward)
    # The root is unchanged. Use the same representation for its behavior and
    # current scores so that the model's zero-cap equality is tested exactly.
    root_scores = _categorical_log_probs(root, 1).tolist()
    paths = [
        ([0], [root_scores[0]], 1 - q),
        ([1, 0], [root_scores[1], math.log(0.25)], q / 4),
        ([1, 1], [root_scores[1], math.log(0.75)], 3 * q / 4),
    ]
    mass = missed = Fraction(0)
    token_mean = path_sum = correction = 0.0
    reward_drop = 0.0
    conditional = math.log(3) / 2
    for rows in product(paths, repeat=batch):
        packed = ppo_online._pack(
            learner,
            [
                {
                    "prompt_ids": [3],
                    "action_ids": ids,
                    "behavior_log_probs": old,
                    "behavior_values": [0] * len(ids),
                    "reward": float(ids == [1, 1]),
                }
                for ids, old, _ in rows
            ],
        )
        mask = packed["response_mask"][:, 1:]
        with torch.no_grad():
            current, _ = learner.compute_log_probs(
                packed["input_ids"], packed["attention_mask"]
            )
        measured = _behavior_kl(current, packed["old_log_probs"], mask)
        rare = sum(ids == [1, 0] for ids, _, _ in rows)
        common = sum(ids == [1, 1] for ids, _, _ in rows)
        length = batch + rare + common
        expected = (rare * (2 - math.log(3)) + common * (math.log(3) - 2 / 3)) / length
        assert measured == pytest.approx(expected, abs=1e-15)
        for cap in (0, 100, 200, 350, 600, 750, 900, 1200, 1500, 1800):
            assert (measured <= cap / 3000) == (
                2704 * rare + 1296 * common <= length * cap
            )
        if expected > learner.config.target_kl:
            with pytest.raises(ValueError, match="before the first update"):
                learner._update_token_batch(**packed)
        else:
            assert learner._update_token_batch(**packed)["approx_kl"] == pytest.approx(
                expected, abs=1e-15
            )
        probability = math.prod(weight for _, _, weight in rows)
        if batch == 1 and common:
            reward_drop += probability - current[mask.bool()].sum().exp().item()
        mass += probability
        if not (rare or common):
            missed += probability
        x = Fraction(rare + common, batch)
        correction += (
            probability * conditional * (x - q) ** 2 / ((1 + x) * (1 + q) ** 2)
        )
        token_mean += probability * measured
        path_sum += probability * measured * length / batch
    assert mass == 1 and missed == (1 - q) ** batch
    assert path_sum == pytest.approx(q * conditional)
    assert q * conditional / (1 + q) - token_mean == pytest.approx(correction)
    if batch == 1:
        assert token_mean == pytest.approx(q * conditional / 2)
        assert reward_drop == pytest.approx(q / 2)


def test_stopped_drift_omits_target_mass_outside_a_visited_prefix():
    # Behavior stops with probability 3/4. When it continues, only its first
    # child can be drawn. The target puts one quarter on the unseen child.
    paths = [
        (0.75, [0.75], [0.5]),
        (0.25, [0.25, 1.0], [0.5, 0.75]),
    ]
    sampled_sum = path_kl = 0.0
    for probability, old, current in paths:
        a, b = [torch.tensor(row, dtype=torch.float64).log() for row in (current, old)]
        sampled_sum += probability * len(old) * _behavior_kl(a, b, torch.ones_like(a))
        path_kl += probability * math.log(probability / math.prod(current))
    assert path_kl - sampled_sum == pytest.approx(1 / 16, abs=2e-15)


def test_epoch_runs_at_the_measured_kl_boundary(monkeypatch):
    learner = trainer_fixture.__wrapped__(monkeypatch)
    ids = torch.tensor([[3, 4]])
    attention = torch.ones_like(ids)
    response = torch.tensor([[0.0, 1.0]])
    with torch.no_grad():
        current, _ = learner.compute_log_probs(ids, attention)
        old = current - 1.0
        learner.config.target_kl = _behavior_kl(current, old, response[:, 1:])
    result = learner._update_token_batch(
        ids,
        attention,
        response,
        torch.ones(1),
        old_log_probs=old,
        old_values=torch.zeros_like(old),
    )
    assert result["ppo_epochs"] == 1 and result["approx_kl"] == learner.config.target_kl


async def test_stale_first_epoch_cannot_advance_learner(monkeypatch):
    m = trainer_fixture.__wrapped__(monkeypatch)
    m.config.target_kl = 1e-4
    ids = torch.tensor([[3, 4]])
    with pytest.raises(ValueError, match="before the first update"):
        m._update_token_batch(
            ids,
            torch.ones_like(ids),
            torch.tensor([[0.0, 1.0]]),
            torch.ones(1),
            old_log_probs=torch.tensor([[-5.0]]),
            old_values=torch.zeros(1, 1),
        )
    assert (
        m.global_step == 0
        and not m.policy_optimizer.state
        and not m.value_optimizer.state
    )


@pytest.mark.parametrize("cap", [None, 1e-8, 0.025, 8.0])
async def test_online_inner_epochs_match_scalar_sgd(monkeypatch, tmp_path, cap):
    m = trainer_fixture.__wrapped__(monkeypatch)
    m.tokenizer = NonRoundTripTokenizer()
    m.tokenizer.eos_token_id = 1
    m.config.output_dir = str(tmp_path)
    m.config.target_kl = cap
    m.config.num_ppo_epochs = 5
    m.config.num_generations = 8
    m.config.mini_batch_size = 2
    m.config.temperature = 0.8
    m.config.entropy_coef = 0.13
    m.config.max_grad_norm = 1e6
    m.config.gamma = 0.75
    m.config.gae_lambda = 0.5
    m.config.value_clip_eps = 0.12
    m.reward_fn = lambda _, text: 2.5 if text.endswith("1") else -2.0
    logits = [-0.3, -0.1] + [-35.0] * 30
    value, frozen_value = 0.25, 0.25
    with torch.no_grad():
        for module in (m.model, m.value_head):
            for parameter in module.parameters():
                parameter.zero_()
                parameter.requires_grad_(False)
        actor = m.model.projection.bias
        actor.copy_(torch.tensor(logits))
        actor.requires_grad_(True)
        critic = m.value_head.net[-1].bias
        critic.fill_(value)
        critic.requires_grad_(True)
    lr = 0.12 if cap == 0.025 else 1.2
    m.policy_optimizer = torch.optim.SGD([actor], lr=lr)
    m.value_optimizer = torch.optim.SGD([critic], lr=lr)
    m.policy_scheduler = torch.optim.lr_scheduler.StepLR(
        m.policy_optimizer, 1, gamma=0.5
    )
    m.value_scheduler = torch.optim.lr_scheduler.StepLR(m.value_optimizer, 1, gamma=0.5)
    torch.manual_seed(190)
    result = await m.train(["ab", "c"], num_episodes=1)
    rows = json.loads(
        (Path(result["rollout_directory"]) / "rollout-0.json").read_text()
    )["rows"]
    tokens, old_scores, advantages, returns = [], [], [], []
    for row in rows:
        actions = row["action_ids"]
        n = len(actions)
        assert row["behavior_values"] == [frozen_value] * n
        a, y = forward_targets(
            [0.0] * (n - 1) + [row["reward"]],
            [frozen_value] * n,
            0.75,
            0.5,
            [False] * (n - 1) + [True],
        )
        tokens.extend(actions)
        old_scores.extend(row["behavior_log_probs"])
        advantages.extend(a)
        returns.extend(y)
    assert {0, 1} == set(tokens) and len({len(r["action_ids"]) for r in rows}) > 1
    avg, std = statistics.mean(advantages), statistics.stdev(advantages)
    advantages = [(a - avg) / (std + 1e-8) for a in advantages]
    N = len(tokens)
    policy_losses, value_losses, clipped = [], [], []
    last_kl = 0.0
    for _ in range(5):
        exps = [math.exp(logit / 0.8) for logit in logits]
        p = [e / sum(exps) for e in exps]
        gaps = [
            math.log(p[action]) - old
            for action, old in zip(tokens, old_scores, strict=True)
        ]
        drift = sum(math.expm1(d) - d for d in gaps) / N
        if cap is not None and drift > cap:
            break
        last_kl = drift
        h = -sum(prob * math.log(prob) for prob in p)
        slope = [0.13 * prob * (math.log(prob) + h) / 0.8 for prob in p]
        dv = pl = vl = outside = 0.0
        for token, d, a, y in zip(tokens, gaps, advantages, returns, strict=True):
            ratio = math.exp(d)
            clipped_ratio = min(1.2, max(0.8, ratio))
            outside += ratio < 0.8 or ratio > 1.2
            pl -= min(ratio * a, clipped_ratio * a) / N
            if not ((a > 0 and ratio > 1.2) or (a < 0 and ratio < 0.8)):
                for j in range(len(p)):
                    slope[j] -= a * ratio * ((j == token) - p[j]) / (N * 0.8)
            clip_val = max(frozen_value - 0.12, min(frozen_value + 0.12, value))
            raw, bounded = (value - y) ** 2, (clip_val - y) ** 2
            vl += 0.5 * max(raw, bounded) / N
            if raw >= bounded or abs(value - frozen_value) <= 0.12:
                dv += m.config.value_coef * (value - y) / N
        logits = [logit - lr * g for logit, g in zip(logits, slope, strict=True)]
        value -= lr * dv
        policy_losses.append(pl)
        value_losses.append(vl)
        clipped.append(outside / N)
    count = len(policy_losses)
    metrics = result["metrics"]
    assert metrics["ppo_epochs"] == count
    if cap == 1e-8:
        assert count == 1
    if cap == 0.025:
        assert 1 < count < 5
    if cap is None or cap == 8:
        assert count == 5
    if cap is not None:
        assert metrics["approx_kl"] == pytest.approx(last_kl, rel=2e-5, abs=1e-13)
    assert metrics["clip_fraction"] == pytest.approx(statistics.mean(clipped))
    assert metrics["policy_loss"] == pytest.approx(
        statistics.mean(policy_losses), abs=3e-7
    )
    assert metrics["value_loss"] == pytest.approx(
        statistics.mean(value_losses), rel=3e-6
    )
    torch.testing.assert_close(
        actor.detach(), torch.tensor(logits), rtol=2e-6, atol=2e-6
    )
    assert critic.item() == pytest.approx(value, rel=3e-6, abs=3e-6)
    assert (
        m.policy_scheduler.last_epoch
        == m.value_scheduler.last_epoch
        == m.global_step
        == 1
    )
