"""Independent scalar oracles for prompt-group baselines and their learner signal."""

import itertools
import math
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

torch = pytest.importorskip("torch")
from stateset_agents.training import objectives  # noqa: E402


def oracle(data, ids, kind, eps):
    r = list(map(Fraction, data))
    eps = Fraction(eps)
    result = []
    for i, x in enumerate(r):
        group = [v for v, gid in zip(r, ids, strict=True) if gid == ids[i]]
        if kind == "leave_one_out":
            result.append(
                float(x - (sum(group) - x) / (len(group) - 1)) if len(group) > 1 else 0
            )
            continue
        delta = x - sum(group) / len(group)
        if kind == "group_mean":
            result.append(float(delta))
            continue
        pop = r if kind == "batch_norm" else group
        # Pairwise squared differences give the population variance without a mean.
        variance = (
            sum((a - b) ** 2 for a, b in itertools.combinations(pop, 2)) / len(pop) ** 2
        )
        if variance <= eps**2:
            result.append(0.0)
        else:

            def dec(x):
                return Decimal(x.numerator) / Decimal(x.denominator)

            with localcontext() as ctx:
                ctx.prec = 110
                result.append(float(dec(delta) / (dec(variance).sqrt() + dec(eps))))
    return torch.tensor(result, dtype=torch.float32)


KINDS = ["group_norm", "group_mean", "leave_one_out", "batch_norm"]


@pytest.mark.parametrize("kind", KINDS)
def test_all_group_model_configurations_match_pairwise_oracle(kind):
    for ticks, cut, shift, eps in itertools.product(
        itertools.product((-1, 0, 1), repeat=4), (1, 2, 3), (0, 17), (0, 1)
    ):
        data = [shift + i for i in ticks]
        ids = [9] * cut + [-7] * (4 - cut)
        obj = objectives.OBJECTIVES["grpo"].with_(advantage=kind, advantage_eps=eps)
        actual = objectives.compute_advantages(data, ids, obj)
        torch.testing.assert_close(
            actual, oracle(data, ids, kind, eps), rtol=1e-7, atol=0
        )


@pytest.mark.parametrize("kind", KINDS)
def test_input_precision_and_cancellation_survive_in_advantages_and_policy_gradient(
    kind,
):
    data = [1e8, 1e30, 1e8 + 1, 1, -1e30]
    ids = [3, 8, 3, 8, 8]
    obj = objectives.OBJECTIVES["grpo"].with_(advantage=kind, kl="none")
    adv = objectives.compute_advantages(data, ids, obj)
    expected = oracle(data, ids, kind, obj.advantage_eps)
    torch.testing.assert_close(adv, expected, rtol=1e-7, atol=0)
    scores = torch.zeros((len(data), 1), requires_grad=True)
    result = objectives.policy_loss(
        objective=obj,
        logp_cur=scores,
        logp_old=scores.detach(),
        advantages=adv,
        mask=torch.ones_like(scores),
    )
    result.loss.backward()
    torch.testing.assert_close(
        scores.grad[:, 0], -expected / len(data), rtol=2e-7, atol=0
    )


@pytest.mark.parametrize("kind", ["group_norm", "batch_norm"])
@pytest.mark.parametrize(
    "data,ids,eps",
    [
        ([1e308, -1e308, 1e308], [0, 0, 0], 0.0),
        ([0.0, math.ulp(0.0)], [0, 0], 0.0),
        ([0.0, 1.0], [0, 0], math.nextafter(0.5, 0.0)),
        ([0.0, 1.0], [0, 0], 0.5),
        ([1e300, 1e300, 1e-300, 2e-300], [0, 0, 1, 1], 0.0),
    ],
)
def test_exact_population_boundary_and_binary64_exponent_range(kind, data, ids, eps):
    obj = objectives.OBJECTIVES["grpo"].with_(advantage=kind, advantage_eps=eps)
    actual = objectives.compute_advantages(data, ids, obj)
    torch.testing.assert_close(actual, oracle(data, ids, kind, eps), rtol=1e-7, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_reward_derivatives_match_differentiable_population_formula(kind):
    rewards = torch.tensor(
        [1e8, 1e8 + 1, 1e8 + 3, 1e8 + 5], dtype=torch.float64, requires_grad=True
    )
    ids = torch.tensor([0, 1, 0, 1])
    obj = objectives.OBJECTIVES["grpo"].with_(advantage=kind)
    out = objectives.compute_advantages(rewards, ids, obj)
    expected = torch.zeros_like(rewards)
    for gid in (0, 1):
        v = rewards[ids == gid] - 1e8
        d = v - v.mean()
        if kind in ("group_norm", "batch_norm"):
            d = d / (
                (v if kind == "group_norm" else (rewards - 1e8)).std(correction=0)
                + obj.advantage_eps
            )
        elif kind == "leave_one_out":
            d = 2 * d
        expected[ids == gid] = d
    coefficients = torch.tensor([2, -1, -4, 3], dtype=torch.float32)
    for order in (1, 2) if kind in ("group_norm", "batch_norm") else (1,):
        if order == 1:
            (observed,) = torch.autograd.grad(
                out @ coefficients, rewards, create_graph=True
            )
            (reference,) = torch.autograd.grad(
                expected @ coefficients.double(), rewards, create_graph=True
            )
        else:
            (observed,) = torch.autograd.grad(observed @ coefficients.double(), rewards)
            (reference,) = torch.autograd.grad(
                reference @ coefficients.double(), rewards
            )
        torch.testing.assert_close(observed, reference, atol=1e-14, rtol=2e-13)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("baseline", ["mean", "median", "min"])
def test_native_and_numpy_group_reductions(native, baseline, monkeypatch):
    import numpy as np

    from stateset_agents.core import rust_accelerator as acc

    if native and not acc.is_rust_available():
        pytest.skip("local Rust extension unavailable")
    monkeypatch.setattr(acc, "_RUST_AVAILABLE", native)
    data = np.array([[1e308, 1e308, -1e308], [1e8, 1e8 + 1, 1e8 + 2], [-1e30, 1, 1e30]])
    view = data[:, ::-1]
    normalized = acc.compute_group_advantages(view, baseline, True).reshape(3, 3)
    for actual, r in zip(normalized, view, strict=True):
        expected = oracle(r, [0] * 3, "group_norm", 1e-8)
        torch.testing.assert_close(
            torch.as_tensor(actual).float(), expected, rtol=1e-7, atol=0
        )
    inputs = np.array([[1e308, 1e308], [1e8, 1e8 + 1], [-1e30, 1]], dtype=np.float64)
    for numbers, raw in zip(
        inputs,
        acc.compute_group_advantages(inputs, baseline, False).reshape(3, 2),
        strict=True,
    ):
        x, y = map(Fraction, numbers)
        b = min(x, y) if baseline == "min" else (x + y) / 2
        assert raw.tolist() == [float(x - b), float(y - b)]


@pytest.mark.parametrize("kind", ["group_norm", "batch_norm", "leave_one_out"])
async def test_dapo_optimizer_uses_uncollapsed_rewards_and_the_configured_population(
    kind, monkeypatch
):
    from tests.unit.test_group_sampler_refinement import learner

    t = learner("dapo", top_p=0.8)
    t._objective = t._objective.with_(advantage=kind)
    t.config.use_overlong_shaping = False
    data = [1e8, 1e8 + 1, 1e8 + 8, 1e8 + 12]
    observed_rewards = iter(data)
    t.reward_fn = lambda *_: next(observed_rewards)
    expected = oracle(data, [0, 0, 1, 1], kind, t._objective.advantage_eps).split(2)
    seen = 0
    original = t.compute_dapo_loss_from_log_probs

    def check(current, old, advantages, mask):
        nonlocal seen
        a = expected[seen % 2]
        seen += 1
        torch.testing.assert_close(advantages, a)
        loss, metrics = original(current, old, advantages, mask)
        (grad,) = torch.autograd.grad(loss, current, retain_graph=True)
        ratio = (current.detach() - old).clamp(-20, 20).exp()
        sign = a[:, None]
        clipped = ((sign > 0) & (ratio > 1 + t._objective.clip_high)) | (
            (sign < 0) & (ratio < 1 - t._objective.clip_low)
        )
        expected_grad = (-sign * ratio * mask / mask.sum()).masked_fill(clipped, 0)
        torch.testing.assert_close(
            grad, expected_grad.to(grad.dtype), atol=3e-7, rtol=1e-6
        )
        return loss, metrics

    monkeypatch.setattr(t, "compute_dapo_loss_from_log_probs", check)
    await t.train_step(["first", "second"])
    assert seen == 2 * t.config.num_gradient_updates


def test_gspo_metrics_and_advantages_for_a_representable_large_mean():
    from stateset_agents.training.gspo_trainer import GSPOTrainer

    for data in ([3e38, 3e38], [1e8, 1e8 + 1]):
        advantage, stats = GSPOTrainer.compute_group_advantages(None, data)
        assert stats["mean_reward"] == sum(map(Fraction, data)) / len(data)
        assert stats["std_reward"] == (data[1] - data[0]) / 2
        torch.testing.assert_close(advantage, oracle(data, [0, 0], "group_norm", 1e-8))


def test_ddp_moment_wire_has_room_for_a_maximum_int64_population():
    import sys

    from stateset_agents.core.reward_moments import PopulationMoments, RewardMoments

    for value in [sys.float_info.max, -sys.float_info.max, math.ulp(0.0)]:
        single = RewardMoments([value])
        n = (1 << 63) - 1
        pop = PopulationMoments(n, single.unit, n * single.total, n * single.squares)
        packet = pop.wire()
        assert len(packet) == 804
        merged = PopulationMoments.merge_wire(n, [packet])
        assert merged.mean == value and merged.variance_numerator == 0


@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("kind", ["group_mean", "leave_one_out"])
def test_independent_bernoulli_groups_have_the_predicted_expected_policy_gradient(
    n, kind
):
    p = Fraction(1, 4)
    expected = float(p * (1 - p)) * (1 if kind == "leave_one_out" else (n - 1) / n)
    weighted_gradient = 0.0
    obj = objectives.OBJECTIVES["grpo"].with_(advantage=kind, kl="none")
    for path in itertools.product((0, 1), repeat=n):
        theta = torch.tensor(
            math.log(p / (1 - p)), dtype=torch.float64, requires_grad=True
        )
        scores = torch.stack((theta * 0, theta)).log_softmax(0)[list(path)][:, None]
        rewards = [10**8 + a for a in path]
        a = objectives.compute_advantages(rewards, [0] * n, obj)
        loss = objectives.policy_loss(
            objective=obj,
            logp_cur=scores,
            logp_old=scores.detach(),
            advantages=a,
            mask=torch.ones_like(scores),
        ).loss
        (gradient,) = torch.autograd.grad(-loss, theta)
        k = sum(path)
        weighted_gradient += float(p**k * (1 - p) ** (n - k)) * gradient.item()
    assert weighted_gradient == pytest.approx(expected, abs=1e-8)


def test_int64_rewards_keep_neighboring_values_above_binary64_precision():
    x = 2**62
    r = torch.tensor([x, x + 1], dtype=torch.int64)
    assert objectives.compute_advantages(
        r, [0, 0], objectives.OBJECTIVES["rloo"]
    ).tolist() == [-1, 1]


@pytest.mark.parametrize("kind", KINDS)
def test_constant_group_has_zero_reward_derivative(kind):
    r = torch.full((4,), 1e100, dtype=torch.float64, requires_grad=True)
    obj = objectives.OBJECTIVES["grpo"].with_(advantage=kind)
    a = objectives.compute_advantages(r, [0] * len(r), obj)
    a.sum().backward()
    assert a.count_nonzero() == 0 and r.grad.count_nonzero() == 0


@pytest.mark.parametrize("preset", [None, "gspo"])
def test_sequence_compatibility_preserves_small_advantages_at_a_large_offset(
    preset, monkeypatch
):
    from types import SimpleNamespace

    from stateset_agents.training import loss_computation as lc
    from tests.unit.test_grpo_token_path import _config, _group, _tiny_model

    data = [1e8, 1e8 + 1, 1e8 + 8, 1e8 + 12]
    groups = [_group(data[:2]), _group(data[2:])]
    del groups[0].trajectories[0].turns[1].metadata["token_ids"]
    cfg = _config(
        allow_sequence_fallback=True,
        objective=preset,
        objective_overrides={"advantage": "batch_norm"} if preset else None,
        baseline_type="leave_one_out",
    )
    captured = []

    def loss(group, advantages, *args, **kwargs):
        captured.extend(advantages.tolist())
        return torch.zeros((), requires_grad=True), None

    monkeypatch.setattr(lc, "_compute_group_policy_loss", loss)
    result = lc.compute_grpo_loss(
        groups, cfg, SimpleNamespace(model=_tiny_model(), tokenizer=None), 0.0, 0, None
    )
    assert result["path"] == "sequence"
    kind = "batch_norm" if preset else "group_norm"
    eps = 1e-8 if preset else 5e-9
    torch.testing.assert_close(
        torch.tensor(captured), oracle(data, [0, 0, 1, 1], kind, eps)
    )


def test_enhanced_compatibility_sample_deviation_at_large_offset():
    from stateset_agents.training.reward_advantages import (
        reward_tensor,
        standardized_rewards,
    )

    a = standardized_rewards(reward_tensor([1e16, 1e16 + 2]), 1e-8, correction=1)
    magnitude = 1 / (math.sqrt(2) + 1e-8)
    torch.testing.assert_close(a, torch.tensor([-magnitude, magnitude]))
