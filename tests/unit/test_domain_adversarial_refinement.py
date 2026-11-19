"""Compare DANN's feature and critic updates to independent domain-mean likelihoods."""

import itertools
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.sim_to_real import (  # noqa: E402
    DomainAdaptationModule,
    SimToRealConfig,
)


def adapter(dim=2, strength=0.5):
    return DomainAdaptationModule(
        SimToRealConfig(
            state_dim=dim, domain_critic_hidden=2, gradient_reversal_lambda=strength
        )
    ).double()


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0, 2.0])
def test_two_joint_adam_steps_match_manual_backpropagation(strength):
    torch.manual_seed(56)
    module = adapter(strength=strength)
    encoder = torch.nn.Linear(2, 2).double()
    layers = [
        encoder,
        *[m for m in module.domain_critic if isinstance(m, torch.nn.Linear)],
    ]
    # Keep nonlinear paths active and include both signs of the encoder's input.
    with torch.no_grad():
        for layer in layers[1:-1]:
            layer.weight.copy_(torch.tensor([[0.8, -0.2], [0.3, 0.9]]))
            layer.bias.copy_(torch.tensor([0.1, 0.35]))
    params = [p for layer in layers for p in layer.parameters()]
    expected = [p.detach().numpy().copy() for p in params]
    moments = [np.zeros_like(x) for x in expected]
    squares = [np.zeros_like(x) for x in expected]
    opt = torch.optim.Adam(params, lr=0.02, betas=(0.8, 0.9), eps=1e-8)
    source = torch.tensor(
        [[0.25, -1.0], [1.25, 0.5], [-0.75, 0.25]], dtype=torch.float64
    )
    target = torch.tensor([[-2.0, 0.5], [0.2, 0.3]], dtype=torch.float64)
    for step in (1, 2):
        opt.zero_grad()
        source_features = encoder(source)
        domain_loss = module.compute_adaptation_loss(source_features, encoder(target))
        # A supervised task on the source still updates its shared encoder at lambda zero.
        task_loss = source_features.square().sum() / len(source)
        (domain_loss + task_loss).backward()
        gradients = [np.zeros_like(x) for x in expected]
        measured = 0.0
        for label, rows in enumerate((source.numpy(), target.numpy())):
            for x in rows:
                feature = expected[0] @ x + expected[1]
                activations = [feature]
                for j in (2, 4):
                    activations.append(
                        np.maximum(expected[j] @ activations[-1] + expected[j + 1], 0)
                    )
                logit = (expected[6] @ activations[-1] + expected[7]).item()
                signed_logit = logit if label == 0 else -logit
                measured += np.logaddexp(0, signed_logit) / len(rows)
                grad = np.array([(1 / (1 + math.exp(-logit)) - label) / len(rows)])
                for j in (6, 4, 2):
                    previous = activations[(j - 2) // 2]
                    gradients[j] += np.outer(grad, previous)
                    gradients[j + 1] += grad
                    grad = grad @ expected[j]
                    if j > 2:
                        grad *= previous > 0
                grad *= -strength
                if label == 0:
                    grad += 2 * feature / len(rows)
                gradients[0] += np.outer(grad, x)
                gradients[1] += grad
        assert domain_loss.item() == pytest.approx(measured, abs=1e-13)
        for param, grad in zip(params, gradients, strict=True):
            np.testing.assert_allclose(param.grad.numpy(), grad, atol=1e-13)
        opt.step()
        for i, (param, grad) in enumerate(zip(params, gradients, strict=True)):
            moments[i] = 0.8 * moments[i] + 0.2 * grad
            squares[i] = 0.9 * squares[i] + 0.1 * grad**2
            expected[i] -= (
                0.02
                * (moments[i] / (1 - 0.8**step))
                / (np.sqrt(squares[i] / (1 - 0.9**step)) + 1e-8)
            )
            for actual, want in (
                (param, expected[i]),
                (opt.state[param]["exp_avg"], moments[i]),
                (opt.state[param]["exp_avg_sq"], squares[i]),
            ):
                np.testing.assert_allclose(actual.detach().numpy(), want, atol=1e-13)


class TableCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, features):
        return features[:, :1] + self.bias * features[:, 1:]


def test_every_valid_tlc_snapshot_matches_public_backward():
    module = adapter()
    head = TableCritic()
    module.domain_critic = head
    jacobian = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    for source in itertools.product((False, True), repeat=3):
        if all(source) or not any(source):
            continue
        selection = torch.tensor(source)
        for bins in itertools.product((0, 1), repeat=3):
            head_jac = torch.tensor(bins, dtype=torch.float64) + 1
            for scores in itertools.product((2, 3, 4), repeat=2):
                values = [scores[b] for b in bins]
                logits = torch.tensor(
                    [math.log(s / (6 - s)) for s in values], dtype=torch.float64
                )
                signed = [
                    (s - (0 if is_source else 6)) / (6 * source.count(is_source))
                    for s, is_source in zip(values, source, strict=True)
                ]
                for gain in (0.0, 0.5, 1.0, 2.0):
                    enc = torch.zeros((), dtype=torch.float64, requires_grad=True)
                    features = torch.stack((logits + enc * jacobian, head_jac), dim=1)
                    loss = module.domain_adversarial_loss(
                        features[selection], features[~selection], gain
                    )
                    ge, gd = torch.autograd.grad(loss, (enc, head.bias))
                    assert ge.item() == pytest.approx(
                        -gain * sum(v * (i - 1) for i, v in enumerate(signed)),
                        abs=1e-14,
                    )
                    assert gd.item() == pytest.approx(
                        sum(v * (1 + b) for v, b in zip(signed, bins, strict=True)),
                        abs=1e-14,
                    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_saturated_wrong_labels_still_move_both_sides(dtype):
    module = adapter(dim=1).to(dtype)
    module.domain_critic = torch.nn.Linear(1, 1, bias=False, dtype=dtype)
    with torch.no_grad():
        module.domain_critic.weight.fill_(1.0)
    sim = torch.tensor([[1000.0]], dtype=dtype, requires_grad=True)
    real = torch.tensor([[-1000.0]], dtype=dtype, requires_grad=True)
    loss = module.compute_adaptation_loss(sim, real)
    assert loss.item() == 2000.0
    sim_grad, real_grad, critic_grad = torch.autograd.grad(
        loss, (sim, real, module.domain_critic.weight)
    )
    assert (sim_grad.item(), real_grad.item(), critic_grad.item()) == (
        -0.5,
        0.5,
        2000.0,
    )


@pytest.mark.parametrize("strength", [-1.0, math.inf, math.nan])
def test_invalid_strength_and_empty_domains_fail_before_scoring(strength):
    module = adapter()
    x = torch.ones((1, 2), dtype=torch.float64)
    with pytest.raises(ValueError, match="strength"):
        module.domain_adversarial_loss(x, x, strength)
    for s, r in ((x[:0], x), (x, x[:0])):
        with pytest.raises(ValueError, match="nonempty"):
            module.compute_adaptation_loss(s, r)


def test_bayes_discriminator_and_vocabulary_metric_use_empirical_word_frequencies():
    from types import SimpleNamespace

    from stateset_agents.evaluation.sim_to_real_metrics import SimToRealEvaluator

    def trajectory(words):
        return SimpleNamespace(
            turns=[SimpleNamespace(role="assistant", content=" ".join(words))]
        )

    for source, target in (
        ([0, 1], [1, 0, 1, 0]),
        ([0, 0, 0, 1], [1, 1, 1, 0]),
        ([0, 1, 2], [2, 3]),
        ([0, 0], [1]),
    ):
        vocab = sorted(set(source + target))
        s = np.array([source.count(i) / len(source) for i in vocab])
        r = np.array([target.count(i) / len(target) for i in vocab])
        safe_logits = [
            math.log(b / a) if a and b else (1000.0 if b else -1000.0)
            for a, b in zip(s, r, strict=True)
        ]
        module = adapter(dim=len(vocab))
        module.domain_critic = torch.nn.Linear(len(vocab), 1, bias=False).double()
        with torch.no_grad():
            module.domain_critic.weight.copy_(
                torch.tensor([safe_logits], dtype=torch.float64)
            )
        eye = torch.eye(len(vocab), dtype=torch.float64)
        best = module.compute_adaptation_loss(
            eye[[vocab.index(i) for i in source]], eye[[vocab.index(i) for i in target]]
        ).item()
        metrics = SimToRealEvaluator().compute_distribution_metrics(
            [trajectory(list(map(str, source)))], [trajectory(list(map(str, target)))]
        )
        want = math.log(2) - best / 2
        assert metrics["vocabulary_js_divergence"] == pytest.approx(want, abs=2e-15)
        if set(source) == set(target) and not np.array_equal(s, r):
            assert metrics["vocabulary_js_divergence"] > 0


@pytest.mark.parametrize("epsilon", [0.0, 1e-10, 0.2])
def test_histogram_divergences_match_decimal_definition(epsilon):
    from decimal import Decimal as D
    from decimal import localcontext

    from stateset_agents.evaluation.sim_to_real_metrics import (
        compute_js_divergence,
        compute_kl_divergence,
    )

    small = float(np.nextafter(1.0, 2.0))
    examples = [
        ([1e-300, 0.0, 0.0], [0.0, 1e308, 0.0]),
        ([1e308, 1e308], [1e-300, 1e-300]),
        ([small, 1.0], [1.0, small]),
        ([1.0, 3.0, 4.0], [2.0, 0.0, 5.0]),
        ([0.9995, 1.0005], [1.0, 1.0]),
    ]
    with localcontext() as ctx:
        ctx.prec = 110

        def probs(row):
            weights = [D(float(x)) for x in row]
            normalized = [x / sum(weights) + D(epsilon) for x in weights]
            return [x / sum(normalized) for x in normalized]

        def kl(s, r):
            return sum(x * (x / y).ln() for x, y in zip(s, r, strict=True) if x)

        for p, q in examples:
            a, b = probs(p), probs(q)
            m = [(x + y) / 2 for x, y in zip(a, b, strict=True)]
            js = max(0.0, float((kl(a, m) + kl(b, m)) / 2))
            expected_kl = (
                math.inf
                if any(x and not y for x, y in zip(a, b, strict=True))
                else max(0.0, float(kl(a, b)))
            )
            actual = compute_js_divergence(np.array(p), np.array(q), epsilon)
            assert actual == pytest.approx(
                js, rel=2e-3 if 0 < js < 1e-30 else 4e-13, abs=1e-110
            )
            actual_kl = compute_kl_divergence(np.array(p), np.array(q), epsilon)
            assert actual_kl == pytest.approx(
                expected_kl, rel=2e-3 if 0 < expected_kl < 1e-30 else 4e-13, abs=1e-110
            )
            assert 0 <= actual <= math.log(2) and actual_kl >= 0
    for p, q in (
        ([0.0, 0.0], [1.0, 1.0]),
        ([1.0, -1.0], [1.0, 1.0]),
        ([1.0], [1.0, 1.0]),
    ):
        with pytest.raises(ValueError):
            compute_js_divergence(np.array(p), np.array(q), epsilon)


def test_domain_mean_preserves_large_finite_losses_and_each_row_gradient():
    module = adapter(dim=1)
    module.domain_critic = torch.nn.Identity()
    sim = torch.full((3, 1), 1.2e308, dtype=torch.float64, requires_grad=True)
    real = torch.full((2, 1), 1000.0, dtype=torch.float64, requires_grad=True)
    loss = module.compute_adaptation_loss(sim, real)
    assert loss.item() == 1.2e308
    sim_grad, real_grad = torch.autograd.grad(loss, (sim, real))
    torch.testing.assert_close(sim_grad, torch.full_like(sim, -0.5 / 3))
    torch.testing.assert_close(real_grad, torch.zeros_like(real))
