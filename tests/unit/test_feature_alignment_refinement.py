"""Independent sample-pair and log-kernel checks for feature alignment."""

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from stateset_agents.evaluation.sim_to_real_metrics import compute_mmd  # noqa: E402
from stateset_agents.training.offline_rl_bear import MMDKernel  # noqa: E402
from stateset_agents.training.sim_to_real import (  # noqa: E402
    DomainAdaptationModule,
    SimToRealConfig,
)


@pytest.mark.parametrize(
    "sigma,offset,scale",
    [(1.0, 1e8, 1.0), (2.0, 0.0, 1e-12), (1e-200, 0.0, 1e-200), (1e200, 0.0, 1e200)],
)
def test_bear_transfer_and_evaluation_match_gaussian_value_and_gradients(
    sigma, offset, scale
):
    source = (
        torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float64) * scale + offset
    )
    target = (
        torch.tensor([[2.0, -1.0], [0.0, 0.0], [2.0, -1.0]], dtype=torch.float64)
        * scale
        + offset
    )
    module = DomainAdaptationModule(
        SimToRealConfig(adaptation_method="mmd", state_dim=2)
    )
    with localcontext() as ctx:
        ctx.prec = 110
        sigma_d = Decimal(sigma)
        data = [
            [Decimal(v) for v in row] for row in torch.cat((source, target)).tolist()
        ]
        weights = [Decimal(1) / len(source)] * len(source) + [
            -Decimal(1) / len(target)
        ] * len(target)
        total = Decimal(0)
        gradients = [[Decimal(0) for _ in row] for row in data]
        for i, a in enumerate(data):
            for j, b in enumerate(data):
                delta = [
                    (first - second) / sigma_d
                    for first, second in zip(a, b, strict=True)
                ]
                k = (-sum(d * d for d in delta) / 2).exp()
                term = weights[i] * weights[j] * k
                total += term
                for axis, d in enumerate(delta):
                    gradients[i][axis] -= 2 * term * d / sigma_d
    expected = float(total)
    deriv = torch.tensor(
        [[float(v) for v in row] for row in gradients], dtype=torch.float64
    )
    assert compute_mmd(source.numpy(), target.numpy(), sigma) == pytest.approx(
        expected, rel=4e-13, abs=0.0
    )
    source.requires_grad_()
    target.requires_grad_()
    actual = module.compute_mmd(source, target, sigma)
    if sigma == 1.0:
        torch.testing.assert_close(
            actual, module.compute_adaptation_loss(source, target)
        )
    bear = MMDKernel("gaussian", sigma).compute_mmd(source[None], target[None])[0]
    assert actual.item() == pytest.approx(expected, rel=4e-13, abs=0.0)
    torch.testing.assert_close(bear, actual, atol=0, rtol=0)
    actual.backward()
    torch.testing.assert_close(
        torch.cat((source.grad, target.grad)), deriv, atol=1e-310, rtol=4e-12
    )


@pytest.mark.parametrize("offset", [0.0, float(2**52)])
def test_coral_pairwise_covariance_and_gradients(offset):
    source = torch.tensor([[0, 1], [4, 1], [3, 7]], dtype=torch.float64) + offset
    target = torch.tensor([[4, 2], [1, 6]], dtype=torch.float64) - offset

    def covariance(rows):
        total = np.zeros((2, 2))
        for i, a in enumerate(rows):
            for b in rows[:i]:
                delta = a - b
                total += np.outer(delta, delta)
        return total / (len(rows) * (len(rows) - 1))

    difference = covariance(source.numpy()) - covariance(target.numpy())
    expected = np.square(difference).sum() / 16
    gradients = []
    for sign, x in ((1, source), (-1, target)):
        # Compute each centered row from a sum of differences to other rows.
        centered = (x[:, None, :] - x[None, :, :]).numpy().mean(axis=1)
        gradients.append(sign * centered @ difference / (4 * (len(x) - 1)))
    module = DomainAdaptationModule(
        SimToRealConfig(adaptation_method="coral", state_dim=2)
    )
    result = module.compute_adaptation_loss(
        source.requires_grad_(), target.requires_grad_()
    )
    assert result.item() == pytest.approx(expected)
    result.backward()
    for g, oracle in zip((source.grad, target.grad), gradients, strict=True):
        np.testing.assert_allclose(g.numpy(), oracle, rtol=1e-13, atol=1e-14)


def test_alignment_population_and_bandwidth_boundaries():
    x = torch.ones((2, 1), dtype=torch.float64)
    module = DomainAdaptationModule(
        SimToRealConfig(adaptation_method="coral", state_dim=1)
    )
    for s, r in ((x[:1], x), (x, x[:1])):
        with pytest.raises(ValueError, match="at least two"):
            module.compute_adaptation_loss(s, r)
    for invalid in (math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            compute_mmd(np.array([invalid]), np.array([0.0]))
    for sigma in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="bandwidth"):
            module.compute_mmd(x, x, sigma)
        with pytest.raises(ValueError, match="bandwidth"):
            compute_mmd(x.numpy(), x.numpy(), sigma)
    for s, r in ((x[:0], x), (x, x[:0])):
        with pytest.raises(ValueError, match="nonempty"):
            module.compute_mmd(s, r)
        with pytest.raises(ValueError, match="nonempty"):
            compute_mmd(s.numpy(), r.numpy())


def test_tlc_feature_partitions_match_both_public_statistics():
    mmd = DomainAdaptationModule(SimToRealConfig(adaptation_method="mmd", state_dim=1))
    coral = DomainAdaptationModule(
        SimToRealConfig(adaptation_method="coral", state_dim=1)
    )
    distance = math.sqrt(2 * math.log(2))
    for available in range(32):
        si = [i for i in range(2) if available & (1 << i)]
        ri = [i for i in range(2, 5) if available & (1 << i)]
        if not si or not ri:
            continue
        for bins in range(32):
            rows = torch.tensor(
                [(bins >> i) & 1 for i in range(5)], dtype=torch.float64
            )[:, None]
            s, r = rows[si], rows[ri]
            ns, nr = len(s), len(r)
            a, b = int(s.sum()), int(r.sum())
            expected = ((a * nr - b * ns) / (ns * nr)) ** 2
            assert mmd.compute_adaptation_loss(
                s * distance, r * distance
            ).item() == pytest.approx(expected, abs=2e-15)
            if min(ns, nr) >= 2:
                delta = a * (ns - a) / (ns * (ns - 1)) - b * (nr - b) / (nr * (nr - 1))
                assert coral.compute_adaptation_loss(s, r).item() == pytest.approx(
                    delta**2 / 4, abs=2e-15
                )


def test_gaussian_backward_recovers_an_underflowed_kernel_at_small_bandwidth():
    sigma = 1e-300
    x = torch.tensor([[[42 * sigma]]], dtype=torch.float64, requires_grad=True)
    y = torch.zeros_like(x)
    loss = MMDKernel("gaussian", sigma).compute_mmd(x, y)
    loss.backward()
    expected = math.exp(math.log(84) - 882 - math.log(sigma))
    assert loss.item() == 2
    assert x.grad.item() == pytest.approx(expected, rel=3e-13, abs=0.0)
