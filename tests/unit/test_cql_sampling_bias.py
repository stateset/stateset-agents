"""Compare CQL's sampled targets and penalties with their proposal-population values."""

import itertools
import math

import pytest
import torch

from stateset_agents.training.offline_rl_algorithms import (
    ConservativeQLearning,
    CQLConfig,
)


def learner(count: int, temperature: float) -> ConservativeQLearning:
    config = CQLConfig(
        num_layers=1,
        num_random_actions=count,
        temperature=temperature,
        discount_factor=0.75,
    )
    return ConservativeQLearning(1, 1, config, device="cpu")


@pytest.mark.parametrize("count", [1, 2, 4, 8])
@pytest.mark.parametrize("p_high", [0.25, 0.5])
def test_soft_partition_bias_and_tail_over_every_finite_bank(count, p_high):
    tau, logged = 0.75, -0.375
    model = learner(count, tau)
    draws = list(itertools.product((0, 1), repeat=count))
    scores = torch.tensor(draws, dtype=torch.float64) * (tau * math.log(3))
    # A score exponentiates to 1 or 3. Keep duplicate banks and their probability mass.
    highs = [sum(row) for row in draws]
    probabilities = [p_high**k * (1 - p_high) ** (count - k) for k in highs]
    reference = [tau * math.log1p(2 * k / count) for k in highs]
    sampled = model._soft_max(scores, normalize=True).tolist()
    penalties = model._soft_max(
        scores, normalize=False, baseline=torch.tensor(logged)
    ).tolist()
    assert sampled == pytest.approx(reference, abs=2e-15)
    offset = tau * math.log(count) - logged
    assert penalties == pytest.approx([x + offset for x in reference], abs=2e-15)

    mass = 1 + 2 * p_high
    variance = 4 * p_high * (1 - p_high)
    population = tau * math.log(mass)
    expected = math.fsum(w * y for w, y in zip(probabilities, sampled, strict=True))
    expected_penalty = math.fsum(
        w * y for w, y in zip(probabilities, penalties, strict=True)
    )
    bias = population - expected
    assert population + offset - expected_penalty == pytest.approx(bias, abs=2e-15)
    assert 0 < bias <= tau * variance / (count * mass)
    assert min(sampled) < population < max(sampled)

    delta = 0.375
    threshold = -tau * math.log1p(-delta)
    outside = math.fsum(
        w
        for w, y in zip(probabilities, sampled, strict=True)
        if abs(y - population) >= threshold
    )
    assert outside <= variance / (count * (mass * delta) ** 2)
    if count == 8 and p_high == 0.5:
        assert 0 < outside < variance / (count * (mass * delta) ** 2) < 1


def test_repeating_one_draw_breaks_the_independent_proposal_rate():
    count = 8
    model = learner(count, 1)
    data = torch.tensor([0, math.log(3)], dtype=torch.float64)
    correlated = data[:, None].expand(-1, count)
    values = model._soft_max(correlated, normalize=True)
    error = math.log(2) - values.mean().item()
    assert error == pytest.approx(math.log(2) - math.log(3) / 2)
    assert error > 1 / (count * 2)


def test_constant_critic_has_zero_bias_for_both_partitions():
    tau, count = 0.625, 7
    model = learner(count, tau)
    scores = torch.full((2, count), 2.0, dtype=torch.float64)
    assert torch.equal(
        model._soft_max(scores, normalize=True), torch.tensor([2.0, 2.0])
    )
    penalty = model._soft_max(scores, normalize=False, baseline=scores[:, 0])
    assert penalty.tolist() == pytest.approx([tau * math.log(count)] * 2, abs=2e-15)


@pytest.mark.parametrize("count", [1, 2])
def test_uniform_action_integrals_through_native_bellman_and_penalty(
    monkeypatch, count
):
    # Evaluate every product of equal-width midpoint cells of the real proposal law.
    # The affine heads and the pointwise minimum are 1-Lipschitz in the action.
    tau, cells = 0.5, 128
    model = learner(count, tau)
    nodes = (torch.arange(cells, dtype=torch.float32) + 0.5) * (2 / cells) - 1
    bank = torch.cartesian_prod(*([nodes] * count)).reshape(-1, count)
    n_rows = bank.shape[0]
    states = torch.zeros((n_rows, 1))
    actions = torch.full_like(states, 0.25)
    with torch.no_grad():
        for net, slope in (
            (model.q1, 1),
            (model.q2, -1),
            (model.q1_target, 1),
            (model.q2_target, -1),
        ):
            layer = net.network[0]
            layer.weight.copy_(torch.tensor([[0, slope]]))
            layer.bias.zero_()

    def uniform_grid(tensor, low, high):
        assert (low, high) == (-1, 1) and tensor.shape == (n_rows * count, 1)
        return tensor.copy_(bank.reshape_as(tensor))

    monkeypatch.setattr(torch.Tensor, "uniform_", uniform_grid)
    gamma, reward = model.config.discount_factor, 1.25
    with torch.no_grad():
        target = model._bellman_target(
            torch.full_like(states, reward), states, torch.zeros_like(states)
        )
        penalties = model._compute_cql_loss(
            states, actions, model.q1(states, actions), model.q2(states, actions)
        )

    # Independent uniform integrals for clipped twin score -|a| and live scores ±a.
    min_mass = -tau * math.expm1(-1 / tau)
    second_moment = -tau / 2 * math.expm1(-2 / tau)
    min_var = second_moment - min_mass**2
    full_target = reward + gamma * tau * math.log(min_mass)
    target_bias = full_target - target.double().mean().item()
    # Moving each proposal to its cell midpoint changes its score by at most 1/cells.
    # A normalized or unnormalized soft partition obeys the same sup-norm bound.
    target_radius = gamma / cells + 2e-6
    assert target_radius < target_bias
    assert target_bias + target_radius <= gamma * tau * min_var / (
        count * math.exp(-1 / tau) * min_mass
    )
    assert target.min().item() < full_target < target.max().item()

    head_mass = tau * math.sinh(1 / tau)
    head_var = tau / 2 * math.sinh(2 / tau) - head_mass**2
    head_bound = tau * head_var / (count * math.exp(-1 / tau) * head_mass)
    head_biases = []
    for penalty, logged in zip(penalties, (0.25, -0.25), strict=True):
        full_gap = tau * (math.log(head_mass) + math.log(count)) - logged
        gap_bias = full_gap - penalty.item()
        assert 1 / cells + 2e-6 < gap_bias
        assert gap_bias + 1 / cells + 2e-6 <= head_bound
        head_biases.append(gap_bias)
    assert head_biases[0] == pytest.approx(head_biases[1], abs=2e-6)
    if count == 1:
        # At N=1 the native values average the score itself; E[-|a|]=-1/2 and E[a]=0.
        assert target_bias == pytest.approx(
            gamma * (tau * math.log(min_mass) + 0.5), abs=2e-6
        )
        assert head_biases[0] == pytest.approx(tau * math.log(head_mass), abs=2e-6)


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_expected_penalty_autograd_against_population_slope(count):
    tau, a, variance, logged_slope = 0.75, 2.0, 1.0, 0.125
    model = learner(count, tau)
    # Equal-probability proposal masses are 1 and 3. The same parameter affects
    # both the proposal and the logged action. The proposal's score slope is
    # either 0 or sign, with population tilted mean 3*sign/4.
    bank = torch.tensor(
        list(itertools.product((0, 1), repeat=count)), dtype=torch.float64
    )
    for sign in (-1, 1):
        theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        q = (tau * math.log(3) + sign * theta) * bank
        penalty = model._soft_max(
            q, normalize=False, baseline=0.375 + logged_slope * theta
        )
        sampled = torch.autograd.grad(penalty.mean(), theta)[0].item()
        exact_sampled = (
            math.fsum(
                math.comb(count, k) * 3 * k / (count + 2 * k) for k in range(count + 1)
            )
            / 2**count
        )
        assert sampled == pytest.approx(sign * exact_sampled - logged_slope, abs=2e-15)
        population = 0.75 * sign - logged_slope
        # X*(D-b) takes opposite values of magnitude 3/4 with equal probability.
        centered_variance = 0.75**2
        scale = math.sqrt(centered_variance / variance)
        bound = (centered_variance / scale + scale * variance) / (2 * count * a)
        assert 0 < sign * (population - sampled) <= bound
        if count == 1:
            assert abs(population - sampled) == pytest.approx(0.25)


@pytest.mark.parametrize("count", [1, 2])
def test_tail_with_no_positive_uniform_mass_floor(count):
    # X is uniform on (0,1), Q=log(X). The low tail of the two-draw mean has
    # CDF 2*x^2 below 1/2, from the area of a triangle in the proposal square.
    cells = 128
    nodes = (torch.arange(cells, dtype=torch.float64) + 0.5) / cells
    masses = torch.cartesian_prod(*([nodes] * count)).reshape(-1, count)
    model = learner(count, 1)
    value = model._soft_max(masses.log(), normalize=True)
    a, v, delta = 0.5, 1 / 12, 0.75
    outside = (
        ((value - math.log(a)).abs() >= -math.log1p(-delta)).double().mean().item()
    )
    # The high-side mass threshold is 2, outside the support.
    threshold = a * (1 - delta)
    radius = 0.5 / cells

    def lower_tail(x):
        return x if count == 1 else 2 * x * x

    assert lower_tail(threshold - radius) <= outside <= lower_tail(threshold + radius)
    assert 0 < lower_tail(threshold) < v / (count * (a * delta) ** 2) < 1


def test_continuous_population_derivative_of_affine_critic():
    # For A uniform on [-1,1] and Q=theta*A, the population partition mass
    # is sinh(k)/k at k=theta/tau. Its soft-value derivative is coth(k)-1/k.
    tau, cells, logged_action = 0.5, 2048, 0.25
    model = learner(cells, tau)
    actions = ((torch.arange(cells, dtype=torch.float64) + 0.5) * (2 / cells) - 1)[
        None, :
    ]
    theta = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    k = theta.item() / tau
    population = tau * math.log(math.sinh(k) / k)
    slope = 1 / math.tanh(k) - 1 / k
    scores = theta * actions
    soft = model._soft_max(scores, normalize=True)
    penalty = model._soft_max(scores, normalize=False, baseline=theta * logged_action)
    soft_grad = torch.autograd.grad(soft, theta, retain_graph=True)[0]
    penalty_grad = torch.autograd.grad(penalty, theta)[0]
    # On this uniform midpoint grid, the exact quadrature mass equals the
    # continuous mass times z/sinh(z), z=k/cells, by the geometric sum.
    z = k / cells
    log_correction = tau * math.log(z / math.sinh(z))
    slope_correction = (1 / z - 1 / math.tanh(z)) / cells
    assert soft.item() == pytest.approx(population + log_correction, abs=2e-15)
    assert soft_grad.item() == pytest.approx(slope + slope_correction, abs=3e-15)
    assert penalty_grad.item() + logged_action == pytest.approx(
        soft_grad.item(), abs=3e-15
    )
