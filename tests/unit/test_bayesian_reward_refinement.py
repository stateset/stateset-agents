"""Native checks for the finite predictive mixture and paired Gaussian loss."""

import math
from itertools import permutations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.trajectory import ConversationTurn  # noqa: E402
from stateset_agents.rewards.bayesian_reward_model import (  # noqa: E402
    ActiveLearningSelector,
    BayesianLinear,
    BayesianRewardConfig,
    BayesianRewardFunction,
    EnsembleRewardModel,
    MCDropoutRewardModel,
    VariationalBayesianRewardModel,
    _paired_gaussian_nll,
    _population_moments,
)


@pytest.mark.parametrize("column", [False, True])
def test_train_step_pairs_one_label_per_row(column):
    ensemble = EnsembleRewardModel(1, 1, num_layers=0, device="cpu")
    head = ensemble.models[0]
    with torch.no_grad():
        head.mean_head.weight.fill_(1)
        head.mean_head.bias.zero_()
        head.log_var_head.weight.zero_()
        head.log_var_head.bias.zero_()
    features = torch.tensor([[1.0], [-1.0]])
    labels = features if column else features.flatten()
    assert ensemble.train_step(features, labels)["ensemble_loss"] == 0
    assert head.mean_head.weight.grad.item() == 0
    assert head.mean_head.bias.grad.item() == 0
    assert head.log_var_head.bias.grad.item() == 0.5


@pytest.mark.parametrize(
    "targets", [torch.ones(1), torch.ones(1, 2), torch.ones(2, 1, 1)]
)
def test_invalid_label_shape_preserves_all_members(targets):
    model = EnsembleRewardModel(1, 2, hidden_size=2, device="cpu")
    before = [
        p.detach().clone() for member in model.models for p in member.parameters()
    ]
    with pytest.raises(ValueError, match="label"):
        model.train_step(torch.ones(2, 1), targets)
    assert all(
        torch.equal(p, old)
        for p, old in zip(
            (p for member in model.models for p in member.parameters()),
            before,
            strict=True,
        )
    )
    assert all(not optimizer.state for optimizer in model.optimizers)


@pytest.mark.parametrize(
    "log_variance,m,y",
    [
        (0.7, 2.3, -1.0),
        (-1000, 0.0, 0.0),
        (-1000, 1e-220, 0.0),
        (-1430, 0.0, 0.0),
        (1430, 1e308, 0.0),
    ],
)
def test_paired_nll_value_and_both_gradients(log_variance, m, y):
    from decimal import Decimal, localcontext

    with localcontext() as ctx:
        ctx.prec = 100
        delta = Decimal.from_float(m) - Decimal.from_float(y)
        log_var = Decimal.from_float(float(log_variance))
        precision = (-log_var).exp()
        expected = tuple(
            float(v)
            for v in (
                (log_var + delta**2 * precision) / 2,
                delta * precision,
                (1 - delta**2 * precision) / 2,
            )
        )
    mu, lv = (
        torch.tensor([[z]], dtype=torch.float64, requires_grad=True)
        for z in (m, log_variance)
    )
    loss = _paired_gaussian_nll(mu, lv, torch.tensor([y], dtype=torch.float64))
    grads = torch.autograd.grad(loss, (mu, lv))
    for actual, ref in zip((loss, *grads), expected, strict=True):
        assert actual.item() == pytest.approx(ref, rel=3e-12, abs=5e-324)


def test_all_draws_contribute_to_ensemble_moments(monkeypatch):
    ensemble = EnsembleRewardModel(1, 2, num_layers=0, device="cpu")
    for member, values in zip(
        ensemble.models, (((0, 1), (2, 9)), ((0, 4), (6, 2))), strict=True
    ):
        draws = iter(values)

        def forward(x, draws=draws):
            mean, var = next(draws)
            return torch.tensor([[mean], [-mean]], dtype=torch.float64), torch.tensor(
                [[math.log(var)], [0.0]], dtype=torch.float64
            )

        monkeypatch.setattr(member, "forward", forward)
    mu, ep, al = ensemble.predict_with_uncertainty(torch.ones(2, 1), 2)
    assert mu.flatten().tolist() == [2, -2]
    assert ep.flatten().tolist() == [6, 6]
    assert al.flatten().tolist() == pytest.approx([4, 1])


def test_population_variance_with_large_common_offset_and_arbitrary_order():
    for pair in permutations((1e16, 1e16 + 2)):
        _, variance = _population_moments(torch.tensor(pair, dtype=torch.float64))
        assert variance == 1


def test_small_mean_survives_cancellation_and_empty_batches_keep_shape():
    for arrangement in permutations((1e100, 1.0, -1e100)):
        mu, _ = _population_moments(torch.tensor(arrangement, dtype=torch.float64))
        assert mu.item() == 1 / 3
    mean, var = _population_moments(torch.empty(3, 0, 1))
    assert mean.shape == var.shape == (0, 1)


@pytest.mark.parametrize("family", ["dropout", "ensemble", "variational"])
def test_single_draw_has_zero_epistemic_variance(family):
    model = {
        "dropout": lambda: MCDropoutRewardModel(2, 4),
        "ensemble": lambda: EnsembleRewardModel(2, 1, hidden_size=4, device="cpu"),
        "variational": lambda: VariationalBayesianRewardModel(2, 4),
    }[family]()
    mu, epistemic, *_ = model.predict_with_uncertainty(torch.ones(3, 2), 1)
    assert mu.shape == epistemic.shape == (3, 1)
    assert torch.isfinite(mu).all()
    assert torch.equal(epistemic, torch.zeros_like(mu))
    with pytest.raises(ValueError, match="positive"):
        model.predict_with_uncertainty(torch.ones(3, 2), 0)


def test_dropout_sampling_restores_mixed_modes_on_failure():
    torch.manual_seed(5)
    model = MCDropoutRewardModel(8, 8, num_layers=1, dropout_rate=0.5).eval()
    model.log_var_head.train()
    flags = [m.training for m in model.modules()]
    x = torch.ones(12, 8)
    deterministic = model(x)[0]
    assert model.predict_with_uncertainty(x, 8)[1].max() > 0
    with pytest.raises(RuntimeError):
        model.predict_with_uncertainty(torch.ones(4, 3), 1)
    assert [m.training for m in model.modules()] == flags
    assert torch.equal(model(x)[0], deterministic)


@pytest.mark.parametrize("log_std", [-100.0, -1e-14, 0.0, 1e-14, 50.0])
def test_gaussian_kl_and_log_scale_gradient(log_std):
    from decimal import Decimal, localcontext

    layer = BayesianLinear(1, 1).double()
    with torch.no_grad():
        layer.weight_mu.zero_()
        layer.bias_mu.zero_()
        layer.weight_log_sigma.fill_(log_std)
        layer.bias_log_sigma.zero_()
    kl = layer.kl_divergence()
    kl.backward()
    with localcontext() as ctx:
        ctx.prec = 90
        d = 2 * Decimal.from_float(log_std)
        ref = float((d.exp() - 1 - d) / 2)
        grad = float(d.exp() - 1)
    assert kl.item() == pytest.approx(ref, rel=1e-13, abs=0)
    assert layer.weight_log_sigma.grad.item() == pytest.approx(grad, rel=1e-13, abs=0)
    assert layer.bias_log_sigma.grad.item() == 0


def test_last_invalid_member_keeps_optimizer_state_unmodified():
    ensemble = EnsembleRewardModel(2, 2, num_layers=0, device="cpu")
    with torch.no_grad():
        ensemble.models[-1].log_var_head.bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        ensemble.train_step(torch.ones(2, 2), torch.ones(2))
    assert all(not opt.state for opt in ensemble.optimizers)


@pytest.mark.parametrize("ensemble", [False, True])
@pytest.mark.asyncio
async def test_labeled_update_through_public_reward(ensemble):
    config = BayesianRewardConfig(
        use_ensemble=ensemble,
        num_ensemble=1,
        num_samples=1,
        num_layers=0,
        learning_rate=0.02,
        weight_decay=0,
    )
    model = BayesianRewardFunction(input_dim=2, config=config, device="cpu")
    head = model.model.models[0] if ensemble else model.model
    head.double()
    turn = ConversationTurn(role="assistant", content="ok")
    features = model._extract_features([turn])
    assert model._extract_features([]).shape == features.shape == (2,)
    before = await model.compute_reward([turn])
    assert before.breakdown["epistemic_uncertainty"] == 0
    assert before.metadata["interval_method"] == "moment_matched_normal_approximation"
    selector = ActiveLearningSelector()
    for step, rate in enumerate([0.1, None], 1):
        metrics = selector.update_with_labels(
            model, [features], [before.score + 1.0], 1, rate
        )
        assert math.isfinite(metrics["update_loss"])
        opt = model.model.optimizers[0] if ensemble else model._optimizer
        assert opt.param_groups[0]["lr"] == config.learning_rate
        assert all(state["step"].item() == step for state in opt.state.values())
    after = await model.compute_reward([turn])
    assert after.score > before.score
    assert selector.select_batch_for_labeling([(features, after)], batch_size=0) == []
    assert selector.selected_samples == []
    with pytest.raises(ValueError, match="same length"):
        selector.update_with_labels(model, [np.zeros(2)], [], 1)


def test_normal_approximation_can_cover_less_than_95_percent():
    # A 32-component mixture has one narrow component at each of ±4 and 30 at 0.
    means = [-4, 4] + [0] * 30
    sigma = 0.01
    upper = 1.96 * math.sqrt(1 + sigma**2)

    def cdf(z):
        return math.erfc(-z / math.sqrt(2)) / 2

    mass = sum(
        cdf((upper - m) / sigma) - cdf((-upper - m) / sigma) for m in means
    ) / len(means)
    assert mass == pytest.approx(30 / 32)


def test_calibration_uses_paired_observations_and_representable_rms():
    model = BayesianRewardFunction(
        1, BayesianRewardConfig(use_ensemble=False, num_layers=0), device="cpu"
    )
    with pytest.raises(ValueError, match="observation"):
        model.calibrate([], [])
    # One residual is 2e308 but the averages over four observations fit.
    stats = model.calibrate([1e308, 0.0, 0.0, 0.0], [-1e308, 0.0, 0.0, 0.0])
    assert stats["mae"] == pytest.approx(5e307)
    assert stats["rmse"] == pytest.approx(1e308)
    for predicted, actual in [([1.0], []), ([math.nan], [0.0])]:
        with pytest.raises(ValueError):
            model.calibrate(predicted, actual)
    assert model.calibrate([], []) == stats


@pytest.mark.asyncio
async def test_query_weights_do_not_change_predictive_variance(monkeypatch):
    config = BayesianRewardConfig(
        use_ensemble=False,
        num_layers=0,
        epistemic_weight=2,
        aleatoric_weight=0,
        high_uncertainty_threshold=4,
    )
    model = BayesianRewardFunction(1, config, device="cpu")
    monkeypatch.setattr(
        model,
        "_predict_with_uncertainty",
        lambda x: tuple(x.new_tensor(v) for v in (0.0, 2.0, 5.0)),
    )
    result = await model.compute_reward([])
    assert result.breakdown["total_uncertainty"] == 7
    assert result.breakdown["weighted_uncertainty"] == 4
    assert result.breakdown["confidence_interval_upper"] == pytest.approx(
        1.96 * math.sqrt(7)
    )
    assert not result.metadata["high_uncertainty"]
    assert not ActiveLearningSelector(uncertainty_threshold=4).should_query_label(
        result
    )
    assert ActiveLearningSelector(uncertainty_threshold=3.9).should_query_label(result)


def test_variational_total_kl_preserves_range_for_fp32_parameters():
    model = VariationalBayesianRewardModel(2, 2, num_layers=1)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, BayesianLinear):
                layer.weight_mu.zero_()
                layer.bias_mu.zero_()
                layer.weight_log_sigma.fill_(50)
                layer.bias_log_sigma.zero_()
    assert model.kl_divergence().item() == pytest.approx(6 * (math.exp(100) - 101) / 2)


@pytest.mark.parametrize("prior", [0, -1, float("inf"), float("nan")])
def test_invalid_gaussian_prior(prior):
    with pytest.raises(ValueError, match="prior_std"):
        BayesianLinear(1, 1, prior)


def test_native_shared_parameter_gradient_matches_paired_batch_rule():
    model = EnsembleRewardModel(
        1, 1, hidden_size=1, num_layers=1, dropout_rate=0, learning_rate=0, device="cpu"
    )
    member = model.models[0].double()
    parameters = dict(member.named_parameters())
    values = {
        "feature_net.0.weight": 2.0,
        "feature_net.0.bias": 0.4,
        "mean_head.weight": 0.3,
        "mean_head.bias": -0.2,
        "log_var_head.weight": -0.4,
        "log_var_head.bias": 0.7,
    }
    with torch.no_grad():
        for name, param in parameters.items():
            param.fill_(values[name])
    x, targets = [-1.0, 0.5, 2.0], [0.3, 2.0, -1.0]
    expected = dict.fromkeys(values, 0.0)
    score = 0.0
    for obs, target in zip(x, targets, strict=True):
        feature = max(0, 2 * obs + 0.4)
        mean = 0.3 * feature - 0.2
        log_var = -0.4 * feature + 0.7
        residual = mean - target
        precision = math.exp(-log_var)
        score += (log_var + residual**2 * precision) / 2
        dm = residual * precision
        dl = (1 - residual**2 * precision) / 2
        shared = (0.3 * dm - 0.4 * dl) if feature > 0 else 0.0
        expected["feature_net.0.weight"] += shared * obs
        expected["feature_net.0.bias"] += shared
        expected["mean_head.weight"] += dm * feature
        expected["mean_head.bias"] += dm
        expected["log_var_head.weight"] += dl * feature
        expected["log_var_head.bias"] += dl
    result = model.train_step(
        torch.tensor(x, dtype=torch.float64).unsqueeze(1),
        torch.tensor(targets, dtype=torch.float64),
    )
    assert result["ensemble_loss"] == pytest.approx(score / len(x), rel=1e-13)
    for name, param in parameters.items():
        assert param.grad.item() == pytest.approx(expected[name] / len(x), rel=1e-13)
