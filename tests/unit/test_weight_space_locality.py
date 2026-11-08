"""Locality objectives must improve the intended metric without relaxing edits."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space import (  # noqa: E402
    WeightSpaceSession,
    constrained_edit,
)


def problem(seed=17):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(4, 12, generator=g) / 3
    k = torch.randn(3, 12, generator=g)
    t = torch.randn(3, 4, generator=g)
    p = torch.randn(3, 12, generator=g)
    h = torch.randn(40, 12, generator=g)
    h[:, :6] *= 8
    b = torch.randn(4, generator=g)
    return w, k, t, p, h, b


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("strength", [0.1, 1.0, 100.0])
def test_weighted_edit_matches_independent_kkt_solution(seed, strength):
    w, k, t, p, h, b = problem(seed)
    rng = torch.random.get_rng_state().clone()
    snapshots = [v.clone() for v in (w, k, t, p, h, b)]
    result = constrained_edit(
        w, k, t, protect=p, locality=h, locality_weight=strength, bias=b
    )
    # Independent KKT solve with a full covariance; no SVD/projection helpers.
    covariance = torch.eye(
        12, dtype=torch.float64
    ) + strength * h.double().T @ h.double() / len(h)
    constraints = torch.cat((k, p)).double()
    rhs = torch.cat(
        (t.double() - k.double() @ w.double().T - b.double(), torch.zeros(3, 4))
    )
    inverse_constraints = torch.linalg.solve(covariance, constraints.T)
    oracle = (
        inverse_constraints @ torch.linalg.solve(constraints @ inverse_constraints, rhs)
    ).T
    torch.testing.assert_close(
        result.delta.dense().double(), oracle, atol=3e-6, rtol=3e-6
    )
    after = w + result.delta.dense()
    torch.testing.assert_close(k @ after.T + b, t, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(p @ after.T, p @ w.T, atol=1e-5, rtol=1e-5)
    drift = (h @ after.T + b).double() - (h @ w.T + b).double()
    assert result.diagnostics.locality_rms_drift == pytest.approx(
        float(drift.square().mean().sqrt()), abs=2e-6
    )
    realized = after.double() - w.double()
    objective = realized.square().sum() + strength * (
        h.double() @ realized.T
    ).square().sum() / len(h)
    assert result.diagnostics.objective_value == pytest.approx(
        float(objective), rel=1e-6
    )
    assert result.diagnostics.protected_rank == 3
    for actual, before in zip((w, k, t, p, h, b), snapshots, strict=True):
        assert torch.equal(actual, before)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_higher_penalty_reduces_calibration_drift_with_identical_constraints():
    w, k, t, p, h, b = problem()
    results = [
        constrained_edit(w, k, t, protect=p, locality=h, locality_weight=lam, bias=b)
        for lam in (0, 0.1, 1, 100)
    ]
    norms = [r.diagnostics.update_norm for r in results]
    drifts = [r.diagnostics.locality_rms_drift for r in results]
    assert norms == sorted(norms)
    assert drifts == sorted(drifts, reverse=True)
    assert drifts[-1] < drifts[0] * 0.8
    for result in results:
        assert result.diagnostics.target_max_error < 1e-5
        assert result.diagnostics.protected_span_error < 1e-5


def test_zero_penalty_or_zero_locality_preserves_default_delta_exactly():
    w, k, t, p, h, b = problem()
    original = constrained_edit(w, k, t, protect=p, bias=b)
    for kwargs in (
        {"locality": h, "locality_weight": 0},
        {"locality": torch.zeros_like(h)},
    ):
        current = constrained_edit(w, k, t, protect=p, bias=b, **kwargs)
        assert torch.equal(original.delta.B, current.delta.B)
        assert torch.equal(original.delta.A, current.delta.A)
    assert original.diagnostics.locality_rms_drift is None


def test_repeating_all_locality_rows_preserves_the_mean_based_objective():
    w, k, t, p, h, b = problem()
    original = constrained_edit(
        w, k, t, protect=p, bias=b, locality=h, locality_weight=7
    )
    repeated = constrained_edit(
        w, k, t, protect=p, bias=b, locality=h.repeat(3, 1), locality_weight=7
    )
    torch.testing.assert_close(
        original.delta.dense(), repeated.delta.dense(), atol=2e-6, rtol=2e-6
    )


def test_weighting_cannot_relax_protection_even_when_targets_conflict():
    w, k, t, _, h, b = problem()
    with pytest.raises(ConfigurationError, match="infeasible"):
        constrained_edit(w, k, t, protect=k, bias=b, locality=h, locality_weight=100)


def test_zero_capacity_noop_remains_possible_with_locality():
    w = torch.eye(3)
    keys = torch.ones(1, 3)
    result = constrained_edit(w, keys, keys, protect=torch.eye(3), locality=keys)
    assert result.diagnostics.available_dimensions == 0
    assert result.diagnostics.locality_rms_drift == 0


def test_session_locality_cap_rejects_before_mutation_and_success_reverts():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
    session = WeightSpaceSession(model, lambda _: [1], budget=1)
    keys = torch.tensor([[1.0, 1.0, 0.0]])
    refs = torch.tensor([[0.0, 0.0, 1.0]])
    h = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    before = session.weight("0")
    targets = model(keys).detach() + 0.5
    with pytest.raises(ConfigurationError, match="locality RMS drift"):
        session.apply_constrained(
            "0",
            keys,
            targets,
            protect=refs,
            locality=h,
            locality_weight=100,
            max_locality_rms=0,
        )
    assert session.active_edit_ids == []
    assert torch.equal(session.weight("0"), before)
    _, result = session.apply_constrained(
        "0",
        keys,
        targets,
        protect=refs,
        locality=h,
        locality_weight=100,
        max_locality_rms=0.01,
    )
    assert result.diagnostics.locality_rms_drift < 0.01
    session.revert()
    assert torch.equal(session.weight("0"), before)
    assert session.budget.used == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"locality_weight": -1},
        {"locality_weight": float("nan")},
        {"max_locality_rms": -1},
        {"max_locality_rms": float("inf")},
        {"locality": None, "max_locality_rms": 1},
    ],
)
def test_invalid_locality_controls_raise_typed_errors(kwargs):
    w, k, t, p, h, b = problem()
    with pytest.raises(ConfigurationError):
        constrained_edit(w, k, t, protect=p, bias=b, **{"locality": h, **kwargs})


@pytest.mark.parametrize("kind", ["empty", "width", "dtype", "nonfinite"])
def test_invalid_locality_arrays_are_rejected(kind):
    w, k, t, p, h, b = problem()
    if kind == "empty":
        h = h[:0]
    elif kind == "width":
        h = h[:, :-1]
    elif kind == "dtype":
        h = h.double()
    else:
        h[0, 0] = float("nan")
    with pytest.raises(ConfigurationError, match="locality"):
        constrained_edit(w, k, t, protect=p, bias=b, locality=h)


def test_extreme_metric_cannot_silently_drop_protected_directions():
    w, k, t, p, h, b = problem()
    with pytest.raises(ConfigurationError, match="numerically singular"):
        constrained_edit(w, k, t, protect=p, bias=b, locality=h, locality_weight=1e30)
