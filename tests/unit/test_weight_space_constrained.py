"""Dense constrained edits: independent solves, numerical gates and composition."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space import constrained_edit  # noqa: E402


def problem(seed=3):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(5, 12, generator=g) / 3
    keys = torch.randn(3, 12, generator=g)
    protect = torch.randn(4, 12, generator=g)
    bias = torch.randn(5, generator=g)
    targets = torch.randn(3, 5, generator=g)
    return w, keys, targets, protect, bias


@pytest.mark.parametrize("seed", range(5))
def test_dense_edit_matches_independent_augmented_minimum_norm_solve(seed):
    w, k, t, p, b = problem(seed)
    before = [x.clone() for x in (w, k, t, p, b)]
    rng = torch.random.get_rng_state().clone()
    edit = constrained_edit(w, k, t, protect=p, bias=b)
    # Independent oracle: solve the full augmented equality system, without
    # projecting protected directions or calling any implementation helpers.
    system = torch.cat((k, p)).double()
    rhs = torch.cat(
        (t.double() - k.double() @ w.double().T - b.double(), torch.zeros(4, 5))
    )
    oracle = torch.linalg.lstsq(system, rhs, driver="gelsd").solution.T
    torch.testing.assert_close(
        edit.delta.dense().double(), oracle, atol=2e-6, rtol=2e-6
    )
    after = w + edit.delta.dense()
    torch.testing.assert_close(k @ after.T + b, t, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(p @ after.T, p @ w.T, atol=1e-5, rtol=1e-5)
    assert edit.diagnostics.protected_rank == 4
    assert edit.diagnostics.edit_rank == 3
    assert edit.diagnostics.available_dimensions == 8
    assert edit.diagnostics.protected_span_error < 1e-5
    for original, snapshot in zip((w, k, t, p, b), before, strict=True):
        assert torch.equal(original, snapshot)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_preservation_extends_to_unsampled_combinations_of_references():
    w, k, t, p, b = problem()
    result = constrained_edit(w, k, t, protect=p, bias=b)
    heldout = torch.randn(100, len(p), generator=torch.Generator().manual_seed(4)) @ p
    drift = heldout.double() @ ((w + result.delta.dense()).double() - w.double()).T
    assert float(drift.norm(dim=1).max()) < 2e-5


def test_duplicate_constraints_and_scaled_protected_rows_do_not_change_solution():
    w, k, t, p, b = problem()
    plain = constrained_edit(w, k, t, protect=p, bias=b)
    repeated = constrained_edit(
        w,
        torch.cat((k, k)),
        torch.cat((t, t)),
        protect=torch.cat((p, p * 1024, p / 1024, torch.zeros(1, 12))),
        bias=b,
    )
    assert repeated.diagnostics.protected_rank == 4
    torch.testing.assert_close(
        plain.delta.dense(), repeated.delta.dense(), atol=2e-6, rtol=2e-6
    )


def test_sequential_edits_protect_previous_targets():
    w, k, t, p, b = problem()
    first = constrained_edit(w, k[:2], t[:2], protect=p, bias=b)
    w1 = w + first.delta.dense()
    second = constrained_edit(w1, k[2:], t[2:], protect=torch.cat((p, k[:2])), bias=b)
    w2 = w1 + second.delta.dense()
    torch.testing.assert_close(k @ w2.T + b, t, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(p @ w2.T, p @ w.T, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "kind", ["protected", "duplicate", "full_span", "near_overlap", "zero_key"]
)
def test_conflicts_are_rejected_instead_of_using_numerical_noise_as_capacity(kind):
    w, k, t, p, b = problem()
    if kind == "protected":
        p = k.clone()
    elif kind == "duplicate":
        k[1] = k[0]
    elif kind == "full_span":
        p = torch.eye(12)
    elif kind == "near_overlap":
        p = torch.eye(12)[:11]
        k = torch.ones(3, 12)
        k[:, -1] = 1e-12
    else:
        k.zero_()
    with pytest.raises(ConfigurationError, match="infeasible"):
        constrained_edit(w, k, t, protect=p, bias=b)


def test_saturated_protection_still_allows_a_noop():
    w = torch.eye(3)
    keys = torch.ones(1, 3)
    result = constrained_edit(w, keys, keys, protect=torch.eye(3))
    assert result.diagnostics.edit_rank == 0
    assert result.diagnostics.available_dimensions == 0
    assert result.diagnostics.update_norm == 0
    assert torch.equal(result.delta.dense(), torch.zeros_like(w))


def test_update_norm_cap_prevents_an_ill_conditioned_large_edit():
    w = torch.zeros(1, 2)
    k = torch.tensor([[1.0, 1e-5]])
    with pytest.raises(ConfigurationError, match="update norm"):
        constrained_edit(
            w,
            k,
            torch.ones(1, 1),
            protect=torch.tensor([[1.0, 0.0]]),
            max_update_norm=10,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_binary_exact_edit_and_bias_in_deployment_dtype(dtype):
    w = torch.ones(2, 3, dtype=dtype)
    k = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype)
    p = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype)
    b = torch.tensor([1.0, 2.0], dtype=dtype)
    t = torch.tensor([[3.0, 5.0]], dtype=dtype)
    result = constrained_edit(
        w, k, t, protect=p, bias=b, target_atol=0, target_rtol=0, preservation_atol=0
    )
    edited = w + result.delta.dense().to(dtype)
    assert torch.equal(k @ edited.T + b, t)
    assert torch.equal(p @ edited.T, p @ w.T)


def test_unrealizable_bfloat16_target_fails_the_rounded_execution_gate():
    with pytest.raises(ConfigurationError, match="target tolerance"):
        constrained_edit(
            torch.ones(1, 2, dtype=torch.bfloat16),
            torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16),
            torch.tensor([[1.001]]),
            target_atol=1e-6,
            target_rtol=0,
        )


@pytest.mark.parametrize("name", ["weight", "keys", "targets", "protect", "bias"])
def test_nonfinite_inputs_are_rejected(name):
    w, k, t, p, b = problem()
    inputs = {"weight": w, "keys": k, "targets": t, "protect": p, "bias": b}
    inputs[name].flatten()[0] = float("nan")
    with pytest.raises(ConfigurationError, match="finite"):
        constrained_edit(**inputs)


@pytest.mark.parametrize(
    "options",
    [
        {"rcond": 0},
        {"rcond": 1},
        {"rcond": float("nan")},
        {"target_atol": -1},
        {"target_rtol": float("inf")},
        {"preservation_atol": -1},
        {"max_update_norm": -1},
    ],
)
def test_invalid_controls_are_rejected(options):
    w, k, t, p, b = problem()
    with pytest.raises(ConfigurationError):
        constrained_edit(w, k, t, protect=p, bias=b, **options)


def test_shape_and_dtype_mismatches_are_rejected():
    w, k, t, p, b = problem()
    for kwargs in (
        {"keys": k[:, :-1]},
        {"targets": t[:1]},
        {"keys": k.double()},
        {"protect": p[:, :-1]},
        {"bias": b[:2]},
        {"keys": k[:0]},
    ):
        inputs = {"weight": w, "keys": k, "targets": t, "protect": p, "bias": b}
        inputs.update(kwargs)
        with pytest.raises(ConfigurationError):
            constrained_edit(**inputs)
