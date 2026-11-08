"""Readout target design against exhaustive independent active-set solutions."""

from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space import (  # noqa: E402
    minimum_margin_target,
)


def exhaustive_projection(readout, residual, token, scale, margin):
    """Enumerate all possible binding sets on small test problems."""
    c = (readout[token] - readout[torch.arange(len(readout)) != token]) * scale
    b = margin - c @ residual
    best = None
    for size in range(min(len(c), len(residual)) + 1):
        for subset in itertools.combinations(range(len(c)), size):
            a = c[list(subset)]
            multipliers = torch.linalg.lstsq(a @ a.T, b[list(subset)]).solution
            d = a.T @ multipliers
            if bool((multipliers >= -1e-9).all() and (c @ d >= b - 1e-8).all()):
                if best is None or float(d.norm()) < float(best.norm()):
                    best = d
    assert best is not None
    return best


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("chunk_size", [1, 3, 100])
def test_projection_matches_exhaustive_oracle_and_dual_witness(seed, chunk_size):
    g = torch.Generator().manual_seed(seed)
    # Six independent rows make every token exposed; positive margin is feasible.
    head = torch.randn(6, 7, generator=g, dtype=torch.float64)
    residual = torch.randn(7, generator=g, dtype=torch.float64) * 3
    scale = torch.randn(7, generator=g, dtype=torch.float64)
    originals = [t.clone() for t in (head, residual, scale)]
    rng = torch.random.get_rng_state().clone()
    answer = minimum_margin_target(
        head, residual, 2, output_scale=scale, margin=0.3, chunk_size=chunk_size
    )
    oracle = exhaustive_projection(head, residual, 2, scale, 0.3)
    torch.testing.assert_close(answer.delta, oracle, atol=2e-7, rtol=2e-7)
    c = (head[2] - head[list(answer.token_ids)]) * scale
    c = c / c.norm(dim=1, keepdim=True)
    witness = c.T @ answer.multipliers
    torch.testing.assert_close(witness, answer.delta, atol=1e-10, rtol=1e-10)
    assert bool((answer.multipliers >= 0).all())
    assert abs(answer.diagnostics.duality_gap) <= 1e-6
    assert answer.diagnostics.minimum_margin >= 0.3 - 1e-7
    for actual, old in zip((head, residual, scale), originals, strict=True):
        assert torch.equal(actual, old)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_already_satisfied_returns_owned_zero_update():
    head = torch.eye(3)
    x = torch.tensor([3.0, 1.0, 0.0])
    result = minimum_margin_target(head, x, 0)
    assert torch.equal(result.delta, torch.zeros(3, dtype=torch.float64))
    assert result.token_ids == ()
    result.target.zero_()
    assert x[0] == 3


def test_all_vocabulary_constraints_checked_not_only_initial_winner():
    head = torch.tensor([[0.0, 0.0], [-1.0, 0.0], [1.0, -1.0]])
    x = torch.tensor([-4.0, -1.0])
    result = minimum_margin_target(head, x, 0, margin=1.0)
    torch.testing.assert_close(
        result.target, torch.tensor([1.0, 2.0], dtype=torch.float64)
    )
    assert len(result.token_ids) == 2


def test_rmsnorm_rank_equivalence_includes_negative_diagonal_scale():
    head = torch.eye(4)
    scale = torch.tensor([1.0, -2.0, 0.5, 4.0])
    x = torch.tensor([0.0, -3.0, 1.0, 1.0])
    result = minimum_margin_target(head, x, 0, output_scale=scale, margin=0.5)
    normalized = result.target * scale / (result.target.square().mean() + 1e-6).sqrt()
    assert int((head.double() @ normalized).argmax()) == 0
    assert result.diagnostics.update_norm < 5


def test_duplicate_impossible_and_nonexposed_target_bounded_failure():
    with pytest.raises(ConfigurationError, match="duplicate"):
        minimum_margin_target(torch.zeros(2, 3), torch.ones(3), 0)
    # Token 0 lies strictly inside the competitors' convex hull.
    with pytest.raises(ConfigurationError, match="max_sweeps.*not established"):
        minimum_margin_target(
            torch.tensor([[0.0], [-1.0], [1.0]]), torch.ones(1), 0, max_sweeps=20
        )


def test_working_set_limit_does_not_claim_infeasibility():
    with pytest.raises(ConfigurationError, match="limit.*not established"):
        minimum_margin_target(
            torch.eye(3), torch.tensor([0.0, 3.0, 3.0]), 0, max_constraints=1
        )


def test_large_readout_scale_cannot_hide_a_tie_within_normalized_tolerance():
    result = minimum_margin_target(
        torch.eye(2, dtype=torch.float64) * 1e12, torch.zeros(2), 1
    )
    assert result.diagnostics.minimum_margin > 0
    assert int((torch.eye(2, dtype=torch.float64) @ result.target).argmax()) == 1


def test_overflow_in_contrast_norm_is_rejected():
    with pytest.raises(ConfigurationError, match="nonfinite"):
        minimum_margin_target(
            torch.eye(2, dtype=torch.float64) * 1e300, torch.zeros(2), 0
        )


def test_underflow_does_not_claim_duplicate_row_infeasibility():
    with pytest.raises(ConfigurationError, match="underflow.*not established"):
        minimum_margin_target(
            torch.eye(2, dtype=torch.float64) * 1e-300, torch.zeros(2), 0
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"margin": 0},
        {"margin": float("nan")},
        {"margin": True},
        {"feasibility_atol": 0},
        {"duality_atol": -1},
        {"chunk_size": 0},
        {"max_constraints": True},
        {"max_sweeps": 0},
        {"target_token_id": True},
        {"target_token_id": 3},
        {"output_scale": torch.ones(2)},
    ],
)
def test_invalid_inputs(kwargs):
    args = {"readout": torch.eye(3), "residual": torch.ones(3), "target_token_id": 0}
    args.update(kwargs)
    with pytest.raises(ConfigurationError):
        minimum_margin_target(**args)


@pytest.mark.parametrize("name", ["readout", "residual", "output_scale"])
def test_nonfinite_values_even_in_unselected_readout_rows(name):
    args = {
        "readout": torch.eye(3),
        "residual": torch.ones(3),
        "output_scale": torch.ones(3),
        "target_token_id": 0,
    }
    args[name].flatten()[-1] = float("nan")
    with pytest.raises(ConfigurationError, match="finite"):
        minimum_margin_target(**args, chunk_size=1)
