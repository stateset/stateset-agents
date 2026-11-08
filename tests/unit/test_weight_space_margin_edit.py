"""Independent full-coordinate optimization checks for joint margin editing."""

import itertools

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space import (  # noqa: E402
    constrained_edit,
    joint_margin_edit,
    minimum_margin_target,
)


def problem(seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "weight": torch.randn(2, 4, generator=g) / 5,
        "keys": torch.randn(2, 4, generator=g),
        "residuals": torch.randn(2, 2, generator=g) * 3,
        "readout": torch.randn(3, 2, generator=g),
        "target_token_ids": [0, 1],
        "output_scale": torch.tensor([1.0, -0.7]),
        "protect": torch.randn(1, 4, generator=g),
        "locality": torch.randn(3, 4, generator=g),
        "margin": 0.4,
    }


def exhaustive_weight_oracle(args):
    """Enumerate all binding inequalities in the unreduced weight coordinates."""
    k, r, e = (args[n].double() for n in ("keys", "residuals", "readout"))
    h, p = args["locality"].double(), args["protect"].double()
    metric = torch.eye(k.shape[1], dtype=torch.float64) + args[
        "locality_weight"
    ] * h.T @ h / len(h)
    objective = torch.kron(torch.eye(e.shape[1], dtype=torch.float64), metric)
    equality = torch.kron(torch.eye(e.shape[1], dtype=torch.float64), p)
    normals, rhs = [], []
    for i, target in enumerate(args["target_token_ids"]):
        for token in range(len(e)):
            if token != target:
                c = (e[target] - e[token]) * args["output_scale"].double()
                normals.append(torch.kron(c, k[i]))
                rhs.append(args["margin"] - c @ r[i])
    normals, rhs = torch.stack(normals), torch.stack(rhs)
    best = None
    for size in range(len(normals) + 1):
        for subset in itertools.combinations(range(len(normals)), size):
            system = torch.cat((equality, normals[list(subset)]))
            bound = torch.cat(
                (torch.zeros(len(equality), dtype=torch.float64), rhs[list(subset)])
            )
            inverse_action = torch.linalg.solve(objective, system.T)
            lam = torch.linalg.lstsq(
                system @ inverse_action, bound, driver="gelsd"
            ).solution
            d = inverse_action @ lam
            if bool(
                (lam[len(equality) :] >= -1e-7).all()
                and (normals @ d >= rhs - 1e-7).all()
            ) and torch.allclose(system @ d, bound, atol=1e-7, rtol=1e-7):
                value = float(d @ objective @ d / 2)
                if best is None or value < best[0]:
                    best = value, d.reshape(e.shape[1], k.shape[1])
    assert best is not None
    return best


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("strength", [0, 5])
def test_joint_edit_matches_full_coordinate_active_set_oracle(seed, strength):
    args = problem(seed)
    args["locality_weight"] = strength
    before = {n: t.clone() for n, t in args.items() if isinstance(t, torch.Tensor)}
    rng = torch.random.get_rng_state().clone()
    result = joint_margin_edit(**args, chunk_size=1)
    value, oracle = exhaustive_weight_oracle(args)
    torch.testing.assert_close(
        result.ideal_B @ result.ideal_A, oracle, atol=2e-6, rtol=2e-6
    )
    assert result.diagnostics.primal_objective == pytest.approx(
        value, abs=2e-7, rel=2e-7
    )
    assert abs(result.diagnostics.duality_gap) <= 1e-8
    assert min(result.diagnostics.deployed_minimum_margins) >= 0.4 - 1e-4
    assert result.delta.rank <= len(args["keys"])
    for name, original in before.items():
        assert torch.equal(args[name], original)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_joint_solve_can_reduce_cost_compared_with_independent_residual_targets():
    args = problem(2)
    args["locality_weight"] = 5
    result = joint_margin_edit(**args)
    changes = torch.stack(
        [
            minimum_margin_target(
                args["readout"],
                r,
                target,
                output_scale=args["output_scale"],
                margin=args["margin"],
            ).delta
            for r, target in zip(
                args["residuals"], args["target_token_ids"], strict=True
            )
        ]
    )
    separate = constrained_edit(
        args["weight"],
        args["keys"],
        (args["keys"] @ args["weight"].T).double() + changes,
        protect=args["protect"],
        locality=args["locality"],
        locality_weight=5,
    )
    assert (
        result.diagnostics.primal_objective
        < separate.diagnostics.objective_value / 2 * 0.99
    )


def test_duplicate_contexts_share_capacity_and_conflicts_fail_boundedly():
    args = problem(1)
    args["keys"][1] = args["keys"][0]
    args["residuals"][1] = args["residuals"][0]
    args["target_token_ids"] = [0, 0]
    result = joint_margin_edit(**args)
    assert min(result.diagnostics.deployed_minimum_margins) > 0
    args["target_token_ids"] = [0, 1]
    with pytest.raises(ConfigurationError, match="not established"):
        joint_margin_edit(**args, max_sweeps=100)


def test_fully_protected_span_cannot_create_capacity_from_roundoff():
    args = problem(3)
    args["protect"] = torch.eye(4)
    with pytest.raises(ConfigurationError, match="capacity"):
        joint_margin_edit(**args)


def test_fully_protected_noop_is_valid_when_all_margins_already_hold():
    result = joint_margin_edit(
        torch.ones(2, 2),
        torch.ones(1, 2),
        torch.tensor([[3.0, 0.0]]),
        torch.eye(2),
        [0],
        protect=torch.eye(2),
    )
    assert result.diagnostics.update_norm == 0
    assert torch.equal(result.delta.dense(), torch.zeros(2, 2))


@pytest.mark.parametrize(
    "option",
    [
        {"margin": 0},
        {"locality_weight": -1},
        {"rcond": 0},
        {"max_constraints": 0},
        {"max_sweeps": True},
        {"margin_atol": -1},
        {"target_token_ids": [0]},
        {"target_token_ids": [False, 1]},
        {"max_update_norm": -1},
        {"chunk_size": 0},
    ],
)
def test_invalid_controls(option):
    args = problem(1)
    args.update(option)
    with pytest.raises(ConfigurationError):
        joint_margin_edit(**args)


@pytest.mark.parametrize(
    "name",
    ["weight", "keys", "residuals", "readout", "protect", "locality", "output_scale"],
)
def test_nonfinite_data(name):
    args = problem(1)
    args[name].flatten()[-1] = float("nan")
    with pytest.raises(ConfigurationError):
        joint_margin_edit(**args)


def test_update_cap_rejects_before_mutating_inputs():
    args = problem(2)
    before = args["weight"].clone()
    with pytest.raises(ConfigurationError, match="max_update_norm"):
        joint_margin_edit(**args, max_update_norm=0)
    assert torch.equal(args["weight"], before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_binary_exact_solution_in_deployment_dtype(dtype):
    result = joint_margin_edit(
        torch.zeros(2, 2, dtype=dtype),
        torch.eye(2, dtype=dtype),
        torch.zeros(2, 2, dtype=dtype),
        torch.eye(2, dtype=dtype),
        [0, 1],
        margin_atol=0,
        preservation_atol=0,
    )
    assert result.diagnostics.deployed_minimum_margins == (1.0, 1.0)
