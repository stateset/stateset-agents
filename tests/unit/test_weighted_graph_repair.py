"""Check exact message replacement, edit isolation, controls, and graph identity."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space.handwritten_weighted_graph import (  # noqa: E402
    encode_weighted_graphs,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/weight_space"))
from learned_weighted_graph_job import LearnedRelaxation  # noqa: E402
from repair_weighted_graph_job import (  # noqa: E402
    addition_edit,
    component_grid,
    from_state,
    matched_control,
    matrix_keys,
)


def test_edit_isolated_to_69_coordinates_without_architecture_or_rng_change() -> None:
    original = LearnedRelaxation(10)
    state = original.state_dict()
    snapshot = {k: v.clone() for k, v in state.items()}
    rng = torch.get_rng_state().clone()
    edited, ledger = addition_edit(state)
    model = from_state(edited)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(ledger) == 69
    assert sum(p.numel() for p in model.parameters()) == 1185
    assert [type(m) for m in model.message] == [type(m) for m in original.message]
    allowed = {(r["tensor"], r["index"]) for r in ledger}
    assert len(allowed) == 69
    for name, before in state.items():
        assert torch.equal(before, snapshot[name])
        for index in (
            (before.flatten() != edited[name].flatten()).nonzero().flatten().tolist()
        ):
            assert (name, index) in allowed
    for row in ledger:
        assert float(state[row["tensor"]].flatten()[row["index"]]) == row["before"]
        assert float(edited[row["tensor"]].flatten()[row["index"]]) == row["after"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_path_adds_nonnegative_inputs_and_repairs_a_full_expensive_chain(
    seed: int,
) -> None:
    state, _ = addition_edit(LearnedRelaxation(seed).state_dict())
    model = from_state(state)
    rng = torch.Generator().manual_seed(17)
    inputs = torch.cat((component_grid(), torch.rand(1024, 2, generator=rng)))
    with torch.inference_mode():
        assert torch.equal(model.message(inputs).squeeze(-1), inputs.sum(1))
    costs = torch.zeros(3, 17, 17)
    costs[0, torch.arange(16), torch.arange(1, 17)] = 7
    costs[1, 0, 1], costs[1, 0, 2], costs[1, 2, 1] = 7, 2, 3
    # Third graph is empty and starts at a different vertex.
    packed = encode_weighted_graphs(costs, torch.tensor([0, 0, 9]))
    expected = torch.full((3, 17), 113)
    expected[0] = torch.arange(17) * 7
    expected[1, :3] = torch.tensor([0, 5, 2])
    expected[2, 9] = 0
    with torch.inference_mode():
        actual = model(packed)
    assert torch.equal(actual.round().long(), expected)
    assert float((actual - expected).abs().max()) < 0.001


@pytest.mark.parametrize("damage", ["missing", "shape", "nan", "dtype", "overflow"])
def test_invalid_checkpoint_is_rejected_before_mutating_input(damage: str) -> None:
    state = LearnedRelaxation(0).state_dict()
    if damage == "missing":
        del state["message.0.bias"]
    elif damage == "shape":
        state["message.0.weight"] = torch.zeros(31, 2)
    elif damage == "nan":
        state["message.0.weight"][1, 0] = float("nan")
    elif damage == "dtype":
        state = {k: v.double() for k, v in state.items()}
    else:
        state["message.0.weight"][1, 0] = 3e38
    before = {k: v.clone() for k, v in state.items()}
    with pytest.raises(ConfigurationError):
        addition_edit(state)
    for key in state:
        torch.testing.assert_close(
            state[key], before[key], rtol=0, atol=0, equal_nan=True
        )


def test_random_control_matches_coordinates_and_delta_norm() -> None:
    initial = LearnedRelaxation(5).state_dict()
    edited, ledger = addition_edit(initial)
    rng = torch.get_rng_state().clone()
    control = matched_control(initial, ledger, 12)
    again = matched_control(initial, ledger, 12)
    assert torch.equal(rng, torch.get_rng_state())
    patch_delta = torch.cat([(edited[k] - initial[k]).flatten() for k in initial])
    control_delta = torch.cat([(control[k] - initial[k]).flatten() for k in initial])
    assert torch.equal(patch_delta != 0, control_delta != 0)
    assert not torch.equal(patch_delta, control_delta)
    assert control_delta.norm().item() == pytest.approx(
        patch_delta.norm().item(), rel=1e-6
    )
    assert all(torch.equal(control[k], again[k]) for k in initial)


def test_graph_identity_ignores_the_source_for_split_exclusion() -> None:
    costs = torch.zeros(2, 17, 17, dtype=torch.long)
    costs[:, 0, 1] = 7
    packed = encode_weighted_graphs(costs, torch.tensor([0, 16]))
    assert len(matrix_keys(packed)) == 1
    costs[1, 0, 1] = 6
    assert len(matrix_keys(encode_weighted_graphs(costs, torch.tensor([0, 16])))) == 2
