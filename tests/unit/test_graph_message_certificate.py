"""Check continuous message-error bounds with known kinks and numeric witnesses."""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/weight_space"))
from certify_graph_repair import certify  # noqa: E402
from learned_weighted_graph_job import LearnedRelaxation  # noqa: E402
from repair_weighted_graph_job import addition_edit, from_state  # noqa: E402


def test_analytic_message_has_zero_error_on_the_entire_interval() -> None:
    state, _ = addition_edit(LearnedRelaxation(0).state_dict())
    result = certify(state)
    assert result["max_error_exact"] == "0"
    assert all(r["max_error_exact"] == "0" for r in result["per_cost"])


def test_first_layer_absolute_value_kink_has_known_quarter_error() -> None:
    state = {
        k: torch.zeros_like(v) for k, v in LearnedRelaxation(0).state_dict().items()
    }
    state["message.0.weight"][0, 0] = 1
    state["message.0.bias"][0] = -0.25
    state["message.0.weight"][1, 0] = -1
    state["message.0.bias"][1] = 0.25
    state["message.0.weight"][2, 1] = 1
    for i in range(3):
        state["message.2.weight"][i, i] = 1
        state["message.4.weight"][0, i] = 1
    # f(d,c)=|d-1/4|+c, so max |f-(d+c)| is exactly 1/4.
    result = certify(state)
    assert result["max_error_exact"] == "1/4"
    assert result["segments"] == 14


def test_second_layer_breakpoint_is_enumerated() -> None:
    state = {
        k: torch.zeros_like(v) for k, v in LearnedRelaxation(0).state_dict().items()
    }
    state["message.0.weight"][0, 0] = 1
    state["message.0.weight"][1, 1] = 1
    state["message.2.weight"][0, 0] = 1
    state["message.2.bias"][0] = -0.375
    state["message.2.weight"][1, 1] = 1
    state["message.4.weight"][0, :2] = 1
    # f(d,c)=ReLU(d-3/8)+c; there is no first-layer interior breakpoint.
    result = certify(state)
    assert result["max_error_exact"] == "3/8"
    assert result["segments"] == 14


def test_exact_bound_encloses_dense_probes_and_witness_matches_forward() -> None:
    state = LearnedRelaxation(7).state_dict()
    result = certify(state)
    model = from_state(state).double()
    distance = torch.linspace(0, 1, 257, dtype=torch.float64)
    cost = (torch.arange(1, 8).float() / 113).double()
    d, c = torch.meshgrid(distance, cost, indexing="ij")
    inputs = torch.stack((d.flatten(), c.flatten()), 1)
    witness = result["witness"]
    point = torch.tensor(
        [
            [
                float(Fraction(witness["distance"])),
                float(Fraction(witness["encoded_cost"])),
            ]
        ],
        dtype=torch.float64,
    )
    with torch.inference_mode():
        error = (model.message(inputs).squeeze(-1) - inputs.sum(1)).abs()
        witness_error = float((model.message(point).squeeze(-1) - point.sum(1)).abs())
    assert float(error.max()) <= result["max_error"] + 1e-12
    assert witness_error == pytest.approx(result["max_error"], abs=1e-12)
