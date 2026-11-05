"""A separately implemented symbolic check must detect incorrect compiled weights."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.experimental.weight_space.relu_program import (  # noqa: E402
    ReLUProgram,
    compile_relu_program,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/weight_space"))
from relu_compiler_job import make_model, programs  # noqa: E402
from verify_relu_compiler import certify, normal_form  # noqa: E402


@pytest.mark.parametrize("name", list(programs()))
def test_exact_identity_on_whole_declared_box(name: str) -> None:
    program, domain = programs()[name]
    result = compile_relu_program(
        make_model(len(domain), (8, 8, 8), 0), program, domain=domain
    )
    certificate = certify(result.state_dict, asdict(program), domain)
    assert certificate["real_arithmetic_identity"] is True


@pytest.mark.parametrize("damage", ["first", "wire", "readout", "bias", "domain"])
def test_checker_rejects_wrong_functions(damage: str) -> None:
    program = ReLUProgram((1, 1))
    domain = ((0, 1),) * 2
    result = compile_relu_program(make_model(2, (8, 8), 0), program, domain=domain)
    state = result.state_dict
    if damage == "first":
        state["0.weight"][0, 0] = 2
    elif damage == "wire":
        state["2.weight"][0, 1] = 0.5
    elif damage == "readout":
        state["4.weight"][0, 0] = -1
    elif damage == "bias":
        state["4.bias"][0] = 1
    elif damage == "domain":
        domain = ((-1, 1),) * 2
    with pytest.raises(ValueError):
        certify(state, asdict(program), domain)


def test_symbolic_normalization_handles_offsets_opposites_and_inactive_hinges() -> None:
    # ReLU(x+1)-ReLU(-x-1) = x+1; ReLU(x-10)=0 on this domain.
    actual = normal_form(
        [0], 0, [([1], 1, 1), ([-1], -1, -1), ([1], -10, 77)], [(-4, 4)]
    )
    expected = normal_form([1], 1, [], [(-4, 4)])
    assert actual == expected
