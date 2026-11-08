"""Selective arithmetic edits preserve complete unrelated output dependency paths."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")

from benchmarks.weight_space.verify_relu_compiler import (  # noqa: E402
    certify,
    certify_preservation,
)
from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space.relu_program import (  # noqa: E402
    Hinge,
    ReLUProgram,
    compile_relu_program,
)

DOMAIN = ((-4.0, 4.0),) * 2
MAXIMUM = ReLUProgram((0, 1), hinges=(Hinge((1, -1)),))


def sparse_model() -> torch.nn.Sequential:
    """Create noncontiguous available rows that differ between hidden layers."""
    with torch.random.fork_rng():
        torch.manual_seed(8421)
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 7),
            torch.nn.ReLU(),
            torch.nn.Linear(7, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 3),
        )
    with torch.no_grad():
        # The protected outputs depend on last-hidden rows 1, 3, 5. Those
        # rows in turn depend on first-hidden rows 0, 2, 4, 6.
        model[4].weight[0, [0, 2, 4]] = 0
        model[4].weight[2, [0, 2, 4]] = 0
        for row in (1, 3, 5):
            model[2].weight[row, [1, 3, 5]] = 0
    return model.eval()


def test_selective_edit_preserves_other_functions_and_exact_assignment_ledger() -> None:
    model = sparse_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.cartesian_prod(torch.linspace(-4, 4, 33), torch.linspace(-4, 4, 33))
    baseline = model(x).detach()
    rng = torch.get_rng_state().clone()
    result = compile_relu_program(model, MAXIMUM, domain=DOMAIN, output_index=1)
    assert result.channel_rows == ((1, 3, 5), (0, 2, 4))
    assert result.protected_outputs == (0, 2)
    assert result.output_index == 1
    assert torch.equal(rng, torch.get_rng_state())
    touched = {(a.tensor, a.index) for a in result.assignments}
    assert len(touched) == len(result.assignments) == 40
    for name, original in before.items():
        assert torch.equal(original, model.state_dict()[name])
        edited = result.state_dict[name]
        changes = (original.flatten() != edited.flatten()).nonzero().flatten()
        assert all((name, int(i)) in touched for i in changes)
        assert original.data_ptr() != edited.data_ptr()
    for assignment in result.assignments:
        assert (
            float(before[assignment.tensor].flatten()[assignment.index])
            == assignment.before
        )
        assert (
            float(result.state_dict[assignment.tensor].flatten()[assignment.index])
            == assignment.after
        )
    model.load_state_dict(result.state_dict)
    actual = model(x).detach()
    assert torch.equal(actual[:, [0, 2]], baseline[:, [0, 2]])
    assert torch.equal(actual[:, 1], x.max(dim=1).values)
    assert not torch.equal(actual[:, 1], baseline[:, 1])
    assert certify(result.state_dict, asdict(MAXIMUM), DOMAIN, output_index=1)[
        "real_arithmetic_identity"
    ]
    proof = certify_preservation(before, result.state_dict, output_index=1)
    assert proof["protected_outputs"] == [0, 2]
    assert proof["reachable_rows"]["0"] == [0, 2, 4, 6]


@pytest.mark.parametrize("damage", ["direct", "indirect", "subnormal"])
def test_connected_capacity_is_rejected_without_mutation(damage: str) -> None:
    model = sparse_model()
    with torch.no_grad():
        if damage == "direct":
            model[4].weight[0, 0] = 0.25
        else:
            # Row 1 contributes to protected outputs; a tiny nonzero edge
            # to spare row 1 is sufficient to disqualify that first-layer row.
            model[2].weight[1, 1] = 2**-149 if damage == "subnormal" else 0.25
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ConfigurationError, match="no nonzero path"):
        compile_relu_program(model, MAXIMUM, domain=DOMAIN, output_index=1)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_dense_network_rejects_program_but_accepts_constant_without_hidden_edits() -> (
    None
):
    with torch.random.fork_rng():
        torch.manual_seed(82)
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        )
    with pytest.raises(ConfigurationError, match="no nonzero path"):
        compile_relu_program(model, MAXIMUM, domain=DOMAIN, output_index=2)
    result = compile_relu_program(
        model, ReLUProgram((0, 0), 7), domain=DOMAIN, output_index=2
    )
    assert result.channels == 0
    assert result.channel_rows == ((),)
    assert len(result.assignments) == 9
    assert torch.equal(result.state_dict["0.weight"], model[0].weight)
    assert certify_preservation(model.state_dict(), result.state_dict, output_index=2)[
        "real_arithmetic_preservation"
    ]


@pytest.mark.parametrize("index", [-1, 3, True, 1.0, "1"])
def test_invalid_output_index_rejected(index: object) -> None:
    with pytest.raises(ConfigurationError, match="output_index"):
        compile_relu_program(sparse_model(), MAXIMUM, domain=DOMAIN, output_index=index)


def test_named_layers_and_scalar_explicit_index() -> None:
    model = torch.nn.Sequential(
        OrderedDict(
            zip(("in", "relu1", "hidden", "relu2", "out"), sparse_model(), strict=True)
        )
    )
    result = compile_relu_program(model, MAXIMUM, domain=DOMAIN, output_index=1)
    assert result.assignments[0].tensor == "in.weight"
    scalar = torch.nn.Sequential(
        torch.nn.Linear(2, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1)
    )
    default = compile_relu_program(scalar, MAXIMUM, domain=DOMAIN)
    explicit = compile_relu_program(scalar, MAXIMUM, domain=DOMAIN, output_index=0)
    assert default.assignments == explicit.assignments
    assert all(
        torch.equal(v, explicit.state_dict[k]) for k, v in default.state_dict.items()
    )


def test_original_dense_overflow_rejected_even_when_edit_would_remove_it() -> None:
    model = sparse_model()
    with torch.no_grad():
        model[0].weight[1, 0] = 3e38
    with pytest.raises(ConfigurationError, match="overflow"):
        compile_relu_program(model, MAXIMUM, domain=DOMAIN, output_index=1)


@pytest.mark.parametrize(
    "damage", ["hidden", "first_bias", "readout", "new_edge", "shape", "nan"]
)
def test_independent_certificate_rejects_protected_path_changes(damage: str) -> None:
    model = sparse_model()
    before = model.state_dict()
    after = compile_relu_program(
        model, MAXIMUM, domain=DOMAIN, output_index=1
    ).state_dict
    if damage == "hidden":
        after["2.weight"][1, 0] += 1
    elif damage == "first_bias":
        after["0.bias"][0] += 1
    elif damage == "readout":
        after["4.bias"][2] += 1
    elif damage == "new_edge":
        after["2.weight"][1, 1] = 2**-149
    elif damage == "shape":
        after["0.weight"] = after["0.weight"][:, :1]
    elif damage == "nan":
        after["4.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        certify_preservation(before, after, output_index=1)


def test_output_identity_and_preservation_are_independent_checks() -> None:
    model = sparse_model()
    after = compile_relu_program(
        model, MAXIMUM, domain=DOMAIN, output_index=1
    ).state_dict
    after["4.bias"][1] += 1
    assert certify_preservation(model.state_dict(), after, output_index=1)[
        "real_arithmetic_preservation"
    ]
    with pytest.raises(ValueError, match="symbolic function"):
        certify(after, asdict(MAXIMUM), DOMAIN, output_index=1)


def test_flushing_subnormals_cannot_hide_stored_connections() -> None:
    model = sparse_model()
    before = model.state_dict()
    after = compile_relu_program(
        model, MAXIMUM, domain=DOMAIN, output_index=1
    ).state_dict
    probe = torch.tensor([1], dtype=torch.int32).view(torch.float32)
    originally_flushing = float(probe[0]) == 0
    try:
        if not torch.set_flush_denormal(True):
            pytest.skip("CPU does not support configuring denormal handling")
        with pytest.raises(ConfigurationError, match="subnormal flushing"):
            compile_relu_program(model, MAXIMUM, domain=DOMAIN, output_index=1)
        with pytest.raises(ValueError, match="subnormal flushing"):
            certify_preservation(before, after, output_index=1)
        with pytest.raises(ValueError, match="subnormal flushing"):
            certify(after, asdict(MAXIMUM), DOMAIN, output_index=1)
    finally:
        torch.set_flush_denormal(originally_flushing)
