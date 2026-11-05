"""Check the compiler's identity, domain contract, and edit isolation."""

from __future__ import annotations

from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.experimental.weight_space.relu_program import (  # noqa: E402
    Hinge,
    ReLUProgram,
    compile_relu_program,
)


def model(widths: tuple[int, ...] = (8, 8)) -> torch.nn.Sequential:
    with torch.random.fork_rng():
        torch.manual_seed(721)
        modules = []
        inputs = 2
        for width in widths:
            modules.extend((torch.nn.Linear(inputs, width), torch.nn.ReLU()))
            inputs = width
        modules.append(torch.nn.Linear(inputs, 1))
    return torch.nn.Sequential(*modules)


@pytest.mark.parametrize("widths", [(4,), (8, 8), (9, 7, 5)])
@pytest.mark.parametrize("operation", ["add", "min", "max", "abs", "clip"])
def test_arithmetic_on_signed_grid(widths: tuple[int, ...], operation: str) -> None:
    specifications = {
        "add": ReLUProgram((1.0, 1.0)),
        "min": ReLUProgram((1.0, 0.0), hinges=(Hinge((1.0, -1.0), scale=-1),)),
        "max": ReLUProgram((0.0, 1.0), hinges=(Hinge((1.0, -1.0)),)),
        "abs": ReLUProgram((0.0, 0.0), hinges=(Hinge((1.0, -1.0)), Hinge((-1.0, 1.0)))),
        "clip": ReLUProgram(
            (0.0, 0.0), -1, (Hinge((1.0, -1.0), 1), Hinge((1.0, -1.0), -1, -1))
        ),
    }
    net = model(widths)
    result = compile_relu_program(net, specifications[operation], domain=((-4, 4),) * 2)
    net.load_state_dict(result.state_dict)
    x = torch.cartesian_prod(torch.arange(-16, 17) / 4, torch.arange(-16, 17) / 4)
    a, b = x.unbind(1)
    expected = {
        "add": a + b,
        "min": torch.minimum(a, b),
        "max": torch.maximum(a, b),
        "abs": (a - b).abs(),
        "clip": (a - b).clamp(-1, 1),
    }[operation]
    with torch.inference_mode():
        assert torch.equal(net(x).squeeze(1), expected)


def test_edit_does_not_mutate_model_or_rng_and_accounts_for_every_change() -> None:
    net = model((32, 32))
    initial = {k: v.clone() for k, v in net.state_dict().items()}
    rng = torch.get_rng_state().clone()
    compiled = compile_relu_program(net, ReLUProgram((1, 1)), domain=((0, 1),) * 2)
    assert compiled.channels == 1
    assert len(compiled.assignments) == compiled.changed_parameters == 69
    assert compiled.total_parameters == 1185
    assert torch.equal(rng, torch.get_rng_state())
    allowed = {(a.tensor, a.index) for a in compiled.assignments}
    assert len(allowed) == 69
    for name, before in initial.items():
        assert torch.equal(before, net.state_dict()[name])
        after = compiled.state_dict[name]
        for index in (before.flatten() != after.flatten()).nonzero().flatten().tolist():
            assert (name, index) in allowed
        assert before.data_ptr() != after.data_ptr()
    for a in compiled.assignments:
        assert float(initial[a.tensor].flatten()[a.index]) == a.before
        assert float(compiled.state_dict[a.tensor].flatten()[a.index]) == a.after
    net.load_state_dict(compiled.state_dict)
    again = compile_relu_program(net, ReLUProgram((1, 1)), domain=((0, 1),) * 2)
    assert len(again.assignments) == 69
    assert again.changed_parameters == 0


@pytest.mark.parametrize(
    "domain, channels", [(((-4, 4),) * 2, 2), (((0, 4),) * 2, 1), (((-4, 0),) * 2, 1)]
)
def test_sign_optimization(domain: tuple, channels: int) -> None:
    net = model()
    result = compile_relu_program(net, ReLUProgram((1, 1)), domain=domain)
    assert result.channels == channels
    net.load_state_dict(result.state_dict)
    x = torch.tensor(
        [[d[0] for d in domain], [d[1] for d in domain]], dtype=torch.float32
    )
    assert torch.equal(net(x).squeeze(1), x.sum(1))


def test_constant_and_named_layers() -> None:
    net = torch.nn.Sequential(
        OrderedDict(zip(("input", "relu", "out"), model((4,)), strict=True))
    )
    result = compile_relu_program(net, ReLUProgram((0, 0), -3), domain=((-1, 1),) * 2)
    assert result.channels == 0
    assert len(result.assignments) == 5
    assert torch.equal(result.state_dict["input.weight"], net[0].weight)
    net.load_state_dict(result.state_dict)
    assert torch.equal(net(torch.zeros(7, 2)), torch.full((7, 1), -3.0))


@pytest.mark.parametrize(
    "damage",
    [
        "narrow",
        "dtype",
        "nan",
        "overflow",
        "activation",
        "shared",
        "no_bias",
        "multi_output",
        "hooks",
        "shape",
    ],
)
def test_invalid_network_rejected_without_mutation(damage: str) -> None:
    net = model()
    if damage == "narrow":
        net = model((1,))
    elif damage == "dtype":
        net.double()
    elif damage == "nan":
        with torch.no_grad():
            net[0].weight[1, 0] = float("nan")
    elif damage == "overflow":
        with torch.no_grad():
            net[0].weight[7, 0] = 3e38
    elif damage == "activation":
        net[1] = torch.nn.Sigmoid()
    elif damage == "shared":
        net[2].bias = net[0].bias
    elif damage == "no_bias":
        net[0] = torch.nn.Linear(2, 8, bias=False)
    elif damage == "multi_output":
        net[-1] = torch.nn.Linear(8, 2)
    elif damage == "hooks":
        net.register_forward_hook(lambda module, args, output: output + 1)
    elif damage == "shape":
        net[0].weight = torch.nn.Parameter(torch.ones(8, 3))
    before = {k: v.clone() for k, v in net.state_dict().items()}
    with pytest.raises(ConfigurationError):
        compile_relu_program(net, ReLUProgram((1, 1)), domain=((-4, 4),) * 2)
    for name, tensor in net.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize(
    "program, domain",
    [
        (ReLUProgram((0.1, 1)), ((0, 1),) * 2),
        (ReLUProgram((float("inf"), 1)), ((0, 1),) * 2),
        (ReLUProgram((1,)), ((0, 1),)),
        (ReLUProgram((1, 1), hinges=(Hinge((1,)),)), ((0, 1),) * 2),
        (ReLUProgram((1, 1)), ((1, 0),) * 2),
        (ReLUProgram((1, 1)), ((0, float("inf")),) * 2),
        (ReLUProgram((1, 1)), ((0, 1),)),
        (ReLUProgram((1, 1)), ((0, 1, 2), (0, 1))),
        (ReLUProgram((1, 1)), (("zero", 1), (0, 1))),
    ],
)
def test_invalid_specifications(program: ReLUProgram, domain: tuple) -> None:
    with pytest.raises(ConfigurationError):
        compile_relu_program(model(), program, domain=domain)


def test_first_layer_coefficients_do_not_depend_on_default_dtype() -> None:
    net = model()
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        result = compile_relu_program(
            net, ReLUProgram((1 + 2**-20, 1)), domain=((0, 1),) * 2
        )
    finally:
        torch.set_default_dtype(original)
    assert float(result.state_dict["0.weight"][0, 0]) == 1 + 2**-20
