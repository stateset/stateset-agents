"""Compile scalar affine-plus-hinge programs into existing dense ReLU MLPs.

The specification is supplied by the caller, not inferred from learned weights.
Selected hidden rows carry nonnegative signals through identity connections;
the entire scalar readout is replaced. Unselected branches remain stored but
do not contribute to the result. This is component replacement, not preservation
of a learned function. No optimizer, data, new layers, or custom forward is used.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from stateset_agents.core.errors import ConfigurationError

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class Hinge:
    """A term ``scale * ReLU(weights @ x + bias)``."""

    weights: tuple[float, ...]
    bias: float = 0.0
    scale: float = 1.0


@dataclass(frozen=True)
class ReLUProgram:
    """Scalar function ``linear @ x + bias + sum(hinges)``.

    Coefficients must be finite and exactly representable in float32. For
    example, 0.5 is accepted; the Python float 0.1 is rejected instead of silently
    quantized. Express an already quantized coefficient explicitly if desired.
    """

    linear: tuple[float, ...]
    bias: float = 0.0
    hinges: tuple[Hinge, ...] = ()


@dataclass(frozen=True)
class ScalarAssignment:
    """One assigned coordinate, including assignments already at their target."""

    tensor: str
    index: int
    before: float
    after: float


@dataclass(frozen=True)
class CompiledProgram:
    """A detached state dictionary and audit information; input model is untouched."""

    state_dict: dict[str, torch.Tensor]
    assignments: tuple[ScalarAssignment, ...]
    channels: int
    total_parameters: int

    @property
    def changed_parameters(self) -> int:
        """Count numerically changed coordinates, excluding no-op assignments."""
        return sum(a.before != a.after for a in self.assignments)


def _coefficient(value: float) -> None:
    try:
        finite = math.isfinite(value)
        exact = struct.unpack("f", struct.pack("f", value))[0] == value
    except (OverflowError, TypeError, struct.error) as exc:
        raise ConfigurationError("coefficients must be finite float32 values") from exc
    if not finite or not exact:
        raise ConfigurationError("coefficients must be finite, exact float32 values")


def _channels(
    program: ReLUProgram, domain: tuple[tuple[float, float], ...]
) -> list[Hinge]:
    dimensions = len(program.linear)
    if not dimensions or len(domain) != dimensions:
        raise ConfigurationError("domain must have one interval per input")
    for interval in domain:
        if len(interval) != 2:
            raise ConfigurationError("domain intervals must contain two bounds")
        low, high = interval
        try:
            valid = math.isfinite(low) and math.isfinite(high) and low <= high
        except (TypeError, OverflowError) as exc:
            raise ConfigurationError("domain bounds must be finite numbers") from exc
        if not valid:
            raise ConfigurationError("domain intervals must be finite and ordered")
    for value in (*program.linear, program.bias):
        _coefficient(value)
    for hinge in program.hinges:
        if len(hinge.weights) != dimensions:
            raise ConfigurationError("hinge input dimension does not match linear term")
        for value in (*hinge.weights, hinge.bias, hinge.scale):
            _coefficient(value)

    result: list[Hinge] = []
    if any(program.linear):
        # Rational bounds prevent floating cancellation from misclassifying a
        # linear form's sign and incorrectly discarding its negative channel.
        low_bound = sum(
            Fraction(w) * Fraction(lo if w >= 0 else hi)
            for w, (lo, hi) in zip(program.linear, domain, strict=True)
        )
        high_bound = sum(
            Fraction(w) * Fraction(hi if w >= 0 else lo)
            for w, (lo, hi) in zip(program.linear, domain, strict=True)
        )
        if low_bound >= 0:
            result.append(Hinge(program.linear))
        elif high_bound <= 0:
            result.append(Hinge(tuple(-w for w in program.linear), scale=-1.0))
        else:
            result.extend(
                (
                    Hinge(program.linear),
                    Hinge(tuple(-w for w in program.linear), scale=-1.0),
                )
            )
    result.extend(hinge for hinge in program.hinges if hinge.scale)
    return result


def compile_relu_program(
    model: torch.nn.Sequential,
    program: ReLUProgram,
    *,
    domain: tuple[tuple[float, float], ...],
) -> CompiledProgram:
    """Write a program into a vanilla float32 CPU Linear/ReLU MLP.

    Require one or more hidden Linear/ReLU pairs followed by a scalar Linear
    readout. All layers need biases, independent parameters, and enough width
    for the program's nonnegative channels. Named Sequential layers are allowed.
    Return a cloned state dictionary suitable for ``model.load_state_dict``.

    In real arithmetic the result equals the specified function on ``domain``:
    ReLU(z)-ReLU(-z)=z, nonnegative channels survive every identity/ReLU pair,
    and the readout forms the specified sum. This identity does not assert zero
    floating-point error. Inputs outside the domain are not covered. The dense
    forward still evaluates disconnected branches; a conservative interval guard
    rejects possible overflow, including there. No RNG state is consumed.
    """
    import torch
    from torch import nn

    if type(model) is not nn.Sequential:
        raise ConfigurationError("expected a vanilla nn.Sequential")
    modules = list(model._modules.items())
    if len(modules) < 3 or len(modules) % 2 != 1:
        raise ConfigurationError(
            "expected hidden Linear/ReLU pairs and a Linear output"
        )
    for index, (_, module) in enumerate(modules):
        if type(module) is not (nn.Linear if index % 2 == 0 else nn.ReLU):
            raise ConfigurationError(
                "only vanilla alternating Linear/ReLU is supported"
            )
    for module in model.modules():
        if module._forward_hooks or module._forward_pre_hooks:
            raise ConfigurationError("forward hooks are not supported")
    layers = [(name, layer) for name, layer in modules if isinstance(layer, nn.Linear)]
    if layers[0][1].in_features != len(program.linear):
        raise ConfigurationError("program input dimension does not match model")
    if layers[-1][1].out_features != 1:
        raise ConfigurationError("expected a scalar output")
    for (_, before), (_, after) in zip(layers[:-1], layers[1:], strict=True):
        if before.out_features != after.in_features:
            raise ConfigurationError("incompatible adjacent Linear dimensions")
    if any(layer.bias is None for _, layer in layers):
        raise ConfigurationError("all Linear layers require biases")
    features = _channels(program, domain)
    count = len(features)
    if any(layer.out_features < count for _, layer in layers[:-1]):
        raise ConfigurationError(f"each hidden layer needs at least {count} channels")

    initial = model.state_dict()
    expected = {f"{name}.{part}" for name, _ in layers for part in ("weight", "bias")}
    if set(initial) != expected:
        raise ConfigurationError("unexpected model state or buffers")
    storage = set()
    for name, tensor in initial.items():
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ConfigurationError(f"expected finite, contiguous float32 CPU {name}")
        pointer = tensor.untyped_storage().data_ptr()
        if pointer in storage:
            raise ConfigurationError("shared parameter storage is not supported")
        storage.add(pointer)
    for name, layer in layers:
        if tuple(initial[f"{name}.weight"].shape) != (
            layer.out_features,
            layer.in_features,
        ) or tuple(initial[f"{name}.bias"].shape) != (layer.out_features,):
            raise ConfigurationError("parameter shapes disagree with Linear dimensions")
    state = {name: value.detach().clone() for name, value in initial.items()}
    touched: dict[str, list[int]] = {}
    first = layers[0][0]
    for row, hinge in enumerate(features):
        state[f"{first}.weight"][row] = torch.tensor(
            hinge.weights, dtype=torch.float32, device="cpu"
        )
        state[f"{first}.bias"][row] = hinge.bias
    for name, layer in layers[:-1]:
        if name != first:
            state[f"{name}.weight"][:count] = 0
            for row in range(count):
                state[f"{name}.weight"][row, row] = 1
            state[f"{name}.bias"][:count] = 0
        touched[f"{name}.weight"] = list(range(count * layer.in_features))
        touched[f"{name}.bias"] = list(range(count))
    last = layers[-1][0]
    state[f"{last}.weight"].zero_()
    for row, hinge in enumerate(features):
        state[f"{last}.weight"][0, row] = hinge.scale
    state[f"{last}.bias"][0] = program.bias
    touched[f"{last}.weight"] = list(range(layers[-1][1].in_features))
    touched[f"{last}.bias"] = [0]

    # Factor-two inflation at every layer gives ample rounding headroom for
    # these bounded-size dense maps; reject very wide layers instead of assuming
    # that headroom covers arbitrary accumulation lengths. Include the readout.
    bound = torch.tensor(
        [max(abs(a), abs(b)) for a, b in domain], dtype=torch.float64, device="cpu"
    )
    limit = torch.finfo(torch.float32).max / 4
    if not bool((bound <= limit).all()):
        raise ConfigurationError("input domain can overflow float32")
    for name, layer in layers:
        if layer.in_features > 65536:
            raise ConfigurationError(
                "interval guard supports at most 65536 inputs/layer"
            )
        bound = 2 * (
            state[f"{name}.weight"].double().abs() @ bound
            + state[f"{name}.bias"].double().abs()
        )
        if not bool((bound <= limit).all()):
            raise ConfigurationError(
                "dense branches can overflow float32 on the domain"
            )
    ledger = tuple(
        ScalarAssignment(
            name,
            index,
            float(initial[name].view(-1)[index]),
            float(state[name].view(-1)[index]),
        )
        for name, indices in touched.items()
        for index in indices
    )
    return CompiledProgram(
        state, ledger, count, sum(t.numel() for t in initial.values())
    )
