"""Compile scalar affine-plus-hinge programs into existing dense ReLU MLPs.

The specification is supplied by the caller, not inferred from learned weights.
Selected hidden rows carry nonnegative signals through identity connections.
By default the entire scalar readout is replaced. An explicit output index
allows editing one output of a multi-output network, using only hidden rows
with no nonzero path to any other output. No optimizer, data, new layers, or
custom forward is used. Dense networks may have no such spare capacity.
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
    output_index: int = 0
    channel_rows: tuple[tuple[int, ...], ...] = ()
    protected_outputs: tuple[int, ...] = ()

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


def _linear_layers(
    model: torch.nn.Sequential, dimensions: int, output_index: int | None
) -> list[tuple[str, torch.nn.Linear]]:
    """Validate the inference architecture and return its named affine layers."""
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
    if layers[0][1].in_features != dimensions:
        raise ConfigurationError("program input dimension does not match model")
    outputs = layers[-1][1].out_features
    if output_index is None and outputs != 1:
        raise ConfigurationError("expected a scalar output")
    if output_index is not None and (
        type(output_index) is not int or not 0 <= output_index < outputs
    ):
        raise ConfigurationError("output_index must be an integer within the readout")
    for (_, before), (_, after) in zip(layers[:-1], layers[1:], strict=True):
        if before.out_features != after.in_features:
            raise ConfigurationError("incompatible adjacent Linear dimensions")
    if any(layer.bias is None for _, layer in layers):
        raise ConfigurationError("all Linear layers require biases")
    return layers


def _initial_state(
    model: torch.nn.Sequential,
    layers: list[tuple[str, torch.nn.Linear]],
) -> dict[str, torch.Tensor]:
    """Reject unsupported tensor layouts before dependency analysis or editing."""
    import torch

    probe = torch.tensor([1], dtype=torch.int32, device="cpu").view(torch.float32)
    if float(probe[0]) != 2**-149:
        raise ConfigurationError("subnormal flushing is not supported")
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
    return dict(initial)


def _channel_rows(
    state: dict[str, torch.Tensor],
    layers: list[tuple[str, torch.nn.Linear]],
    count: int,
    protected_outputs: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Select deterministic rows outside every protected output's ancestor set."""
    protected = list(protected_outputs)
    free_rows = []
    for name, _ in reversed(layers[1:]):
        # Exact nonzero edges, including subnormals: near-zero is not unused.
        used = (state[f"{name}.weight"][protected] != 0).any(dim=0)
        free_rows.append(tuple((~used).nonzero().flatten().tolist()[:count]))
        protected = used.nonzero().flatten().tolist()
    rows = tuple(reversed(free_rows))
    if any(len(indices) < count for indices in rows):
        raise ConfigurationError(
            f"each hidden layer needs at least {count} channels with no nonzero "
            "path to protected outputs"
        )
    return rows


def _guard_overflow(
    state: dict[str, torch.Tensor],
    layers: list[tuple[str, torch.nn.Linear]],
    domain: tuple[tuple[float, float], ...],
) -> None:
    """Bound all dense branches, including disconnected ones evaluated at runtime."""
    import torch

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


def compile_relu_program(
    model: torch.nn.Sequential,
    program: ReLUProgram,
    *,
    domain: tuple[tuple[float, float], ...],
    output_index: int | None = None,
) -> CompiledProgram:
    """Write a program into a vanilla float32 CPU Linear/ReLU MLP.

    Require hidden Linear/ReLU pairs and a Linear readout, with independent
    parameters and biases. Named Sequential layers are allowed. Return a cloned
    state dictionary; neither the input model nor RNG state is changed.

    With no ``output_index``, require a scalar readout and replace its function.
    Otherwise replace only that output and protect every other output. Follow
    all exact nonzero connections backward from protected outputs, then choose
    the lowest available hidden rows outside those paths. Reject insufficient
    capacity, including densely connected networks; no pruning is performed.
    Protected paths retain all their coefficients, so their real-arithmetic
    functions are unchanged, regardless of the training data or activations.

    In real arithmetic the selected output equals the program on ``domain``:
    ReLU(z)-ReLU(-z)=z and nonnegative channels survive identity/ReLU pairs.
    This does not promise zero floating-point error or equality outside the
    domain. Conservative interval bounds reject possible dense-branch overflow;
    when protecting outputs, both the original and edited network are checked.
    CPU modes that flush subnormal values to zero are rejected because they can
    hide stored nonzero connections from dependency and coefficient checks.
    """
    import torch

    layers = _linear_layers(model, len(program.linear), output_index)
    features = _channels(program, domain)
    count = len(features)
    initial = _initial_state(model, layers)
    target = 0 if output_index is None else output_index
    protected = tuple(i for i in range(layers[-1][1].out_features) if i != target)
    rows = _channel_rows(initial, layers, count, protected)
    if protected:
        _guard_overflow(initial, layers, domain)
    state = {name: value.detach().clone() for name, value in initial.items()}
    touched: dict[str, list[int]] = {}
    for depth, ((name, layer), indices) in enumerate(
        zip(layers[:-1], rows, strict=True)
    ):
        for channel, row in enumerate(indices):
            if depth == 0:
                state[f"{name}.weight"][row] = torch.tensor(
                    features[channel].weights, dtype=torch.float32, device="cpu"
                )
                state[f"{name}.bias"][row] = features[channel].bias
            else:
                state[f"{name}.weight"][row] = 0
                state[f"{name}.weight"][row, rows[depth - 1][channel]] = 1
                state[f"{name}.bias"][row] = 0
        touched[f"{name}.weight"] = [
            row * layer.in_features + col
            for row in indices
            for col in range(layer.in_features)
        ]
        touched[f"{name}.bias"] = list(indices)
    last, readout = layers[-1]
    state[f"{last}.weight"][target] = 0
    for row, hinge in zip(rows[-1], features, strict=True):
        state[f"{last}.weight"][target, row] = hinge.scale
    state[f"{last}.bias"][target] = program.bias
    touched[f"{last}.weight"] = list(
        range(target * readout.in_features, (target + 1) * readout.in_features)
    )
    touched[f"{last}.bias"] = [target]
    _guard_overflow(state, layers, domain)
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
        state,
        ledger,
        count,
        sum(t.numel() for t in initial.values()),
        target,
        rows,
        protected,
    )
