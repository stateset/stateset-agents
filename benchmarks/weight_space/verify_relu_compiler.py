"""Independently check stored ReLU programs using rational algebra and replay.

The symbolic checker does not call the compiler. It follows nonzero readout
connections backward through identity/ReLU layers, extracts first-layer hinges,
and compares their rational normal form with the specification on its domain.
It certifies real-arithmetic identities, not zero floating-point execution error.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path

import torch
from safetensors.torch import load_file


def require(condition: bool, message: str) -> None:
    """Keep verification active with python -O as well."""
    if not condition:
        raise ValueError(message)


def normal_form(linear: list, bias: float, hinges: list, domain: list) -> tuple:
    """Canonicalize affine and opposite hinge terms with exact rational bounds."""
    affine = [Fraction(v) for v in [*linear, bias]]
    terms: dict[tuple, Fraction] = {}
    for weights, offset, scale in hinges:
        key = tuple(Fraction(v) for v in [*weights, offset])
        factor = Fraction(scale)
        low = key[-1] + sum(
            w * Fraction(lo if w >= 0 else hi)
            for w, (lo, hi) in zip(key[:-1], domain, strict=True)
        )
        high = key[-1] + sum(
            w * Fraction(hi if w >= 0 else lo)
            for w, (lo, hi) in zip(key[:-1], domain, strict=True)
        )
        if low >= 0:
            affine = [a + factor * w for a, w in zip(affine, key, strict=True)]
        elif high > 0:
            opposite = tuple(-v for v in key)
            if key < opposite:
                # ReLU(-z) = ReLU(z) - z, including the affine offset.
                key = opposite
                affine = [a - factor * w for a, w in zip(affine, key, strict=True)]
            terms[key] = terms.get(key, Fraction(0)) + factor
    return tuple(affine), {k: v for k, v in terms.items() if v}


def _checked_layers(state: dict[str, torch.Tensor]) -> list[int]:
    """Validate serialized vanilla Sequential dimensions independently."""
    probe = torch.tensor([1], dtype=torch.int32, device="cpu").view(torch.float32)
    require(float(probe[0]) == 2**-149, "subnormal flushing is not supported")
    indices = sorted(int(k.split(".")[0]) for k in state if k.endswith(".weight"))
    require(indices == list(range(0, len(indices) * 2, 2)), "unexpected layer indices")
    require(len(indices) >= 2, "need a hidden layer")
    require(
        set(state) == {f"{i}.{part}" for i in indices for part in ("weight", "bias")},
        "unexpected state keys",
    )
    for value in state.values():
        require(
            value.device.type == "cpu"
            and value.dtype == torch.float32
            and bool(torch.isfinite(value).all()),
            "invalid parameters",
        )
    for position, index in enumerate(indices):
        weights, bias = state[f"{index}.weight"], state[f"{index}.bias"]
        require(
            weights.ndim == 2
            and min(weights.shape) > 0
            and bias.shape == (weights.shape[0],),
            "invalid affine dimensions",
        )
        if position:
            require(
                weights.shape[1] == state[f"{indices[position - 1]}.weight"].shape[0],
                "incompatible adjacent dimensions",
            )
    return indices


def certify(
    state: dict[str, torch.Tensor],
    program: dict,
    domain: list,
    *,
    output_index: int | None = None,
) -> dict:
    """Check one output's complete reachable circuit against its specification."""
    indices = _checked_layers(state)
    output = indices[-1]
    outputs = state[f"{output}.weight"].shape[0]
    if output_index is None:
        require(outputs == 1, "not a scalar output")
        output_index = 0
    require(
        type(output_index) is int and 0 <= output_index < outputs,
        "invalid output index",
    )
    require(state["0.weight"].shape[1] == len(domain), "invalid domain dimension")
    active = {
        i: Fraction(float(v))
        for i, v in enumerate(state[f"{output}.weight"][output_index])
        if v != 0
    }
    bias = float(state[f"{output}.bias"][output_index])
    for index in reversed(indices[1:-1]):
        previous: dict[int, Fraction] = {}
        for row, scale in active.items():
            weights = state[f"{index}.weight"][row]
            nonzero = weights.nonzero().flatten().tolist()
            require(len(nonzero) == 1, "reachable hidden row is not an identity wire")
            col = nonzero[0]
            require(
                float(weights[col]) == 1 and float(state[f"{index}.bias"][row]) == 0,
                "reachable hidden row changes its signal",
            )
            previous[col] = previous.get(col, Fraction(0)) + scale
        active = {k: v for k, v in previous.items() if v}
    actual_hinges = [
        (state["0.weight"][row].tolist(), float(state["0.bias"][row]), scale)
        for row, scale in active.items()
    ]
    expected_hinges = [(h["weights"], h["bias"], h["scale"]) for h in program["hinges"]]
    actual = normal_form([0] * len(domain), bias, actual_hinges, domain)
    expected = normal_form(program["linear"], program["bias"], expected_hinges, domain)
    require(actual == expected, "symbolic function differs from the program")
    return {
        "real_arithmetic_identity": True,
        "reachable_first_layer_units": len(active),
        "remaining_hinge_terms": len(actual[1]),
    }


def certify_preservation(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    *,
    output_index: int,
) -> dict:
    """Prove unchanged real-arithmetic functions for all other output rows.

    Walk backward through every nonzero edge from protected outputs. Every
    reachable affine row (including its zero edges and bias) must be identical.
    This independent check reads saved tensors, not the compiler's assignment
    ledger, chosen channels, training data, or claimed certificate. It assumes
    the serialized vanilla Linear/ReLU architecture and finite arithmetic;
    floating-point overflow and empirical equality are separate checks.
    """
    indices = _checked_layers(before)
    require(_checked_layers(after) == indices, "layer structure changed")
    for name, value in before.items():
        require(value.shape == after[name].shape, "parameter shape changed")
    outputs = before[f"{indices[-1]}.weight"].shape[0]
    require(
        type(output_index) is int and 0 <= output_index < outputs,
        "invalid output index",
    )
    protected = [i for i in range(outputs) if i != output_index]
    rows = protected
    reachable = {}
    parameters = 0
    for index in reversed(indices):
        reachable[str(index)] = list(rows)
        for part in ("weight", "bias"):
            old = before[f"{index}.{part}"][rows]
            new = after[f"{index}.{part}"][rows]
            require(torch.equal(old, new), f"protected {index}.{part} changed")
            parameters += old.numel()
        weights = before[f"{index}.weight"]
        rows = [
            col
            for col in range(weights.shape[1])
            if any(float(weights[row, col]) != 0 for row in rows)
        ]
    return {
        "real_arithmetic_preservation": True,
        "protected_outputs": protected,
        "reachable_rows": reachable,
        "unchanged_reachable_parameters": parameters,
    }


def oracle(name: str, row: list[float]) -> float:
    """Python arithmetic oracle independent of the job's torch formulas."""
    a, b = row[:2]
    if name in ("signed_add", "nonnegative_add"):
        return a + b
    if name == "minimum":
        return min(a, b)
    if name == "maximum":
        return max(a, b)
    if name == "absolute_difference":
        return abs(a - b)
    if name == "clipped_difference":
        return min(1.0, max(-1.0, a - b))
    if name == "triangle":
        return max(0.0, 1 - abs(a))
    if name == "manhattan_distance":
        return abs(a - row[2]) + abs(b - row[3])
    raise ValueError(name)


def model(
    dimensions: int, widths: list[int], state: dict, *, outputs: int = 1
) -> torch.nn.Sequential:
    """Rebuild ordinary Linear/ReLU inference from the recorded architecture."""
    layers = []
    for width in widths:
        layers.extend(
            (torch.nn.Linear(dimensions, width, device="meta"), torch.nn.ReLU())
        )
        dimensions = width
    layers.append(torch.nn.Linear(dimensions, outputs, device="meta"))
    net = torch.nn.Sequential(*layers).to_empty(device="cpu")
    net.load_state_dict(state)
    return net.eval()


def edit_counts(base: dict, edited: dict, channels: int) -> tuple[int, int]:
    """Check all unassigned coordinates and count assignments independently."""
    layers = sorted(int(k.split(".")[0]) for k in edited if k.endswith(".weight"))
    assigned, changed = 0, 0
    require(base.keys() == edited.keys(), "state keys changed")
    for layer in layers:
        for part in ("weight", "bias"):
            key = f"{layer}.{part}"
            before, after = base[key], edited[key]
            require(before.shape == after.shape, "parameter shape changed")
            end = before.shape[0] if layer == layers[-1] else channels
            require(torch.equal(before[end:], after[end:]), "unassigned row changed")
            assigned += before[:end].numel()
            changed += int((before != after).sum())
    return assigned, changed


@torch.inference_mode()
def verify(root: Path, graph: Path) -> dict:
    """Verify hashes, every condition, all predictions, identities and graph reuse."""
    manifest = json.loads((root / "artifact_manifest.json").read_text())
    for name, expected in manifest.items():
        require(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
            f"hash mismatch: {name}",
        )
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    for name, expected in protocol["sources"].items():
        require(
            hashlib.sha256((root / "sources" / name).read_bytes()).hexdigest()
            == expected,
            "source mismatch",
        )
    for name, expected in protocol["graph_dependencies"].items():
        require(
            hashlib.sha256((graph / name).read_bytes()).hexdigest() == expected,
            "graph dependency mismatch",
        )
    tensors = load_file(str(root / "checkpoints.safetensors"))
    cases = load_file(str(root / "inputs.safetensors"))
    expected_conditions = {
        (name, tuple(widths), seed)
        for name, widths, seed in itertools.product(
            protocol["tasks"], protocol["architectures"], protocol["seeds"]
        )
    }
    observed = [
        (row["task"], tuple(row["widths"]), row["seed"]) for row in summary["variants"]
    ]
    require(
        len(observed) == len(set(observed)) and set(observed) == expected_conditions,
        "conditions missing or duplicated",
    )
    truths = {
        name: torch.tensor(
            [oracle(name, x) for x in inputs.tolist()], dtype=torch.float64
        )
        for name, inputs in cases.items()
    }
    certificates = []
    total = 0
    maxima = []
    for row in summary["variants"]:
        key, name = row["key"], row["task"]
        extracted = {}
        for kind in ("base", "edited"):
            prefix = f"{key}.{kind}."
            extracted[kind] = {
                k.removeprefix(prefix): v
                for k, v in tensors.items()
                if k.startswith(prefix)
            }
        base, state = extracted["base"], extracted["edited"]
        task = protocol["tasks"][name]
        certificate = certify(state, task["program"], task["domain"])
        assigned, changed = edit_counts(base, state, row["channels"])
        require(
            (assigned, changed) == (row["assigned"], row["changed"]),
            "edit counts mismatch",
        )
        require(
            sum(v.numel() for v in state.values()) == row["parameters"],
            "parameter count mismatch",
        )
        net = model(len(task["domain"]), row["widths"], state)
        error = (net(cases[name]).squeeze(1).double() - truths[name]).abs()
        require(bool(torch.isfinite(error).all()), "non-finite replay")
        grid = row["grid_cases"]
        require(
            len(error) == row["cases"]
            and int((error[:grid] == 0).sum()) == row["grid_exact"] == grid,
            "grid replay failed",
        )
        require(
            int((error <= protocol["absolute_tolerance"]).sum())
            == row["within_tolerance"]
            == len(error),
            "floating replay failed",
        )
        require(
            float(error.max()) == row["max_abs_error"]
            and float(error.mean()) == row["mean_abs_error"],
            "saved metrics mismatch",
        )
        maxima.append(float(error.max()))
        total += len(error)
        certificates.append({"key": key, **certificate})
    require(
        total == summary["total_predictions"]
        and max(maxima) == summary["max_abs_error"],
        "aggregate metrics mismatch",
    )
    require(
        len(summary["graph_compatibility"]) == len(protocol["seeds"]),
        "missing graph seed",
    )
    for row in summary["graph_compatibility"]:
        seed = row["seed"]
        prefix = f"graph{seed}."
        compiled = {
            k.removeprefix(prefix): v
            for k, v in tensors.items()
            if k.startswith(prefix)
        }
        expected = {
            k.removeprefix("message."): v
            for k, v in load_file(str(graph / f"edited_seed{seed}.safetensors")).items()
        }
        base = {
            k.removeprefix("message."): v
            for k, v in load_file(
                str(graph / f"original_seed{seed}.safetensors")
            ).items()
        }
        require(
            compiled.keys() == expected.keys()
            and all(torch.equal(compiled[k], expected[k]) for k in compiled),
            "graph compiler compatibility failed",
        )
        require(edit_counts(base, compiled, 1) == (69, 69), "graph edit count mismatch")
        task = protocol["tasks"]["nonnegative_add"]
        certify(compiled, task["program"], task["domain"])
    return {
        "passed": True,
        "variants": len(certificates),
        "predictions_replayed": total,
        "graph_checkpoints_equal": len(summary["graph_compatibility"]),
        "max_abs_error": max(maxima),
        "certificates": certificates,
    }


def main() -> None:
    """Replay retained artifacts and emit deterministic verification evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--graph-evidence", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = verify(args.root, args.graph_evidence)
    (args.root / "verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "certificates"}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
