"""Exact real-arithmetic message-error bounds by rational ReLU breakpoints.

For each of the seven encoded edge costs, the two-input MLP becomes a continuous
piecewise-affine function of distance on [0,1]. Enumerating all first- and second-
layer breakpoints makes its maximum addition error exact in real arithmetic.
This does not certify floating-point execution of arbitrary learned networks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import torch
from repair_weighted_graph_job import SHAPES, State, graphs
from safetensors.torch import load_file

from stateset_agents.core.errors import ConfigurationError


def fraction(value: torch.Tensor) -> Fraction:
    """Represent a stored binary floating-point value without decimal rounding."""
    return Fraction.from_float(float(value))


def certify(state: State) -> dict:
    """Bound real-arithmetic addition error for every distance and real edge cost."""
    if set(state) != set(SHAPES):
        raise ConfigurationError("unexpected message parameter names")
    for name, shape in SHAPES.items():
        tensor = state[name]
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ConfigurationError(f"invalid float32 CPU parameter: {name}")
    w1 = [[fraction(v) for v in row] for row in state["message.0.weight"]]
    b1 = [fraction(v) for v in state["message.0.bias"]]
    w2 = [[fraction(v) for v in row] for row in state["message.2.weight"]]
    b2 = [fraction(v) for v in state["message.2.bias"]]
    w3 = [fraction(v) for v in state["message.4.weight"][0]]
    b3 = fraction(state["message.4.bias"][0])
    worst = Fraction(-1)
    witness: dict = {}
    pieces = 0
    per_cost = []
    for cost in range(1, graphs.MAX_COST + 1):
        # Match the float32 normalization at the existing graph-model boundary.
        c = fraction(torch.tensor(float(cost)) / graphs.SENTINEL)
        offsets = [w[1] * c + bias for w, bias in zip(w1, b1, strict=True)]
        boundaries = {Fraction(0), Fraction(1)}
        for w, offset in zip(w1, offsets, strict=True):
            if w[0]:
                root = -offset / w[0]
                if 0 < root < 1:
                    boundaries.add(root)
        boundaries = sorted(boundaries)
        cost_max = Fraction(0)
        cost_pieces = 0
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            midpoint = (left + right) / 2
            active = [w[0] * midpoint + o > 0 for w, o in zip(w1, offsets, strict=True)]
            slopes = [
                w[0] if on else Fraction(0) for w, on in zip(w1, active, strict=True)
            ]
            intercepts = [
                o if on else Fraction(0) for o, on in zip(offsets, active, strict=True)
            ]
            slopes2 = [
                sum(w * s for w, s in zip(row, slopes, strict=True)) for row in w2
            ]
            intercepts2 = [
                sum(w * o for w, o in zip(row, intercepts, strict=True)) + b
                for row, b in zip(w2, b2, strict=True)
            ]
            inner = {left, right}
            for slope, intercept in zip(slopes2, intercepts2, strict=True):
                if slope:
                    root = -intercept / slope
                    if left < root < right:
                        inner.add(root)
            inner = sorted(inner)
            for lo, hi in zip(inner[:-1], inner[1:], strict=True):
                midpoint2 = (lo + hi) / 2
                active2 = [
                    s * midpoint2 + o > 0
                    for s, o in zip(slopes2, intercepts2, strict=True)
                ]
                out_slope = sum(
                    w * s for w, s, on in zip(w3, slopes2, active2, strict=True) if on
                )
                out_offset = b3 + sum(
                    w * o
                    for w, o, on in zip(w3, intercepts2, active2, strict=True)
                    if on
                )
                cost_pieces += 1
                for d in (lo, hi):
                    output = out_slope * d + out_offset
                    error = abs(output - (d + c))
                    cost_max = max(cost_max, error)
                    if error > worst:
                        worst = error
                        witness = {
                            "cost": cost,
                            "encoded_cost": str(c),
                            "distance": str(d),
                            "output": str(output),
                            "expected": str(d + c),
                        }
        pieces += cost_pieces
        per_cost.append(
            {
                "cost": cost,
                "segments": cost_pieces,
                "max_error_exact": str(cost_max),
                "max_error": float(cost_max),
            }
        )
    return {
        "max_error_exact": str(worst),
        "max_error": float(worst),
        "segments": pieces,
        "per_cost": per_cost,
        "witness": witness,
    }


def main() -> None:
    """Analyze already selected checkpoints; never feed bounds into selection."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    a = ap.parse_args()
    summary = json.loads((a.root / "summary.json").read_text())
    rows = []
    for variant in summary["variants"]:
        certificate = certify(load_file(str(a.root / variant["checkpoint"])))
        rows.append(
            {
                "name": variant["name"],
                "seed": variant["seed"],
                "checkpoint": variant["checkpoint"],
                **certificate,
            }
        )
        print(
            json.dumps(
                {
                    "name": variant["name"],
                    "seed": variant["seed"],
                    "max_error": certificate["max_error"],
                    "segments": certificate["segments"],
                }
            ),
            flush=True,
        )
    result = {
        "method": "exact Fraction arithmetic, all ReLU breakpoints on distance [0,1] for seven encoded costs",
        "scope": "real-arithmetic component error; not a floating-point execution certificate",
        "used_for_model_selection": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "variants": rows,
    }
    (a.root / "certificates.json").write_text(json.dumps(result, indent=2) + "\n")
    (a.root / "sources" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
