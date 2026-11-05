"""Exercise one arithmetic compiler across programs, MLP shapes, and initial states.

This tests compilation, not a new comparison with training. Most initial states
are random; the three previously trained graph models are compatibility checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from stateset_agents.experimental.weight_space.relu_program import (
    Hinge,
    ReLUProgram,
    compile_relu_program,
)

LOGGER = logging.getLogger(__name__)
ARCHITECTURES = ((4,), (8, 8), (32, 32), (24, 16, 8))
SEEDS = (0, 1, 2)
TOLERANCE = 2e-6


def digest(path: Path) -> str:
    """Hash an evidence or source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write deterministic, readable evidence."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def programs() -> dict[str, tuple[ReLUProgram, tuple]]:
    """Fixed task specifications supplied analytically, before evaluation."""
    signed = ((-4.0, 4.0),) * 2
    return {
        "nonnegative_add": (ReLUProgram((1, 1)), ((0.0, 1.0),) * 2),
        "signed_add": (ReLUProgram((1, 1)), signed),
        "minimum": (ReLUProgram((1, 0), hinges=(Hinge((1, -1), scale=-1),)), signed),
        "maximum": (ReLUProgram((0, 1), hinges=(Hinge((1, -1)),)), signed),
        "absolute_difference": (
            ReLUProgram((0, 0), hinges=(Hinge((1, -1)), Hinge((-1, 1)))),
            signed,
        ),
        "clipped_difference": (
            ReLUProgram((0, 0), -1, (Hinge((1, -1), 1), Hinge((1, -1), -1, -1))),
            signed,
        ),
        "triangle": (
            ReLUProgram(
                (0, 0),
                hinges=(Hinge((1, 0), 1), Hinge((1, 0), 0, -2), Hinge((1, 0), -1)),
            ),
            signed,
        ),
        "manhattan_distance": (
            ReLUProgram(
                (0, 0, 0, 0),
                hinges=(
                    Hinge((1, 0, -1, 0)),
                    Hinge((-1, 0, 1, 0)),
                    Hinge((0, 1, 0, -1)),
                    Hinge((0, -1, 0, 1)),
                ),
            ),
            signed * 2,
        ),
    }


def make_model(
    dimensions: int, widths: tuple[int, ...], seed: int
) -> torch.nn.Sequential:
    """Create deterministic initial tensors without consuming the caller's RNG."""
    with torch.random.fork_rng():
        torch.manual_seed(72000 + seed)
        layers = []
        for width in widths:
            layers.extend((torch.nn.Linear(dimensions, width), torch.nn.ReLU()))
            dimensions = width
        layers.append(torch.nn.Linear(dimensions, 1))
    return torch.nn.Sequential(*layers).eval()


def inputs(domain: tuple, seed: int) -> tuple[torch.Tensor, int]:
    """Exact dyadic grid plus independently rounded continuous uniform samples."""
    steps = 33 if len(domain) == 2 else 5
    grid = torch.cartesian_prod(*(torch.linspace(lo, hi, steps) for lo, hi in domain))
    rng = torch.Generator().manual_seed(seed)
    random = torch.rand((4096, len(domain)), generator=rng, dtype=torch.float64)
    low, high = torch.tensor(domain, dtype=torch.float64).unbind(1)
    random = (low + random * (high - low)).float()
    return torch.cat((grid, random)), len(grid)


def reference(name: str, x: torch.Tensor) -> torch.Tensor:
    """Direct mathematical operations, evaluated in float64 on float32 inputs."""
    x = x.double()
    a, b = x[:, 0], x[:, 1]
    if name in ("nonnegative_add", "signed_add"):
        return a + b
    if name == "minimum":
        return torch.minimum(a, b)
    if name == "maximum":
        return torch.maximum(a, b)
    if name == "absolute_difference":
        return (a - b).abs()
    if name == "clipped_difference":
        return (a - b).clamp(-1, 1)
    if name == "triangle":
        return (1 - a.abs()).clamp_min(0)
    if name == "manhattan_distance":
        return (x[:, :2] - x[:, 2:]).abs().sum(1)
    raise ValueError(name)


@torch.inference_mode()
def metrics(model: torch.nn.Sequential, x: torch.Tensor, name: str, grid: int) -> dict:
    """Record exact grid results separately from general floating-point error."""
    actual = model(x).squeeze(1).double()
    error = (actual - reference(name, x)).abs()
    if not bool(torch.isfinite(error).all()):
        raise RuntimeError("non-finite compiler result")
    return {
        "cases": len(x),
        "grid_cases": grid,
        "grid_exact": int((error[:grid] == 0).sum()),
        "within_tolerance": int((error <= TOLERANCE).sum()),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
    }


def main() -> None:
    """Freeze specifications, run all fixed conditions, and retain replay artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--graph-evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("use a new output directory to preserve earlier evidence")
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("verify_relu_compiler.py"),
        root / "stateset_agents/experimental/weight_space/relu_program.py",
    ]
    sources = args.out / "sources"
    sources.mkdir()
    for source in source_paths:
        shutil.copy2(source, sources / source.name)
    tasks = programs()
    graph_files = [
        args.graph_evidence / f"{kind}_seed{seed}.safetensors"
        for kind in ("original", "edited")
        for seed in SEEDS
    ]
    protocol = {
        "architectures": ARCHITECTURES,
        "seeds": SEEDS,
        "tasks": {
            name: {"program": asdict(program), "domain": domain}
            for name, (program, domain) in tasks.items()
        },
        "input_seed_start": 73000,
        "absolute_tolerance": TOLERANCE,
        "optimizer_steps": 0,
        "model_selection": "none; every fixed condition is reported",
        "graph_evidence_directory": args.graph_evidence.name,
        "graph_dependencies": {p.name: digest(p) for p in graph_files},
        "sources": {p.name: digest(p) for p in source_paths},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
        },
    }
    write_json(args.out / "protocol.json", protocol)
    checkpoints, cases, rows = {}, {}, []
    for task_index, (name, (program, domain)) in enumerate(tasks.items()):
        x, grid = inputs(domain, 73000 + task_index)
        cases[name] = x
        for widths in ARCHITECTURES:
            for seed in SEEDS:
                key = f"{name}__{'x'.join(map(str, widths))}__{seed}"
                net = make_model(len(domain), widths, seed)
                initial = {k: v.clone() for k, v in net.state_dict().items()}
                compiled = compile_relu_program(net, program, domain=domain)
                net.load_state_dict(compiled.state_dict)
                for kind, state in (("base", initial), ("edited", compiled.state_dict)):
                    checkpoints.update(
                        {f"{key}.{kind}.{k}": v for k, v in state.items()}
                    )
                measured = metrics(net, x, name, grid)
                row = {
                    "key": key,
                    "task": name,
                    "widths": widths,
                    "seed": seed,
                    "channels": compiled.channels,
                    "assigned": len(compiled.assignments),
                    "changed": compiled.changed_parameters,
                    "parameters": compiled.total_parameters,
                    **measured,
                }
                rows.append(row)
                if (
                    measured["within_tolerance"] != len(x)
                    or measured["grid_exact"] != grid
                ):
                    raise RuntimeError(f"compiler arithmetic check failed: {row}")
        LOGGER.info("checked %s across 12 starting networks", name)
    graph_rows = []
    for seed in SEEDS:
        initial = load_file(
            str(args.graph_evidence / f"original_seed{seed}.safetensors")
        )
        expected = load_file(
            str(args.graph_evidence / f"edited_seed{seed}.safetensors")
        )
        net = make_model(2, (32, 32), 0)
        net.load_state_dict({k.removeprefix("message."): v for k, v in initial.items()})
        compiled = compile_relu_program(net, ReLUProgram((1, 1)), domain=((0, 1),) * 2)
        equal = all(
            torch.equal(v, expected[f"message.{k}"])
            for k, v in compiled.state_dict.items()
        )
        if not equal:
            raise RuntimeError("generic compiler disagrees with frozen graph repair")
        checkpoints.update(
            {f"graph{seed}.{k}": v for k, v in compiled.state_dict.items()}
        )
        graph_rows.append(
            {
                "seed": seed,
                "assigned": len(compiled.assignments),
                "changed": compiled.changed_parameters,
                "matches_original_repair": equal,
            }
        )
    save_file(checkpoints, str(args.out / "checkpoints.safetensors"))
    save_file(cases, str(args.out / "inputs.safetensors"))
    write_json(
        args.out / "summary.json",
        {
            "variants": rows,
            "graph_compatibility": graph_rows,
            "total_predictions": sum(r["cases"] for r in rows),
            "max_abs_error": max(r["max_abs_error"] for r in rows),
        },
    )
    write_json(
        args.out / "artifact_manifest.json",
        {
            p.relative_to(args.out).as_posix(): digest(p)
            for p in sorted(args.out.rglob("*"))
            if p.is_file()
        },
    )
    LOGGER.info("finished %d variants", len(rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
