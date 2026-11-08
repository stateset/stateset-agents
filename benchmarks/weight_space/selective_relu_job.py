"""Edit one arithmetic output while preserving two trained output functions.

The sparse dependency structure is supplied before training. This experiment
does not discover spare capacity in a dense pretrained network, prune a model,
or compare editing with retraining. Every architecture, seed, and edit is kept.
"""

from __future__ import annotations

import argparse
import logging
import platform
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from benchmarks.weight_space.relu_compiler_job import (
    digest,
    inputs,
    programs,
    reference,
    write_json,
)
from stateset_agents.core.errors import ConfigurationError
from stateset_agents.experimental.weight_space.relu_program import compile_relu_program

LOGGER = logging.getLogger(__name__)
ARCHITECTURES = ((8,), (12, 8), (16, 12, 8))
SEEDS = (0, 1, 2)
TASKS = ("signed_add", "maximum", "clipped_difference")
DOMAIN = ((-4.0, 4.0),) * 2
TRAIN_STEPS = 400
TRAIN_CASES = 512
LEARNING_RATE = 0.01
TOLERANCE = 2e-6


def targets(x: torch.Tensor) -> torch.Tensor:
    """Fixed smooth training functions; the middle output is later replaced."""
    a, b = x.unbind(1)
    return torch.stack((a.sin() + 0.5 * b, a - b, b.cos() - 0.25 * a), dim=1)


def make_sparse_model(
    widths: tuple[int, ...], seed: int
) -> tuple[torch.nn.Sequential, dict[str, torch.Tensor]]:
    """Reserve even hidden rows outside both protected output dependency paths."""
    with torch.random.fork_rng():
        torch.manual_seed(84000 + seed)
        layers = []
        dimensions = 2
        for width in widths:
            layers.extend((torch.nn.Linear(dimensions, width), torch.nn.ReLU()))
            dimensions = width
        layers.append(torch.nn.Linear(dimensions, 3))
        net = torch.nn.Sequential(*layers)
    masks = {name: torch.ones_like(value) for name, value in net.named_parameters()}
    for index in range(2, len(net) - 1, 2):
        masks[f"{index}.weight"][1::2, ::2] = 0
    masks[f"{len(net) - 1}.weight"][[0, 2], ::2] = 0
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            parameter.mul_(masks[name])
    return net, masks


def train_model(
    widths: tuple[int, ...], seed: int, x: torch.Tensor
) -> tuple[torch.nn.Sequential, dict]:
    """Fit all three outputs with fixed steps, preserving the declared sparsity."""
    net, masks = make_sparse_model(widths, seed)
    y = targets(x)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    history = []
    start = time.perf_counter()
    for step in range(TRAIN_STEPS + 1):
        predicted = net(x)
        errors = (predicted - y).square().mean(dim=0)
        if step % 100 == 0 or step == TRAIN_STEPS:
            history.append({"step": step, "mse_by_output": errors.detach().tolist()})
        if step == TRAIN_STEPS:
            break
        optimizer.zero_grad(set_to_none=True)
        errors.mean().backward()
        for name, parameter in net.named_parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(masks[name])
        optimizer.step()
        with torch.no_grad():
            for name, parameter in net.named_parameters():
                parameter.mul_(masks[name])
    return net.eval(), {
        "widths": widths,
        "seed": seed,
        "steps": TRAIN_STEPS,
        "elapsed_seconds": time.perf_counter() - start,
        "history": history,
    }


def shared_path_rejection(net: torch.nn.Sequential) -> str:
    """Show that sufficient width alone does not authorize a selective edit."""
    from copy import deepcopy

    connected = deepcopy(net)
    with torch.no_grad():
        # One protected output now reads every last-hidden row. Every spare
        # channel is thus connected, regardless of earlier sparse structure.
        connected[-1].weight[0].fill_(0.25)
    try:
        compile_relu_program(
            connected, programs()["signed_add"][0], domain=DOMAIN, output_index=1
        )
    except ConfigurationError as exc:
        if "no nonzero path" not in str(exc):
            raise
        return str(exc)
    raise RuntimeError("connected control was incorrectly accepted")


def run(out: Path) -> dict:
    """Freeze a protocol, train all fixed cases, and retain tensors and edit ledgers."""
    if out.exists():
        raise ValueError("use a new output directory to preserve earlier evidence")
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("verify_selective_relu.py"),
        Path(__file__).with_name("relu_compiler_job.py"),
        Path(__file__).with_name("verify_relu_compiler.py"),
        root / "stateset_agents/experimental/weight_space/relu_program.py",
    ]
    (out / "sources").mkdir()
    for source in source_paths:
        shutil.copy2(source, out / "sources" / source.name)
    tasks = {name: programs()[name] for name in TASKS}
    protocol = {
        "kind": "stateset-selective-relu-v1",
        "architectures": ARCHITECTURES,
        "seeds": SEEDS,
        "tasks": {name: asdict(program) for name, (program, _) in tasks.items()},
        "domain": DOMAIN,
        "target_output": 1,
        "protected_outputs": [0, 2],
        "sparsity": "protected odd hidden rows read only odd rows; outputs 0 and 2 read odd rows",
        "training": {
            "steps": TRAIN_STEPS,
            "cases": TRAIN_CASES,
            "input_seed": 85000,
            "model_seed_start": 84000,
            "optimizer": "Adam, default betas/epsilon, no weight decay",
            "learning_rate": LEARNING_RATE,
            "targets": ["sin(x0)+0.5*x1", "x0-x1", "cos(x1)-0.25*x0"],
        },
        "evaluation_input_seed": 86000,
        "grid_steps": 33,
        "uniform_cases": 4096,
        "absolute_tolerance": TOLERANCE,
        "edit_optimizer_steps": 0,
        "selection": "none; every fixed condition and trained final checkpoint is reported",
        "sources": {p.name: digest(p) for p in source_paths},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "threads": 1,
        },
    }
    write_json(out / "protocol.json", protocol)
    x, grid = inputs(DOMAIN, 86000)
    train_x = (
        8 * torch.rand((TRAIN_CASES, 2), generator=torch.Generator().manual_seed(85000))
        - 4
    )
    save_file({"training": train_x, "evaluation": x}, str(out / "inputs.safetensors"))
    checkpoints, histories, outcomes, ledgers, controls = {}, [], [], {}, []
    for widths in ARCHITECTURES:
        for seed in SEEDS:
            key = f"{'x'.join(map(str, widths))}__{seed}"
            net, history = train_model(widths, seed, train_x)
            histories.append({"key": key, **history})
            base = {k: v.detach().clone() for k, v in net.state_dict().items()}
            checkpoints.update({f"{key}.base.{k}": v for k, v in base.items()})
            with torch.inference_mode():
                baseline = net(x)
            controls.append({"key": key, "rejection": shared_path_rejection(net)})
            for name, (program, _) in tasks.items():
                net.load_state_dict(base)
                compiled = compile_relu_program(
                    net, program, domain=DOMAIN, output_index=1
                )
                net.load_state_dict(compiled.state_dict)
                checkpoints.update(
                    {f"{key}.{name}.{k}": v for k, v in compiled.state_dict.items()}
                )
                ledgers[f"{key}.{name}"] = [asdict(a) for a in compiled.assignments]
                with torch.inference_mode():
                    actual = net(x)
                    error = (actual[:, 1].double() - reference(name, x)).abs()
                if not torch.equal(actual[:, [0, 2]], baseline[:, [0, 2]]):
                    raise RuntimeError("protected outputs changed")
                if not bool(torch.isfinite(actual).all()) or not bool(
                    (error <= TOLERANCE).all()
                ):
                    raise RuntimeError("compiled output failed the fixed tolerance")
                outcomes.append(
                    {
                        "key": key,
                        "task": name,
                        "widths": widths,
                        "seed": seed,
                        "cases": len(x),
                        "grid_cases": grid,
                        "grid_exact": int((error[:grid] == 0).sum()),
                        "within_tolerance": int((error <= TOLERANCE).sum()),
                        "max_abs_error": float(error.max()),
                        "protected_values_equal": int(
                            (actual[:, [0, 2]] == baseline[:, [0, 2]]).sum()
                        ),
                        "baseline_mse_by_output": (baseline - targets(x))
                        .square()
                        .mean(dim=0)
                        .tolist(),
                        "baseline_target_error": float(
                            (baseline[:, 1].double() - reference(name, x)).abs().max()
                        ),
                        "assigned": len(compiled.assignments),
                        "changed": compiled.changed_parameters,
                        "parameters": compiled.total_parameters,
                        "channel_rows": compiled.channel_rows,
                    }
                )
    summary = {
        "variants": outcomes,
        "shared_path_controls": controls,
        "target_predictions": sum(row["cases"] for row in outcomes),
        "protected_values_equal": sum(
            row["protected_values_equal"] for row in outcomes
        ),
        "max_abs_error": max(row["max_abs_error"] for row in outcomes),
    }
    save_file(checkpoints, str(out / "checkpoints.safetensors"))
    write_json(out / "training.json", histories)
    write_json(out / "edit_ledgers.json", ledgers)
    write_json(out / "summary.json", summary)
    write_json(
        out / "artifact_manifest.json",
        {
            p.relative_to(out).as_posix(): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )
    return summary


def main() -> None:
    """Run the fixed local CPU preservation experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.out)
    LOGGER.info(
        "Verified numerical checks for %d selective edits", len(summary["variants"])
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
