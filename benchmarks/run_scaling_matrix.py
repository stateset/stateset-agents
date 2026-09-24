#!/usr/bin/env python3
"""Run a matched 1/2/4/8-GPU StateSet DDP scaling matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from benchmark_provenance import resolve_harness_commit
from distributed_scaling_workload import DEFAULT_WORKLOAD
from scaling_comparison import (
    load_scaling_evidence,
    render_markdown,
    summarize_scaling,
    validate_scaling_comparison,
    validate_scaling_performance,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_LAUNCHER_PLACEHOLDERS = {
    "{gpu_count}",
    "{node_count}",
    "{nproc_per_node}",
    "{seed}",
    "{harness_commit}",
    "{config_json}",
    "{command_label}",
    "{output}",
    "{workload}",
}


class ScalingRunnerError(ValueError):
    """Raised when a scaling launcher cannot produce defensible evidence."""


def load_launcher_manifest(path: Path, gpu_counts: Sequence[int]) -> dict[str, object]:
    """Load a shell-free local or provider multi-node launcher contract."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScalingRunnerError(f"{path}: invalid JSON") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ScalingRunnerError("launcher manifest must have schema_version=1")
    if raw.get("kind") != "stateset-scaling-launcher-manifest":
        raise ScalingRunnerError("launcher manifest kind is invalid")
    command = raw.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise ScalingRunnerError("launcher command must be a non-empty string list")
    missing = REQUIRED_LAUNCHER_PLACEHOLDERS.difference(command)
    if missing:
        raise ScalingRunnerError(
            "launcher command is missing placeholders: " + ", ".join(sorted(missing))
        )
    topologies = raw.get("topologies")
    if not isinstance(topologies, dict) or set(topologies) != {
        str(count) for count in gpu_counts
    }:
        raise ScalingRunnerError("launcher topologies must exactly match gpu_counts")
    for gpu_count in gpu_counts:
        topology = topologies[str(gpu_count)]
        if not isinstance(topology, dict):
            raise ScalingRunnerError(f"topology {gpu_count} must be an object")
        node_count = topology.get("node_count")
        nproc = topology.get("nproc_per_node")
        if (
            isinstance(node_count, bool)
            or not isinstance(node_count, int)
            or node_count < 1
            or isinstance(nproc, bool)
            or not isinstance(nproc, int)
            or nproc < 1
            or node_count * nproc != gpu_count
        ):
            raise ScalingRunnerError(
                f"topology {gpu_count} must satisfy node_count * nproc_per_node = gpu_count"
            )
    largest = topologies[str(max(gpu_counts))]
    assert isinstance(largest, dict)
    if int(largest["node_count"]) < 2:
        raise ScalingRunnerError(
            "largest launcher topology must span at least two nodes"
        )
    return raw


def _format_launcher(command: Sequence[str], values: dict[str, object]) -> list[str]:
    formatted: list[str] = []
    for part in command:
        match = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", part)
        if match is None:
            formatted.append(part)
            continue
        name = match.group(1)
        if name not in values:
            raise ScalingRunnerError(f"unknown launcher placeholder: {name}")
        formatted.append(str(values[name]))
    return formatted


def _validate_launched_topology(output: Path, expected: dict[str, object]) -> None:
    """Match workload-observed physical nodes to the declared launcher topology."""
    try:
        evidence = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScalingRunnerError(
            f"launcher did not produce valid evidence: {output}"
        ) from exc
    hardware = evidence.get("hardware")
    if not isinstance(hardware, dict):
        raise ScalingRunnerError(f"{output}: hardware evidence is missing")
    if hardware.get("node_count") != expected["node_count"]:
        raise ScalingRunnerError(
            f"{output}: observed node_count does not match launcher topology"
        )
    ranks = hardware.get("ranks_per_node")
    if not isinstance(ranks, dict) or any(
        value != expected["nproc_per_node"] for value in ranks.values()
    ):
        raise ScalingRunnerError(
            f"{output}: observed ranks_per_node does not match launcher topology"
        )


def run_matrix(args: argparse.Namespace) -> None:
    """Execute every topology and seed, then fail closed through the validator."""
    launcher = (
        load_launcher_manifest(args.launcher_manifest, args.gpu_counts)
        if args.launcher_manifest is not None
        else None
    )
    if launcher is None:
        available = torch.cuda.device_count()
        if available < max(args.gpu_counts):
            raise RuntimeError(
                f"matrix needs {max(args.gpu_counts)} GPUs, but CUDA exposes {available}"
            )
        gpu_names = {torch.cuda.get_device_name(index) for index in range(available)}
        if len(gpu_names) != 1:
            raise RuntimeError(
                f"mixed GPU models are not comparable: {sorted(gpu_names)}"
            )

    commit = resolve_harness_commit(REPOSITORY_ROOT, claimed=args.harness_commit)
    config = dict(DEFAULT_WORKLOAD)
    config.update(json.loads(args.config_json))
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    evidence_dir = args.output_dir / "evidence"
    report_dir = args.output_dir / "report"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    base_order = list(args.gpu_counts)
    for seed_index, seed in enumerate(args.seeds):
        rotation = seed_index % len(base_order)
        topology_order = base_order[rotation:] + base_order[:rotation]
        for gpu_count in topology_order:
            output = evidence_dir / f"gpu{gpu_count}-seed{seed}.json"
            workload = Path(__file__).with_name("distributed_scaling_workload.py")
            if launcher is None:
                command_label = (
                    f"CUDA_VISIBLE_DEVICES=0-{gpu_count - 1} "
                    f"python -m torch.distributed.run --standalone "
                    f"--nproc-per-node={gpu_count} "
                    "benchmarks/distributed_scaling_workload.py "
                    f"--gpu-count={gpu_count} --seed={seed}"
                )
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={gpu_count}",
                    str(workload),
                    "--gpu-count",
                    str(gpu_count),
                    "--seed",
                    str(seed),
                    "--harness-commit",
                    commit,
                    "--config-json",
                    config_json,
                    "--command-label",
                    command_label,
                    "--output",
                    str(output),
                ]
            else:
                topologies = launcher["topologies"]
                launcher_command = launcher["command"]
                assert isinstance(topologies, dict)
                assert isinstance(launcher_command, list)
                topology = topologies[str(gpu_count)]
                assert isinstance(topology, dict)
                command_label = (
                    "provider scaling launcher "
                    f"--gpu-count={gpu_count} --node-count={topology['node_count']} "
                    f"--nproc-per-node={topology['nproc_per_node']} --seed={seed}"
                )
                command = _format_launcher(
                    launcher_command,
                    {
                        "gpu_count": gpu_count,
                        "node_count": topology["node_count"],
                        "nproc_per_node": topology["nproc_per_node"],
                        "seed": seed,
                        "harness_commit": commit,
                        "config_json": config_json,
                        "command_label": command_label,
                        "output": output.resolve(),
                        "workload": workload.resolve(),
                    },
                )
            env = dict(os.environ)
            if launcher is None:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(gpu_count)))
            env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
            subprocess.run(command, check=True, env=env, timeout=args.timeout_seconds)
            if launcher is not None:
                _validate_launched_topology(output, topology)

    runs = load_scaling_evidence([evidence_dir])
    validate_scaling_comparison(runs, args.gpu_counts, len(args.seeds))
    summary = summarize_scaling(runs)
    summary["publication_gate"] = {
        "enforced": True,
        "min_efficiency": args.min_efficiency,
        "require_monotonic": True,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "scaling.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report_dir / "scaling.md").write_text(render_markdown(summary), encoding="utf-8")
    validate_scaling_performance(
        summary,
        min_efficiency=args.min_efficiency,
        require_monotonic=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2026])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results/scaling")
    )
    parser.add_argument("--harness-commit")
    parser.add_argument(
        "--launcher-manifest",
        type=Path,
        help="Shell-free provider launcher contract for physical multi-node runs.",
    )
    parser.add_argument(
        "--validate-launcher",
        action="store_true",
        help="Validate the launcher contract without provisioning or running GPUs.",
    )
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--min-efficiency", type=float, default=0.7)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_launcher:
        if args.launcher_manifest is None:
            raise ScalingRunnerError("--validate-launcher requires --launcher-manifest")
        load_launcher_manifest(args.launcher_manifest, args.gpu_counts)
        print(f"validated multi-node scaling launcher: {args.launcher_manifest}")
        return 0
    run_matrix(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
