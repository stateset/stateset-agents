#!/usr/bin/env python3
"""Run a matched 1/2/4/8-GPU StateSet DDP scaling matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from distributed_scaling_workload import DEFAULT_WORKLOAD
from scaling_comparison import (
    load_scaling_evidence,
    render_markdown,
    summarize_scaling,
    validate_scaling_comparison,
    validate_scaling_performance,
)


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_matrix(args: argparse.Namespace) -> None:
    """Execute every topology and seed, then fail closed through the validator."""
    available = torch.cuda.device_count()
    if available < max(args.gpu_counts):
        raise RuntimeError(
            f"matrix needs {max(args.gpu_counts)} GPUs, but CUDA exposes {available}"
        )
    gpu_names = {torch.cuda.get_device_name(index) for index in range(available)}
    if len(gpu_names) != 1:
        raise RuntimeError(f"mixed GPU models are not comparable: {sorted(gpu_names)}")

    commit = args.harness_commit or _commit()
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
                str(Path(__file__).with_name("distributed_scaling_workload.py")),
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
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(gpu_count)))
            env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
            subprocess.run(command, check=True, env=env, timeout=args.timeout_seconds)

    runs = load_scaling_evidence([evidence_dir])
    validate_scaling_comparison(runs, args.gpu_counts, len(args.seeds))
    summary = summarize_scaling(runs)
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
    parser.add_argument("--config-json", default="{}")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--min-efficiency", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    run_matrix(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
