#!/usr/bin/env python3
"""Validate and summarize measured distributed-scaling evidence."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from framework_comparison import (
    EvidenceError,
    RunEvidence,
    discover_inputs,
    validate_document,
)

MATCH_FIELDS = (
    "framework",
    "framework_version",
    "harness_commit",
    "protocol",
    "cache_policy",
    "algorithm",
    "algorithm_revision",
    "model",
    "model_revision",
    "task",
    "dataset_revision",
    "workload_config_sha256",
)


def load_scaling_evidence(inputs: Sequence[Path]) -> list[RunEvidence]:
    """Load scaling documents using the strict measured-run schema."""
    runs: list[RunEvidence] = []
    for path in discover_inputs(inputs):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"{path}: invalid JSON: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise EvidenceError(f"{path}: top-level value must be an object")
        digest = raw.get("workload_config_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise EvidenceError(
                f"{path}: workload_config_sha256 must contain 64 hex characters"
            )
        try:
            bytes.fromhex(digest)
        except ValueError as exc:
            raise EvidenceError(
                f"{path}: workload_config_sha256 is not hexadecimal"
            ) from exc
        runs.append(validate_document(raw, path))
    return runs


def validate_scaling_comparison(
    runs: Sequence[RunEvidence],
    gpu_counts: Sequence[int] = (1, 2, 4, 8),
    min_seeds: int = 3,
) -> None:
    """Require a complete topology matrix with matched seeds and workload."""
    if not runs:
        raise EvidenceError("no scaling evidence supplied")
    expected = tuple(gpu_counts)
    if not expected or any(count < 1 for count in expected):
        raise EvidenceError("gpu_counts must contain positive integers")
    if len(expected) != len(set(expected)):
        raise EvidenceError("gpu_counts must not contain duplicates")
    if min_seeds < 1:
        raise EvidenceError("min_seeds must be >= 1")

    first = runs[0]
    first_hardware = first.data["hardware"]
    for run in runs[1:]:
        differences = [
            field for field in MATCH_FIELDS if run.data[field] != first.data[field]
        ]
        if run.data["hardware"]["gpu"] != first_hardware["gpu"]:
            differences.append("hardware.gpu")
        if differences:
            raise EvidenceError(
                f"{run.source}: incomparable fields differ: {', '.join(differences)}"
            )

    grouped: dict[int, list[RunEvidence]] = {}
    for run in runs:
        grouped.setdefault(int(run.data["hardware"]["gpu_count"]), []).append(run)
    actual = set(grouped)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        raise EvidenceError(
            f"incomplete topology matrix: missing={missing}, unexpected={unexpected}"
        )

    expected_seeds: set[int] | None = None
    for gpu_count in expected:
        seeds = [run.seed for run in grouped[gpu_count]]
        if len(seeds) != len(set(seeds)):
            raise EvidenceError(
                f"{gpu_count} GPU: duplicate seed evidence is forbidden"
            )
        if len(seeds) < min_seeds:
            raise EvidenceError(
                f"{gpu_count} GPU: only {len(seeds)} seeds; at least {min_seeds} required"
            )
        seed_set = set(seeds)
        if expected_seeds is None:
            expected_seeds = seed_set
        elif seed_set != expected_seeds:
            raise EvidenceError(
                f"{gpu_count} GPU: seed set {sorted(seed_set)} does not match "
                f"{sorted(expected_seeds)}"
            )


def summarize_scaling(runs: Sequence[RunEvidence]) -> dict[str, Any]:
    """Compute throughput speedup and efficiency against the 1-GPU baseline."""
    grouped: dict[int, list[RunEvidence]] = {}
    for run in runs:
        grouped.setdefault(int(run.data["hardware"]["gpu_count"]), []).append(run)
    baseline = statistics.mean(run.metrics["samples_per_second"] for run in grouped[1])
    topologies: dict[str, Any] = {}
    for gpu_count, topology_runs in sorted(grouped.items()):
        throughput = [run.metrics["samples_per_second"] for run in topology_runs]
        wall_clock = [run.metrics["wall_clock_seconds"] for run in topology_runs]
        peak_vram = [run.metrics["peak_vram_mb"] for run in topology_runs]
        mean_throughput = statistics.mean(throughput)
        speedup = mean_throughput / baseline
        topologies[str(gpu_count)] = {
            "seeds": sorted(run.seed for run in topology_runs),
            "samples_per_second_mean": mean_throughput,
            "samples_per_second_std": (
                statistics.stdev(throughput) if len(throughput) > 1 else 0.0
            ),
            "wall_clock_seconds_mean": statistics.mean(wall_clock),
            "peak_vram_mb_mean": statistics.mean(peak_vram),
            "speedup_vs_1_gpu": speedup,
            "scaling_efficiency": speedup / gpu_count,
        }
    return {
        "schema_version": 1,
        "report_kind": "measured-scaling-comparison",
        "workload_config_sha256": runs[0].data["workload_config_sha256"],
        "gpu": runs[0].data["hardware"]["gpu"],
        "topologies": topologies,
    }


def validate_scaling_performance(
    summary: Mapping[str, Any],
    *,
    min_efficiency: float = 0.5,
    require_monotonic: bool = True,
) -> None:
    """Apply the predeclared publication threshold to measured means."""
    if not 0 < min_efficiency <= 1:
        raise EvidenceError("min_efficiency must be in (0, 1]")
    topologies = summary["topologies"]
    ordered = sorted((int(count), values) for count, values in topologies.items())
    if require_monotonic:
        for (previous_count, previous), (count, current) in zip(
            ordered, ordered[1:], strict=False
        ):
            if (
                current["samples_per_second_mean"]
                <= previous["samples_per_second_mean"]
            ):
                raise EvidenceError(
                    f"throughput is not monotonic from {previous_count} to {count} GPUs"
                )
    failures = [
        (count, float(values["scaling_efficiency"]))
        for count, values in ordered
        if count > 1 and float(values["scaling_efficiency"]) < min_efficiency
    ]
    if failures:
        rendered = ", ".join(
            f"{count} GPU={efficiency:.1%}" for count, efficiency in failures
        )
        raise EvidenceError(
            f"scaling efficiency below {min_efficiency:.1%}: {rendered}"
        )


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render scale results and the predeclared publication gate."""
    lines = [
        "# Measured distributed-scaling comparison",
        "",
        f"Workload digest: `{summary['workload_config_sha256']}`",
        f"GPU: {summary['gpu']}",
        "Default publication gate: monotonic throughput and at least 50% efficiency",
        "",
        "| GPUs | Seeds | Samples/s | Speedup | Scaling efficiency | Wall clock (s) | Peak VRAM/GPU (MiB) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for gpu_count, values in summary["topologies"].items():
        lines.append(
            f"| {gpu_count} | {len(values['seeds'])} |"
            f" {values['samples_per_second_mean']:.3f} ± {values['samples_per_second_std']:.3f} |"
            f" {values['speedup_vs_1_gpu']:.3f}× |"
            f" {values['scaling_efficiency']:.1%} |"
            f" {values['wall_clock_seconds_mean']:.1f} |"
            f" {values['peak_vram_mb_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "> Scope: these measurements apply only to the retained workload, model,",
            "> software, GPU, and interconnect configuration.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate measured GPU scaling evidence"
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--gpu-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--min-efficiency", type=float, default=0.5)
    parser.add_argument(
        "--allow-non-monotonic",
        action="store_true",
        help="Do not require mean throughput to increase at every topology.",
    )
    parser.add_argument(
        "--no-performance-gate",
        action="store_true",
        help="Validate provenance/completeness without enforcing scale performance.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/scaling/report"),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        runs = load_scaling_evidence(args.inputs)
        validate_scaling_comparison(runs, args.gpu_counts, args.min_seeds)
        result = summarize_scaling(runs)
        if not args.no_performance_gate:
            validate_scaling_performance(
                result,
                min_efficiency=args.min_efficiency,
                require_monotonic=not args.allow_non_monotonic,
            )
        if not args.validate_only:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "scaling.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (args.output_dir / "scaling.md").write_text(
                render_markdown(result), encoding="utf-8"
            )
    except EvidenceError as exc:
        print(f"scaling comparison rejected: {exc}", file=sys.stderr)
        return 2
    print(f"validated {len(runs)} measured scaling runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
