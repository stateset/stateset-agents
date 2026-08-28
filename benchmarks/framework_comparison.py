#!/usr/bin/env python3
"""Compare measured LLM RL framework runs without manufacturing results.

Each framework must run the same published protocol and emit one evidence JSON
document per seed. This tool validates provenance, rejects mismatched runs, and
produces a descriptive comparison. It never invents feature scores, simulates
competitors, or declares a winner from incomparable experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    """Raised when benchmark evidence is incomplete or incomparable."""


REQUIRED_STRINGS = (
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
    "timestamp",
    "command",
)
REQUIRED_METRICS = (
    "samples_per_second",
    "wall_clock_seconds",
    "peak_vram_mb",
    "eval_score_baseline",
    "eval_score_final",
)
REQUIRED_HARDWARE = ("gpu", "gpu_count", "cuda")
MATCH_FIELDS = (
    "protocol",
    "cache_policy",
    "algorithm",
    "algorithm_revision",
    "model",
    "model_revision",
    "task",
    "dataset_revision",
)


@dataclass(frozen=True)
class RunEvidence:
    """One measured framework run with enough provenance to reproduce it."""

    source: Path
    data: Mapping[str, Any]

    @property
    def framework(self) -> str:
        return str(self.data["framework"])

    @property
    def seed(self) -> int:
        return int(self.data["seed"])

    @property
    def metrics(self) -> Mapping[str, float]:
        return self.data["metrics"]

    @property
    def comparison_key(self) -> tuple[Any, ...]:
        hardware = self.data["hardware"]
        return tuple(self.data[field] for field in MATCH_FIELDS) + (
            json.dumps(self.data["config"], sort_keys=True, separators=(",", ":")),
            hardware["gpu"],
            hardware["gpu_count"],
            hardware["cuda"],
        )


def _require_nonempty_string(data: Mapping[str, Any], field: str, source: Path) -> None:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{source}: {field!r} must be a non-empty string")


def _require_finite_number(
    data: Mapping[str, Any], field: str, source: Path, *, minimum: float | None = None
) -> float:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{source}: {field!r} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        requirement = "finite" if minimum is None else f"finite and >= {minimum}"
        raise EvidenceError(f"{source}: {field!r} must be {requirement}, got {value!r}")
    return number


def validate_document(data: Mapping[str, Any], source: Path) -> RunEvidence:
    """Validate one evidence document and return its typed representation."""
    if data.get("schema_version") != 1:
        raise EvidenceError(f"{source}: schema_version must be 1")
    if data.get("measured") is not True:
        raise EvidenceError(
            f"{source}: measured must be true; simulated or estimated runs are forbidden"
        )

    for field in REQUIRED_STRINGS:
        _require_nonempty_string(data, field, source)
    for field in ("harness_commit", "model_revision", "dataset_revision"):
        value = str(data[field])
        if len(value) != 40:
            raise EvidenceError(f"{source}: {field} must be a full 40-character commit")

    try:
        parsed_timestamp = datetime.fromisoformat(
            str(data["timestamp"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise EvidenceError(f"{source}: timestamp is not ISO-8601") from exc
    if parsed_timestamp.tzinfo is None:
        raise EvidenceError(f"{source}: timestamp must include a UTC offset")

    seed = data.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvidenceError(f"{source}: seed must be a non-negative integer")

    config = data.get("config")
    if not isinstance(config, Mapping) or not config:
        raise EvidenceError(f"{source}: config must be a non-empty object")

    hardware = data.get("hardware")
    if not isinstance(hardware, Mapping):
        raise EvidenceError(f"{source}: hardware must be an object")
    for field in REQUIRED_HARDWARE:
        if field == "gpu_count":
            count = hardware.get(field)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise EvidenceError(f"{source}: hardware.gpu_count must be >= 1")
        else:
            _require_nonempty_string(hardware, field, source)

    metrics = data.get("metrics")
    if not isinstance(metrics, Mapping):
        raise EvidenceError(f"{source}: metrics must be an object")
    for field in REQUIRED_METRICS:
        number = _require_finite_number(metrics, field, source)
        if field in {"samples_per_second", "wall_clock_seconds", "peak_vram_mb"}:
            if number <= 0:
                raise EvidenceError(f"{source}: {field!r} must be greater than zero")

    artifact_sha256 = data.get("artifact_sha256")
    if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
        raise EvidenceError(f"{source}: artifact_sha256 must contain 64 hex characters")
    try:
        bytes.fromhex(artifact_sha256)
    except ValueError as exc:
        raise EvidenceError(f"{source}: artifact_sha256 is not hexadecimal") from exc

    return RunEvidence(source=source, data=data)


def discover_inputs(inputs: Sequence[Path]) -> list[Path]:
    """Resolve files and directories into a deterministic JSON file list."""
    paths: list[Path] = []
    for candidate in inputs:
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.json")))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            raise EvidenceError(f"benchmark input does not exist: {candidate}")
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise EvidenceError("no benchmark evidence JSON files found")
    return unique


def load_evidence(inputs: Sequence[Path]) -> list[RunEvidence]:
    """Load and validate all evidence, failing closed on any bad document."""
    runs: list[RunEvidence] = []
    for path in discover_inputs(inputs):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"{path}: invalid JSON: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise EvidenceError(f"{path}: top-level value must be an object")
        runs.append(validate_document(raw, path))
    return runs


def validate_comparison(
    runs: Sequence[RunEvidence],
    min_seeds: int = 3,
    required_frameworks: Sequence[str] = (),
) -> None:
    """Require matched protocols, unique seeds, and adequate replication."""
    if min_seeds < 1:
        raise EvidenceError("min_seeds must be >= 1")
    keys = {run.comparison_key for run in runs}
    if len(keys) != 1:
        differing: list[str] = []
        fields = (
            *MATCH_FIELDS,
            "config",
            "hardware.gpu",
            "hardware.gpu_count",
            "hardware.cuda",
        )
        for field_index, field in enumerate(fields):
            values = {key[field_index] for key in keys}
            if len(values) > 1:
                differing.append(f"{field}={sorted(map(str, values))}")
        raise EvidenceError("runs are not comparable: " + "; ".join(differing))

    by_framework: dict[str, list[RunEvidence]] = {}
    for run in runs:
        by_framework.setdefault(run.framework, []).append(run)
    if len(by_framework) < 2:
        raise EvidenceError("comparison requires evidence from at least two frameworks")

    required = tuple(required_frameworks)
    if any(not name.strip() for name in required):
        raise EvidenceError("required_frameworks must contain non-empty names")
    if len(required) != len(set(required)):
        raise EvidenceError("required_frameworks must not contain duplicates")
    missing_frameworks = sorted(set(required) - set(by_framework))
    if missing_frameworks:
        raise EvidenceError(
            "comparison is missing required frameworks: "
            + ", ".join(missing_frameworks)
        )

    expected_seeds: set[int] | None = None
    for framework, framework_runs in sorted(by_framework.items()):
        seeds = [run.seed for run in framework_runs]
        if len(seeds) != len(set(seeds)):
            raise EvidenceError(f"{framework}: duplicate seed evidence is forbidden")
        if len(seeds) < min_seeds:
            raise EvidenceError(
                f"{framework}: only {len(seeds)} seeds; at least {min_seeds} required"
            )
        seed_set = set(seeds)
        if expected_seeds is None:
            expected_seeds = seed_set
        elif seed_set != expected_seeds:
            raise EvidenceError(
                f"{framework}: seed set {sorted(seed_set)} does not match "
                f"{sorted(expected_seeds)}"
            )
        versions = {run.data["framework_version"] for run in framework_runs}
        if len(versions) != 1:
            raise EvidenceError(f"{framework}: runs span framework versions {versions}")


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    numbers = list(values)
    return {
        "mean": statistics.mean(numbers),
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "n": len(numbers),
    }


def summarize(runs: Sequence[RunEvidence]) -> dict[str, Any]:
    """Return descriptive statistics without subjective feature scoring."""
    grouped: dict[str, list[RunEvidence]] = {}
    for run in runs:
        grouped.setdefault(run.framework, []).append(run)

    first = runs[0]
    result: dict[str, Any] = {
        "schema_version": 1,
        "comparison": {field: first.data[field] for field in MATCH_FIELDS},
        "hardware": dict(first.data["hardware"]),
        "frameworks": {},
    }
    for framework, framework_runs in sorted(grouped.items()):
        result["frameworks"][framework] = {
            "version": framework_runs[0].data["framework_version"],
            "seeds": sorted(run.seed for run in framework_runs),
            "samples_per_second": _stats(
                run.metrics["samples_per_second"] for run in framework_runs
            ),
            "wall_clock_seconds": _stats(
                run.metrics["wall_clock_seconds"] for run in framework_runs
            ),
            "peak_vram_mb": _stats(
                run.metrics["peak_vram_mb"] for run in framework_runs
            ),
            "eval_score_baseline": _stats(
                run.metrics["eval_score_baseline"] for run in framework_runs
            ),
            "eval_score_final": _stats(
                run.metrics["eval_score_final"] for run in framework_runs
            ),
            "improvement": _stats(
                run.metrics["eval_score_final"] - run.metrics["eval_score_baseline"]
                for run in framework_runs
            ),
            "evidence": [run.source.name for run in framework_runs],
        }
    return result


def _format_stats(values: Mapping[str, Any], metric: str) -> str:
    stats = values[metric]
    return f"{stats['mean']:.3f} ± {stats['std']:.3f}"


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render an auditable report and explicitly scope its conclusions."""
    comparison = summary["comparison"]
    hardware = summary["hardware"]
    lines = [
        "# Measured framework comparison",
        "",
        "> Descriptive results only. Every row uses the same protocol, model, data,",
        "> task, and hardware. This report does not assign subjective feature scores.",
        "",
        f"- Protocol: `{comparison['protocol']}`",
        f"- Model: `{comparison['model']}` at `{comparison['model_revision']}`",
        f"- Task/data: `{comparison['task']}` at `{comparison['dataset_revision']}`",
        f"- Hardware: {hardware['gpu_count']}× {hardware['gpu']} (CUDA {hardware['cuda']})",
        "",
        "| Framework | Version | Seeds | Samples/s | Wall clock (s) | Peak VRAM (MiB) | Baseline | Final | Improvement |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for framework, values in summary["frameworks"].items():
        lines.append(
            f"| {framework} | {values['version']} | {len(values['seeds'])} |"
            f" {_format_stats(values, 'samples_per_second')} |"
            f" {_format_stats(values, 'wall_clock_seconds')} |"
            f" {_format_stats(values, 'peak_vram_mb')} |"
            f" {_format_stats(values, 'eval_score_baseline')} |"
            f" {_format_stats(values, 'eval_score_final')} |"
            f" {_format_stats(values, 'improvement')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The table establishes results only for the protocol and hardware above.",
            "It is not evidence of ecosystem maturity, developer experience, or",
            "performance on other models and clusters.",
            "",
        ]
    )
    return "\n".join(lines)


def evidence_digest(runs: Sequence[RunEvidence]) -> str:
    """Hash canonical inputs so a report identifies its exact evidence."""
    canonical = [
        json.dumps(run.data, sort_keys=True, separators=(",", ":")) for run in runs
    ]
    return hashlib.sha256("\n".join(sorted(canonical)).encode()).hexdigest()


def write_report(runs: Sequence[RunEvidence], output_dir: Path) -> None:
    """Write machine-readable and human-readable comparison artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = summarize(runs)
    result["evidence_sha256"] = evidence_digest(runs)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "comparison.md").write_text(render_markdown(result), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and compare real, matched framework benchmark evidence"
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="Evidence files/directories"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/framework_comparison/report"),
    )
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument(
        "--required-framework",
        action="append",
        default=[],
        help="Framework required in the comparison (repeatable).",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        runs = load_evidence(args.inputs)
        validate_comparison(
            runs,
            min_seeds=args.min_seeds,
            required_frameworks=args.required_framework,
        )
        if not args.validate_only:
            write_report(runs, args.output_dir)
    except EvidenceError as exc:
        print(f"framework comparison rejected: {exc}", file=sys.stderr)
        return 2
    print(
        f"validated {len(runs)} measured runs across "
        f"{len({run.framework for run in runs})} frameworks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
