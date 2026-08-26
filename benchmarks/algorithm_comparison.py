#!/usr/bin/env python3
"""Aggregate real, matched StateSet algorithm runs.

The former version of this file generated synthetic reward curves for several
algorithms and presented them as benchmark results. That behavior was removed:
publication-facing comparisons must now use measured evidence and three or
more seeds per algorithm.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from framework_comparison import (
    EvidenceError,
    RunEvidence,
    discover_inputs,
    render_markdown,
    summarize,
    validate_document,
)

MATCH_FIELDS = (
    "framework",
    "framework_version",
    "harness_commit",
    "protocol",
    "cache_policy",
    "model",
    "model_revision",
    "task",
    "dataset_revision",
)


def load_algorithm_evidence(inputs: Sequence[Path]) -> list[RunEvidence]:
    """Load strict evidence documents shared with the framework shootout."""
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


def validate_algorithm_comparison(
    runs: Sequence[RunEvidence], min_seeds: int = 3
) -> None:
    """Reject mixed protocols and under-replicated algorithm comparisons."""
    if min_seeds < 1:
        raise EvidenceError("min_seeds must be >= 1")
    if not runs:
        raise EvidenceError("no algorithm evidence supplied")

    first = runs[0]
    first_hardware = first.data["hardware"]
    for run in runs[1:]:
        differences = [
            field for field in MATCH_FIELDS if run.data[field] != first.data[field]
        ]
        hardware = run.data["hardware"]
        for field in ("gpu", "gpu_count"):
            if hardware[field] != first_hardware[field]:
                differences.append(f"hardware.{field}")
        if differences:
            raise EvidenceError(
                f"{run.source}: incomparable fields differ: {', '.join(differences)}"
            )

    grouped: dict[str, list[RunEvidence]] = {}
    for run in runs:
        grouped.setdefault(str(run.data["algorithm"]), []).append(run)
    if len(grouped) < 2:
        raise EvidenceError("comparison requires at least two algorithms")

    for algorithm, algorithm_runs in sorted(grouped.items()):
        seeds = [run.seed for run in algorithm_runs]
        if len(seeds) != len(set(seeds)):
            raise EvidenceError(f"{algorithm}: duplicate seed evidence is forbidden")
        if len(seeds) < min_seeds:
            raise EvidenceError(
                f"{algorithm}: only {len(seeds)} seeds; at least {min_seeds} required"
            )
        revisions = {run.data["algorithm_revision"] for run in algorithm_runs}
        if len(revisions) != 1:
            raise EvidenceError(
                f"{algorithm}: runs span algorithm revisions {sorted(revisions)}"
            )


def as_algorithm_runs(runs: Sequence[RunEvidence]) -> list[RunEvidence]:
    """Adapt the neutral report renderer to group rows by algorithm."""
    adapted: list[RunEvidence] = []
    for run in runs:
        data: dict[str, Any] = dict(run.data)
        data["framework"] = data["algorithm"]
        data["framework_version"] = data["algorithm_revision"]
        adapted.append(RunEvidence(run.source, data))
    return adapted


def write_algorithm_report(runs: Sequence[RunEvidence], output_dir: Path) -> None:
    """Write measured algorithm comparison artifacts."""
    result = summarize(as_algorithm_runs(runs))
    result["report_kind"] = "measured-algorithm-comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(result).replace(
        "# Measured framework comparison", "# Measured algorithm comparison", 1
    )
    (output_dir / "comparison.md").write_text(markdown, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate real, matched algorithm evidence"
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/algorithm_comparison/report"),
    )
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        runs = load_algorithm_evidence(args.inputs)
        validate_algorithm_comparison(runs, args.min_seeds)
        if not args.validate_only:
            write_algorithm_report(runs, args.output_dir)
    except EvidenceError as exc:
        print(f"algorithm comparison rejected: {exc}", file=sys.stderr)
        return 2
    print(
        f"validated {len(runs)} measured runs across "
        f"{len({run.data['algorithm'] for run in runs})} algorithms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
