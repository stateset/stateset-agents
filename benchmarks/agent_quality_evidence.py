#!/usr/bin/env python3
"""Gate measured agent quality across standard held-out suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REQUIRED_SUITES = ("tau3-bench", "bfcl-v4", "swe-bench-verified")
COST_SOURCES = frozenset(
    {"provider-api", "provider-invoice", "local-meter", "sponsored"}
)
HEX_DIGITS = frozenset("0123456789abcdef")


class AgentQualityEvidenceError(ValueError):
    """Raised when agent benchmark evidence is incomplete or inconsistent."""


def _string(data: Mapping[str, Any], key: str, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgentQualityEvidenceError(f"{source}: {key} must be non-empty")
    return value


def _number(data: Mapping[str, Any], key: str, source: Path) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentQualityEvidenceError(f"{source}: {key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AgentQualityEvidenceError(f"{source}: {key} must be finite")
    return result


def _integer(
    data: Mapping[str, Any], key: str, source: Path, *, minimum: int = 0
) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AgentQualityEvidenceError(
            f"{source}: {key} must be an integer >= {minimum}"
        )
    return value


def _digest(data: Mapping[str, Any], key: str, source: Path, length: int) -> str:
    value = _string(data, key, source)
    if len(value) != length or any(char not in HEX_DIGITS for char in value):
        raise AgentQualityEvidenceError(
            f"{source}: {key} must be {length} lowercase hex characters"
        )
    return value


def validate_run(data: Mapping[str, Any], source: Path) -> dict[str, Any]:
    """Validate one paired base-versus-trained held-out evaluation."""
    if (
        data.get("schema_version") != 2
        or data.get("kind") != "stateset-agent-quality-evidence"
        or data.get("status") != "completed"
        or data.get("measured") is not True
    ):
        raise AgentQualityEvidenceError(
            f"{source}: completed measured schema_version=2 evidence is required"
        )
    suite = _string(data, "suite", source)
    if suite not in REQUIRED_SUITES:
        raise AgentQualityEvidenceError(f"{source}: unsupported suite {suite!r}")
    for key in (
        "run_id",
        "protocol",
        "framework_version",
        "baseline_model",
        "trained_model",
        "split",
        "timestamp",
        "cost_source",
    ):
        _string(data, key, source)
    if data["cost_source"] not in COST_SOURCES:
        raise AgentQualityEvidenceError(
            f"{source}: cost_source must be one of {sorted(COST_SOURCES)}"
        )
    _digest(data, "suite_revision", source, 40)
    _digest(data, "baseline_model_revision", source, 40)
    _digest(data, "trained_model_revision", source, 40)
    _digest(data, "training_artifact_sha256", source, 64)
    _digest(data, "harness_commit", source, 40)
    _digest(data, "paired_task_ids_sha256", source, 64)
    _digest(data, "artifact_sha256", source, 64)
    _integer(data, "seed", source)

    config = data.get("evaluation_config")
    if not isinstance(config, Mapping) or not config:
        raise AgentQualityEvidenceError(
            f"{source}: evaluation_config must be a non-empty object"
        )
    expected = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if _digest(data, "evaluation_config_sha256", source, 64) != expected:
        raise AgentQualityEvidenceError(
            f"{source}: evaluation_config_sha256 does not match config"
        )

    tasks = _integer(data, "tasks", source, minimum=1)
    baseline_successful = _integer(data, "baseline_successful_episodes", source)
    trained_successful = _integer(data, "trained_successful_episodes", source)
    if baseline_successful > tasks or trained_successful > tasks:
        raise AgentQualityEvidenceError(
            f"{source}: successful episode count exceeds tasks"
        )
    baseline = _number(data, "baseline_score", source)
    trained = _number(data, "trained_score", source)
    if not 0.0 <= baseline <= 1.0 or not 0.0 <= trained <= 1.0:
        raise AgentQualityEvidenceError(f"{source}: scores must be within [0, 1]")
    duration = _number(data, "evaluation_seconds", source)
    cost = _number(data, "evaluation_cost_usd", source)
    if duration <= 0 or cost < 0:
        raise AgentQualityEvidenceError(
            f"{source}: duration must be positive and cost non-negative"
        )
    per_success = _number(data, "cost_per_successful_episode_usd", source)
    expected_cost = cost / trained_successful if trained_successful else 0.0
    if not math.isclose(per_success, expected_cost, rel_tol=0.02, abs_tol=1e-9):
        raise AgentQualityEvidenceError(
            f"{source}: cost per successful episode is inconsistent"
        )
    return dict(data)


def load_runs(inputs: Sequence[Path]) -> list[dict[str, Any]]:
    """Load and validate every JSON document found in the inputs."""
    paths: list[Path] = []
    for candidate in inputs:
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.json")))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            raise AgentQualityEvidenceError(f"input does not exist: {candidate}")
    if not paths:
        raise AgentQualityEvidenceError("no agent-quality evidence found")
    runs: list[dict[str, Any]] = []
    for path in dict.fromkeys(item.resolve() for item in paths):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentQualityEvidenceError(f"{path}: invalid JSON") from exc
        if not isinstance(raw, Mapping):
            raise AgentQualityEvidenceError(f"{path}: evidence must be an object")
        runs.append(validate_run(raw, path))
    return runs


def validate_matrix(
    runs: Sequence[Mapping[str, Any]],
    *,
    min_seeds: int = 3,
    minimum_mean_improvement: float = 0.03,
) -> None:
    """Require matched suites and a positive paired 95% confidence bound."""
    if min_seeds < 3 or minimum_mean_improvement <= 0:
        raise AgentQualityEvidenceError("invalid matrix gate configuration")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    run_ids: set[str] = set()
    for run in runs:
        run_id = str(run["run_id"])
        if run_id in run_ids:
            raise AgentQualityEvidenceError(f"duplicate run_id: {run_id}")
        run_ids.add(run_id)
        grouped.setdefault(str(run["suite"]), []).append(run)
    if set(grouped) != set(REQUIRED_SUITES):
        raise AgentQualityEvidenceError(
            f"suite matrix mismatch: required={sorted(REQUIRED_SUITES)}, "
            f"actual={sorted(grouped)}"
        )
    first = runs[0]
    for run in runs[1:]:
        changed = [
            field
            for field in (
                "framework_version",
                "protocol",
                "baseline_model",
                "baseline_model_revision",
                "trained_model",
                "trained_model_revision",
                "training_artifact_sha256",
                "harness_commit",
                "evaluation_config_sha256",
            )
            if run[field] != first[field]
        ]
        if changed:
            raise AgentQualityEvidenceError("matrix mixes " + ", ".join(changed))

    expected_seeds: set[int] | None = None
    for suite, suite_runs in sorted(grouped.items()):
        for field in (
            "suite_revision",
            "split",
            "tasks",
            "paired_task_ids_sha256",
        ):
            if len({str(run[field]) for run in suite_runs}) != 1:
                raise AgentQualityEvidenceError(f"{suite}: {field} drift")
        seeds = [int(run["seed"]) for run in suite_runs]
        if len(seeds) != len(set(seeds)) or len(seeds) < min_seeds:
            raise AgentQualityEvidenceError(
                f"{suite}: at least {min_seeds} unique seeds are required"
            )
        seed_set = set(seeds)
        if expected_seeds is None:
            expected_seeds = seed_set
        elif seed_set != expected_seeds:
            raise AgentQualityEvidenceError(f"{suite}: seed set mismatch")
        improvements = [
            float(run["trained_score"]) - float(run["baseline_score"])
            for run in suite_runs
        ]
        mean = statistics.fmean(improvements)
        if mean < minimum_mean_improvement:
            raise AgentQualityEvidenceError(
                f"{suite}: mean improvement {mean:.6f} is below the floor"
            )
        standard_error = statistics.stdev(improvements) / math.sqrt(len(improvements))
        # Conservative for n>3; for three seeds 4.303 is the 95% t critical value.
        if mean - 4.303 * standard_error <= 0:
            raise AgentQualityEvidenceError(
                f"{suite}: paired 95% confidence bound does not exclude zero"
            )


def summarize(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create a compact result retaining suite effect sizes and cost."""
    suites: dict[str, Any] = {}
    for suite in REQUIRED_SUITES:
        selected = [run for run in runs if run["suite"] == suite]
        improvements = [
            float(run["trained_score"]) - float(run["baseline_score"])
            for run in selected
        ]
        suites[suite] = {
            "seeds": sorted(int(run["seed"]) for run in selected),
            "mean_improvement": statistics.fmean(improvements),
            "total_tasks": sum(int(run["tasks"]) for run in selected),
            "total_evaluation_cost_usd": sum(
                float(run["evaluation_cost_usd"]) for run in selected
            ),
        }
    return {
        "schema_version": 2,
        "kind": "stateset-agent-quality-matrix",
        "passed": True,
        "protocol": runs[0]["protocol"],
        "baseline_model": runs[0]["baseline_model"],
        "baseline_model_revision": runs[0]["baseline_model_revision"],
        "trained_model": runs[0]["trained_model"],
        "trained_model_revision": runs[0]["trained_model_revision"],
        "training_artifact_sha256": runs[0]["training_artifact_sha256"],
        "suites": suites,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the standard agent-quality publication gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--minimum-mean-improvement", type=float, default=0.03)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        runs = load_runs(args.inputs)
        validate_matrix(
            runs,
            min_seeds=args.min_seeds,
            minimum_mean_improvement=args.minimum_mean_improvement,
        )
        report = summarize(runs)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
    except AgentQualityEvidenceError as exc:
        print(f"agent-quality evidence rejected: {exc}", file=sys.stderr)
        return 2
    print(f"validated {len(runs)} measured agent-quality runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
