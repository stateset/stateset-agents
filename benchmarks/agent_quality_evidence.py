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
from fractions import Fraction
from pathlib import Path
from typing import Any

# Suites every published agent-quality matrix must include. The flagship is a
# customer-support / tool-use policy, so the gate is anchored on tau3-bench
# (multi-turn tool-agent-user dialogue) and BFCL V4 (function calling).
REQUIRED_SUITES = ("tau3-bench", "bfcl-v4")
# Suites a matrix may additionally include. An optional suite is never
# exempt: once it is present it must pass the same per-suite gate.
OPTIONAL_SUITES = ("swe-bench-verified",)
SUPPORTED_SUITES = REQUIRED_SUITES + OPTIONAL_SUITES
COST_SOURCES = frozenset(
    {"provider-api", "provider-invoice", "local-meter", "sponsored"}
)
HEX_DIGITS = frozenset("0123456789abcdef")


class AgentQualityEvidenceError(ValueError):
    """Raised when agent benchmark evidence is incomplete or inconsistent."""


def validate_suite_roster(names: Sequence[str]) -> tuple[str, ...]:
    """Check a suite roster against the required/optional suite policy.

    The roster must name every required suite, may name optional suites, and
    may not repeat or invent suites. Returns the roster in canonical order.
    """
    duplicates = sorted({name for name in names if list(names).count(name) > 1})
    if duplicates:
        raise AgentQualityEvidenceError(f"duplicate suites in roster: {duplicates}")
    unknown = sorted(set(names).difference(SUPPORTED_SUITES))
    if unknown:
        raise AgentQualityEvidenceError(
            f"unsupported suites {unknown}; supported={list(SUPPORTED_SUITES)}"
        )
    missing = [suite for suite in REQUIRED_SUITES if suite not in names]
    if missing:
        raise AgentQualityEvidenceError(
            f"missing required suites {missing}; required={list(REQUIRED_SUITES)}"
        )
    return tuple(suite for suite in SUPPORTED_SUITES if suite in names)


def hash_artifact(path: Path) -> str:
    """Hash a retained file or directory tree without following symlinks."""
    if not path.exists() or path.is_symlink():
        raise AgentQualityEvidenceError(f"artifact is missing or unsafe: {path}")
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    entries = sorted(path.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise AgentQualityEvidenceError(f"artifact tree contains a symlink: {path}")
    files = [item for item in entries if item.is_file()]
    if not files:
        raise AgentQualityEvidenceError(f"artifact directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def verify_retained_artifact(data: Mapping[str, Any], source: Path) -> Path:
    """Resolve and verify the portable artifact referenced by one evidence row."""
    raw_path = data.get("artifact_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise AgentQualityEvidenceError(f"{source}: artifact_path must be non-empty")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise AgentQualityEvidenceError(f"{source}: artifact_path must be relative")
    bundle_root = source.resolve().parent.parent
    artifact = (source.resolve().parent / candidate).resolve()
    if not artifact.is_relative_to(bundle_root):
        raise AgentQualityEvidenceError(
            f"{source}: artifact_path escapes the evidence bundle"
        )
    actual = hash_artifact(artifact)
    if actual != data.get("artifact_sha256"):
        raise AgentQualityEvidenceError(
            f"{source}: retained artifact digest does not match artifact_sha256"
        )
    return artifact


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


#: Scores are serialized floats; ``successes / tasks`` is the exact rational
#: they must encode. Any deviation beyond float rounding is a tampered score.
SCORE_TOLERANCE = 1e-12


def _require_score_matches(
    score: float, successes: int, tasks: int, *, key: str, source: Path
) -> None:
    expected = Fraction(successes, tasks)
    if abs(Fraction(score) - expected) > SCORE_TOLERANCE:
        raise AgentQualityEvidenceError(
            f"{source}: {key} {score!r} does not equal {successes}/{tasks}"
        )


def _load_task_records(path: Path, source: Path) -> list[tuple[str, bool]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AgentQualityEvidenceError(
            f"{source}: retained artifact lacks per-task records {path.name} "
            f"under {path.parent.name}/"
        ) from exc
    records: list[tuple[str, bool]] = []
    for line in lines:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AgentQualityEvidenceError(
                f"{source}: invalid record in {path}"
            ) from exc
        task_id = raw.get("task_id") if isinstance(raw, Mapping) else None
        success = raw.get("success") if isinstance(raw, Mapping) else None
        if not isinstance(task_id, str) or not isinstance(success, bool):
            raise AgentQualityEvidenceError(f"{source}: invalid record in {path}")
        records.append((task_id, success))
    return records


def verify_task_records(data: Mapping[str, Any], artifact: Path, source: Path) -> None:
    """Recompute counts, scores and the task digest from retained records.

    The paired harness retains ``baseline/tasks.jsonl`` and
    ``trained/tasks.jsonl`` inside the hashed artifact. A row whose reported
    successes, scores or paired task digest disagree with those per-task
    records is rejected even when it is internally consistent.
    """
    tasks = int(data["tasks"])
    task_ids: list[str] | None = None
    for policy in ("baseline", "trained"):
        records = _load_task_records(artifact / policy / "tasks.jsonl", source)
        ids = [task_id for task_id, _ in records]
        if len(ids) != tasks or len(set(ids)) != tasks:
            raise AgentQualityEvidenceError(
                f"{source}: {policy} per-task records do not cover {tasks} tasks"
            )
        if task_ids is not None and ids != task_ids:
            raise AgentQualityEvidenceError(
                f"{source}: per-task records are not identically ordered"
            )
        task_ids = ids
        successes = sum(success for _, success in records)
        if successes != data[f"{policy}_successful_episodes"]:
            raise AgentQualityEvidenceError(
                f"{source}: {policy}_successful_episodes disagrees with the "
                f"retained per-task records ({successes}/{tasks})"
            )
        _require_score_matches(
            data[f"{policy}_score"],
            successes,
            tasks,
            key=f"{policy}_score (per-task records)",
            source=source,
        )
    digest = hashlib.sha256(
        json.dumps(task_ids, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != data["paired_task_ids_sha256"]:
        raise AgentQualityEvidenceError(
            f"{source}: paired_task_ids_sha256 disagrees with the per-task records"
        )


def validate_run(data: Mapping[str, Any], source: Path) -> dict[str, Any]:
    """Validate one paired base-versus-trained held-out evaluation."""
    if (
        data.get("schema_version") != 3
        or data.get("kind") != "stateset-agent-quality-evidence"
        or data.get("status") != "completed"
        or data.get("measured") is not True
    ):
        raise AgentQualityEvidenceError(
            f"{source}: completed measured schema_version=3 evidence is required"
        )
    suite = _string(data, "suite", source)
    if suite not in SUPPORTED_SUITES:
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
        "artifact_path",
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
    for key, successes in (
        ("baseline_score", baseline_successful),
        ("trained_score", trained_successful),
    ):
        _require_score_matches(data[key], successes, tasks, key=key, source=source)
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
        run = validate_run(raw, path)
        artifact = verify_retained_artifact(run, path)
        verify_task_records(run, artifact, path)
        runs.append(run)
    return runs


def validate_matrix(
    runs: Sequence[Mapping[str, Any]],
    *,
    min_seeds: int = 3,
    minimum_mean_improvement: float = 0.03,
    expected_suites: Sequence[str] | None = None,
) -> None:
    """Require matched suites and a positive paired 95% confidence bound.

    Every required suite must be present. Optional suites may be present, and
    every present suite must clear the same per-suite gate. When
    ``expected_suites`` is given (the executed manifest roster), the evidence
    must cover exactly that roster so no attempted suite can be dropped.
    """
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
    if not runs:
        raise AgentQualityEvidenceError("no agent-quality evidence runs")
    try:
        validate_suite_roster(list(grouped))
    except AgentQualityEvidenceError as exc:
        raise AgentQualityEvidenceError(
            f"suite matrix mismatch: {exc}; actual={sorted(grouped)}"
        ) from exc
    if expected_suites is not None:
        expected = set(validate_suite_roster(list(expected_suites)))
        if set(grouped) != expected:
            raise AgentQualityEvidenceError(
                "suite matrix mismatch: evidence must cover exactly the executed "
                f"roster {sorted(expected)}; actual={sorted(grouped)}"
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
    present = {str(run["suite"]) for run in runs}
    for suite in (name for name in SUPPORTED_SUITES if name in present):
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
        "schema_version": 3,
        "kind": "stateset-agent-quality-matrix",
        "passed": True,
        "protocol": runs[0]["protocol"],
        "baseline_model": runs[0]["baseline_model"],
        "baseline_model_revision": runs[0]["baseline_model_revision"],
        "trained_model": runs[0]["trained_model"],
        "trained_model_revision": runs[0]["trained_model_revision"],
        "training_artifact_sha256": runs[0]["training_artifact_sha256"],
        "required_suites": list(REQUIRED_SUITES),
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
