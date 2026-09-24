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


def hash_artifact(path: Path) -> str:
    """Hash a retained result file or directory tree without following links."""
    if not path.exists() or path.is_symlink():
        raise EvidenceError(f"artifact is missing or unsafe: {path}")
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    entries = sorted(path.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise EvidenceError(f"artifact tree contains a symlink: {path}")
    files = [item for item in entries if item.is_file()]
    if not files:
        raise EvidenceError(f"artifact directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def verify_retained_artifact(data: Mapping[str, Any], source: Path) -> Path:
    """Verify a schema-v2 artifact path and digest within its evidence bundle."""
    raw_path = data.get("artifact_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise EvidenceError(f"{source}: artifact_path must be non-empty")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise EvidenceError(f"{source}: artifact_path must be relative")
    bundle_root = source.resolve().parent
    artifact = (bundle_root / candidate).resolve()
    if not artifact.is_relative_to(bundle_root):
        raise EvidenceError(f"{source}: artifact_path escapes the evidence bundle")
    if hash_artifact(artifact) != data.get("artifact_sha256"):
        raise EvidenceError(
            f"{source}: retained artifact digest does not match artifact_sha256"
        )
    return artifact


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
    "harness_commit",
    "protocol",
    "cache_policy",
    "algorithm",
    "algorithm_revision",
    "model",
    "model_revision",
    "task",
    "dataset_revision",
)
PROVIDER_COST_SOURCE = "authoritative-provider-rate-x-observed-lifetime"


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
        return (
            (self.data["schema_version"], self.data.get("manifest_sha256"))
            + tuple(self.data[field] for field in MATCH_FIELDS)
            + (
                json.dumps(self.data["config"], sort_keys=True, separators=(",", ":")),
                hardware["gpu"],
                hardware["gpu_count"],
                hardware["cuda"],
            )
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
    schema_version = data.get("schema_version")
    if schema_version not in {1, 2}:
        raise EvidenceError(f"{source}: schema_version must be 1 or 2")
    if data.get("measured") is not True:
        raise EvidenceError(
            f"{source}: measured must be true; simulated or estimated runs are forbidden"
        )

    for field in REQUIRED_STRINGS:
        _require_nonempty_string(data, field, source)
    for field in ("harness_commit", "model_revision", "dataset_revision"):
        value = str(data[field])
        if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise EvidenceError(
                f"{source}: {field} must be a full 40-character lowercase hex commit"
            )

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
    zero_fraction = metrics.get("train_reward_zero_fraction")
    if isinstance(zero_fraction, (int, float)) and not isinstance(zero_fraction, bool):
        if float(zero_fraction) >= 1.0:
            raise EvidenceError(
                f"{source}: training reward was identically zero at every step "
                "(no learning signal); the run is not comparable"
            )

    artifact_sha256 = data.get("artifact_sha256")
    if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
        raise EvidenceError(f"{source}: artifact_sha256 must contain 64 hex characters")
    try:
        bytes.fromhex(artifact_sha256)
    except ValueError as exc:
        raise EvidenceError(f"{source}: artifact_sha256 is not hexadecimal") from exc

    if schema_version == 2:
        manifest_sha256 = data.get("manifest_sha256")
        if (
            not isinstance(manifest_sha256, str)
            or len(manifest_sha256) != 64
            or any(char not in "0123456789abcdef" for char in manifest_sha256)
        ):
            raise EvidenceError(
                f"{source}: manifest_sha256 must be 64 lowercase hex characters"
            )
        verify_retained_artifact(data, source)

    return RunEvidence(source=source, data=data)


# Launcher-owned records that may sit next to evidence files; they are not
# runs and are skipped when a directory is given.
_NON_EVIDENCE_KINDS = (
    "stateset-runpod-shootout-provider-record",
    "stateset-runpod-conformance-provider-record",
    "framework-shootout-accounting",
)


def _is_launcher_record(path: Path) -> bool:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(raw, dict) and raw.get("kind") in _NON_EVIDENCE_KINDS


def discover_inputs(inputs: Sequence[Path]) -> list[Path]:
    """Resolve files and directories into a deterministic JSON file list.

    Directories contribute their top-level ``*.json`` files minus launcher
    records (provider/accounting documents identified by ``kind``); files
    named explicitly are always included.
    """
    paths: list[Path] = []
    for candidate in inputs:
        if candidate.is_dir():
            paths.extend(
                p
                for p in sorted(candidate.glob("*.json"))
                if not _is_launcher_record(p)
            )
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


def load_provider_cost_bundle(
    bundle: Path, runs: Sequence[RunEvidence]
) -> dict[str, Any]:
    """Validate every RunPod lifecycle record and return complete matrix cost."""
    if not bundle.is_dir():
        raise EvidenceError("provider cost accounting requires one evidence directory")
    paths = sorted(bundle.glob("runpod-provider*.json"))
    if not paths:
        raise EvidenceError(f"{bundle}: no RunPod provider cost records found")
    expected_manifests = {str(run.data.get("manifest_sha256")) for run in runs}
    expected_harnesses = {str(run.data["harness_commit"]) for run in runs}
    if len(expected_manifests) != 1 or "None" in expected_manifests:
        raise EvidenceError("evidence does not share one shootout manifest digest")
    if len(expected_harnesses) != 1:
        raise EvidenceError("evidence does not share one harness revision")
    total_cost = 0.0
    total_lifetime = 0.0
    pod_ids: set[str] = set()
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvidenceError(f"{path}: invalid provider record") from exc
        if (
            not isinstance(record, Mapping)
            or record.get("schema_version") != 2
            or record.get("kind") != "stateset-runpod-shootout-provider-record"
            or record.get("provider") != "runpod"
            or record.get("cost_source") != PROVIDER_COST_SOURCE
        ):
            raise EvidenceError(f"{path}: invalid provider cost schema")
        if record.get("status") not in {"completed", "failed"}:
            raise EvidenceError(f"{path}: provider lifecycle status is incomplete")
        if record.get("shootout_manifest_sha256") not in expected_manifests:
            raise EvidenceError(f"{path}: provider record manifest mismatch")
        if record.get("harness_revision") not in expected_harnesses:
            raise EvidenceError(f"{path}: provider record harness mismatch")
        if record.get("termination_confirmed") is not True:
            raise EvidenceError(f"{path}: provider cleanup is not confirmed")
        pod_id = record.get("pod_id")
        if not isinstance(pod_id, str) or not pod_id or pod_id in pod_ids:
            raise EvidenceError(f"{path}: provider pod_id must be unique")
        pod_ids.add(pod_id)
        rate = _require_finite_number(
            record, "authoritative_pod_cost_per_hr_usd", path, minimum=0.0
        )
        lifetime = _require_finite_number(
            record, "pod_lifetime_seconds", path, minimum=0.0
        )
        cost = _require_finite_number(record, "estimated_cost_usd", path, minimum=0.0)
        if rate <= 0 or lifetime <= 0 or cost <= 0:
            raise EvidenceError(
                f"{path}: provider rate, lifetime, and cost must be > 0"
            )
        if not math.isclose(cost, rate * lifetime / 3600.0, rel_tol=0.0, abs_tol=1e-4):
            raise EvidenceError(f"{path}: provider cost arithmetic is inconsistent")
        total_cost += cost
        total_lifetime += lifetime
    return {
        "provider": "runpod",
        "cost_source": PROVIDER_COST_SOURCE,
        "records": len(paths),
        "pod_ids": sorted(pod_ids),
        "total_lifetime_seconds": round(total_lifetime, 3),
        "total_cost_usd": round(total_cost, 6),
    }


def validate_comparison(
    runs: Sequence[RunEvidence],
    min_seeds: int = 3,
    required_frameworks: Sequence[str] = (),
    minimum_schema_version: int = 1,
) -> None:
    """Require matched protocols, unique seeds, and adequate replication."""
    if min_seeds < 1:
        raise EvidenceError("min_seeds must be >= 1")
    if minimum_schema_version not in {1, 2}:
        raise EvidenceError("minimum_schema_version must be 1 or 2")
    legacy = [
        run.source
        for run in runs
        if int(run.data["schema_version"]) < minimum_schema_version
    ]
    if legacy:
        raise EvidenceError(
            "comparison requires verifiable schema_version>=2 evidence; "
            "legacy schema-v1 rows need explicit historical opt-in"
        )
    keys = {run.comparison_key for run in runs}
    if len(keys) != 1:
        differing: list[str] = []
        fields = (
            "schema_version",
            "manifest_sha256",
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
        "evidence_schema_version": first.data["schema_version"],
        "manifest_sha256": first.data.get("manifest_sha256"),
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
        f"- Harness commit: `{comparison['harness_commit']}`",
        f"- Model: `{comparison['model']}` at `{comparison['model_revision']}`",
        f"- Task/data: `{comparison['task']}` at `{comparison['dataset_revision']}`",
        f"- Hardware: {hardware['gpu_count']}× {hardware['gpu']} (CUDA {hardware['cuda']})",
    ]
    manifest_sha256 = summary.get("manifest_sha256")
    if isinstance(manifest_sha256, str):
        lines.append(f"- Shootout manifest SHA-256: `{manifest_sha256}`")
    provider_cost = summary.get("provider_cost")
    if isinstance(provider_cost, Mapping):
        lines.extend(
            [
                f"- Provider cost: ${float(provider_cost['total_cost_usd']):.4f} "
                f"across {provider_cost['records']} pod lifecycle record(s)",
                f"- Cost source: `{provider_cost['cost_source']}`",
            ]
        )
    lines.extend(
        [
            "",
            "| Framework | Version | Seeds | Samples/s | Wall clock (s) | Peak VRAM (MiB) | Baseline | Final | Improvement |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
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


def write_report(
    runs: Sequence[RunEvidence],
    output_dir: Path,
    provider_cost: Mapping[str, Any] | None = None,
) -> None:
    """Write machine-readable and human-readable comparison artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = summarize(runs)
    if provider_cost is not None:
        result["provider_cost"] = dict(provider_cost)
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
    parser.add_argument(
        "--require-provider-cost",
        action="store_true",
        help="require and total schema-v2 RunPod records in one input directory",
    )
    parser.add_argument(
        "--allow-legacy-schema-v1",
        action="store_true",
        help="allow historical rows whose retained artifact cannot be re-hashed",
    )
    args = parser.parse_args(argv)
    try:
        runs = load_evidence(args.inputs)
        validate_comparison(
            runs,
            min_seeds=args.min_seeds,
            required_frameworks=args.required_framework,
            minimum_schema_version=1 if args.allow_legacy_schema_v1 else 2,
        )
        provider_cost = None
        if args.require_provider_cost:
            if len(args.inputs) != 1:
                raise EvidenceError(
                    "--require-provider-cost needs exactly one evidence directory"
                )
            provider_cost = load_provider_cost_bundle(args.inputs[0], runs)
        if not args.validate_only:
            write_report(runs, args.output_dir, provider_cost)
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
