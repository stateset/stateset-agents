#!/usr/bin/env python3
"""Gate a complete, artifact-valid external-backend conformance roster."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend_conformance import ConformanceError, load_evidence, write_json_once

REQUIRED_BACKENDS = ("nemo-rl", "openrlhf", "verl")
MATCHED_EXPERIMENT_FIELDS = (
    "algorithm",
    "model",
    "model_revision",
    "seed",
    "task",
)
MATCHED_EXECUTION_FIELDS = (
    "provider",
    "provider_tier",
    "gpu_name",
    "gpu_count",
    "timeout_seconds",
    "max_cost_usd",
)


class ConformanceSuiteError(ValueError):
    """Raised when a conformance roster is incomplete or inconsistent."""


@dataclass(frozen=True)
class ConformanceRecord:
    """One validated conformance document and its exact source bytes."""

    source: Path
    evidence: Mapping[str, Any]
    evidence_sha256: str

    @property
    def backend(self) -> str:
        return str(self.evidence["backend"])


def discover_evidence(inputs: Sequence[Path]) -> list[Path]:
    """Find only canonical conformance documents, deterministically."""
    discovered: list[Path] = []
    for candidate in inputs:
        if candidate.is_dir():
            discovered.extend(sorted(candidate.rglob("conformance.json")))
        elif candidate.is_file():
            discovered.append(candidate)
        else:
            raise ConformanceSuiteError(f"input does not exist: {candidate}")
    paths = list(dict.fromkeys(path.resolve() for path in discovered))
    if not paths:
        raise ConformanceSuiteError("no conformance.json evidence found")
    return paths


def load_records(inputs: Sequence[Path]) -> list[ConformanceRecord]:
    """Load every document and rehash its colocated artifact bytes."""
    records: list[ConformanceRecord] = []
    for path in discover_evidence(inputs):
        try:
            evidence = load_evidence(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except (ConformanceError, OSError) as exc:
            raise ConformanceSuiteError(f"{path}: {exc}") from exc
        records.append(
            ConformanceRecord(
                source=path,
                evidence=evidence,
                evidence_sha256=digest,
            )
        )
    return records


def _experiment_value(evidence: Mapping[str, Any], field: str) -> Any:
    experiment = evidence["manifest"]["experiment"]
    if field == "task":
        return experiment.get("task", "conformance")
    return experiment[field]


def validate_suite(
    records: Sequence[ConformanceRecord],
    required_backends: Sequence[str] = REQUIRED_BACKENDS,
) -> None:
    """Require one consistent, measured record for every requested backend."""
    required = tuple(name.strip() for name in required_backends)
    if not required or any(not name for name in required):
        raise ConformanceSuiteError("required_backends must contain non-empty names")
    if len(required) != len(set(required)):
        raise ConformanceSuiteError("required_backends must not contain duplicates")
    unsupported = sorted(set(required) - set(REQUIRED_BACKENDS))
    if unsupported:
        raise ConformanceSuiteError(
            "unsupported required backends: " + ", ".join(unsupported)
        )

    grouped: dict[str, list[ConformanceRecord]] = {}
    for record in records:
        grouped.setdefault(record.backend, []).append(record)
    duplicates = sorted(name for name, values in grouped.items() if len(values) != 1)
    if duplicates:
        raise ConformanceSuiteError(
            "duplicate backend evidence is forbidden: " + ", ".join(duplicates)
        )
    actual = set(grouped)
    expected = set(required)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ConformanceSuiteError(
            f"backend roster mismatch; missing={missing}, unexpected={unexpected}"
        )

    first = records[0].evidence
    for record in records[1:]:
        changed: list[str] = []
        for field in ("harness_revision", "stateset_agents_version"):
            if record.evidence[field] != first[field]:
                changed.append(field)
        for field in MATCHED_EXPERIMENT_FIELDS:
            if _experiment_value(record.evidence, field) != _experiment_value(
                first, field
            ):
                changed.append(f"experiment.{field}")
        for field in MATCHED_EXECUTION_FIELDS:
            if record.evidence["execution"][field] != first["execution"][field]:
                changed.append(f"execution.{field}")
        if changed:
            raise ConformanceSuiteError(
                f"{record.backend}: conformance roster mixes {', '.join(changed)}"
            )


def summarize(records: Sequence[ConformanceRecord]) -> dict[str, Any]:
    """Build a compact report bound to every evidence document and artifact."""
    ordered = sorted(records, key=lambda record: record.backend)
    first = ordered[0].evidence
    experiment = first["manifest"]["experiment"]
    entries = []
    for record in ordered:
        evidence = record.evidence
        entries.append(
            {
                "backend": record.backend,
                "backend_version": evidence["backend_version"],
                "source": record.source.as_posix(),
                "evidence_sha256": record.evidence_sha256,
                "manifest_sha256": evidence["manifest_sha256"],
                "experiment_sha256": evidence["experiment_sha256"],
                "artifact_uri": evidence["artifact_uri"],
                "artifact_sha256": evidence["artifact_sha256"],
                "wall_time_seconds": evidence["wall_time_seconds"],
                "hardware": evidence["hardware"],
                "execution": evidence["execution"],
            }
        )
    suite_digest = hashlib.sha256(
        "\n".join(sorted(record.evidence_sha256 for record in ordered)).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "kind": "stateset-external-backend-conformance-suite",
        "status": "completed",
        "measured": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "harness_revision": first["harness_revision"],
        "stateset_agents_version": first["stateset_agents_version"],
        "required_backends": sorted(record.backend for record in ordered),
        "backend_count": len(ordered),
        "experiment": {
            field: (
                experiment.get(field, "conformance")
                if field == "task"
                else experiment[field]
            )
            for field in MATCHED_EXPERIMENT_FIELDS
        },
        "suite_sha256": suite_digest,
        "backends": entries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the complete external-backend gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--required-backend",
        action="append",
        default=[],
        help="Exact backend roster (repeatable; defaults to all supported backends).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/backend_conformance/suite.json"),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    required = args.required_backend or list(REQUIRED_BACKENDS)
    try:
        records = load_records(args.inputs)
        validate_suite(records, required)
        if not args.validate_only:
            write_json_once(args.output, summarize(records))
    except (ConformanceSuiteError, ConformanceError, OSError) as exc:
        print(f"conformance suite rejected: {exc}", file=sys.stderr)
        return 2
    print(f"validated {len(records)} measured external-backend conformance records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
