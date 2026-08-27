#!/usr/bin/env python3
"""Gate measured fault-injection and checkpoint-recovery evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REQUIRED_FAULTS = ("worker_exit", "controller_restart", "network_interruption")


class ReliabilityEvidenceError(ValueError):
    """Raised when recovery evidence is incomplete or violates an invariant."""


def _nonempty(data: Mapping[str, Any], key: str, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReliabilityEvidenceError(f"{source}: {key} must be a non-empty string")
    return value


def validate_run(data: Mapping[str, Any], source: Path) -> dict[str, Any]:
    """Validate the schema and no-corruption recovery invariants for one run."""
    if data.get("schema_version") != 1 or data.get("measured") is not True:
        raise ReliabilityEvidenceError(
            f"{source}: schema_version=1 and measured=true are required"
        )
    for key in (
        "run_id",
        "framework_version",
        "harness_commit",
        "protocol",
        "model",
        "model_revision",
        "timestamp",
        "command",
    ):
        _nonempty(data, key, source)
    seed = data.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ReliabilityEvidenceError(f"{source}: seed must be non-negative integer")

    config = data.get("config")
    if not isinstance(config, Mapping) or not config:
        raise ReliabilityEvidenceError(f"{source}: config must be a non-empty object")
    config_digest = data.get("config_sha256")
    if not isinstance(config_digest, str) or len(config_digest) != 64:
        raise ReliabilityEvidenceError(f"{source}: config_sha256 must be 64 hex chars")
    try:
        bytes.fromhex(config_digest)
    except ValueError as exc:
        raise ReliabilityEvidenceError(
            f"{source}: config_sha256 is not hexadecimal"
        ) from exc
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != config_digest:
        raise ReliabilityEvidenceError(
            f"{source}: config_sha256 does not match canonical config"
        )

    for section, fields in (
        ("hardware", ("accelerator", "cuda")),
        ("software", ("python", "torch")),
    ):
        values = data.get(section)
        if not isinstance(values, Mapping):
            raise ReliabilityEvidenceError(f"{source}: {section} must be an object")
        for field in fields:
            _nonempty(values, field, source)

    fault = data.get("fault")
    if not isinstance(fault, Mapping):
        raise ReliabilityEvidenceError(f"{source}: fault must be an object")
    fault_type = _nonempty(fault, "type", source)
    if fault_type not in REQUIRED_FAULTS:
        raise ReliabilityEvidenceError(
            f"{source}: unsupported fault type {fault_type!r}"
        )
    injected_at = fault.get("injected_at_step")
    if not isinstance(injected_at, int) or injected_at < 1:
        raise ReliabilityEvidenceError(f"{source}: injected_at_step must be >= 1")

    recovery = data.get("recovery")
    if not isinstance(recovery, Mapping):
        raise ReliabilityEvidenceError(f"{source}: recovery must be an object")
    for flag in ("resumed", "completed"):
        if recovery.get(flag) is not True:
            raise ReliabilityEvidenceError(f"{source}: recovery.{flag} must be true")
    for key in (
        "checkpoint_step",
        "resumed_step",
        "duplicate_updates",
        "data_loss_steps",
        "final_step",
        "expected_final_step",
        "resources_remaining",
    ):
        value = recovery.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReliabilityEvidenceError(f"{source}: recovery.{key} must be >= 0")
    seconds = recovery.get("recovery_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ReliabilityEvidenceError(f"{source}: recovery_seconds must be numeric")
    if not math.isfinite(float(seconds)) or float(seconds) <= 0:
        raise ReliabilityEvidenceError(
            f"{source}: recovery_seconds must be finite and > 0"
        )

    if recovery["checkpoint_step"] > injected_at:
        raise ReliabilityEvidenceError(
            f"{source}: checkpoint occurs after injected fault"
        )
    if recovery["resumed_step"] != recovery["checkpoint_step"]:
        raise ReliabilityEvidenceError(
            f"{source}: resumed_step must equal checkpoint_step for exact replay"
        )
    expected_loss = injected_at - recovery["checkpoint_step"]
    if recovery["data_loss_steps"] != expected_loss:
        raise ReliabilityEvidenceError(
            f"{source}: data_loss_steps does not match the checkpoint gap"
        )
    if recovery["duplicate_updates"] != 0:
        raise ReliabilityEvidenceError(
            f"{source}: duplicate optimizer updates detected"
        )
    if recovery["final_step"] != recovery["expected_final_step"]:
        raise ReliabilityEvidenceError(
            f"{source}: training did not reach expected final step"
        )
    if recovery["resources_remaining"] != 0:
        raise ReliabilityEvidenceError(f"{source}: resources remain after recovery run")

    digest = data.get("artifact_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ReliabilityEvidenceError(
            f"{source}: artifact_sha256 must be 64 hex chars"
        )
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ReliabilityEvidenceError(
            f"{source}: artifact_sha256 is not hexadecimal"
        ) from exc
    return dict(data)


def load_runs(inputs: Sequence[Path]) -> list[dict[str, Any]]:
    """Load files/directories and fail on the first invalid evidence document."""
    paths: list[Path] = []
    for candidate in inputs:
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.json")))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            raise ReliabilityEvidenceError(f"input does not exist: {candidate}")
    if not paths:
        raise ReliabilityEvidenceError("no reliability evidence found")
    runs = []
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReliabilityEvidenceError(f"{path}: invalid JSON") from exc
        if not isinstance(raw, Mapping):
            raise ReliabilityEvidenceError(f"{path}: top-level value must be an object")
        runs.append(validate_run(raw, path))
    return runs


def validate_matrix(
    runs: Sequence[Mapping[str, Any]],
    required_faults: Sequence[str] = REQUIRED_FAULTS,
    min_seeds: int = 3,
    max_data_loss_steps: int = 10,
) -> None:
    """Require matched three-seed evidence for every prescribed fault."""
    if min_seeds < 1 or max_data_loss_steps < 0:
        raise ReliabilityEvidenceError("invalid matrix gate configuration")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["fault"]["type"]), []).append(run)
    if set(grouped) != set(required_faults):
        raise ReliabilityEvidenceError(
            f"fault matrix mismatch: required={sorted(required_faults)}, "
            f"actual={sorted(grouped)}"
        )
    match_fields = (
        "framework_version",
        "harness_commit",
        "protocol",
        "model",
        "model_revision",
        "config_sha256",
    )
    first = runs[0]
    for run in runs[1:]:
        changed = [field for field in match_fields if run[field] != first[field]]
        for section, field in (
            ("hardware", "accelerator"),
            ("hardware", "cuda"),
            ("software", "torch"),
        ):
            if run[section][field] != first[section][field]:
                changed.append(f"{section}.{field}")
        if changed:
            raise ReliabilityEvidenceError(f"run matrix mixes {', '.join(changed)}")
    expected_seeds: set[int] | None = None
    for fault_type, fault_runs in sorted(grouped.items()):
        seeds = [int(run["seed"]) for run in fault_runs]
        if len(seeds) != len(set(seeds)):
            raise ReliabilityEvidenceError(f"{fault_type}: duplicate seeds")
        if len(seeds) < min_seeds:
            raise ReliabilityEvidenceError(
                f"{fault_type}: only {len(seeds)} seeds; need {min_seeds}"
            )
        seed_set = set(seeds)
        if expected_seeds is None:
            expected_seeds = seed_set
        elif seed_set != expected_seeds:
            raise ReliabilityEvidenceError(f"{fault_type}: seed set mismatch")
        worst_loss = max(int(run["recovery"]["data_loss_steps"]) for run in fault_runs)
        if worst_loss > max_data_loss_steps:
            raise ReliabilityEvidenceError(
                f"{fault_type}: data loss {worst_loss} exceeds {max_data_loss_steps} steps"
            )


def summarize(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Produce a compact machine-readable reliability report."""
    faults: dict[str, Any] = {}
    for fault_type in sorted({str(run["fault"]["type"]) for run in runs}):
        selected = [run for run in runs if run["fault"]["type"] == fault_type]
        faults[fault_type] = {
            "seeds": sorted(int(run["seed"]) for run in selected),
            "max_data_loss_steps": max(
                int(run["recovery"]["data_loss_steps"]) for run in selected
            ),
            "max_recovery_seconds": max(
                float(run["recovery"]["recovery_seconds"]) for run in selected
            ),
            "duplicate_updates": sum(
                int(run["recovery"]["duplicate_updates"]) for run in selected
            ),
            "resources_remaining": sum(
                int(run["recovery"]["resources_remaining"]) for run in selected
            ),
        }
    return {"schema_version": 1, "passed": True, "faults": faults}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate measured recovery evidence")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--max-data-loss-steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        runs = load_runs(args.inputs)
        validate_matrix(
            runs, min_seeds=args.min_seeds, max_data_loss_steps=args.max_data_loss_steps
        )
        report = summarize(runs)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
    except ReliabilityEvidenceError as exc:
        print(f"reliability evidence rejected: {exc}", file=sys.stderr)
        return 2
    print(f"validated {len(runs)} measured fault-injection runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
