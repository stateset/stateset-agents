#!/usr/bin/env python3
"""Execute a matched cross-framework benchmark manifest and emit evidence.

Framework adapters are ordinary commands. They must write a tiny neutral JSON
result; this orchestrator measures end-to-end wall time, hashes the resulting
artifact, adds immutable protocol provenance, and validates the final document
with :mod:`framework_comparison` before it can enter a report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from framework_comparison import EvidenceError, validate_document


class ShootoutError(ValueError):
    """Raised when a manifest or adapter result violates the protocol."""


MANIFEST_FIELDS = (
    "protocol",
    "algorithm",
    "algorithm_revision",
    "model",
    "model_revision",
    "task",
    "dataset_revision",
    "cache_policy",
)
REQUIRED_COMMAND_PLACEHOLDERS = {
    "{seed}",
    "{adapter_output}",
    "{artifact_dir}",
    "{model}",
    "{model_revision}",
    "{dataset_revision}",
    "{task}",
    "{config_json}",
}


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a framework-neutral shootout manifest."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ShootoutError(f"{path}: invalid JSON") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ShootoutError("manifest must be an object with schema_version=1")
    for field in MANIFEST_FIELDS:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ShootoutError(f"manifest.{field} must be a non-empty string")
    for field in ("model_revision", "dataset_revision"):
        value = str(raw[field])
        if len(value) != 40:
            raise ShootoutError(f"manifest.{field} must be a full 40-character commit")
    config = raw.get("config")
    if not isinstance(config, Mapping) or not config:
        raise ShootoutError("manifest.config must be a non-empty object")
    seeds = raw.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        )
        or len(seeds) != len(set(seeds))
    ):
        raise ShootoutError(
            "manifest.seeds must contain at least three unique integers"
        )
    hardware = raw.get("hardware")
    if not isinstance(hardware, Mapping):
        raise ShootoutError("manifest.hardware must be an object")
    if not isinstance(hardware.get("gpu"), str) or not hardware["gpu"].strip():
        raise ShootoutError("manifest.hardware.gpu must be a non-empty string")
    count = hardware.get("gpu_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ShootoutError("manifest.hardware.gpu_count must be >= 1")

    implementations = raw.get("implementations")
    if not isinstance(implementations, list) or len(implementations) < 2:
        raise ShootoutError("manifest requires at least two implementations")
    names: list[str] = []
    for implementation in implementations:
        if not isinstance(implementation, Mapping):
            raise ShootoutError("each implementation must be an object")
        for field in ("name", "version"):
            value = implementation.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ShootoutError(f"implementation.{field} must be non-empty")
        command = implementation.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise ShootoutError(
                "implementation.command must be a non-empty string list"
            )
        missing_placeholders = REQUIRED_COMMAND_PLACEHOLDERS.difference(command)
        if missing_placeholders:
            raise ShootoutError(
                "implementation.command is missing protocol placeholders: "
                + ", ".join(sorted(missing_placeholders))
            )
        names.append(str(implementation["name"]))
    if len(names) != len(set(names)):
        raise ShootoutError("implementation names must be unique")
    return dict(raw)


def git_commit(root: Path) -> str:
    """Return the immutable harness commit, rejecting non-repository runs."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ShootoutError("benchmark harness worktree must be clean")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) != 40:
        raise ShootoutError("could not resolve a full harness git commit")
    return commit


def hash_artifact(path: Path) -> str:
    """Hash a file or a directory tree deterministically."""
    if not path.exists():
        raise ShootoutError(f"adapter artifact does not exist: {path}")
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ShootoutError(f"adapter artifact directory is empty: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with candidate.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Mapping[str, Any]) -> str:
    """Hash JSON configuration with stable ordering and separators."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _format_command(command: Sequence[str], values: Mapping[str, Any]) -> list[str]:
    formatted: list[str] = []
    for part in command:
        match = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", part)
        if match is None:
            formatted.append(part)
            continue
        name = match.group(1)
        if name not in values:
            raise ShootoutError(f"unknown command placeholder: {name}")
        formatted.append(str(values[name]))
    return formatted


def validate_adapter_result(
    raw: Mapping[str, Any],
    expected_hardware: Mapping[str, Any],
    expected_config_sha256: str,
    expected_version: str,
    source: Path,
) -> dict[str, Any]:
    """Validate the minimal result emitted by a framework adapter."""
    if raw.get("status") != "completed" or raw.get("measured") is not True:
        raise ShootoutError(f"{source}: adapter did not report measured completion")
    artifact_path = raw.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise ShootoutError(f"{source}: artifact_path must be non-empty")
    if raw.get("config_sha256") != expected_config_sha256:
        raise ShootoutError(f"{source}: adapter config digest does not match manifest")
    if raw.get("framework_version") != expected_version:
        raise ShootoutError(
            f"{source}: installed framework version does not match manifest"
        )
    hardware = raw.get("hardware")
    if not isinstance(hardware, Mapping):
        raise ShootoutError(f"{source}: hardware must be an object")
    for field in ("gpu", "gpu_count"):
        if hardware.get(field) != expected_hardware.get(field):
            raise ShootoutError(f"{source}: hardware.{field} does not match manifest")
    cuda = hardware.get("cuda")
    if not isinstance(cuda, str) or not cuda:
        raise ShootoutError(f"{source}: hardware.cuda must be non-empty")
    metrics = raw.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ShootoutError(f"{source}: metrics must be an object")
    for field in (
        "samples_processed",
        "peak_vram_mb",
        "eval_score_baseline",
        "eval_score_final",
    ):
        value = metrics.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ShootoutError(f"{source}: metrics.{field} must be numeric")
    if float(metrics["samples_processed"]) <= 0 or float(metrics["peak_vram_mb"]) <= 0:
        raise ShootoutError(f"{source}: samples_processed and peak_vram_mb must be > 0")
    return dict(raw)


def run_implementation(
    manifest: Mapping[str, Any],
    implementation: Mapping[str, Any],
    seed: int,
    output_dir: Path,
    root: Path,
    timeout_seconds: int,
) -> Path:
    """Run one seed, retain logs/failure evidence, and emit validated evidence."""
    slug = str(implementation["name"]).lower().replace(" ", "-")
    run_dir = output_dir / "runs" / f"{slug}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter_output = run_dir / "adapter-result.json"
    artifact_dir = run_dir / "artifact"
    artifact_dir.mkdir()
    values = {
        "seed": seed,
        "adapter_output": str(adapter_output.resolve()),
        "phase0_output": str((run_dir / "phase0-result.json").resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "dataset_revision": manifest["dataset_revision"],
        "task": manifest["task"],
        "config_json": json.dumps(
            manifest["config"], sort_keys=True, separators=(",", ":")
        ),
    }
    command = _format_command(implementation["command"], values)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env=os.environ.copy(),
        )
        elapsed = time.monotonic() - started
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        (run_dir / "failure.json").write_text(
            json.dumps(
                {"kind": "timeout", "elapsed_seconds": elapsed, "command": command},
                indent=2,
            ),
            encoding="utf-8",
        )
        raise ShootoutError(f"{implementation['name']} seed {seed} timed out") from exc
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        (run_dir / "failure.json").write_text(
            json.dumps(
                {
                    "kind": "exit",
                    "returncode": completed.returncode,
                    "command": command,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise ShootoutError(
            f"{implementation['name']} seed {seed} exited {completed.returncode}"
        )
    if not adapter_output.exists():
        raise ShootoutError(
            f"{implementation['name']} seed {seed} wrote no adapter result"
        )
    try:
        adapter_raw = json.loads(adapter_output.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ShootoutError(f"{adapter_output}: invalid JSON") from exc
    if not isinstance(adapter_raw, Mapping):
        raise ShootoutError(f"{adapter_output}: result must be an object")
    adapter = validate_adapter_result(
        adapter_raw,
        manifest["hardware"],
        canonical_digest(manifest["config"]),
        str(implementation["version"]),
        adapter_output,
    )
    artifact_path = Path(adapter["artifact_path"])
    if not artifact_path.is_absolute():
        artifact_path = root / artifact_path
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_relative_to(artifact_dir.resolve()):
        raise ShootoutError("adapter artifact_path must stay inside {artifact_dir}")
    samples = float(adapter["metrics"]["samples_processed"])
    evidence = {
        "schema_version": 1,
        "measured": True,
        "framework": implementation["name"],
        "framework_version": implementation["version"],
        "harness_commit": git_commit(root),
        **{
            field: manifest[field]
            for field in MANIFEST_FIELDS
            if field != "cache_policy"
        },
        "cache_policy": manifest["cache_policy"],
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(command),
        "config": dict(manifest["config"]),
        "hardware": dict(adapter["hardware"]),
        "metrics": {
            "samples_per_second": samples / elapsed,
            "wall_clock_seconds": elapsed,
            "peak_vram_mb": float(adapter["metrics"]["peak_vram_mb"]),
            "eval_score_baseline": float(adapter["metrics"]["eval_score_baseline"]),
            "eval_score_final": float(adapter["metrics"]["eval_score_final"]),
        },
        "artifact_sha256": hash_artifact(artifact_path),
    }
    destination = output_dir / f"{slug}-seed{seed}.json"
    validate_document(evidence, destination)
    destination.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return destination


def execution_order(
    implementations: Sequence[Mapping[str, Any]], seed_index: int
) -> list[Mapping[str, Any]]:
    """Rotate implementation order by seed to reduce systematic order bias."""
    offset = seed_index % len(implementations)
    return [*implementations[offset:], *implementations[:offset]]


def validate_required_frameworks(
    manifest: Mapping[str, Any], required_frameworks: Sequence[str]
) -> None:
    """Reject a manifest that omits any explicitly required framework."""
    required = {name.strip() for name in required_frameworks if name.strip()}
    configured = {
        str(implementation["name"]) for implementation in manifest["implementations"]
    }
    missing = sorted(required - configured)
    if missing:
        raise ShootoutError(
            "manifest is missing required frameworks: " + ", ".join(missing)
        )


def write_run_summary(
    output_dir: Path,
    *,
    mode: str,
    manifest: Path,
    attempts: Sequence[Mapping[str, Any]],
) -> None:
    """Write an accounting record for every attempted framework/seed pair."""
    succeeded = sum(attempt["status"] == "completed" for attempt in attempts)
    payload = {
        "schema_version": 1,
        "kind": "framework-shootout-accounting",
        "mode": mode,
        "manifest": str(manifest),
        "attempted": len(attempts),
        "completed": succeeded,
        "failed": len(attempts) - succeeded,
        "attempts": list(attempts),
    }
    accounting_dir = output_dir / "_accounting"
    accounting_dir.mkdir(parents=True, exist_ok=True)
    destination = accounting_dir / "shootout-summary.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a measured framework shootout manifest"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="run the first seed for every framework; never publication-complete",
    )
    parser.add_argument(
        "--required-framework",
        action="append",
        default=[],
        help="fail before execution when the manifest omits this framework",
    )
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        validate_required_frameworks(manifest, args.required_framework)
        if args.timeout_seconds < 1:
            raise ShootoutError("timeout-seconds must be >= 1")
        if args.dry_run and args.preflight:
            raise ShootoutError("--dry-run and --preflight are mutually exclusive")
        if args.dry_run:
            for index, seed in enumerate(manifest["seeds"]):
                for implementation in execution_order(
                    manifest["implementations"], index
                ):
                    print(f"seed={seed} framework={implementation['name']}")
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        attempts: list[dict[str, Any]] = []
        seeds = manifest["seeds"][:1] if args.preflight else manifest["seeds"]
        for index, seed in enumerate(seeds):
            for implementation in execution_order(manifest["implementations"], index):
                framework = str(implementation["name"])
                try:
                    evidence = run_implementation(
                        manifest,
                        implementation,
                        seed,
                        args.output_dir,
                        args.root,
                        args.timeout_seconds,
                    )
                except (ShootoutError, EvidenceError) as exc:
                    attempts.append(
                        {
                            "framework": framework,
                            "seed": seed,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    continue
                attempts.append(
                    {
                        "framework": framework,
                        "seed": seed,
                        "status": "completed",
                        "evidence": str(evidence),
                    }
                )
        write_run_summary(
            args.output_dir,
            mode="preflight" if args.preflight else "measured",
            manifest=args.manifest,
            attempts=attempts,
        )
    except (ShootoutError, EvidenceError) as exc:
        print(f"shootout rejected: {exc}", file=sys.stderr)
        return 2
    failed = sum(attempt["status"] == "failed" for attempt in attempts)
    completed = len(attempts) - failed
    if failed:
        print(
            f"shootout completed {completed}/{len(attempts)} attempts; "
            f"{failed} failed (see _accounting/shootout-summary.json)",
            file=sys.stderr,
        )
        return 2
    if args.preflight:
        print(f"preflight completed {completed} framework runs")
        return 0
    print(f"wrote {completed} measured evidence documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
