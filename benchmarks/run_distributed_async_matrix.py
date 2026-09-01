#!/usr/bin/env python3
"""Execute and retain the publication-grade distributed async scenario matrix.

The provider driver is an argv-only adapter.  It owns provisioning and fault
injection, while this runner owns immutable experiment metadata, external wall
time, artifact hashing, failure accounting, and the final publication gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .distributed_async_evidence import (
        REQUIRED_SCENARIOS,
        DistributedAsyncEvidenceError,
        load_runs,
        summarize,
        validate_matrix,
        validate_run,
    )
except ImportError:  # pragma: no cover - direct script execution
    from distributed_async_evidence import (
        REQUIRED_SCENARIOS,
        DistributedAsyncEvidenceError,
        load_runs,
        summarize,
        validate_matrix,
        validate_run,
    )


class DistributedAsyncRunnerError(ValueError):
    """Raised when measured distributed collection is not defensible."""


HEX_DIGITS = frozenset("0123456789abcdef")
REQUIRED_PLACEHOLDERS = {
    "{scenario}",
    "{seed}",
    "{mode}",
    "{protocol}",
    "{framework_version}",
    "{config_json}",
    "{config_sha256}",
    "{adapter_output}",
    "{artifact_dir}",
}
MATCHED_TOPOLOGY_FIELDS = (
    "node_count",
    "worker_count",
    "accelerator",
    "accelerator_driver",
    "interconnect",
)
SENSITIVE_CONFIG_KEYS = frozenset(
    {"api_key", "authorization", "credential", "password", "secret", "token"}
)


def canonical_digest(value: Mapping[str, Any]) -> str:
    """Return the SHA-256 of canonical JSON configuration."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def hash_artifact(path: Path) -> str:
    """Hash one retained file or directory tree deterministically."""
    if not path.exists():
        raise DistributedAsyncRunnerError(f"artifact does not exist: {path}")
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise DistributedAsyncRunnerError(f"artifact directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _text(data: Mapping[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DistributedAsyncRunnerError(f"{label}.{key} must be non-empty")
    return value


def _positive_number(data: Mapping[str, Any], key: str, label: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DistributedAsyncRunnerError(f"{label}.{key} must be numeric")
    number = float(value)
    if not 0 < number < float("inf"):
        raise DistributedAsyncRunnerError(f"{label}.{key} must be finite and positive")
    return number


def _reject_secrets(value: Any, path: str = "manifest.config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            components = set(normalized.split("_"))
            if (
                normalized in SENSITIVE_CONFIG_KEYS
                or normalized.endswith("_api_key")
                or components.intersection(SENSITIVE_CONFIG_KEYS)
            ):
                raise DistributedAsyncRunnerError(
                    f"{path}.{key}: secrets must be supplied through the environment"
                )
            _reject_secrets(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_secrets(child, f"{path}[{index}]")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the matched topology, scenarios, and shell-free driver contract."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DistributedAsyncRunnerError(f"{path}: invalid JSON") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise DistributedAsyncRunnerError(
            "manifest must be an object with schema_version=1"
        )
    if raw.get("kind") != "stateset-distributed-async-manifest":
        raise DistributedAsyncRunnerError("manifest.kind is invalid")
    for key in ("protocol", "framework_version", "provider", "cost_source"):
        _text(raw, key, "manifest")
    config = raw.get("config")
    if not isinstance(config, Mapping) or not config:
        raise DistributedAsyncRunnerError("manifest.config must be a non-empty object")
    _reject_secrets(config)
    max_lag = config.get("max_policy_lag")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 0:
        raise DistributedAsyncRunnerError(
            "manifest.config.max_policy_lag must be a non-negative integer"
        )
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
        raise DistributedAsyncRunnerError(
            "manifest.seeds must contain at least three unique non-negative integers"
        )
    scenarios = raw.get("scenarios")
    if (
        not isinstance(scenarios, list)
        or set(scenarios) != set(REQUIRED_SCENARIOS)
        or len(scenarios) != len(set(scenarios))
    ):
        raise DistributedAsyncRunnerError(
            f"manifest.scenarios must be exactly {sorted(REQUIRED_SCENARIOS)}"
        )
    durations = raw.get("minimum_duration_seconds")
    if not isinstance(durations, Mapping) or set(durations) != set(REQUIRED_SCENARIOS):
        raise DistributedAsyncRunnerError(
            "manifest.minimum_duration_seconds must cover every scenario"
        )
    for scenario in REQUIRED_SCENARIOS:
        _positive_number(durations, scenario, "manifest.minimum_duration_seconds")
    if float(durations["steady_state_soak"]) < 43_200:
        raise DistributedAsyncRunnerError(
            "steady_state_soak minimum must be at least 43200 seconds"
        )
    topology = raw.get("topology")
    if not isinstance(topology, Mapping):
        raise DistributedAsyncRunnerError("manifest.topology must be an object")
    for key in ("node_count", "worker_count"):
        value = topology.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise DistributedAsyncRunnerError(f"manifest.topology.{key} must be >= 2")
    if int(topology["worker_count"]) < int(topology["node_count"]):
        raise DistributedAsyncRunnerError("worker_count must cover every node")
    for key in ("accelerator", "accelerator_driver", "interconnect"):
        _text(topology, key, "manifest.topology")
    command = raw.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise DistributedAsyncRunnerError("manifest.command must be a string list")
    missing = REQUIRED_PLACEHOLDERS.difference(command)
    if missing:
        raise DistributedAsyncRunnerError(
            "manifest.command is missing placeholders: " + ", ".join(sorted(missing))
        )
    return dict(raw)


def git_commit(root: Path) -> str:
    """Resolve the harness revision and reject a dirty collection tree."""
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    if status.returncode != 0 or status.stdout.strip():
        raise DistributedAsyncRunnerError("benchmark harness worktree must be clean")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    commit = result.stdout.strip()
    if (
        result.returncode != 0
        or len(commit) != 40
        or any(c not in HEX_DIGITS for c in commit)
    ):
        raise DistributedAsyncRunnerError("could not resolve a full harness commit")
    return commit


def _format_command(command: Sequence[str], values: Mapping[str, Any]) -> list[str]:
    formatted: list[str] = []
    for part in command:
        match = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", part)
        if match is None:
            formatted.append(part)
            continue
        name = match.group(1)
        if name not in values:
            raise DistributedAsyncRunnerError(f"unknown command placeholder: {name}")
        formatted.append(str(values[name]))
    return formatted


def _validate_adapter_result(
    raw: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    scenario: str,
    seed: int,
    source: Path,
) -> dict[str, Any]:
    if raw.get("status") != "completed" or raw.get("measured") is not True:
        raise DistributedAsyncRunnerError(f"{source}: measured completion is required")
    expected = {
        "scenario": scenario,
        "seed": seed,
        "protocol": manifest["protocol"],
        "framework_version": manifest["framework_version"],
        "provider": manifest["provider"],
        "config_sha256": canonical_digest(manifest["config"]),
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise DistributedAsyncRunnerError(
                f"{source}: {key} does not match manifest"
            )
    topology = raw.get("topology")
    if not isinstance(topology, Mapping):
        raise DistributedAsyncRunnerError(f"{source}: topology must be an object")
    for key in MATCHED_TOPOLOGY_FIELDS:
        if topology.get(key) != manifest["topology"][key]:
            raise DistributedAsyncRunnerError(
                f"{source}: topology.{key} does not match manifest"
            )
    for key in ("metrics", "cost", "recovery"):
        if not isinstance(raw.get(key), Mapping):
            raise DistributedAsyncRunnerError(f"{source}: {key} must be an object")
    if raw["cost"].get("source") != manifest["cost_source"]:
        raise DistributedAsyncRunnerError(
            f"{source}: cost.source does not match manifest"
        )
    _positive_number(raw, "duration_seconds", str(source))
    artifact_path = raw.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise DistributedAsyncRunnerError(f"{source}: artifact_path must be non-empty")
    return dict(raw)


def run_scenario(
    manifest: Mapping[str, Any],
    scenario: str,
    seed: int,
    *,
    output_dir: Path,
    root: Path,
    harness_commit: str,
    timeout_seconds: int,
    preflight: bool = False,
) -> Path:
    """Run one provider scenario and bind its result to retained bytes."""
    run_dir = output_dir / "runs" / f"{scenario}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir = run_dir / "artifact"
    artifact_dir.mkdir()
    adapter_output = run_dir / "adapter-result.json"
    config_json = json.dumps(manifest["config"], sort_keys=True, separators=(",", ":"))
    values = {
        "scenario": scenario,
        "seed": seed,
        "mode": "preflight" if preflight else "measured",
        "protocol": manifest["protocol"],
        "framework_version": manifest["framework_version"],
        "config_json": config_json,
        "config_sha256": canonical_digest(manifest["config"]),
        "adapter_output": str(adapter_output.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
    }
    command = _format_command(manifest["command"], values)
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
        elapsed = max(time.monotonic() - started, 1e-9)
    except subprocess.TimeoutExpired as exc:
        elapsed = max(time.monotonic() - started, 1e-9)
        (run_dir / "failure.json").write_text(
            json.dumps({"kind": "timeout", "elapsed_seconds": elapsed}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise DistributedAsyncRunnerError(f"{scenario} seed {seed} timed out") from exc
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        (run_dir / "failure.json").write_text(
            json.dumps({"kind": "exit", "returncode": completed.returncode}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise DistributedAsyncRunnerError(
            f"{scenario} seed {seed} exited {completed.returncode}"
        )
    try:
        raw = json.loads(adapter_output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DistributedAsyncRunnerError(
            f"{scenario} seed {seed} wrote no valid adapter result"
        ) from exc
    if not isinstance(raw, Mapping):
        raise DistributedAsyncRunnerError(f"{adapter_output}: result must be an object")
    result = _validate_adapter_result(
        raw, manifest=manifest, scenario=scenario, seed=seed, source=adapter_output
    )
    duration = float(result["duration_seconds"])
    if not preflight:
        minimum = float(manifest["minimum_duration_seconds"][scenario])
        if duration < minimum:
            raise DistributedAsyncRunnerError(
                f"{adapter_output}: duration is below the manifest minimum"
            )
        if duration > elapsed * 1.02:
            raise DistributedAsyncRunnerError(
                f"{adapter_output}: reported duration exceeds external wall time"
            )
    artifact_path = Path(str(result["artifact_path"]))
    if not artifact_path.is_absolute():
        artifact_path = root / artifact_path
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_relative_to(artifact_dir.resolve()):
        raise DistributedAsyncRunnerError(
            f"{adapter_output}: artifact_path must stay inside artifact_dir"
        )
    evidence = {
        "schema_version": 1,
        "kind": (
            "stateset-distributed-async-preflight"
            if preflight
            else "stateset-distributed-async-evidence"
        ),
        "status": "preflight" if preflight else "completed",
        "measured": not preflight,
        "run_id": f"{scenario}-{seed}-{harness_commit[:12]}",
        "framework_version": manifest["framework_version"],
        "harness_commit": harness_commit,
        "protocol": manifest["protocol"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": manifest["provider"],
        "seed": seed,
        "scenario": scenario,
        "duration_seconds": duration,
        "external_wall_seconds": elapsed,
        "config": dict(manifest["config"]),
        "config_sha256": canonical_digest(manifest["config"]),
        "topology": dict(result["topology"]),
        "metrics": dict(result["metrics"]),
        "cost": dict(result["cost"]),
        "recovery": dict(result["recovery"]),
        "artifact_sha256": hash_artifact(artifact_path),
    }
    category = "preflight" if preflight else "evidence"
    destination = output_dir / category / f"{scenario}-seed{seed}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not preflight:
        validate_run(evidence, destination)
    destination.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return destination


def _rotated_scenarios(scenarios: Sequence[str], seed_index: int) -> list[str]:
    offset = seed_index % len(scenarios)
    return list(scenarios[offset:]) + list(scenarios[:offset])


def _write_accounting(
    output_dir: Path,
    *,
    mode: str,
    manifest_path: Path,
    attempts: Sequence[Mapping[str, Any]],
) -> None:
    completed = sum(item["status"] == "completed" for item in attempts)
    payload = {
        "schema_version": 1,
        "kind": "stateset-distributed-async-accounting",
        "mode": mode,
        "manifest": str(manifest_path),
        "attempted": len(attempts),
        "completed": completed,
        "failed": len(attempts) - completed,
        "attempts": list(attempts),
    }
    destination = output_dir / "_accounting" / "matrix-summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=int, default=50_400)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every scenario/seed, retain failures, and invoke the strict gate."""
    args = parse_args(argv)
    attempts: list[dict[str, Any]] = []
    try:
        manifest = load_manifest(args.manifest)
        if args.timeout_seconds < 1:
            raise DistributedAsyncRunnerError("timeout-seconds must be >= 1")
        if args.dry_run and args.preflight:
            raise DistributedAsyncRunnerError("--dry-run and --preflight are exclusive")
        seeds = manifest["seeds"][:1] if args.preflight else manifest["seeds"]
        ordered: list[tuple[int, str]] = []
        for index, seed in enumerate(seeds):
            ordered.extend(
                (seed, scenario)
                for scenario in _rotated_scenarios(manifest["scenarios"], index)
            )
        if args.dry_run:
            for seed, scenario in ordered:
                print(f"seed={seed} scenario={scenario}")
            return 0
        commit = git_commit(args.root)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for seed, scenario in ordered:
            try:
                evidence = run_scenario(
                    manifest,
                    scenario,
                    seed,
                    output_dir=args.output_dir,
                    root=args.root,
                    harness_commit=commit,
                    timeout_seconds=args.timeout_seconds,
                    preflight=args.preflight,
                )
            except (DistributedAsyncRunnerError, DistributedAsyncEvidenceError) as exc:
                attempts.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                continue
            attempts.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "status": "completed",
                    "evidence": str(evidence),
                }
            )
        _write_accounting(
            args.output_dir,
            mode="preflight" if args.preflight else "measured",
            manifest_path=args.manifest,
            attempts=attempts,
        )
        failures = sum(item["status"] == "failed" for item in attempts)
        if failures:
            raise DistributedAsyncRunnerError(
                f"matrix completed {len(attempts) - failures}/{len(attempts)} attempts"
            )
        if not args.preflight:
            runs = load_runs([args.output_dir / "evidence"])
            validate_matrix(runs, min_seeds=len(manifest["seeds"]))
            (args.output_dir / "report.json").write_text(
                json.dumps(summarize(runs), indent=2) + "\n", encoding="utf-8"
            )
    except (DistributedAsyncRunnerError, DistributedAsyncEvidenceError, OSError) as exc:
        print(f"distributed async matrix rejected: {exc}", file=sys.stderr)
        return 2
    print(
        f"completed {len(attempts)} distributed async attempts"
        + (" (preflight only)" if args.preflight else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
