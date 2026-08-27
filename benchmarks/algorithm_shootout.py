#!/usr/bin/env python3
"""Execute a matched StateSet algorithm benchmark and emit strict evidence.

Each algorithm adapter runs as a separate process.  This orchestrator rotates
execution order by seed, measures wall time outside the adapter, verifies both
the shared protocol and algorithm-specific configuration attestations, hashes
the normalized policy artifact, and emits documents accepted by
``algorithm_comparison.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from framework_comparison import EvidenceError, validate_document
from shootout import (
    ShootoutError,
    _format_command,
    canonical_digest,
    git_commit,
    hash_artifact,
)

COMMON_MANIFEST_FIELDS = (
    "protocol",
    "model",
    "model_revision",
    "task",
    "dataset_revision",
    "cache_policy",
)
REQUIRED_COMMAND_PLACEHOLDERS = {
    "{algorithm}",
    "{seed}",
    "{adapter_output}",
    "{artifact_dir}",
    "{phase0_output}",
    "{model}",
    "{model_revision}",
    "{dataset_revision}",
    "{task}",
    "{config_json}",
    "{num_train_examples}",
    "{num_eval_examples}",
}


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ShootoutError(f"{field} must be a non-empty string")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a strict algorithm-shootout manifest."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ShootoutError(f"{path}: invalid JSON") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ShootoutError("manifest must be an object with schema_version=1")
    for field in COMMON_MANIFEST_FIELDS:
        _nonempty_string(raw.get(field), f"manifest.{field}")
    for field in ("model_revision", "dataset_revision"):
        if len(str(raw[field])) != 40:
            raise ShootoutError(f"manifest.{field} must be a full commit SHA")

    framework = raw.get("framework")
    if not isinstance(framework, Mapping):
        raise ShootoutError("manifest.framework must be an object")
    for field in ("name", "version"):
        _nonempty_string(framework.get(field), f"manifest.framework.{field}")

    config = raw.get("config")
    if not isinstance(config, Mapping) or not config:
        raise ShootoutError("manifest.config must be a non-empty object")
    for field in (
        "num_train_examples",
        "num_eval_examples",
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_generations",
        "num_iterations",
    ):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ShootoutError(f"manifest.config.{field} must be an integer >= 1")
    # Several native objectives expose gradient accumulation in their config
    # for compatibility but do not all implement the same accumulation
    # semantics.  Refuse an allegedly matched matrix until that is true.
    if config["gradient_accumulation_steps"] != 1:
        raise ShootoutError(
            "algorithm shootouts currently require gradient_accumulation_steps=1"
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
        raise ShootoutError(
            "manifest.seeds must contain at least three unique non-negative integers"
        )

    hardware = raw.get("hardware")
    if not isinstance(hardware, Mapping):
        raise ShootoutError("manifest.hardware must be an object")
    _nonempty_string(hardware.get("gpu"), "manifest.hardware.gpu")
    _nonempty_string(hardware.get("cuda"), "manifest.hardware.cuda")
    gpu_count = hardware.get("gpu_count")
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
        raise ShootoutError("manifest.hardware.gpu_count must be >= 1")

    algorithms = raw.get("algorithms")
    if not isinstance(algorithms, list) or len(algorithms) < 2:
        raise ShootoutError("manifest requires at least two algorithms")
    names: list[str] = []
    for algorithm in algorithms:
        if not isinstance(algorithm, Mapping):
            raise ShootoutError("each algorithm must be an object")
        name = _nonempty_string(algorithm.get("name"), "algorithm.name")
        _nonempty_string(algorithm.get("revision"), f"algorithm[{name}].revision")
        algorithm_config = algorithm.get("config")
        if not isinstance(algorithm_config, Mapping) or not algorithm_config:
            raise ShootoutError(f"algorithm[{name}].config must be a non-empty object")
        if not isinstance(algorithm_config.get("objective"), str):
            raise ShootoutError(f"algorithm[{name}].config.objective must be a string")
        command = algorithm.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise ShootoutError(
                f"algorithm[{name}].command must be a non-empty string list"
            )
        missing = REQUIRED_COMMAND_PLACEHOLDERS.difference(command)
        if missing:
            raise ShootoutError(
                f"algorithm[{name}].command is missing placeholders: "
                + ", ".join(sorted(missing))
            )
        names.append(name)
    if len(names) != len(set(names)):
        raise ShootoutError("algorithm names must be unique")
    return dict(raw)


def execution_order(
    algorithms: Sequence[Mapping[str, Any]], seed_index: int
) -> list[Mapping[str, Any]]:
    """Rotate algorithm order each seed to reduce thermal/cache order bias."""
    offset = seed_index % len(algorithms)
    return [*algorithms[offset:], *algorithms[:offset]]


def validate_adapter_result(
    raw: Mapping[str, Any],
    manifest: Mapping[str, Any],
    algorithm: Mapping[str, Any],
    source: Path,
) -> dict[str, Any]:
    """Validate Phase-0's neutral result and both config attestations."""
    if raw.get("status") != "completed" or raw.get("measured") is not True:
        raise ShootoutError(f"{source}: adapter did not report measured completion")
    if raw.get("framework_version") != manifest["framework"]["version"]:
        raise ShootoutError(f"{source}: framework version does not match manifest")
    if raw.get("config_sha256") != canonical_digest(manifest["config"]):
        raise ShootoutError(f"{source}: shared config digest does not match manifest")
    algorithm_config = raw.get("algorithm_config")
    if algorithm_config != algorithm["config"]:
        raise ShootoutError(f"{source}: algorithm config does not match manifest")
    if raw.get("algorithm_config_sha256") != canonical_digest(algorithm["config"]):
        raise ShootoutError(
            f"{source}: algorithm config digest does not match manifest"
        )

    artifact_path = raw.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise ShootoutError(f"{source}: artifact_path must be non-empty")
    hardware = raw.get("hardware")
    if not isinstance(hardware, Mapping):
        raise ShootoutError(f"{source}: hardware must be an object")
    for field in ("gpu", "gpu_count", "cuda"):
        if hardware.get(field) != manifest["hardware"].get(field):
            raise ShootoutError(f"{source}: hardware.{field} does not match manifest")
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
    if float(metrics["samples_processed"]) <= 0:
        raise ShootoutError(f"{source}: samples_processed must be > 0")
    if float(metrics["peak_vram_mb"]) <= 0:
        raise ShootoutError(f"{source}: peak_vram_mb must be > 0")
    return dict(raw)


def run_algorithm(
    manifest: Mapping[str, Any],
    algorithm: Mapping[str, Any],
    seed: int,
    output_dir: Path,
    root: Path,
    timeout_seconds: int,
) -> Path:
    """Run one algorithm/seed and emit one validated evidence document."""
    slug = str(algorithm["name"]).lower().replace(" ", "-")
    run_dir = output_dir / "runs" / f"{slug}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    adapter_output = run_dir / "adapter-result.json"
    artifact_dir = run_dir / "artifact"
    artifact_dir.mkdir()
    values = {
        "algorithm": algorithm["name"],
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
        "num_train_examples": manifest["config"]["num_train_examples"],
        "num_eval_examples": manifest["config"]["num_eval_examples"],
    }
    command = _format_command(algorithm["command"], values)
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
            )
            + "\n",
            encoding="utf-8",
        )
        raise ShootoutError(f"{algorithm['name']} seed {seed} timed out") from exc

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
            )
            + "\n",
            encoding="utf-8",
        )
        raise ShootoutError(
            f"{algorithm['name']} seed {seed} exited {completed.returncode}"
        )
    if not adapter_output.is_file():
        raise ShootoutError(f"{algorithm['name']} seed {seed} wrote no adapter result")
    try:
        raw = json.loads(adapter_output.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ShootoutError(f"{adapter_output}: invalid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ShootoutError(f"{adapter_output}: result must be an object")
    adapter = validate_adapter_result(raw, manifest, algorithm, adapter_output)

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
        "framework": manifest["framework"]["name"],
        "framework_version": manifest["framework"]["version"],
        "harness_commit": git_commit(root),
        "protocol": manifest["protocol"],
        "cache_policy": manifest["cache_policy"],
        "algorithm": algorithm["name"],
        "algorithm_revision": algorithm["revision"],
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "task": manifest["task"],
        "dataset_revision": manifest["dataset_revision"],
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(command),
        "config": {
            "shared": dict(manifest["config"]),
            "algorithm": dict(algorithm["config"]),
        },
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument(
        "--required-algorithm",
        action="append",
        default=[],
        help="Algorithm required in the manifest (repeatable).",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run only the first seed across every algorithm as non-publishable diagnostics.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.timeout_seconds < 1:
            raise ShootoutError("timeout-seconds must be >= 1")
        names = {str(algorithm["name"]) for algorithm in manifest["algorithms"]}
        missing = sorted(set(args.required_algorithm) - names)
        if missing:
            raise ShootoutError(
                "manifest is missing required algorithms: " + ", ".join(missing)
            )
        seeds = manifest["seeds"][:1] if args.preflight else manifest["seeds"]
        if args.dry_run:
            for index, seed in enumerate(seeds):
                for algorithm in execution_order(manifest["algorithms"], index):
                    print(f"seed={seed} algorithm={algorithm['name']}")
            return 0
        args.output_dir.mkdir(parents=True, exist_ok=True)
        produced: list[Path] = []
        failures: list[str] = []
        for index, seed in enumerate(seeds):
            for algorithm in execution_order(manifest["algorithms"], index):
                try:
                    produced.append(
                        run_algorithm(
                            manifest,
                            algorithm,
                            seed,
                            args.output_dir,
                            args.root,
                            args.timeout_seconds,
                        )
                    )
                except (ShootoutError, EvidenceError) as exc:
                    failure = f"{algorithm['name']} seed {seed}: {exc}"
                    failures.append(failure)
                    print(f"algorithm run failed: {failure}", file=sys.stderr)
        if failures:
            raise ShootoutError(
                f"{len(failures)} of {len(produced) + len(failures)} runs failed; "
                "all attempted runs and failure logs were retained"
            )
    except (ShootoutError, EvidenceError) as exc:
        print(f"algorithm shootout rejected: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {len(produced)} measured algorithm evidence documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
