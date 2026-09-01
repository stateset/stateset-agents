#!/usr/bin/env python3
"""Execute paired standard-agent evaluations and emit publishable evidence."""

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
    from .adapters.official_suite_pipeline import (
        OfficialPipelineError,
        load_pipeline_config,
    )
    from .agent_quality_evidence import (
        REQUIRED_SUITES,
        AgentQualityEvidenceError,
        load_runs,
        summarize,
        validate_matrix,
        validate_run,
    )
except ImportError:  # pragma: no cover - direct script execution
    from adapters.official_suite_pipeline import (
        OfficialPipelineError,
        load_pipeline_config,
    )
    from agent_quality_evidence import (
        REQUIRED_SUITES,
        AgentQualityEvidenceError,
        load_runs,
        summarize,
        validate_matrix,
        validate_run,
    )


class AgentQualityRunnerError(ValueError):
    """Raised when collection cannot produce defensible paired evidence."""


HEX_DIGITS = frozenset("0123456789abcdef")
REQUIRED_PLACEHOLDERS = {
    "{seed}",
    "{suite}",
    "{suite_revision}",
    "{split}",
    "{framework_version}",
    "{baseline_model}",
    "{baseline_revision}",
    "{trained_model}",
    "{trained_revision}",
    "{evaluation_config_json}",
    "{adapter_output}",
    "{artifact_dir}",
}


def _text(data: Mapping[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgentQualityRunnerError(f"{label}.{key} must be non-empty")
    return value


def _digest(data: Mapping[str, Any], key: str, label: str, length: int) -> str:
    value = _text(data, key, label)
    if len(value) != length or any(char not in HEX_DIGITS for char in value):
        raise AgentQualityRunnerError(
            f"{label}.{key} must be {length} lowercase hex characters"
        )
    return value


def canonical_digest(value: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 for JSON configuration."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def hash_artifact(path: Path) -> str:
    """Hash a retained result file or directory tree deterministically."""
    if not path.exists():
        raise AgentQualityRunnerError(f"artifact does not exist: {path}")
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise AgentQualityRunnerError(f"artifact directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the immutable policy-pair and upstream-suite execution contract."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentQualityRunnerError(f"{path}: invalid JSON") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise AgentQualityRunnerError(
            "manifest must be an object with schema_version=1"
        )
    if raw.get("kind") != "stateset-agent-quality-manifest":
        raise AgentQualityRunnerError("manifest.kind is invalid")
    for key in ("protocol", "framework_version"):
        _text(raw, key, "manifest")
    for policy_name in ("baseline_policy", "trained_policy"):
        policy = raw.get(policy_name)
        if not isinstance(policy, Mapping):
            raise AgentQualityRunnerError(f"manifest.{policy_name} must be an object")
        _text(policy, "model", f"manifest.{policy_name}")
        _digest(policy, "revision", f"manifest.{policy_name}", 40)
    trained = raw["trained_policy"]
    _digest(trained, "artifact_sha256", "manifest.trained_policy", 64)

    config = raw.get("evaluation_config")
    if not isinstance(config, Mapping) or not config:
        raise AgentQualityRunnerError(
            "manifest.evaluation_config must be a non-empty object"
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
        raise AgentQualityRunnerError(
            "manifest.seeds must contain at least three unique non-negative integers"
        )

    suites = raw.get("suites")
    if not isinstance(suites, list) or len(suites) != len(REQUIRED_SUITES):
        raise AgentQualityRunnerError("manifest must configure every required suite")
    names: list[str] = []
    for index, suite in enumerate(suites):
        label = f"manifest.suites[{index}]"
        if not isinstance(suite, Mapping):
            raise AgentQualityRunnerError(f"{label} must be an object")
        names.append(_text(suite, "name", label))
        _digest(suite, "revision", label, 40)
        _text(suite, "split", label)
        command = suite.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise AgentQualityRunnerError(f"{label}.command must be a string list")
        missing = REQUIRED_PLACEHOLDERS.difference(command)
        if missing:
            raise AgentQualityRunnerError(
                f"{label}.command is missing placeholders: "
                + ", ".join(sorted(missing))
            )
    if set(names) != set(REQUIRED_SUITES) or len(names) != len(set(names)):
        raise AgentQualityRunnerError(
            f"suite roster must be exactly {sorted(REQUIRED_SUITES)}"
        )
    if "official_suite_pipelines" in config:
        contract_dir = Path("/tmp/stateset-agent-quality-contract")
        try:
            for suite in REQUIRED_SUITES:
                load_pipeline_config(config, suite, contract_dir / suite)
        except OfficialPipelineError as exc:
            raise AgentQualityRunnerError(
                f"manifest.evaluation_config pipeline is invalid: {exc}"
            ) from exc
    return dict(raw)


def git_commit(root: Path) -> str:
    """Resolve the harness commit and reject collection from a dirty tree."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise AgentQualityRunnerError("benchmark harness worktree must be clean")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) != 40:
        raise AgentQualityRunnerError("could not resolve a full harness commit")
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
            raise AgentQualityRunnerError(f"unknown command placeholder: {name}")
        formatted.append(str(values[name]))
    return formatted


def _adapter_number(data: Mapping[str, Any], key: str, source: Path) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentQualityRunnerError(f"{source}: {key} must be numeric")
    result = float(value)
    if not result == result or result in (float("inf"), float("-inf")):
        raise AgentQualityRunnerError(f"{source}: {key} must be finite")
    return result


def validate_adapter_result(
    raw: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    suite: Mapping[str, Any],
    seed: int,
    source: Path,
) -> dict[str, Any]:
    """Validate the neutral paired result written by an upstream harness."""
    if raw.get("status") != "completed" or raw.get("measured") is not True:
        raise AgentQualityRunnerError(f"{source}: measured completion is required")
    expected = {
        "suite": suite["name"],
        "suite_revision": suite["revision"],
        "split": suite["split"],
        "seed": seed,
        "framework_version": manifest["framework_version"],
        "baseline_model": manifest["baseline_policy"]["model"],
        "baseline_model_revision": manifest["baseline_policy"]["revision"],
        "trained_model": manifest["trained_policy"]["model"],
        "trained_model_revision": manifest["trained_policy"]["revision"],
        "evaluation_config_sha256": canonical_digest(manifest["evaluation_config"]),
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise AgentQualityRunnerError(f"{source}: {key} does not match manifest")
    paired_digest = raw.get("paired_task_ids_sha256")
    if (
        not isinstance(paired_digest, str)
        or len(paired_digest) != 64
        or any(char not in HEX_DIGITS for char in paired_digest)
    ):
        raise AgentQualityRunnerError(
            f"{source}: paired_task_ids_sha256 must be lowercase SHA-256"
        )
    tasks = raw.get("tasks")
    baseline_successes = raw.get("baseline_successful_episodes")
    trained_successes = raw.get("trained_successful_episodes")
    for key, value in (
        ("tasks", tasks),
        ("baseline_successful_episodes", baseline_successes),
        ("trained_successful_episodes", trained_successes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AgentQualityRunnerError(f"{source}: {key} must be non-negative")
    if tasks < 1 or baseline_successes > tasks or trained_successes > tasks:
        raise AgentQualityRunnerError(f"{source}: invalid task/success counts")
    for key in ("baseline_score", "trained_score"):
        score = _adapter_number(raw, key, source)
        if not 0.0 <= score <= 1.0:
            raise AgentQualityRunnerError(f"{source}: {key} must be within [0, 1]")
    cost = _adapter_number(raw, "evaluation_cost_usd", source)
    if cost < 0:
        raise AgentQualityRunnerError(f"{source}: evaluation cost cannot be negative")
    _text(raw, "cost_source", str(source))
    artifact_path = raw.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise AgentQualityRunnerError(f"{source}: artifact_path must be non-empty")
    return dict(raw)


def run_suite(
    manifest: Mapping[str, Any],
    suite: Mapping[str, Any],
    seed: int,
    *,
    output_dir: Path,
    root: Path,
    harness_commit: str,
    timeout_seconds: int,
) -> Path:
    """Execute one paired suite/seed evaluation and retain its raw output."""
    suite_name = str(suite["name"])
    run_dir = output_dir / "runs" / f"{suite_name}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir = run_dir / "artifact"
    artifact_dir.mkdir()
    adapter_output = run_dir / "adapter-result.json"
    config_json = json.dumps(
        manifest["evaluation_config"], sort_keys=True, separators=(",", ":")
    )
    values = {
        "seed": seed,
        "suite": suite_name,
        "suite_revision": suite["revision"],
        "split": suite["split"],
        "framework_version": manifest["framework_version"],
        "baseline_model": manifest["baseline_policy"]["model"],
        "baseline_revision": manifest["baseline_policy"]["revision"],
        "trained_model": manifest["trained_policy"]["model"],
        "trained_revision": manifest["trained_policy"]["revision"],
        "evaluation_config_json": config_json,
        "adapter_output": str(adapter_output.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
    }
    command = _format_command(suite["command"], values)
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
        # Some Windows timer implementations can return the same tick for a
        # fast adapter (and tests replace the subprocess with an in-process
        # stub). Evidence requires a strictly positive duration, so preserve
        # that invariant at the measurement boundary.
        elapsed = max(time.monotonic() - started, 1e-9)
    except subprocess.TimeoutExpired as exc:
        elapsed = max(time.monotonic() - started, 1e-9)
        (run_dir / "failure.json").write_text(
            json.dumps({"kind": "timeout", "elapsed_seconds": elapsed}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise AgentQualityRunnerError(f"{suite_name} seed {seed} timed out") from exc
    (run_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        (run_dir / "failure.json").write_text(
            json.dumps({"kind": "exit", "returncode": completed.returncode}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise AgentQualityRunnerError(
            f"{suite_name} seed {seed} exited {completed.returncode}"
        )
    try:
        raw = json.loads(adapter_output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentQualityRunnerError(
            f"{suite_name} seed {seed} wrote no valid adapter result"
        ) from exc
    if not isinstance(raw, Mapping):
        raise AgentQualityRunnerError(f"{adapter_output}: result must be an object")
    result = validate_adapter_result(
        raw, manifest=manifest, suite=suite, seed=seed, source=adapter_output
    )
    artifact_path = Path(result["artifact_path"])
    if not artifact_path.is_absolute():
        artifact_path = root / artifact_path
    artifact_path = artifact_path.resolve()
    if not artifact_path.is_relative_to(artifact_dir.resolve()):
        raise AgentQualityRunnerError(
            f"{adapter_output}: artifact_path must stay inside artifact_dir"
        )
    trained_successes = int(result["trained_successful_episodes"])
    cost = float(result["evaluation_cost_usd"])
    evidence = {
        "schema_version": 2,
        "kind": "stateset-agent-quality-evidence",
        "status": "completed",
        "measured": True,
        "run_id": f"{suite_name}-{seed}-{harness_commit[:12]}",
        "suite": suite_name,
        "suite_revision": suite["revision"],
        "protocol": manifest["protocol"],
        "framework_version": manifest["framework_version"],
        "baseline_model": manifest["baseline_policy"]["model"],
        "baseline_model_revision": manifest["baseline_policy"]["revision"],
        "trained_model": manifest["trained_policy"]["model"],
        "trained_model_revision": manifest["trained_policy"]["revision"],
        "training_artifact_sha256": manifest["trained_policy"]["artifact_sha256"],
        "harness_commit": harness_commit,
        "split": suite["split"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "evaluation_config": dict(manifest["evaluation_config"]),
        "evaluation_config_sha256": canonical_digest(manifest["evaluation_config"]),
        "paired_task_ids_sha256": result["paired_task_ids_sha256"],
        "tasks": int(result["tasks"]),
        "baseline_successful_episodes": int(result["baseline_successful_episodes"]),
        "trained_successful_episodes": trained_successes,
        "baseline_score": float(result["baseline_score"]),
        "trained_score": float(result["trained_score"]),
        "evaluation_seconds": elapsed,
        "evaluation_cost_usd": cost,
        "cost_source": result["cost_source"],
        "cost_per_successful_episode_usd": (
            cost / trained_successes if trained_successes else 0.0
        ),
        "artifact_sha256": hash_artifact(artifact_path),
    }
    destination = output_dir / "evidence" / f"{suite_name}-seed{seed}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_run(evidence, destination)
    destination.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return destination


def write_accounting(
    output_dir: Path,
    *,
    mode: str,
    manifest_path: Path,
    attempts: Sequence[Mapping[str, Any]],
) -> None:
    """Account for every requested suite/seed, including retained failures."""
    completed = sum(item["status"] == "completed" for item in attempts)
    payload = {
        "schema_version": 1,
        "kind": "stateset-agent-quality-accounting",
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
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--minimum-mean-improvement", type=float, default=0.03)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the matrix, retaining failures and validating publication output."""
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.timeout_seconds < 1:
            raise AgentQualityRunnerError("timeout-seconds must be >= 1")
        if args.dry_run and args.preflight:
            raise AgentQualityRunnerError("--dry-run and --preflight are exclusive")
        seeds = manifest["seeds"][:1] if args.preflight else manifest["seeds"]
        if args.dry_run:
            for seed in seeds:
                for suite in manifest["suites"]:
                    print(f"seed={seed} suite={suite['name']}")
            return 0
        commit = git_commit(args.root)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        attempts: list[dict[str, Any]] = []
        for seed in seeds:
            for suite in manifest["suites"]:
                suite_name = str(suite["name"])
                try:
                    evidence = run_suite(
                        manifest,
                        suite,
                        seed,
                        output_dir=args.output_dir,
                        root=args.root,
                        harness_commit=commit,
                        timeout_seconds=args.timeout_seconds,
                    )
                except (AgentQualityRunnerError, AgentQualityEvidenceError) as exc:
                    attempts.append(
                        {
                            "suite": suite_name,
                            "seed": seed,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    continue
                attempts.append(
                    {
                        "suite": suite_name,
                        "seed": seed,
                        "status": "completed",
                        "evidence": str(evidence),
                    }
                )
        write_accounting(
            args.output_dir,
            mode="preflight" if args.preflight else "measured",
            manifest_path=args.manifest,
            attempts=attempts,
        )
        failures = sum(item["status"] == "failed" for item in attempts)
        if failures:
            raise AgentQualityRunnerError(
                f"matrix completed {len(attempts) - failures}/{len(attempts)} attempts"
            )
        if not args.preflight:
            runs = load_runs([args.output_dir / "evidence"])
            validate_matrix(
                runs,
                min_seeds=len(manifest["seeds"]),
                minimum_mean_improvement=args.minimum_mean_improvement,
            )
            report = summarize(runs)
            report_path = args.output_dir / "report.json"
            report_path.write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
    except (AgentQualityRunnerError, AgentQualityEvidenceError) as exc:
        print(f"agent-quality matrix rejected: {exc}", file=sys.stderr)
        return 2
    print(
        f"completed {len(attempts)} agent-quality attempts"
        + (" (preflight only)" if args.preflight else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
