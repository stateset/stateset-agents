#!/usr/bin/env python3
"""Run a pinned upstream agent harness for two policies and normalize evidence.

The upstream command is intentionally configured as an argv list and executed
without a shell. It must write one JSON object per line to ``{output}`` using
the task record contract documented in ``benchmarks/README.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class PairedHarnessError(ValueError):
    """Raised when paired upstream evidence is incomplete or untrustworthy."""


ALLOWED_COST_SOURCES = {
    "provider-api",
    "provider-invoice",
    "local-meter",
    "sponsored",
}
COMMAND_PLACEHOLDERS = {
    "{policy}",
    "{model}",
    "{model_revision}",
    "{seed}",
    "{suite}",
    "{suite_revision}",
    "{split}",
    "{evaluation_config_json}",
    "{output}",
    "{artifact_dir}",
}


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 for a JSON-compatible value."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairedHarnessError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise PairedHarnessError(f"{label} must be a JSON object: {path}")
    return dict(value)


def load_suite_config(path: Path, suite: str) -> dict[str, Any]:
    """Load and validate one suite's pinned command configuration."""
    config = _load_object(path, label="harness config")
    if config.get("schema_version") != 1:
        raise PairedHarnessError("harness config must have schema_version=1")
    if config.get("kind") != "stateset-paired-agent-harness-config":
        raise PairedHarnessError("harness config kind is invalid")
    suites = config.get("suites")
    if not isinstance(suites, Mapping) or suite not in suites:
        raise PairedHarnessError(f"harness config does not define {suite}")
    value = suites[suite]
    if not isinstance(value, Mapping):
        raise PairedHarnessError(f"harness config for {suite} must be an object")
    command = value.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise PairedHarnessError(f"harness config for {suite} needs string argv")
    missing = COMMAND_PLACEHOLDERS.difference(command)
    if missing:
        raise PairedHarnessError(
            f"harness config for {suite} is missing placeholders: "
            + ", ".join(sorted(missing))
        )
    repository_path = value.get("repository_path")
    if not isinstance(repository_path, str) or not repository_path.strip():
        raise PairedHarnessError(f"harness config for {suite} needs repository_path")
    timeout = value.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise PairedHarnessError(f"harness config for {suite} needs positive timeout")
    if value.get("cost_source") not in ALLOWED_COST_SOURCES:
        raise PairedHarnessError(f"harness config for {suite} has invalid cost_source")
    return dict(value)


def verify_repository(path: Path, expected_revision: str) -> None:
    """Require a clean upstream checkout at the manifest-pinned revision."""
    if not path.is_dir():
        raise PairedHarnessError(f"upstream repository does not exist: {path}")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    actual = revision.stdout.strip()
    if revision.returncode != 0 or actual != expected_revision:
        raise PairedHarnessError(
            f"upstream repository revision mismatch: expected {expected_revision}, "
            f"got {actual or 'unresolved'}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise PairedHarnessError("upstream repository worktree must be clean")


def format_command(command: Sequence[str], values: Mapping[str, Any]) -> list[str]:
    """Substitute only whole-token placeholders in an argv template."""
    formatted: list[str] = []
    for part in command:
        match = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", part)
        if match is None:
            formatted.append(part)
            continue
        name = match.group(1)
        if name not in values:
            raise PairedHarnessError(f"unknown command placeholder: {name}")
        formatted.append(str(values[name]))
    return formatted


def load_task_records(path: Path, *, policy: str) -> list[dict[str, Any]]:
    """Load strict, ordered per-task JSONL emitted by an upstream command."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PairedHarnessError(f"{policy} wrote no task records: {path}") from exc
    if not lines:
        raise PairedHarnessError(f"{policy} task records are empty")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PairedHarnessError(
                f"{policy} record {line_number} is invalid JSON"
            ) from exc
        if not isinstance(raw, Mapping):
            raise PairedHarnessError(f"{policy} record {line_number} must be an object")
        task_id = raw.get("task_id")
        success = raw.get("success")
        cost = raw.get("cost_usd")
        if not isinstance(task_id, str) or not task_id.strip():
            raise PairedHarnessError(
                f"{policy} record {line_number} needs a non-empty task_id"
            )
        if task_id in seen:
            raise PairedHarnessError(f"{policy} contains duplicate task_id {task_id}")
        if not isinstance(success, bool):
            raise PairedHarnessError(
                f"{policy} record {task_id} success must be boolean"
            )
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            raise PairedHarnessError(
                f"{policy} record {task_id} cost_usd must be numeric"
            )
        numeric_cost = float(cost)
        if not math.isfinite(numeric_cost) or numeric_cost < 0:
            raise PairedHarnessError(
                f"{policy} record {task_id} cost_usd must be finite and non-negative"
            )
        seen.add(task_id)
        records.append(
            {"task_id": task_id, "success": success, "cost_usd": numeric_cost}
        )
    return records


def run_policy(
    policy: str,
    *,
    command: Sequence[str],
    values: Mapping[str, Any],
    repository_path: Path,
    policy_dir: Path,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Run one policy, retaining its streams and normalized task records."""
    output = policy_dir / "tasks.jsonl"
    policy_dir.mkdir(parents=True, exist_ok=False)
    policy_values = dict(values)
    policy_values.update(
        {
            "policy": policy,
            "output": output.resolve(),
            "artifact_dir": policy_dir.resolve(),
        }
    )
    argv = format_command(command, policy_values)
    try:
        completed = subprocess.run(
            argv,
            cwd=repository_path,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        (policy_dir / "failure.json").write_text(
            json.dumps(
                {"kind": "timeout", "timeout_seconds": timeout_seconds}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        raise PairedHarnessError(f"{policy} harness timed out") from exc
    (policy_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (policy_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        (policy_dir / "failure.json").write_text(
            json.dumps({"kind": "exit", "returncode": completed.returncode}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise PairedHarnessError(f"{policy} harness exited {completed.returncode}")
    return load_task_records(output, policy=policy)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the policy pair and return the neutral matrix-runner payload."""
    if args.suite not in {"tau3-bench", "bfcl-v4", "swe-bench-verified"}:
        raise PairedHarnessError(f"unsupported suite: {args.suite}")
    try:
        evaluation_config = json.loads(args.evaluation_config_json)
    except json.JSONDecodeError as exc:
        raise PairedHarnessError("evaluation config is invalid JSON") from exc
    if not isinstance(evaluation_config, Mapping) or not evaluation_config:
        raise PairedHarnessError("evaluation config must be a non-empty object")
    config_path = Path(args.harness_config).resolve()
    suite_config = load_suite_config(config_path, args.suite)
    repository_path = Path(suite_config["repository_path"])
    if not repository_path.is_absolute():
        repository_path = (config_path.parent / repository_path).resolve()
    verify_repository(repository_path, args.suite_revision)
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "seed": args.seed,
        "suite": args.suite,
        "suite_revision": args.suite_revision,
        "split": args.split,
        "evaluation_config_json": json.dumps(
            evaluation_config, sort_keys=True, separators=(",", ":")
        ),
    }
    command = suite_config["command"]
    baseline = run_policy(
        "baseline",
        command=command,
        values={
            **values,
            "model": args.baseline_model,
            "model_revision": args.baseline_revision,
        },
        repository_path=repository_path,
        policy_dir=artifact_dir / "baseline",
        timeout_seconds=suite_config["timeout_seconds"],
    )
    trained = run_policy(
        "trained",
        command=command,
        values={
            **values,
            "model": args.trained_model,
            "model_revision": args.trained_revision,
        },
        repository_path=repository_path,
        policy_dir=artifact_dir / "trained",
        timeout_seconds=suite_config["timeout_seconds"],
    )
    baseline_ids = [record["task_id"] for record in baseline]
    trained_ids = [record["task_id"] for record in trained]
    if baseline_ids != trained_ids:
        raise PairedHarnessError(
            "baseline and trained policies must emit identical ordered task IDs"
        )
    baseline_successes = sum(record["success"] for record in baseline)
    trained_successes = sum(record["success"] for record in trained)
    cost = sum(record["cost_usd"] for record in baseline + trained)
    summary = {
        "schema_version": 1,
        "kind": "stateset-paired-agent-task-summary",
        "task_ids": baseline_ids,
        "baseline_successful_episodes": baseline_successes,
        "trained_successful_episodes": trained_successes,
        "evaluation_cost_usd": cost,
        "cost_source": suite_config["cost_source"],
    }
    (artifact_dir / "paired-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    tasks = len(baseline_ids)
    return {
        "status": "completed",
        "measured": True,
        "suite": args.suite,
        "suite_revision": args.suite_revision,
        "split": args.split,
        "seed": args.seed,
        "framework_version": args.framework_version,
        "baseline_model": args.baseline_model,
        "baseline_model_revision": args.baseline_revision,
        "trained_model": args.trained_model,
        "trained_model_revision": args.trained_revision,
        "evaluation_config_sha256": canonical_digest(evaluation_config),
        "paired_task_ids_sha256": canonical_digest(baseline_ids),
        "tasks": tasks,
        "baseline_successful_episodes": baseline_successes,
        "trained_successful_episodes": trained_successes,
        "baseline_score": baseline_successes / tasks,
        "trained_score": trained_successes / tasks,
        "evaluation_cost_usd": cost,
        "cost_source": suite_config["cost_source"],
        "artifact_path": str(artifact_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract consumed by the matrix manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="adapter_command", required=True)
    validate = commands.add_parser(
        "validate", help="validate every suite command without executing it"
    )
    validate.add_argument("--harness-config", required=True)
    execute = commands.add_parser("run", help="execute one paired suite/seed")
    execute.add_argument("--harness-config", required=True)
    execute.add_argument("--seed", required=True, type=int)
    execute.add_argument("--suite", required=True)
    execute.add_argument("--suite-revision", required=True)
    execute.add_argument("--split", required=True)
    execute.add_argument("--framework-version", required=True)
    execute.add_argument("--baseline-model", required=True)
    execute.add_argument("--baseline-revision", required=True)
    execute.add_argument("--trained-model", required=True)
    execute.add_argument("--trained-revision", required=True)
    execute.add_argument("--evaluation-config-json", required=True)
    execute.add_argument("--adapter-output", required=True)
    execute.add_argument("--artifact-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the adapter CLI and write its neutral result exactly once."""
    args = build_parser().parse_args(argv)
    try:
        if args.adapter_command == "validate":
            config_path = Path(args.harness_config).resolve()
            for suite in ("tau3-bench", "bfcl-v4", "swe-bench-verified"):
                load_suite_config(config_path, suite)
            print("validated paired harness commands for 3 suites")
            return 0
        result = run(args)
        output = Path(args.adapter_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except PairedHarnessError as exc:
        print(f"paired harness rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
