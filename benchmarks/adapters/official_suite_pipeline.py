#!/usr/bin/env python3
"""Execute official suite commands and normalize their retained artifacts.

The pipeline is the concrete bridge between the paired benchmark harness and
upstream CLIs. Commands are argv arrays and run without a shell; every declared
artifact is confined to the policy artifact directory. Official outputs are
normalized by ``official_result_normalizer`` after every configured command
succeeds.
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
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.adapters.official_result_normalizer import (  # noqa: E402
    OfficialResultError,
    normalize_bfcl_v4,
    normalize_swe_bench_verified,
    normalize_tau3,
    write_jsonl,
)


class OfficialPipelineError(ValueError):
    """Raised when an official suite pipeline is unsafe or incomplete."""


SUPPORTED_SUITES = {"tau3-bench", "bfcl-v4", "swe-bench-verified"}
REQUIRED_COMMAND_PLACEHOLDERS = {"{model}"}
RESERVED_VALUES = {
    "policy",
    "model",
    "model_revision",
    "seed",
    "suite",
    "suite_revision",
    "split",
    "artifact_dir",
    "upstream_repository",
    "official_results",
    "official_scores",
    "cost_records",
}


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _safe_artifact_path(artifact_dir: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise OfficialPipelineError(f"{label} must be a non-empty relative path")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise OfficialPipelineError(f"{label} must be relative to artifact_dir")
    resolved = (artifact_dir / candidate).resolve()
    try:
        resolved.relative_to(artifact_dir.resolve())
    except ValueError as exc:
        raise OfficialPipelineError(f"{label} escapes artifact_dir") from exc
    return resolved


def _command_list(value: Any, *, suite: str) -> list[list[str]]:
    if not isinstance(value, list) or not value:
        raise OfficialPipelineError(f"{suite} requires at least one command")
    commands: list[list[str]] = []
    for index, command in enumerate(value, start=1):
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(part, str) or not part for part in command)
        ):
            raise OfficialPipelineError(
                f"{suite} command {index} must be a non-empty string argv list"
            )
        commands.append(list(command))
    flattened = {part for command in commands for part in command}
    missing = REQUIRED_COMMAND_PLACEHOLDERS - flattened
    if missing:
        raise OfficialPipelineError(
            f"{suite} commands are missing placeholders: {', '.join(sorted(missing))}"
        )
    return commands


def load_pipeline_config(
    evaluation_config: Mapping[str, Any], suite: str, artifact_dir: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Validate one suite pipeline and resolve its confined artifact paths."""
    pipelines = evaluation_config.get("official_suite_pipelines")
    if not isinstance(pipelines, Mapping):
        raise OfficialPipelineError(
            "evaluation_config requires official_suite_pipelines"
        )
    raw = pipelines.get(suite)
    if not isinstance(raw, Mapping):
        raise OfficialPipelineError(f"no official pipeline configured for {suite}")
    config = dict(raw)
    config["commands"] = _command_list(config.get("commands"), suite=suite)
    timeout = config.get("command_timeout_seconds", 14400)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise OfficialPipelineError(f"{suite} command timeout must be positive")
    config["command_timeout_seconds"] = timeout

    paths = {
        "official_results": _safe_artifact_path(
            artifact_dir, config.get("results_path"), label="results_path"
        )
    }
    if suite == "bfcl-v4":
        paths["official_scores"] = _safe_artifact_path(
            artifact_dir, config.get("scores_path"), label="scores_path"
        )
    costs_path = config.get("cost_records_path")
    if costs_path is not None:
        paths["cost_records"] = _safe_artifact_path(
            artifact_dir, costs_path, label="cost_records_path"
        )
    if suite != "tau3-bench" and "cost_records" not in paths:
        raise OfficialPipelineError(f"{suite} requires cost_records_path")
    extra_paths = config.get("artifact_paths", {})
    if not isinstance(extra_paths, Mapping):
        raise OfficialPipelineError(f"{suite} artifact_paths must be an object")
    for name, raw_path in extra_paths.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            or name in RESERVED_VALUES
            or name in paths
        ):
            raise OfficialPipelineError(f"{suite} has invalid artifact path name")
        paths[name] = _safe_artifact_path(
            artifact_dir, raw_path, label=f"artifact_paths.{name}"
        )
    if len(set(paths.values())) != len(paths):
        raise OfficialPipelineError(f"{suite} artifact paths must be distinct")
    return config, paths


def _format_command(command: Sequence[str], values: Mapping[str, Any]) -> list[str]:
    formatted: list[str] = []
    for part in command:
        match = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", part)
        if match is None:
            if "{" in part or "}" in part:
                raise OfficialPipelineError(
                    "placeholders must occupy a complete command argument"
                )
            formatted.append(part)
            continue
        name = match.group(1)
        if name not in values:
            raise OfficialPipelineError(f"unknown command placeholder: {name}")
        formatted.append(str(values[name]))
    return formatted


def _scalar_config_values(config: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in config.items():
        if key in RESERVED_VALUES or key == "official_suite_pipelines":
            continue
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and not math.isfinite(value):
                raise OfficialPipelineError(f"evaluation_config.{key} must be finite")
            values[key] = value
    return values


def _repository_is_clean(repository: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and not completed.stdout.strip()


def execute_pipeline(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run configured upstream commands and return normalized task records."""
    if args.suite not in SUPPORTED_SUITES:
        raise OfficialPipelineError(f"unsupported suite: {args.suite}")
    try:
        evaluation_config = json.loads(args.evaluation_config_json)
    except json.JSONDecodeError as exc:
        raise OfficialPipelineError("evaluation config is invalid JSON") from exc
    if not isinstance(evaluation_config, Mapping):
        raise OfficialPipelineError("evaluation config must be an object")
    artifact_dir = Path(args.artifact_dir).resolve()
    output = Path(args.output).resolve()
    repository = Path(args.upstream_repository).resolve()
    if not artifact_dir.is_dir():
        raise OfficialPipelineError("artifact_dir must already exist")
    try:
        output.relative_to(artifact_dir)
    except ValueError as exc:
        raise OfficialPipelineError("output must be inside artifact_dir") from exc
    if not _repository_is_clean(repository):
        raise OfficialPipelineError(
            "upstream repository must be clean before execution"
        )

    pipeline, paths = load_pipeline_config(evaluation_config, args.suite, artifact_dir)
    pipeline_owned = [
        artifact_dir / "execution-manifest.json",
        *[
            artifact_dir / f"command-{index:02d}.{stream}.log"
            for index in range(1, len(pipeline["commands"]) + 1)
            for stream in ("stdout", "stderr")
        ],
    ]
    distinct_outputs = [output, *paths.values(), *pipeline_owned]
    if len(set(distinct_outputs)) != len(distinct_outputs):
        raise OfficialPipelineError(
            "pipeline output and artifact paths must be distinct"
        )
    occupied = [path for path in distinct_outputs if path.exists()]
    if occupied:
        raise OfficialPipelineError(
            "pipeline artifact already exists: " + ", ".join(map(str, occupied))
        )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    values: dict[str, Any] = {
        **_scalar_config_values(evaluation_config),
        **{name: str(path) for name, path in paths.items()},
        "policy": args.policy,
        "model": args.model,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "suite": args.suite,
        "suite_revision": args.suite_revision,
        "split": args.split,
        "artifact_dir": str(artifact_dir),
        "upstream_repository": str(repository),
    }
    commands = [_format_command(command, values) for command in pipeline["commands"]]
    manifest = {
        "schema_version": 1,
        "kind": "stateset-official-suite-execution",
        "policy": args.policy,
        "model": args.model,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "suite": args.suite,
        "suite_revision": args.suite_revision,
        "split": args.split,
        "evaluation_config_sha256": _canonical_digest(evaluation_config),
        "commands": commands,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    (artifact_dir / "execution-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    started = time.monotonic()
    timeout = pipeline["command_timeout_seconds"]
    for index, command in enumerate(commands, start=1):
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise OfficialPipelineError(f"{args.suite} pipeline timed out")
        try:
            completed = subprocess.run(
                command,
                cwd=repository,
                capture_output=True,
                text=True,
                check=False,
                timeout=remaining,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as exc:
            raise OfficialPipelineError(
                f"{args.suite} command {index} timed out"
            ) from exc
        (artifact_dir / f"command-{index:02d}.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (artifact_dir / f"command-{index:02d}.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise OfficialPipelineError(
                f"{args.suite} command {index} exited {completed.returncode}"
            )
        if not _repository_is_clean(repository):
            raise OfficialPipelineError(
                f"{args.suite} command {index} modified the upstream checkout"
            )

    costs = paths.get("cost_records")
    try:
        if args.suite == "tau3-bench":
            records = normalize_tau3(paths["official_results"], costs_path=costs)
        elif args.suite == "bfcl-v4":
            assert costs is not None
            records = normalize_bfcl_v4(
                paths["official_results"],
                paths["official_scores"],
                costs_path=costs,
            )
        else:
            assert costs is not None
            records = normalize_swe_bench_verified(
                paths["official_results"], costs_path=costs
            )
    except OfficialResultError as exc:
        raise OfficialPipelineError(str(exc)) from exc
    write_jsonl(records, output)
    return records


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface used by the paired harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--suite-revision", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--evaluation-config-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--upstream-repository", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute an official suite pipeline, returning two on rejected evidence."""
    args = build_parser().parse_args(argv)
    try:
        execute_pipeline(args)
    except OfficialPipelineError as exc:
        print(f"official suite pipeline rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
