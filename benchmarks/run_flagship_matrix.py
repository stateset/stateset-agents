#!/usr/bin/env python3
"""Collect and validate the three-seed StateSet flagship benchmark matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FlagshipError(ValueError):
    """Raised when flagship evidence is incomplete or not reproducible."""


HEX = frozenset("0123456789abcdef")
REQUIRED_PLACEHOLDERS = {
    "{seed}",
    "{mode}",
    "{framework_version}",
    "{model}",
    "{model_revision}",
    "{dataset_revision}",
    "{config_json}",
    "{config_sha256}",
    "{judge_json}",
    "{judge_sha256}",
    "{adapter_output}",
    "{artifact_dir}",
}
SECRET_FRAGMENTS = ("api_key", "token", "password", "credential", "secret")
T_CRITICAL_95_DF2 = 4.303


def canonical_json(value: Mapping[str, Any]) -> str:
    """Serialize a mapping deterministically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest_json(value: Mapping[str, Any]) -> str:
    """Return the SHA-256 of canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _text(data: Mapping[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FlagshipError(f"{label}.{key} must be a non-empty string")
    return value


def _hex(data: Mapping[str, Any], key: str, label: str, length: int) -> str:
    value = _text(data, key, label)
    if len(value) != length or any(char not in HEX for char in value):
        raise FlagshipError(f"{label}.{key} must be {length} lowercase hex characters")
    return value


def _number(data: Mapping[str, Any], key: str, label: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FlagshipError(f"{label}.{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FlagshipError(f"{label}.{key} must be finite")
    return result


def _reject_secrets(value: Any, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in SECRET_FRAGMENTS):
                raise FlagshipError(
                    f"{path}.{key} may not contain credentials; use environment variables"
                )
            _reject_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{path}[{index}]")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the immutable flagship experiment contract."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FlagshipError(f"{path}: invalid JSON") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise FlagshipError("manifest must be an object with schema_version=1")
    if raw.get("kind") != "stateset-flagship-manifest":
        raise FlagshipError("manifest.kind is invalid")
    _reject_secrets(raw)
    for key in (
        "protocol",
        "framework_version",
        "provider",
        "cost_source",
        "model",
        "dataset",
        "trainer",
        "task",
    ):
        _text(raw, key, "manifest")
    _hex(raw, "model_revision", "manifest", 40)
    _hex(raw, "dataset_revision", "manifest", 40)
    parameters = _number(raw, "model_parameter_count", "manifest")
    if not 7_000_000_000 <= parameters <= 9_000_000_000:
        raise FlagshipError("manifest.model_parameter_count must be between 7B and 9B")
    if raw["trainer"] != "gspo" or raw["task"] != "customer_support":
        raise FlagshipError("flagship protocol requires gspo on customer_support")

    seeds = raw.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or len(seeds) != len(set(seeds))
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        )
    ):
        raise FlagshipError("manifest.seeds must contain exactly three unique integers")

    config = raw.get("config")
    if not isinstance(config, Mapping):
        raise FlagshipError("manifest.config must be an object")
    if _number(config, "num_train_examples", "manifest.config") < 500:
        raise FlagshipError("manifest.config.num_train_examples must be >= 500")
    if _number(config, "num_eval_examples", "manifest.config") < 200:
        raise FlagshipError("manifest.config.num_eval_examples must be >= 200")
    if _number(config, "max_wall_clock_seconds", "manifest.config") <= 0:
        raise FlagshipError("manifest.config.max_wall_clock_seconds must be > 0")
    if _number(config, "max_cost_usd_per_seed", "manifest.config") <= 0:
        raise FlagshipError("manifest.config.max_cost_usd_per_seed must be > 0")

    judge = raw.get("judge")
    if not isinstance(judge, Mapping):
        raise FlagshipError("manifest.judge must be an object")
    _text(judge, "model", "manifest.judge")
    _hex(judge, "revision", "manifest.judge", 40)
    _hex(judge, "rubric_revision", "manifest.judge", 40)

    hardware = raw.get("hardware")
    if not isinstance(hardware, Mapping):
        raise FlagshipError("manifest.hardware must be an object")
    _text(hardware, "gpu", "manifest.hardware")
    if _number(hardware, "gpu_count", "manifest.hardware") < 1:
        raise FlagshipError("manifest.hardware.gpu_count must be >= 1")

    command = raw.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
    ):
        raise FlagshipError("manifest.command must be a non-empty string list")
    missing = REQUIRED_PLACEHOLDERS.difference(command)
    if missing:
        raise FlagshipError(
            "manifest.command is missing placeholders: " + ", ".join(sorted(missing))
        )
    return dict(raw)


def validate_execution_ready(
    manifest: Mapping[str, Any], expected_framework_version: str
) -> None:
    """Reject template or stale contracts before any measured execution."""
    placeholders: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str) and value.startswith("REPLACE_WITH_"):
            placeholders.append(path)

    walk(manifest, "manifest")
    if placeholders:
        raise FlagshipError(
            "measured execution rejects template placeholders: "
            + ", ".join(placeholders)
        )
    for path, revision in (
        ("manifest.model_revision", manifest["model_revision"]),
        ("manifest.dataset_revision", manifest["dataset_revision"]),
        ("manifest.judge.revision", manifest["judge"]["revision"]),
        ("manifest.judge.rubric_revision", manifest["judge"]["rubric_revision"]),
    ):
        if revision == "0" * 40:
            raise FlagshipError(f"{path} must not be the template zero revision")
    if manifest["framework_version"] != expected_framework_version:
        raise FlagshipError(
            "manifest.framework_version must match the installed StateSet version "
            f"({expected_framework_version})"
        )


def git_commit(root: Path) -> str:
    """Resolve the harness commit and reject measured runs from dirty trees."""
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    if status.returncode != 0 or status.stdout.strip():
        raise FlagshipError("flagship harness worktree must be clean")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) != 40:
        raise FlagshipError("could not resolve a full harness commit")
    return commit


def hash_artifact(path: Path) -> str:
    """Hash a retained file or directory tree without following symlinks."""
    if not path.exists() or path.is_symlink():
        raise FlagshipError(f"artifact does not exist or is unsafe: {path}")
    digest = hashlib.sha256()
    entries = [] if path.is_file() else sorted(path.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise FlagshipError(f"artifact tree contains a symlink: {path}")
    files = [path] if path.is_file() else [item for item in entries if item.is_file()]
    if not files:
        raise FlagshipError(f"artifact is empty: {path}")
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with item.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def verify_retained_artifact(data: Mapping[str, Any], source: Path) -> Path:
    """Resolve and re-hash one portable artifact inside the evidence bundle."""
    raw_path = _text(data, "artifact_path", str(source))
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise FlagshipError(f"{source}: artifact_path must be relative")
    bundle_root = source.resolve().parent
    artifact = (bundle_root / candidate).resolve()
    if not artifact.is_relative_to(bundle_root):
        raise FlagshipError(f"{source}: artifact_path escapes the evidence bundle")
    if hash_artifact(artifact) != data.get("artifact_sha256"):
        raise FlagshipError(f"{source}: retained artifact digest mismatch")
    return artifact


def validate_retained_evidence(
    data: Mapping[str, Any], manifest: Mapping[str, Any], source: Path
) -> dict[str, Any]:
    """Validate one collected schema-v2 flagship document independently."""
    if (
        data.get("schema_version") != 2
        or data.get("kind") != "stateset-flagship-evidence"
        or data.get("measured") is not True
    ):
        raise FlagshipError(f"{source}: measured flagship schema_version=2 required")
    if data.get("preflight") is not False:
        raise FlagshipError(f"{source}: preflight evidence is not publishable")
    seed = data.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in manifest["seeds"]
    ):
        raise FlagshipError(f"{source}: seed is not declared by the manifest")
    expected = {
        "manifest_sha256": digest_json(manifest),
        "framework_version": manifest["framework_version"],
        "provider": manifest["provider"],
        "cost_source": manifest["cost_source"],
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "model_parameter_count": manifest["model_parameter_count"],
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["dataset_revision"],
        "trainer": manifest["trainer"],
        "task": manifest["task"],
        "config_sha256": digest_json(manifest["config"]),
        "judge_sha256": digest_json(manifest["judge"]),
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise FlagshipError(f"{source}: {key} does not match the manifest")
    _hex(data, "harness_commit", str(source), 40)
    if data.get("judge") != manifest["judge"]:
        raise FlagshipError(f"{source}: judge does not match the manifest")
    hardware = data.get("hardware")
    if not isinstance(hardware, Mapping):
        raise FlagshipError(f"{source}: hardware must be an object")
    for key in ("gpu", "gpu_count"):
        if hardware.get(key) != manifest["hardware"][key]:
            raise FlagshipError(f"{source}: hardware.{key} does not match manifest")
    _text(hardware, "cuda", f"{source}.hardware")
    _text(hardware, "driver", f"{source}.hardware")
    metrics = data.get("metrics")
    if not isinstance(metrics, Mapping):
        raise FlagshipError(f"{source}: metrics must be an object")
    adapter_shape = {
        **expected,
        "status": "completed",
        "measured": True,
        "artifact_path": raw_path if (raw_path := data.get("artifact_path")) else "",
        "hardware": hardware,
        "metrics": metrics,
    }
    validate_adapter(adapter_shape, manifest, expected["config_sha256"], source)
    if _number(data, "wall_clock_seconds", str(source)) <= 0:
        raise FlagshipError(f"{source}: wall_clock_seconds must be positive")
    collected_at = _text(data, "collected_at", str(source))
    try:
        collected = datetime.fromisoformat(collected_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FlagshipError(f"{source}: collected_at must be ISO-8601") from exc
    if collected.tzinfo is None:
        raise FlagshipError(f"{source}: collected_at must include a UTC offset")
    _hex(data, "artifact_sha256", str(source), 64)
    verify_retained_artifact(data, source)
    return dict(data)


def load_retained_evidence(
    inputs: Sequence[Path], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Load a portable retained flagship matrix and re-verify every artifact."""
    paths: list[Path] = []
    for candidate in inputs:
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.json")))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            raise FlagshipError(f"retained evidence does not exist: {candidate}")
    if not paths:
        raise FlagshipError("no retained flagship evidence found")
    runs: list[dict[str, Any]] = []
    for path in dict.fromkeys(item.resolve() for item in paths):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FlagshipError(f"{path}: invalid retained evidence JSON") from exc
        if not isinstance(raw, Mapping):
            raise FlagshipError(f"{path}: retained evidence must be an object")
        runs.append(validate_retained_evidence(raw, manifest, path))
    return runs


def _format_command(command: Sequence[str], values: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for part in command:
        match = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", part)
        if match is None:
            result.append(part)
        else:
            name = match.group(1)
            if name not in values:
                raise FlagshipError(f"unknown command placeholder: {name}")
            result.append(str(values[name]))
    return result


def validate_adapter(
    raw: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config_sha256: str,
    source: Path,
) -> dict[str, Any]:
    """Validate provider-driver output before accepting it as measured evidence."""
    if raw.get("status") != "completed" or raw.get("measured") is not True:
        raise FlagshipError(f"{source}: adapter did not report measured completion")
    for key in (
        "framework_version",
        "model_revision",
        "dataset_revision",
        "config_sha256",
        "judge_sha256",
    ):
        if key == "config_sha256":
            expected = config_sha256
        elif key == "judge_sha256":
            expected = digest_json(manifest["judge"])
        else:
            expected = manifest[key]
        if raw.get(key) != expected:
            raise FlagshipError(f"{source}: {key} does not match the manifest")
    if raw.get("cost_source") != manifest["cost_source"]:
        raise FlagshipError(f"{source}: cost_source does not match the manifest")
    artifact = _text(raw, "artifact_path", str(source))
    hardware = raw.get("hardware")
    if not isinstance(hardware, Mapping):
        raise FlagshipError(f"{source}: hardware must be an object")
    for key in ("gpu", "gpu_count"):
        if hardware.get(key) != manifest["hardware"][key]:
            raise FlagshipError(f"{source}: hardware.{key} does not match the manifest")
    _text(hardware, "cuda", f"{source}.hardware")
    _text(hardware, "driver", f"{source}.hardware")
    metrics = raw.get("metrics")
    if not isinstance(metrics, Mapping):
        raise FlagshipError(f"{source}: metrics must be an object")
    for key in (
        "baseline_score",
        "final_score",
        "judge_self_disagreement",
        "train_examples",
        "eval_examples",
        "peak_vram_mb",
        "cost_usd",
    ):
        _number(metrics, key, f"{source}.metrics")
    for key in ("baseline_score", "final_score", "judge_self_disagreement"):
        value = float(metrics[key])
        if not 0 <= value <= 1:
            raise FlagshipError(f"{source}: metrics.{key} must be in [0, 1]")
    config = manifest["config"]
    if metrics["train_examples"] < config["num_train_examples"]:
        raise FlagshipError(f"{source}: insufficient training examples")
    if metrics["eval_examples"] < config["num_eval_examples"]:
        raise FlagshipError(f"{source}: insufficient evaluation examples")
    if metrics["peak_vram_mb"] <= 0 or metrics["cost_usd"] <= 0:
        raise FlagshipError(f"{source}: peak VRAM and provider cost must be positive")
    if metrics["cost_usd"] > config["max_cost_usd_per_seed"]:
        raise FlagshipError(f"{source}: provider cost exceeded the declared ceiling")
    return {**dict(raw), "artifact_path": artifact}


def validate_matrix(
    runs: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the publication gates to a complete measured seed matrix."""
    expected = set(manifest["seeds"])
    observed = [run.get("seed") for run in runs]
    if (
        len(runs) != len(expected)
        or set(observed) != expected
        or len(observed) != len(set(observed))
    ):
        raise FlagshipError("evidence must contain every declared seed exactly once")
    if any(run.get("preflight") is not False for run in runs):
        raise FlagshipError("preflight evidence cannot pass the publication gate")
    harnesses = {run.get("harness_commit") for run in runs}
    if len(harnesses) != 1:
        raise FlagshipError("evidence spans multiple harness commits")
    manifest_digests = {run.get("manifest_sha256") for run in runs}
    if len(manifest_digests) != 1:
        raise FlagshipError("evidence spans multiple flagship manifests")
    artifact_paths = [
        run.get("artifact_path") for run in runs if "artifact_path" in run
    ]
    if artifact_paths and len(artifact_paths) != len(set(artifact_paths)):
        raise FlagshipError("each flagship seed must retain a distinct artifact path")
    improvements = [
        float(run["metrics"]["final_score"]) - float(run["metrics"]["baseline_score"])
        for run in runs
    ]
    mean = statistics.mean(improvements)
    std = statistics.stdev(improvements)
    lower = mean - T_CRITICAL_95_DF2 * std / math.sqrt(len(improvements))
    if mean < 0.03:
        raise FlagshipError(f"mean improvement {mean:+.4f} is below +0.0300")
    if lower <= 0:
        raise FlagshipError(
            f"paired 95% confidence lower bound {lower:+.4f} is not positive"
        )
    final_std = statistics.stdev(float(run["metrics"]["final_score"]) for run in runs)
    if final_std > 0.10:
        raise FlagshipError(
            f"final-score standard deviation {final_std:.4f} exceeds 0.10"
        )
    if any(float(run["metrics"]["judge_self_disagreement"]) > 0.05 for run in runs):
        raise FlagshipError("judge self-disagreement exceeded 0.05")
    return {
        "passed": True,
        "seed_count": len(runs),
        "seeds": sorted(expected),
        "mean_improvement": mean,
        "improvement_stddev": std,
        "paired_95pct_lower_bound": lower,
        "final_score_stddev": final_std,
        "total_cost_usd": sum(float(run["metrics"]["cost_usd"]) for run in runs),
        "total_wall_clock_seconds": sum(
            float(run["wall_clock_seconds"]) for run in runs
        ),
    }


def run_seed(
    manifest: Mapping[str, Any],
    seed: int,
    output_dir: Path,
    commit: str,
    preflight: bool,
) -> dict[str, Any]:
    """Execute one seed, retain logs, and produce normalized evidence."""
    attempt = output_dir / "attempts" / f"seed-{seed}"
    artifact_dir = output_dir / "evidence" / "artifacts" / f"seed-{seed}"
    adapter_output = attempt / "adapter.json"
    attempt.mkdir(parents=True, exist_ok=False)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    config_json = canonical_json(manifest["config"])
    config_sha256 = hashlib.sha256(config_json.encode()).hexdigest()
    judge_json = canonical_json(manifest["judge"])
    judge_sha256 = hashlib.sha256(judge_json.encode()).hexdigest()
    values = {
        "seed": seed,
        "mode": "preflight" if preflight else "measured",
        "framework_version": manifest["framework_version"],
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "dataset_revision": manifest["dataset_revision"],
        "config_json": config_json,
        "config_sha256": config_sha256,
        "judge_json": judge_json,
        "judge_sha256": judge_sha256,
        "adapter_output": adapter_output.resolve(),
        "artifact_dir": artifact_dir.resolve(),
    }
    command = _format_command(manifest["command"], values)
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        shell=False,
    )
    elapsed = time.monotonic() - started
    (attempt / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (attempt / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise FlagshipError(
            f"seed {seed}: driver exited {completed.returncode}; logs retained in {attempt}"
        )
    try:
        raw = json.loads(adapter_output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FlagshipError(
            f"seed {seed}: driver did not write valid adapter JSON"
        ) from exc
    if not isinstance(raw, Mapping):
        raise FlagshipError(f"seed {seed}: adapter output must be an object")
    adapter = validate_adapter(raw, manifest, config_sha256, adapter_output)
    artifact_path = Path(adapter["artifact_path"]).resolve()
    owned = artifact_dir.resolve()
    if artifact_path != owned and owned not in artifact_path.parents:
        raise FlagshipError(f"seed {seed}: artifact must stay inside {owned}")
    if elapsed > float(manifest["config"]["max_wall_clock_seconds"]):
        raise FlagshipError(
            f"seed {seed}: external wall time exceeded the declared ceiling"
        )
    evidence = {
        "schema_version": 2,
        "kind": "stateset-flagship-evidence",
        "measured": True,
        "preflight": preflight,
        "seed": seed,
        "harness_commit": commit,
        "manifest_sha256": digest_json(manifest),
        "framework_version": manifest["framework_version"],
        "provider": manifest["provider"],
        "cost_source": manifest["cost_source"],
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "model_parameter_count": manifest["model_parameter_count"],
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["dataset_revision"],
        "trainer": manifest["trainer"],
        "task": manifest["task"],
        "config_sha256": config_sha256,
        "judge": dict(manifest["judge"]),
        "judge_sha256": judge_sha256,
        "hardware": adapter["hardware"],
        "metrics": adapter["metrics"],
        "artifact_sha256": hash_artifact(artifact_path),
        "artifact_path": artifact_path.relative_to(
            (output_dir / "evidence").resolve()
        ).as_posix(),
        "wall_clock_seconds": elapsed,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    path = output_dir / "evidence" / f"seed-{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def _dry_run(manifest: Mapping[str, Any], preflight: bool) -> None:
    seeds = manifest["seeds"][:1] if preflight else manifest["seeds"]
    for seed in seeds:
        print(f"seed={seed} mode={'preflight' if preflight else 'measured'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--validate-existing",
        nargs="+",
        type=Path,
        help="re-hash and validate an existing evidence matrix without execution",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.validate_existing:
            from stateset_agents import __version__

            validate_execution_ready(manifest, __version__)
            runs = load_retained_evidence(args.validate_existing, manifest)
            report = validate_matrix(runs, manifest)
            if args.output_dir is not None:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                (args.output_dir / "report.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
            print(f"validated {len(runs)} retained flagship runs")
            return 0
        if args.dry_run:
            _dry_run(manifest, args.preflight)
            return 0
        if args.output_dir is None:
            raise FlagshipError("--output-dir is required for benchmark execution")
        from stateset_agents import __version__

        validate_execution_ready(manifest, __version__)
        commit = git_commit(Path(__file__).resolve().parents[1])
        args.output_dir.mkdir(parents=True, exist_ok=False)
        seeds = manifest["seeds"][:1] if args.preflight else manifest["seeds"]
        runs: list[dict[str, Any]] = []
        failures: list[str] = []
        for seed in seeds:
            try:
                runs.append(
                    run_seed(manifest, seed, args.output_dir, commit, args.preflight)
                )
            except FlagshipError as exc:
                failures.append(str(exc))
        report: dict[str, Any] = {
            "passed": False,
            "preflight": args.preflight,
            "failures": failures,
        }
        if not failures and not args.preflight:
            try:
                report.update(validate_matrix(runs, manifest))
            except FlagshipError as exc:
                failures.append(str(exc))
                report["failures"] = failures
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return 0 if not failures and (args.preflight or report["passed"]) else 1
    except FlagshipError as exc:
        print(f"flagship benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
