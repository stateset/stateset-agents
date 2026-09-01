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
    """Hash a retained file or directory tree deterministically."""
    if not path.exists():
        raise FlagshipError(f"artifact does not exist: {path}")
    digest = hashlib.sha256()
    files = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    if not files:
        raise FlagshipError(f"artifact is empty: {path}")
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        digest.update(relative.encode())
        with item.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


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
    ):
        expected = config_sha256 if key == "config_sha256" else manifest[key]
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
    artifact_dir = attempt / "artifact"
    adapter_output = attempt / "adapter.json"
    attempt.mkdir(parents=True, exist_ok=False)
    artifact_dir.mkdir()
    config_json = canonical_json(manifest["config"])
    config_sha256 = hashlib.sha256(config_json.encode()).hexdigest()
    values = {
        "seed": seed,
        "mode": "preflight" if preflight else "measured",
        "framework_version": manifest["framework_version"],
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "dataset_revision": manifest["dataset_revision"],
        "config_json": config_json,
        "config_sha256": config_sha256,
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
        "schema_version": 1,
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
        "hardware": adapter["hardware"],
        "metrics": adapter["metrics"],
        "artifact_sha256": hash_artifact(artifact_path),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.dry_run:
            _dry_run(manifest, args.preflight)
            return 0
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
