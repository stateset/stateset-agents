#!/usr/bin/env python3
"""Run one external training backend and retain strict conformance evidence.

This is an execution-compatibility gate, not a performance comparison.  Each
backend may run in its own dependency-compatible image while producing the
same evidence schema.  The manifest binds the workload timeout; paid provider
wrappers must additionally reject quotes above its cost ceiling before rental.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil

# All subprocess calls use fixed argument arrays and never enable a shell.
import subprocess  # nosec B404
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from stateset_agents.training import TrainingExperiment
from stateset_agents.training.adapters.nemo_rl import nemo_rl_backend
from stateset_agents.training.adapters.openrlhf import openrlhf_backend
from stateset_agents.training.adapters.verl import verl_backend

_BACKEND_FACTORIES = {
    "nemo-rl": nemo_rl_backend,
    "openrlhf": openrlhf_backend,
    "verl": verl_backend,
}
_EXPERIMENT_FIELDS = frozenset(
    {
        "algorithm",
        "model",
        "model_revision",
        "dataset_uri",
        "dataset_sha256",
        "seed",
        "config",
        "task",
        "environment",
        "reward",
        "requirements",
    }
)
_EXECUTION_FIELDS = frozenset(
    {
        "provider",
        "provider_tier",
        "container_image",
        "gpu_name",
        "gpu_count",
        "container_disk_gb",
        "timeout_seconds",
        "max_lifetime_seconds",
        "max_cost_usd",
    }
)
_IMAGE_DIGEST = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


class ConformanceError(ValueError):
    """Raised when a conformance manifest or result is not auditable."""


def _required_executable(name: str) -> str:
    """Resolve a fixed system executable before launching a shell-free argv."""
    path = shutil.which(name)
    if path is None:
        raise ConformanceError(f"required executable is unavailable: {name}")
    return path


def canonical_digest(value: Mapping[str, Any]) -> str:
    """Hash a JSON object using the backend protocol's canonical encoding."""
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as exc:
        raise ConformanceError(
            "manifest must contain JSON-serializable values"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def validate_manifest(raw: Any) -> dict[str, Any]:
    """Validate and normalize one external-backend conformance manifest."""
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 3:
        raise ConformanceError("manifest must be an object with schema_version=3")
    allowed = {
        "schema_version",
        "backend",
        "backend_version",
        "harness_revision",
        "execution",
        "experiment",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConformanceError("unknown manifest fields: " + ", ".join(unknown))
    backend = raw.get("backend")
    if backend not in _BACKEND_FACTORIES:
        raise ConformanceError(
            "backend must be one of: " + ", ".join(sorted(_BACKEND_FACTORIES))
        )
    version = raw.get("backend_version")
    if not isinstance(version, str) or not version.strip():
        raise ConformanceError("backend_version must be a non-empty string")
    revision = raw.get("harness_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ConformanceError("harness_revision must be a full lowercase git commit")
    execution = raw.get("execution")
    if not isinstance(execution, Mapping):
        raise ConformanceError("execution must be an object")
    execution_unknown = sorted(set(execution) - _EXECUTION_FIELDS)
    execution_missing = sorted(_EXECUTION_FIELDS - set(execution))
    if execution_unknown:
        raise ConformanceError(
            "unknown execution fields: " + ", ".join(execution_unknown)
        )
    if execution_missing:
        raise ConformanceError(
            "missing execution fields: " + ", ".join(execution_missing)
        )
    provider = execution.get("provider")
    if not isinstance(provider, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9-]{0,63}", provider
    ):
        raise ConformanceError("execution.provider must be a lowercase provider id")
    for field in ("provider_tier", "gpu_name"):
        value = execution.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ConformanceError(f"execution.{field} must be a non-empty string")
    image = execution.get("container_image")
    if not isinstance(image, str) or _IMAGE_DIGEST.fullmatch(image) is None:
        raise ConformanceError(
            "execution.container_image must use an immutable @sha256 digest"
        )
    for field in (
        "gpu_count",
        "container_disk_gb",
        "timeout_seconds",
        "max_lifetime_seconds",
    ):
        value = execution.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConformanceError(f"execution.{field} must be a positive integer")
    if execution["max_lifetime_seconds"] < execution["timeout_seconds"]:
        raise ConformanceError(
            "execution.max_lifetime_seconds must be at least timeout_seconds"
        )
    max_cost = execution.get("max_cost_usd")
    if (
        isinstance(max_cost, bool)
        or not isinstance(max_cost, (int, float))
        or not math.isfinite(float(max_cost))
        or max_cost <= 0
    ):
        raise ConformanceError("execution.max_cost_usd must be finite and positive")
    experiment = raw.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ConformanceError("experiment must be an object")
    experiment_unknown = sorted(set(experiment) - _EXPERIMENT_FIELDS)
    missing = sorted(
        {
            "algorithm",
            "model",
            "model_revision",
            "dataset_uri",
            "dataset_sha256",
            "seed",
            "config",
        }
        - set(experiment)
    )
    if experiment_unknown:
        raise ConformanceError(
            "unknown experiment fields: " + ", ".join(experiment_unknown)
        )
    if missing:
        raise ConformanceError("missing experiment fields: " + ", ".join(missing))
    canonical_digest(raw)
    return dict(raw)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate one external-backend conformance manifest."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConformanceError(f"{path}: invalid JSON") from exc
    return validate_manifest(raw)


def build_experiment(
    manifest: Mapping[str, Any], output_dir: Path
) -> TrainingExperiment:
    """Create the protocol object while keeping output location out of semantics."""
    value = manifest["experiment"]
    if not isinstance(value, Mapping):  # defensive for programmatic callers
        raise ConformanceError("experiment must be an object")
    try:
        return TrainingExperiment(
            algorithm=str(value["algorithm"]),
            model=str(value["model"]),
            model_revision=str(value["model_revision"]),
            dataset_uri=str(value["dataset_uri"]),
            dataset_sha256=str(value["dataset_sha256"]),
            output_dir=output_dir,
            seed=value["seed"],
            config=value["config"],
            task=str(value.get("task", "conformance")),
            environment=value.get("environment", {}),
            reward=value.get("reward", {}),
            requirements=frozenset(value.get("requirements", ())),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConformanceError(f"invalid experiment: {exc}") from exc


def hash_artifact(path: Path) -> str:
    """Hash artifact names and bytes deterministically."""
    resolved = path.resolve()
    if not resolved.exists():
        raise ConformanceError(f"artifact does not exist: {resolved}")
    digest = hashlib.sha256()
    files = (
        [resolved]
        if resolved.is_file()
        else sorted(
            candidate for candidate in resolved.rglob("*") if candidate.is_file()
        )
    )
    if not files:
        raise ConformanceError(f"artifact is empty: {resolved}")
    for candidate in files:
        relative = (
            candidate.name
            if resolved.is_file()
            else candidate.relative_to(resolved).as_posix()
        )
        encoded = relative.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_artifact_uri(value: Any) -> str:
    """Require an evidence-relative, portable artifact location."""
    if not isinstance(value, str) or not value.strip():
        raise ConformanceError("evidence artifact_uri must be non-empty")
    if "\\" in value:
        raise ConformanceError("evidence artifact_uri must use POSIX separators")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative == PurePosixPath(".")
        or ".." in relative.parts
    ):
        raise ConformanceError(
            "evidence artifact_uri must be a safe path relative to the evidence file"
        )
    return relative.as_posix()


def resolve_evidence_artifact(evidence_path: Path, artifact_uri: Any) -> Path:
    """Resolve an evidence artifact without allowing directory traversal."""
    relative = validate_artifact_uri(artifact_uri)
    root = evidence_path.resolve().parent
    artifact = (root / relative).resolve()
    if not artifact.is_relative_to(root):  # defensive against unusual path semantics
        raise ConformanceError("evidence artifact escaped its evidence directory")
    return artifact


def collect_nvidia_hardware() -> dict[str, Any]:
    """Collect exact NVIDIA identities and fail when no live GPU is visible."""
    command = [
        _required_executable("nvidia-smi"),
        "--query-gpu=name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(  # nosec B603
            command, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConformanceError(f"could not inspect NVIDIA hardware: {exc}") from exc
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not rows:
        detail = completed.stderr.strip() or "no GPUs reported"
        raise ConformanceError(f"NVIDIA GPU preflight failed: {detail}")
    gpus: list[dict[str, Any]] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 4:
            raise ConformanceError(f"unexpected nvidia-smi row: {row!r}")
        try:
            memory_mb = int(parts[2])
        except ValueError as exc:
            raise ConformanceError(f"invalid GPU memory value: {parts[2]!r}") from exc
        gpus.append(
            {
                "name": parts[0],
                "uuid": parts[1],
                "memory_total_mb": memory_mb,
                "driver_version": parts[3],
            }
        )
    return {"gpu_count": len(gpus), "gpus": gpus}


def validate_hardware_contract(
    hardware: Mapping[str, Any], execution: Mapping[str, Any]
) -> None:
    """Fail before training when visible accelerators differ from the manifest."""
    gpus = hardware.get("gpus")
    if hardware.get("gpu_count") != execution["gpu_count"]:
        raise ConformanceError("visible GPU count does not match execution contract")
    if not isinstance(gpus, list) or any(
        not isinstance(gpu, Mapping) or gpu.get("name") != execution["gpu_name"]
        for gpu in gpus
    ):
        raise ConformanceError("visible GPU name does not match execution contract")


def verify_harness_revision(expected: str, root: Path) -> None:
    """Require the running checkout to be clean and at the declared commit."""
    resolved = root.resolve()
    git = _required_executable("git")
    status = subprocess.run(  # nosec B603
        [git, "status", "--porcelain"],
        cwd=resolved,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if status.returncode != 0:
        raise ConformanceError("could not inspect harness git status")
    if status.stdout.strip():
        raise ConformanceError("conformance harness worktree must be clean")
    revision = subprocess.run(  # nosec B603
        [git, "rev-parse", "HEAD"],
        cwd=resolved,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if revision.returncode != 0 or revision.stdout.strip() != expected:
        raise ConformanceError("running harness revision does not match manifest")


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    """Write evidence without replacing an earlier run record."""
    if path.exists():
        raise ConformanceError(f"refusing to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_evidence(evidence: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Reject incomplete or self-inconsistent conformance evidence."""
    manifest = validate_manifest(manifest)
    required = {
        "schema_version",
        "kind",
        "status",
        "measured",
        "backend",
        "backend_version",
        "stateset_agents_version",
        "harness_revision",
        "manifest",
        "manifest_sha256",
        "experiment_sha256",
        "started_at",
        "completed_at",
        "wall_time_seconds",
        "hardware",
        "runtime",
        "artifact_uri",
        "artifact_sha256",
        "backend_metrics",
        "backend_metadata",
        "execution",
    }
    if set(evidence) != required:
        missing = sorted(required - set(evidence))
        unknown = sorted(set(evidence) - required)
        raise ConformanceError(
            f"evidence schema mismatch; missing={missing}, unknown={unknown}"
        )
    if (
        evidence.get("schema_version") != 3
        or evidence.get("kind") != "stateset-external-backend-conformance"
        or evidence.get("status") != "completed"
        or evidence.get("measured") is not True
    ):
        raise ConformanceError("evidence does not report measured completion")
    for field in ("backend", "backend_version", "harness_revision"):
        if evidence.get(field) != manifest.get(field):
            raise ConformanceError(f"evidence {field} does not match manifest")
    if evidence.get("execution") != manifest.get("execution"):
        raise ConformanceError("evidence execution does not match manifest")
    embedded_manifest = evidence.get("manifest")
    if not isinstance(embedded_manifest, Mapping):
        raise ConformanceError("evidence manifest must be an object")
    validated_embedded = validate_manifest(embedded_manifest)
    if validated_embedded != manifest:
        raise ConformanceError("embedded evidence manifest does not match")
    if evidence.get("manifest_sha256") != canonical_digest(manifest):
        raise ConformanceError("evidence manifest digest does not match")
    for field in ("experiment_sha256", "artifact_sha256"):
        if not isinstance(evidence.get(field), str) or not re.fullmatch(
            r"[0-9a-f]{64}", str(evidence[field])
        ):
            raise ConformanceError(f"evidence {field} must be a SHA-256 digest")
    expected_experiment_digest = build_experiment(manifest, Path(".")).sha256
    if evidence.get("experiment_sha256") != expected_experiment_digest:
        raise ConformanceError("evidence experiment digest does not match manifest")
    if (
        not isinstance(evidence.get("stateset_agents_version"), str)
        or not str(evidence["stateset_agents_version"]).strip()
    ):
        raise ConformanceError("evidence stateset_agents_version must be non-empty")
    validate_artifact_uri(evidence.get("artifact_uri"))
    duration = evidence.get("wall_time_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or duration <= 0
    ):
        raise ConformanceError("evidence wall_time_seconds must be positive")
    for field in ("started_at", "completed_at"):
        value = evidence.get(field)
        if not isinstance(value, str):
            raise ConformanceError(f"evidence {field} must be an ISO timestamp")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ConformanceError(
                f"evidence {field} must be an ISO timestamp"
            ) from exc
        if parsed.tzinfo is None:
            raise ConformanceError(f"evidence {field} must include a timezone")
    hardware = evidence.get("hardware")
    if not isinstance(hardware, Mapping):
        raise ConformanceError("evidence hardware must be an object")
    gpu_count = hardware.get("gpu_count")
    gpus = hardware.get("gpus")
    if (
        isinstance(gpu_count, bool)
        or not isinstance(gpu_count, int)
        or gpu_count < 1
        or not isinstance(gpus, list)
        or len(gpus) != gpu_count
    ):
        raise ConformanceError("evidence hardware must contain every visible GPU")
    uuids: list[str] = []
    for gpu in gpus:
        if not isinstance(gpu, Mapping):
            raise ConformanceError("each evidence GPU must be an object")
        if set(gpu) != {"name", "uuid", "memory_total_mb", "driver_version"}:
            raise ConformanceError("evidence GPU schema is invalid")
        for field in ("name", "uuid", "driver_version"):
            if not isinstance(gpu.get(field), str) or not str(gpu[field]).strip():
                raise ConformanceError(f"evidence GPU {field} must be non-empty")
        memory = gpu.get("memory_total_mb")
        if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
            raise ConformanceError("evidence GPU memory_total_mb must be positive")
        uuids.append(str(gpu["uuid"]))
    if len(uuids) != len(set(uuids)):
        raise ConformanceError("evidence GPU UUIDs must be unique")
    validate_hardware_contract(hardware, manifest["execution"])
    runtime = evidence.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {"python", "platform"}:
        raise ConformanceError("evidence runtime schema is invalid")
    if any(not isinstance(value, str) or not value for value in runtime.values()):
        raise ConformanceError("evidence runtime values must be non-empty")
    metrics = evidence.get("backend_metrics")
    if not isinstance(metrics, Mapping) or metrics.get("completed") != 1.0:
        raise ConformanceError("backend metrics must attest completed=1.0")
    backend_duration = metrics.get("wall_time_seconds")
    if (
        isinstance(backend_duration, bool)
        or not isinstance(backend_duration, (int, float))
        or backend_duration < 0
    ):
        raise ConformanceError("backend wall_time_seconds must be non-negative")
    if not isinstance(evidence.get("backend_metadata"), Mapping):
        raise ConformanceError("backend_metadata must be an object")


def load_evidence(path: Path, *, verify_artifact: bool = True) -> dict[str, Any]:
    """Reload evidence, validate its embedded manifest, and rehash its artifact."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConformanceError(f"{path}: invalid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ConformanceError(f"{path}: evidence must be an object")
    manifest = validate_manifest(raw.get("manifest"))
    validate_evidence(raw, manifest)
    if verify_artifact:
        artifact = resolve_evidence_artifact(path, raw["artifact_uri"])
        if hash_artifact(artifact) != raw["artifact_sha256"]:
            raise ConformanceError("evidence artifact digest does not match bytes")
    return dict(raw)


def run_conformance(
    manifest: Mapping[str, Any],
    output_dir: Path,
    timeout_seconds: int | None,
    root: Path,
) -> dict[str, Any]:
    """Execute one backend and return completed, artifact-bound evidence."""
    manifest = validate_manifest(manifest)
    declared_timeout = int(manifest["execution"]["timeout_seconds"])
    if timeout_seconds is not None and timeout_seconds != declared_timeout:
        raise ConformanceError(
            "CLI timeout_seconds must equal execution.timeout_seconds"
        )
    timeout_seconds = declared_timeout
    verify_harness_revision(str(manifest["harness_revision"]), root)
    hardware = collect_nvidia_hardware()
    validate_hardware_contract(hardware, manifest["execution"])
    backend_name = str(manifest["backend"])
    backend_version = str(manifest["backend_version"])
    backend = _BACKEND_FACTORIES[backend_name](
        version=backend_version, timeout_seconds=timeout_seconds
    )
    experiment = build_experiment(manifest, output_dir / "run")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    result = backend.run(experiment)
    duration = max(
        time.monotonic() - started,
        time.get_clock_info("monotonic").resolution,
    )
    output_root = output_dir.resolve()
    artifact = Path(result.artifact_uri).resolve()
    if not artifact.is_relative_to((output_root / "run").resolve()):
        raise ConformanceError("backend artifact escaped the conformance run directory")
    artifact_uri = artifact.relative_to(output_root).as_posix()
    evidence = {
        "schema_version": 3,
        "kind": "stateset-external-backend-conformance",
        "status": "completed",
        "measured": True,
        "backend": backend_name,
        "backend_version": backend_version,
        "stateset_agents_version": importlib.metadata.version("stateset-agents"),
        "harness_revision": manifest["harness_revision"],
        "execution": dict(manifest["execution"]),
        "manifest": dict(manifest),
        "manifest_sha256": canonical_digest(manifest),
        "experiment_sha256": experiment.sha256,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": duration,
        "hardware": hardware,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "artifact_uri": artifact_uri,
        "artifact_sha256": hash_artifact(artifact),
        "backend_metrics": dict(result.metrics),
        "backend_metadata": dict(result.metadata),
    }
    validate_evidence(evidence, manifest)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point that always retains a success or failure record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="must equal the immutable manifest execution timeout when supplied",
    )
    parser.add_argument(
        "--validate-evidence",
        type=Path,
        help="validate an existing conformance file and its artifact, then exit",
    )
    args = parser.parse_args(argv)
    if args.validate_evidence is not None:
        if args.manifest is not None or args.output_dir is not None:
            parser.error("--validate-evidence cannot be combined with a run")
        try:
            load_evidence(args.validate_evidence)
        except ConformanceError as exc:
            print(f"conformance evidence rejected: {exc}", file=sys.stderr)
            return 2
        print(args.validate_evidence)
        return 0
    if args.manifest is None or args.output_dir is None:
        parser.error("a manifest and --output-dir are required for a run")
    manifest: dict[str, Any] | None = None
    try:
        manifest = load_manifest(args.manifest)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        evidence = run_conformance(
            manifest, args.output_dir, args.timeout_seconds, args.root
        )
        write_json_once(args.output_dir / "conformance.json", evidence)
    except Exception as exc:
        failure = {
            "schema_version": 3,
            "kind": "stateset-external-backend-conformance",
            "status": "failed",
            "measured": True,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if manifest is not None:
            failure.update(
                {
                    "backend": manifest["backend"],
                    "backend_version": manifest["backend_version"],
                    "harness_revision": manifest["harness_revision"],
                    "execution": manifest["execution"],
                    "manifest_sha256": canonical_digest(manifest),
                }
            )
        try:
            write_json_once(args.output_dir / "failure.json", failure)
        except (ConformanceError, OSError):
            pass
        print(f"conformance failed: {exc}", file=sys.stderr)
        return 2
    print(args.output_dir / "conformance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
