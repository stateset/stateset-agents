#!/usr/bin/env python3
"""Fail-closed, evidence-backed A+ publication gate.

This is the single top-level decision surface for leadership claims.  It
revalidates raw evidence through each domain-specific validator and requires
every result to describe one immutable release commit.  Missing evidence is a
failure; diagnostic or historical opt-outs are intentionally unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from . import (
        agent_quality_evidence,
        distributed_async_evidence,
        framework_comparison,
        provider_evidence,
        reliability_evidence,
        run_flagship_matrix,
        scaling_comparison,
    )
except ImportError:
    import agent_quality_evidence  # type: ignore[no-redef]
    import distributed_async_evidence  # type: ignore[no-redef]
    import framework_comparison  # type: ignore[no-redef]
    import provider_evidence  # type: ignore[no-redef]
    import reliability_evidence  # type: ignore[no-redef]
    import run_flagship_matrix  # type: ignore[no-redef]
    import scaling_comparison  # type: ignore[no-redef]


class APlusGateError(ValueError):
    """Raised when evidence cannot support the complete A+ claim."""


REQUIRED_FRAMEWORKS = (
    "stateset-agents-gspo",
    "trl",
    "verl",
    "nemo-rl",
    "openrlhf",
)
STATESET_FRAMEWORK = "stateset-agents-gspo"
HEX = frozenset("0123456789abcdef")
ROOT = Path(__file__).resolve().parents[1]
SCALING_DOCKERFILE = ROOT / "deployment/docker/Dockerfile.scaling"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in HEX for char in value)
        or value == "0" * 40
    ):
        raise APlusGateError(f"{label} must be a nonzero 40-character commit")
    return value


def current_commit(root: Path) -> str:
    """Resolve the immutable repository revision used by the gate."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise APlusGateError("could not resolve the repository commit")
    return _commit(result.stdout.strip(), "repository commit")


def validate_repository_state(root: Path, expected_commit: str) -> None:
    """Require the decision gate itself to run from the claimed clean commit."""
    if current_commit(root) != expected_commit:
        raise APlusGateError("gate checkout does not match the expected commit")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    if status.returncode != 0:
        raise APlusGateError("could not inspect the gate checkout")
    if status.stdout.strip():
        raise APlusGateError("A+ evidence gate requires a clean checkout")


def validate_security_report_payloads(bandit: Any, safety: Any) -> dict[str, int]:
    """Re-evaluate retained Bandit and Safety results fail-closed."""
    if not isinstance(bandit, Mapping):
        raise APlusGateError("Bandit report must be a JSON object")
    errors = bandit.get("errors")
    results = bandit.get("results")
    if not isinstance(errors, list) or not isinstance(results, list):
        raise APlusGateError("Bandit report is missing errors/results arrays")
    if errors:
        raise APlusGateError("Bandit scan contains execution errors")
    if any(not isinstance(item, Mapping) for item in results):
        raise APlusGateError("Bandit findings must be JSON objects")
    blocked = [
        item
        for item in results
        if str(item.get("issue_severity", "")).upper() in {"MEDIUM", "HIGH", "CRITICAL"}
    ]
    if blocked:
        raise APlusGateError(
            "Bandit report contains medium/high/critical security findings"
        )

    if not isinstance(safety, Mapping):
        raise APlusGateError("Safety report must be a JSON object")
    vulnerabilities = safety.get("vulnerabilities")
    scanned = safety.get("scanned_packages")
    if not isinstance(vulnerabilities, list) or not isinstance(scanned, list):
        raise APlusGateError(
            "Safety report is missing vulnerabilities/scanned_packages arrays"
        )
    if vulnerabilities:
        raise APlusGateError("Safety report contains known vulnerabilities")
    return {
        "bandit_findings": len(results),
        "safety_vulnerabilities": len(vulnerabilities),
        "safety_scanned_packages": len(scanned),
    }


def _archive_member_is_safe(name: str) -> bool:
    member = PurePosixPath(name)
    return bool(name) and not member.is_absolute() and ".." not in member.parts


def _validate_distribution_archives(
    wheel_path: Path, sdist_path: Path, expected_version: str
) -> dict[str, Any]:
    """Inspect distribution structure and core metadata without extracting it."""
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            wheel_members = archive.infolist()
            wheel_names = [item.filename for item in wheel_members]
            dist_info = f"stateset_agents-{expected_version}.dist-info"
            if (
                len(wheel_names) != len(set(wheel_names))
                or any(not _archive_member_is_safe(name) for name in wheel_names)
                or any(stat.S_ISLNK(item.external_attr >> 16) for item in wheel_members)
                or "stateset_agents/__init__.py" not in wheel_names
                or f"{dist_info}/WHEEL" not in wheel_names
                or f"{dist_info}/RECORD" not in wheel_names
            ):
                raise APlusGateError("wheel structure is unsafe or incomplete")
            metadata_names = [
                name for name in wheel_names if name == f"{dist_info}/METADATA"
            ]
            all_metadata = [
                name for name in wheel_names if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1 or len(all_metadata) != 1:
                raise APlusGateError(
                    "wheel must contain exactly one bound METADATA file"
                )
            wheel_metadata = Parser().parsestr(
                archive.read(metadata_names[0]).decode("utf-8")
            )
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise APlusGateError("wheel is not a valid readable archive") from exc

    expected_root = f"stateset_agents-{expected_version}"
    try:
        with tarfile.open(sdist_path, mode="r:*") as archive:
            members = archive.getmembers()
            member_names = [member.name for member in members]
            roots = {PurePosixPath(name).parts[0] for name in member_names if name}
            if (
                not member_names
                or len(member_names) != len(set(member_names))
                or any(not _archive_member_is_safe(name) for name in member_names)
                or any(
                    member.issym() or member.islnk() or member.isdev()
                    for member in members
                )
                or roots != {expected_root}
                or f"{expected_root}/stateset_agents/__init__.py" not in member_names
                or f"{expected_root}/PKG-INFO" not in member_names
                or f"{expected_root}/pyproject.toml" not in member_names
            ):
                raise APlusGateError("source archive structure is unsafe or incomplete")
            pkg_info = archive.extractfile(f"{expected_root}/PKG-INFO")
            if pkg_info is None:
                raise APlusGateError("source archive PKG-INFO is unreadable")
            sdist_metadata = Parser().parsestr(pkg_info.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, tarfile.TarError) as exc:
        raise APlusGateError(
            "source distribution is not a valid readable archive"
        ) from exc

    def normalized(value: str | None) -> str:
        return re.sub(r"[-_.]+", "-", str(value or "")).lower()

    for label, metadata in (("wheel", wheel_metadata), ("sdist", sdist_metadata)):
        if (
            normalized(metadata.get("Name")) != "stateset-agents"
            or metadata.get("Version") != expected_version
        ):
            raise APlusGateError(f"{label} package metadata identity mismatch")
    return {
        "wheel_members": len(wheel_names),
        "sdist_members": len(member_names),
        "project": "stateset-agents",
        "version": expected_version,
    }


def validate_release_readiness(
    path: Path, expected_commit: str, expected_version: str
) -> dict[str, Any]:
    """Require a passed publish-readiness summary for the exact release commit."""
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise APlusGateError("publish-readiness summary must not be a symlink")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise APlusGateError(f"{path}: invalid publish-readiness summary") from exc
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema_version") != 2
        or raw.get("kind") != "stateset-publish-readiness-summary"
        or raw.get("status") != "passed"
    ):
        raise APlusGateError(f"{path}: publish readiness did not pass")
    git = raw.get("git")
    if not isinstance(git, Mapping):
        raise APlusGateError(f"{path}: publish readiness has no git identity")
    observed = _commit(git.get("sha"), "publish-readiness git.sha")
    if observed != expected_commit:
        raise APlusGateError(
            f"{path}: publish readiness is for {observed}, expected {expected_commit}"
        )
    if raw.get("framework_version") != expected_version:
        raise APlusGateError(f"{path}: publish-readiness version mismatch")
    expected_checks = {
        "linters",
        "type_checks",
        "api_compatibility",
        "release_governance",
        "agent_quality_contract",
        "tests_with_coverage",
        "security_scans",
        "build",
        "twine_check",
        "isolated_wheel_smoke",
        "working_tree_clean",
    }
    checks = raw.get("checks")
    if (
        not isinstance(checks, list)
        or len(checks) != len(expected_checks)
        or set(checks) != expected_checks
        or raw.get("working_tree_clean") is not True
    ):
        raise APlusGateError(f"{path}: publish-readiness checklist is incomplete")

    def retained_artifacts(key: str, expected_count: int) -> list[dict[str, Any]]:
        values = raw.get(key)
        if not isinstance(values, list) or len(values) != expected_count:
            raise APlusGateError(f"{path}: {key} inventory is incomplete")
        retained: list[dict[str, Any]] = []
        seen: set[Path] = set()
        for value in values:
            if not isinstance(value, Mapping):
                raise APlusGateError(f"{path}: {key} entry is invalid")
            relative = value.get("path")
            digest = value.get("sha256")
            size = value.get("size_bytes")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or not isinstance(digest, str)
                or len(digest) != 64
                or digest == "0" * 64
                or any(char not in HEX for char in digest)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
            ):
                raise APlusGateError(f"{path}: {key} entry identity is invalid")
            candidate = path.parent / relative
            resolved = candidate.resolve()
            try:
                resolved.relative_to(path.parent.resolve())
            except ValueError as exc:
                raise APlusGateError(f"{path}: {key} path escapes bundle") from exc
            if (
                resolved in seen
                or candidate.is_symlink()
                or any(parent.is_symlink() for parent in candidate.parents)
                or not resolved.is_file()
                or resolved.stat().st_size != size
                or _sha256(resolved) != digest
            ):
                raise APlusGateError(f"{path}: {key} artifact verification failed")
            seen.add(resolved)
            retained.append(dict(value))
        return retained

    distributions = retained_artifacts("distributions", 2)
    names = [str(item["path"]) for item in distributions]
    if (
        sum(name.endswith(".whl") for name in names) != 1
        or sum(name.endswith(".tar.gz") for name in names) != 1
    ):
        raise APlusGateError(f"{path}: wheel/source distribution pair is invalid")
    if any(expected_version not in Path(name).name for name in names):
        raise APlusGateError(f"{path}: distribution version is inconsistent")
    wheel_path = path.parent / next(
        str(item["path"])
        for item in distributions
        if str(item["path"]).endswith(".whl")
    )
    sdist_path = path.parent / next(
        str(item["path"])
        for item in distributions
        if str(item["path"]).endswith(".tar.gz")
    )
    distribution_validation = _validate_distribution_archives(
        wheel_path, sdist_path, expected_version
    )
    security_reports = retained_artifacts("security_reports", 2)
    report_names = {Path(str(item["path"])).name for item in security_reports}
    if report_names != {"bandit-report.json", "safety-report.json"}:
        raise APlusGateError(f"{path}: security report pair is invalid")
    security_payloads: dict[str, Any] = {}
    for report in security_reports:
        report_path = path.parent / str(report["path"])
        try:
            security_payloads[report_path.name] = json.loads(
                report_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise APlusGateError(f"{path}: security report is not valid JSON") from exc
    security_validation = validate_security_report_payloads(
        security_payloads["bandit-report.json"],
        security_payloads["safety-report.json"],
    )
    coverage_reports = retained_artifacts("coverage_reports", 1)
    coverage_path = path.parent / str(coverage_reports[0]["path"])
    if coverage_path.name != "coverage.xml":
        raise APlusGateError(f"{path}: coverage report identity is invalid")
    try:
        measured_coverage = (
            float(ET.parse(coverage_path).getroot().attrib["line-rate"]) * 100
        )
        declared_coverage = float(raw.get("coverage_percent", -1.0))
    except (ET.ParseError, KeyError, TypeError, ValueError) as exc:
        raise APlusGateError(f"{path}: coverage report is invalid") from exc
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    threshold_match = re.search(
        r"(?ms)^\[tool\.coverage\.report\].*?^fail_under\s*=\s*([0-9.]+)",
        pyproject,
    )
    if threshold_match is None:
        raise APlusGateError("repository coverage threshold is unavailable")
    threshold = float(threshold_match.group(1))
    if (
        not math.isfinite(measured_coverage)
        or not math.isclose(measured_coverage, declared_coverage, abs_tol=0.01)
        or measured_coverage < threshold
    ):
        raise APlusGateError(f"{path}: coverage is below or inconsistent with gate")
    generated = raw.get("generated_at_unix")
    if isinstance(generated, bool) or not isinstance(generated, int) or generated <= 0:
        raise APlusGateError(f"{path}: generated_at_unix must be positive")
    return {
        "commit": observed,
        "generated_at_unix": generated,
        "distributions": distributions,
        "distribution_validation": distribution_validation,
        "security_reports": security_reports,
        "security_validation": security_validation,
        "coverage_percent": measured_coverage,
    }


def validate_scaling_image_attestation(
    path: Path, expected_commit: str, expected_version: str
) -> dict[str, Any]:
    """Require a pushed, source-bound scaling image with SBOM/provenance requests."""
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise APlusGateError("scaling image attestation must not be a symlink")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise APlusGateError(f"{path}: invalid scaling image attestation") from exc
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema_version") != 1
        or raw.get("kind") != "stateset-scaling-image-attestation"
        or raw.get("status") != "pushed"
    ):
        raise APlusGateError(f"{path}: scaling image was not attestably pushed")
    commit = _commit(raw.get("source_commit"), "scaling image source_commit")
    if commit != expected_commit:
        raise APlusGateError(f"{path}: scaling image commit mismatch")
    if raw.get("framework_version") != expected_version:
        raise APlusGateError(f"{path}: scaling image version mismatch")
    resolved = raw.get("resolved_image")
    digest = raw.get("image_digest")
    if not isinstance(resolved, str) or not isinstance(digest, str):
        raise APlusGateError(f"{path}: scaling image digest is missing")
    observed_digest = resolved.rsplit("@sha256:", 1)[-1]
    if (
        "@sha256:" not in resolved
        or observed_digest != digest
        or len(digest) != 64
        or digest == "0" * 64
        or any(char not in HEX for char in digest)
    ):
        raise APlusGateError(f"{path}: scaling image digest is invalid")
    requested = raw.get("requested_image")
    if not isinstance(requested, str) or "@" in requested:
        raise APlusGateError(f"{path}: requested image tag is invalid")
    slash = requested.rfind("/")
    colon = requested.rfind(":")
    if colon <= slash or requested[:colon] != resolved.split("@", 1)[0]:
        raise APlusGateError(
            f"{path}: requested and resolved image repositories differ"
        )
    pushed_at = raw.get("pushed_at")
    try:
        pushed_time = datetime.fromisoformat(str(pushed_at))
    except ValueError as exc:
        raise APlusGateError(f"{path}: pushed_at is invalid") from exc
    if pushed_time.tzinfo is None or pushed_time > datetime.now(timezone.utc):
        raise APlusGateError(f"{path}: pushed_at must be timezone-aware and not future")
    for key in ("base_image_digest", "dockerfile_sha256", "build_metadata_sha256"):
        value = raw.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or value == "0" * 64
            or any(char not in HEX for char in value)
        ):
            raise APlusGateError(f"{path}: {key} is invalid")
    if raw.get("dockerfile") != "deployment/docker/Dockerfile.scaling":
        raise APlusGateError(f"{path}: scaling Dockerfile identity is invalid")
    if not SCALING_DOCKERFILE.is_file() or SCALING_DOCKERFILE.is_symlink():
        raise APlusGateError("repository scaling Dockerfile is unavailable")
    if _sha256(SCALING_DOCKERFILE) != raw["dockerfile_sha256"]:
        raise APlusGateError(f"{path}: scaling Dockerfile digest mismatch")
    metadata = raw.get("build_metadata")
    if not isinstance(metadata, Mapping):
        raise APlusGateError(f"{path}: Buildx metadata is missing")
    metadata_digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if metadata_digest != raw["build_metadata_sha256"]:
        raise APlusGateError(f"{path}: Buildx metadata digest mismatch")
    if metadata.get("containerimage.digest") != f"sha256:{digest}":
        raise APlusGateError(f"{path}: Buildx image digest mismatch")
    base_image = raw.get("base_image")
    if (
        not isinstance(base_image, str)
        or "@sha256:" not in base_image
        or base_image.rsplit("@sha256:", 1)[-1] != raw["base_image_digest"]
    ):
        raise APlusGateError(f"{path}: base image digest is not bound")
    if raw.get("provenance_mode") != "max" or raw.get("sbom_requested") is not True:
        raise APlusGateError(f"{path}: provenance or SBOM request is missing")
    inspection = raw.get("registry_inspection")
    if raw.get("registry_attestations_verified") is not True or not isinstance(
        inspection, Mapping
    ):
        raise APlusGateError(f"{path}: registry attestations were not verified")
    if set(inspection) != {"provenance", "sbom", "image", "manifest"}:
        raise APlusGateError(f"{path}: registry inspection is incomplete")
    provenance = inspection["provenance"]
    sbom = inspection["sbom"]
    image = inspection["image"]
    manifest = inspection["manifest"]
    if not all(isinstance(value, Mapping) and value for value in inspection.values()):
        raise APlusGateError(f"{path}: registry inspection is malformed")
    provenance_text = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    if (
        expected_commit not in provenance_text
        or raw["base_image_digest"] not in provenance_text
    ):
        raise APlusGateError(f"{path}: SLSA provenance is not source/base bound")
    if "SLSA" not in provenance_text or "buildType" not in provenance_text:
        raise APlusGateError(f"{path}: registry provenance is not SLSA evidence")
    sbom_text = json.dumps(sbom, sort_keys=True, separators=(",", ":"))
    if "SPDXRef-DOCUMENT" not in sbom_text or "spdxVersion" not in sbom_text:
        raise APlusGateError(f"{path}: registry SBOM is not an SPDX document")
    manifest_digest = manifest.get("digest")
    if manifest_digest != f"sha256:{digest}":
        raise APlusGateError(f"{path}: registry manifest digest mismatch")
    config = image.get("config")
    labels = config.get("Labels") if isinstance(config, Mapping) else None
    if not isinstance(labels, Mapping):
        raise APlusGateError(f"{path}: registry image labels are missing")
    required_labels = {
        "org.opencontainers.image.revision": expected_commit,
        "org.opencontainers.image.version": expected_version,
        "ai.stateset.image.purpose": "distributed-scaling-evidence",
    }
    if any(labels.get(key) != value for key, value in required_labels.items()):
        raise APlusGateError(f"{path}: registry image labels do not match source")
    return {"resolved_image": resolved, "image_digest": digest}


def validate_competitive_efficiency(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Require StateSet throughput within 10% of the fastest matched framework."""
    frameworks = summary.get("frameworks")
    if not isinstance(frameworks, Mapping) or STATESET_FRAMEWORK not in frameworks:
        raise APlusGateError(f"comparison is missing {STATESET_FRAMEWORK}")
    means: dict[str, float] = {}
    for name, values in frameworks.items():
        if not isinstance(values, Mapping):
            raise APlusGateError(f"{name}: invalid framework summary")
        throughput = values.get("samples_per_second")
        if not isinstance(throughput, Mapping):
            raise APlusGateError(f"{name}: throughput summary is missing")
        value = throughput.get("mean")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise APlusGateError(f"{name}: mean throughput must be numeric")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise APlusGateError(f"{name}: mean throughput must be finite and positive")
        means[str(name)] = number
    fastest_name, fastest = max(means.items(), key=lambda item: item[1])
    stateset = means[STATESET_FRAMEWORK]
    ratio = stateset / fastest
    if ratio < 0.90:
        raise APlusGateError(
            f"StateSet throughput is {ratio:.1%} of fastest ({fastest_name}); need 90%"
        )
    return {
        "stateset_samples_per_second": stateset,
        "fastest_framework": fastest_name,
        "fastest_samples_per_second": fastest,
        "stateset_to_fastest_ratio": ratio,
    }


def _require_commit(values: Sequence[Any], expected: str, gate: str) -> None:
    observed = {_commit(value, f"{gate} harness commit") for value in values}
    if observed != {expected}:
        raise APlusGateError(
            f"{gate} evidence commits {sorted(observed)} do not match {expected}"
        )


def _require_version(values: Sequence[Any], expected: str, gate: str) -> None:
    observed = {str(value) for value in values}
    if observed != {expected}:
        raise APlusGateError(
            f"{gate} framework versions {sorted(observed)} do not match {expected}"
        )


def _multi_node_scaling(runs: Sequence[Any]) -> dict[str, Any]:
    """Require the largest strong-scaling topology to span physical nodes."""
    summary = scaling_comparison.summarize_scaling(runs)
    if summary.get("scaling_mode") != "strong":
        raise APlusGateError("A+ scaling evidence must use strong scaling")
    if any(
        not str(run.data.get("protocol", "")).endswith("-scaling-v4") for run in runs
    ):
        raise APlusGateError(
            "A+ scaling evidence must use v4 topology- and artifact-bound protocol"
        )
    largest = max(runs, key=lambda run: int(run.data["hardware"]["gpu_count"]))
    hardware = largest.data["hardware"]
    node_count = hardware.get("node_count")
    node_ids = hardware.get("node_ids")
    node_identity_sources = hardware.get("node_identity_sources")
    if (
        isinstance(node_count, bool)
        or not isinstance(node_count, int)
        or node_count < 2
    ):
        raise APlusGateError("largest scaling topology must span at least two nodes")
    if (
        not isinstance(node_ids, list)
        or len(node_ids) != node_count
        or len(set(node_ids)) != node_count
        or any(not isinstance(item, str) or not item.strip() for item in node_ids)
    ):
        raise APlusGateError("multi-node scaling requires one unique ID per node")
    if (
        not isinstance(node_identity_sources, Mapping)
        or set(node_identity_sources) != set(node_ids)
        or set(node_identity_sources.values()) != {"dmi-product-uuid"}
    ):
        raise APlusGateError(
            "A+ physical-node proof requires a DMI product UUID from every node"
        )
    scaling_comparison.validate_scaling_performance(
        summary, min_efficiency=0.70, require_monotonic=True
    )
    return {
        "mode": "strong",
        "node_count": node_count,
        "max_gpu_count": int(largest.data["hardware"]["gpu_count"]),
        "max_topology_efficiency": summary["topologies"][
            str(largest.data["hardware"]["gpu_count"])
        ]["scaling_efficiency"],
    }


def validate_scaling_provider_evidence(
    root: Path,
    runs: Sequence[Any],
    expected_commit: str,
    expected_image: str,
    billing_root: Path | None = None,
) -> dict[str, Any]:
    """Bind every scaling row to one completed provider lifecycle record."""
    if root.is_symlink():
        raise APlusGateError("scaling provider evidence root must not be a symlink")
    paths = sorted(path for path in root.rglob("*.json") if path.is_file())
    if not paths:
        raise APlusGateError("no scaling provider lifecycle records supplied")
    records: list[tuple[Path, Mapping[str, Any]]] = []

    def digest_list(
        record: Mapping[str, Any], key: str, count: int, source: Path
    ) -> list[str]:
        values = record.get(key)
        if not isinstance(values, list) or len(values) != count:
            raise APlusGateError(f"{source}: {key} attestation is incomplete")
        if any(not isinstance(value, str) for value in values):
            raise APlusGateError(f"{source}: {key} attestation is incomplete")
        if len(set(values)) != count or any(
            len(value) != 64
            or value == "0" * 64
            or any(char not in HEX for char in value)
            for value in values
        ):
            raise APlusGateError(f"{source}: {key} attestation is incomplete")
        return values

    for path in paths:
        relative = path.relative_to(root)
        ancestors = [
            root / Path(*relative.parts[:index])
            for index in range(1, len(relative.parts))
        ]
        if path.is_symlink() or any(ancestor.is_symlink() for ancestor in ancestors):
            raise APlusGateError(f"{path}: provider record must not be a symlink")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise APlusGateError(f"{path}: invalid provider record") from exc
        if not isinstance(value, Mapping):
            raise APlusGateError(f"{path}: provider record must be an object")
        records.append((path, value))

    expected = {
        (int(run.data["hardware"]["gpu_count"]), int(run.seed)): run for run in runs
    }
    observed: dict[tuple[int, int], tuple[Path, Mapping[str, Any]]] = {}
    kinds: set[str] = set()
    provider_names: set[str] = set()
    total_cost_usd = 0.0
    for path, record in records:
        if record.get("schema_version") != 1 or record.get("status") != "completed":
            raise APlusGateError(f"{path}: provider lifecycle did not complete")
        kind = str(record.get("kind") or "")
        if kind not in {
            "stateset-runpod-scaling-provider-record",
            "stateset-kubernetes-scaling-provider-record",
        }:
            raise APlusGateError(f"{path}: unsupported scaling provider record kind")
        kinds.add(kind)
        provider = (
            "runpod"
            if kind.startswith("stateset-runpod")
            else str(record.get("provider") or "")
        )
        if provider not in {"runpod", "coreweave", "nebius", "kubernetes"}:
            raise APlusGateError(f"{path}: scaling provider is invalid")
        provider_names.add(provider)
        gpu_count = record.get("gpu_count")
        seed = record.get("seed")
        if (
            isinstance(gpu_count, bool)
            or not isinstance(gpu_count, int)
            or gpu_count < 1
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise APlusGateError(f"{path}: invalid provider topology identity")
        key = (gpu_count, seed)
        if key in observed:
            raise APlusGateError(f"{path}: duplicate scaling provider record {key}")
        observed[key] = (path, record)
    if set(observed) != set(expected):
        raise APlusGateError(
            "scaling provider records do not exactly cover the measured matrix"
        )
    if len(kinds) != 1 or len(provider_names) != 1:
        raise APlusGateError("scaling matrix must use one provider lifecycle protocol")

    for key, run in expected.items():
        path, record = observed[key]
        if record.get("harness_commit") != expected_commit:
            raise APlusGateError(f"{path}: provider record commit mismatch")
        if record.get("output") != run.source.name:
            raise APlusGateError(f"{path}: provider record output mismatch")
        if record.get("config_sha256") != run.data["workload_config_sha256"]:
            raise APlusGateError(f"{path}: provider record config mismatch")
        hardware = run.data["hardware"]
        if record.get("node_count") != hardware["node_count"]:
            raise APlusGateError(f"{path}: provider node count mismatch")
        if record.get("image") != expected_image:
            raise APlusGateError(f"{path}: provider image does not match attestation")
        if record.get("kind") == "stateset-kubernetes-scaling-provider-record":
            if record.get("resources_deleted") is not True:
                raise APlusGateError(f"{path}: Kubernetes resources were not deleted")
            if not record.get("fabric") or not record.get("fabric_resource"):
                raise APlusGateError(f"{path}: Kubernetes fabric is not enforced")
            image = str(record.get("image") or "")
            image_digest = image.rsplit("@sha256:", 1)[-1]
            if (
                "@sha256:" not in image
                or len(image_digest) != 64
                or image_digest == "0" * 64
                or any(char not in HEX for char in image_digest)
            ):
                raise APlusGateError(f"{path}: Kubernetes image is not digest pinned")
            digest_list(record, "node_names_sha256", hardware["node_count"], path)
            digest_list(record, "pod_uids_sha256", hardware["node_count"], path)
        else:
            pods = record.get("pods")
            if (
                record.get("all_terminations_confirmed") is not True
                or not isinstance(pods, list)
                or len(pods) != hardware["node_count"]
                or any(not isinstance(pod, Mapping) for pod in pods)
            ):
                raise APlusGateError(
                    f"{path}: RunPod cleanup attestation is incomplete"
                )
            machine_ids = {str(pod.get("machine_id") or "") for pod in pods}
            if "" in machine_ids or len(machine_ids) != hardware["node_count"]:
                raise APlusGateError(
                    f"{path}: RunPod machine attestation is incomplete"
                )
            computed = 0.0
            for pod in pods:
                try:
                    rate = float(pod.get("authoritative_cost_per_hr_usd", 0.0))
                    lifetime = float(pod.get("lifetime_seconds", 0.0))
                    cost = float(pod.get("estimated_cost_usd", -1.0))
                except (TypeError, ValueError) as exc:
                    raise APlusGateError(
                        f"{path}: RunPod cost fields must be numeric"
                    ) from exc
                if (
                    rate <= 0
                    or lifetime <= 0
                    or not math.isclose(
                        cost, rate * lifetime / 3600, rel_tol=0.0, abs_tol=1e-5
                    )
                ):
                    raise APlusGateError(f"{path}: RunPod cost arithmetic mismatch")
                computed += cost
            if not math.isclose(
                computed,
                float(record.get("total_estimated_cost_usd", -1.0)),
                rel_tol=0.0,
                abs_tol=1e-5,
            ):
                raise APlusGateError(f"{path}: RunPod total cost mismatch")
            total_cost_usd += computed

    provider = next(iter(provider_names))
    lifecycle_protocol = next(iter(kinds))
    if lifecycle_protocol == "stateset-kubernetes-scaling-provider-record":
        if billing_root is None:
            raise APlusGateError(
                "Kubernetes scaling requires provider billing-export evidence"
            )
        total_cost_usd = validate_kubernetes_billing_evidence(
            billing_root,
            [record for _, record in records],
            expected_commit,
            provider,
        )["total_cost_usd"]
    try:
        total_steps = sum(int(run.data["config"]["measured_steps"]) for run in runs)
    except (KeyError, TypeError, ValueError) as exc:
        raise APlusGateError("scaling measured-step accounting is missing") from exc
    if total_steps <= 0 or total_cost_usd <= 0:
        raise APlusGateError(
            "scaling cost and measured optimizer steps must be positive"
        )
    return {
        "provider": provider,
        "records": len(records),
        "lifecycle_protocol": lifecycle_protocol,
        "total_cost_usd": total_cost_usd,
        "measured_optimizer_steps": total_steps,
        "cost_per_measured_optimizer_step_usd": total_cost_usd / total_steps,
    }


def _timestamp(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise APlusGateError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise APlusGateError(f"{label} must be timezone-aware")
    return parsed


def validate_kubernetes_billing_evidence(
    root: Path,
    lifecycle_records: Sequence[Mapping[str, Any]],
    expected_commit: str,
    expected_provider: str,
) -> dict[str, Any]:
    """Validate one raw-export-bound provider bill for every Kubernetes Job."""
    if root.is_symlink():
        raise APlusGateError("Kubernetes billing root must not be a symlink")
    paths = sorted(path for path in root.rglob("*.billing.json") if path.is_file())
    expected = {str(record["job"]): record for record in lifecycle_records}
    if len(expected) != len(lifecycle_records):
        raise APlusGateError("Kubernetes lifecycle Job identities are not unique")
    observed: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    allocation_ids: set[str] = set()
    line_item_ids: set[str] = set()
    total = 0.0
    for path in paths:
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise APlusGateError(f"{path}: billing evidence must not be a symlink")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise APlusGateError(
                f"{path}: invalid Kubernetes billing evidence"
            ) from exc
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != 1
            or value.get("kind") != "stateset-kubernetes-scaling-billing"
            or value.get("status") not in {"settled", "provider-exported"}
        ):
            raise APlusGateError(f"{path}: unsupported Kubernetes billing evidence")
        job = str(value.get("job") or "")
        if not job or job in observed:
            raise APlusGateError(f"{path}: duplicate or empty billed Job")
        observed[job] = (path, value)
    if set(observed) != set(expected):
        raise APlusGateError(
            "Kubernetes billing evidence does not exactly cover scaling Jobs"
        )
    for job, lifecycle in expected.items():
        path, bill = observed[job]
        allocation_id = str(bill.get("allocation_id") or "")
        if (
            bill.get("provider") != expected_provider
            or bill.get("harness_commit") != expected_commit
            or bill.get("namespace") != lifecycle.get("namespace")
            or bill.get("gpu_count") != lifecycle.get("gpu_count")
            or bill.get("seed") != lifecycle.get("seed")
            or bill.get("currency") != "USD"
            or not str(bill.get("billing_export_id") or "").strip()
            or not allocation_id
            or allocation_id in allocation_ids
        ):
            raise APlusGateError(f"{path}: billing identity does not match lifecycle")
        allocation_ids.add(allocation_id)
        window = bill.get("billing_window")
        if not isinstance(window, Mapping):
            raise APlusGateError(f"{path}: billing window is missing")
        window_start = _timestamp(window.get("started_at"), f"{path} window start")
        window_end = _timestamp(window.get("finished_at"), f"{path} window finish")
        lifecycle_start = _timestamp(
            lifecycle.get("started_at"), f"{path} lifecycle start"
        )
        lifecycle_end = _timestamp(
            lifecycle.get("finished_at"), f"{path} lifecycle finish"
        )
        if window_start > lifecycle_start or window_end < lifecycle_end:
            raise APlusGateError(f"{path}: billing window does not cover Job lifecycle")
        items = bill.get("line_items")
        if not isinstance(items, list) or not items:
            raise APlusGateError(f"{path}: billing line items are missing")
        item_total = 0.0
        for item in items:
            line_item_id = (
                str(item.get("provider_line_item_id") or "")
                if isinstance(item, Mapping)
                else ""
            )
            if (
                not isinstance(item, Mapping)
                or not str(item.get("meter") or "").strip()
                or not line_item_id
                or line_item_id in line_item_ids
            ):
                raise APlusGateError(f"{path}: billing line item is invalid")
            line_item_ids.add(line_item_id)
            try:
                quantity = float(item.get("quantity", 0.0))
                cost = float(item.get("cost_usd", -1.0))
            except (TypeError, ValueError) as exc:
                raise APlusGateError(f"{path}: billing values must be numeric") from exc
            if (
                not math.isfinite(quantity)
                or quantity <= 0
                or not math.isfinite(cost)
                or cost < 0
            ):
                raise APlusGateError(f"{path}: billing values are invalid")
            item_total += cost
        try:
            declared_total = float(bill.get("total_cost_usd", -1.0))
        except (TypeError, ValueError) as exc:
            raise APlusGateError(f"{path}: total cost must be numeric") from exc
        if declared_total <= 0 or not math.isclose(
            item_total, declared_total, rel_tol=0.0, abs_tol=1e-6
        ):
            raise APlusGateError(f"{path}: billing line-item arithmetic mismatch")
        raw_path_value = bill.get("raw_export_path")
        raw_digest = bill.get("raw_export_sha256")
        if not isinstance(raw_path_value, str) or not isinstance(raw_digest, str):
            raise APlusGateError(f"{path}: raw billing export identity is missing")
        if Path(raw_path_value).is_absolute():
            raise APlusGateError(f"{path}: raw billing export path must be relative")
        raw_candidate = path.parent / raw_path_value
        raw_path = raw_candidate.resolve()
        try:
            raw_path.relative_to(root.resolve())
        except ValueError as exc:
            raise APlusGateError(f"{path}: raw billing export escapes bundle") from exc
        if (
            not raw_path.is_file()
            or raw_candidate.is_symlink()
            or any(parent.is_symlink() for parent in raw_candidate.parents)
            or len(raw_digest) != 64
            or raw_digest == "0" * 64
            or any(char not in HEX for char in raw_digest)
            or _sha256(raw_path) != raw_digest
        ):
            raise APlusGateError(f"{path}: raw billing export digest mismatch")
        try:
            raw_text = raw_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise APlusGateError(f"{path}: raw billing export must be UTF-8") from exc
        required_raw_ids = {
            job,
            str(bill["billing_export_id"]),
            allocation_id,
            *(str(item["provider_line_item_id"]) for item in items),
        }
        if any(identifier not in raw_text for identifier in required_raw_ids):
            raise APlusGateError(
                f"{path}: billing envelope identity is absent from raw export"
            )
        total += declared_total
    return {"records": len(observed), "total_cost_usd": total}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run every raw-evidence validator and return one complete report."""
    from stateset_agents import __version__

    expected_commit = _commit(args.expected_commit, "expected commit")
    validate_repository_state(Path(__file__).resolve().parents[1], expected_commit)
    gates: dict[str, Any] = {}

    framework_runs = framework_comparison.load_evidence([args.framework_evidence])
    framework_comparison.validate_comparison(
        framework_runs,
        min_seeds=3,
        required_frameworks=REQUIRED_FRAMEWORKS,
        minimum_schema_version=2,
    )
    _require_commit(
        [run.data["harness_commit"] for run in framework_runs],
        expected_commit,
        "framework comparison",
    )
    _require_version(
        [run.data["framework_version"] for run in framework_runs],
        __version__,
        "framework comparison",
    )
    framework_summary = framework_comparison.summarize(framework_runs)
    gates["competitive_frameworks"] = {
        "runs": len(framework_runs),
        "frameworks": list(REQUIRED_FRAMEWORKS),
        "efficiency": validate_competitive_efficiency(framework_summary),
        "provider_cost": framework_comparison.load_provider_cost_bundle(
            args.framework_evidence, framework_runs
        ),
    }

    scaling_runs = scaling_comparison.load_scaling_evidence([args.scaling_evidence])
    scaling_comparison.validate_scaling_comparison(
        scaling_runs, gpu_counts=(1, 2, 4, 8), min_seeds=3
    )
    _require_commit(
        [run.data["harness_commit"] for run in scaling_runs],
        expected_commit,
        "scaling",
    )
    _require_version(
        [run.data["framework_version"] for run in scaling_runs],
        __version__,
        "scaling",
    )
    gates["scaling"] = _multi_node_scaling(scaling_runs)
    gates["scaling_image"] = validate_scaling_image_attestation(
        args.scaling_image_attestation, expected_commit, __version__
    )
    gates["scaling_provider"] = validate_scaling_provider_evidence(
        args.scaling_provider_evidence,
        scaling_runs,
        expected_commit,
        gates["scaling_image"]["resolved_image"],
        args.scaling_billing_evidence,
    )

    reliability_runs = reliability_evidence.load_runs([args.reliability_evidence])
    reliability_evidence.validate_matrix(
        reliability_runs, min_seeds=3, max_data_loss_steps=0
    )
    _require_commit(
        [run["harness_commit"] for run in reliability_runs],
        expected_commit,
        "reliability",
    )
    _require_version(
        [run["framework_version"] for run in reliability_runs],
        __version__,
        "reliability",
    )
    gates["reliability"] = reliability_evidence.summarize(reliability_runs)

    async_runs = distributed_async_evidence.load_runs([args.distributed_async_evidence])
    distributed_async_evidence.validate_matrix(
        async_runs, min_seeds=3, min_nodes=2, min_soak_seconds=43_200.0
    )
    _require_commit(
        [run["harness_commit"] for run in async_runs],
        expected_commit,
        "distributed async",
    )
    _require_version(
        [run["framework_version"] for run in async_runs],
        __version__,
        "distributed async",
    )
    gates["distributed_async"] = distributed_async_evidence.summarize(async_runs)

    provider_reports = provider_evidence.load_reports([args.provider_evidence])
    gates["providers"] = provider_evidence.validate_matrix(
        provider_reports,
        required=provider_evidence.REQUIRED_PROVIDERS,
        max_age_days=30,
        minimum_schema_version=2,
        expected_commit=expected_commit,
        expected_version=__version__,
    )

    quality_runs = agent_quality_evidence.load_runs([args.agent_quality_evidence])
    agent_quality_evidence.validate_matrix(
        quality_runs, min_seeds=3, minimum_mean_improvement=0.03
    )
    _require_commit(
        [run["harness_commit"] for run in quality_runs],
        expected_commit,
        "agent quality",
    )
    _require_version(
        [run["framework_version"] for run in quality_runs],
        __version__,
        "agent quality",
    )
    gates["agent_quality"] = agent_quality_evidence.summarize(quality_runs)

    flagship_manifest = run_flagship_matrix.load_manifest(args.flagship_manifest)
    _require_version([flagship_manifest["framework_version"]], __version__, "flagship")
    flagship_runs = run_flagship_matrix.load_retained_evidence(
        [args.flagship_evidence], flagship_manifest
    )
    _require_commit(
        [run["harness_commit"] for run in flagship_runs],
        expected_commit,
        "flagship",
    )
    gates["flagship"] = run_flagship_matrix.validate_matrix(
        flagship_runs, flagship_manifest
    )

    gates["release_readiness"] = validate_release_readiness(
        args.release_readiness, expected_commit, __version__
    )
    return {
        "schema_version": 1,
        "kind": "stateset-a-plus-evidence-report",
        "passed": True,
        "expected_commit": expected_commit,
        "framework_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--expected-commit", default=current_commit(root))
    parser.add_argument("--framework-evidence", type=Path, required=True)
    parser.add_argument("--scaling-evidence", type=Path, required=True)
    parser.add_argument("--scaling-provider-evidence", type=Path, required=True)
    parser.add_argument("--scaling-image-attestation", type=Path, required=True)
    parser.add_argument("--scaling-billing-evidence", type=Path)
    parser.add_argument("--reliability-evidence", type=Path, required=True)
    parser.add_argument("--distributed-async-evidence", type=Path, required=True)
    parser.add_argument("--provider-evidence", type=Path, required=True)
    parser.add_argument("--agent-quality-evidence", type=Path, required=True)
    parser.add_argument("--flagship-manifest", type=Path, required=True)
    parser.add_argument("--flagship-evidence", type=Path, required=True)
    parser.add_argument("--release-readiness", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/a_plus/report.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = evaluate(args)
    except Exception as exc:  # noqa: BLE001 - retain the failed top-level gate
        report = {
            "schema_version": 1,
            "kind": "stateset-a-plus-evidence-report",
            "passed": False,
            "expected_commit": args.expected_commit,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["passed"]:
        print(f"A+ evidence gate rejected: {report['error']}", file=sys.stderr)
        return 2
    print(f"A+ evidence gate passed for {report['expected_commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
