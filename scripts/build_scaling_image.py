#!/usr/bin/env python3
"""Build and attest the immutable image used by scaling evidence.

The default is a read-only plan. ``--execute`` performs one external registry
push and therefore requires an exact ``--confirm-push`` value. BuildKit is
asked to publish maximal provenance and an SBOM; its returned image digest is
bound to the source commit, package version, base digest, and Dockerfile hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCKERFILE = ROOT / "deployment/docker/Dockerfile.scaling"
HEX = frozenset("0123456789abcdef")


class ScalingImageError(ValueError):
    """Raised when an image cannot provide immutable scaling provenance."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: str, label: str) -> str:
    digest = value.rsplit("@sha256:", 1)[-1]
    if (
        "@sha256:" not in value
        or len(digest) != 64
        or digest == "0" * 64
        or any(char not in HEX for char in digest)
    ):
        raise ScalingImageError(f"{label} must be pinned by a nonzero sha256 digest")
    return digest


def _repository(image: str) -> str:
    """Strip a tag from an OCI name without confusing a registry port."""
    if "@" in image:
        image = image.split("@", 1)[0]
    slash = image.rfind("/")
    colon = image.rfind(":")
    return image[:colon] if colon > slash else image


def _version(root: Path) -> str:
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    package = (root / "stateset_agents/__init__.py").read_text(encoding="utf-8")
    project_match = re.search(r'^version = "(\d+\.\d+\.\d+)"$', pyproject, re.M)
    package_match = re.search(r'^__version__ = "(\d+\.\d+\.\d+)"$', package, re.M)
    if not project_match or not package_match:
        raise ScalingImageError("could not resolve package version")
    if project_match.group(1) != package_match.group(1):
        raise ScalingImageError("package version surfaces do not agree")
    return project_match.group(1)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ScalingImageError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def validate_request(args: argparse.Namespace, root: Path = ROOT) -> dict[str, str]:
    """Validate immutable inputs and return their resolved identities."""
    base_digest = _digest(args.base_image, "base image")
    if (
        "@" in args.image
        or not _repository(args.image)
        or _repository(args.image) == args.image
    ):
        raise ScalingImageError("destination image must include a tag, not a digest")
    if not args.dockerfile.is_file() or args.dockerfile.is_symlink():
        raise ScalingImageError("Dockerfile must be a regular non-symlink file")
    try:
        args.dockerfile.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ScalingImageError("Dockerfile must be inside the repository") from exc
    version = _version(root)
    commit = _git(root, "rev-parse", "HEAD")
    if (
        len(commit) != 40
        or commit == "0" * 40
        or any(char not in HEX for char in commit)
    ):
        raise ScalingImageError("repository commit must be a nonzero full SHA")
    if args.execute:
        if args.confirm_push != args.image:
            raise ScalingImageError("--confirm-push must exactly equal --image")
        if _git(root, "status", "--porcelain"):
            raise ScalingImageError("image publication requires a clean checkout")
        if args.output.exists():
            raise ScalingImageError(f"refusing to overwrite attestation: {args.output}")
    return {
        "base_digest": base_digest,
        "commit": commit,
        "version": version,
        "dockerfile_sha256": _sha256(args.dockerfile),
    }


def build(
    args: argparse.Namespace,
    identity: Mapping[str, str],
    *,
    runner: Callable[..., Any] = subprocess.run,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Push through BuildKit and return an exact source-to-image attestation."""

    def inspect(resolved_image: str, field: str) -> Any:
        command = [
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            resolved_image,
            "--format",
            f"{{{{json .{field}}}}}",
        ]
        try:
            result = runner(
                command, cwd=root, capture_output=True, text=True, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ScalingImageError(
                f"could not inspect registry {field}: {exc}"
            ) from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            raise ScalingImageError(f"registry {field} inspection failed: {detail}")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ScalingImageError(
                f"registry {field} inspection returned invalid JSON"
            ) from exc
        if value is None or value == {} or value == []:
            raise ScalingImageError(f"registry has no retained {field} attestation")
        return value

    with tempfile.TemporaryDirectory(prefix="stateset-scaling-image-") as temp:
        metadata_path = Path(temp) / "metadata.json"
        command = [
            "docker",
            "buildx",
            "build",
            "--file",
            str(args.dockerfile),
            "--tag",
            args.image,
            "--build-arg",
            f"BASE_IMAGE={args.base_image}",
            "--build-arg",
            f"SOURCE_COMMIT={identity['commit']}",
            "--build-arg",
            f"PACKAGE_VERSION={identity['version']}",
            "--provenance=mode=max",
            "--sbom=true",
            "--push",
            "--metadata-file",
            str(metadata_path),
            str(root),
        ]
        try:
            result = runner(
                command, cwd=root, capture_output=True, text=True, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ScalingImageError(f"could not execute Docker Buildx: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            raise ScalingImageError(f"Docker Buildx failed: {detail}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ScalingImageError("Buildx did not return valid metadata") from exc
    if not isinstance(metadata, Mapping):
        raise ScalingImageError("Buildx metadata must be an object")
    digest_value = metadata.get("containerimage.digest")
    if not isinstance(digest_value, str):
        raise ScalingImageError("Buildx metadata has no containerimage.digest")
    image_digest = _digest(f"image@{digest_value}", "built image")
    resolved = f"{_repository(args.image)}@sha256:{image_digest}"
    registry_inspection = {
        field.lower(): inspect(resolved, field)
        for field in ("Provenance", "SBOM", "Image", "Manifest")
    }
    provenance_text = json.dumps(registry_inspection["provenance"], sort_keys=True)
    if (
        "SLSA" not in provenance_text
        or "buildType" not in provenance_text
        or identity["commit"] not in provenance_text
        or identity["base_digest"] not in provenance_text
    ):
        raise ScalingImageError("registry SLSA provenance is not source/base bound")
    sbom_text = json.dumps(registry_inspection["sbom"], sort_keys=True)
    if "SPDXRef-DOCUMENT" not in sbom_text or "spdxVersion" not in sbom_text:
        raise ScalingImageError("registry SBOM is not a retained SPDX document")
    image_config = registry_inspection["image"].get("config", {})
    labels = image_config.get("Labels", {}) if isinstance(image_config, Mapping) else {}
    required_labels = {
        "org.opencontainers.image.revision": identity["commit"],
        "org.opencontainers.image.version": identity["version"],
        "ai.stateset.image.purpose": "distributed-scaling-evidence",
    }
    if not isinstance(labels, Mapping) or any(
        labels.get(key) != value for key, value in required_labels.items()
    ):
        raise ScalingImageError("registry image labels do not match source")
    if registry_inspection["manifest"].get("digest") != f"sha256:{image_digest}":
        raise ScalingImageError("registry manifest digest does not match pushed image")
    canonical_metadata = json.dumps(
        metadata, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "schema_version": 1,
        "kind": "stateset-scaling-image-attestation",
        "status": "pushed",
        "source_commit": identity["commit"],
        "framework_version": identity["version"],
        "base_image": args.base_image,
        "base_image_digest": identity["base_digest"],
        "requested_image": args.image,
        "resolved_image": resolved,
        "image_digest": image_digest,
        "dockerfile": str(args.dockerfile.resolve().relative_to(root.resolve())),
        "dockerfile_sha256": identity["dockerfile_sha256"],
        "build_metadata": metadata,
        "build_metadata_sha256": hashlib.sha256(canonical_metadata).hexdigest(),
        "provenance_mode": "max",
        "sbom_requested": True,
        "registry_attestations_verified": True,
        "registry_inspection": registry_inspection,
        "pushed_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--dockerfile", type=Path, default=DEFAULT_DOCKERFILE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/scaling/image-attestation.json"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-push")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        identity = validate_request(args)
        if not args.execute:
            print(
                json.dumps(
                    {
                        "action": "plan-only",
                        "base_image": args.base_image,
                        "destination_image": args.image,
                        **identity,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        attestation = build(args, identity)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(attestation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(attestation["resolved_image"])
    except (OSError, ScalingImageError) as exc:
        print(f"scaling image rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
