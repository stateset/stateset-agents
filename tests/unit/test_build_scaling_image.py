"""Contracts for the immutable scaling image builder."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_scaling_image", ROOT / "scripts/build_scaling_image.py"
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _args(tmp_path: Path, *extra: str) -> Any:
    dockerfile = tmp_path / "Dockerfile.scaling"
    dockerfile.write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n", encoding="utf-8")
    return builder.parse_args(
        [
            "--base-image",
            "registry.example/pytorch@sha256:" + "a" * 64,
            "--image",
            "registry.example:5000/stateset/scaling:0.54.0",
            "--dockerfile",
            str(dockerfile),
            "--output",
            str(tmp_path / "attestation.json"),
            *extra,
        ]
    )


def test_request_rejects_mutable_base_and_requires_exact_push_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.base_image = "registry.example/pytorch:latest"
    with pytest.raises(builder.ScalingImageError, match="pinned"):
        builder.validate_request(args)

    args = _args(tmp_path, "--execute", "--confirm-push", "wrong:image")
    monkeypatch.setattr(builder, "_version", lambda _root: "0.54.0")
    monkeypatch.setattr(
        builder,
        "_git",
        lambda _root, *command: "a" * 40 if command[0] == "rev-parse" else "",
    )
    with pytest.raises(builder.ScalingImageError, match="exactly equal"):
        builder.validate_request(args, root=tmp_path)


def test_build_binds_buildkit_digest_source_and_supply_chain_requests(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    identity = {
        "commit": "b" * 40,
        "version": "0.54.0",
        "base_digest": "a" * 64,
        "dockerfile_sha256": "c" * 64,
    }
    observed: list[str] = []

    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        observed.extend(command)
        if "--metadata-file" in command:
            metadata = Path(command[command.index("--metadata-file") + 1])
            metadata.write_text(
                json.dumps({"containerimage.digest": "sha256:" + "d" * 64}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "", "")
        field = command[-1].removeprefix("{{json .").removesuffix("}}")
        values = {
            "Provenance": {
                "SLSA": {
                    "buildType": "buildkit",
                    "materials": ["a" * 64, "b" * 40],
                }
            },
            "SBOM": {
                "SPDX": {
                    "SPDXID": "SPDXRef-DOCUMENT",
                    "spdxVersion": "SPDX-2.3",
                }
            },
            "Image": {
                "config": {
                    "Labels": {
                        "org.opencontainers.image.revision": "b" * 40,
                        "org.opencontainers.image.version": "0.54.0",
                        "ai.stateset.image.purpose": "distributed-scaling-evidence",
                    }
                }
            },
            "Manifest": {"digest": "sha256:" + "d" * 64},
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(values[field]), "")

    attestation = builder.build(args, identity, runner=run, root=tmp_path)
    assert attestation["resolved_image"] == (
        "registry.example:5000/stateset/scaling@sha256:" + "d" * 64
    )
    assert attestation["source_commit"] == "b" * 40
    assert attestation["base_image_digest"] == "a" * 64
    assert attestation["provenance_mode"] == "max"
    assert attestation["sbom_requested"] is True
    assert attestation["registry_attestations_verified"] is True
    assert set(attestation["registry_inspection"]) == {
        "provenance",
        "sbom",
        "image",
        "manifest",
    }
    assert "--push" in observed
    assert "--provenance=mode=max" in observed
    assert "--sbom=true" in observed


def test_build_fails_when_registry_drops_attestation(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    identity = {
        "commit": "b" * 40,
        "version": "0.54.0",
        "base_digest": "a" * 64,
        "dockerfile_sha256": "c" * 64,
    }

    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        if "--metadata-file" in command:
            metadata = Path(command[command.index("--metadata-file") + 1])
            metadata.write_text(
                json.dumps({"containerimage.digest": "sha256:" + "d" * 64}),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "null", "")

    with pytest.raises(builder.ScalingImageError, match="no retained Provenance"):
        builder.build(args, identity, runner=run, root=tmp_path)
