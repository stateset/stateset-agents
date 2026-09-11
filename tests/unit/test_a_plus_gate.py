"""Contract tests for the top-level A+ evidence decision."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateset_agents import __version__ as PACKAGE_VERSION

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location(
    "a_plus_gate", BENCHMARKS / "a_plus_gate.py"
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _comparison(stateset: float, trl: float) -> dict:
    return {
        "frameworks": {
            gate.STATESET_FRAMEWORK: {"samples_per_second": {"mean": stateset}},
            "trl": {"samples_per_second": {"mean": trl}},
        }
    }


def _distribution_pair(
    root_path: Path, *, metadata_version: str = PACKAGE_VERSION
) -> tuple[Path, Path]:
    wheel = root_path / f"stateset_agents-{PACKAGE_VERSION}-py3-none-any.whl"
    sdist = root_path / f"stateset_agents-{PACKAGE_VERSION}.tar.gz"
    metadata = f"Name: stateset-agents\nVersion: {metadata_version}\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("stateset_agents/__init__.py", "")
        dist_info = f"stateset_agents-{PACKAGE_VERSION}.dist-info"
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", "Wheel-Version: 1.0\n")
        archive.writestr(f"{dist_info}/RECORD", "")
    with tarfile.open(sdist, "w:gz") as archive:
        archive_root = f"stateset_agents-{PACKAGE_VERSION}"
        for name, content in (
            (f"{archive_root}/stateset_agents/__init__.py", b""),
            (f"{archive_root}/PKG-INFO", metadata.encode()),
            (f"{archive_root}/pyproject.toml", b"[build-system]\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return wheel, sdist


def test_competitive_efficiency_requires_90_percent_of_fastest() -> None:
    result = gate.validate_competitive_efficiency(_comparison(9.0, 10.0))
    assert result["stateset_to_fastest_ratio"] == pytest.approx(0.9)

    with pytest.raises(gate.APlusGateError, match="need 90%"):
        gate.validate_competitive_efficiency(_comparison(8.99, 10.0))


def test_release_readiness_is_bound_to_exact_commit(tmp_path: Path) -> None:
    commit = "a" * 40
    path = tmp_path / "readiness.json"
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel, sdist = _distribution_pair(dist)
    bandit = tmp_path / "bandit-report.json"
    safety = tmp_path / "safety-report.json"
    coverage = tmp_path / "coverage.xml"
    for artifact, content in (
        (bandit, b'{"errors": [], "results": []}'),
        (safety, b'{"scanned_packages": [], "vulnerabilities": []}'),
        (coverage, b'<coverage line-rate="0.64"/>'),
    ):
        artifact.write_bytes(content)

    def retained(artifact: Path) -> dict[str, object]:
        return {
            "path": str(artifact.relative_to(tmp_path)),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "size_bytes": artifact.stat().st_size,
        }

    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "stateset-publish-readiness-summary",
                "status": "passed",
                "generated_at_unix": 1,
                "git": {"sha": commit},
                "framework_version": PACKAGE_VERSION,
                "checks": [
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
                ],
                "working_tree_clean": True,
                "distributions": [retained(wheel), retained(sdist)],
                "security_reports": [retained(bandit), retained(safety)],
                "coverage_reports": [retained(coverage)],
                "coverage_percent": 64.0,
            }
        ),
        encoding="utf-8",
    )
    assert (
        gate.validate_release_readiness(path, commit, PACKAGE_VERSION)["commit"]
        == commit
    )
    with pytest.raises(gate.APlusGateError, match="expected"):
        gate.validate_release_readiness(path, "b" * 40, PACKAGE_VERSION)
    wheel.write_bytes(b"tampered")
    with pytest.raises(gate.APlusGateError, match="verification failed"):
        gate.validate_release_readiness(path, commit, PACKAGE_VERSION)


def test_distribution_archive_validation_rejects_unsafe_and_wrong_identity(
    tmp_path: Path,
) -> None:
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    wheel, sdist = _distribution_pair(valid_dir)
    assert (
        gate._validate_distribution_archives(wheel, sdist, PACKAGE_VERSION)["project"]
        == "stateset-agents"
    )

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("../escape.py", "")
    with pytest.raises(gate.APlusGateError, match="unsafe or incomplete"):
        gate._validate_distribution_archives(wheel, sdist, PACKAGE_VERSION)

    mismatch_dir = tmp_path / "mismatch"
    mismatch_dir.mkdir()
    mismatch_wheel, mismatch_sdist = _distribution_pair(
        mismatch_dir, metadata_version="999.0.0"
    )
    with pytest.raises(gate.APlusGateError, match="metadata identity"):
        gate._validate_distribution_archives(
            mismatch_wheel, mismatch_sdist, PACKAGE_VERSION
        )


def test_release_security_reports_are_revalidated_fail_closed() -> None:
    valid = gate.validate_security_report_payloads(
        {"errors": [], "results": []},
        {"scanned_packages": [{"name": "numpy"}], "vulnerabilities": []},
    )
    assert valid == {
        "bandit_findings": 0,
        "safety_vulnerabilities": 0,
        "safety_scanned_packages": 1,
    }

    with pytest.raises(gate.APlusGateError, match="execution errors"):
        gate.validate_security_report_payloads(
            {"errors": ["parse failed"], "results": []},
            {"scanned_packages": [], "vulnerabilities": []},
        )
    with pytest.raises(gate.APlusGateError, match="security findings"):
        gate.validate_security_report_payloads(
            {
                "errors": [],
                "results": [{"issue_severity": "MEDIUM"}],
            },
            {"scanned_packages": [], "vulnerabilities": []},
        )
    with pytest.raises(gate.APlusGateError, match="known vulnerabilities"):
        gate.validate_security_report_payloads(
            {"errors": [], "results": []},
            {
                "scanned_packages": [{"name": "example"}],
                "vulnerabilities": [{"package_name": "example"}],
            },
        )


def test_scaling_image_attestation_binds_source_version_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    path = tmp_path / "image.json"
    metadata = {"containerimage.digest": "sha256:" + "c" * 64}
    metadata_digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    dockerfile = tmp_path / "Dockerfile.scaling"
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    monkeypatch.setattr(gate, "SCALING_DOCKERFILE", dockerfile)
    dockerfile_digest = hashlib.sha256(dockerfile.read_bytes()).hexdigest()
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "stateset-scaling-image-attestation",
                "status": "pushed",
                "source_commit": commit,
                "framework_version": PACKAGE_VERSION,
                "base_image": "registry/base@sha256:" + "b" * 64,
                "base_image_digest": "b" * 64,
                "resolved_image": "registry/scaling@sha256:" + "c" * 64,
                "requested_image": "registry/scaling:0.54.0",
                "image_digest": "c" * 64,
                "dockerfile": "deployment/docker/Dockerfile.scaling",
                "dockerfile_sha256": dockerfile_digest,
                "build_metadata": metadata,
                "build_metadata_sha256": metadata_digest,
                "provenance_mode": "max",
                "sbom_requested": True,
                "registry_attestations_verified": True,
                "registry_inspection": {
                    "provenance": {
                        "SLSA": {
                            "buildType": "buildkit",
                            "materials": [
                                {"digest": {"sha256": "b" * 64}},
                                {"source_commit": commit},
                            ],
                        }
                    },
                    "sbom": {
                        "SPDX": {
                            "SPDXID": "SPDXRef-DOCUMENT",
                            "spdxVersion": "SPDX-2.3",
                        }
                    },
                    "image": {
                        "config": {
                            "Labels": {
                                "org.opencontainers.image.revision": commit,
                                "org.opencontainers.image.version": PACKAGE_VERSION,
                                "ai.stateset.image.purpose": "distributed-scaling-evidence",
                            }
                        }
                    },
                    "manifest": {"digest": "sha256:" + "c" * 64},
                },
                "pushed_at": "2026-09-10T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    assert gate.validate_scaling_image_attestation(path, commit, PACKAGE_VERSION) == {
        "resolved_image": "registry/scaling@sha256:" + "c" * 64,
        "image_digest": "c" * 64,
    }
    value = json.loads(path.read_text(encoding="utf-8"))
    value["sbom_requested"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(gate.APlusGateError, match="SBOM"):
        gate.validate_scaling_image_attestation(path, commit, PACKAGE_VERSION)
    value["sbom_requested"] = True
    value["registry_inspection"]["image"]["config"]["Labels"][
        "org.opencontainers.image.revision"
    ] = ("f" * 40)
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(gate.APlusGateError, match="labels"):
        gate.validate_scaling_image_attestation(path, commit, PACKAGE_VERSION)


def test_repository_state_must_be_clean_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    monkeypatch.setattr(gate, "current_commit", lambda _: commit)
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, " M README.md\n", ""),
    )
    with pytest.raises(gate.APlusGateError, match="clean checkout"):
        gate.validate_repository_state(tmp_path, commit)

    monkeypatch.setattr(gate, "current_commit", lambda _: "b" * 40)
    with pytest.raises(gate.APlusGateError, match="does not match"):
        gate.validate_repository_state(tmp_path, commit)


def test_scaling_provider_records_exactly_bind_matrix_and_cleanup(
    tmp_path: Path,
) -> None:
    commit = "a" * 40
    evidence = tmp_path / "evidence"
    provider = tmp_path / "provider"
    billing = tmp_path / "billing"
    evidence.mkdir()
    provider.mkdir()
    billing.mkdir()
    runs = []
    for count in (1, 2, 4, 8):
        node_count = 2 if count == 8 else 1
        for seed in (42, 1337, 2026):
            source = evidence / f"gpu{count}-seed{seed}.json"
            source.write_text("{}", encoding="utf-8")
            runs.append(
                SimpleNamespace(
                    source=source,
                    seed=seed,
                    data={
                        "workload_config_sha256": "c" * 64,
                        "config": {"measured_steps": 6},
                        "hardware": {
                            "gpu_count": count,
                            "node_count": node_count,
                        },
                    },
                )
            )
            record = {
                "schema_version": 1,
                "kind": "stateset-kubernetes-scaling-provider-record",
                "provider": "coreweave",
                "status": "completed",
                "job": f"stateset-scale-{count}-{seed}",
                "namespace": "stateset-benchmarks",
                "harness_commit": commit,
                "output": source.name,
                "config_sha256": "c" * 64,
                "gpu_count": count,
                "node_count": node_count,
                "seed": seed,
                "started_at": "2026-09-10T10:00:00+00:00",
                "finished_at": "2026-09-10T10:10:00+00:00",
                "resources_deleted": True,
                "fabric": "400-gbps-infiniband",
                "fabric_resource": "rdma/ib",
                "image": "registry/stateset@sha256:" + "d" * 64,
                "node_names_sha256": [
                    f"{index + 1:064x}" for index in range(node_count)
                ],
                "pod_uids_sha256": [
                    f"{index + 101:064x}" for index in range(node_count)
                ],
            }
            (provider / source.name).write_text(json.dumps(record), encoding="utf-8")
            raw = billing / f"gpu{count}-seed{seed}.billing.raw"
            raw.write_text(
                " ".join(
                    (
                        f"export-{count}-{seed}",
                        f"allocation-{count}-{seed}",
                        f"item-{count}-{seed}",
                        record["job"],
                    )
                ),
                encoding="utf-8",
            )
            bill = {
                "schema_version": 1,
                "kind": "stateset-kubernetes-scaling-billing",
                "status": "provider-exported",
                "provider": "coreweave",
                "harness_commit": commit,
                "job": record["job"],
                "namespace": "stateset-benchmarks",
                "gpu_count": count,
                "seed": seed,
                "currency": "USD",
                "billing_export_id": f"export-{count}-{seed}",
                "allocation_id": f"allocation-{count}-{seed}",
                "billing_window": {
                    "started_at": "2026-09-10T09:55:00+00:00",
                    "finished_at": "2026-09-10T10:15:00+00:00",
                },
                "line_items": [
                    {
                        "provider_line_item_id": f"item-{count}-{seed}",
                        "meter": "gpu-node-seconds",
                        "quantity": 600,
                        "cost_usd": 1.0,
                    }
                ],
                "total_cost_usd": 1.0,
                "raw_export_path": raw.name,
                "raw_export_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            }
            (billing / f"gpu{count}-seed{seed}.billing.json").write_text(
                json.dumps(bill), encoding="utf-8"
            )

    image = "registry/stateset@sha256:" + "d" * 64
    result = gate.validate_scaling_provider_evidence(
        provider, runs, commit, image, billing
    )
    assert result == {
        "provider": "coreweave",
        "records": 12,
        "lifecycle_protocol": "stateset-kubernetes-scaling-provider-record",
        "total_cost_usd": 12.0,
        "measured_optimizer_steps": 72,
        "cost_per_measured_optimizer_step_usd": pytest.approx(1 / 6),
    }

    changed = provider / "gpu8-seed42.json"
    record = json.loads(changed.read_text(encoding="utf-8"))
    record["resources_deleted"] = False
    changed.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(gate.APlusGateError, match="were not deleted"):
        gate.validate_scaling_provider_evidence(provider, runs, commit, image, billing)

    record["resources_deleted"] = True
    changed.write_text(json.dumps(record), encoding="utf-8")
    raw = billing / "gpu8-seed42.billing.raw"
    raw.write_text("tampered provider export", encoding="utf-8")
    with pytest.raises(gate.APlusGateError, match="raw billing export digest"):
        gate.validate_scaling_provider_evidence(provider, runs, commit, image, billing)
    bill_path = billing / "gpu8-seed42.billing.json"
    bill = json.loads(bill_path.read_text(encoding="utf-8"))
    bill["raw_export_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    bill_path.write_text(json.dumps(bill), encoding="utf-8")
    with pytest.raises(gate.APlusGateError, match="absent from raw export"):
        gate.validate_scaling_provider_evidence(provider, runs, commit, image, billing)


def test_evaluate_requires_every_gate_on_one_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    monkeypatch.setattr(gate, "validate_repository_state", lambda *_: None)
    framework_run = SimpleNamespace(
        data={"harness_commit": commit, "framework_version": PACKAGE_VERSION}
    )
    monkeypatch.setattr(
        gate.framework_comparison, "load_evidence", lambda _: [framework_run]
    )
    monkeypatch.setattr(
        gate.framework_comparison, "validate_comparison", lambda *a, **k: None
    )
    monkeypatch.setattr(
        gate.framework_comparison,
        "summarize",
        lambda _: _comparison(9.5, 10.0),
    )
    monkeypatch.setattr(
        gate.framework_comparison,
        "load_provider_cost_bundle",
        lambda *_: {"total_cost_usd": 1.0},
    )

    scaling_runs = [
        SimpleNamespace(
            data={
                "harness_commit": commit,
                "framework_version": PACKAGE_VERSION,
                "protocol": "stateset-ddp-policy-strong-scaling-v4",
                "config": {"measured_steps": 6},
                "hardware": {
                    "gpu_count": count,
                    "node_count": 2 if count == 8 else 1,
                    "node_ids": ["node-a", "node-b"] if count == 8 else ["node-a"],
                    "node_identity_sources": (
                        {
                            "node-a": "dmi-product-uuid",
                            "node-b": "dmi-product-uuid",
                        }
                        if count == 8
                        else {"node-a": "dmi-product-uuid"}
                    ),
                },
            }
        )
        for count in (1, 2, 4, 8)
    ]
    monkeypatch.setattr(
        gate.scaling_comparison, "load_scaling_evidence", lambda _: scaling_runs
    )
    monkeypatch.setattr(
        gate.scaling_comparison, "validate_scaling_comparison", lambda *a, **k: None
    )
    monkeypatch.setattr(
        gate.scaling_comparison,
        "summarize_scaling",
        lambda _: {
            "scaling_mode": "strong",
            "topologies": {"8": {"scaling_efficiency": 0.8}},
        },
    )
    monkeypatch.setattr(
        gate.scaling_comparison, "validate_scaling_performance", lambda *a, **k: None
    )
    monkeypatch.setattr(
        gate,
        "validate_scaling_image_attestation",
        lambda *_: {"resolved_image": "registry/scaling@sha256:" + "d" * 64},
    )
    monkeypatch.setattr(
        gate,
        "validate_scaling_provider_evidence",
        lambda *_: {"provider": "coreweave", "records": 12},
    )

    one_run = [{"harness_commit": commit, "framework_version": PACKAGE_VERSION}]
    monkeypatch.setattr(gate.reliability_evidence, "load_runs", lambda _: one_run)
    monkeypatch.setattr(
        gate.reliability_evidence, "validate_matrix", lambda *a, **k: None
    )
    monkeypatch.setattr(
        gate.reliability_evidence, "summarize", lambda _: {"passed": True}
    )
    monkeypatch.setattr(gate.distributed_async_evidence, "load_runs", lambda _: one_run)
    monkeypatch.setattr(
        gate.distributed_async_evidence, "validate_matrix", lambda *a, **k: None
    )
    monkeypatch.setattr(
        gate.distributed_async_evidence, "summarize", lambda _: {"passed": True}
    )
    monkeypatch.setattr(gate.provider_evidence, "load_reports", lambda _: [{}])
    provider_gate_options: dict[str, object] = {}

    def validate_provider_gate(*args, **kwargs):
        provider_gate_options.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(
        gate.provider_evidence, "validate_matrix", validate_provider_gate
    )
    monkeypatch.setattr(gate.agent_quality_evidence, "load_runs", lambda _: one_run)
    monkeypatch.setattr(
        gate.agent_quality_evidence, "validate_matrix", lambda *a, **k: None
    )
    monkeypatch.setattr(
        gate.agent_quality_evidence, "summarize", lambda _: {"passed": True}
    )
    monkeypatch.setattr(
        gate.run_flagship_matrix,
        "load_manifest",
        lambda _: {"framework_version": PACKAGE_VERSION},
    )
    monkeypatch.setattr(
        gate.run_flagship_matrix, "load_retained_evidence", lambda *a: one_run
    )
    monkeypatch.setattr(
        gate.run_flagship_matrix, "validate_matrix", lambda *a: {"passed": True}
    )
    monkeypatch.setattr(
        gate,
        "validate_release_readiness",
        lambda *_: {"commit": commit},
    )

    readiness = tmp_path / "readiness.json"
    readiness.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        expected_commit=commit,
        framework_evidence=tmp_path,
        scaling_evidence=tmp_path,
        scaling_provider_evidence=tmp_path,
        scaling_image_attestation=tmp_path / "image.json",
        scaling_billing_evidence=tmp_path,
        reliability_evidence=tmp_path,
        distributed_async_evidence=tmp_path,
        provider_evidence=tmp_path,
        agent_quality_evidence=tmp_path,
        flagship_manifest=tmp_path / "manifest.json",
        flagship_evidence=tmp_path,
        release_readiness=readiness,
    )
    report = gate.evaluate(args)
    assert report["passed"] is True
    assert provider_gate_options == {
        "required": gate.provider_evidence.REQUIRED_PROVIDERS,
        "max_age_days": 30,
        "minimum_schema_version": 2,
        "expected_commit": commit,
        "expected_version": PACKAGE_VERSION,
    }
    assert set(report["gates"]) == {
        "competitive_frameworks",
        "scaling",
        "scaling_provider",
        "scaling_image",
        "reliability",
        "distributed_async",
        "providers",
        "agent_quality",
        "flagship",
        "release_readiness",
    }

    framework_run.data["harness_commit"] = "b" * 40
    with pytest.raises(gate.APlusGateError, match="do not match"):
        gate.evaluate(args)
