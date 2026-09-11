"""Contracts for provider-neutral high-bandwidth Kubernetes scaling."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from benchmarks import kubernetes_scaling_launcher as launcher


def _args(tmp_path: Path, *extra: str) -> Any:
    workload = tmp_path / "distributed_scaling_workload.py"
    workload.write_text("# workload\n", encoding="utf-8")
    return launcher.parse_args(
        [
            "--gpu-count",
            "8",
            "--node-count",
            "2",
            "--nproc-per-node",
            "4",
            "--seed",
            "42",
            "--harness-commit",
            "a" * 40,
            "--config-json",
            '{"scaling_mode":"strong"}',
            "--command-label",
            "matched strong scaling",
            "--output",
            str(tmp_path / "evidence" / "gpu8-seed42.json"),
            "--workload",
            str(workload),
            "--provider",
            "coreweave",
            "--context",
            "coreweave-prod",
            "--namespace",
            "stateset-benchmarks",
            "--image",
            "registry.example/stateset@sha256:" + "b" * 64,
            "--fabric",
            "400-gbps-infiniband",
            "--fabric-resource",
            "rdma/ib",
            "--node-selector-key",
            "gpu.nvidia.com/class",
            "--node-selector-value",
            "H100",
            *extra,
        ]
    )


def test_indexed_job_has_stable_dns_gpu_shape_and_hard_anti_affinity(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    launcher.validate_request(args)
    service, job = launcher.build_resources(args, "stateset-scale-contract")
    assert service["spec"]["clusterIP"] == "None"
    assert service["spec"]["publishNotReadyAddresses"] is True
    assert job["spec"]["completionMode"] == "Indexed"
    assert job["spec"]["parallelism"] == 2
    pod = job["spec"]["template"]["spec"]
    assert pod["subdomain"] == "stateset-scale-contract-headless"
    assert (
        pod["affinity"]["podAntiAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ][0]["topologyKey"]
        == "kubernetes.io/hostname"
    )
    container = pod["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 4
    assert container["resources"]["limits"]["rdma/ib"] == 1
    script = container["args"][0]
    assert "--nnodes=2" in script
    assert '--node-rank="${JOB_COMPLETION_INDEX}"' in script
    assert "stateset-scale-contract-0.stateset-scale-contract-headless" in script


def test_request_rejects_mutable_image_and_missing_fabric(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.image = "registry.example/stateset:latest"
    with pytest.raises(launcher.KubernetesScalingError, match="sha256"):
        launcher.validate_request(args)
    args = _args(tmp_path)
    args.image = "registry.example/stateset@sha256:" + "0" * 64
    with pytest.raises(launcher.KubernetesScalingError, match="sha256"):
        launcher.validate_request(args)
    args = _args(tmp_path)
    args.fabric = ""
    with pytest.raises(launcher.KubernetesScalingError, match="fabric"):
        launcher.validate_request(args)


def test_kubectl_transport_is_context_namespace_and_stdin_bound() -> None:
    calls: list[tuple[list[str], str | None]] = []

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs.get("input")))
        return subprocess.CompletedProcess(command, 0, "", "")

    kube = launcher.KubernetesApi(
        context="coreweave-prod", namespace="benchmarks", runner=runner
    )
    kube.apply({"apiVersion": "v1", "kind": "Service"})
    command, stdin = calls[0]
    assert command[:5] == [
        "kubectl",
        "--context",
        "coreweave-prod",
        "--namespace",
        "benchmarks",
    ]
    assert command[-3:] == ["apply", "-f", "-"]
    assert json.loads(stdin or "{}")["kind"] == "Service"


class _FailAfterService:
    def __init__(self) -> None:
        self.applied = 0
        self.deleted: list[tuple[str, str]] = []

    def require_permissions(self) -> None:
        pass

    def apply(self, _: dict[str, Any]) -> None:
        self.applied += 1
        if self.applied == 2:
            raise RuntimeError("job admission denied")

    def delete(self, kind: str, name: str) -> None:
        self.deleted.append((kind, name))


def test_job_admission_failure_deletes_already_created_service(tmp_path: Path) -> None:
    args = _args(tmp_path, "--execute")
    kube = _FailAfterService()
    with pytest.raises(RuntimeError, match="admission denied"):
        launcher.execute(args, kube)  # type: ignore[arg-type]
    assert len(kube.deleted) == 1
    assert kube.deleted[0][0] == "service"
    record = args.output.parent.parent / "kubernetes-provider" / args.output.name
    assert json.loads(record.read_text(encoding="utf-8"))["status"] == "failed"


class _SuccessfulKube:
    def __init__(self) -> None:
        self.applied: list[str] = []
        self.deleted: list[str] = []

    def require_permissions(self) -> None:
        pass

    def apply(self, manifest: dict[str, Any]) -> None:
        self.applied.append(manifest["kind"])

    def get_json(self, kind: str, name: str) -> dict[str, Any]:
        return {"status": {"conditions": [{"type": "Complete", "status": "True"}]}}

    def list_pods(self, selector: str) -> list[dict[str, Any]]:
        return [
            {
                "metadata": {
                    "name": f"rank-{rank}",
                    "uid": f"uid-{rank}",
                    "annotations": {
                        "batch.kubernetes.io/job-completion-index": str(rank)
                    },
                },
                "spec": {"nodeName": f"gpu-node-{rank}"},
            }
            for rank in range(2)
        ]

    def copy_from(self, pod: str, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.suffix == ".json":
            local.write_text("{}", encoding="utf-8")
        else:
            local.write_bytes(b"policy")

    def delete(self, kind: str, name: str) -> None:
        self.deleted.append(kind)


def test_success_retrieves_rank_zero_attests_nodes_and_cleans_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, "--execute")
    kube = _SuccessfulKube()
    monkeypatch.setattr(launcher, "load_scaling_evidence", lambda _: [object()])
    assert launcher.execute(args, kube) == args.output  # type: ignore[arg-type]
    assert kube.applied == ["Service", "Job"]
    assert kube.deleted == ["job", "service"]
    record = args.output.parent.parent / "kubernetes-provider" / args.output.name
    retained = json.loads(record.read_text(encoding="utf-8"))
    assert retained["status"] == "completed"
    assert retained["resources_deleted"] is True
    assert retained["started_at"].endswith("+00:00")
    assert retained["finished_at"].endswith("+00:00")
    assert len(retained["node_names_sha256"]) == 2
    assert len(retained["pod_uids_sha256"]) == 2
