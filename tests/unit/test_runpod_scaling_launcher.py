"""Safety contracts for the RunPod multi-node scaling launcher."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks import runpod_scaling_launcher as launcher


def _args(tmp_path: Path, *extra: str) -> Any:
    workload = tmp_path / "distributed_scaling_workload.py"
    workload.write_text("# pinned workload\n", encoding="utf-8")
    argv = [
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
        "matched scaling",
        "--output",
        str(tmp_path / "evidence" / "gpu8-seed42.json"),
        "--workload",
        str(workload),
        "--gpu-name",
        "NVIDIA A40",
        "--image",
        "runpod/pytorch:test",
        "--data-center-id",
        "US-TX-3",
        "--max-cost-usd",
        "3.0",
        *extra,
    ]
    return launcher.parse_args(argv)


def _catalog() -> list[dict[str, Any]]:
    return [
        {
            "id": "NVIDIA A40",
            "secureCloud": True,
            "communityCloud": True,
            "securePrice": 0.4,
            "communityPrice": 0.3,
        }
    ]


def test_plan_prices_every_gpu_for_full_lifetime(tmp_path: Path) -> None:
    args = _args(tmp_path)
    launcher.validate_request(args)
    plan = launcher.build_plan(args, _catalog())
    assert plan["catalog_total_cost_per_hr_usd"] == pytest.approx(3.2)
    assert plan["worst_case_cost_usd"] == pytest.approx(2.4)
    assert plan["global_networking"] is True


def test_execution_requires_exact_cost_confirmation(tmp_path: Path) -> None:
    args = _args(tmp_path, "--execute", "--confirm-max-cost-usd", "2.99")
    with pytest.raises(launcher.RunPodScalingError, match="exactly equal"):
        launcher.validate_request(args)


def test_multi_node_requires_secure_pinned_datacenter(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.cloud_type = "COMMUNITY"
    with pytest.raises(launcher.RunPodScalingError, match="SECURE"):
        launcher.validate_request(args)
    args.cloud_type = "SECURE"
    args.data_center_id = None
    with pytest.raises(launcher.RunPodScalingError, match="data-center-id"):
        launcher.validate_request(args)


class _Api:
    api_key = "secret"
    root = "https://api.invalid"

    def __init__(self) -> None:
        self.created = 0
        self.terminated: list[str] = []

    def create_pod(self, **_: Any) -> dict[str, Any]:
        self.created += 1
        if self.created == 2:
            raise RuntimeError("capacity disappeared")
        return {"id": "pod-1", "costPerHr": 1.0, "machineId": "machine-1"}

    def terminate_pod(self, pod_id: str) -> None:
        self.terminated.append(pod_id)


def test_partial_provision_failure_terminates_every_created_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, "--execute", "--confirm-max-cost-usd", "3.0")
    api = _Api()
    monkeypatch.setattr(
        launcher,
        "_write_lease",
        lambda *_a, **_k: tmp_path / "lease.json",
    )
    with pytest.raises(RuntimeError, match="capacity disappeared"):
        launcher.execute(
            args,
            launcher.build_plan(args, _catalog()),
            api=api,
            ssh_factory=lambda: SimpleNamespace(),
            public_key="ssh-ed25519 test",
        )
    assert api.terminated == ["pod-1"]
    record = args.output.parent.parent / "runpod-provider" / args.output.name
    assert record.is_file()
    retained = __import__("json").loads(record.read_text(encoding="utf-8"))
    assert retained["status"] == "failed"
    assert retained["image"] == "runpod/pytorch:test"
    assert retained["pods"][0]["termination_confirmed"] is True


def test_remote_command_uses_private_dns_and_exact_topology(tmp_path: Path) -> None:
    command = launcher._remote_command(_args(tmp_path), 1, "pod-1.runpod.internal")
    assert "--nnodes=2" in command
    assert "--nproc-per-node=4" in command
    assert "--node-rank=1" in command
    assert "--master-addr=pod-1.runpod.internal" in command
