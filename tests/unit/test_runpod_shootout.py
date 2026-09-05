"""Tests for the budget-bounded RunPod shootout launcher."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _name in (
    "backend_conformance",
    "runpod_backend_conformance",
    "framework_comparison",
    "shootout",
):
    if _name not in sys.modules:
        _load(_name, BENCHMARKS / f"{_name}.py")
launcher = _load("runpod_shootout", BENCHMARKS / "runpod_shootout.py")

SHA = "0" * 40


def _shootout_manifest() -> dict[str, Any]:
    example = json.loads((BENCHMARKS / "shootout_manifest.example.json").read_text())
    example["hardware"] = {"gpu": "NVIDIA A40", "gpu_count": 1}
    return example


def _write_inputs(tmp_path: Path, **execution_updates: Any) -> Path:
    (tmp_path / "benchmarks").mkdir(exist_ok=True)
    (tmp_path / "benchmarks" / "shootout.json").write_text(
        json.dumps(_shootout_manifest())
    )
    execution = {
        "provider": "runpod",
        "provider_tier": "SECURE",
        "container_image": "runpod/pytorch:2.4.0",
        "gpu_name": "NVIDIA A40",
        "gpu_count": 1,
        "container_disk_gb": 60,
        "timeout_seconds": 1800,
        "max_lifetime_seconds": 2400,
        "max_cost_usd": 1.0,
    }
    execution.update(execution_updates)
    manifest = {
        "schema_version": 1,
        "harness_revision": SHA,
        "shootout_manifest": "benchmarks/shootout.json",
        "execution": execution,
        "extra_pip": ["trl==1.9.1"],
        "prewarm_command": "python -c 'print(1)'",
    }
    path = tmp_path / "launcher.json"
    path.write_text(json.dumps(manifest))
    return path


def _catalog(**updates: Any) -> list[dict[str, Any]]:
    row = {
        "id": "NVIDIA A40",
        "displayName": "A40",
        "secureCloud": True,
        "communityCloud": True,
        "securePrice": 0.44,
        "communityPrice": 0.39,
    }
    row.update(updates)
    return [row]


class _Api:
    api_key = "test-secret"
    root = "https://rest.runpod.test/v1"

    def __init__(self, price: float = 0.44) -> None:
        self.price = price
        self.terminated: list[str] = []

    def create_pod(self, **kwargs: Any) -> dict[str, Any]:
        self.create_kwargs = kwargs
        return {"id": "pod-123", "costPerHr": self.price}

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return {
            "id": pod_id,
            "costPerHr": self.price,
            "desiredStatus": "RUNNING",
            "publicIp": "127.0.0.1",
            "portMappings": {"22": 22022},
        }

    def terminate_pod(self, pod_id: str) -> None:
        self.terminated.append(pod_id)


class _TerminationFailsApi(_Api):
    def terminate_pod(self, pod_id: str) -> None:
        self.terminated.append(pod_id)
        raise RuntimeError("provider unavailable")


class _Ssh:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self.commands: list[str] = []
        self.secrets: list[str] = []

    def wait_until_reachable(self, host: str, port: int, timeout: int) -> None:
        self.endpoint = (host, port, timeout)

    def upload(self, local: Path, remote: str) -> None:
        pass

    def upload_secret(self, secret: str, remote: str) -> None:
        self.secrets.append(remote)

    def run(self, command: str) -> tuple[int, str]:
        self.commands.append(command)
        if "benchmarks/shootout.py" in command:
            return self.exit_code, "shootout output"
        return 0, "ok"

    def download_dir(self, remote: str, local: Path) -> list[Path]:
        path = local / "shootout-summary.json"
        path.write_text("{}", encoding="utf-8")
        return [path]


def test_plan_prices_full_lifetime_and_counts_runs(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    plan = launcher.build_plan(manifest, _catalog())
    assert plan["provisions_hardware"] is False
    assert plan["planned_runs"] == 3 * 2
    assert plan["worst_case_cost_usd"] == pytest.approx(0.44 * 2400 / 3600, abs=1e-3)
    assert plan["frameworks"] == ["stateset-agents", "trl"]


def test_plan_fails_closed_when_lifetime_exceeds_ceiling(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(
        _write_inputs(tmp_path, max_cost_usd=0.01), tmp_path
    )
    with pytest.raises(launcher.RunPodShootoutError):
        launcher.build_plan(manifest, _catalog())


def test_manifest_requires_lifetime_headroom_and_matching_gpu(tmp_path: Path) -> None:
    with pytest.raises(launcher.RunPodShootoutError, match="headroom"):
        launcher.load_launcher_manifest(
            _write_inputs(tmp_path, max_lifetime_seconds=1800), tmp_path
        )
    with pytest.raises(launcher.RunPodShootoutError, match="hardware"):
        launcher.load_launcher_manifest(
            _write_inputs(tmp_path, gpu_name="NVIDIA H100 80GB HBM3"), tmp_path
        )


def test_cli_refuses_execution_without_exact_spend_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    path = _write_inputs(tmp_path)
    monkeypatch.setattr(launcher, "fetch_catalog", lambda: _catalog())
    code = launcher.main(
        [
            str(path),
            "--root",
            str(tmp_path),
            "--execute",
            "--confirm-max-cost-usd",
            "0.5",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2
    assert "exactly equal" in capsys.readouterr().err
    assert launcher.main([str(path), "--root", str(tmp_path)]) == 0


def test_execute_runs_shootout_downloads_records_and_terminates(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    plan = launcher.build_plan(manifest, _catalog())
    api, ssh = _Api(), _Ssh()
    out = tmp_path / "evidence"
    result = launcher.execute(
        manifest,
        out,
        plan,
        api=api,
        ssh=ssh,
        public_key="ssh-ed25519 AAAA",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    assert result == out and (out / "shootout-summary.json").exists()
    joined = "\n".join(ssh.commands)
    assert f"checkout --detach {SHA}" in joined
    assert "pip install --quiet -e" in joined and "trl==1.9.1" in joined
    assert "print(1)" in joined  # prewarm ran inside the repo
    run_cmd = [c for c in ssh.commands if "benchmarks/shootout.py" in c][0]
    assert "--required-framework stateset-agents" in run_cmd
    assert "--required-framework trl" in run_cmd
    assert "--timeout-seconds 1800" in run_cmd
    assert launcher._REMOTE_SHOOTOUT_MANIFEST in ssh.secrets
    assert api.terminated == ["pod-123"]
    record = json.loads((out / "runpod-provider.json").read_text())
    assert record["status"] == "completed" and record["termination_confirmed"]
    assert record["harness_revision"] == SHA
    ledger = (tmp_path / "ledger.jsonl").read_text()
    assert "pod-123" in ledger and "runpod" in ledger
    assert not (tmp_path / "leases").exists() or not any(
        (tmp_path / "leases").iterdir()
    )


def test_remote_failure_keeps_evidence_terminates_and_raises(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    plan = launcher.build_plan(manifest, _catalog())
    api = _Api()
    out = tmp_path / "evidence"
    with pytest.raises(launcher.RunPodShootoutError, match="remote shootout failed"):
        launcher.execute(
            manifest,
            out,
            plan,
            api=api,
            ssh=_Ssh(exit_code=3),
            public_key="k",
            lease_dir=tmp_path / "leases",
            ledger_path=tmp_path / "ledger.jsonl",
        )
    assert api.terminated == ["pod-123"]
    record = json.loads((out / "runpod-provider.json").read_text())
    assert record["status"] == "failed" and record["remote_exit_code"] == 3


def test_price_drift_after_allocation_terminates_before_running(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    plan = launcher.build_plan(manifest, _catalog())
    api, ssh = _Api(price=40.0), _Ssh()
    with pytest.raises(launcher.RunPodShootoutError):
        launcher.execute(
            manifest,
            tmp_path / "evidence",
            plan,
            api=api,
            ssh=ssh,
            public_key="k",
            lease_dir=tmp_path / "leases",
            ledger_path=tmp_path / "ledger.jsonl",
        )
    assert api.terminated == ["pod-123"] and ssh.commands == []


def test_unconfirmed_termination_fails_closed_with_lease(tmp_path: Path) -> None:
    manifest = launcher.load_launcher_manifest(_write_inputs(tmp_path), tmp_path)
    plan = launcher.build_plan(manifest, _catalog())
    with pytest.raises(launcher.RunPodShootoutError, match="not confirmed"):
        launcher.execute(
            manifest,
            tmp_path / "evidence",
            plan,
            api=_TerminationFailsApi(),
            ssh=_Ssh(),
            public_key="k",
            lease_dir=tmp_path / "leases",
            ledger_path=tmp_path / "ledger.jsonl",
        )
    assert any((tmp_path / "leases").iterdir())
