"""Tests for the zero-surprise RunPod conformance launcher."""

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


if "backend_conformance" not in sys.modules:
    _load("backend_conformance", BENCHMARKS / "backend_conformance.py")
launcher = _load(
    "runpod_backend_conformance", BENCHMARKS / "runpod_backend_conformance.py"
)


def _manifest(dataset_sha256: str, **execution_updates: Any) -> dict[str, Any]:
    execution = {
        "provider": "runpod",
        "provider_tier": "SECURE",
        "container_image": "registry.example/nemo@sha256:" + "d" * 64,
        "gpu_name": "NVIDIA A40",
        "gpu_count": 1,
        "container_disk_gb": 80,
        "timeout_seconds": 60,
        "max_lifetime_seconds": 120,
        "max_cost_usd": 1.0,
    }
    execution.update(execution_updates)
    return {
        "schema_version": 3,
        "backend": "nemo-rl",
        "backend_version": "0.6.0+abcdef0",
        "harness_revision": "a" * 40,
        "execution": execution,
        "experiment": {
            "algorithm": "grpo",
            "model": "Qwen/example",
            "model_revision": "b" * 40,
            "dataset_uri": "/workspace/data/train.jsonl",
            "dataset_sha256": dataset_sha256,
            "seed": 42,
            "config": {"max_steps": 1},
        },
    }


def _inputs(tmp_path: Path, **execution_updates: Any) -> tuple[dict[str, Any], Path]:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"prompt":"2+2"}\n', encoding="utf-8")
    digest = launcher.hashlib.sha256(dataset.read_bytes()).hexdigest()
    return _manifest(digest, **execution_updates), dataset


def _catalog(**updates: Any) -> list[dict[str, Any]]:
    row = {
        "id": "NVIDIA A40",
        "displayName": "A40",
        "secureCloud": True,
        "communityCloud": True,
        "securePrice": 0.44,
        "communityPrice": 0.35,
    }
    row.update(updates)
    return [row]


def test_plan_binds_full_lifetime_and_performs_no_provisioning(tmp_path: Path) -> None:
    manifest, dataset = _inputs(tmp_path, gpu_count=2)
    plan = launcher.build_plan(manifest, dataset, _catalog())
    assert plan["gpu_type_id"] == "NVIDIA A40"
    assert plan["catalog_total_cost_per_hr_usd"] == 0.88
    assert plan["max_lifetime_seconds"] == 120
    assert plan["provisions_hardware"] is False
    assert "api_key" not in json.dumps(plan).lower()


@pytest.mark.parametrize(
    ("updates", "catalog_updates", "message"),
    [
        ({"provider": "other"}, {}, "must be runpod"),
        ({"provider_tier": "secure"}, {}, "SECURE or COMMUNITY"),
        ({}, {"secureCloud": False}, "unavailable"),
        ({"gpu_name": "missing"}, {}, "exactly one GPU"),
        ({"max_cost_usd": 0.001}, {}, "above the --max-cost ceiling"),
    ],
)
def test_plan_fails_closed_on_provider_or_budget_drift(
    tmp_path: Path,
    updates: dict[str, Any],
    catalog_updates: dict[str, Any],
    message: str,
) -> None:
    manifest, dataset = _inputs(tmp_path, **updates)
    with pytest.raises(launcher.RunPodConformanceError, match=message):
        launcher.build_plan(manifest, dataset, _catalog(**catalog_updates))


def test_plan_rejects_launcher_owned_dataset_path(tmp_path: Path) -> None:
    manifest, dataset = _inputs(tmp_path)
    manifest["experiment"]["dataset_uri"] = "/workspace/stateset-agents/data.jsonl"
    with pytest.raises(launcher.RunPodConformanceError, match="launcher-owned"):
        launcher.build_plan(manifest, dataset, _catalog())


def test_default_cli_is_public_plan_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, dataset = _inputs(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(launcher, "fetch_catalog", _catalog)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert launcher.main([str(path), "--dataset", str(dataset)]) == 0
    assert '"provisions_hardware": false' in capsys.readouterr().out


def test_cli_refuses_execution_without_exact_spend_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, dataset = _inputs(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(launcher, "fetch_catalog", _catalog)
    monkeypatch.setattr(
        launcher,
        "RunPodApi",
        lambda *_: pytest.fail("API client must not be created before confirmation"),
    )
    assert (
        launcher.main(
            [
                str(path),
                "--dataset",
                str(dataset),
                "--execute",
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
        == 2
    )


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
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.uploaded: list[tuple[Path, str]] = []

    def wait_until_reachable(self, host: str, port: int, timeout: int) -> None:
        self.endpoint = (host, port, timeout)

    def upload(self, local: Path, remote: str) -> None:
        self.uploaded.append((local, remote))

    def upload_secret(self, secret: str, remote: str) -> None:
        self.secret_remote = remote

    def run(self, command: str) -> tuple[int, str]:
        self.commands.append(command)
        return 0, "ok"

    def download_dir(self, remote: str, local: Path) -> list[Path]:
        path = local / "conformance.json"
        path.write_text("{}", encoding="utf-8")
        return [path]


def test_authoritative_price_drift_terminates_and_clears_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, dataset = _inputs(tmp_path, max_cost_usd=0.01)
    plan = launcher.build_plan(manifest, dataset, _catalog(securePrice=0.01))
    api = _Api(price=10.0)
    monkeypatch.setattr(launcher, "verify_harness_revision", lambda *_: None)
    with pytest.raises(launcher.RunPodConformanceError, match="above"):
        launcher.execute(
            manifest,
            dataset,
            tmp_path / "output",
            tmp_path,
            plan,
            api=api,
            ssh=_Ssh(),
            public_key="ssh-ed25519 test",
            lease_dir=tmp_path / "leases",
            ledger_path=tmp_path / "ledger.jsonl",
        )
    assert api.terminated == ["pod-123"]
    assert list((tmp_path / "leases").glob("*.json")) == []


def test_success_downloads_validates_records_and_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, dataset = _inputs(tmp_path)
    plan = launcher.build_plan(manifest, dataset, _catalog())
    api, ssh = _Api(), _Ssh()
    monkeypatch.setattr(launcher, "verify_harness_revision", lambda *_: None)
    monkeypatch.setattr(launcher, "_arm_self_destruct", lambda *_: None)
    monkeypatch.setattr(launcher, "load_evidence", lambda path: {"path": str(path)})
    output = tmp_path / "output"
    evidence = launcher.execute(
        manifest,
        dataset,
        output,
        tmp_path,
        plan,
        api=api,
        ssh=ssh,
        public_key="ssh-ed25519 test",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
    )
    assert evidence == output / "conformance.json"
    assert api.terminated == ["pod-123"]
    assert api.create_kwargs["container_disk_gb"] == 80
    assert any("backend_conformance.py" in command for command in ssh.commands)
    provider = json.loads((output / "runpod-provider.json").read_text())
    assert provider["status"] == "completed"
    assert provider["termination_confirmed"] is True
    assert provider["cleanup_lease_retained"] is False


def test_successful_evidence_fails_closed_when_cleanup_is_unconfirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, dataset = _inputs(tmp_path)
    plan = launcher.build_plan(manifest, dataset, _catalog())
    api, ssh = _TerminationFailsApi(), _Ssh()
    monkeypatch.setattr(launcher, "verify_harness_revision", lambda *_: None)
    monkeypatch.setattr(launcher, "_arm_self_destruct", lambda *_: None)
    monkeypatch.setattr(launcher, "load_evidence", lambda path: {"path": str(path)})
    output = tmp_path / "output"
    with pytest.raises(launcher.RunPodConformanceError, match="not confirmed"):
        launcher.execute(
            manifest,
            dataset,
            output,
            tmp_path,
            plan,
            api=api,
            ssh=ssh,
            public_key="ssh-ed25519 test",
            lease_dir=tmp_path / "leases",
            ledger_path=tmp_path / "ledger.jsonl",
        )
    provider = json.loads((output / "runpod-provider.json").read_text())
    assert provider["status"] == "cleanup-pending"
    assert provider["termination_confirmed"] is False
    assert provider["cleanup_lease_retained"] is True
