"""Argv contract, safety, and adapter tests for the distributed async drivers.

No network and no spend: the RunPod API, SSH, and catalogs are fakes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from benchmarks.distributed_async_evidence import (
    DistributedAsyncEvidenceError,
    validate_matrix,
    validate_run,
)
from benchmarks.run_distributed_async_matrix import (
    _validate_adapter_result,
    canonical_digest,
    load_manifest,
)
from stateset_agents import __version__ as PACKAGE_VERSION

from .distributed_async_fakes import (
    BENCHMARKS,
    load_bench,
    workload,
    write_synthetic_run,
)

load_bench("runpod_backend_conformance")
local_driver = load_bench("distributed_async_local_driver")
runpod_driver = load_bench("distributed_async_runpod_driver")

RUNPOD_MANIFEST = BENCHMARKS / "distributed_async_manifest.json"
LOCAL_MANIFEST = BENCHMARKS / "distributed_async_local_manifest.json"


def _manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _argv(
    manifest: dict[str, Any], tmp_path: Path, *, scenario: str = "worker_exit"
) -> list[str]:
    config_json = json.dumps(manifest["config"], sort_keys=True, separators=(",", ":"))
    values = {
        "scenario": scenario,
        "seed": 42,
        "mode": "measured",
        "protocol": manifest["protocol"],
        "framework_version": manifest["framework_version"],
        "config_json": config_json,
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "adapter_output": str(tmp_path / "run" / "adapter-result.json"),
        "artifact_dir": str(tmp_path / "run" / "artifact"),
    }
    argv = []
    for part in manifest["command"][2:]:
        argv.append(str(values[part[1:-1]]) if part.startswith("{") else part)
    return argv


# --------------------------------------------------------------------------- #
# Manifests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", [RUNPOD_MANIFEST, LOCAL_MANIFEST])
def test_manifests_load_and_pin_current_version(path: Path) -> None:
    manifest = load_manifest(path)
    assert manifest["framework_version"] == PACKAGE_VERSION == "0.64.1"
    config = workload.parse_config(manifest["config"])
    for scenario, minimum in manifest["minimum_duration_seconds"].items():
        if manifest["provider"] == "runpod":
            assert config.scenario_duration_seconds[scenario] >= minimum
    assert manifest["command"][1].endswith(
        "distributed_async_runpod_driver.py"
        if manifest["provider"] == "runpod"
        else "distributed_async_local_driver.py"
    )


def test_runpod_manifest_shape() -> None:
    manifest = _manifest(RUNPOD_MANIFEST)
    assert manifest["provider"] == "runpod" and manifest["cost_source"] == "runpod-api"
    assert manifest["topology"]["node_count"] == 2
    assert manifest["topology"]["worker_count"] == 4
    assert manifest["topology"]["interconnect"] == runpod_driver.INTERCONNECT
    caps = manifest["config"]["max_cost_usd_per_scenario"]
    assert set(caps) == set(workload.SCENARIOS)


# --------------------------------------------------------------------------- #
# Local driver
# --------------------------------------------------------------------------- #


def test_local_driver_refuses_mislabelled_hardware() -> None:
    detected = {"accelerator": "CPU", "accelerator_driver": "none (CPU-only PyTorch)"}
    local_driver.check_accelerator("CPU", "none (CPU-only PyTorch)", detected)
    with pytest.raises(local_driver.LocalDriverError, match="declared accelerator"):
        local_driver.check_accelerator("NVIDIA H100", "CUDA 12", detected)
    assert local_driver.split_workers(5, 2) == [3, 2]
    with pytest.raises(local_driver.LocalDriverError):
        local_driver.split_workers(1, 2)


def test_local_driver_rejects_bad_inputs(tmp_path: Path, capsys: Any) -> None:
    manifest = _manifest(LOCAL_MANIFEST)
    argv = _argv(manifest, tmp_path)
    tampered = list(argv)
    tampered[tampered.index("--config-sha256") + 1] = "0" * 64
    assert local_driver.main(tampered) == 2
    assert "config_sha256" in capsys.readouterr().err
    wrong_version = list(argv)
    wrong_version[wrong_version.index("--framework-version") + 1] = "0.0.1"
    assert local_driver.main(wrong_version) == 2
    wrong_link = list(argv)
    wrong_link[wrong_link.index("--interconnect") + 1] = "100 GbE"
    assert local_driver.main(wrong_link) == 2
    assert not (tmp_path / "run" / "adapter-result.json").exists()


def test_local_driver_writes_no_adapter_when_nodes_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_nodes(args: Any, **_kwargs: Any) -> dict[str, Any]:
        write_synthetic_run(Path(args.artifact_dir) / "nodes")
        return {
            "node_ids": ["node-0", "node-1"],
            "exit_codes": {"node-0": 0, "node-1": 1},
            "resources_remaining": 1,
            "wall_seconds": 1.0,
        }

    monkeypatch.setattr(local_driver, "run_nodes", fake_run_nodes)
    argv = _argv(_manifest(LOCAL_MANIFEST), tmp_path)
    assert local_driver.main(argv) == 1
    assert not (tmp_path / "run" / "adapter-result.json").exists()
    record = json.loads((tmp_path / "run" / "artifact" / "local-run.json").read_text())
    assert any("node-1 agent exited 1" in item for item in record["problems"])
    assert record["publishable"] is False


def test_local_adapter_validates_per_run_but_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_nodes(args: Any, **_kwargs: Any) -> dict[str, Any]:
        write_synthetic_run(Path(args.artifact_dir) / "nodes")
        return {
            "node_ids": ["node-0", "node-1"],
            "exit_codes": {"node-0": 0, "node-1": 0},
            "resources_remaining": 0,
            "wall_seconds": 1.0,
        }

    monkeypatch.setattr(local_driver, "run_nodes", fake_run_nodes)
    manifest = _manifest(LOCAL_MANIFEST)
    assert local_driver.main(_argv(manifest, tmp_path)) == 0
    output = tmp_path / "run" / "adapter-result.json"
    adapter = json.loads(output.read_text())
    assert adapter["provider"] == "local" and adapter["cost"]["total"] == 0.0
    assert adapter["topology"]["diagnostic"] is True
    assert adapter["topology"]["physical_hosts"] == 1
    result = _validate_adapter_result(
        adapter, manifest=manifest, scenario="worker_exit", seed=42, source=output
    )
    evidence = {
        **result,
        "schema_version": 1,
        "kind": "stateset-distributed-async-evidence",
        "run_id": "r",
        "harness_commit": "a" * 40,
        "timestamp": "2026-10-12T00:00:00+00:00",
        "config": manifest["config"],
        "config_sha256": canonical_digest(manifest["config"]),
        "artifact_sha256": "b" * 64,
    }
    validate_run(evidence, output)
    runs = []
    for scenario in workload.SCENARIOS:
        for seed in (1, 2, 3):
            runs.append({**evidence, "scenario": scenario, "seed": seed})
    with pytest.raises(DistributedAsyncEvidenceError, match="not publishable"):
        validate_matrix(runs, min_seeds=3, min_soak_seconds=1)


def test_publication_gate_rejects_too_few_physical_hosts() -> None:
    base = {
        "scenario": "worker_exit",
        "seed": 1,
        "provider": "runpod",
        "topology": {"physical_hosts": 1},
    }
    with pytest.raises(DistributedAsyncEvidenceError, match="physical hosts"):
        validate_matrix([base], min_seeds=1, min_soak_seconds=1)


# --------------------------------------------------------------------------- #
# RunPod driver (mocked API, SSH, and catalogs)
# --------------------------------------------------------------------------- #


def _cpu_catalog(flavor: str, vcpus: int) -> dict[str, Any]:
    return {
        "id": flavor,
        "ramMultiplier": 2,
        "minVcpu": 2,
        "maxVcpu": 32,
        "instance_id": f"{flavor}-{vcpus}-{vcpus * 2}",
        "securePrice": 0.12,
        "stockStatus": "High",
    }


def _no_gpu_catalog() -> list[dict[str, Any]]:
    raise AssertionError("CPU runs must not query the GPU catalog")


class FakeApi:
    root = "https://rest.example.invalid/v1"
    api_key = "fake-key-not-real"

    def __init__(
        self,
        *,
        fail_create_at: int | None = None,
        price: float = 0.12,
        fail_terminate: bool = False,
    ) -> None:
        self.created: list[dict[str, Any]] = []
        self.terminated: list[str] = []
        self.fail_create_at = fail_create_at
        self.price = price
        self.fail_terminate = fail_terminate

    def create_cpu_pod(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail_create_at is not None and len(self.created) == self.fail_create_at:
            raise runpod_driver.RemoteExecutionError("capacity", provider="runpod")
        pod_id = f"pod{len(self.created)}"
        self.created.append({"id": pod_id, **kwargs})
        return {"id": pod_id, "costPerHr": self.price}

    def create_pod(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("CPU runs must not create GPU pods")

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        index = int(pod_id[3:])
        return {
            "desiredStatus": "RUNNING",
            "publicIp": f"203.0.113.{index + 10}",
            "portMappings": {"22": 40000 + index, "29500": 41000},
            "machineId": f"machine-{index}",
            "costPerHr": self.price,
        }

    def terminate_pod(self, pod_id: str) -> None:
        if self.fail_terminate:
            raise RuntimeError("API down")
        self.terminated.append(pod_id)


class FakeSsh:
    instances: list[FakeSsh] = []

    def __init__(self, *, source: Path | None, exit_code: int = 0) -> None:
        self.source = source
        self.exit_code = exit_code
        self.commands: list[str] = []
        self.secrets: list[str] = []
        self.host: tuple[str, int] | None = None
        FakeSsh.instances.append(self)

    def wait_until_reachable(self, host: str, port: int, timeout_s: int) -> None:
        self.host = (host, port)

    def upload(self, local: Path, remote: str) -> None:
        self.commands.append(f"upload {remote}")

    def upload_secret(self, secret: str, remote: str) -> None:
        self.secrets.append(remote)

    def run(self, command: str) -> tuple[int, str]:
        self.commands.append(command)
        if command.startswith("cat /workspace/da-node.exit"):
            return 0, f"{self.exit_code}\n"
        if command.startswith("pgrep"):
            return 0, "0\n"
        return 0, ""

    def download_dir(self, remote_dir: str, local_dir: Path) -> list[Path]:
        if self.source is None:
            raise RuntimeError("no artifacts")
        node = Path(remote_dir).name
        shutil.copytree(self.source / node, local_dir, dirs_exist_ok=True)
        return list(local_dir.rglob("*"))


def _run_runpod(
    tmp_path: Path,
    api: FakeApi,
    *,
    extra: list[str] | None = None,
    environ: dict[str, str] | None = None,
    exit_code: int = 0,
    with_artifacts: bool = True,
    duration: float = 400.0,
) -> int:
    source = None
    if with_artifacts:
        source = tmp_path / "remote"
        write_synthetic_run(source, duration=duration)
    FakeSsh.instances = []
    argv = _argv(_manifest(RUNPOD_MANIFEST), tmp_path) + [
        "--harness-revision",
        "c" * 40,
        "--lease-dir",
        str(tmp_path / "leases"),
        "--ledger",
        str(tmp_path / "ledger.jsonl"),
        "--ssh-public-key",
        str(_pubkey(tmp_path)),
        "--poll-seconds",
        "0",
        *(extra or []),
    ]
    env = {"RUNPOD_API_KEY": "fake-key-not-real"}
    env.update(environ or {})
    return runpod_driver.main(
        argv,
        environ=env,
        gpu_catalog=_no_gpu_catalog,
        cpu_catalog=_cpu_catalog,
        api_factory=lambda _key: api,
        ssh_factory=lambda **_kw: FakeSsh(source=source, exit_code=exit_code),
        sleep=lambda _s: None,
    )


def _pubkey(tmp_path: Path) -> Path:
    path = tmp_path / "id_test.pub"
    path.write_text("ssh-ed25519 AAAAtest test@example\n")
    return path


def test_runpod_plan_prices_full_matrix_without_provisioning(
    tmp_path: Path, capsys: Any
) -> None:
    api = FakeApi()
    assert _run_runpod(tmp_path, api, extra=["--plan"], with_artifacts=False) == 0
    plan = json.loads(capsys.readouterr().out)
    assert api.created == []
    assert plan["catalog_cost_per_pod_hr_usd"] == 0.12
    assert plan["catalog_cost_per_hr_usd_all_pods"] == 0.24
    assert plan["matrix_runs"] == 12
    soak = plan["per_scenario"]["steady_state_soak"]
    assert soak["lifetime_seconds"] == 43_500 + runpod_driver.SETUP_HEADROOM_SECONDS
    assert soak["worst_case_cost_usd"] == pytest.approx(0.24 * 45_300 / 3600, abs=1e-3)
    expected = 3 * sum(
        item["worst_case_cost_usd"] for item in plan["per_scenario"].values()
    )
    assert plan["matrix_worst_case_cost_usd"] == pytest.approx(expected, abs=1e-3)
    assert all(item["within_cap"] for item in plan["per_scenario"].values())
    assert plan["provisions_hardware"] is False


def test_runpod_dry_run_is_offline(tmp_path: Path, capsys: Any) -> None:
    api = FakeApi()
    assert _run_runpod(tmp_path, api, extra=["--dry-run"], with_artifacts=False) == 0
    preview = json.loads(capsys.readouterr().out)
    assert api.created == []
    launch = preview["commands"][0]["launch"]
    assert "--controller" in launch and "STATESET_DA_WORKER_KEY" in launch
    assert "fake-key-not-real" not in json.dumps(preview)
    assert "--controller-url" in preview["commands"][1]["launch"]


def test_runpod_refuses_without_ceiling(tmp_path: Path, capsys: Any) -> None:
    api = FakeApi()
    assert _run_runpod(tmp_path, api, with_artifacts=False) == 2
    assert "explicit spend ceiling" in capsys.readouterr().err
    assert api.created == []


def test_runpod_refuses_ceiling_above_cap_or_below_worst_case(
    tmp_path: Path, capsys: Any
) -> None:
    api = FakeApi()
    assert (
        _run_runpod(
            tmp_path, api, extra=["--confirm-max-cost-usd", "5"], with_artifacts=False
        )
        == 2
    )
    assert "exceeds config.max_cost_usd_per_scenario" in capsys.readouterr().err
    assert (
        _run_runpod(
            tmp_path,
            api,
            extra=["--confirm-max-cost-usd", "0.01"],
            with_artifacts=False,
        )
        == 2
    )
    assert "worst case" in capsys.readouterr().err
    assert api.created == []


def test_runpod_refuses_gpu_label_on_cpu_pods(tmp_path: Path) -> None:
    manifest = _manifest(RUNPOD_MANIFEST)
    argv = _argv(manifest, tmp_path)
    argv[argv.index("--accelerator") + 1] = "NVIDIA H100 80GB HBM3"
    assert runpod_driver.main(argv, environ={}, cpu_catalog=_cpu_catalog) == 2


def test_runpod_success_writes_valid_adapter_and_terminates(tmp_path: Path) -> None:
    api = FakeApi()
    code = _run_runpod(
        tmp_path,
        api,
        environ={runpod_driver.ENV_CEILING: "0.5"},
    )
    assert code == 0
    assert sorted(api.terminated) == ["pod0", "pod1"]
    assert not list((tmp_path / "leases").glob("*.json"))
    assert len(api.created) == 2
    assert api.created[0]["ports"] == ["22/tcp", "29500/tcp"]
    assert api.created[1]["ports"] == ["22/tcp"]
    assert (
        api.created[0]["cpu_flavor_id"] == "cpu3c" and api.created[0]["vcpu_count"] == 4
    )
    controller_ssh, worker_ssh = FakeSsh.instances
    launch = next(c for c in worker_ssh.commands if "distributed_async_workload" in c)
    assert "http://203.0.113.10:41000" in launch
    assert "/workspace/.stateset-da-worker-key" in controller_ssh.secrets
    output = tmp_path / "run" / "adapter-result.json"
    adapter = json.loads(output.read_text())
    manifest = _manifest(RUNPOD_MANIFEST)
    _validate_adapter_result(
        adapter, manifest=manifest, scenario="worker_exit", seed=42, source=output
    )
    assert adapter["topology"]["node_ids"] == ["runpod:pod0", "runpod:pod1"]
    assert adapter["topology"]["physical_hosts"] == 2
    assert adapter["cost"]["source"] == "runpod-api" and adapter["cost"]["total"] > 0
    ledger = [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
    ]
    assert len(ledger) == 2 and all(row["status"] == "completed" for row in ledger)
    record = json.loads(
        (tmp_path / "run" / "artifact" / "runpod-provider.json").read_text()
    )
    assert "fake-key-not-real" not in json.dumps(record)


def test_runpod_second_pod_failure_terminates_first(tmp_path: Path) -> None:
    api = FakeApi(fail_create_at=1)
    code = _run_runpod(tmp_path, api, environ={runpod_driver.ENV_CEILING: "0.5"})
    assert code == 1
    assert api.terminated == ["pod0"]
    assert not (tmp_path / "run" / "adapter-result.json").exists()


def test_runpod_authoritative_price_above_ceiling_terminates(tmp_path: Path) -> None:
    api = FakeApi(price=5.0)
    code = _run_runpod(tmp_path, api, environ={runpod_driver.ENV_CEILING: "0.5"})
    assert code == 1
    assert sorted(api.terminated) == ["pod0", "pod1"]
    assert FakeSsh.instances == []
    assert not (tmp_path / "run" / "adapter-result.json").exists()


def test_runpod_remote_failure_terminates_and_writes_no_adapter(tmp_path: Path) -> None:
    api = FakeApi()
    code = _run_runpod(
        tmp_path, api, environ={runpod_driver.ENV_CEILING: "0.5"}, exit_code=3
    )
    assert code == 1
    assert sorted(api.terminated) == ["pod0", "pod1"]
    assert not (tmp_path / "run" / "adapter-result.json").exists()


def test_runpod_missing_artifacts_terminates_and_writes_no_adapter(
    tmp_path: Path,
) -> None:
    api = FakeApi()
    code = _run_runpod(
        tmp_path, api, environ={runpod_driver.ENV_CEILING: "0.5"}, with_artifacts=False
    )
    assert code == 1
    assert sorted(api.terminated) == ["pod0", "pod1"]
    assert not (tmp_path / "run" / "adapter-result.json").exists()


def test_runpod_short_run_is_not_evidence(tmp_path: Path, capsys: Any) -> None:
    api = FakeApi()
    code = _run_runpod(
        tmp_path, api, environ={runpod_driver.ENV_CEILING: "0.5"}, duration=10.0
    )
    assert code == 1
    assert "before its planned duration" in capsys.readouterr().err
    assert not (tmp_path / "run" / "adapter-result.json").exists()


def test_runpod_unconfirmed_termination_retains_leases(tmp_path: Path) -> None:
    api = FakeApi(fail_terminate=True)
    code = _run_runpod(tmp_path, api, environ={runpod_driver.ENV_CEILING: "0.5"})
    assert code == 2
    assert len(list((tmp_path / "leases").glob("*.json"))) == 2
    assert not (tmp_path / "run" / "adapter-result.json").exists()


def test_create_cpu_pod_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    from stateset_agents.remote.runpod import RunPodApi

    captured: dict[str, Any] = {}

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"id": "p1"}

    def fake_post(url: str, **kwargs: Any) -> Response:
        captured.update(kwargs)
        captured["url"] = url
        return Response()

    monkeypatch.setattr(requests, "post", fake_post)
    pod = RunPodApi("k").create_cpu_pod(
        name="n",
        image="img",
        cpu_flavor_id="cpu3c",
        vcpu_count=4,
        ports=["22/tcp"],
        env={},
    )
    assert pod == {"id": "p1"}
    payload = captured["json"]
    assert payload["computeType"] == "CPU" and payload["cpuFlavorIds"] == ["cpu3c"]
    assert payload["vcpuCount"] == 4 and payload["cloudType"] == "SECURE"
    assert "gpuTypeIds" not in payload
