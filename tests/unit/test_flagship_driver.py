"""Safety and contract tests for the flagship RunPod driver (no network, no spend)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from stateset_agents import __version__ as PACKAGE_VERSION

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"


def _load(name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _name in ("backend_conformance", "dataset_content", "runpod_backend_conformance"):
    _load(_name)
driver = _load("flagship_runpod_driver")
runner = _load("run_flagship_matrix")

GPU = "NVIDIA H100 80GB HBM3"


def _manifest() -> dict[str, Any]:
    manifest = json.loads((BENCHMARKS / "flagship_manifest.json").read_text())
    manifest["framework_version"] = PACKAGE_VERSION
    manifest["dataset_revision"] = "b" * 40
    manifest["judge"]["rubric_revision"] = "c" * 40
    return manifest


def _argv(tmp_path: Path, manifest: dict[str, Any], *extra: str) -> list[str]:
    config_json = runner.canonical_json(manifest["config"])
    judge_json = runner.canonical_json(manifest["judge"])
    return [
        "--seed",
        "42",
        "--mode",
        "measured",
        "--framework-version",
        manifest["framework_version"],
        "--model",
        manifest["model"],
        "--model-revision",
        manifest["model_revision"],
        "--dataset-revision",
        manifest["dataset_revision"],
        "--config-json",
        config_json,
        "--config-sha256",
        hashlib.sha256(config_json.encode()).hexdigest(),
        "--judge-json",
        judge_json,
        "--judge-sha256",
        hashlib.sha256(judge_json.encode()).hexdigest(),
        "--output",
        str(tmp_path / "attempt" / "adapter.json"),
        "--artifact-dir",
        str(tmp_path / "artifacts"),
        "--lease-dir",
        str(tmp_path / "leases"),
        "--ledger",
        str(tmp_path / "ledger.jsonl"),
        "--harness-revision",
        "e" * 40,
        "--poll-seconds",
        "0",
        *extra,
    ]


def _catalog(price: float = 2.99) -> list[dict[str, Any]]:
    return [
        {
            "id": GPU,
            "displayName": "H100 SXM",
            "secureCloud": True,
            "communityCloud": True,
            "securePrice": price,
            "communityPrice": price,
        }
    ]


class ForbiddenApi:
    """Any construction means a paid code path was reached."""

    created = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ForbiddenApi.created += 1
        raise AssertionError("RunPod API must not be constructed")


class FakeApi:
    api_key = "test-key"
    root = "https://rest.example.invalid/v1"

    def __init__(self, price: float = 2.99, terminate_error: bool = False) -> None:
        self.price = price
        self.terminate_error = terminate_error
        self.created: list[dict[str, Any]] = []
        self.terminated: list[str] = []

    def create_pod(self, **kwargs: Any) -> dict[str, Any]:
        self.created.append(kwargs)
        return {"id": "pod-1", "costPerHr": self.price}

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return {
            "id": pod_id,
            "desiredStatus": "RUNNING",
            "publicIp": "203.0.113.5",
            "portMappings": {"22": 22022},
            "costPerHr": self.price,
        }

    def terminate_pod(self, pod_id: str) -> None:
        self.terminated.append(pod_id)
        if self.terminate_error:
            raise RuntimeError("terminate failed")


def _result(manifest: dict[str, Any], **updates: Any) -> dict[str, Any]:
    config_json = runner.canonical_json(manifest["config"])
    judge_json = runner.canonical_json(manifest["judge"])
    result = {
        "schema_version": 1,
        "kind": "stateset-flagship-workload-result",
        "status": "completed",
        "measured": True,
        "mode": "measured",
        "smoke": False,
        "seed": 42,
        "framework_version": manifest["framework_version"],
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "dataset_revision": manifest["dataset_revision"],
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "judge_sha256": hashlib.sha256(judge_json.encode()).hexdigest(),
        "hardware": {"gpu": GPU, "gpu_count": 1, "cuda": "12.9", "driver": "570.1"},
        "metrics": {
            "baseline_score": 0.51,
            "final_score": 0.58,
            "judge_self_disagreement": 0.03,
            "judge_parse_failure_rate": 0.0,
            "train_examples": 500,
            "eval_examples": 200,
            "peak_vram_mb": 61234.0,
        },
        "diagnostics": {"baseline_parse_failure_rate": 0.0},
    }
    result.update(updates)
    return result


class FakeSsh:
    def __init__(
        self,
        result: dict[str, Any] | None,
        *,
        exit_code: str = "0",
        fail_on: str | None = None,
    ) -> None:
        self.result = result
        self.exit_code = exit_code
        self.fail_on = fail_on
        self.commands: list[str] = []
        self.secrets: list[str] = []

    def wait_until_reachable(self, host: str, port: int, timeout_s: int) -> None:
        return None

    def upload(self, local: Path, remote: str) -> None:
        return None

    def upload_secret(self, secret: str, remote: str) -> None:
        self.secrets.append(remote)

    def run(self, command: str) -> tuple[int, str]:
        self.commands.append(command)
        if self.fail_on and self.fail_on in command:
            return 1, "boom"
        if command.startswith("cat /workspace/flagship.exit"):
            return 0, self.exit_code
        return 0, ""

    def download_dir(self, remote_dir: str, local_dir: Path) -> list[Path]:
        local_dir.mkdir(parents=True, exist_ok=True)
        if remote_dir.endswith("/artifacts") and self.result is not None:
            (local_dir / "adapter").mkdir(exist_ok=True)
            (local_dir / "adapter" / "adapter_config.json").write_text("{}")
            (local_dir / "workload_result.json").write_text(json.dumps(self.result))
        return [p for p in local_dir.rglob("*") if p.is_file()]


@pytest.fixture
def ssh_key(tmp_path: Path) -> Path:
    key = tmp_path / "id_ed25519.pub"
    key.write_text("ssh-ed25519 AAAA test")
    return key


def _run(
    tmp_path: Path,
    manifest: dict[str, Any],
    api: FakeApi,
    ssh: FakeSsh,
    ssh_key: Path,
    *extra: str,
    environ: dict[str, str] | None = None,
) -> int:
    env = {
        driver.ENV_CEILING: "25",
        "RUNPOD_API_KEY": "test-key",
        **(environ or {}),
    }
    return driver.main(
        _argv(tmp_path, manifest, "--ssh-public-key", str(ssh_key), *extra),
        environ=env,
        catalog_fetcher=_catalog,
        api_factory=lambda key: api,
        ssh_factory=lambda key_path=None: ssh,
    )


# --------------------------------------------------------------------------- #
# Spend-ceiling refusal
# --------------------------------------------------------------------------- #


def test_refuses_without_spend_ceiling(tmp_path: Path) -> None:
    ForbiddenApi.created = 0
    code = driver.main(
        _argv(tmp_path, _manifest()),
        environ={"RUNPOD_API_KEY": "test-key"},
        catalog_fetcher=_catalog,
        api_factory=ForbiddenApi,
    )
    assert code == 2
    assert ForbiddenApi.created == 0
    assert not (tmp_path / "attempt" / "adapter.json").exists()


@pytest.mark.parametrize(
    ("flag", "env", "message"),
    [
        (None, {}, "explicit spend ceiling"),
        (30.0, {}, "exceeds"),
        (None, {driver.ENV_CEILING: "25.01"}, "exceeds"),
        (None, {driver.ENV_CEILING: "abc"}, "must be a number"),
        (-1.0, {}, "positive finite"),
    ],
)
def test_spend_ceiling_validation(
    flag: float | None, env: dict[str, str], message: str
) -> None:
    with pytest.raises(driver.FlagshipDriverError, match=message):
        driver.resolve_spend_ceiling(flag, env, 25.0)


def test_spend_ceiling_accepts_env_and_flag() -> None:
    assert driver.resolve_spend_ceiling(None, {driver.ENV_CEILING: "20"}, 25.0) == 20.0
    assert driver.resolve_spend_ceiling(25.0, {}, 25.0) == 25.0


def test_ceiling_above_cap_refuses_before_api(tmp_path: Path) -> None:
    ForbiddenApi.created = 0
    code = driver.main(
        _argv(tmp_path, _manifest(), "--confirm-max-cost-usd", "40"),
        environ={"RUNPOD_API_KEY": "test-key"},
        catalog_fetcher=_catalog,
        api_factory=ForbiddenApi,
    )
    assert code == 2 and ForbiddenApi.created == 0


def test_worst_case_above_ceiling_refuses_before_api(tmp_path: Path) -> None:
    ForbiddenApi.created = 0
    code = driver.main(
        _argv(tmp_path, _manifest(), "--confirm-max-cost-usd", "5"),
        environ={"RUNPOD_API_KEY": "test-key"},
        catalog_fetcher=_catalog,
        api_factory=ForbiddenApi,
    )
    assert code == 2 and ForbiddenApi.created == 0


def test_template_revisions_refuse_paid_execution(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["dataset_revision"] = "0" * 40
    ForbiddenApi.created = 0
    code = driver.main(
        _argv(tmp_path, manifest),
        environ={driver.ENV_CEILING: "25", "RUNPOD_API_KEY": "k"},
        catalog_fetcher=_catalog,
        api_factory=ForbiddenApi,
    )
    assert code == 2 and ForbiddenApi.created == 0


# --------------------------------------------------------------------------- #
# Free modes
# --------------------------------------------------------------------------- #


def test_plan_makes_no_paid_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ForbiddenApi.created = 0
    fetched: list[int] = []

    def catalog() -> list[dict[str, Any]]:
        fetched.append(1)
        return _catalog()

    code = driver.main(
        _argv(tmp_path, _manifest(), "--plan"),
        environ={},
        catalog_fetcher=catalog,
        api_factory=ForbiddenApi,
        ssh_factory=ForbiddenApi,
    )
    assert code == 0 and fetched == [1] and ForbiddenApi.created == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["provisions_hardware"] is False
    assert plan["gpu_name"] == GPU
    assert plan["worst_case_cost_usd"] <= 25.0
    assert not (tmp_path / "attempt" / "adapter.json").exists()


def test_dry_run_is_offline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def no_catalog() -> list[dict[str, Any]]:
        raise AssertionError("dry run must not fetch the catalog")

    code = driver.main(
        _argv(tmp_path, _manifest(), "--dry-run"),
        environ={},
        catalog_fetcher=no_catalog,
        api_factory=ForbiddenApi,
    )
    assert code == 0
    preview = json.loads(capsys.readouterr().out)
    assert "flagship_workload.py" in preview["commands"]["launch"]
    assert "/workspace/flagship.exit" in preview["commands"]["launch"]
    assert "nohup" in preview["commands"]["launch"]


def test_gpu_override_from_env(tmp_path: Path) -> None:
    args = driver.build_parser().parse_args(_argv(tmp_path, _manifest()))
    settings = driver.execution_settings(
        args, _manifest()["config"], {driver.ENV_GPU: "NVIDIA A100 80GB PCIe"}
    )
    assert settings["gpu_name"] == "NVIDIA A100 80GB PCIe"
    assert settings["max_lifetime_seconds"] == 21600 - driver.LOCAL_HEADROOM_SECONDS


# --------------------------------------------------------------------------- #
# Paid path with fakes: termination, accounting, adapter contract
# --------------------------------------------------------------------------- #


def test_complete_run_writes_valid_adapter(tmp_path: Path, ssh_key: Path) -> None:
    manifest = _manifest()
    api = FakeApi()
    ssh = FakeSsh(_result(manifest))
    hf = {"HF_TOKEN": "hf_secret"}
    assert _run(tmp_path, manifest, api, ssh, ssh_key, environ=hf) == 0

    assert api.terminated == ["pod-1"]
    assert len(api.created) == 1 and api.created[0]["gpu_type_id"] == GPU
    assert driver._REMOTE_HF_TOKEN in ssh.secrets
    assert not any("hf_secret" in command for command in ssh.commands)
    adapter_path = tmp_path / "attempt" / "adapter.json"
    adapter = json.loads(adapter_path.read_text())
    assert adapter["status"] == "completed" and adapter["measured"] is True
    assert adapter["cost_source"] == "runpod-api"
    assert 0 < adapter["metrics"]["cost_usd"] <= 25.0
    config_sha = hashlib.sha256(
        runner.canonical_json(manifest["config"]).encode()
    ).hexdigest()
    validated = runner.validate_adapter(adapter, manifest, config_sha, adapter_path)
    assert Path(validated["artifact_path"]) == (tmp_path / "artifacts").resolve()
    ledger = [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
    ]
    assert ledger[-1]["status"] == "completed" and ledger[-1]["job_id"] == "pod-1"
    record = json.loads((tmp_path / "artifacts" / "runpod-provider.json").read_text())
    assert record["termination_confirmed"] is True
    assert not list((tmp_path / "leases").glob("*.json"))


@pytest.mark.parametrize(
    ("ssh_kwargs", "result_updates"),
    [
        ({"fail_on": "git clone"}, {}),  # setup failure
        ({"exit_code": "1"}, {}),  # workload crashed
        ({}, {"status": "failed", "measured": False}),  # partial workload
        ({}, {"smoke": True, "status": "smoke", "measured": False}),
        (
            {},
            {
                "hardware": {
                    "gpu": "NVIDIA A40",
                    "gpu_count": 1,
                    "cuda": "1",
                    "driver": "1",
                }
            },
        ),
        ({}, {"config_sha256": "f" * 64}),
    ],
)
def test_incomplete_runs_terminate_and_never_complete(
    tmp_path: Path,
    ssh_key: Path,
    ssh_kwargs: dict[str, Any],
    result_updates: dict[str, Any],
) -> None:
    manifest = _manifest()
    api = FakeApi()
    ssh = FakeSsh(_result(manifest, **result_updates), **ssh_kwargs)
    assert _run(tmp_path, manifest, api, ssh, ssh_key) == 1
    assert api.terminated == ["pod-1"]
    adapter_path = tmp_path / "attempt" / "adapter.json"
    adapter = json.loads(adapter_path.read_text())
    assert adapter["status"] == "failed" and adapter["measured"] is False
    assert adapter["problems"]
    with pytest.raises(runner.FlagshipError):
        runner.validate_adapter(
            adapter, manifest, adapter["config_sha256"], adapter_path
        )


def test_partial_metrics_never_complete(tmp_path: Path, ssh_key: Path) -> None:
    manifest = _manifest()
    result = _result(manifest)
    del result["metrics"]["final_score"]
    api = FakeApi()
    assert _run(tmp_path, manifest, api, FakeSsh(result), ssh_key) == 1
    adapter = json.loads((tmp_path / "attempt" / "adapter.json").read_text())
    assert adapter["status"] == "failed"
    assert any("final_score" in problem for problem in adapter["problems"])
    assert api.terminated == ["pod-1"]


def test_post_allocation_price_over_budget_terminates(
    tmp_path: Path, ssh_key: Path
) -> None:
    manifest = _manifest()
    api = FakeApi(price=9.0)  # 9.0 $/h x ~5.7 h > $25
    ssh = FakeSsh(_result(manifest))
    assert _run(tmp_path, manifest, api, ssh, ssh_key) == 1
    assert api.terminated == ["pod-1"]
    assert ssh.commands == []  # never reached setup


def test_unexpected_exception_still_terminates(tmp_path: Path, ssh_key: Path) -> None:
    manifest = _manifest()
    api = FakeApi()

    class ExplodingSsh(FakeSsh):
        def wait_until_reachable(self, host: str, port: int, timeout_s: int) -> None:
            raise ValueError("network gremlin")

    assert _run(tmp_path, manifest, api, ExplodingSsh(None), ssh_key) == 1
    assert api.terminated == ["pod-1"]
    adapter = json.loads((tmp_path / "attempt" / "adapter.json").read_text())
    assert "network gremlin" in " ".join(adapter["problems"])


def test_unconfirmed_termination_is_an_error(tmp_path: Path, ssh_key: Path) -> None:
    manifest = _manifest()
    api = FakeApi(terminate_error=True)
    assert _run(tmp_path, manifest, api, FakeSsh(_result(manifest)), ssh_key) == 2
    assert api.terminated == ["pod-1"]
    assert not (tmp_path / "attempt" / "adapter.json").exists()
    assert list((tmp_path / "leases").glob("*.json"))  # lease retained


def test_build_adapter_rejects_cost_above_ceiling(tmp_path: Path) -> None:
    manifest = _manifest()
    args = argparse.Namespace(
        **vars(driver.build_parser().parse_args(_argv(tmp_path, manifest)))
    )
    settings = driver.execution_settings(args, manifest["config"], {})
    outcome = {"status": "completed", "cost_usd": 30.0, "result": _result(manifest)}
    adapter = driver.build_adapter(args, outcome, manifest["config"], settings, 25.0)
    assert adapter["status"] == "failed"
    assert any("exceeded" in problem for problem in adapter["problems"])


def test_poll_fails_fast_when_the_provider_exits_the_pod(tmp_path) -> None:
    """A pod the provider stops must not leave the driver waiting out its lifetime."""
    import benchmarks.flagship_runpod_driver as drv

    class DeadSsh:
        calls = 0

        def download_dir(self, *_a, **_k):
            raise OSError("connection refused")

        def run(self, *_a, **_k):
            DeadSsh.calls += 1
            return 255, ""

    statuses = iter(["RUNNING", "EXITED"])
    with pytest.raises(drv.FlagshipDriverError, match="EXITED"):
        drv.poll_until_exit(
            DeadSsh(),
            tmp_path,
            deadline_s=3600,
            poll_s=0,
            pod_status=lambda: next(statuses),
        )
    assert DeadSsh.calls < 10


def test_poll_without_status_callback_keeps_old_behavior(tmp_path) -> None:
    import benchmarks.flagship_runpod_driver as drv

    class DoneSsh:
        def download_dir(self, *_a, **_k):
            return None

        def run(self, command, *_a, **_k):
            return (0, "0\n") if "cat" in command else (0, "")

    assert drv.poll_until_exit(DoneSsh(), tmp_path, deadline_s=60, poll_s=0) == 0


def test_low_runpod_credit_fails_before_provisioning() -> None:
    import benchmarks.flagship_runpod_driver as drv

    class BrokeApi:
        created = False

        def account_balance(self) -> float:
            return -0.05

        def create_pod(self, **_kw):
            BrokeApi.created = True
            raise AssertionError("must not provision")

    with pytest.raises(drv.FlagshipDriverError, match="top up"):
        drv._require_credit(BrokeApi(), {"worst_case_cost_usd": 19.3})
    assert not BrokeApi.created
