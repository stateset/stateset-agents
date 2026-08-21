"""Tests for ``FireworksExecutor`` — against a fake Fireworks client.

Unlike River, this adapter was written against the real ``fireworks-ai``
SDK surface (typed resources, generated from Fireworks' OpenAPI spec), so
the fakes below mirror actual method signatures and response models rather
than a documented guess. What is still unverified is the *service*
behaviour — state transition timing, whether a PEFT addon's weights are
downloadable, and the deploy/addon-load sequence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.fireworks import (
    CHECKPOINT_POINTER_NAME,
    FIREWORKS_ACCOUNT_ENV,
    FIREWORKS_API_KEY_ENV,
    FireworksExecutor,
)
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec
from stateset_agents.remote.registry import available_providers, get_executor
from stateset_agents.training.lineage import MANIFEST_NAME


# --- fakes mirroring the fireworks-ai SDK -------------------------------


@dataclass
class FakeDatasets:
    created: list[dict[str, Any]] = field(default_factory=list)
    uploaded: list[tuple[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        return SimpleNamespace(
            name=f"datasets/{kwargs['dataset_id']}", state="UPLOADING"
        )

    def upload(self, dataset_id: str, *, file: Any = None, **_: Any) -> Any:
        self.uploaded.append((dataset_id, file))
        return SimpleNamespace(id=dataset_id, filename="train.jsonl")


@dataclass
class FakeSFTJobs:
    #: States returned by successive ``get()`` calls; the last one sticks.
    states: list[str] = field(
        default_factory=lambda: ["JOB_STATE_RUNNING", "JOB_STATE_COMPLETED"]
    )
    created: list[dict[str, Any]] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    polls: int = 0
    output_model: str = "accounts/acct/models/sft-abc"

    def _job(self, state: str) -> Any:
        return SimpleNamespace(
            name="accounts/acct/supervisedFineTuningJobs/sftj-1",
            state=state,
            output_model=self.output_model,
            base_model="Qwen/Qwen3.5-9B",
            lora_rank=16,
            epochs=2,
            status=SimpleNamespace(code="OK", message=None),
            job_progress=SimpleNamespace(percent=50, epoch=1, output_tokens=1234),
            estimated_cost=SimpleNamespace(
                currency_code="USD", units="3", nanos=250000000
            ),
            create_time=None,
            completed_time=None,
        )

    def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        return self._job("JOB_STATE_CREATING")

    def get(self, job_id: str, **_: Any) -> Any:
        state = self.states[min(self.polls, len(self.states) - 1)]
        self.polls += 1
        return self._job(state)

    def delete(self, job_id: str, **_: Any) -> Any:
        self.deleted.append(job_id)
        return {}


@dataclass
class FakeDeployments:
    created: list[dict[str, Any]] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        return SimpleNamespace(name="accounts/acct/deployments/dep-1", state="CREATING")

    def delete(self, deployment_id: str, **_: Any) -> Any:
        self.deleted.append(deployment_id)
        return {}


@dataclass
class FakeLora:
    loaded: list[dict[str, Any]] = field(default_factory=list)

    def load(self, **kwargs: Any) -> Any:
        self.loaded.append(kwargs)
        return SimpleNamespace(name="accounts/acct/deployedModels/dm-1")


@dataclass
class FakeModels:
    urls: dict[str, str] | None = None

    def get_download_endpoint(self, model_id: str, **_: Any) -> Any:
        return SimpleNamespace(filename_to_signed_urls=self.urls)


class FakeFireworks:
    """Stand-in for ``fireworks.Fireworks``."""

    def __init__(self, urls: dict[str, str] | None = None) -> None:
        self.datasets = FakeDatasets()
        self.supervised_fine_tuning_jobs = FakeSFTJobs()
        self.deployments = FakeDeployments()
        self.lora = FakeLora()
        self.models = FakeModels(urls=urls)


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv(FIREWORKS_API_KEY_ENV, "fw_test_key")
    monkeypatch.setenv(FIREWORKS_ACCOUNT_ENV, "acct")


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "curated.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"question {i}"},
                        {"role": "assistant", "content": f"answer {i}"},
                    ]
                }
            )
            for i in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def spec(dataset, tmp_path):
    return RemoteJobSpec(
        dataset=dataset,
        base_model="Qwen/Qwen3.5-9B",
        output_dir=tmp_path / "out",
        num_epochs=2,
        lora_r=16,
        learning_rate=1e-4,
        max_length=2048,
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
    )


@pytest.fixture
def client():
    return FakeFireworks()


@pytest.fixture
def executor(client, tmp_path):
    return FireworksExecutor(client=client, ledger_path=tmp_path / "ledger.jsonl")


# --- registry -----------------------------------------------------------


def test_fireworks_is_a_registered_provider():
    assert "fireworks" in available_providers()


def test_registry_constructs_the_fireworks_executor():
    assert isinstance(get_executor("fireworks"), FireworksExecutor)


# --- submit -------------------------------------------------------------


def test_submit_uploads_the_dataset_then_creates_the_job(executor, client, spec):
    handle = executor.submit(spec)

    assert handle.provider == "fireworks"
    assert handle.job_id == "sftj-1"
    created = client.datasets.created[0]
    assert created["dataset"]["format"] == "CHAT"
    assert created["dataset"]["example_count"] == "4"
    assert client.datasets.uploaded[0][0] == created["dataset_id"]


def test_submit_maps_spec_hyperparameters_onto_the_job(executor, client, spec):
    executor.submit(spec)

    job = client.supervised_fine_tuning_jobs.created[0]
    assert job["base_model"] == "Qwen/Qwen3.5-9B"
    assert job["epochs"] == 2
    assert job["lora_rank"] == 16
    assert job["learning_rate"] == 1e-4
    assert job["max_context_length"] == 2048
    assert job["batch_size"] == 2
    assert job["gradient_accumulation_steps"] == 4
    assert job["dataset"] == client.datasets.created[0]["dataset_id"]


def test_submit_without_an_api_key_names_the_variable(monkeypatch, tmp_path, spec):
    monkeypatch.delenv(FIREWORKS_API_KEY_ENV, raising=False)
    # No injected client: this is the path that must read the environment.
    executor = FireworksExecutor(ledger_path=tmp_path / "ledger.jsonl")

    with pytest.raises(RemoteExecutionError, match=FIREWORKS_API_KEY_ENV):
        executor.submit(spec)


def test_submit_without_an_account_id_names_the_variable(monkeypatch, tmp_path, spec):
    monkeypatch.delenv(FIREWORKS_ACCOUNT_ENV, raising=False)
    executor = FireworksExecutor(ledger_path=tmp_path / "ledger.jsonl")

    with pytest.raises(RemoteExecutionError, match=FIREWORKS_ACCOUNT_ENV):
        executor.submit(spec)


def test_submit_rejects_a_dataset_row_without_messages(executor, spec, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"prompt": "hi"}) + "\n", encoding="utf-8")
    spec.dataset = bad

    with pytest.raises(RemoteExecutionError, match="messages"):
        executor.submit(spec)


def test_submit_reports_machine_shaped_spec_fields_as_ignored(executor, spec, caplog):
    spec.gpu = "H100"
    spec.container_disk_gb = 200

    with caplog.at_level("INFO"):
        executor.submit(spec)

    ignored = "\n".join(caplog.messages)
    assert "gpu" in ignored and "container_disk_gb" in ignored


# --- status / logs / cost ----------------------------------------------


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("JOB_STATE_PENDING", JobStatus.PENDING),
        ("JOB_STATE_CREATING", JobStatus.PENDING),
        ("JOB_STATE_VALIDATING", JobStatus.PENDING),
        ("JOB_STATE_RUNNING", JobStatus.RUNNING),
        ("JOB_STATE_WRITING_RESULTS", JobStatus.RUNNING),
        ("JOB_STATE_COMPLETED", JobStatus.SUCCEEDED),
        ("JOB_STATE_EARLY_STOPPED", JobStatus.SUCCEEDED),
        ("JOB_STATE_FAILED", JobStatus.FAILED),
        ("JOB_STATE_EXPIRED", JobStatus.FAILED),
        ("JOB_STATE_CANCELLED", JobStatus.CANCELLED),
    ],
)
def test_status_maps_fireworks_job_states(executor, client, spec, state, expected):
    handle = executor.submit(spec)
    client.supervised_fine_tuning_jobs.states = [state]

    assert executor.status(handle) is expected


def test_status_of_an_unknown_job_still_queries_the_api(executor, client):
    client.supervised_fine_tuning_jobs.states = ["JOB_STATE_RUNNING"]

    assert executor.status(JobHandle("fireworks", "sftj-unseen")) is JobStatus.RUNNING


def test_logs_report_progress_observed_while_polling(executor, client, spec):
    handle = executor.submit(spec)
    executor.status(handle)

    lines = list(executor.logs(handle))
    assert any("50%" in line for line in lines)


def test_job_cost_converts_fireworks_units_and_nanos_to_dollars(executor, client, spec):
    handle = executor.submit(spec)
    client.supervised_fine_tuning_jobs.states = ["JOB_STATE_COMPLETED"]
    executor.status(handle)

    _duration, cost = executor.job_cost(handle)
    assert cost == pytest.approx(3.25)


# --- fetch --------------------------------------------------------------


def test_fetch_refuses_an_unfinished_job(executor, client, spec):
    handle = executor.submit(spec)
    client.supervised_fine_tuning_jobs.states = ["JOB_STATE_RUNNING"]

    with pytest.raises(RemoteExecutionError, match="not finished"):
        executor.fetch(handle)


def test_fetch_writes_a_checkpoint_pointer_and_manifest(executor, client, spec):
    handle = executor.submit(spec)
    client.supervised_fine_tuning_jobs.states = ["JOB_STATE_COMPLETED"]

    out = executor.fetch(handle)

    pointer = json.loads((out / CHECKPOINT_POINTER_NAME).read_text())
    assert pointer["provider"] == "fireworks"
    assert pointer["model"] == "accounts/acct/models/sft-abc"
    assert pointer["base_model"] == "Qwen/Qwen3.5-9B"
    assert pointer["weights_downloaded"] is False
    assert (out / MANIFEST_NAME).exists()


def test_fetch_downloads_addon_weights_when_the_api_offers_them(
    tmp_path, spec, monkeypatch
):
    client = FakeFireworks(urls={"adapter_model.safetensors": "https://signed/x"})
    executor = FireworksExecutor(client=client, ledger_path=tmp_path / "ledger.jsonl")
    monkeypatch.setattr(
        "stateset_agents.remote.fireworks._download",
        lambda url, dest: dest.write_bytes(b"weights"),
    )
    handle = executor.submit(spec)
    client.supervised_fine_tuning_jobs.states = ["JOB_STATE_COMPLETED"]

    out = executor.fetch(handle)

    assert (out / "adapter_model.safetensors").read_bytes() == b"weights"
    pointer = json.loads((out / CHECKPOINT_POINTER_NAME).read_text())
    assert pointer["weights_downloaded"] is True


def test_fetch_keeps_the_pointer_when_the_download_fails(tmp_path, spec, monkeypatch):
    client = FakeFireworks(urls={"adapter_model.safetensors": "https://signed/x"})
    executor = FireworksExecutor(client=client, ledger_path=tmp_path / "ledger.jsonl")

    def boom(url, dest):
        raise OSError("connection reset")

    monkeypatch.setattr("stateset_agents.remote.fireworks._download", boom)
    handle = executor.submit(spec)
    client.supervised_fine_tuning_jobs.states = ["JOB_STATE_COMPLETED"]

    out = executor.fetch(handle)

    pointer = json.loads((out / CHECKPOINT_POINTER_NAME).read_text())
    assert pointer["weights_downloaded"] is False


def test_fetch_of_a_job_this_process_did_not_submit_is_refused(executor, client):
    client.supervised_fine_tuning_jobs.states = ["JOB_STATE_COMPLETED"]

    with pytest.raises(RemoteExecutionError, match="submitted by this process"):
        executor.fetch(JobHandle("fireworks", "sftj-unseen"))


# --- cancel / deploy ----------------------------------------------------


def test_cancel_deletes_the_running_job(executor, client, spec):
    handle = executor.submit(spec)

    executor.cancel(handle)

    assert client.supervised_fine_tuning_jobs.deleted == ["sftj-1"]


def test_deploy_creates_a_deployment_with_addons_and_loads_the_lora(
    executor, client, spec
):
    handle = executor.submit(spec)
    client.supervised_fine_tuning_jobs.states = ["JOB_STATE_COMPLETED"]

    result = executor.deploy(handle, accelerator_type="NVIDIA_H100_80GB")

    deployment = client.deployments.created[0]
    assert deployment["base_model"] == "Qwen/Qwen3.5-9B"
    assert deployment["enable_addons"] is True
    assert deployment["accelerator_type"] == "NVIDIA_H100_80GB"
    assert client.lora.loaded[0]["model"] == "accounts/acct/models/sft-abc"
    assert client.lora.loaded[0]["deployment"] == "accounts/acct/deployments/dep-1"
    assert result["deployment"] == "accounts/acct/deployments/dep-1"
    assert result["model"] == "accounts/acct/models/sft-abc"
    assert result["base_url"].startswith("https://")


def test_deploy_refuses_a_job_that_has_not_finished(executor, client, spec):
    handle = executor.submit(spec)
    client.supervised_fine_tuning_jobs.states = ["JOB_STATE_RUNNING"]

    with pytest.raises(RemoteExecutionError, match="not finished"):
        executor.deploy(handle)


def test_undeploy_deletes_the_deployment(executor, client):
    executor.undeploy("accounts/acct/deployments/dep-1")

    assert client.deployments.deleted == ["dep-1"]


def test_wait_polls_on_a_managed_service_cadence_not_once_a_second():
    """A managed fine-tune changes state on the order of minutes.

    Inheriting the 1s default would hammer the control plane for an hour to
    learn nothing, so the executor widens it.
    """
    import inspect

    default = (
        inspect.signature(FireworksExecutor.wait).parameters["poll_interval_s"].default
    )
    assert default >= 10.0


def test_a_partial_addon_download_leaves_no_half_set_of_weights(
    tmp_path, spec, monkeypatch
):
    """A directory holding one of two shards is worse than holding neither.

    `serve --checkpoint` would find weights, try to load them, and fail
    somewhere far from the download that actually went wrong.
    """
    client = FakeFireworks(
        urls={
            "adapter_config.json": "https://signed/a",
            "adapter_model.safetensors": "https://signed/b",
        }
    )
    executor = FireworksExecutor(client=client, ledger_path=tmp_path / "ledger.jsonl")

    def fail_on_the_second(url, dest):
        if url.endswith("/b"):
            raise OSError("connection reset")
        dest.write_bytes(b"{}")

    monkeypatch.setattr(
        "stateset_agents.remote.fireworks._download", fail_on_the_second
    )
    handle = executor.submit(spec)
    client.supervised_fine_tuning_jobs.states = ["JOB_STATE_COMPLETED"]

    out = executor.fetch(handle)

    assert not (out / "adapter_config.json").exists()
    assert (
        json.loads((out / CHECKPOINT_POINTER_NAME).read_text())["weights_downloaded"]
        is False
    )
