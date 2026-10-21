"""Source bytes consumed at submit must determine later SFT lineage."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from stateset_agents.data.source import DatasetSourceIdentity
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.fireworks import FireworksExecutor
from stateset_agents.remote.job import RemoteJobSpec
from stateset_agents.remote.river import RiverExecutor
from stateset_agents.training.lineage import MANIFEST_NAME
from tests.unit.test_remote_fireworks_executor import FakeFireworks
from tests.unit.test_remote_river_executor import FakeRiverClient
from tests.unit.test_river_batches import FakeTokenizer


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"original request {i}"},
                {"role": "assistant", "content": f"original response {i}"},
            ],
            "metadata": {"split": "train"},
        }
        for i in range(4)
    ]
    path.write_bytes(
        ("\r\n".join(json.dumps(row) for row in rows) + "\r\n\r\n").encode()
    )
    return path


@pytest.fixture
def spec(source, tmp_path):
    return RemoteJobSpec(
        dataset=source,
        base_model="Qwen/Qwen3.5-9B",
        output_dir=tmp_path / "out",
        num_epochs=1,
    )


@pytest.fixture(autouse=True)
def raw_text_renderer(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ValueError("recording fake uses raw-text sampling")

    monkeypatch.setattr(
        "stateset_agents.remote.river_renderers.get_river_renderer", unavailable
    )


def test_source_identity_binds_exact_bytes_and_nonblank_source_lines(source):
    raw = source.read_bytes()
    captured = DatasetSourceIdentity.from_bytes(source, raw)
    assert captured.path == str(source)
    assert captured.sha256 == hashlib.sha256(raw).hexdigest()
    assert captured.rows == 4
    assert DatasetSourceIdentity.from_dict(captured.to_dict()) == captured


def test_metadata_restore_does_not_relax_new_job_configuration_checks(source, spec):
    payload = spec.to_dict()
    assert "_require_dataset" not in payload
    source.unlink()
    with pytest.raises(ValueError, match="dataset does not exist"):
        RemoteJobSpec.from_dict(payload)
    restored = RemoteJobSpec.from_dict(payload, require_dataset=False)
    assert restored.dataset == source
    assert restored.to_dict() == payload
    for key, value in (("num_epochs", 0), ("job_kind", "unknown"), ("seed", True)):
        with pytest.raises(ValueError):
            RemoteJobSpec.from_dict({**payload, key: value}, require_dataset=False)
    with pytest.raises(TypeError):
        RemoteJobSpec.from_dict({**payload, "_require_dataset": False})


@pytest.mark.parametrize("mutation", ["replace", "remove", "repoint_spec"])
def test_river_captures_source_before_client_calls_and_fetch_does_not_reopen(
    source, spec, tmp_path, monkeypatch, mutation
):
    original = source.read_bytes()
    expected = DatasetSourceIdentity.from_bytes(source, original)
    client = FakeRiverClient()
    capabilities = client.get_capabilities

    def change_source():
        if mutation == "remove":
            source.unlink()
        elif mutation == "repoint_spec":
            spec.dataset = tmp_path / "missing-new-dataset.jsonl"
        else:
            source.write_text('{"metadata":{"split":"test"}}\n')
        return capabilities()

    client.get_capabilities = change_source
    executor = RiverExecutor(
        client=client, tokenizer=FakeTokenizer(), ledger_path=tmp_path / "ledger.jsonl"
    )
    handle = executor.submit(spec)
    assert client.model.forward_backward_calls

    def no_late_hash(*args, **kwargs):
        pytest.fail("Source path was reread during fetch")

    monkeypatch.setattr("stateset_agents.training.lineage.hash_dataset", no_late_hash)
    output = executor.fetch(handle)
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert manifest["dataset_sha256"] == expected.sha256
    assert manifest["dataset_rows"] == expected.rows
    assert manifest["dataset_path"] == expected.path
    assert manifest["hyperparameters"]["dataset_provenance"] == "captured_at_submission"


@pytest.mark.parametrize("mutation", ["replace", "remove", "repoint_spec"])
def test_fireworks_upload_and_restarted_fetch_use_captured_source(
    source, spec, tmp_path, monkeypatch, mutation
):
    original = source.read_bytes()
    expected = DatasetSourceIdentity.from_bytes(source, original)
    client = FakeFireworks()
    create = client.datasets.create
    uploaded = []
    temp_paths = []

    def change_source(**kwargs):
        if mutation == "remove":
            source.unlink()
        elif mutation == "repoint_spec":
            spec.dataset = tmp_path / "missing-new-dataset.jsonl"
        else:
            source.write_text('{"metadata":{"split":"test"}}\n')
        return create(**kwargs)

    def record_upload(dataset_id, *, file):
        temp_path = Path(file.name)
        temp_paths.append(temp_path)
        assert stat.S_IMODE(temp_path.stat().st_mode) == 0o600
        uploaded.append(file.read())

    client.datasets.create = change_source
    client.datasets.upload = record_upload
    executor = FireworksExecutor(client=client, ledger_path=tmp_path / "ledger.jsonl")
    handle = executor.submit(spec)
    assert uploaded == [original]
    assert temp_paths and all(not p.exists() for p in temp_paths)
    client.supervised_fine_tuning_jobs.states = ["JOB_STATE_COMPLETED"]
    restarted = FireworksExecutor(
        client=client,
        account_id="acct",
        state_dir=executor.state_dir,
        ledger_path=tmp_path / "ledger.jsonl",
    )

    def no_late_hash(*args, **kwargs):
        pytest.fail("Source path was reread during fetch")

    monkeypatch.setattr("stateset_agents.training.lineage.hash_dataset", no_late_hash)
    output = restarted.fetch(handle)
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert manifest["dataset_sha256"] == expected.sha256
    assert manifest["dataset_rows"] == expected.rows
    assert manifest["dataset_path"] == expected.path
    assert manifest["hyperparameters"]["dataset_provenance"] == "captured_at_submission"


def test_fireworks_failed_upload_removes_private_snapshot(source, spec, tmp_path):
    client = FakeFireworks()
    temp_paths = []

    def fail_upload(dataset_id, *, file):
        temp_paths.append(Path(file.name))
        assert file.read() == source.read_bytes()
        raise OSError("simulated transport failure")

    client.datasets.upload = fail_upload
    executor = FireworksExecutor(client=client, ledger_path=tmp_path / "ledger.jsonl")
    with pytest.raises(RemoteExecutionError, match="could not upload"):
        executor.submit(spec)
    assert temp_paths and all(not path.exists() for path in temp_paths)
    assert not client.supervised_fine_tuning_jobs.created


@pytest.mark.parametrize("provider", ["river", "fireworks"])
@pytest.mark.parametrize("change", ["validation", "test", "duplicate"])
def test_capture_rechecks_declarations_after_shared_preflight(
    source, spec, tmp_path, monkeypatch, provider, change
):
    if provider == "river":
        executor = RiverExecutor(
            client=FakeRiverClient(),
            tokenizer=FakeTokenizer(),
            ledger_path=tmp_path / "ledger.jsonl",
        )
        monkeypatch.setattr(
            executor,
            "_checked_client",
            lambda *_: pytest.fail("Provider setup reached"),
        )
    else:
        executor = FireworksExecutor(
            client=FakeFireworks(), ledger_path=tmp_path / "ledger.jsonl"
        )
        monkeypatch.setattr(
            executor, "_get_client", lambda: pytest.fail("Provider setup reached")
        )
    validate = executor.validate_spec
    original = source.read_text()

    def change_after_preflight(job):
        validate(job)
        replacement = (
            '"split": "test", "split": "train"'
            if change == "duplicate"
            else f'"split": "{change}"'
        )
        source.write_text(original.replace('"split": "train"', replacement))

    monkeypatch.setattr(executor, "validate_spec", change_after_preflight)
    with pytest.raises((ValueError, RemoteExecutionError)):
        executor.submit(spec)
    assert not spec.output_dir.exists()


@pytest.mark.parametrize(
    "bad",
    [
        {"path": "input", "sha256": "a" * 64, "rows": True},
        {"path": "input", "sha256": "a" * 64, "rows": -1},
        {"path": "input", "sha256": "a" * 64, "rows": 1.5},
        {"path": None, "sha256": "a" * 64, "rows": 1},
        {"path": "input", "sha256": "private-value-never-echo", "rows": 1},
        {
            "path": "input",
            "sha256": "a" * 64,
            "rows": 1,
            "extra": "private-value-never-echo",
        },
    ],
)
def test_invalid_retained_source_identity_fails_before_provider_poll(
    source, spec, tmp_path, monkeypatch, bad
):
    client = FakeFireworks()
    executor = FireworksExecutor(client=client, ledger_path=tmp_path / "ledger.jsonl")
    handle = executor.submit(spec)
    state = executor._state_path(handle.job_id)
    payload = json.loads(state.read_text())
    payload["dataset_source"] = bad
    state.write_text(json.dumps(payload))
    restarted = FireworksExecutor(client=client, state_dir=executor.state_dir)
    monkeypatch.setattr(
        client.supervised_fine_tuning_jobs,
        "get",
        lambda *_: pytest.fail("Provider poll reached"),
    )
    with pytest.raises(RemoteExecutionError, match="metadata.*invalid") as error:
        restarted.fetch(handle)
    assert "private-value-never-echo" not in str(error.value)
    assert not spec.output_dir.exists()


def test_legacy_fireworks_state_retains_unknown_lineage_instead_of_hashing_today(
    source, spec, tmp_path, monkeypatch
):
    client = FakeFireworks()
    executor = FireworksExecutor(client=client, ledger_path=tmp_path / "ledger.jsonl")
    handle = executor.submit(spec)
    state = executor._state_path(handle.job_id)
    payload = json.loads(state.read_text())
    del payload["dataset_source"]
    state.write_text(json.dumps(payload))
    source.write_text('{"messages":[]}\n')
    client.supervised_fine_tuning_jobs.states = ["JOB_STATE_COMPLETED"]
    restarted = FireworksExecutor(
        client=client, account_id="acct", state_dir=executor.state_dir
    )
    monkeypatch.setattr(
        "stateset_agents.training.lineage.hash_dataset",
        lambda *_: pytest.fail("Late hash reached"),
    )
    output = restarted.fetch(handle)
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert manifest["dataset_sha256"] is None and manifest["dataset_rows"] is None
    assert (
        manifest["hyperparameters"]["dataset_provenance"]
        == "unavailable_legacy_metadata"
    )
