"""Modal call identities and honest cleanup observations without provider calls."""

import json
import multiprocessing
import os
import sys
from pathlib import Path

import pytest

from stateset_agents.remote import modal as modal_provider
from stateset_agents.remote.executor import RemoteExecutionError, SubmissionReceiptError
from stateset_agents.remote.job import JobStatus, RemoteJobSpec
from tests.unit import fake_modal


def install_sdk(monkeypatch, root):
    sdk = fake_modal.build(root / "volumes")
    sdk.calls, sdk.lifecycle = [], []
    sdk.get_error = sdk.spawn_error = sdk.cancel_error = None
    sdk.crash_boundary = None
    sdk.hydrate_supported = True
    from_name = sdk.Volume.from_name
    rewrite_paths = fake_modal.FakeFunction._rewrite_paths

    def rewrite_mount_prefix(value, source, target):
        # A mount applies once at the beginning of a path. The default
        # /outputs mount also occurs in /outputs/outputs/{job_id}; replacing
        # both occurrences would write outside the fake volume's output tree.
        if isinstance(value, str):
            if value == source or value.startswith(source + "/"):
                return target + value[len(source) :]
            return value
        return rewrite_paths(value, source, target)

    monkeypatch.setattr(
        fake_modal.FakeFunction,
        "_rewrite_paths",
        staticmethod(rewrite_mount_prefix),
    )

    def create_volume(name, **kwargs):
        sdk.lifecycle.append("volume_create")
        volume = from_name(name, **kwargs)

        def hydrate():
            sdk.lifecycle.append("hydrate")
            volume.object_id = "vo-" + name
            return volume

        volume.hydrate = hydrate if sdk.hydrate_supported else None
        return volume

    sdk.Volume.from_name = create_volume

    class Call:
        def __init__(self, function, payload):
            self.function, self.payload = function, payload
            self.object_id = "fc-" + str(len(sdk.calls) + 1)
            self.cancelled = False

        def get(self):
            sdk.lifecycle.append("get")
            if sdk.crash_boundary == "after_id":
                os._exit(19)
            if sdk.get_error:
                raise sdk.get_error
            return self.function.remote(self.payload)

        def cancel(self):
            sdk.lifecycle.append("cancel")
            if sdk.cancel_error:
                raise sdk.cancel_error
            self.cancelled = True

    def spawn(function, payload):
        sdk.lifecycle.append("spawn")
        if sdk.spawn_error:
            raise sdk.spawn_error
        call = Call(function, payload)
        sdk.calls.append(call)
        if sdk.crash_boundary == "before_id":
            os._exit(19)
        return call

    def train(payload):
        output = Path(payload["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "adapter_model.safetensors").write_bytes(b"synthetic adapter")
        return {
            "returncode": 0,
            "logs": [],
            "output_dir": str(output),
            "artifacts": ["adapter_model.safetensors"],
        }

    monkeypatch.setitem(sys.modules, "modal", sdk)
    monkeypatch.setattr(fake_modal.FakeFunction, "spawn", spawn)
    monkeypatch.setattr(modal_provider, "MODAL_AVAILABLE", True)
    monkeypatch.setattr(modal_provider, "_remote_entrypoint", train)
    return sdk


def crash_submission(root, boundary):
    root = Path(root)
    patch = pytest.MonkeyPatch()
    sdk = install_sdk(patch, root)
    sdk.crash_boundary = boundary
    executor = modal_provider.ModalExecutor(secret_names=[])

    def retain(event):
        descriptor = os.open(
            root / "receipt.jsonl", os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    executor.set_submission_observer(retain)
    executor.submit(RemoteJobSpec(dataset=root / "data.jsonl", base_model="acme/model"))


@pytest.fixture
def submission(tmp_path, monkeypatch):
    sdk = install_sdk(monkeypatch, tmp_path)
    data = tmp_path / "data.jsonl"
    data.write_text('{"messages":[]}\n')
    spec = RemoteJobSpec(
        dataset=data, base_model="acme/model", output_dir=tmp_path / "artifacts"
    )
    executor = modal_provider.ModalExecutor(secret_names=[])
    events = []
    executor.set_submission_observer(events.append)
    return executor, spec, sdk, events


def test_observed_submit_publishes_hydrated_volume_and_call_before_wait(submission):
    executor, spec, sdk, events = submission

    def observe(event):
        events.append(event)
        if event["event"] == "allocation_intent" and event["resource_kind"] == "volume":
            assert "volume_create" not in sdk.lifecycle
        elif event["event"] == "allocated" and event["resource_kind"] == "volume":
            assert "hydrate" in sdk.lifecycle and "spawn" not in sdk.lifecycle
            assert event["resource_id"].startswith("vo-stateset-sft-")
        elif (
            event["event"] == "allocated" and event["resource_kind"] == "function_call"
        ):
            assert "get" not in sdk.lifecycle and event["resource_id"] == "fc-1"

    executor.set_submission_observer(observe)
    result = executor.wait(executor.submit(spec))
    assert result.status is JobStatus.SUCCEEDED
    assert (
        result.output_dir / "adapter_model.safetensors"
    ).read_bytes() == b"synthetic adapter"
    assert [(event["resource_kind"], event["event"]) for event in events] == [
        ("volume", "allocation_intent"),
        ("volume", "allocated"),
        ("function_call", "allocation_intent"),
        ("function_call", "allocated"),
        ("function_call", "execution_finished"),
        ("volume", "cleanup_acknowledged"),
    ]
    assert all(
        event["schema_version"] == 1 and event["provider"] == "modal"
        for event in events
    )
    assert sdk.calls[0].cancelled is False
    assert not any(event["event"] == "cleanup_confirmed" for event in events)


@pytest.mark.parametrize(
    "kind,event",
    [
        ("volume", "allocation_intent"),
        ("volume", "allocated"),
        ("function_call", "allocation_intent"),
        ("function_call", "allocated"),
        ("function_call", "execution_finished"),
    ],
)
def test_observer_failure_cleans_known_resources_without_replacement_call(
    submission, kind, event
):
    executor, spec, sdk, events = submission
    cause = OSError("PRIVATE receipt failure")

    def observe(value):
        events.append(value)
        if (value["resource_kind"], value["event"]) == (kind, event):
            raise cause

    executor.set_submission_observer(observe)
    with pytest.raises(SubmissionReceiptError) as error:
        executor.submit(spec)
    assert error.value.__cause__ is cause and "PRIVATE" not in str(error.value)
    if event == "allocation_intent" and kind == "volume":
        assert sdk.lifecycle == [] and fake_modal.FakeVolume.deleted_names == []
    else:
        assert len(fake_modal.FakeVolume.deleted_names) == 1
    assert sdk.lifecycle.count("spawn") <= 1
    if kind == "function_call" and event == "allocated":
        assert sdk.calls[0].cancelled and "get" not in sdk.lifecycle
        assert any(
            value["resource_kind"] == "function_call"
            and value["event"] == "cleanup_acknowledged"
            for value in events
        )
    if event == "execution_finished":
        assert not sdk.calls[0].cancelled
    assert executor._jobs == {}


@pytest.mark.parametrize(
    "failure",
    [NotImplementedError("not implemented"), ConnectionError("ambiguous response")],
)
def test_spawn_failure_never_falls_back_to_remote_or_retries(submission, failure):
    executor, spec, sdk, events = submission
    sdk.spawn_error = failure
    with pytest.raises(RemoteExecutionError) as error:
        executor.submit(spec)
    if isinstance(failure, NotImplementedError):
        assert "working Function.spawn" in str(error.value)
    assert sdk.lifecycle.count("spawn") == 1 and "get" not in sdk.lifecycle
    assert sdk.calls == [] and not spec.output_dir.exists()
    assert any(
        event["resource_kind"] == "function_call"
        and event["event"] == "allocation_intent"
        for event in events
    )
    assert not any(
        event["resource_kind"] == "function_call"
        and event["event"] == "cleanup_acknowledged"
        for event in events
    )


def test_unsupported_hydration_sdk_refuses_before_call_and_cleans_volume(submission):
    executor, spec, sdk, _events = submission
    sdk.hydrate_supported = False
    with pytest.raises(RemoteExecutionError, match="Volume.hydrate"):
        executor.submit(spec)
    assert (
        "spawn" not in sdk.lifecycle and len(fake_modal.FakeVolume.deleted_names) == 1
    )


def test_primary_error_survives_failed_call_cancel_volume_cleanup_and_observer(
    submission, monkeypatch
):
    executor, spec, sdk, events = submission
    primary = ValueError("synthetic execution error")
    sdk.get_error, sdk.cancel_error = primary, ConnectionError(
        "PRIVATE cancel response"
    )

    def delete(_name):
        raise ConnectionError("PRIVATE delete response")

    def observe(event):
        events.append(event)
        if event["event"] == "cleanup_unconfirmed":
            raise OSError("PRIVATE cleanup receipt failure")

    monkeypatch.setattr(executor, "_delete_volume", delete)
    executor.set_submission_observer(observe)
    with pytest.raises(RemoteExecutionError) as error:
        executor.submit(spec)
    assert error.value.__cause__ is primary
    assert sdk.lifecycle.count("spawn") == 1 and sdk.lifecycle.count("cancel") == 1
    assert {
        event["resource_kind"]
        for event in events
        if event["event"] == "cleanup_unconfirmed"
    } == {"volume", "function_call"}
    assert "PRIVATE" not in json.dumps(events)


def test_cleanup_receipt_failure_cannot_report_success(submission):
    executor, spec, _sdk, events = submission

    def observe(event):
        events.append(event)
        if event["event"] == "cleanup_acknowledged":
            raise OSError("synthetic receipt failure")

    executor.set_submission_observer(observe)
    result = executor.wait(executor.submit(spec))
    assert result.status is JobStatus.FAILED
    assert (spec.output_dir / "adapter_model.safetensors").exists()


def test_unobserved_legacy_submission_uses_remote_without_spawn(submission):
    executor, spec, sdk, events = submission
    executor.set_submission_observer(None)
    sdk.spawn_error = AssertionError("Legacy submission must not use spawn")
    assert executor.wait(executor.submit(spec)).succeeded
    assert sdk.calls == [] and "spawn" not in sdk.lifecycle and events == []


@pytest.mark.parametrize("boundary", ["before_id", "after_id"])
def test_process_death_retains_modal_volume_and_call_uncertainty(tmp_path, boundary):
    (tmp_path / "data.jsonl").write_text('{"messages":[]}\n')
    process = multiprocessing.get_context("spawn").Process(
        target=crash_submission, args=(str(tmp_path), boundary)
    )
    process.start()
    try:
        process.join(15)
        assert not process.is_alive() and process.exitcode == 19
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
    events = [
        json.loads(line)
        for line in (tmp_path / "receipt.jsonl").read_text().splitlines()
    ]
    volume = next(
        event
        for event in events
        if event["resource_kind"] == "volume" and event["event"] == "allocated"
    )
    assert (tmp_path / "volumes" / volume["resource_name"]).is_dir()
    calls = [event for event in events if event["resource_kind"] == "function_call"]
    assert calls[0]["event"] == "allocation_intent" and calls[0]["resource_id"] is None
    assert not any(event["event"].startswith("cleanup") for event in events)
    if boundary == "before_id":
        assert len(calls) == 1
    else:
        assert calls[1]["event"] == "allocated" and calls[1]["resource_id"] == "fc-1"
