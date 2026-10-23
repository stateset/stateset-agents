"""Durable allocation observations without contacting RunPod or running models."""

import hashlib
import json
import multiprocessing
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from stateset_agents.remote.executor import RemoteExecutionError, SubmissionReceiptError
from stateset_agents.remote.job import JobStatus, RemoteJobSpec
from stateset_agents.remote.runpod import RunPodApi, RunPodExecutor
from tests.unit.test_remote_runpod_executor import FakePodApi, FakeSsh


def append_receipt(path, event):
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def crash_submission(root, boundary):
    root = Path(root)

    class Api(FakePodApi):
        def create_pod(self, **kwargs):
            result = super().create_pod(**kwargs)
            if boundary == "before_id":
                os._exit(17)
            return result

    class Ssh(FakeSsh):
        def wait_until_reachable(self, *_args):
            os._exit(17)

    executor = RunPodExecutor(
        api=Api(),
        ssh=Ssh(),
        public_key="synthetic-key",
        lease_dir=root / "leases",
        poll_interval_s=0,
        allow_environment_wheel=False,
    )
    executor.set_submission_observer(
        lambda event: append_receipt(root / "receipt.jsonl", event)
    )
    executor.submit(RemoteJobSpec(dataset=root / "data.jsonl", base_model="acme/model"))


@pytest.fixture
def submission(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{"messages":[]}\n')
    spec = RemoteJobSpec(
        dataset=data,
        base_model="acme/model",
        output_dir=tmp_path / "artifacts",
        timeout_s=600,
    )
    api, ssh = FakePodApi(), FakeSsh()
    executor = RunPodExecutor(
        api=api,
        ssh=ssh,
        public_key="PRIVATE synthetic public key",
        lease_dir=tmp_path / "leases",
        ledger_path=tmp_path / "ledger.jsonl",
        poll_interval_s=0,
        max_provision_attempts=3,
        allow_environment_wheel=False,
    )
    events = []
    executor.set_submission_observer(events.append)
    return executor, spec, api, ssh, events


def cleanup_clock(monkeypatch, api, durations):
    """Make each owned allocation's observed interval deterministic."""
    clock = [100.0]
    remaining = iter(durations)
    original_terminate = api.terminate_pod
    monkeypatch.setattr(
        "stateset_agents.remote.runpod.time.monotonic", lambda: clock[0]
    )

    def terminate(pod_id):
        clock[0] += next(remaining)
        return original_terminate(pod_id)

    monkeypatch.setattr(api, "terminate_pod", terminate)


def cost_rows(executor):
    return [json.loads(line) for line in executor.ledger_path.read_text().splitlines()]


def test_submit_retains_identity_before_work_and_distinguishes_cleanup_acknowledgement(
    submission,
):
    executor, spec, api, ssh, events = submission
    original_create = api.create_pod
    original_wait = ssh.wait_until_reachable

    def create(**kwargs):
        assert events[-1]["event"] == "allocation_intent"
        assert events[-1]["resource_name"] == kwargs["name"]
        assert events[-1]["resource_id"] is None
        return original_create(**kwargs)

    def wait(*args):
        assert events[-1]["event"] == "allocated"
        assert events[-1]["resource_id"] == "pod-abc"
        return original_wait(*args)

    api.create_pod, ssh.wait_until_reachable = create, wait
    result = executor.wait(executor.submit(spec))
    assert result.status is JobStatus.SUCCEEDED and result.output_dir.exists()
    assert [event["event"] for event in events] == [
        "allocation_intent",
        "allocated",
        "execution_finished",
        "cleanup_acknowledged",
    ]
    assert all(
        event["provider"] == "runpod" and event["schema_version"] == 1
        for event in events
    )
    assert all(
        event["attempt"] == 1 and event["resource_kind"] == "pod" for event in events
    )
    assert "PRIVATE" not in json.dumps(events) and "data.jsonl" not in json.dumps(
        events
    )
    assert api.terminated == ["pod-abc"] and executor.leases() == []


@pytest.mark.parametrize(
    "failure_event",
    ["allocation_intent", "allocated", "execution_finished", "cleanup_acknowledged"],
)
def test_receipt_failure_fails_closed_without_automatic_reprovision(
    submission, failure_event
):
    executor, spec, api, ssh, events = submission
    cause = OSError("PRIVATE storage failure")

    def observe(event):
        events.append(event)
        if event["event"] == failure_event:
            raise cause

    executor.set_submission_observer(observe)
    with pytest.raises(SubmissionReceiptError) as error:
        executor.submit(spec)
    assert error.value.__cause__ is cause and "PRIVATE" not in str(error.value)
    assert len(api.created) == (0 if failure_event == "allocation_intent" else 1)
    assert api.terminated == (
        [] if failure_event == "allocation_intent" else ["pod-abc"]
    )
    if failure_event in {"allocation_intent", "allocated"}:
        assert ssh.uploaded == []
    assert executor._jobs == {}
    if failure_event != "allocation_intent":
        assert executor._last_duration_s is not None
        ledger = [
            json.loads(line) for line in executor.ledger_path.read_text().splitlines()
        ]
        assert len(ledger) == 1 and ledger[0]["status"] == "failed"
        assert ledger[0]["duration_s"] >= 0
    else:
        assert executor._last_duration_s is None and not executor.ledger_path.exists()


def test_primary_transport_failure_survives_cleanup_receipt_failure_without_retry(
    submission,
):
    executor, spec, api, ssh, events = submission
    ssh.run_failures_left = 1

    def observe(event):
        events.append(event)
        if event["event"] == "cleanup_acknowledged":
            raise OSError("synthetic receipt failure")

    executor.set_submission_observer(observe)
    with pytest.raises(RemoteExecutionError, match="failed mid-job") as error:
        executor.submit(spec)
    assert not isinstance(error.value, SubmissionReceiptError)
    assert isinstance(error.value.__cause__, OSError)
    assert len(api.created) == 1 and api.terminated == ["pod-abc"]


def test_budget_refusal_still_reports_allocated_pod_cleanup(submission):
    executor, spec, api, ssh, events = submission
    api.cost_per_hr, spec.max_cost_usd = 100.0, 0.01
    with pytest.raises(RemoteExecutionError):
        executor.submit(spec)
    assert [event["event"] for event in events] == [
        "allocation_intent",
        "allocated",
        "cleanup_acknowledged",
    ]
    assert api.terminated == ["pod-abc"] and ssh.uploaded == []


def test_failed_lease_retains_original_error_and_reports_acknowledged_cleanup(
    submission, monkeypatch
):
    executor, spec, api, ssh, events = submission
    primary = RemoteExecutionError("synthetic lease failure")

    def fail(*_args):
        raise primary

    monkeypatch.setattr(executor, "_write_lease", fail)
    with pytest.raises(RemoteExecutionError) as error:
        executor.submit(spec)
    assert error.value is primary and api.terminated == ["pod-abc"]
    assert [event["event"] for event in events] == [
        "allocation_intent",
        "cleanup_acknowledged",
    ]
    assert events[-1]["resource_id"] == "pod-abc" and ssh.uploaded == []


def test_malformed_allocated_identity_retains_intent_without_inventing_an_id(
    submission,
):
    executor, spec, api, ssh, events = submission

    def create(**kwargs):
        api.created.append(kwargs)
        return {"id": {"PRIVATE": "untrusted provider response"}}

    api.create_pod = create
    with pytest.raises(RemoteExecutionError, match="allocation may exist") as error:
        executor.submit(spec)
    assert len(api.created) == 1 and api.terminated == [] and ssh.uploaded == []
    assert len(events) == 1 and events[0]["resource_id"] is None
    assert "PRIVATE" not in str(error.value) and "PRIVATE" not in json.dumps(events)
    assert executor._last_duration_s is None and not executor.ledger_path.exists()


def test_failed_termination_records_uncertainty_and_retains_cleanup_lease(submission):
    executor, spec, api, _ssh, events = submission

    def fail(_pod_id):
        raise ConnectionError("PRIVATE provider response")

    api.terminate_pod = fail
    executor.submit(spec)
    assert events[-1]["event"] == "cleanup_unconfirmed"
    assert "PRIVATE" not in json.dumps(events)
    assert executor.leases()[0]["pod_id"] == "pod-abc"


def test_existing_transport_retry_has_separate_attempt_receipts(submission):
    executor, spec, api, ssh, events = submission
    ssh.run_failures_left = 1
    assert executor.wait(executor.submit(spec)).succeeded
    allocated = [event for event in events if event["event"] == "allocated"]
    assert [event["attempt"] for event in allocated] == [1, 2]
    assert [event["resource_id"] for event in allocated] == ["pod-abc", "pod-abc2"]
    assert api.terminated == ["pod-abc", "pod-abc2"]


@pytest.mark.parametrize("boundary", ["before_id", "after_id"])
def test_process_death_retains_intent_or_identity_without_false_cleanup(
    tmp_path, boundary
):
    (tmp_path / "data.jsonl").write_text('{"messages":[]}\n')
    process = multiprocessing.get_context("spawn").Process(
        target=crash_submission, args=(str(tmp_path), boundary)
    )
    process.start()
    try:
        process.join(15)
        assert not process.is_alive() and process.exitcode == 17
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
    events = [
        json.loads(line)
        for line in (tmp_path / "receipt.jsonl").read_text().splitlines()
    ]
    assert events[0]["event"] == "allocation_intent"
    assert events[0]["resource_name"].startswith("stateset-sft-")
    assert not any(event["event"].startswith("cleanup") for event in events)
    if boundary == "before_id":
        assert len(events) == 1 and events[0]["resource_id"] is None
    else:
        assert (
            events[1]["event"] == "allocated" and events[1]["resource_id"] == "pod-abc"
        )
        assert (
            RunPodExecutor(lease_dir=tmp_path / "leases").leases()[0]["pod_id"]
            == "pod-abc"
        )


def test_reviewed_release_can_disable_environment_wheel_without_disabling_explicit_wheel(
    tmp_path, monkeypatch
):
    wheel = tmp_path / "unreviewed.whl"
    monkeypatch.setenv("STATESET_AGENTS_WHEEL", str(wheel))
    assert RunPodExecutor().wheel == wheel
    assert RunPodExecutor(allow_environment_wheel=False).wheel is None
    assert RunPodExecutor(wheel=wheel, allow_environment_wheel=False).wheel == wheel


def test_submission_observer_is_optional_and_capability_is_explicit(submission):
    executor, spec, _api, _ssh, events = submission
    with pytest.raises(TypeError, match="callable"):
        executor.set_submission_observer(123)
    executor.set_submission_observer(None)
    assert executor.wait(executor.submit(spec)).succeeded
    assert (
        events == [] and executor.capabilities()["durable_submission_receipts"] is True
    )


@pytest.mark.parametrize(
    "digest,size,wheel",
    [
        ("a" * 64, None, True),
        (None, 12, True),
        ("wrong", 12, True),
        ("a" * 64, True, True),
        ("a" * 64, 0, True),
        ("a" * 64, 64 * 1024**2 + 1, True),
        ("a" * 64, 12, False),
    ],
)
def test_incomplete_reviewed_binding_is_rejected(tmp_path, digest, size, wheel):
    with pytest.raises(ValueError, match="Reviewed wheel requires"):
        RunPodExecutor(
            wheel=tmp_path / "wheel.whl" if wheel else None,
            allow_environment_wheel=False,
            reviewed_wheel_sha256=digest,
            reviewed_wheel_bytes=size,
        )


@pytest.mark.parametrize("mutation", ["bytes", "symlink", "fifo", "missing"])
def test_changed_or_special_reviewed_wheel_refuses_before_client_and_pod(
    submission, tmp_path, monkeypatch, mutation
):
    executor, spec, api, ssh, events = submission
    wheel = tmp_path / "reviewed.whl"
    original = b"reviewed fixture archive"
    wheel.write_bytes(original)
    executor.bind_reviewed_wheel(
        wheel, sha256=hashlib.sha256(original).hexdigest(), size_bytes=len(original)
    )
    wheel.unlink()
    if mutation == "bytes":
        wheel.write_bytes(b"changed! fixture archive")
    elif mutation == "symlink":
        outside = tmp_path / "outside.whl"
        outside.write_bytes(original)
        try:
            wheel.symlink_to(outside)
        except OSError:
            pytest.skip("Symlinks unavailable")
    elif mutation == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFOs unavailable")
        os.mkfifo(wheel, 0o600)
    monkeypatch.setattr(
        executor,
        "_require_api",
        lambda: pytest.fail("Client accessed before wheel verification"),
    )
    with pytest.raises(RemoteExecutionError, match="Reviewed wheel bytes"):
        executor.submit(spec)
    assert api.created == [] and ssh.uploaded == [] and events == []


@pytest.mark.parametrize("mutate_upload", [False, True])
@pytest.mark.parametrize("job_kind", ["sft", "harvest"])
def test_uploaded_reviewed_wheel_is_checked_remotely_before_pip(
    submission, tmp_path, mutate_upload, job_kind
):
    executor, spec, api, _ssh, events = submission
    wheel = tmp_path / "reviewed '$ archive.whl"
    original = b"reviewed fixture archive"
    wheel.write_bytes(original)
    executor.bind_reviewed_wheel(
        wheel, sha256=hashlib.sha256(original).hexdigest(), size_bytes=len(original)
    )
    spec.job_kind, spec.dry_run = job_kind, True

    class WheelSsh(FakeSsh):
        def upload(self, local, remote):
            if local == wheel and mutate_upload:
                local.write_bytes(b"X" * len(original))
            return super().upload(local, remote)

        def run(self, command):
            arguments = shlex.split(command)
            if arguments[:2] == ["python", "-c"]:
                self.commands.append(command)
                remote = arguments[3]
                assert remote == "/workspace/" + wheel.name
                local = tmp_path / "remote-copy.whl"
                local.write_bytes(self.remote_files[remote])
                checked = subprocess.run(
                    [sys.executable, "-c", arguments[2], str(local), *arguments[4:]],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                return checked.returncode, checked.stdout + checked.stderr
            return super().run(command)

    ssh = WheelSsh()
    executor._ssh = ssh
    result = executor.wait(executor.submit(spec))
    assert result.status is (JobStatus.FAILED if mutate_upload else JobStatus.SUCCEEDED)
    assert len(api.created) == 1 and api.terminated == ["pod-abc"]
    assert shlex.split(ssh.commands[0])[:2] == ["python", "-c"]
    assert any(command.startswith("pip install") for command in ssh.commands) is (
        not mutate_upload
    )
    if mutate_upload:
        assert len(ssh.commands) == 1
    assert events[-1]["event"] == "cleanup_acknowledged"


def test_review_binding_clears_environment_and_cannot_change_after_submission_starts(
    submission, tmp_path, monkeypatch
):
    executor, spec, _api, _ssh, events = submission
    wheel = tmp_path / "ambient.whl"
    monkeypatch.setenv("STATESET_AGENTS_WHEEL", str(wheel))
    ambient = RunPodExecutor()
    assert ambient.wheel == wheel
    ambient.bind_reviewed_wheel(None, sha256=None, size_bytes=None)
    assert ambient.wheel is None and ambient.reviewed_wheel_sha256 is None

    def observe(event):
        events.append(event)
        with pytest.raises(ValueError, match="after submission starts"):
            executor.bind_reviewed_wheel(None, sha256=None, size_bytes=None)

    executor.set_submission_observer(observe)
    executor.submit(spec)
    with pytest.raises(ValueError, match="after submission starts"):
        executor.bind_reviewed_wheel(None, sha256=None, size_bytes=None)


def test_explicit_review_binder_cannot_silently_disable_archive_verification(
    submission, tmp_path
):
    executor, _spec, _api, _ssh, _events = submission
    with pytest.raises(ValueError, match="requires both SHA256 and byte count"):
        executor.bind_reviewed_wheel(
            tmp_path / "unbound.whl", sha256=None, size_bytes=None
        )
    assert executor.wheel is None and executor.reviewed_wheel_sha256 is None


def test_uncertain_cleanup_blocks_automatic_replacement_for_observed_submission(
    submission,
):
    executor, spec, api, ssh, events = submission
    ssh.run_failures_left = 1

    def unavailable_cleanup(pod_id):
        raise OSError("synthetic termination unavailable")

    api.terminate_pod = unavailable_cleanup
    with pytest.raises(RemoteExecutionError, match="cleanup is unconfirmed"):
        executor.submit(spec)
    assert len(api.created) == 1
    assert events[-1]["event"] == "cleanup_unconfirmed"
    assert executor.leases()[0]["pod_id"] == "pod-abc"
    assert executor._jobs == {}
    rows = [json.loads(line) for line in executor.ledger_path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["status"] == "failed"


@pytest.mark.parametrize("exhausted", [False, True])
def test_each_allocated_retry_records_its_own_cost_once(
    submission, monkeypatch, exhausted
):
    executor, spec, api, ssh, events = submission
    executor.max_provision_attempts = 2
    api.cost_per_hr = 3600.0
    ssh.run_failures_left = 2 if exhausted else 1
    cleanup_clock(monkeypatch, api, [2.0, 3.0])

    if exhausted:
        with pytest.raises(RemoteExecutionError, match="every attempt"):
            executor.submit(spec)
        assert executor._jobs == {}
    else:
        handle = executor.submit(spec)
        result = executor.wait(handle)
        assert result.succeeded
        assert executor.job_cost(handle) == (5.0, 5.0)
        assert (result.duration_s, result.cost_usd) == (5.0, 5.0)

    rows = cost_rows(executor)
    assert len(rows) == 2
    assert [row["duration_s"] for row in rows] == [2.0, 3.0]
    assert [row["cost_usd"] for row in rows] == [2.0, 3.0]
    assert [row["status"] for row in rows] == [
        "failed",
        "failed" if exhausted else "succeeded",
    ]
    assert len({row["job_id"] for row in rows}) == 1
    assert api.terminated == ["pod-abc", "pod-abc2"]
    assert [event["attempt"] for event in events if event["event"] == "allocated"] == [
        1,
        2,
    ]


@pytest.mark.parametrize("boundary", ["lease", "allocated_receipt", "budget"])
def test_prework_failure_records_owned_allocation_once_and_preserves_diagnosis(
    submission, monkeypatch, boundary
):
    executor, spec, api, ssh, events = submission
    api.cost_per_hr = 3600.0
    cleanup_clock(monkeypatch, api, [4.0])
    primary = OSError("synthetic private persistence failure")
    if boundary == "lease":

        def fail_lease(*_args):
            raise primary

        monkeypatch.setattr(executor, "_write_lease", fail_lease)
    elif boundary == "allocated_receipt":

        def observe(event):
            events.append(event)
            if event["event"] == "allocated":
                raise primary

        executor.set_submission_observer(observe)
    else:
        spec.max_cost_usd = 0.01

    with pytest.raises((OSError, RemoteExecutionError)) as caught:
        executor.submit(spec)
    if boundary == "lease":
        assert caught.value is primary
    elif boundary == "allocated_receipt":
        assert isinstance(caught.value, SubmissionReceiptError)
        assert caught.value.__cause__ is primary
    else:
        assert "above the --max-cost ceiling" in str(caught.value)
    rows = cost_rows(executor)
    assert len(rows) == 1
    assert (rows[0]["duration_s"], rows[0]["cost_usd"], rows[0]["status"]) == (
        4.0,
        4.0,
        "failed",
    )
    assert len(api.created) == 1 and api.terminated == ["pod-abc"]
    assert ssh.uploaded == [] and executor._jobs == {}
    assert events[-1]["event"] == "cleanup_acknowledged"


@pytest.mark.parametrize("quote", ["NaN", float("inf"), float("-inf"), -1.0])
def test_invalid_quote_refuses_budget_before_work_and_retains_unknown_cost(
    submission, monkeypatch, quote
):
    executor, spec, api, ssh, _events = submission
    api.cost_per_hr, spec.max_cost_usd = quote, 10.0
    cleanup_clock(monkeypatch, api, [2.0])
    with pytest.raises(RemoteExecutionError, match="did not report a price"):
        executor.submit(spec)
    rows = cost_rows(executor)
    assert len(rows) == 1 and rows[0]["duration_s"] == 2.0
    assert rows[0]["cost_usd"] is None and rows[0]["cost_per_hr"] is None
    assert rows[0]["status"] == "failed"
    assert api.terminated == ["pod-abc"] and ssh.uploaded == []


def test_unknown_price_on_prior_attempt_keeps_logical_result_cost_unknown(
    submission, monkeypatch
):
    executor, spec, api, ssh, _events = submission
    original_create = api.create_pod

    def create(**kwargs):
        api.cost_per_hr = None if not api.created else 3600.0
        return original_create(**kwargs)

    monkeypatch.setattr(api, "create_pod", create)
    cleanup_clock(monkeypatch, api, [2.0, 3.0])
    ssh.run_failures_left = 1
    handle = executor.submit(spec)
    assert executor.wait(handle).succeeded
    assert executor.job_cost(handle) == (5.0, None)
    assert [row["cost_usd"] for row in cost_rows(executor)] == [None, 3.0]


def test_accounting_error_does_not_replace_primary_failure_or_repeat_cleanup(
    submission, monkeypatch
):
    executor, spec, api, _ssh, _events = submission
    primary = OSError("synthetic primary lease error")
    accounted = []

    def fail_lease(*_args):
        raise primary

    def fail_accounting(*args):
        accounted.append(args)
        raise ValueError("synthetic accounting error")

    monkeypatch.setattr(executor, "_write_lease", fail_lease)
    monkeypatch.setattr(executor, "_record_cost", fail_accounting)
    cleanup_clock(monkeypatch, api, [2.0])
    with pytest.raises(OSError) as caught:
        executor.submit(spec)
    assert caught.value is primary
    assert len(accounted) == 1 and accounted[0][-1] is JobStatus.FAILED
    assert api.terminated == ["pod-abc"]


def test_unconfirmed_cleanup_records_only_observed_window_and_warns_billing_may_continue(
    submission, monkeypatch
):
    executor, spec, api, _ssh, events = submission
    api.cost_per_hr = 3600.0

    def fail_cleanup(pod_id):
        api.terminated.append(pod_id)
        raise OSError("synthetic cleanup unavailable")

    monkeypatch.setattr(api, "terminate_pod", fail_cleanup)
    cleanup_clock(monkeypatch, api, [7.0])
    handle = executor.submit(spec)
    assert executor.wait(handle).succeeded
    rows = cost_rows(executor)
    assert len(rows) == 1 and rows[0]["duration_s"] == 7.0
    assert rows[0]["cost_usd"] == 7.0
    assert executor.job_cost(handle) == (7.0, 7.0)
    assert events[-1]["event"] == "cleanup_unconfirmed"
    assert executor.leases()[0]["pod_id"] == "pod-abc"
    assert api.terminated == ["pod-abc"]
    assert any("billing may still be active" in line for line in executor.logs(handle))


@pytest.mark.parametrize("failure", [500, 429, "disconnect"])
def test_ambiguous_allocation_failure_never_repeats_post_or_fabricates_cost(
    submission, monkeypatch, failure
):
    import requests

    executor, spec, _api, ssh, events = submission
    posts, sleeps = [], []
    executor._api = RunPodApi("synthetic-test-key")

    def post(*args, **kwargs):
        posts.append((args, kwargs))
        if failure == "disconnect":
            raise requests.ConnectionError("synthetic ambiguous disconnect")
        response = requests.Response()
        response.status_code = failure
        return response

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr("stateset_agents.remote.runpod.time.sleep", sleeps.append)
    with pytest.raises(RemoteExecutionError, match="allocation may exist") as caught:
        executor.submit(spec)
    assert caught.value.context.details["attempts"] == 1
    assert len(posts) == 1 and sleeps == []
    assert len(events) == 1 and events[0]["event"] == "allocation_intent"
    assert events[0]["resource_id"] is None
    assert ssh.uploaded == [] and executor._jobs == {}
    assert executor._last_duration_s is None and not executor.ledger_path.exists()
