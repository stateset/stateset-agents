"""Reviewed package, real worker, journal and CLI integration without providers."""

import builtins
import io
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as cli
from stateset_agents import finetune_provider_submission as submission
from stateset_agents import finetune_remote_code as remote_code
from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace
from stateset_agents.remote import registry
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobResult
from stateset_agents.remote.runpod import RunPodExecutor

JOB_ID = "integration-job"
POD_ID = "synthetic-pod"


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    monkeypatch.delenv(remote_code.WHEEL_ENV, raising=False)
    monkeypatch.setattr(worker.os, "umask", lambda _mask: 0)
    dataset = tmp_path / "chat.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Synthetic customer question"},
                    {"role": "assistant", "content": "Synthetic support response"},
                ]
            }
        )
        + "\n"
    )

    def prepare(wheel=None):
        if wheel is not None:
            monkeypatch.setenv(remote_code.WHEEL_ENV, str(wheel))
        config = workspace.FinetuneConfig(
            provider="runpod",
            dataset=str(dataset),
            seed=0,
            cost_policy="provider_billed",
        )
        plan = workspace.plan_run(config)
        run = workspace.create_run(tmp_path / "workspace", plan)
        output = run / "artifacts"
        output.mkdir()
        (output / "adapter_model.safetensors").write_bytes(b"inert fixture artifact")
        return run, config

    return prepare


class NoProvider:
    def __getattr__(self, name):
        raise AssertionError(f"No provider/SSH operation is allowed: {name}")


class ObservedRunPod(RunPodExecutor):
    """Use the real binder/observer, replacing only external submission work."""

    def __init__(self, run, *, fault=None, status=JobStatus.SUCCEEDED):
        super().__init__(api=NoProvider(), ssh=NoProvider(), public_key="synthetic-key")
        self.run, self.fault, self.result_status = run, fault, status
        self.calls, self.early = [], None
        self.submitted_wheel = None

    def submit(self, spec):
        self.calls.append("submit")
        self._submission_started = True
        self.submitted_wheel = self.wheel
        assert spec.dataset == self.run / "dataset" and spec.seed == 0
        assert (
            spec.package_version
            == workspace.read_run_plan(self.run)[0]["framework_version"]
        )
        base = {
            "job_id": JOB_ID,
            "resource_kind": "pod",
            "resource_name": "stateset-sft-" + JOB_ID,
            "attempt": 1,
        }
        self._observe_submission({**base, "event": "allocation_intent"})
        intent = submission.inspect_submission(self.run)
        assert intent[
            "unresolved_intents"
        ], "Intent must be visible before a billable hook"
        self.calls.append("inert-allocation-boundary")
        self._observe_submission({**base, "event": "allocated", "resource_id": POD_ID})
        self.early = submission.inspect_submission(self.run)
        assert self.early["resources"][0]["id"] == POD_ID
        assert workspace.read_json(self.run / "provider-job.json")["job_id"] == JOB_ID
        assert self.early["submission_state"] == "submitting"
        if self.fault == "submit":
            raise RuntimeError("injected synchronous submit failure")
        self._observe_submission(
            {**base, "event": "cleanup_acknowledged", "resource_id": POD_ID}
        )
        return JobHandle("runpod", JOB_ID)

    def wait(self, handle):
        self.calls.append("wait")
        return RemoteJobResult(
            handle,
            self.result_status,
            self.run / "artifacts",
            ["synthetic log"],
            duration_s=8.0,
            cost_usd=2.75,
        )


def install_executor(monkeypatch, executor):
    def acquire(provider):
        assert provider == "runpod"
        return executor

    monkeypatch.setattr(registry, "get_executor", acquire)


def invoke(*args):
    result = CliRunner().invoke(cli.app, list(map(str, args)))
    response = json.loads(result.stdout)
    assert response["schema_version"] == 1 and response["ok"] == (result.exit_code == 0)
    return result, response


def test_prepared_wheel_survives_source_removal_and_environment_drift(
    make_run, tmp_path, monkeypatch
):
    source = tmp_path / "stateset_agents-0.66.1-py3-none-any.whl"
    source.write_bytes(b"reviewed wheel archive")
    run, _ = make_run(source)
    record = workspace.read_run_plan(run)[0]["remote_code"]
    snapshot = remote_code.verify_remote_code(run, record)
    source.unlink()
    later = tmp_path / "later-unreviewed.whl"
    monkeypatch.setenv(remote_code.WHEEL_ENV, str(later))
    review = workspace.review_prepared_run(run)
    assert review.plan.describe()["remote_code"] == record
    executor = ObservedRunPod(run)
    assert executor.wheel == later
    install_executor(monkeypatch, executor)
    assert worker.execute(run) == 0
    assert executor.submitted_wheel == snapshot
    assert executor.reviewed_wheel_sha256 == record["wheel_sha256"]
    assert executor.reviewed_wheel_bytes == record["wheel_bytes"]
    assert workspace.read_json(run / "state.json")["status"] == "succeeded"
    assert not workspace.read_json(run / "completion.json")["quality_gain_verified"]


@pytest.mark.parametrize("entrypoint", ["settings", "review", "launch", "worker"])
def test_changed_snapshot_stops_before_executor_or_supervisor_access(
    make_run, tmp_path, monkeypatch, entrypoint
):
    source = tmp_path / "stateset_agents-0.66.1-py3-none-any.whl"
    source.write_bytes(b"reviewed wheel archive")
    run, _ = make_run(source)
    plan, digest = workspace.read_run_plan(run)
    snapshot = remote_code.verify_remote_code(run, plan["remote_code"])
    snapshot.chmod(0o600)
    snapshot.write_bytes(b"X" * plan["remote_code"]["wheel_bytes"])
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("unreviewed work")
        raise AssertionError("Changed wheel must fail before executor or Popen")

    monkeypatch.setattr(registry, "get_executor", forbidden)
    monkeypatch.setattr(workspace.subprocess, "Popen", forbidden)
    if entrypoint == "worker":
        assert worker.execute(run) == 1
    else:
        with pytest.raises(ValueError):
            if entrypoint == "settings":
                workspace.verified_run_settings(run)
            elif entrypoint == "review":
                workspace.review_prepared_run(run)
            else:
                workspace.launch_run(run, expected_plan_sha256=digest)
    assert calls == [] and not (run / "provider-submission.json").exists()


def test_release_review_clears_later_ambient_wheel_in_actual_executor_binder(
    make_run, tmp_path, monkeypatch
):
    run, _ = make_run()
    assert workspace.read_run_plan(run)[0]["remote_code"]["mode"] == "release"
    later = tmp_path / "later-unreviewed.whl"
    monkeypatch.setenv(remote_code.WHEEL_ENV, str(later))
    executor = ObservedRunPod(run)
    assert executor.wheel == later
    install_executor(monkeypatch, executor)
    assert worker.execute(run) == 0
    assert executor.submitted_wheel is None and executor.wheel is None
    assert (
        executor.reviewed_wheel_sha256 is None and executor.reviewed_wheel_bytes is None
    )
    assert executor.early["job_id"] == JOB_ID
    assert executor.calls == ["submit", "inert-allocation-boundary", "wait"]


@pytest.mark.parametrize("fault", ["submit", "callback"])
def test_submission_failure_retains_early_identity_and_execution_uncertainty(
    make_run, monkeypatch, fault
):
    run, _ = make_run()
    executor = ObservedRunPod(run, fault=fault)
    install_executor(monkeypatch, executor)
    if fault == "callback":
        publish = submission.SubmissionJournal.publish

        def failed(self, value):
            if value["event"] == "allocated":
                raise OSError("injected callback persistence failure")
            return publish(self, value)

        monkeypatch.setattr(submission.SubmissionJournal, "publish", failed)
    assert worker.execute(run) == 1
    report = submission.inspect_submission(run)
    assert workspace.read_json(run / "state.json")["status"] == "failed"
    assert report["job_id"] == JOB_ID and report["submission_state"] == "submitting"
    assert report["cleanup_required_or_unconfirmed"] is True
    assert report["provider_cleanup_verified"] is False
    if fault == "submit":
        assert report["resources"]
    else:
        assert report["unresolved_intents"]
    assert executor.calls == ["submit", "inert-allocation-boundary"]
    assert not (run / "completion.json").exists()


def test_returned_failure_keeps_cost_duration_and_provider_diagnosis(
    make_run, monkeypatch
):
    run, _ = make_run()
    executor = ObservedRunPod(run, status=JobStatus.FAILED)
    install_executor(monkeypatch, executor)
    assert worker.execute(run) == 1
    assert workspace.read_json(run / "provider-result.json") == {
        "duration_s": 8.0,
        "cost_usd": 2.75,
    }
    assert workspace.read_json(run / "failure.json")["code"] == "provider_failed"
    assert submission.inspect_submission(run)["observed_status"] == "failed"
    _, response = invoke("status", run.name, "--workspace", run.parent)
    assert response["result"]["run"]["provider_result"]["cost_usd"] == 2.75


def test_cli_inspects_persisted_resources_without_provider_or_wheel_reads(
    make_run, monkeypatch
):
    run, _ = make_run()
    executor = ObservedRunPod(run)
    install_executor(monkeypatch, executor)
    assert worker.execute(run) == 0

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Passive inspection must not contact providers or hash wheels"
        )

    monkeypatch.setattr(registry, "get_executor", forbidden)
    monkeypatch.setattr(remote_code, "_wheel_identity", forbidden)
    _, response = invoke("provider-status", run.name, "--workspace", run.parent)
    report = response["result"]
    assert report["resources"][0]["id"] == POD_ID
    assert not report["live_status_checked"] and not report["provider_cleanup_verified"]
    assert report["cleanup_required_or_unconfirmed"]
    _, direct = invoke("status", run.name, "--workspace", run.parent)
    _, history = invoke("status", "--workspace", run.parent)
    for summary in (direct["result"]["run"], history["result"]["runs"][0]):
        assert summary["provider_submission"]["job_id"] == JOB_ID
        assert (
            summary["provider_submission"]["resources"][0]["state"]
            == "cleanup_acknowledged"
        )


@pytest.mark.parametrize("kind", ["malformed", "symlink", "fifo"])
def test_cli_bad_provider_receipts_are_controlled_and_do_not_hang(
    make_run, tmp_path, monkeypatch, kind
):
    run, _ = make_run()
    submission.SubmissionJournal(run, "runpod", durable_supported=True)
    path = run / "provider-submission.json"
    path.unlink()
    if kind == "malformed":
        path.write_text('{"private":"PRIVATE-UNRELATED-VALUE"}')
        path.chmod(0o600)
    elif kind == "symlink":
        outside = tmp_path / "private-outside.json"
        outside.write_text('{"private":"PRIVATE-UNRELATED-VALUE"}')
        path.symlink_to(outside)
    else:
        if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
            pytest.skip("Nonblocking named pipes are unavailable")
        os.mkfifo(path, 0o600)
    originals = (os.open, Path.open, builtins.open, io.open)

    def matches(candidate):
        return not isinstance(candidate, int) and os.fspath(candidate) == os.fspath(
            path
        )

    def opened(candidate, flags, *args, **kwargs):
        if matches(candidate):
            if hasattr(os, "O_NONBLOCK"):
                assert flags & os.O_NONBLOCK
            if hasattr(os, "O_NOFOLLOW"):
                assert flags & os.O_NOFOLLOW
        return originals[0](candidate, flags, *args, **kwargs)

    def guarded(original):
        def checked(candidate, *args, **kwargs):
            assert not matches(candidate), "Provider receipts need descriptor admission"
            return original(candidate, *args, **kwargs)

        return checked

    monkeypatch.setattr(submission.os, "open", opened)
    monkeypatch.setattr(Path, "open", guarded(originals[1]))
    monkeypatch.setattr(builtins, "open", guarded(originals[2]))
    monkeypatch.setattr(io, "open", guarded(originals[3]))
    result, response = invoke("provider-status", run.name, "--workspace", run.parent)
    assert (
        not response["ok"] and response["error"]["code"] == "invalid_provider_receipt"
    )
    assert "PRIVATE-UNRELATED-VALUE" not in result.stdout
    result, response = invoke("status", run.name, "--workspace", run.parent)
    assert response["ok"]
    assert "provider-submission.json" in response["result"]["run"]["unreadable_records"]
    assert "provider_submission" not in response["result"]["run"]
    assert "PRIVATE-UNRELATED-VALUE" not in result.stdout
