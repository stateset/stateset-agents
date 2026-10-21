"""Durable, plan-bound provider observations without SDKs or paid resources."""

import builtins
import io
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from stateset_agents import finetune_provider_submission as submission
from stateset_agents import finetune_workspace as workspace

JOB_ID = "synthetic-job-1"
RESOURCE_ID = "synthetic-pod-1"


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.delenv("STATESET_AGENTS_WHEEL", raising=False)
    source = tmp_path / "chat.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Synthetic customer question"},
                    {"role": "assistant", "content": "Synthetic customer response"},
                ]
            }
        )
        + "\n"
    )
    config = workspace.FinetuneConfig(
        provider="runpod",
        dataset=str(source),
        seed=0,
        cost_policy="provider_billed",
    )
    return workspace.create_run(tmp_path / "workspace", workspace.plan_run(config))


def event(name, resource_id=None):
    value = {
        "schema_version": 1,
        "provider": "runpod",
        "job_id": JOB_ID,
        "event": name,
        "resource_kind": "pod",
        "resource_name": "stateset-sft-" + JOB_ID,
        "attempt": 1,
    }
    if name != "allocation_intent":
        value["resource_id"] = resource_id or RESOURCE_ID
    return value


def journal_path(run):
    return run / "provider-submission.json"


def guarded_record_open(monkeypatch, path):
    """A blocking-open regression fails before attempting a real FIFO open."""
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
            assert not matches(candidate), "Journal reads need descriptor admission"
            return original(candidate, *args, **kwargs)

        return checked

    monkeypatch.setattr(submission.os, "open", opened)
    monkeypatch.setattr(Path, "open", guarded(originals[1]))
    monkeypatch.setattr(builtins, "open", guarded(originals[2]))
    monkeypatch.setattr(io, "open", guarded(originals[3]))


def test_preallocation_intent_is_retained_before_mock_billable_action(run):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    observed = []

    def mock_billable_action():
        record = workspace.read_json(journal_path(run))
        assert record["plan_sha256"] == workspace.read_run_plan(run)[1]
        assert record["events"][-1]["event"] == "allocation_intent"
        assert workspace.read_json(run / "provider-job.json")["job_id"] == JOB_ID
        observed.append("billable action would start here")

    journal.publish(event("allocation_intent"))
    mock_billable_action()
    assert len(observed) == 1
    assert submission.inspect_submission(run)["live_status_checked"] is False
    if os.name != "nt":
        assert journal_path(run).stat().st_mode & 0o077 == 0


def test_allocated_identity_is_inspectable_while_submission_has_not_returned(run):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    allocated, release = threading.Event(), threading.Event()

    def mock_synchronous_submit():
        journal.publish(event("allocation_intent"))
        journal.publish(event("allocated"))
        allocated.set()
        assert release.wait(5), "mock submission was not released"
        return RESOURCE_ID

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(mock_synchronous_submit)
        try:
            assert allocated.wait(5), "allocation callback did not complete"
            report = submission.inspect_submission(run)
            assert report["job_id"] == JOB_ID
            assert report["resources"][0]["id"] == RESOURCE_ID
            assert report["submission_state"] == "submitting"
            assert not future.done()
        finally:
            release.set()
        assert future.result(timeout=5) == RESOURCE_ID


def test_fresh_process_inspects_receipt_after_original_journal_object_is_gone(
    run, tmp_path
):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    journal.publish(event("allocation_intent"))
    journal.publish(event("allocated"))
    del journal
    before = journal_path(run).read_bytes()
    child_cwd = tmp_path / "empty-child-cwd"
    child_cwd.mkdir()
    program = (
        "import json,sys; from pathlib import Path; "
        "from stateset_agents.finetune_provider_submission import inspect_submission; "
        "print(json.dumps(inspect_submission(Path(sys.argv[1])), allow_nan=False))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(run)],
        cwd=child_cwd,
        capture_output=True,
        text=True,
        timeout=20,
        env={
            "PATH": os.defpath,
            "PYTHONPATH": str(Path(submission.__file__).parents[1]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
        },
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["resources"][0]["id"] == RESOURCE_ID
    assert report["cleanup_required_or_unconfirmed"] is True
    assert (
        not report["live_status_checked"] and not report["automatic_restart_supported"]
    )
    assert journal_path(run).read_bytes() == before


@pytest.mark.parametrize(
    "cleanup", ["cleanup_acknowledged", "cleanup_unconfirmed", "cleanup_confirmed"]
)
def test_cleanup_acknowledgement_does_not_become_live_verified_cleanup(run, cleanup):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    for name in ("allocation_intent", "allocated", cleanup):
        journal.publish(event(name))
    journal.finish("succeeded")
    report = submission.inspect_submission(run)
    assert report["resources"][0]["state"] == cleanup
    assert report["cleanup_required_or_unconfirmed"] is (cleanup != "cleanup_confirmed")
    assert report["provider_cleanup_verified"] is False
    assert report["live_status_checked"] is False


def test_intent_only_failure_remains_ambiguous_instead_of_claiming_no_resources(run):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    journal.publish(event("allocation_intent"))
    journal.finish("failed")
    report = submission.inspect_submission(run)
    assert report["resources"] == [] and report["unresolved_intents"]
    assert report["observed_status"] == "failed"
    assert report["cleanup_required_or_unconfirmed"] is True
    assert report["provider_cleanup_verified"] is False


def test_nondurable_handle_and_success_never_prove_resource_cleanup(run):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=False)
    journal.record_handle("runpod", JOB_ID)
    journal.finish("succeeded")
    report = submission.inspect_submission(run)
    assert report["durable_receipts"] is False and report["resources"] == []
    assert report["cleanup_required_or_unconfirmed"] is True
    assert report["provider_cleanup_verified"] is False


def test_plan_mutation_rejects_callback_before_mock_allocation(run):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    before = journal_path(run).read_bytes()
    plan = workspace.read_json(run / "plan.json")
    plan["config"]["epochs"] += 1
    workspace.write_json(run / "plan.json", plan)
    allocations = []

    def submit():
        journal.publish(event("allocation_intent"))
        allocations.append("must not allocate")

    with pytest.raises(ValueError, match="plan changed"):
        submit()
    assert allocations == [] and journal_path(run).read_bytes() == before
    assert not (run / "provider-job.json").exists()


def test_inspection_rejects_receipt_after_reviewed_plan_changes(run):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    journal.record_handle("runpod", JOB_ID)
    plan = workspace.read_json(run / "plan.json")
    plan["config"]["epochs"] += 1
    workspace.write_json(run / "plan.json", plan)
    with pytest.raises(ValueError, match="reviewed plan"):
        submission.inspect_submission(run)


def test_exclusive_journal_refuses_resubmission_after_owner_object_disappears(run):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    del journal
    before = journal_path(run).read_bytes()
    with pytest.raises(FileExistsError):
        submission.SubmissionJournal(run, "runpod", durable_supported=True)
    assert journal_path(run).read_bytes() == before
    assert submission.inspect_submission(run)["automatic_restart_supported"] is False


@pytest.mark.parametrize(
    "kind", ["invalid_json", "oversized", "symlink", "fifo", "directory"]
)
def test_invalid_or_nonregular_journal_is_bounded_and_cannot_hang(
    run, tmp_path, monkeypatch, kind
):
    submission.SubmissionJournal(run, "runpod", durable_supported=True)
    path = journal_path(run)
    path.unlink()
    if kind == "invalid_json":
        path.write_text('{"partial":')
        path.chmod(0o600)
    elif kind == "oversized":
        path.write_bytes(b" " * (128 * 1024 + 1))
        path.chmod(0o600)
    elif kind == "symlink":
        outside = tmp_path / "private-outside.json"
        outside.write_text('{"secret":"PRIVATE-UNRELATED-VALUE"}')
        path.symlink_to(outside)
    elif kind == "fifo":
        if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
            pytest.skip("Nonblocking named pipes are unavailable")
        os.mkfifo(path, 0o600)
    else:
        path.mkdir()
    guarded_record_open(monkeypatch, path)
    with pytest.raises((OSError, ValueError)) as error:
        submission.inspect_submission(run)
    assert "PRIVATE-UNRELATED-VALUE" not in str(error.value)


@pytest.mark.parametrize(
    "field,value", [("created_at", float("nan")), ("job_id", ["not-an-identifier"])]
)
def test_malformed_retained_observations_fail_closed(run, field, value):
    submission.SubmissionJournal(run, "runpod", durable_supported=True)
    record = workspace.read_json(journal_path(run))
    record[field] = value
    journal_path(run).write_text(json.dumps(record))
    with pytest.raises(ValueError):
        submission.inspect_submission(run)


def test_unknown_event_fields_and_unrelated_secret_never_reach_disk(run):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    before = journal_path(run).read_bytes()
    value = {**event("allocation_intent"), "api_key": "PRIVATE-UNRELATED-VALUE"}
    with pytest.raises(ValueError) as error:
        journal.publish(value)
    assert journal_path(run).read_bytes() == before
    assert "PRIVATE-UNRELATED-VALUE" not in str(error.value)
    assert "PRIVATE-UNRELATED-VALUE" not in json.dumps(
        submission.inspect_submission(run)
    )


@pytest.mark.parametrize("field", ["event", "resource_kind"])
def test_unhashable_event_fields_return_controlled_validation_error(run, field):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    before = journal_path(run).read_bytes()
    value = event("allocation_intent")
    value[field] = []
    with pytest.raises(ValueError):
        journal.publish(value)
    assert journal_path(run).read_bytes() == before


def test_event_limit_preserves_the_last_complete_receipt(run, monkeypatch):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    monkeypatch.setattr(submission, "_EVENT_LIMIT", 2)
    journal.publish(event("allocation_intent"))
    journal.publish(event("allocated"))
    before = journal_path(run).read_bytes()
    with pytest.raises(ValueError, match="event limit"):
        journal.publish(event("cleanup_acknowledged"))
    assert journal_path(run).read_bytes() == before


def test_missing_journal_is_read_only_and_does_not_invent_allocation_state(run):
    before = set(run.iterdir())
    assert submission.inspect_submission(run) == {
        "available": False,
        "live_status_checked": False,
    }
    assert set(run.iterdir()) == before


def test_public_journal_permissions_are_rejected(run):
    if os.name == "nt":
        pytest.skip("POSIX private-mode admission")
    submission.SubmissionJournal(run, "runpod", durable_supported=True)
    journal_path(run).chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        submission.inspect_submission(run)


def test_nonstring_handle_is_rejected_without_changing_the_journal(run):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    before = journal_path(run).read_bytes()
    with pytest.raises(ValueError):
        journal.record_handle("runpod", ["not-an-identifier"])
    assert journal_path(run).read_bytes() == before
    assert not (run / "provider-job.json").exists()


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
def test_terminal_return_without_lifecycle_receipts_keeps_cleanup_uncertain(
    run, status
):
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    journal.record_handle("runpod", JOB_ID)
    journal.finish(status)
    report = submission.inspect_submission(run)
    assert report["submission_state"] == "returned"
    assert report["resources"] == []
    assert report["cleanup_required_or_unconfirmed"] is True
    assert report["provider_cleanup_verified"] is False
