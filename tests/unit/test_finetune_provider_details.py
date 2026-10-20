"""Retained provider identity stays read-only and honest in terminal details."""

import json
import os

import pytest
from rich.cells import cell_len

from stateset_agents import finetune_provider_submission as submission
from stateset_agents import finetune_workspace as workspace
from stateset_agents.remote import registry
from stateset_agents.tui import Studio
from tests.unit.test_finetune_provider_submission import guarded_record_open

JOB_ID = "synthetic-provider-job"
POD_ID = "synthetic-provider-pod"


@pytest.fixture(autouse=True)
def no_ambient_wheel(monkeypatch):
    monkeypatch.delenv("STATESET_AGENTS_WHEEL", raising=False)


def make_provider_run(tmp_path, provider="runpod"):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "PRIVATE synthetic customer question"},
                    {
                        "role": "assistant",
                        "content": "PRIVATE synthetic customer reply",
                    },
                ]
            }
        )
        + "\n"
    )
    config = workspace.FinetuneConfig(
        dataset=str(source),
        provider=provider,
        cost_policy="provider_billed",
    )
    return workspace.create_run(tmp_path / "workspace", workspace.plan_run(config))


def event(name, resource_id=None, *, kind="pod", provider="runpod"):
    value = {
        "schema_version": 1,
        "provider": provider,
        "job_id": JOB_ID,
        "event": name,
        "resource_kind": kind,
        "resource_id": resource_id,
    }
    if kind in {"pod", "volume"}:
        value["resource_name"] = "stateset-sft-" + JOB_ID
    return value


def damage_receipt(run, kind, tmp_path):
    path = run / "provider-submission.json"
    if kind == "malformed":
        path.write_text("PRIVATE malformed receipt")
    elif kind == "plan":
        plan = workspace.read_json(run / "plan.json")
        plan["config"]["model"] = "changed/model"
        workspace.write_json(run / "plan.json", plan)
    elif kind == "symlink":
        outside = tmp_path / "outside-receipt.json"
        outside.write_text("PRIVATE outside record")
        path.unlink()
        try:
            path.symlink_to(outside)
        except OSError:
            pytest.skip("Symlinks unavailable")
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFOs unavailable")
        path.unlink()
        os.mkfifo(path, 0o600)
    elif kind == "oversize":
        path.write_bytes(b"PRIVATE" * 20000)
    elif kind == "public":
        if os.name == "nt":
            pytest.skip("POSIX permissions unavailable")
        path.chmod(0o644)


def test_details_show_retained_ids_and_acknowledgement_without_live_claims(
    tmp_path, monkeypatch
):
    run = make_provider_run(tmp_path)
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    journal.publish(event("allocation_intent"))
    journal.publish(event("allocated", POD_ID))
    journal.publish(event("cleanup_acknowledged", POD_ID))
    journal.finish("succeeded")
    monkeypatch.setattr(
        registry,
        "get_executor",
        lambda *_: pytest.fail("View must not contact providers"),
    )
    before = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
    studio = Studio(run.parent)
    text = "\n".join(studio._run_details())
    assert "PROVIDER SUBMISSION / RETAINED OBSERVATIONS" in text
    assert JOB_ID in text and POD_ID in text and "cleanup acknowledged" in text
    assert "Cleanup required or unconfirmed" in text
    assert "does not prove billing stopped" in text and "No automatic restart" in text
    assert "not checked live" in text and "PRIVATE" not in text
    assert before == {
        path.name: path.read_bytes() for path in run.iterdir() if path.is_file()
    }


def test_details_distinguish_unresolved_intent_from_a_known_allocation(tmp_path):
    run = make_provider_run(tmp_path)
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    journal.publish(event("allocation_intent"))
    text = "\n".join(Studio(run.parent)._run_details())
    assert "Unresolved allocation: pod | stateset-sft-" + JOB_ID in text
    assert POD_ID not in text and "Cleanup required or unconfirmed" in text


def test_modal_details_keep_volume_and_function_call_identities_distinct(tmp_path):
    run = make_provider_run(tmp_path, "modal")
    journal = submission.SubmissionJournal(run, "modal", durable_supported=True)
    journal.publish(event("allocated", "vo-synthetic", kind="volume", provider="modal"))
    journal.publish(
        event("allocated", "fc-synthetic", kind="function_call", provider="modal")
    )
    journal.publish(
        event(
            "execution_finished", "fc-synthetic", kind="function_call", provider="modal"
        )
    )
    text = "\n".join(Studio(run.parent)._run_details())
    assert "volume: vo-synthetic | allocated" in text
    assert "function_call: fc-synthetic | execution finished" in text
    assert "Cleanup required or unconfirmed" in text


def test_details_missing_receipt_is_unknown_and_never_creates_one(tmp_path):
    run = make_provider_run(tmp_path)
    text = "\n".join(Studio(run.parent)._run_details())
    assert "No submission receipt retained" in text
    assert "identity and cleanup are unknown" in text
    assert not (run / "provider-submission.json").exists()


@pytest.mark.parametrize(
    "kind", ["malformed", "plan", "symlink", "fifo", "oversize", "public"]
)
def test_invalid_receipt_uses_controlled_terminal_message(tmp_path, monkeypatch, kind):
    run = make_provider_run(tmp_path)
    submission.SubmissionJournal(run, "runpod", durable_supported=True)
    damage_receipt(run, kind, tmp_path)
    guarded_record_open(monkeypatch, run / "provider-submission.json")
    text = "\n".join(Studio(run.parent)._run_details())
    assert "Provider receipt unavailable" in text and "could not be verified" in text
    assert "PRIVATE" not in text
    assert not (run / "admission.json").exists()


def test_known_credentials_are_redacted_from_retained_terminal_identity(
    tmp_path, monkeypatch
):
    run = make_provider_run(tmp_path)
    secret = "synthetic-secret-credential"
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    journal.publish(event("allocated", secret))
    monkeypatch.setenv("RUNPOD_API_KEY", secret)
    text = "\n".join(Studio(run.parent)._run_details())
    assert secret not in text and "[REDACTED]" in text


@pytest.mark.parametrize("width,height", [(32, 16), (72, 22), (74, 44), (110, 32)])
def test_provider_receipt_details_remain_clipped_and_scrollable(
    tmp_path, width, height
):
    run = make_provider_run(tmp_path)
    journal = submission.SubmissionJournal(run, "runpod", durable_supported=True)
    journal.publish(event("allocation_intent"))
    journal.publish(event("allocated", POD_ID))
    studio = Studio(run.parent)
    studio.handle("TAB")
    studio.handle("ENTER")
    assert studio.overlay == "details"
    for _ in range(20):
        studio.handle("PAGEDOWN")
    lines = studio.lines(width, height)
    assert len(lines) <= height and all(cell_len(line) <= width for line in lines)
    assert not (run / "admission.json").exists()
