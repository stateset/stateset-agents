"""Every state.json change goes through one locked, legality-checked transition."""

from __future__ import annotations

import json

import pytest

from stateset_agents import finetune_evaluation_worker as evaluation_worker
from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace


@pytest.fixture
def run(tmp_path, monkeypatch):
    source = tmp_path / "chat.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        )
        + "\n"
    )
    monkeypatch.setattr(worker.os, "umask", lambda _mask: 0)
    config = workspace.FinetuneConfig(dataset=str(source))
    return workspace.create_run(tmp_path / "workspace", workspace.plan_run(config))


def _commit_success(run):
    output = run / "artifacts"
    output.mkdir()
    (output / "adapter_model.safetensors").write_bytes(b"weights")
    workspace.write_json(
        run / "completion.json",
        {"output_dir": str(output), "weights": ["adapter_model.safetensors"]},
    )
    # Written directly, as a racing worker would, independent of the helper.
    workspace.write_json(run / "state.json", {"status": "succeeded"})


@pytest.mark.parametrize(
    "final,attempt",
    [
        ("failed", "succeeded"),
        ("failed", "running"),
        ("cancelled", "failed"),
        ("cancelled", "queued"),
        ("succeeded", "running"),
        ("succeeded", "cancelled"),
    ],
)
def test_final_outcomes_never_move_to_another_status(run, final, attempt):
    if final == "succeeded":
        _commit_success(run)
    else:
        assert workspace.transition_run(run, status=final)
    before = (run / "state.json").read_bytes()
    assert workspace.transition_run(run, status=attempt) is False
    assert (run / "state.json").read_bytes() == before
    # Detail may still be added to a final outcome without changing it.
    assert workspace.transition_run(run, status=final, exit_code=3) is True
    assert workspace.read_json(run / "state.json")["status"] == final


def test_unverified_success_may_still_be_replaced_by_a_failure(run):
    assert workspace.transition_run(run, status="succeeded")  # no artifact
    assert workspace.transition_run(run, status="failed") is True


def test_allowed_from_is_checked_under_the_lock(run):
    assert (
        workspace.transition_run(run, allowed_from={"running"}, status="failed")
        is False
    )
    assert workspace.read_json(run / "state.json")["status"] == "prepared"
    assert workspace.transition_run(run, allowed_from={"prepared"}, status="queued")


def test_refused_failure_leaves_no_diagnosis_behind(run):
    _commit_success(run)
    worker._record_failure(
        run, workspace.FinetuneFailure("worker_exit"), stage="supervisor"
    )
    assert not (run / "failure.json").exists()
    assert workspace.read_json(run / "state.json")["teardown_error"] == "worker_exit"


def test_stop_racing_a_finished_run_is_a_no_op_not_an_overwrite(run, monkeypatch):
    """The run succeeds between request_stop's status read and its write."""
    _, digest = workspace.read_run_plan(run)
    workspace.write_json(
        run / "launch.lock", {"schema_version": 1, "plan_sha256": digest}
    )
    unclaimed, status = workspace.launch_unclaimed, workspace.run_status
    armed = []

    def observed_status(directory):
        state = status(directory)
        armed.append(state["status"])
        return state

    def finish_meanwhile(directory):
        if armed and not (run / "completion.json").exists():
            _commit_success(run)
        return unclaimed(directory)

    monkeypatch.setattr(workspace, "run_status", observed_status)
    monkeypatch.setattr(workspace, "launch_unclaimed", finish_meanwhile)
    workspace.request_stop(run, expected_plan_sha256=digest)
    assert armed[0] == "prepared"  # the stop was reviewed against a live run
    assert workspace.read_json(run / "state.json")["status"] == "succeeded"


def test_launch_failure_cannot_overwrite_an_outcome_recorded_meanwhile(
    run, monkeypatch
):
    def fail_after_cancel(*_args, **_kwargs):
        workspace.write_json(
            run / "state.json", {"status": "cancelled", "error": "stopped elsewhere"}
        )
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(workspace.subprocess, "Popen", fail_after_cancel)
    with pytest.raises(OSError):
        workspace.launch_run(run)
    state = workspace.read_json(run / "state.json")
    assert state["status"] == "cancelled" and state["error"] == "stopped elsewhere"


def test_evaluation_supervisor_starts_no_queue_work_after_an_outcome(
    tmp_path, monkeypatch
):
    from tests.unit.test_finetune_evaluation import prepared

    path, review = prepared(tmp_path)
    workspace.write_json(path / "launch.lock", {"plan_sha256": review.sha256})
    workspace.write_json(path / "state.json", {"status": "cancelled"})
    before = (path / "state.json").read_bytes()
    monkeypatch.setattr(evaluation_worker, "queue_executable", lambda: "/queue")
    monkeypatch.setattr(
        evaluation_worker.subprocess,
        "call",
        lambda *_a, **_k: pytest.fail("No queue work after a recorded outcome"),
    )
    assert evaluation_worker.supervise(path) == 1
    assert (path / "state.json").read_bytes() == before
