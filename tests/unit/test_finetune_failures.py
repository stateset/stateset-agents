"""Failure diagnosis, terminal lifecycle and recovery guidance without paid compute."""

import errno
import json
import os
import subprocess
from types import SimpleNamespace

import pytest
from rich.cells import cell_len

from stateset_agents import finetune_worker as worker
from stateset_agents.finetune_workspace import (
    FinetuneConfig,
    FinetuneFailure,
    create_run,
    failure_info,
    plan_run,
    read_json,
    write_json,
)
from stateset_agents.tui import Studio


@pytest.fixture
def run(tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "How can I help?"},
                ]
            }
        )
        + "\n"
    )
    monkeypatch.setattr(worker.os, "umask", lambda _: 0)
    monkeypatch.setattr(worker, "queue_executable", lambda: None)
    return create_run(
        tmp_path / "workspace", plan_run(FinetuneConfig(dataset=str(data)))
    )


@pytest.mark.parametrize(
    "error,stage,code",
    [
        (FinetuneFailure("cuda_unavailable"), "runtime", "cuda_unavailable"),
        (MemoryError("sensitive details"), "training", "out_of_memory"),
        (OSError(errno.ENOSPC, "sensitive details"), "artifacts", "disk_full"),
        (OSError(errno.EDQUOT, "sensitive details"), "training", "disk_full"),
        (PermissionError("sensitive details"), "supervisor", "permissions"),
        (subprocess.TimeoutExpired("private command", 10), "supervisor", "timeout"),
        (TimeoutError("private endpoint"), "training", "timeout"),
        (ModuleNotFoundError("private module"), "training", "dependencies"),
        (ValueError("private review"), "review", "review_invalid"),
        (RuntimeError("CUDA out of memory (untrusted text)"), "training", "unknown"),
    ],
)
def test_failure_diagnosis_is_controlled_and_terminal(run, error, stage, code):
    worker._record_failure(run, error, stage=stage)
    state = read_json(run / "state.json")
    info = failure_info(run, state)
    assert state["status"] == "failed" and state["finished_at"] > 0
    assert info["recorded"] and info["code"] == code and info["stage"] == stage
    record = read_json(run / "failure.json")
    assert set(record) == {"schema_version", "code", "stage", "recorded_at"}
    if os.name != "nt":  # Windows st_mode carries no POSIX group/other bits.
        assert (run / "failure.json").stat().st_mode & 0o077 == 0
    assert "sensitive" not in json.dumps(record) + state["error"]
    assert str(error) not in json.dumps(record)


@pytest.mark.parametrize("module", ["torch", "torch.cuda"])
def test_torch_oom_can_be_classified_without_importing_torch(run, module):
    error_type = type("OutOfMemoryError", (RuntimeError,), {"__module__": module})
    worker._record_failure(run, error_type("private detail"), stage="training")
    assert read_json(run / "failure.json")["code"] == "out_of_memory"


@pytest.mark.parametrize("entrypoint", [worker.execute, worker.supervise])
def test_review_failure_is_terminal_even_before_worker_starts(
    run, monkeypatch, entrypoint
):
    (run / "dataset").write_text("changed after review")
    monkeypatch.setattr(
        worker.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("Must not start a process"),
    )
    assert entrypoint(run) == 1
    assert read_json(run / "state.json")["status"] == "failed"
    assert read_json(run / "failure.json")["code"] == "review_invalid"
    assert not (run / "completion.json").exists()


def test_supervisor_preserves_child_diagnosis_and_exit_code(run, monkeypatch):
    retained = []

    def start(*_a, **_k):
        worker._record_failure(
            run, FinetuneFailure("cuda_unavailable"), stage="runtime"
        )
        retained.append((run / "failure.json").read_bytes())
        return SimpleNamespace(wait=lambda **_: 3)

    monkeypatch.setattr(worker.subprocess, "Popen", start)
    assert worker.supervise(run) == 3
    assert (run / "failure.json").read_bytes() == retained[0]
    state = read_json(run / "state.json")
    assert state["status"] == "failed" and state["exit_code"] == 3
    assert "CUDA training is unavailable" in state["error"]


def test_supervisor_initial_queue_configuration_failure_is_recorded(run, monkeypatch):
    def invalid_queue():
        raise ValueError("private path or credential")

    monkeypatch.setattr(worker, "queue_executable", invalid_queue)
    assert worker.supervise(run) == 1
    assert read_json(run / "state.json")["status"] == "failed"
    assert failure_info(run, read_json(run / "state.json"))["stage"] == "supervisor"
    assert "private" not in (run / "failure.json").read_text()


def test_direct_worker_timeout_records_action_after_terminating_its_group(
    run, monkeypatch
):
    signals = []
    waits = []

    def wait(**kwargs):
        waits.append(kwargs)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("private command", 1)
        return -15

    monkeypatch.setattr(
        worker.subprocess,
        "Popen",
        lambda *_a, **_k: SimpleNamespace(pid=123, wait=wait),
    )
    if os.name == "nt":  # Windows stops the worker tree with taskkill instead.
        monkeypatch.setattr(
            worker.subprocess,
            "run",
            lambda command, **_k: signals.append((int(command[-1]), command)),
        )
    else:
        monkeypatch.setattr(
            "os.killpg", lambda pid, signal: signals.append((pid, signal))
        )
    assert worker.supervise(run) == 1
    assert signals and signals[0][0] == 123
    info = failure_info(run, read_json(run / "state.json"))
    assert info["code"] == "timeout" and "spending budget" in info["action"]


@pytest.mark.parametrize("exit_code,code", [(0, "missing_receipt"), (2, "worker_exit")])
def test_unknown_child_exit_does_not_claim_success(run, monkeypatch, exit_code, code):
    monkeypatch.setattr(
        worker.subprocess,
        "Popen",
        lambda *_a, **_k: SimpleNamespace(wait=lambda **_: exit_code),
    )
    assert worker.supervise(run) != 0
    assert failure_info(run, read_json(run / "state.json"))["code"] == code


def test_corrupt_state_does_not_mask_original_failure(run):
    (run / "state.json").write_text("damaged")
    worker._record_failure(run, FinetuneFailure("missing_weights"), stage="artifacts")
    state = read_json(run / "state.json")
    assert state["status"] == "failed" and state["previous_state_unreadable"]
    assert failure_info(run, state)["code"] == "missing_weights"


def test_full_storage_does_not_turn_diagnosis_failure_into_success(
    run, monkeypatch, caplog
):
    def full(*_):
        raise OSError(errno.ENOSPC, "sensitive details")

    monkeypatch.setattr(worker, "write_json", full)
    assert worker.execute(run) == 1
    assert "Could not persist" in caplog.text
    assert not (run / "completion.json").exists()


@pytest.mark.parametrize(
    "record",
    [
        [],
        {"schema_version": True},
        {"schema_version": 2},
        {"schema_version": 1, "code": [], "stage": "runtime"},
        {"schema_version": 1, "code": "invented", "stage": "runtime"},
        {"schema_version": 1, "code": "timeout", "stage": []},
    ],
)
def test_untrusted_or_old_diagnosis_falls_back_without_guessing(run, record):
    write_json(run / "failure.json", record)
    info = failure_info(run, {"status": "failed"})
    assert info["code"] == "unknown" and not info["recorded"]


def test_diagnosis_size_corruption_and_nonterminal_records_are_safe(run):
    (run / "failure.json").write_text("x" * (16 * 1024 + 1))
    assert not failure_info(run, {"status": "failed"})["recorded"]
    (run / "failure.json").write_text("invalid json")
    assert failure_info(run, {"status": "interrupted"})["code"] == "interrupted"
    worker._record_failure(run, FinetuneFailure("timeout"), stage="training")
    assert failure_info(run, {"status": "succeeded"}) is None
    assert failure_info(run, {"status": "running"}) is None


def test_ui_displays_whitelisted_guidance_not_recorded_arbitrary_text(run):
    worker._record_failure(run, FinetuneFailure("out_of_memory"), stage="training")
    record = read_json(run / "failure.json")
    write_json(
        run / "failure.json",
        {**record, "title": "injected credential", "action": "injected command"},
    )
    studio = Studio(run.parent)
    studio.handle("TAB")
    studio.handle("ENTER")
    text = "\n".join(studio._run_details())
    assert "FAILURE AND NEXT STEPS" in text and "Training ran out of memory" in text
    assert "Reduce context length" in text and "injected" not in text
    assert "does not resume a checkpoint" in text
    for _ in range(100):
        studio.handle("PAGEDOWN")
    lines = studio.lines(74, 24)
    assert len(lines) <= 24 and all(cell_len(line) <= 74 for line in lines)


def test_remote_failure_details_warn_before_a_new_paid_launch(run):
    plan = read_json(run / "plan.json")
    plan["config"] = {**plan["config"], "provider": "runpod"}
    write_json(run / "plan.json", plan)
    worker._record_failure(run, FinetuneFailure("provider_failed"), stage="training")
    studio = Studio(run.parent)
    studio.handle("TAB")
    assert "active resources or charges" in " ".join(studio._run_details())


def test_known_failure_requires_a_registered_code():
    with pytest.raises(ValueError, match="Unknown"):
        FinetuneFailure("arbitrary")


def test_deeply_nested_diagnosis_does_not_crash_readers(run):
    (run / "failure.json").write_text("[" * 2000 + "]" * 2000)
    assert not failure_info(run, {"status": "failed"})["recorded"]
