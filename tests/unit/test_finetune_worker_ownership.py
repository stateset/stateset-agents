"""Repeated studio worker entry must preserve the winning run and its work."""

import errno
import json
import os
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobResult

RECORDS = (
    "admission.json",
    "state.json",
    "failure.json",
    "completion.json",
    "supervisor.json",
    "execution.json",
    "command.json",
    "provider-job.json",
    "provider-result.json",
    "run.log",
)


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    source = tmp_path / "dataset.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Fixture question"},
                    {"role": "assistant", "content": "Fixture answer"},
                ]
            }
        )
        + "\n"
    )
    monkeypatch.setattr(worker.os, "umask", lambda _mask: 0)

    def create(provider="local"):
        config = workspace.FinetuneConfig(
            dataset=str(source), provider=provider, cost_policy="provider_billed"
        )
        run = workspace.create_run(tmp_path / "workspace", workspace.plan_run(config))
        return run, config

    return create


def records(run, *, exclude=()):
    return {
        name: (run / name).read_bytes() if (run / name).exists() else None
        for name in RECORDS
        if name not in exclude
    }


def install_training(monkeypatch, run, config, observe):
    output = run / "artifacts"
    output.mkdir()
    (output / "adapter_model.safetensors").write_bytes(b"synthetic local adapter")
    if config.provider == "local":
        from stateset_agents.training import sft

        monkeypatch.setattr(sft, "gpu_available", lambda: True)

        def train(*_args):
            observe()
            return output

        monkeypatch.setattr(worker, "_train_sft", train)
    else:
        from stateset_agents.remote import registry

        class Executor:
            def submit(self, spec):
                assert spec.dataset == run / "dataset"
                observe()
                return JobHandle(config.provider, "fixture-observed-submission")

            def wait(self, handle):
                return RemoteJobResult(handle, JobStatus.SUCCEEDED, output, [])

        monkeypatch.setattr(registry, "get_executor", lambda _name: Executor())


@pytest.mark.parametrize("provider", ["local", "modal"])
def test_concurrent_execution_has_one_training_owner(make_run, monkeypatch, provider):
    run, config = make_run(provider)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def train():
        calls.append("training")
        entered.set()
        assert release.wait(5), "Test did not release its injected training call"

    install_training(monkeypatch, run, config, train)
    settings = MagicMock(wraps=worker._settings)
    monkeypatch.setattr(worker, "_settings", settings)
    result = []
    thread = threading.Thread(
        target=lambda: result.append(worker.execute(run)), daemon=True
    )
    thread.start()
    try:
        assert entered.wait(5), "First execution did not reach injected training"
        before = records(run)
        assert worker.execute(run) == 1
        assert records(run) == before
        assert calls == ["training"] and settings.call_count == 1
        owner = workspace.read_json(run / "execution.json")
        assert owner == {
            "pid": os.getpid(),
            "process_token": worker.process_token(os.getpid()),
        }
        if os.name != "nt":
            assert (run / "execution.json").stat().st_mode & 0o077 == 0
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and result == [0]
    assert (run / "completion.json").exists()
    assert not (run / "failure.json").exists()


def test_concurrent_supervisor_does_not_enqueue_or_overwrite_winner(
    make_run, monkeypatch
):
    run, _ = make_run()
    entered, release = threading.Event(), threading.Event()
    calls = []
    monkeypatch.setattr(worker, "queue_executable", lambda: "/fixture/buildr")

    def start(*_args, **_kwargs):
        calls.append("queue")

        def wait(**_kwargs):
            entered.set()
            assert release.wait(5), "Test did not release its injected queue wait"
            workspace.write_json(run / "completion.json", {"output_dir": "artifacts"})
            return 0

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(worker.subprocess, "Popen", start)
    settings = MagicMock(wraps=worker._settings)
    monkeypatch.setattr(worker, "_settings", settings)
    result = []
    thread = threading.Thread(
        target=lambda: result.append(worker.supervise(run)), daemon=True
    )
    thread.start()
    try:
        assert entered.wait(5), "First supervisor did not reach injected queue wait"
        before = records(run)
        assert worker.supervise(run) == 1
        assert records(run) == before
        assert calls == ["queue"] and settings.call_count == 1
        assert workspace.read_json(run / "state.json")["status"] == "queued"
        if os.name != "nt":
            assert (run / "supervisor.json").stat().st_mode & 0o077 == 0
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive() and result == [0]
    assert workspace.read_json(run / "state.json")["status"] == "succeeded"


@pytest.mark.parametrize("provider", ["local", "modal"])
def test_completed_execution_cannot_be_repeated_or_later_supervised(
    make_run, monkeypatch, provider
):
    run, config = make_run(provider)
    calls = []
    install_training(monkeypatch, run, config, lambda: calls.append("training"))
    assert worker.execute(run) == 0
    assert not (run / "supervisor.json").exists()
    before = records(run)
    assert worker.execute(run) == 1
    assert worker.supervise(run) == 1
    assert records(run) == before
    assert calls == ["training"]


def test_supervision_and_execution_have_separate_private_claims(make_run, monkeypatch):
    run, config = make_run()
    calls = []
    install_training(monkeypatch, run, config, lambda: calls.append("training"))
    monkeypatch.setattr(worker, "queue_executable", lambda: None)
    monkeypatch.setattr(
        worker.subprocess,
        "Popen",
        lambda *_args, **process_kwargs: SimpleNamespace(
            wait=lambda **_kwargs: worker.execute(
                run, supervisor_token=process_kwargs["env"][worker.SUPERVISOR_TOKEN_ENV]
            )
        ),
    )
    assert worker.supervise(run) == 0
    assert (run / "supervisor.json").exists() and (run / "execution.json").exists()
    assert workspace.read_json(run / "state.json")["status"] == "succeeded"
    before = records(run)
    assert worker.supervise(run) == 1 and worker.execute(run) == 1
    assert records(run) == before and calls == ["training"]


@pytest.mark.parametrize(
    "entrypoint,filename",
    [("execute", "execution.json"), ("supervise", "supervisor.json")],
)
@pytest.mark.parametrize(
    "kind", ["crashed", "empty", "partial", "invalid", "symlink", "fifo", "directory"]
)
def test_existing_or_uncertain_owners_are_never_replaced(
    make_run, monkeypatch, entrypoint, filename, kind
):
    run, _ = make_run()
    owner = run / filename
    if kind == "crashed":
        workspace.write_json(owner, {"pid": 99999999, "process_token": "dead-owner"})
    elif kind == "empty":
        owner.touch()
    elif kind == "partial":
        owner.write_text('{"pid":')
    elif kind == "invalid":
        owner.write_text("[]")
    elif kind == "symlink":
        target = run.parent / "PRIVATE-owner.txt"
        target.write_bytes(b"PRIVATE external owner")
        owner.symlink_to(target)
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("Named pipes are unavailable")
        os.mkfifo(owner)
    else:
        owner.mkdir()
    initial = owner.lstat()
    if kind == "fifo":
        original_open = os.open

        def safe_open(path, flags, *args, **kwargs):
            if os.fspath(path) == os.fspath(owner):
                # Fail before an incorrectly blocking FIFO open can hang pytest.
                assert flags & os.O_EXCL
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(worker.os, "open", safe_open)
    before = records(run, exclude=(filename,))
    forbidden = MagicMock(
        side_effect=AssertionError("Duplicate must not inspect or mutate")
    )
    monkeypatch.setattr(worker, "_settings", forbidden)
    monkeypatch.setattr(worker, "_record_failure", forbidden)
    assert getattr(worker, entrypoint)(run) == 1
    assert records(run, exclude=(filename,)) == before
    after = owner.lstat()
    assert (after.st_ino, after.st_mode, after.st_size) == (
        initial.st_ino,
        initial.st_mode,
        initial.st_size,
    )
    forbidden.assert_not_called()
    if kind == "symlink":
        assert target.read_bytes() == b"PRIVATE external owner"


@pytest.mark.parametrize("kind", ["empty", "invalid", "symlink", "fifo", "directory"])
def test_preexisting_execution_claim_prevents_a_new_supervisor(
    make_run, monkeypatch, kind
):
    run, _ = make_run()
    owner = run / "execution.json"
    if kind == "empty":
        owner.touch()
    elif kind == "invalid":
        owner.write_text("not valid JSON")
    elif kind == "symlink":
        owner.symlink_to(run.parent / "missing-owner")
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("Named pipes are unavailable")
        os.mkfifo(owner)
    else:
        owner.mkdir()
    before = records(run, exclude=("execution.json",))
    forbidden = MagicMock(
        side_effect=AssertionError("Must not queue an existing execution")
    )
    monkeypatch.setattr(worker, "_settings", forbidden)
    monkeypatch.setattr(worker.subprocess, "Popen", forbidden)
    monkeypatch.setattr(worker, "_record_failure", forbidden)
    assert worker.supervise(run) == 1
    assert records(run, exclude=("execution.json",)) == before
    assert not (run / "supervisor.json").exists()
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "status", ["succeeded", "failed", "cancelled", "interrupted", "queued", "running"]
)
def test_nonprepared_run_cannot_start_a_new_supervisor(make_run, monkeypatch, status):
    run, _ = make_run()
    workspace.write_json(run / "state.json", {"status": status})
    before = records(run)
    forbidden = MagicMock(side_effect=AssertionError("Must not enter supervisor setup"))
    monkeypatch.setattr(worker, "_settings", forbidden)
    monkeypatch.setattr(worker.subprocess, "Popen", forbidden)
    assert worker.supervise(run) == 1
    assert records(run) == before
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "status", ["succeeded", "failed", "cancelled", "interrupted", "running"]
)
def test_terminal_or_uncertain_execution_without_owner_is_not_reclaimed(
    make_run, monkeypatch, status
):
    run, _ = make_run()
    workspace.write_json(run / "state.json", {"status": status})
    workspace.write_json(run / "completion.json", {"historical": "preserved"})
    before = records(run)
    forbidden = MagicMock(
        side_effect=AssertionError("Must not reclaim uncertain execution")
    )
    monkeypatch.setattr(worker, "_settings", forbidden)
    monkeypatch.setattr(worker, "_record_failure", forbidden)
    assert not (run / "execution.json").exists()
    assert worker.execute(run) == 1
    assert records(run) == before
    forbidden.assert_not_called()


@pytest.mark.parametrize("status", ["starting", "queued"])
def test_first_admitted_execution_preserves_supported_phase_entry(
    make_run, monkeypatch, status
):
    run, config = make_run()
    token = "a" * 64
    assert worker._claim_admission(run, "supervised", token)
    assert worker._claim_owner(run, "supervisor.json")
    workspace.write_json(run / "state.json", {"status": status})
    calls = []
    install_training(monkeypatch, run, config, lambda: calls.append("training"))
    assert worker.execute(run, supervisor_token=token) == 0
    assert calls == ["training"] and (run / "completion.json").exists()


def test_execution_rechecks_state_after_its_claim(make_run, monkeypatch):
    run, _ = make_run()
    claim = worker._claim_owner
    captured = []

    def changed(path, filename):
        assert claim(path, filename)
        workspace.write_json(run / "state.json", {"status": "cancelled"})
        captured.append(records(run))
        return True

    monkeypatch.setattr(worker, "_claim_owner", changed)
    forbidden = MagicMock(
        side_effect=AssertionError("Changed phase must not start work")
    )
    monkeypatch.setattr(worker, "_settings", forbidden)
    monkeypatch.setattr(worker, "_record_failure", forbidden)
    assert worker.execute(run) == 1
    assert records(run) == captured[0]
    forbidden.assert_not_called()


def test_supervisor_rechecks_for_execution_before_queueing(make_run, monkeypatch):
    run, _ = make_run()
    settings = worker._settings
    captured = []

    def changed(path):
        result = settings(path)
        workspace.write_json(
            run / "execution.json", {"pid": 99999999, "process_token": "existing"}
        )
        workspace.write_json(run / "state.json", {"status": "running"})
        captured.append(records(run))
        return result

    monkeypatch.setattr(worker, "_settings", changed)
    forbidden = MagicMock(
        side_effect=AssertionError("Must not queue an existing execution")
    )
    monkeypatch.setattr(worker.subprocess, "Popen", forbidden)
    monkeypatch.setattr(worker, "_record_failure", forbidden)
    assert worker.supervise(run) == 1
    assert records(run) == captured[0]
    forbidden.assert_not_called()


@pytest.mark.parametrize("foreign_receipt", [False, True])
def test_private_supervisor_loop_requires_previously_claimed_current_owner(
    make_run, monkeypatch, foreign_receipt
):
    run, config = make_run()
    if foreign_receipt:
        workspace.write_json(
            run / "supervisor.json", {"pid": 99999999, "process_token": "foreign"}
        )
    before = records(run)
    forbidden = MagicMock(
        side_effect=AssertionError("Unowned private entry must not queue")
    )
    monkeypatch.setattr(worker.subprocess, "Popen", forbidden)
    with pytest.raises((OSError, ValueError)):
        worker._supervise(run, config)
    assert records(run) == before
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "entrypoint,filename",
    [("execute", "execution.json"), ("supervise", "supervisor.json")],
)
def test_ownership_permission_failure_cannot_mutate_an_unowned_run(
    make_run, monkeypatch, entrypoint, filename
):
    run, _ = make_run()
    original_open = os.open

    def denied(path, flags, *args, **kwargs):
        if os.fspath(path) == os.fspath(run / "admission.json"):
            raise PermissionError("PRIVATE ownership detail")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(worker.os, "open", denied)
    forbidden = MagicMock(side_effect=AssertionError("Unowned caller must not mutate"))
    monkeypatch.setattr(worker, "_settings", forbidden)
    monkeypatch.setattr(worker, "_record_failure", forbidden)
    before = records(run)
    assert getattr(worker, entrypoint)(run) == 1
    assert records(run) == before
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "entrypoint,filename",
    [("execute", "execution.json"), ("supervise", "supervisor.json")],
)
def test_incomplete_owner_publication_never_starts_work_or_retries(
    make_run, monkeypatch, entrypoint, filename
):
    run, _ = make_run()
    fsync = MagicMock(side_effect=OSError(errno.ENOSPC, "PRIVATE disk detail"))
    monkeypatch.setattr(worker.os, "fsync", fsync)
    forbidden = MagicMock(
        side_effect=AssertionError("Uncertain owner must not start work")
    )
    monkeypatch.setattr(worker, "_settings", forbidden)
    monkeypatch.setattr(worker, "_record_failure", forbidden)
    before = records(run, exclude=("admission.json",))
    assert getattr(worker, entrypoint)(run) == 1
    assert (run / "admission.json").exists()
    owner = (run / "admission.json").read_bytes()
    assert records(run, exclude=("admission.json",)) == before
    assert getattr(worker, entrypoint)(run) == 1
    assert (run / "admission.json").read_bytes() == owner
    assert fsync.call_count == 1
    forbidden.assert_not_called()


@pytest.mark.parametrize("entrypoint", ["execute", "supervise"])
def test_claimed_worker_keeps_controlled_review_failure_diagnostics(
    make_run, monkeypatch, entrypoint
):
    run, _ = make_run()
    (run / "dataset").write_text("changed since review")
    forbidden = MagicMock(side_effect=AssertionError("Invalid input must not train"))
    monkeypatch.setattr(worker, "_train_sft", forbidden)
    monkeypatch.setattr(worker.subprocess, "Popen", forbidden)
    assert getattr(worker, entrypoint)(run) == 1
    assert workspace.read_json(run / "failure.json")["code"] == "review_invalid"
    assert workspace.read_json(run / "state.json")["status"] == "failed"
    forbidden.assert_not_called()


@pytest.mark.parametrize("boundary", ["before_owner", "queue", "popen"])
def test_supervised_admission_excludes_direct_execution_at_each_boundary(
    make_run, monkeypatch, boundary
):
    run, config = make_run()
    calls, rejected = [], []
    install_training(monkeypatch, run, config, lambda: calls.append("training"))

    def direct_attempt():
        before = records(run)
        rejected.append(worker.execute(run))
        assert records(run) == before

    claim = worker._claim_owner

    def owner(path, filename):
        if boundary == "before_owner" and filename == "supervisor.json":
            direct_attempt()
        return claim(path, filename)

    monkeypatch.setattr(worker, "_claim_owner", owner)

    def queue():
        if boundary == "queue":
            direct_attempt()
        return None

    monkeypatch.setattr(worker, "queue_executable", queue)

    def start(argv, **kwargs):
        if boundary == "popen":
            direct_attempt()
        token = kwargs["env"][worker.SUPERVISOR_TOKEN_ENV]
        assert "--supervised" in argv
        assert token not in json.dumps(argv)
        assert token not in (run / "command.json").read_text()
        if os.name != "nt":
            assert (run / "admission.json").stat().st_mode & 0o077 == 0
        return SimpleNamespace(
            wait=lambda **_kwargs: worker.execute(run, supervisor_token=token)
        )

    monkeypatch.setattr(worker.subprocess, "Popen", start)
    assert worker.supervise(run) == 0
    assert rejected == [1] and calls == ["training"]
    assert workspace.read_json(run / "state.json")["status"] == "succeeded"
    assert not (run / "failure.json").exists()


def test_direct_admission_excludes_supervisor_before_execution_owner_exists(
    make_run, monkeypatch
):
    run, config = make_run()
    calls, rejected = [], []
    install_training(monkeypatch, run, config, lambda: calls.append("training"))
    claim = worker._claim_owner

    def owner(path, filename):
        if filename == "execution.json":
            assert not (run / "execution.json").exists()
            before = records(run)
            rejected.append(worker.supervise(run))
            assert records(run) == before
        return claim(path, filename)

    monkeypatch.setattr(worker, "_claim_owner", owner)
    forbidden = MagicMock(
        side_effect=AssertionError("Losing supervisor must not queue")
    )
    monkeypatch.setattr(worker.subprocess, "Popen", forbidden)
    assert worker.execute(run) == 0
    assert rejected == [1] and calls == ["training"]
    assert not (run / "supervisor.json").exists()
    assert not (run / "failure.json").exists()
    forbidden.assert_not_called()


@pytest.mark.parametrize("status", ["queued", "starting"])
@pytest.mark.parametrize("admission_present", [False, True])
def test_tokenless_execution_never_impersonates_a_supervised_child(
    make_run, monkeypatch, status, admission_present
):
    run, _ = make_run()
    token = "a" * 64
    if admission_present:
        assert worker._claim_admission(run, "supervised", token)
        assert worker._claim_owner(run, "supervisor.json")
    workspace.write_json(run / "state.json", {"status": status})
    monkeypatch.setenv(worker.SUPERVISOR_TOKEN_ENV, token)
    before = records(run)
    forbidden = MagicMock(
        side_effect=AssertionError("Ambient authority must not start work")
    )
    monkeypatch.setattr(worker, "_settings", forbidden)
    assert worker.execute(run) == 1
    assert records(run) == before
    assert worker.SUPERVISOR_TOKEN_ENV not in os.environ
    forbidden.assert_not_called()


@pytest.mark.parametrize("wrong", ["", "b" * 64, "PRIVATE-invalid-capability"])
def test_wrong_child_capability_cannot_mutate_admitted_run(
    make_run, monkeypatch, caplog, wrong
):
    run, _ = make_run()
    assert worker._claim_admission(run, "supervised", "a" * 64)
    assert worker._claim_owner(run, "supervisor.json")
    workspace.write_json(run / "state.json", {"status": "queued"})
    before = records(run)
    forbidden = MagicMock(
        side_effect=AssertionError("Invalid child must not start work")
    )
    monkeypatch.setattr(worker, "_settings", forbidden)
    assert worker.execute(run, supervisor_token=wrong) == 1
    assert records(run) == before
    if wrong:
        assert wrong not in caplog.text
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "partial",
        "invalid",
        "oversize",
        "symlink",
        "fifo",
        "bad_schema",
        "bad_owner",
    ],
)
def test_uncertain_child_admission_is_bounded_and_never_repaired(
    make_run, monkeypatch, kind
):
    run, _ = make_run()
    token = "a" * 64
    assert worker._claim_admission(run, "supervised", token)
    assert worker._claim_owner(run, "supervisor.json")
    workspace.write_json(run / "state.json", {"status": "queued"})
    path = run / "admission.json"
    if kind == "empty":
        path.write_text("")
    elif kind == "partial":
        path.write_text('{"mode":')
    elif kind == "invalid":
        path.write_text("[]")
    elif kind == "oversize":
        path.write_text(" " * 4097)
    elif kind == "symlink":
        path.unlink()
        path.symlink_to(run.parent / "missing-admission")
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("Named pipes are unavailable")
        path.unlink()
        os.mkfifo(path)
    elif kind == "bad_schema":
        value = workspace.read_json(path)
        value["schema_version"] = True
        workspace.write_json(path, value)
    else:
        workspace.write_json(
            run / "supervisor.json", {"pid": 99999999, "process_token": "foreign"}
        )
    before = records(run, exclude=("admission.json",))
    initial = path.lstat()
    forbidden = MagicMock(
        side_effect=AssertionError("Uncertain admission must not start work")
    )
    monkeypatch.setattr(worker, "_settings", forbidden)
    assert worker.execute(run, supervisor_token=token) == 1
    assert records(run, exclude=("admission.json",)) == before
    after = path.lstat()
    assert (after.st_ino, after.st_size, after.st_mode) == (
        initial.st_ino,
        initial.st_size,
        initial.st_mode,
    )
    forbidden.assert_not_called()


@pytest.mark.parametrize("supervised", [False, True])
def test_worker_cli_consumes_private_child_environment_before_entry(
    make_run, monkeypatch, supervised
):
    run, _ = make_run()
    token = "a" * 64
    monkeypatch.setenv(worker.SUPERVISOR_TOKEN_ENV, token)
    argv = ["worker", "--execute", str(run)]
    if supervised:
        argv.append("--supervised")
    monkeypatch.setattr(sys, "argv", argv)

    def execute(path, *, supervisor_token=None):
        assert path == run
        assert supervisor_token == (token if supervised else None)
        assert worker.SUPERVISOR_TOKEN_ENV not in os.environ
        return 0

    monkeypatch.setattr(worker, "execute", execute)
    assert worker.main() == 0


def test_supervised_cli_without_capability_never_falls_back_to_direct(
    make_run, monkeypatch
):
    run, _ = make_run()
    monkeypatch.delenv(worker.SUPERVISOR_TOKEN_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", ["worker", "--supervised", "--execute", str(run)])
    forbidden = MagicMock(
        side_effect=AssertionError("Missing capability must not enter worker")
    )
    monkeypatch.setattr(worker, "execute", forbidden)
    before = records(run)
    assert worker.main() == 1
    assert records(run) == before
    forbidden.assert_not_called()


def test_cli_unowned_dispatch_exception_cannot_overwrite_a_run(
    make_run, monkeypatch, caplog
):
    run, _ = make_run()
    workspace.write_json(run / "state.json", {"status": "running"})
    workspace.write_json(
        run / "execution.json", {"pid": 99999999, "process_token": "winner"}
    )
    before = records(run)
    monkeypatch.setattr(sys, "argv", ["worker", "--execute", str(run)])
    monkeypatch.setattr(
        worker,
        "execute",
        MagicMock(side_effect=RuntimeError("PRIVATE dispatch detail")),
    )
    forbidden = MagicMock(
        side_effect=AssertionError("Unowned CLI must not diagnose shared state")
    )
    monkeypatch.setattr(worker, "_record_failure", forbidden)
    assert worker.main() == 1
    assert records(run) == before
    assert "PRIVATE dispatch detail" not in caplog.text
    forbidden.assert_not_called()
