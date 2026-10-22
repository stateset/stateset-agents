"""Owned worker outcomes and unconfirmed direct execution without paid compute."""

import errno
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as cli
from stateset_agents import finetune_quality as quality
from stateset_agents import finetune_recovery as recovery
from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobResult
from stateset_agents.tui import Studio


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    source = tmp_path / "chat.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "PRIVATE training question"},
                    {"role": "assistant", "content": "PRIVATE training response"},
                ]
            }
        )
        + "\n"
    )
    monkeypatch.setattr(worker.os, "umask", lambda _mask: 0)

    def create(provider="local"):
        config = workspace.FinetuneConfig(
            dataset=str(source),
            provider=provider,
            revision="a" * 40 if provider == "local" else "",
            seed=0,
            epochs=3,
            cost_policy="provider_billed",
        )
        run = workspace.create_run(tmp_path / "workspace", workspace.plan_run(config))
        return run, config

    return create


def install_training(monkeypatch, run, config, observe=lambda: None):
    output = run / "artifacts"
    output.mkdir()
    (output / "adapter_model.safetensors").write_bytes(b"synthetic fixture adapter")
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
                return JobHandle(config.provider, "synthetic-submission")

            def wait(self, handle):
                return RemoteJobResult(handle, JobStatus.SUCCEEDED, output, [])

        monkeypatch.setattr(registry, "get_executor", lambda _name: Executor())


def lose_owners(monkeypatch):
    """Report every recorded owner as exited on Linux and non-Linux probes."""

    def absent(_pid):
        raise ProcessLookupError()

    monkeypatch.setattr(workspace, "process_token", lambda _pid: None)
    monkeypatch.setattr(workspace, "probe_process", absent)


def supervised(run):
    token = "a" * 64
    assert worker._claim_admission(run, "supervised", token)
    assert worker._claim_owner(run, "supervisor.json")
    workspace.write_json(run / "state.json", {"status": "queued"})
    return token


@pytest.mark.parametrize("provider", ["local", "modal"])
def test_successful_direct_execution_commits_terminal_metadata(
    make_run, monkeypatch, provider
):
    run, config = make_run(provider)
    install_training(monkeypatch, run, config)
    assert worker.execute(run) == 0
    state = workspace.read_json(run / "state.json")
    assert state["status"] == "succeeded" and state["error"] is None
    assert state["finished_at"] >= state["training_started_at"]
    assert "exit_code" not in state
    assert not (run / "supervisor.json").exists()
    assert not workspace.read_json(run / "completion.json")["quality_gain_verified"]
    monkeypatch.setattr(workspace, "process_token", lambda _pid: None)
    reported = workspace.run_status(run)
    assert reported["status"] == "succeeded"
    assert workspace.run_terminal_status(reported) is True
    assert reported["execution_unconfirmed"] is False

    result = CliRunner().invoke(
        cli.app, ["status", run.name, "--workspace", str(run.parent)]
    )
    assert result.exit_code == 0
    metadata = json.loads(result.stdout)["result"]["run"]
    assert metadata["terminal"] is True and metadata["state"]["status"] == "succeeded"
    studio = Studio(run.parent)
    studio.run_filter = "active"
    studio.refresh()
    assert studio.runs == []


@pytest.mark.parametrize("provider", ["local", "modal"])
def test_worker_success_survives_lost_supervision_without_claiming_judge_quality(
    make_run, monkeypatch, provider
):
    run, config = make_run(provider)
    token = supervised(run)
    lose_owners(monkeypatch)

    def during_training():
        status = workspace.run_status(run)
        assert status["status"] == "interrupted"
        assert status["execution_unconfirmed"] is True

    install_training(monkeypatch, run, config, during_training)
    assert worker.execute(run, supervisor_token=token) == 0
    status = workspace.run_status(run)
    assert status["status"] == "succeeded" and status["execution_unconfirmed"] is False
    assert workspace.run_terminal_status(status) is True
    assert "exit_code" not in status
    assert not workspace.read_json(run / "completion.json")["quality_gain_verified"]


@pytest.mark.parametrize("observed", [None, "reused-owner", "original-owner"])
def test_direct_owner_loss_or_reuse_derives_uncertainty_without_writing_state(
    make_run, monkeypatch, observed
):
    run, _ = make_run()
    workspace.write_json(run / "state.json", {"status": "running"})
    workspace.write_json(
        run / "execution.json", {"pid": 200001, "process_token": "original-owner"}
    )
    monkeypatch.setattr(workspace, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(workspace, "process_token", lambda _pid: observed)
    before = (run / "state.json").read_bytes()
    status = workspace.run_status(run)
    if observed == "original-owner":
        assert status["status"] == "running" and not status["execution_unconfirmed"]
        assert workspace.run_terminal_status(status) is False
    else:
        assert status["status"] == "interrupted" and status["execution_unconfirmed"]
        assert workspace.run_terminal_status(status) is None
    assert (run / "state.json").read_bytes() == before


@pytest.mark.parametrize("outcome", ["alive", "permission", "missing", "overflow"])
def test_nonlinux_direct_owner_probe_preserves_existing_platform_semantics(
    make_run, monkeypatch, outcome
):
    run, _ = make_run()
    workspace.write_json(run / "state.json", {"status": "running"})
    workspace.write_json(run / "execution.json", {"pid": 200001, "process_token": None})
    monkeypatch.setattr(workspace, "sys", SimpleNamespace(platform="win32"))
    calls = []

    def probe(pid):
        calls.append(pid)
        if outcome == "permission":
            raise PermissionError()
        if outcome == "missing":
            raise ProcessLookupError()
        if outcome == "overflow":
            raise OverflowError()

    monkeypatch.setattr(workspace, "probe_process", probe)
    status = workspace.run_status(run)
    assert status["status"] == (
        "running" if outcome in {"alive", "permission"} else "interrupted"
    )
    assert calls == [200001]


def test_prepared_run_without_owners_remains_available(make_run, monkeypatch):
    run, _ = make_run()
    monkeypatch.setattr(
        workspace,
        "process_token",
        lambda *_args: pytest.fail("Unstarted run has no process"),
    )
    status = workspace.run_status(run)
    assert status["status"] == "prepared" and not status["execution_unconfirmed"]
    assert workspace.run_terminal_status(status) is False


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "empty",
        "partial",
        "invalid",
        "oversize",
        "symlink",
        "fifo",
        "directory",
        "boolean_pid",
    ],
)
def test_unreadable_direct_owner_remains_unconfirmed_with_bounded_reads(
    make_run, monkeypatch, kind
):
    run, _ = make_run()
    workspace.write_json(run / "state.json", {"status": "running"})
    path = run / "execution.json"
    if kind == "empty":
        path.touch()
    elif kind == "partial":
        path.write_text('{"pid":')
    elif kind == "invalid":
        path.write_text("[]")
    elif kind == "oversize":
        path.write_text(" " * 4097)
    elif kind == "symlink":
        target = run.parent / "PRIVATE-owner.json"
        workspace.write_json(target, {"pid": 200001, "process_token": "private"})
        path.symlink_to(target)
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("Named pipes are unavailable")
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    elif kind == "boolean_pid":
        workspace.write_json(path, {"pid": True, "process_token": "invalid"})
    monkeypatch.setattr(
        workspace,
        "process_token",
        lambda *_args: pytest.fail("Invalid owner must not probe a process"),
    )
    before = (run / "state.json").read_bytes()
    status = workspace.run_status(run)
    assert status["status"] == "interrupted" and status["execution_unconfirmed"]
    assert workspace.run_terminal_status(status) is None
    assert (run / "state.json").read_bytes() == before
    assert not (run / "failure.json").exists()


@pytest.mark.parametrize("kind", ["valid", "partial", "fifo"])
def test_completion_file_alone_never_promotes_an_unfinished_direct_worker(
    make_run, monkeypatch, kind
):
    run, _ = make_run()
    workspace.write_json(run / "state.json", {"status": "running"})
    workspace.write_json(
        run / "execution.json", {"pid": 200001, "process_token": "missing"}
    )
    if kind == "valid":
        workspace.write_json(
            run / "completion.json", {"output_dir": str(run / "artifacts")}
        )
    elif kind == "partial":
        (run / "completion.json").write_text('{"output_dir":')
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("Named pipes are unavailable")
        os.mkfifo(run / "completion.json")
    monkeypatch.setattr(workspace, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(workspace, "process_token", lambda _pid: None)
    read = workspace.read_json

    def metadata(path, **kwargs):
        assert (
            path.name != "completion.json"
        ), "Status must not inspect completion artifacts"
        return read(path, **kwargs)

    monkeypatch.setattr(workspace, "read_json", metadata)
    status = workspace.run_status(run)
    assert (
        status["status"] == "interrupted"
        and workspace.run_terminal_status(status) is None
    )


@pytest.mark.parametrize("failure", ["temporary", "persistent", "crash"])
def test_success_metadata_failure_never_reports_a_successful_worker_return(
    make_run, monkeypatch, failure
):
    run, config = make_run()
    install_training(monkeypatch, run, config)
    state = worker._state

    class SimulatedDeath(BaseException):
        pass

    def fail(path, **changes):
        if changes.get("status") == "succeeded":
            assert (run / "completion.json").is_file()
            if failure == "crash":
                raise SimulatedDeath()
            raise OSError(errno.ENOSPC, "synthetic full disk")
        if failure == "persistent" and changes.get("status") == "failed":
            raise OSError(errno.ENOSPC, "synthetic full disk")
        return state(path, **changes)

    monkeypatch.setattr(worker, "_state", fail)
    if failure == "crash":
        with pytest.raises(SimulatedDeath):
            worker.execute(run)
    else:
        assert worker.execute(run) == 1
    assert (run / "completion.json").exists()
    lose_owners(monkeypatch)
    reported = workspace.run_status(run)
    if failure == "temporary":
        assert reported["status"] == "failed"
        assert workspace.read_json(run / "failure.json")["code"] == "disk_full"
    else:
        assert reported["status"] == "interrupted"
        assert (
            reported["execution_unconfirmed"]
            and workspace.run_terminal_status(reported) is None
        )


def test_late_supervisor_failure_can_override_committed_worker_outcome(
    make_run, monkeypatch
):
    run, config = make_run()
    install_training(monkeypatch, run, config)
    monkeypatch.setattr(worker, "queue_executable", lambda: None)

    def start(*_args, **kwargs):
        def wait(**_kwargs):
            token = kwargs["env"][worker.SUPERVISOR_TOKEN_ENV]
            assert worker.execute(run, supervisor_token=token) == 0
            outcome = workspace.read_json(run / "state.json")
            assert outcome["status"] == "succeeded" and "exit_code" not in outcome
            return 2

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(worker.subprocess, "Popen", start)
    assert worker.supervise(run) == 2
    reported = workspace.run_status(run)
    assert reported["status"] == "failed" and reported["exit_code"] == 2
    assert workspace.read_json(run / "failure.json")["code"] == "worker_exit"
    assert (run / "completion.json").exists()


@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("boundary", ["before_running", "during_training"])
def test_earlier_supervisor_outcome_survives_late_worker(
    make_run, monkeypatch, status, boundary
):
    run, config = make_run()
    recorded = {}

    def supervisor_failure():
        worker._record_failure(
            run,
            workspace.FinetuneFailure(
                "cancelled" if status == "cancelled" else "worker_exit"
            ),
            stage="supervisor",
            exit_code=2,
        )
        recorded["state"] = (run / "state.json").read_bytes()
        recorded["failure"] = (run / "failure.json").read_bytes()

    install_training(
        monkeypatch,
        run,
        config,
        (
            supervisor_failure
            if boundary == "during_training"
            else lambda: pytest.fail("training started")
        ),
    )
    if boundary == "before_running":
        settings = worker._settings

        def before_running(path):
            value = settings(path)
            supervisor_failure()
            return value

        monkeypatch.setattr(worker, "_settings", before_running)
    assert worker.execute(run) == 1
    assert (run / "state.json").read_bytes() == recorded["state"]
    assert (run / "failure.json").read_bytes() == recorded["failure"]
    assert workspace.read_json(run / "state.json")["exit_code"] == 2
    assert (run / "completion.json").exists() == (boundary == "during_training")


@pytest.mark.parametrize("first", ["succeeded", "failed"])
def test_concurrent_state_transitions_serialize_and_preserve_failure(
    make_run, monkeypatch, first
):
    run, _ = make_run()
    entered, release, contended = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    write = worker.write_json
    lock_operation = worker._state_lock_operation
    second = "failed" if first == "succeeded" else "succeeded"

    def held_write(path, value):
        if path == run / "state.json" and value.get("status") == first:
            entered.set()
            assert release.wait(2), "state writer was not released"
        return write(path, value)

    def observed_lock(descriptor, *, release=False):
        try:
            return lock_operation(descriptor, release=release)
        except OSError as error:
            if not release and error.errno in {
                errno.EAGAIN,
                errno.EACCES,
                errno.EWOULDBLOCK,
            }:
                contended.set()
            raise

    monkeypatch.setattr(worker, "write_json", held_write)
    monkeypatch.setattr(worker, "_state_lock_operation", observed_lock)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_result = executor.submit(
            worker._state, run, status=first, committed_updates=3
        )
        try:
            assert entered.wait(2), "first writer did not hold the state lock"
            second_result = executor.submit(
                worker._state, run, status=second, exit_code=2
            )
            assert contended.wait(
                2
            ), "second writer did not contend on the descriptor lock"
        finally:
            release.set()
        assert first_result.result(timeout=2) is True
        assert second_result.result(timeout=2) is (second == "failed")
    state = workspace.read_json(run / "state.json")
    assert state["status"] == "failed" and state["committed_updates"] == 3
    before = (run / "state.json").read_bytes()
    assert worker._state(run, status="running") is False
    assert (run / "state.json").read_bytes() == before
    assert worker._state(run, recovery_checkpoint="checkpoint-3") is True
    assert workspace.read_json(run / "state.json")["status"] == "failed"


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "public_file"])
def test_state_lock_rejects_unsafe_paths_without_mutating_state(make_run, kind):
    run, _ = make_run()
    lock = run / ".state.lock"
    outside = run.parent / "outside-lock"
    outside.write_bytes(b"unchanged")
    if kind == "symlink":
        try:
            lock.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks unavailable")
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFOs unavailable")
        os.mkfifo(lock, 0o600)
    elif kind == "directory":
        lock.mkdir()
    else:
        if os.name == "nt":
            pytest.skip("POSIX permissions unavailable")
        lock.write_bytes(b"")
        lock.chmod(0o644)
    before = (run / "state.json").read_bytes()
    with pytest.raises((OSError, ValueError)):
        worker._state(run, status="running")
    assert (run / "state.json").read_bytes() == before
    assert outside.read_bytes() == b"unchanged"


def test_busy_state_lock_refuses_boundedly_and_retains_stable_private_inode(
    make_run, monkeypatch
):
    run, _ = make_run()
    monkeypatch.setattr(worker, "_STATE_LOCK_TIMEOUT", 0.0)
    before = (run / "state.json").read_bytes()
    with worker._state_lock(run):
        original = (run / ".state.lock").stat()
        with pytest.raises(ValueError, match="Fine-tuning state is busy"):
            worker._state(run, status="running")
    assert (run / "state.json").read_bytes() == before
    assert worker._state(run, status="running") is True
    retained = (run / ".state.lock").stat()
    assert (original.st_dev, original.st_ino) == (retained.st_dev, retained.st_ino)
    if os.name != "nt":
        assert retained.st_mode & 0o077 == 0


def test_owned_stop_boundary_cannot_commit_success(make_run, monkeypatch):
    run, config = make_run()

    def stop():
        digest = workspace.read_run_plan(run)[1]
        workspace.request_stop(run, expected_plan_sha256=digest)

    install_training(monkeypatch, run, config, stop)
    assert worker.execute(run) == 1
    reported = workspace.run_status(run)
    assert reported["status"] == "cancelled" and workspace.run_terminal_status(reported)
    assert not (run / "completion.json").exists()


def test_committed_local_success_is_visible_to_quality_without_certifying_gain(
    make_run, monkeypatch
):
    from tests.unit.test_finetune_quality import evidence_for

    run, config = make_run()
    install_training(monkeypatch, run, config)
    assert worker.execute(run) == 0
    summary = quality.record_quality_document(run, evidence_for(run, config))
    assert summary["seeds"] == 1 and not summary["recorded_study_passed"]
    assert quality.inspect_quality(run)["artifact_sha256"] == summary["artifact_sha256"]


def test_recovery_still_requires_exited_owner_after_committed_success(
    make_run, monkeypatch
):
    if not sys.platform.startswith("linux"):
        pytest.skip("Studio recovery verifies Linux process ownership")
    run, config = make_run()
    install_training(monkeypatch, run, config)
    checkpoint = run / "artifacts/checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"synthetic checkpoint")
    workspace.write_json(
        checkpoint / "adapter_config.json", {"base_model_name_or_path": config.model}
    )
    workspace.write_json(
        checkpoint / "trainer_state.json", {"global_step": 1, "epoch": 1.0}
    )
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(b"synthetic saved state")
    recovery.publish_checkpoint(run, checkpoint, step=1)
    assert worker.execute(run) == 0
    with pytest.raises(ValueError, match="Wait for the parent worker"):
        recovery.inspect_checkpoints(run)
    monkeypatch.setattr(recovery, "process_token", lambda _pid: None)

    def exited(*_args):
        raise ProcessLookupError()

    monkeypatch.setattr(recovery.os, "kill", exited)
    report = recovery.inspect_checkpoints(run)
    assert report["checkpoints"][0]["available"]
    assert report["checkpoints"][0]["remaining"] == 2.0


@pytest.mark.parametrize("raw_status", ["running", "interrupted"])
def test_unconfirmed_ownerless_parent_cannot_be_inspected_or_resumed(
    tmp_path, monkeypatch, raw_status
):
    from tests.unit.test_finetune_recovery import make_parent

    run, _ = make_parent(tmp_path)
    selected = recovery.inspect_checkpoints(run)["checkpoints"][0]
    workspace.write_json(run / "state.json", {"status": raw_status})
    assert not (run / "supervisor.json").exists()
    assert not (run / "execution.json").exists()
    reported = workspace.run_status(run)
    assert reported["status"] == "interrupted" and reported["execution_unconfirmed"]
    before = (run / "state.json").read_bytes()

    def no_child(*_args, **_kwargs):
        pytest.fail("unconfirmed parent reached child creation")

    monkeypatch.setattr(recovery, "create_run", no_child)
    with pytest.raises(ValueError, match="Cannot verify the parent worker's lifecycle"):
        recovery.inspect_checkpoints(run)
    with pytest.raises(ValueError, match="Cannot verify the parent worker's lifecycle"):
        recovery.prepare_resume(
            run,
            selected["checkpoint"],
            expected_recovery_sha256=selected["recovery_sha256"],
        )
    assert (run / "state.json").read_bytes() == before
    assert len(workspace.list_runs(run.parent)) == 1
    assert not list(run.parent.glob(".resume-*"))


@pytest.mark.parametrize("raw_status", ["failed", "succeeded", "cancelled"])
def test_confirmed_legacy_terminal_parent_without_owners_can_still_resume(
    tmp_path, raw_status
):
    from tests.unit.test_finetune_recovery import make_parent

    run, _ = make_parent(tmp_path)
    workspace.write_json(run / "state.json", {"status": raw_status})
    selected = recovery.inspect_checkpoints(run)["checkpoints"][0]
    child = recovery.prepare_resume(
        run,
        selected["checkpoint"],
        expected_recovery_sha256=selected["recovery_sha256"],
    )
    assert child != run and workspace.run_status(child)["status"] == "prepared"
    assert workspace.read_json(run / "state.json")["status"] == raw_status
