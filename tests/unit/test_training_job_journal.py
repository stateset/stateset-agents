"""Durable records recover ownership without replaying uncertain training."""

import asyncio
import json
import select
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from stateset_agents.training import advanced_training_orchestrator as module
from stateset_agents.training.advanced_training_models import serialize_training_job
from stateset_agents.training.job_journal import JobJournal


def job(name="job", status=module.TrainingStatus.QUEUED):
    return module.TrainingJob(
        name,
        module.TrainingJobSpec("test", "test", {}, "unused"),
        status=status,
    )


@pytest.fixture
def build(monkeypatch, tmp_path):
    instances = []
    monkeypatch.setattr(module.ResourceManager, "_detect_resources", lambda self: None)
    monkeypatch.setattr(module, "get_monitoring_service", MagicMock())

    def create(**kwargs):
        state = SimpleNamespace(set=AsyncMock(), get=AsyncMock(return_value=None))
        monkeypatch.setattr(
            module, "get_state_service", lambda: SimpleNamespace(state_manager=state)
        )
        runner = kwargs.pop("training_runner", AsyncMock())
        result = module.AdvancedTrainingOrchestrator(
            journal_dir=kwargs.pop("journal_dir", tmp_path / "journal"),
            start_background_tasks=False,
            enable_experiment_tracking=False,
            training_runner=runner,
            **kwargs,
        )
        instances.append(result)
        return result

    yield create
    for instance in instances:
        if instance.journal is not None:
            instance.journal.close()


@pytest.mark.asyncio
async def test_journal_is_authoritative_when_cache_unavailable(build):
    orchestrator = build()
    job_id = await orchestrator.submit_training_job(job().config)
    orchestrator.state_service.state_manager.get.side_effect = OSError("offline")
    status = await orchestrator.get_job_status(job_id)
    assert status["status"] == "queued"
    assert (await orchestrator.get_system_status())["durable_journal_enabled"]
    await orchestrator.shutdown()
    assert (await orchestrator.get_job_status(job_id))["status"] == "cancelled"
    await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_durable_terminal_write_survives_failed_cache_update(build):
    original = build()
    job_id = await original.submit_training_job(job().config)
    original.state_service.state_manager.set.side_effect = OSError("offline")
    assert await original.cancel_job(job_id)
    assert job_id in original._pending_state_jobs
    assert original.journal.get(job_id)["status"] == "cancelled"
    original.journal.close()  # Simulate losing the process after the disk commit.
    recovered = build()
    assert not recovered.scheduler.job_queue
    assert (
        recovered.scheduler.completed_jobs[job_id].status
        is module.TrainingStatus.CANCELLED
    )
    await recovered.shutdown()
    recovered.training_runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_cache_outage_does_not_reject_durable_admission_or_shutdown(build):
    orchestrator = build()
    orchestrator.state_service.state_manager.set.side_effect = OSError("offline")
    job_id = await orchestrator.submit_training_job(job().config)
    assert orchestrator.journal.get(job_id)["status"] == "queued"
    assert job_id in orchestrator._pending_state_jobs
    await orchestrator.shutdown()
    assert orchestrator.journal.get(job_id)["status"] == "cancelled"
    await orchestrator.shutdown()
    recovered = build()
    assert not recovered.scheduler.job_queue
    await recovered.shutdown()


@pytest.mark.asyncio
async def test_failed_journal_commit_prevents_runner_start(build, monkeypatch):
    orchestrator = build()
    job_id = await orchestrator.submit_training_job(job().config)
    selected = await orchestrator.scheduler.get_next_job(orchestrator.resource_manager)

    def fail(snapshot):
        raise OSError("disk full")

    monkeypatch.setattr(orchestrator.journal, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        await orchestrator._start_job(selected)
    orchestrator.training_runner.assert_not_awaited()
    assert not orchestrator.resource_manager.allocated_resources
    assert job_id in orchestrator._pending_state_jobs


@pytest.mark.asyncio
async def test_recovered_queue_runs_once_after_durable_running_record(build, tmp_path):
    original = build()
    job_id = await original.submit_training_job(job().config)
    original.journal.close()
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"test artifact")
    recovered = build()

    async def runner(selected):
        assert recovered.journal.get(job_id)["status"] == "running"
        return module.TrainingRunResult(artifact, {"loss": 0.5}, 1, 1)

    recovered.training_runner = AsyncMock(side_effect=runner)
    selected = await recovered.scheduler.get_next_job(recovered.resource_manager)
    assert selected.job_id == job_id
    await recovered._start_job(selected)
    await recovered.worker_tasks[f"worker_{job_id}"]
    await recovered._cleanup_completed_tasks()
    assert recovered.journal.get(job_id)["status"] == "completed"
    recovered.training_runner.assert_awaited_once()
    await recovered.shutdown()
    again = build()
    assert not again.scheduler.job_queue
    assert (
        again.scheduler.completed_jobs[job_id].status is module.TrainingStatus.COMPLETED
    )
    await again.shutdown()


def test_recovered_queue_requires_explicit_runner(build):
    original = build()
    original.journal.save(serialize_training_job(job()))
    original.journal.close()
    with pytest.raises(RuntimeError, match="No training runner"):
        build(training_runner=None)
    recovered = build()  # Failed construction releases the ownership lock.
    assert len(recovered.scheduler.job_queue) == 1


def test_only_one_orchestrator_can_own_journal(build):
    first = build()
    with pytest.raises(ValueError, match="Another driver"):
        build()
    first.journal.close()
    build()


def test_interrupted_atomic_replace_preserves_previous_record(build, monkeypatch):
    from stateset_agents.remote import river_rl

    orchestrator = build()
    record = job()
    orchestrator.journal.save(serialize_training_job(record))
    record.status = module.TrainingStatus.RUNNING

    def fail_replace(*args):
        raise OSError("interrupted replacement")

    with monkeypatch.context() as patcher:
        patcher.setattr(river_rl.os, "replace", fail_replace)
        with pytest.raises(OSError, match="interrupted replacement"):
            orchestrator.journal.save(serialize_training_job(record))
    assert orchestrator.journal.get(record.job_id)["status"] == "queued"
    assert len(orchestrator.journal.load()) == 1
    orchestrator.journal.save(serialize_training_job(record))
    assert orchestrator.journal.get(record.job_id)["status"] == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        module.TrainingStatus.RUNNING,
        module.TrainingStatus.PAUSED,
        module.TrainingStatus.CANCELLED,
    ],
)
async def test_unfinished_execution_is_interrupted_and_never_requeued(build, status):
    original = build()
    record = job(status=status)
    record.started_at = record.created_at
    original.journal.save(serialize_training_job(record))
    original.journal.close()
    recovered = build()
    assert not recovered.scheduler.job_queue
    interrupted = recovered.scheduler.completed_jobs[record.job_id]
    assert interrupted.status is module.TrainingStatus.INTERRUPTED
    assert "External training may still be active" in interrupted.last_error
    assert recovered.journal.get(record.job_id)["status"] == "interrupted"
    await recovered.shutdown()
    recovered.training_runner.assert_not_awaited()


@pytest.mark.parametrize(
    "damage", ["schema", "identity", "missing", "nonfinite", "timestamp", "duplicate"]
)
def test_invalid_records_fail_before_recovery_or_execution(build, damage):
    original = build()
    original.journal.save(serialize_training_job(job()))
    path = next(original.journal.directory.glob("*.json"))
    record = json.loads(path.read_text())
    if damage == "schema":
        record["schema_version"] = True
    elif damage == "identity":
        record["job"]["job_id"] = "different"
    elif damage == "missing":
        del record["job"]["metrics"]
    elif damage == "nonfinite":
        record["job"]["created_at"] = float("inf")
    elif damage == "timestamp":
        record["job"]["created_at"] = "yesterday"
    path.write_text(json.dumps(record))
    if damage == "duplicate":
        path.write_text(
            path.read_text().replace(
                '"schema_version": 2', '"schema_version": 2, "schema_version": 2'
            )
        )
    original.journal.close()
    with pytest.raises(ValueError):
        build()
    # Even a malformed journal must not retain an ownership lock on failure.
    lock = JobJournal(path.parent)
    lock.close()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="select on child pipes is POSIX")
async def test_process_kill_releases_lock_and_preserves_committed_jobs(build, tmp_path):
    script = """
import sys, time
from stateset_agents.training.job_journal import JobJournal
from stateset_agents.training.advanced_training_models import TrainingJob, TrainingJobSpec, TrainingStatus, serialize_training_job
journal = JobJournal(sys.argv[1])
for name, status in [('queued', TrainingStatus.QUEUED), ('started', TrainingStatus.RUNNING)]:
    job = TrainingJob(name, TrainingJobSpec('test', 'test', {}, 'unused'), status=status)
    if status == TrainingStatus.RUNNING:
        job.started_at = job.created_at
    journal.save(serialize_training_job(job))
print('committed', flush=True)
time.sleep(60)
"""
    with subprocess.Popen(
        [sys.executable, "-u", "-c", script, str(tmp_path / "journal")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            assert select.select([child.stdout], [], [], 30)[0], "child did not commit"
            assert child.stdout.readline().strip() == "committed"
            with pytest.raises(ValueError, match="Another driver"):
                build()
        finally:
            child.kill()
            child.communicate(timeout=10)
    recovered = build()
    assert [item.job_id for item in recovered.scheduler.job_queue] == ["queued"]
    assert (
        recovered.scheduler.completed_jobs["started"].status
        is module.TrainingStatus.INTERRUPTED
    )
    await recovered.shutdown()
    recovered.training_runner.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_concurrent_idempotent_submissions_create_one_job(
    build, tmp_path, durable
):
    orchestrator = build(journal_dir=tmp_path / "journal" if durable else None)
    ids = await asyncio.gather(
        *(
            orchestrator.submit_training_job(job().config, idempotency_key="request")
            for _ in range(8)
        )
    )
    assert len(set(ids)) == 1
    assert len(orchestrator.scheduler.job_queue) == 1
    orchestrator.state_service.state_manager.set.assert_awaited_once()
    orchestrator.training_runner.assert_not_awaited()
    await orchestrator.shutdown()
    # A replay is a lookup even when admission has closed.
    assert (
        await orchestrator.submit_training_job(job().config, idempotency_key="request")
        == ids[0]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["config", "priority"])
async def test_key_cannot_be_reused_for_different_work(build, change):
    orchestrator = build()
    config = job().config
    job_id = await orchestrator.submit_training_job(config, idempotency_key="request")
    priority = 1
    if change == "config":
        config.model_config["temperature"] = 0.25
    else:
        priority = 2
    with pytest.raises(ValueError, match="different request"):
        await orchestrator.submit_training_job(
            config, priority, idempotency_key="request"
        )
    assert [item.job_id for item in orchestrator.scheduler.job_queue] == [job_id]
    assert not orchestrator.scheduler.job_queue[0].config.model_config
    orchestrator.state_service.state_manager.set.assert_awaited_once()
    await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_key_is_scoped_to_user_and_plain_submissions_remain_distinct(build):
    orchestrator = build()
    first = await orchestrator.submit_training_job(
        job().config, user_id="one", idempotency_key="key"
    )
    second = await orchestrator.submit_training_job(
        job().config, user_id="two", idempotency_key="key"
    )
    third = await orchestrator.submit_training_job(job().config)
    fourth = await orchestrator.submit_training_job(job().config)
    assert len({first, second, third, fourth}) == 4
    assert (
        await orchestrator.submit_training_job(
            job().config, user_id="one", idempotency_key="key"
        )
        == first
    )
    await orchestrator.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["", " ", 1, True, "a" * 257])
async def test_invalid_key_has_no_persistence_or_admission_side_effects(build, key):
    orchestrator = build()
    with pytest.raises(ValueError, match="idempotency_key"):
        await orchestrator.submit_training_job(job().config, idempotency_key=key)
    orchestrator.state_service.state_manager.set.assert_not_awaited()
    assert not orchestrator.scheduler.job_queue
    assert not list(orchestrator.journal.directory.glob("*.json"))
    await orchestrator.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value", [{1: "nonstring key"}, {"x": float("nan")}, {"x": {1, 2}}]
)
async def test_ambiguous_or_non_json_keyed_config_is_rejected(build, value):
    orchestrator = build()
    config = job().config
    config.model_config = value
    with pytest.raises(ValueError, match="JSON"):
        await orchestrator.submit_training_job(config, idempotency_key="request")
    assert not orchestrator.scheduler.job_queue
    orchestrator.state_service.state_manager.set.assert_not_awaited()
    await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_submission_snapshots_nested_config_before_waiting_for_admission(build):
    orchestrator = build()
    config = job().config
    config.model_config = {"nested": {"value": 1}}
    await orchestrator._admission_lock.acquire()
    task = asyncio.create_task(
        orchestrator.submit_training_job(config, idempotency_key="request")
    )
    try:
        await asyncio.sleep(0)
        config.model_config["nested"]["value"] = 2
    finally:
        orchestrator._admission_lock.release()
    job_id = await task
    saved = orchestrator.journal.get(job_id)
    assert saved["config"]["model_config"] == {"nested": {"value": 1}}
    assert orchestrator.scheduler.job_queue[0].config.model_config == {
        "nested": {"value": 1}
    }
    await orchestrator.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_idempotent_replay_after_restart_returns_original_attempt(build, started):
    original = build()
    job_id = await original.submit_training_job(
        job().config, idempotency_key="request-secret"
    )
    selected = original.scheduler.job_queue[0]
    if started:
        selected.status = module.TrainingStatus.RUNNING
        selected.started_at = selected.created_at
        await original._persist_job(selected)
    assert (
        "request-secret"
        not in next(original.journal.directory.glob("*.json")).read_text()
    )
    original.journal.close()
    recovered = build()
    retry = await recovered.submit_training_job(
        job().config, idempotency_key="request-secret"
    )
    assert retry == job_id
    assert len(recovered.scheduler.job_queue) == (not started)
    if started:
        assert (
            recovered.scheduler.completed_jobs[job_id].status
            is module.TrainingStatus.INTERRUPTED
        )
    recovered.training_runner.assert_not_awaited()
    await recovered.shutdown()


@pytest.mark.asyncio
async def test_terminal_replay_needs_no_runner_and_new_key_starts_new_attempt(build):
    original = build()
    job_id = await original.submit_training_job(job().config, idempotency_key="request")
    await original.shutdown()
    recovered = build(training_runner=None)
    assert (
        await recovered.submit_training_job(job().config, idempotency_key="request")
        == job_id
    )
    with pytest.raises(RuntimeError, match="No training runner"):
        await recovered.submit_training_job(job().config, idempotency_key="new")
    recovered.training_runner = AsyncMock()
    assert (
        await recovered.submit_training_job(job().config, idempotency_key="new")
        != job_id
    )
    assert len(recovered.scheduler.job_queue) == 1
    await recovered.shutdown()


def test_legacy_journal_records_migrate_without_dropping_jobs(build):
    original = build()
    original.journal.save(serialize_training_job(job()))
    path = next(original.journal.directory.glob("*.json"))
    record = json.loads(path.read_text())
    record["schema_version"] = 1
    del record["job"]["submission_fingerprint"]
    path.write_text(json.dumps(record))
    original.journal.close()
    recovered = build()
    assert len(recovered.scheduler.job_queue) == 1
    assert recovered.scheduler.job_queue[0].submission_fingerprint is None


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "null", "invalid"])
async def test_new_journal_cannot_lose_idempotency_fingerprint(build, damage):
    original = build()
    await original.submit_training_job(job().config, idempotency_key="request")
    path = next(original.journal.directory.glob("*.json"))
    record = json.loads(path.read_text())
    if damage == "missing":
        del record["job"]["submission_fingerprint"]
    else:
        record["job"]["submission_fingerprint"] = (
            None if damage == "null" else "invalid"
        )
    path.write_text(json.dumps(record))
    original.journal.close()
    with pytest.raises(ValueError):
        build()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="select on child pipes is POSIX")
async def test_retry_after_process_dies_before_submission_response(build, tmp_path):
    script = """
import asyncio, sys
from types import SimpleNamespace
from stateset_agents.training import advanced_training_orchestrator as module
async def set_state(*args):
    print('committed-before-response', flush=True)
    await asyncio.Event().wait()
async def runner(job):
    raise AssertionError('runner must not execute')
module.get_state_service = lambda: SimpleNamespace(state_manager=SimpleNamespace(set=set_state))
module.get_monitoring_service = lambda: SimpleNamespace()
module.ResourceManager._detect_resources = lambda self: None
async def main():
    orchestrator = module.AdvancedTrainingOrchestrator(
        journal_dir=sys.argv[1], training_runner=runner,
        start_background_tasks=False, enable_experiment_tracking=False)
    config = module.TrainingJobSpec('test', 'test', {}, 'unused')
    await orchestrator.submit_training_job(config, idempotency_key='lost-response')
    raise AssertionError('submission unexpectedly returned')
asyncio.run(main())
"""
    with subprocess.Popen(
        [sys.executable, "-u", "-c", script, str(tmp_path / "journal")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        try:
            assert select.select([child.stdout], [], [], 30)[0], "child did not commit"
            assert child.stdout.readline().strip() == "committed-before-response"
        finally:
            child.kill()
            child.communicate(timeout=10)
    recovered = build()
    assert len(recovered.scheduler.job_queue) == 1
    original = recovered.scheduler.job_queue[0].job_id
    assert (
        await recovered.submit_training_job(
            job().config, idempotency_key="lost-response"
        )
        == original
    )
    assert len(recovered.scheduler.job_queue) == 1
    assert len(list(recovered.journal.directory.glob("*.json"))) == 1
    recovered.training_runner.assert_not_awaited()
    await recovered.shutdown()


@pytest.mark.asyncio
async def test_request_digest_ignores_mapping_order_and_survives_runner_config_changes(
    build,
):
    orchestrator = build()
    config = job().config
    config.model_config = {"a": 1, "b": {"x": 2, "y": 3}}
    job_id = await orchestrator.submit_training_job(config, idempotency_key="request")
    orchestrator.scheduler.job_queue[0].config.model_config["a"] = 99
    config.model_config = {"b": {"y": 3, "x": 2}, "a": 1}
    assert (
        await orchestrator.submit_training_job(config, idempotency_key="request")
        == job_id
    )
    assert len(orchestrator.scheduler.job_queue) == 1
    await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_state_write_snapshot_does_not_share_nested_configuration(build):
    orchestrator = build()
    selected = job()
    selected.config.model_config = {"nested": [1]}
    orchestrator._pending_state_jobs[selected.job_id] = selected
    entered, release = asyncio.Event(), asyncio.Event()
    snapshots = []

    async def write(key, snapshot):
        snapshots.append(snapshot)
        entered.set()
        await release.wait()

    orchestrator.state_service.state_manager.set.side_effect = write
    task = asyncio.create_task(orchestrator._persist_job(selected))
    await asyncio.wait_for(entered.wait(), 2)
    selected.config.model_config["nested"].append(2)
    try:
        assert snapshots[0]["config"]["model_config"] == {"nested": [1]}
    finally:
        release.set()
        await task
    assert selected.job_id in orchestrator._pending_state_jobs
    await orchestrator._cleanup_completed_tasks()
    assert orchestrator.journal.get(selected.job_id)["config"]["model_config"] == {
        "nested": [1, 2]
    }
    assert not orchestrator._pending_state_jobs
    await orchestrator.shutdown()
