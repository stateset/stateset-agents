"""Exercise cancellation and failure at scheduler ownership boundaries."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from stateset_agents.training import advanced_training_orchestrator as module


@pytest.fixture
def orchestrator(monkeypatch):
    state = SimpleNamespace(set=AsyncMock(), get=AsyncMock(return_value=None))
    monkeypatch.setattr(
        module, "get_state_service", lambda: SimpleNamespace(state_manager=state)
    )
    monkeypatch.setattr(module, "get_monitoring_service", MagicMock())
    monkeypatch.setattr(module.ResourceManager, "_detect_resources", lambda self: None)
    manager = module.ResourceManager(capacity_overrides={module.ResourceType.CPU: 1})
    return module.AdvancedTrainingOrchestrator(
        resource_manager=manager,
        training_runner=AsyncMock(),
        enable_experiment_tracking=False,
        start_background_tasks=False,
    )


def make_job(name="job"):
    return module.TrainingJob(
        name,
        module.TrainingJobSpec(
            experiment_name="lifecycle",
            agent_type="test",
            model_config={},
            training_data="unused",
            resource_requirements=[
                module.ResourceRequirement(module.ResourceType.CPU, 1)
            ],
        ),
    )


async def claim(orchestrator, job):
    await orchestrator.scheduler.submit_job(job)
    assert (
        await orchestrator.scheduler.get_next_job(orchestrator.resource_manager) is job
    )


def assert_released(orchestrator, job, status):
    assert not orchestrator.resource_manager.allocated_resources
    assert not orchestrator.workers
    assert not orchestrator.worker_tasks
    assert not orchestrator.worker_jobs
    assert job.job_id not in orchestrator.scheduler.running_jobs
    assert orchestrator.scheduler.completed_jobs[job.job_id] is job
    assert job.status is status
    assert job.completed_at is not None


@pytest.mark.asyncio
async def test_claim_reserves_before_removing_job_and_keeps_blocked_job(orchestrator):
    first, second = make_job("first"), make_job("second")
    await claim(orchestrator, first)
    await orchestrator.scheduler.submit_job(second)
    assert (
        await orchestrator.scheduler.get_next_job(orchestrator.resource_manager) is None
    )
    assert orchestrator.scheduler.job_queue == [second]
    assert orchestrator.scheduler.running_jobs == {"first": first}
    assert orchestrator.resource_manager.allocated_resources == {
        "first": {module.ResourceType.CPU: 1}
    }


@pytest.mark.asyncio
async def test_cancellation_while_claim_waits_preserves_queue(orchestrator):
    job = make_job()
    await orchestrator.scheduler.submit_job(job)
    lock = orchestrator.resource_manager._allocation_lock
    await lock.acquire()
    task = asyncio.create_task(
        orchestrator.scheduler.get_next_job(orchestrator.resource_manager)
    )
    try:
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        lock.release()
    assert orchestrator.scheduler.job_queue == [job]
    assert not orchestrator.scheduler.running_jobs
    assert not orchestrator.resource_manager.allocated_resources


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_start", [False, True])
async def test_interrupted_startup_releases_reservation(orchestrator, cancel_start):
    job = make_job()
    await claim(orchestrator, job)
    entered = asyncio.Event()

    async def blocked_write(*args):
        entered.set()
        await asyncio.Event().wait()

    state = orchestrator.state_service.state_manager
    if cancel_start:
        state.set.side_effect = blocked_write
        task = asyncio.create_task(orchestrator._start_job(job))
        await asyncio.wait_for(entered.wait(), 2)
        state.set.side_effect = None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        status = module.TrainingStatus.CANCELLED
    else:
        state.set.side_effect = OSError("state store unavailable")
        with pytest.raises(OSError, match="state store unavailable"):
            await orchestrator._start_job(job)
        status = module.TrainingStatus.FAILED
    orchestrator.training_runner.assert_not_awaited()
    assert_released(orchestrator, job, status)


@pytest.mark.asyncio
async def test_cancel_during_startup_never_launches_runner(orchestrator):
    job = make_job()
    await claim(orchestrator, job)
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked_write(*args):
        entered.set()
        await release.wait()

    state = orchestrator.state_service.state_manager
    state.set.side_effect = blocked_write
    task = asyncio.create_task(orchestrator._start_job(job))
    await asyncio.wait_for(entered.wait(), 2)
    cancellation = asyncio.create_task(orchestrator.cancel_job(job.job_id))
    await asyncio.sleep(0)
    assert job.status is module.TrainingStatus.CANCELLED
    release.set()
    assert await asyncio.wait_for(cancellation, 2)
    await asyncio.wait_for(task, 2)
    assert_released(orchestrator, job, module.TrainingStatus.CANCELLED)
    orchestrator.training_runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_before_worker_first_instruction_is_reaped(orchestrator):
    job = make_job()
    await claim(orchestrator, job)
    await orchestrator._start_job(job)
    task = orchestrator.worker_tasks["worker_job"]
    assert await orchestrator.cancel_job(job.job_id)
    await asyncio.gather(task, return_exceptions=True)
    await orchestrator._cleanup_completed_tasks()
    orchestrator.training_runner.assert_not_awaited()
    assert_released(orchestrator, job, module.TrainingStatus.CANCELLED)


@pytest.mark.asyncio
async def test_cancel_keeps_capacity_until_runner_cleanup_finishes(orchestrator):
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def runner(job):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    orchestrator.training_runner = runner
    job = make_job()
    await claim(orchestrator, job)
    await orchestrator._start_job(job)
    task = orchestrator.worker_tasks["worker_job"]
    await asyncio.wait_for(entered.wait(), 2)
    try:
        assert await orchestrator.cancel_job(job.job_id)
        await asyncio.wait_for(cleaning.wait(), 2)
        assert await orchestrator.cancel_job(job.job_id)
        await orchestrator._cleanup_completed_tasks()
        assert not task.done()
        assert job.job_id in orchestrator.resource_manager.allocated_resources
        assert not await orchestrator.resource_manager.can_allocate(
            job.config.resource_requirements
        )
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    await orchestrator._cleanup_completed_tasks()
    assert_released(orchestrator, job, module.TrainingStatus.CANCELLED)


@pytest.mark.asyncio
async def test_cancel_stops_worker_even_when_state_store_fails(orchestrator):
    job = make_job()
    await claim(orchestrator, job)
    await orchestrator._start_job(job)
    task = orchestrator.worker_tasks["worker_job"]
    orchestrator.state_service.state_manager.set.side_effect = OSError("offline")
    with pytest.raises(OSError, match="offline"):
        await orchestrator.cancel_job(job.job_id)
    await asyncio.gather(task, return_exceptions=True)
    orchestrator.state_service.state_manager.set.side_effect = None
    await orchestrator._cleanup_completed_tasks()
    assert_released(orchestrator, job, module.TrainingStatus.CANCELLED)


@pytest.mark.asyncio
async def test_shutdown_cancels_and_reaps_running_worker(orchestrator):
    entered, exited = asyncio.Event(), asyncio.Event()

    async def runner(job):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()

    orchestrator.training_runner = runner
    job = make_job()
    await claim(orchestrator, job)
    await orchestrator._start_job(job)
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.wait_for(orchestrator.shutdown(), 2)
    assert exited.is_set()
    assert_released(orchestrator, job, module.TrainingStatus.CANCELLED)
    with pytest.raises(RuntimeError, match="shutting down"):
        await orchestrator.submit_training_job(make_job().config)
    await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_failed_completion_write_can_be_retried_without_resource_leak(
    orchestrator,
):
    job = make_job()
    await claim(orchestrator, job)
    await orchestrator._start_job(job)
    task = orchestrator.worker_tasks["worker_job"]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    state = orchestrator.state_service.state_manager
    state.set.side_effect = OSError("offline")
    with pytest.raises(OSError, match="offline"):
        await orchestrator._cleanup_completed_tasks()
    assert not orchestrator.resource_manager.allocated_resources
    state.set.side_effect = None
    await orchestrator._cleanup_completed_tasks()
    assert_released(orchestrator, job, module.TrainingStatus.CANCELLED)
    assert state.set.call_args.args[1]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_shutdown_releases_every_worker_when_completion_writes_fail(orchestrator):
    orchestrator.resource_manager.available_resources[module.ResourceType.CPU] = 2
    jobs = [make_job("first"), make_job("second")]
    for job in jobs:
        await claim(orchestrator, job)
        await orchestrator._start_job(job)
    state = orchestrator.state_service.state_manager
    state.set.side_effect = OSError("offline")
    with pytest.raises(OSError, match="offline"):
        await asyncio.wait_for(orchestrator.shutdown(), 2)
    assert not orchestrator.resource_manager.allocated_resources
    assert all(task.done() for task in orchestrator.worker_tasks.values())
    assert all(job.status is module.TrainingStatus.CANCELLED for job in jobs)
    state.set.side_effect = None
    await orchestrator.shutdown()
    assert not orchestrator.worker_tasks


@pytest.mark.asyncio
async def test_shutdown_interrupts_startup_without_launching_runner(orchestrator):
    job = make_job()
    await orchestrator.scheduler.submit_job(job)
    entered = asyncio.Event()
    calls = 0

    async def first_write_blocks(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await asyncio.Event().wait()

    orchestrator.state_service.state_manager.set.side_effect = first_write_blocks
    orchestrator._orchestration_task = asyncio.create_task(
        orchestrator._orchestration_loop()
    )
    await asyncio.wait_for(entered.wait(), 2)
    await asyncio.wait_for(orchestrator.shutdown(), 2)
    assert orchestrator._orchestration_task.done()
    orchestrator.training_runner.assert_not_awaited()
    assert_released(orchestrator, job, module.TrainingStatus.CANCELLED)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [module.TrainingStatus.COMPLETED, module.TrainingStatus.FAILED]
)
async def test_cancellation_cannot_overwrite_terminal_result(orchestrator, status):
    job = make_job()
    job.status = status
    orchestrator.scheduler.running_jobs[job.job_id] = job
    assert not await orchestrator.cancel_job(job.job_id)
    assert job.status is status


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["queued", "running", "completed"])
async def test_scheduler_rejects_duplicate_identity_without_mutating_jobs(
    orchestrator, location
):
    original, duplicate = make_job(), make_job()
    scheduler = orchestrator.scheduler
    if location == "queued":
        await scheduler.submit_job(original)
    elif location == "running":
        await claim(orchestrator, original)
    else:
        original.status = module.TrainingStatus.COMPLETED
        scheduler.completed_jobs[original.job_id] = original
    original_status = original.status
    with pytest.raises(ValueError, match="already been submitted"):
        await scheduler.submit_job(duplicate)
    assert duplicate.status is module.TrainingStatus.PENDING
    assert original.status is original_status
    assert len(scheduler.job_queue) == (location == "queued")


@pytest.mark.asyncio
async def test_concurrent_duplicate_submissions_have_one_owner(orchestrator):
    jobs = [make_job(), make_job()]
    results = await asyncio.gather(
        *(orchestrator.scheduler.submit_job(job) for job in jobs),
        return_exceptions=True,
    )
    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert len(orchestrator.scheduler.job_queue) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", ["", "  ", None])
async def test_invalid_identity_rejected_before_queue_mutation(orchestrator, identity):
    with pytest.raises(ValueError, match="nonempty"):
        await orchestrator.scheduler.submit_job(make_job(identity))
    assert not orchestrator.scheduler.job_queue


@pytest.mark.asyncio
async def test_submission_persists_queued_status(orchestrator):
    job_id = await orchestrator.submit_training_job(make_job().config)
    saved = orchestrator.state_service.state_manager.set.call_args.args[1]
    assert saved["job_id"] == job_id
    assert saved["status"] == "queued"


@pytest.mark.asyncio
async def test_shutdown_cancels_and_persists_every_queued_job(orchestrator):
    ids = [await orchestrator.submit_training_job(make_job().config) for _ in range(2)]
    await orchestrator.shutdown()
    assert not orchestrator.scheduler.job_queue
    for job_id in ids:
        job = orchestrator.scheduler.completed_jobs[job_id]
        assert job.status is module.TrainingStatus.CANCELLED
        assert job.completed_at is not None
    saved = {
        call.args[0]: call.args[1]
        for call in orchestrator.state_service.state_manager.set.call_args_list
    }
    assert all(value["status"] == "cancelled" for value in saved.values())
    assert not orchestrator._pending_state_jobs
    with pytest.raises(RuntimeError, match="closed"):
        await orchestrator.scheduler.submit_job(make_job())
    assert (
        await orchestrator.scheduler.get_next_job(orchestrator.resource_manager) is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", ["shutdown", "cancel"])
async def test_interrupted_submission_retains_terminal_owner(orchestrator, interrupt):
    entered, release = asyncio.Event(), asyncio.Event()
    snapshots = []

    async def write(key, snapshot):
        if not entered.is_set():
            entered.set()
            await release.wait()
        snapshots.append(snapshot)

    orchestrator.state_service.state_manager.set.side_effect = write
    submission = asyncio.create_task(
        orchestrator.submit_training_job(make_job().config)
    )
    await asyncio.wait_for(entered.wait(), 2)
    if interrupt == "shutdown":
        shutdown = asyncio.create_task(orchestrator.shutdown())
        await asyncio.sleep(0)
        assert orchestrator._closing
        assert not shutdown.done()
        release.set()
        with pytest.raises(RuntimeError, match="shutting down"):
            await asyncio.wait_for(submission, 2)
        await asyncio.wait_for(shutdown, 2)
    else:
        submission.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(submission, 2)
    assert not orchestrator.scheduler.job_queue
    assert not orchestrator.scheduler.running_jobs
    assert len(orchestrator.scheduler.completed_jobs) == 1
    job = next(iter(orchestrator.scheduler.completed_jobs.values()))
    assert job.status is module.TrainingStatus.CANCELLED
    assert snapshots[-1]["status"] == "cancelled"
    assert snapshots[-1]["completed_at"] is not None
    orchestrator.training_runner.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["submission", "queued_cancel", "shutdown", "startup"]
)
async def test_failed_terminal_writes_are_retained_for_retry(orchestrator, failure):
    state = orchestrator.state_service.state_manager
    if failure == "submission":
        state.set.side_effect = OSError("offline")
        with pytest.raises(OSError, match="offline"):
            await orchestrator.submit_training_job(make_job().config)
    elif failure == "startup":
        job = make_job()
        await claim(orchestrator, job)
        state.set.side_effect = OSError("offline")
        with pytest.raises(OSError, match="offline"):
            await orchestrator._start_job(job)
    else:
        job_id = await orchestrator.submit_training_job(make_job().config)
        state.set.side_effect = OSError("offline")
        with pytest.raises(OSError, match="offline"):
            if failure == "queued_cancel":
                await orchestrator.cancel_job(job_id)
            else:
                await orchestrator.shutdown()
    assert len(orchestrator._pending_state_jobs) == 1
    assert not orchestrator.scheduler.job_queue
    assert not orchestrator.resource_manager.allocated_resources
    state.set.side_effect = None
    await orchestrator.shutdown()
    assert not orchestrator._pending_state_jobs
    expected = "failed" if failure == "startup" else "cancelled"
    assert state.set.call_args.args[1]["status"] == expected
    assert state.set.call_args.args[1]["completed_at"] is not None


@pytest.mark.asyncio
async def test_old_write_does_not_clear_new_terminal_retry(orchestrator):
    job = make_job()
    job.status = module.TrainingStatus.QUEUED
    entered, release = asyncio.Event(), asyncio.Event()
    persisted = []

    async def write(key, snapshot):
        entered.set()
        await release.wait()
        persisted.append(snapshot)

    orchestrator.state_service.state_manager.set.side_effect = write
    old_write = asyncio.create_task(orchestrator._persist_job(job))
    await asyncio.wait_for(entered.wait(), 2)
    job.status = module.TrainingStatus.CANCELLED
    job.completed_at = 1.0
    orchestrator._pending_state_jobs[job.job_id] = job
    release.set()
    await old_write
    assert job.job_id in orchestrator._pending_state_jobs
    assert persisted[-1]["status"] == "queued"
    await orchestrator._cleanup_completed_tasks()
    assert persisted[-1]["status"] == "cancelled"
    assert not orchestrator._pending_state_jobs


@pytest.mark.asyncio
async def test_concurrent_shutdown_calls_do_not_interrupt_runner_cleanup(orchestrator):
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def runner(job):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    orchestrator.training_runner = runner
    job = make_job()
    await claim(orchestrator, job)
    await orchestrator._start_job(job)
    await asyncio.wait_for(entered.wait(), 2)
    first = asyncio.create_task(orchestrator.shutdown())
    await asyncio.wait_for(cleaning.wait(), 2)
    second = asyncio.create_task(orchestrator.shutdown())
    try:
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()
        assert job.job_id in orchestrator.resource_manager.allocated_resources
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 2)
    assert_released(orchestrator, job, module.TrainingStatus.CANCELLED)


@pytest.mark.asyncio
async def test_cancelling_tracker_closure_does_not_erase_completed_training(
    orchestrator, tmp_path
):
    artifact = tmp_path / "weights"
    artifact.write_bytes(b"trained output")
    orchestrator.training_runner = AsyncMock(
        return_value=module.TrainingRunResult(artifact, {"loss": 0.2}, 1, 1)
    )
    closing = asyncio.Event()
    tracker = AsyncMock()

    async def finish(*args, **kwargs):
        closing.set()
        await asyncio.Event().wait()

    tracker.finish_experiment.side_effect = finish
    orchestrator.experiment_tracker = tracker
    job = make_job()
    await claim(orchestrator, job)
    await orchestrator._start_job(job)
    task = orchestrator.worker_tasks["worker_job"]
    await asyncio.wait_for(closing.wait(), 2)
    assert job.status is module.TrainingStatus.COMPLETED
    assert not await orchestrator.cancel_job(job.job_id)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await orchestrator._cleanup_completed_tasks()
    assert_released(orchestrator, job, module.TrainingStatus.COMPLETED)
    assert job.checkpoint_path == str(artifact)
