"""Deadlines must preserve cleanup and never publish late training success."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from stateset_agents.training import advanced_training_orchestrator as module


def config(limit=0.1):
    return module.TrainingJobSpec(
        "deadline",
        "test",
        {},
        "unused",
        max_runtime=limit,
        resource_requirements=[module.ResourceRequirement(module.ResourceType.CPU, 1)],
    )


@pytest.fixture
def result(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"test artifact")
    return module.TrainingRunResult(artifact, {"loss": 0.5}, 1, 1)


@pytest.fixture
def build(monkeypatch, tmp_path):
    instances = []
    monkeypatch.setattr(module, "get_monitoring_service", MagicMock())
    monkeypatch.setattr(module.ResourceManager, "_detect_resources", lambda self: None)

    def create(runner=None):
        state = SimpleNamespace(set=AsyncMock(), get=AsyncMock(return_value=None))
        monkeypatch.setattr(
            module, "get_state_service", lambda: SimpleNamespace(state_manager=state)
        )
        manager = module.ResourceManager(
            capacity_overrides={module.ResourceType.CPU: 1}
        )
        instance = module.AdvancedTrainingOrchestrator(
            training_runner=runner or AsyncMock(),
            resource_manager=manager,
            journal_dir=tmp_path / "journal",
            start_background_tasks=False,
            enable_experiment_tracking=False,
        )
        instances.append(instance)
        return instance

    yield create
    for instance in instances:
        instance.journal.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limit", [0, -1, True, False, float("nan"), float("inf"), "1", 10**400]
)
async def test_invalid_runtime_never_enters_queue_or_state(build, limit):
    orchestrator = build()
    spec = config(limit)
    with pytest.raises(ValueError, match="max_runtime"):
        await orchestrator.submit_training_job(spec)
    with pytest.raises(ValueError, match="max_runtime"):
        await orchestrator.scheduler.submit_job(module.TrainingJob("bad", spec))
    orchestrator.state_service.state_manager.set.assert_not_awaited()
    assert not orchestrator.scheduler.job_queue
    assert not orchestrator.journal.load()
    await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_direct_worker_validates_deadline_before_tracking_or_running():
    runner, tracker = AsyncMock(), AsyncMock()
    job = module.TrainingJob("bad", config(-1))
    assert not await module.TrainingWorker("test", runner).execute_job(job, tracker)
    assert job.status is module.TrainingStatus.FAILED
    runner.assert_not_awaited()
    tracker.start_experiment.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ["cooperative", "returns_result", "raises_error"])
async def test_timeout_cannot_publish_success_even_when_runner_suppresses_cancel(
    result, cleanup
):
    cleaned = asyncio.Event()

    async def runner(job):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if cleanup == "returns_result":
                return result
            if cleanup == "raises_error":
                raise RuntimeError("cleanup failed") from None
            raise
        finally:
            cleaned.set()

    job = module.TrainingJob("timeout", config())
    tracker = AsyncMock()
    worker = module.TrainingWorker("test", runner)
    assert not await asyncio.wait_for(worker.execute_job(job, tracker), 2)
    assert cleaned.is_set()
    assert job.status is module.TrainingStatus.TIMED_OUT
    assert "max_runtime" in job.last_error
    if cleanup == "raises_error":
        assert "RuntimeError" in job.last_error
    assert job.completed_at is not None
    assert not job.metrics and job.current_step == 0
    assert job.checkpoint_path is None
    assert worker.current_job is None
    tracker.log_metrics.assert_not_awaited()
    tracker.log_artifact.assert_not_awaited()
    tracker.finish_experiment.assert_awaited_once()


@pytest.mark.asyncio
async def test_capacity_is_held_until_timed_out_runner_cleanup_finishes(build):
    cleaning, release = asyncio.Event(), asyncio.Event()

    async def runner(job):
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    orchestrator = build(runner)
    job_id = await orchestrator.submit_training_job(
        config(), idempotency_key="deadline"
    )
    selected = await orchestrator.scheduler.get_next_job(orchestrator.resource_manager)
    await orchestrator._start_job(selected)
    task = orchestrator.worker_tasks[f"worker_{job_id}"]
    try:
        await asyncio.wait_for(cleaning.wait(), 2)
        await orchestrator._cleanup_completed_tasks()
        assert not task.done()
        assert job_id in orchestrator.resource_manager.allocated_resources
        assert not await orchestrator.resource_manager.can_allocate(
            config().resource_requirements
        )
    finally:
        release.set()
        await asyncio.wait_for(task, 2)
    await orchestrator._cleanup_completed_tasks()
    assert not orchestrator.resource_manager.allocated_resources
    assert orchestrator.journal.get(job_id)["status"] == "timed_out"
    assert not await orchestrator.cancel_job(job_id)
    await orchestrator.shutdown()
    recovered = build()
    assert (
        recovered.scheduler.completed_jobs[job_id].status
        is module.TrainingStatus.TIMED_OUT
    )
    assert (
        await recovered.submit_training_job(config(), idempotency_key="deadline")
        == job_id
    )
    assert not recovered.scheduler.job_queue
    recovered.training_runner.assert_not_awaited()
    await recovered.shutdown()


@pytest.mark.asyncio
async def test_repeated_parent_cancellation_does_not_interrupt_runner_cleanup():
    entered, cleaning, release, cleaned = (asyncio.Event() for _ in range(4))

    async def runner(job):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            cleaned.set()

    job = module.TrainingJob("cancel", config(100))
    worker = module.TrainingWorker("test", runner)
    task = asyncio.create_task(worker.execute_job(job, None))
    await asyncio.wait_for(entered.wait(), 2)
    try:
        task.cancel()
        await asyncio.wait_for(cleaning.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not cleaned.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
    assert cleaned.is_set()
    assert job.status is module.TrainingStatus.CANCELLED


@pytest.mark.asyncio
async def test_late_blocking_runner_result_is_rejected(result):
    async def runner(job):
        time.sleep(0.2)  # Emulate a runner that incorrectly blocks the event loop.
        return result

    job = module.TrainingJob("blocking", config(0.1))
    assert not await module.TrainingWorker("test", runner).execute_job(job, None)
    assert job.status is module.TrainingStatus.TIMED_OUT
    assert job.checkpoint_path is None


@pytest.mark.asyncio
async def test_timely_result_completes_normally(result):
    job = module.TrainingJob("fast", config(10))
    runner = AsyncMock(return_value=result)
    assert await module.TrainingWorker("test", runner).execute_job(job, None)
    assert job.status is module.TrainingStatus.COMPLETED
    assert job.current_step == 1
    runner.assert_awaited_once()


@pytest.mark.asyncio
async def test_unrelated_event_loop_stall_does_not_reject_timely_result(result):
    async def runner(job):
        asyncio.get_running_loop().call_soon(time.sleep, 0.2)
        return result

    job = module.TrainingJob("timely", config(0.1))
    assert await module.TrainingWorker("test", runner).execute_job(job, None)
    assert job.status is module.TrainingStatus.COMPLETED


@pytest.mark.asyncio
async def test_provider_timeout_is_distinct_and_never_retried():
    job = module.TrainingJob("provider", config(10))
    runner = AsyncMock(side_effect=TimeoutError("optimizer outcome uncertain"))
    assert not await module.TrainingWorker("test", runner).execute_job(job, None)
    assert job.status is module.TrainingStatus.FAILED
    assert job.last_error == "optimizer outcome uncertain"
    runner.assert_awaited_once()


def test_invalid_recovered_deadline_fails_before_work_is_scheduled(build):
    original = build()
    bad = module.TrainingJob("bad", config(True), status=module.TrainingStatus.QUEUED)
    original.journal.save(module.serialize_training_job(bad))
    original.journal.close()
    with pytest.raises(ValueError, match="max_runtime"):
        build()
