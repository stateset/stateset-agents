"""Admission fairness uses reserved resources and rotates equal-share owners."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from stateset_agents.training import advanced_training_orchestrator as module
from stateset_agents.training.job_journal import JobJournal


def job(name, owner=None, cpu=1.0, gpu=0.0):
    return module.TrainingJob(
        name,
        module.TrainingJobSpec(
            "fair",
            "test",
            {},
            "unused",
            resource_requirements=[
                module.ResourceRequirement(module.ResourceType.CPU, cpu),
                module.ResourceRequirement(module.ResourceType.GPU, gpu),
            ],
        ),
        user_id=owner,
    )


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(module.ResourceManager, "_detect_resources", lambda self: None)
    return module.ResourceManager(
        capacity_overrides={module.ResourceType.CPU: 4, module.ResourceType.GPU: 2}
    )


@pytest.fixture
def scheduler():
    return module.JobScheduler("fair_share")


@pytest.fixture
def services(monkeypatch):
    state = SimpleNamespace(set=AsyncMock(), get=AsyncMock(return_value=None))
    monkeypatch.setattr(
        module, "get_state_service", lambda: SimpleNamespace(state_manager=state)
    )
    monkeypatch.setattr(module, "get_monitoring_service", MagicMock())
    return state


async def enqueue(scheduler, jobs):
    for item in jobs:
        await scheduler.submit_job(item)


async def reserve(scheduler, manager, item):
    assert await manager.allocate_resources(
        item.job_id, item.config.resource_requirements
    )
    scheduler.running_jobs[item.job_id] = item


@pytest.mark.asyncio
async def test_user_backlog_cannot_take_every_available_cpu(scheduler, manager):
    await enqueue(
        scheduler,
        [job(f"a{i}", "a") for i in range(3)] + [job(f"b{i}", "b") for i in range(2)],
    )
    selected = [await scheduler.get_next_job(manager) for _ in range(4)]
    assert [item.job_id for item in selected] == ["a0", "b0", "a1", "b1"]
    assert await scheduler.get_next_job(manager) is None
    assert [item.job_id for item in scheduler.job_queue] == ["a2"]
    assert manager.get_resource_utilization()[module.ResourceType.CPU] == 1


@pytest.mark.asyncio
async def test_fairness_compares_largest_reserved_fraction_not_job_count(
    scheduler, manager
):
    manager.available_resources[module.ResourceType.CPU] = 8
    gpu_job = job("gpu-active", "gpu-owner", cpu=0, gpu=1)
    await reserve(scheduler, manager, gpu_job)
    for i in range(3):
        await reserve(scheduler, manager, job(f"cpu-active-{i}", "cpu-owner"))
    # GPU owner holds 1/2; CPU owner holds 3/8 despite having more running jobs.
    gpu_job.config.resource_requirements = []  # Configuration is not the ledger.
    await enqueue(
        scheduler,
        [job("gpu-next", "gpu-owner", cpu=0, gpu=1), job("cpu-next", "cpu-owner")],
    )
    assert (await scheduler.get_next_job(manager)).job_id == "cpu-next"


@pytest.mark.asyncio
async def test_equal_shares_rotate_even_for_jobs_without_resource_requests(
    scheduler, manager
):
    await enqueue(
        scheduler,
        [
            job(f"{owner}{i}", owner, cpu=0)
            for owner in ("a", "b", "c")
            for i in range(2)
        ],
    )
    selected = [await scheduler.get_next_job(manager) for _ in range(6)]
    assert [item.job_id for item in selected] == ["a0", "b0", "c0", "a1", "b1", "c1"]


@pytest.mark.asyncio
async def test_new_owner_joins_rotation_without_jumping_existing_waiter(
    scheduler, manager
):
    await enqueue(
        scheduler,
        [
            job("a0", "a", cpu=0),
            job("a1", "a", cpu=0),
            job("b0", "b", cpu=0),
            job("b1", "b", cpu=0),
        ],
    )
    assert (await scheduler.get_next_job(manager)).job_id == "a0"
    assert (await scheduler.get_next_job(manager)).job_id == "b0"
    await scheduler.submit_job(job("new", "new-owner", cpu=0))
    assert (await scheduler.get_next_job(manager)).job_id == "a1"
    assert (await scheduler.get_next_job(manager)).job_id == "b1"
    assert (await scheduler.get_next_job(manager)).job_id == "new"


@pytest.mark.asyncio
async def test_blocked_owner_does_not_prevent_other_admissible_work(scheduler, manager):
    await reserve(scheduler, manager, job("active", "a"))
    await enqueue(scheduler, [job("oversized", "b", gpu=3), job("fits", "a")])
    assert (await scheduler.get_next_job(manager)).job_id == "fits"
    assert [item.job_id for item in scheduler.job_queue] == ["oversized"]
    before = scheduler._dispatch_sequence
    assert await scheduler.get_next_job(manager) is None
    assert scheduler._dispatch_sequence == before
    assert "oversized" not in manager.allocated_resources


@pytest.mark.asyncio
async def test_cancelled_work_counts_until_reservations_are_released(
    scheduler, manager
):
    first, second = job("active-a", "a", cpu=2), job("active-b", "b")
    await reserve(scheduler, manager, first)
    await reserve(scheduler, manager, second)
    assert await scheduler.cancel_job(first.job_id)
    await enqueue(scheduler, [job("next-a", "a", cpu=0.5), job("next-b", "b", cpu=0.5)])
    assert (await scheduler.get_next_job(manager)).job_id == "next-b"
    await manager.deallocate_resources(first.job_id)
    assert (await scheduler.get_next_job(manager)).job_id == "next-a"


@pytest.mark.asyncio
async def test_missing_user_ids_share_one_pool(scheduler, manager):
    await enqueue(
        scheduler,
        [
            job("anonymous-1"),
            job("anonymous-2"),
            job("named-1", "named"),
            job("named-2", "named"),
        ],
    )
    selected = [await scheduler.get_next_job(manager) for _ in range(4)]
    assert [item.user_id for item in selected] == [None, "named", None, "named"]


@pytest.mark.asyncio
async def test_concurrent_dequeues_preserve_fairness_and_capacity(scheduler, manager):
    await enqueue(
        scheduler, [job(f"{owner}{i}", owner) for owner in ("a", "b") for i in range(4)]
    )
    selected = await asyncio.gather(
        *(scheduler.get_next_job(manager) for _ in range(8))
    )
    assert [item.user_id for item in selected if item is not None] == [
        "a",
        "b",
        "a",
        "b",
    ]
    assert sum(item is not None for item in selected) == 4
    assert len(manager.allocated_resources) == 4


@pytest.mark.asyncio
async def test_cancelled_admission_never_advances_the_rotation(scheduler, manager):
    await enqueue(scheduler, [job("a", "a"), job("b", "b")])
    await manager._allocation_lock.acquire()
    pending = asyncio.create_task(scheduler.get_next_job(manager))
    try:
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    finally:
        manager._allocation_lock.release()
    assert scheduler._dispatch_sequence == 0
    assert not manager.allocated_resources
    assert [item.job_id for item in scheduler.job_queue] == ["a", "b"]
    assert (await scheduler.get_next_job(manager)).job_id == "a"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy,expected",
    [
        ("fifo", ["first", "second", "third"]),
        ("resource_aware", ["first", "second", "third"]),
        ("priority", ["second", "third", "first"]),
        ("shortest_job_first", ["third", "second", "first"]),
    ],
)
async def test_existing_ordering_policies_remain_distinct(manager, strategy, expected):
    scheduler = module.JobScheduler(strategy)
    jobs = [job("first"), job("second"), job("third")]
    for item, priority, epochs in zip(jobs, [-1, 2, 1], [3, 2, 1], strict=True):
        item.priority, item.config.num_epochs = priority, epochs
    await enqueue(scheduler, jobs)
    assert [(await scheduler.get_next_job(manager)).job_id for _ in jobs] == expected
    assert scheduler.get_queue_status()["strategy"] == strategy


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_concurrent_jobs": 0},
        {"max_concurrent_jobs": -1},
        {"max_concurrent_jobs": True},
        {"max_concurrent_jobs": 1.5},
        {"scheduling_strategy": "unknown"},
    ],
)
def test_invalid_scheduler_settings_fail_before_service_initialization(
    monkeypatch, kwargs
):
    manager = MagicMock()
    state = MagicMock()
    monkeypatch.setattr(module, "ResourceManager", manager)
    monkeypatch.setattr(module, "get_state_service", state)
    with pytest.raises(ValueError):
        module.AdvancedTrainingOrchestrator(**kwargs)
    manager.assert_not_called()
    state.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("priority", True),
        ("priority", 1.5),
        ("priority", "high"),
        ("num_epochs", 0),
        ("num_epochs", -1),
        ("num_epochs", True),
        ("user_id", []),
        ("user_id", 1),
    ],
)
async def test_invalid_scheduling_values_never_persist_or_enqueue(
    manager, services, scheduler, field, value
):
    item = job("invalid")
    if field == "num_epochs":
        item.config.num_epochs = value
    else:
        setattr(item, field, value)
    with pytest.raises(ValueError):
        await scheduler.submit_job(item)
    orchestrator = module.AdvancedTrainingOrchestrator(
        resource_manager=manager,
        training_runner=AsyncMock(),
        start_background_tasks=False,
        enable_experiment_tracking=False,
    )
    with pytest.raises(ValueError):
        await orchestrator.submit_training_job(item.config, item.priority, item.user_id)
    services.set.assert_not_awaited()
    assert not scheduler.job_queue and not orchestrator.scheduler.job_queue
    assert item.status is module.TrainingStatus.PENDING
    await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_restored_queued_jobs_participate_in_fair_rotation(
    manager, services, tmp_path
):
    journal = JobJournal(tmp_path / "journal")
    for item in [job("a0", "a"), job("a1", "a"), job("b0", "b")]:
        item.status = module.TrainingStatus.QUEUED
        journal.save(module.serialize_training_job(item))
    journal.close()
    artifact = tmp_path / "model"
    artifact.write_bytes(b"test artifact")
    runner = AsyncMock(
        return_value=module.TrainingRunResult(artifact, {"loss": 0.2}, 1, 1)
    )
    orchestrator = module.AdvancedTrainingOrchestrator(
        resource_manager=manager,
        training_runner=runner,
        scheduling_strategy="fair_share",
        journal_dir=tmp_path / "journal",
        start_background_tasks=False,
        enable_experiment_tracking=False,
    )
    try:
        selected = [
            await orchestrator.scheduler.get_next_job(manager) for _ in range(2)
        ]
        assert [item.user_id for item in selected] == ["a", "b"]
        for item in selected:
            await orchestrator._start_job(item)
        await asyncio.gather(*orchestrator.worker_tasks.values())
    finally:
        await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_live_orchestration_starts_different_owners_before_backlog(
    manager, services, tmp_path
):
    started = []
    both_started, release = asyncio.Event(), asyncio.Event()
    artifact = tmp_path / "model"
    artifact.write_bytes(b"test artifact")

    async def runner(item):
        started.append(item.user_id)
        if len(started) == 2:
            both_started.set()
        await release.wait()
        return module.TrainingRunResult(artifact, {"loss": 0.2}, 1, 1)

    orchestrator = module.AdvancedTrainingOrchestrator(
        max_concurrent_jobs=2,
        scheduling_strategy="fair_share",
        resource_manager=manager,
        training_runner=runner,
        enable_experiment_tracking=False,
        start_background_tasks=False,
    )
    for owner in ["a", "a", "a", "b", "b"]:
        await orchestrator.submit_training_job(job("unused").config, user_id=owner)
    orchestrator._orchestration_task = asyncio.create_task(
        orchestrator._orchestration_loop()
    )
    try:
        await asyncio.wait_for(both_started.wait(), 8)
        assert started == ["a", "b"]
        assert len(orchestrator.worker_tasks) == 2
        release.set()
        await asyncio.gather(*orchestrator.worker_tasks.values())
    finally:
        release.set()
        await orchestrator.shutdown()
    assert not manager.allocated_resources


@pytest.mark.parametrize(
    "field,value", [("priority", True), ("num_epochs", 0), ("user_id", [])]
)
def test_invalid_recovered_scheduling_inputs_fail_before_startup(
    manager, services, tmp_path, field, value
):
    directory = tmp_path / "journal"
    journal = JobJournal(directory)
    item = job("invalid")
    item.status = module.TrainingStatus.QUEUED
    if field == "num_epochs":
        item.config.num_epochs = value
    else:
        setattr(item, field, value)
    journal.save(module.serialize_training_job(item))
    journal.close()
    with pytest.raises(ValueError):
        module.AdvancedTrainingOrchestrator(
            resource_manager=manager,
            training_runner=AsyncMock(),
            scheduling_strategy="fair_share",
            journal_dir=directory,
            start_background_tasks=False,
            enable_experiment_tracking=False,
        )
    services.set.assert_not_awaited()
    assert not manager.allocated_resources
    journal = JobJournal(directory)  # Failed recovery releases ownership.
    journal.close()
