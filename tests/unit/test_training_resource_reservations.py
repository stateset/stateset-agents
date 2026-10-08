"""Contended admissions must conserve the training scheduler's resource ledger."""

import asyncio

import pytest

from stateset_agents.training.advanced_training_models import (
    ResourceRequirement,
    ResourceType,
)
from stateset_agents.training.advanced_training_orchestrator import ResourceManager


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr(ResourceManager, "_detect_resources", lambda self: None)
    result = ResourceManager()
    result.available_resources = {ResourceType.CPU: 1.0, ResourceType.GPU: 2.0}
    return result


@pytest.mark.asyncio
async def test_contended_multi_resource_admission_is_atomic(manager):
    requirements = [
        ResourceRequirement(ResourceType.CPU, 1),
        ResourceRequirement(ResourceType.GPU, 1),
    ]
    gpu = manager.resource_locks[ResourceType.GPU]
    await gpu.acquire()
    started = 0
    both_started = asyncio.Event()

    async def reserve(job_id):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        return await manager.allocate_resources(job_id, requirements)

    tasks = [asyncio.create_task(reserve(str(i))) for i in range(2)]
    try:
        await asyncio.wait_for(both_started.wait(), timeout=2)
    finally:
        gpu.release()
    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
    assert sum(results) == 1
    assert len(manager.allocated_resources) == 1
    assert manager.get_resource_utilization()[ResourceType.CPU] == 1


@pytest.mark.asyncio
async def test_repeated_requirements_are_summed_before_admission(manager):
    duplicate = [ResourceRequirement(ResourceType.CPU, 0.75)] * 2
    assert not await manager.can_allocate(duplicate)
    assert not await manager.allocate_resources("too-large", duplicate)
    assert manager.allocated_resources == {}
    assert await manager.allocate_resources(
        "fits", [ResourceRequirement(ResourceType.CPU, 0.5)] * 2
    )
    assert manager.allocated_resources["fits"][ResourceType.CPU] == 1


@pytest.mark.asyncio
async def test_duplicate_finite_amounts_cannot_overflow_total_demand(manager):
    with pytest.raises(ValueError, match="finite"):
        await manager.allocate_resources(
            "overflow", [ResourceRequirement(ResourceType.CPU, 1e308)] * 2
        )
    assert not manager.allocated_resources


@pytest.mark.asyncio
@pytest.mark.parametrize("amount", [-1, True, float("nan"), float("inf"), "1", 10**400])
async def test_invalid_amounts_cannot_create_capacity(manager, amount):
    requirement = [ResourceRequirement(ResourceType.CPU, amount)]
    with pytest.raises(ValueError):
        await manager.can_allocate(requirement)
    with pytest.raises(ValueError):
        await manager.allocate_resources("invalid", requirement)
    assert manager.allocated_resources == {}


@pytest.mark.asyncio
async def test_same_job_retry_is_idempotent_and_cannot_resize_reservation(manager):
    requirement = [ResourceRequirement(ResourceType.CPU, 1)]
    assert await manager.allocate_resources("job", requirement)
    assert await manager.allocate_resources("job", requirement)
    assert not await manager.allocate_resources("other", requirement)
    with pytest.raises(ValueError, match="different"):
        await manager.allocate_resources(
            "job", [ResourceRequirement(ResourceType.CPU, 0)]
        )
    await manager.deallocate_resources("job")
    await manager.deallocate_resources("job")
    assert await manager.allocate_resources("other", requirement)


@pytest.mark.asyncio
async def test_cancelled_admission_has_no_partial_reservation(manager):
    gpu = manager.resource_locks[ResourceType.GPU]
    await gpu.acquire()
    started = asyncio.Event()
    requirements = [
        ResourceRequirement(ResourceType.CPU, 1),
        ResourceRequirement(ResourceType.GPU, 1),
    ]

    async def reserve():
        started.set()
        return await manager.allocate_resources("cancelled", requirements)

    task = asyncio.create_task(reserve())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert manager.allocated_resources == {}
    finally:
        gpu.release()
    assert await asyncio.wait_for(
        manager.allocate_resources("replacement", requirements), timeout=2
    )


@pytest.mark.asyncio
async def test_invalid_resource_and_empty_job_identity_are_rejected(manager):
    with pytest.raises(ValueError):
        await manager.allocate_resources("", [])
    with pytest.raises(ValueError):
        await manager.allocate_resources("bad", [ResourceRequirement("cpu", 1)])
    assert manager.allocated_resources == {}
