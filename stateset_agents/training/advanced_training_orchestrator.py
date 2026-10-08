"""
Job scheduling and experiment tracking for explicitly supplied training runners.

Runners own optimization and checkpoint recovery. No simulated training backend
is supplied, and unconfigured jobs are rejected before queueing.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import pickle
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    psutil = None
    PSUTIL_AVAILABLE = False

from stateset_agents.core.advanced_monitoring import (
    get_monitoring_service,
    monitor_async_function,
)
from stateset_agents.core.enhanced_state_management import get_state_service

from .advanced_training_models import (
    ResourceRequirement,
    ResourceType,
    SchedulingStrategy,
    TrainingJob,
    TrainingJobSpec,
    TrainingStatus,
    deserialize_training_job,
    serialize_training_config,
    serialize_training_job,
    validate_runtime_limit,
    validate_scheduling_inputs,
)
from .experiment_tracking import ExperimentTracker
from .job_journal import JobJournal
from .resource_capacity import read_cgroup_capacity

logger = logging.getLogger(__name__)

TRAINING_ORCH_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    OSError,
    asyncio.TimeoutError,
    pickle.PickleError,
    shutil.Error,
)


class ResourceManager:
    """Atomic in-process reservations against a detected resource snapshot.

    These reservations coordinate jobs on one event loop; they do not enforce
    operating-system quotas or coordinate separate orchestrator processes.
    """

    def __init__(
        self,
        *,
        capacity_overrides: Mapping[ResourceType, float] | None = None,
        storage_path: str | Path = ".",
    ) -> None:
        """Detect local capacity or use explicitly configured resource allowances.

        Memory/storage amounts are GiB. Network has no trustworthy automatic
        allowance; configure it explicitly in the same units as job requests.
        Overrides are operator-supplied allowances, not measured OS guarantees.
        """
        overrides = self.normalize_requirements(
            [
                ResourceRequirement(kind, value)
                for kind, value in (capacity_overrides or {}).items()
            ]
        )
        self.storage_path = Path(storage_path)
        self.detection_issues: list[str] = []
        self.resource_sources: dict[ResourceType, str] = {}
        self.available_resources: dict[ResourceType, float] = {}
        self.allocated_resources: dict[str, dict[ResourceType, float]] = (
            {}
        )  # job_id -> resources
        self.resource_locks: dict[ResourceType, asyncio.Lock] = {}
        self._allocation_lock = asyncio.Lock()

        # Initialize locks
        for resource_type in ResourceType:
            self.resource_locks[resource_type] = asyncio.Lock()

        # Detect available resources
        self._detect_resources()
        self.available_resources.update(overrides)
        self.resource_sources.update(dict.fromkeys(overrides, "configured"))

    def _detect_resources(self) -> None:
        """Read independent measurements; unknown capacity remains unavailable."""

        def measure(label: str, probe: Callable[[], Any]) -> float:
            try:
                value = probe()
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("Expected numeric capacity")
                measured = float(value)
                if not math.isfinite(measured) or measured < 0:
                    raise ValueError("Invalid capacity")
                return measured
            except Exception as exc:
                self.detection_issues.append(
                    f"{label} unavailable ({type(exc).__name__})"
                )
                return 0.0

        cpu = measure("CPU count", lambda: os.cpu_count() or 0)
        if hasattr(os, "sched_getaffinity"):
            affinity = measure("CPU affinity", lambda: len(os.sched_getaffinity(0)))
            cpu = min(cpu, affinity) if cpu else affinity
        if PSUTIL_AVAILABLE and psutil is not None:
            memory = measure(
                "Available memory", lambda: psutil.virtual_memory().available
            )
        else:
            memory = measure(
                "Available memory",
                lambda: os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
            )
        limits = read_cgroup_capacity()
        self.detection_issues.extend(limits.issues)
        if limits.cpu is not None:
            cpu = min(cpu, limits.cpu)
        if limits.memory_bytes is not None:
            memory = min(memory, limits.memory_bytes)
        gpu = measure(
            "CUDA devices",
            lambda: (
                torch.cuda.device_count()
                if TORCH_AVAILABLE and torch.cuda.is_available()
                else 0
            ),
        )
        storage = measure("Storage", lambda: shutil.disk_usage(self.storage_path).free)
        self.available_resources = {
            ResourceType.CPU: cpu,
            ResourceType.MEMORY: memory / (1024**3),
            ResourceType.GPU: gpu,
            ResourceType.STORAGE: storage / (1024**3),
            ResourceType.NETWORK: 0.0,
        }
        self.resource_sources = {
            ResourceType.CPU: "host_count_process_affinity_visible_cgroups",
            ResourceType.MEMORY: "available_memory_visible_cgroup_headroom",
            ResourceType.GPU: "visible_cuda_devices",
            ResourceType.STORAGE: str(self.storage_path.absolute()),
            ResourceType.NETWORK: "unconfigured",
        }

    @staticmethod
    def normalize_requirements(
        requirements: list[ResourceRequirement],
    ) -> dict[ResourceType, float]:
        """Validate finite nonnegative demand and sum repeated resource types."""
        requested: dict[ResourceType, float] = {}
        for requirement in requirements:
            if not isinstance(requirement, ResourceRequirement) or not isinstance(
                requirement.resource_type, ResourceType
            ):
                raise ValueError("Expected a requirement with a valid ResourceType")
            if isinstance(requirement.amount, bool) or not isinstance(
                requirement.amount, (int, float)
            ):
                raise ValueError("Resource amounts must be finite nonnegative numbers")
            try:
                amount = float(requirement.amount)
            except OverflowError as exc:
                raise ValueError("Resource amount exceeds the supported range") from exc
            total = requested.get(requirement.resource_type, 0.0) + amount
            if amount < 0 or not math.isfinite(total):
                raise ValueError("Resource amounts must be finite nonnegative numbers")
            requested[requirement.resource_type] = total
        return requested

    async def _has_capacity(self, requested: dict[ResourceType, float]) -> bool:
        """Check demand while the caller holds the allocation lock."""
        for resource_type, amount in requested.items():
            async with self.resource_locks[resource_type]:
                available = self.available_resources.get(resource_type, 0)
                allocated = sum(
                    alloc.get(resource_type, 0)
                    for alloc in self.allocated_resources.values()
                )
                if available - allocated < amount:
                    return False
        return True

    async def can_allocate(self, requirements: list[ResourceRequirement]) -> bool:
        """Inspect capacity without reserving it; allocation checks again atomically."""
        requested = self.normalize_requirements(requirements)
        async with self._allocation_lock:
            return await self._has_capacity(requested)

    async def allocate_resources(
        self, job_id: str, requirements: list[ResourceRequirement]
    ) -> bool:
        """Reserve all resources or none; an identical job retry is idempotent."""
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id must be nonempty text")
        requested = self.normalize_requirements(requirements)
        async with self._allocation_lock:
            if job_id in self.allocated_resources:
                if self.allocated_resources[job_id] != requested:
                    raise ValueError("Job already has a different resource reservation")
                return True
            if not await self._has_capacity(requested):
                return False
            # No await between this commit and releasing the lock. Cancellation
            # during a capacity check cannot leave a partial reservation behind.
            self.allocated_resources[job_id] = requested
            return True

    async def deallocate_resources(self, job_id: str) -> None:
        """Release a reservation atomically; repeated release is harmless."""
        async with self._allocation_lock:
            self.allocated_resources.pop(job_id, None)

    def get_resource_utilization(self) -> dict[ResourceType, float]:
        """Get current resource utilization"""
        utilization = {}

        for resource_type in ResourceType:
            available = self.available_resources.get(resource_type, 0)
            allocated = sum(
                alloc.get(resource_type, 0)
                for alloc in self.allocated_resources.values()
            )

            if available > 0:
                utilization[resource_type] = allocated / available
            else:
                utilization[resource_type] = 0.0

        return utilization


class JobScheduler:
    """Intelligent job scheduling"""

    def __init__(
        self, strategy: SchedulingStrategy | str = SchedulingStrategy.RESOURCE_AWARE
    ) -> None:
        self.strategy = SchedulingStrategy(strategy)
        self.job_queue: list[TrainingJob] = []
        self.running_jobs: dict[str, TrainingJob] = {}
        self.completed_jobs: dict[str, TrainingJob] = {}
        self.queue_lock = asyncio.Lock()
        self._closed = False
        self._dispatch_sequence = 0
        self._fair_turn: dict[str | None, int] = {}

    async def submit_job(self, job: TrainingJob) -> None:
        """Admit a new job identity while this scheduler is open."""
        if not isinstance(job.job_id, str) or not job.job_id.strip():
            raise ValueError("job_id must be nonempty text")
        ResourceManager.normalize_requirements(job.config.resource_requirements)
        validate_runtime_limit(job.config.max_runtime)
        validate_scheduling_inputs(job.config, job.priority, job.user_id)
        async with self.queue_lock:
            if self._closed:
                raise RuntimeError("Training scheduler is closed")
            if (
                job.job_id in self.running_jobs
                or job.job_id in self.completed_jobs
                or any(queued.job_id == job.job_id for queued in self.job_queue)
            ):
                raise ValueError("Job identity has already been submitted")
            if job.status != TrainingStatus.PENDING:
                raise ValueError("Only pending jobs can be submitted")
            job.status = TrainingStatus.QUEUED
            self.job_queue.append(job)
            self._sort_queue()

    async def get_next_job(
        self, resource_manager: ResourceManager
    ) -> TrainingJob | None:
        """Reserve and dequeue the next job; the caller must start or release it."""
        async with self.queue_lock:
            if self._closed:
                return None
            candidates = list(enumerate(self.job_queue))
            if self.strategy == SchedulingStrategy.FAIR_SHARE:
                candidates = self._fair_candidates(resource_manager, candidates)
            for i, job in candidates:
                if await resource_manager.allocate_resources(
                    job.job_id, job.config.resource_requirements
                ):
                    # No suspension between reservation and scheduler ownership.
                    self.running_jobs[job.job_id] = job
                    self._dispatch_sequence += 1
                    self._fair_turn[job.user_id] = self._dispatch_sequence
                    return self.job_queue.pop(i)
            return None

    def _fair_candidates(
        self,
        resource_manager: ResourceManager,
        candidates: list[tuple[int, TrainingJob]],
    ) -> list[tuple[int, TrainingJob]]:
        """Prefer owners with the smallest largest fraction of reserved capacity."""
        for _, job in candidates:
            # New owners join the current rotation, rather than indefinitely
            # jumping ahead of existing waiters with an artificial zero turn.
            self._fair_turn.setdefault(job.user_id, self._dispatch_sequence)
        totals: dict[str | None, dict[ResourceType, float]] = {}
        for job_id, running in self.running_jobs.items():
            reserved = resource_manager.allocated_resources.get(job_id, {})
            owner = totals.setdefault(running.user_id, {})
            for resource, amount in reserved.items():
                owner[resource] = owner.get(resource, 0.0) + amount
        shares: dict[str | None, float] = {}
        for user_id, resources in totals.items():
            shares[user_id] = 0.0
            for resource, amount in resources.items():
                if amount <= 0:
                    continue
                capacity = resource_manager.available_resources.get(resource, 0.0)
                share = amount / capacity if capacity > 0 else math.inf
                shares[user_id] = max(shares[user_id], share)
        return sorted(
            candidates,
            key=lambda candidate: (
                shares.get(candidate[1].user_id, 0.0),
                self._fair_turn[candidate[1].user_id],
                candidate[0],
            ),
        )

    async def close(self) -> list[TrainingJob]:
        """Stop admission and cancel queued jobs, returning newly cancelled jobs."""
        self._closed = True
        async with self.queue_lock:
            cancelled = list(self.job_queue)
            self.job_queue.clear()
            for job in cancelled:
                job.status = TrainingStatus.CANCELLED
                job.completed_at = time.time()
                self.completed_jobs[job.job_id] = job
            return cancelled

    def _sort_queue(self):
        """Sort job queue based on scheduling strategy"""
        if self.strategy == SchedulingStrategy.FIFO:
            # Already in FIFO order
            pass
        elif self.strategy == SchedulingStrategy.PRIORITY:
            self.job_queue.sort(key=lambda job: -job.priority)
        elif self.strategy == SchedulingStrategy.SHORTEST_JOB_FIRST:
            self.job_queue.sort(key=lambda job: job.config.num_epochs)
        # RESOURCE_AWARE retains submission order and backfills jobs that fit.
        # FAIR_SHARE is ordered at admission using the current reservation ledger.

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        async with self.queue_lock:
            # Remove from queue
            for i, job in enumerate(self.job_queue):
                if job.job_id == job_id:
                    job.status = TrainingStatus.CANCELLED
                    job.completed_at = time.time()
                    self.job_queue.pop(i)
                    self.completed_jobs[job_id] = job
                    return True

            # Cancel running job
            if job_id in self.running_jobs:
                job = self.running_jobs[job_id]
                if job.status in (
                    TrainingStatus.COMPLETED,
                    TrainingStatus.FAILED,
                    TrainingStatus.TIMED_OUT,
                ):
                    return False
                job.status = TrainingStatus.CANCELLED
                # The orchestrator requests cancellation of the owning worker.
                return True

        return False

    def get_queue_status(self) -> dict[str, Any]:
        """Get scheduler status"""
        return {
            "queued_jobs": len(self.job_queue),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": len(self.completed_jobs),
            "strategy": self.strategy.value,
        }


@dataclass(frozen=True)
class TrainingRunResult:
    """Measured output of a real runner; the runner owns training and recovery."""

    artifact_path: Path
    metrics: dict[str, float]
    steps: int
    epochs: int

    def validate(self) -> None:
        """Reject incomplete runs before publishing successful job state."""
        for name in ("steps", "epochs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Training result {name} must be a positive integer")
        if not self.metrics or any(
            not isinstance(k, str)
            or not k
            or isinstance(v, bool)
            or not isinstance(v, (float, int))
            or not math.isfinite(v)
            for k, v in self.metrics.items()
        ):
            raise ValueError("Training result requires finite measured metrics")
        path = Path(self.artifact_path)
        files = (
            [path] if path.is_file() else list(path.rglob("*")) if path.is_dir() else []
        )
        if not any(p.is_file() and p.stat().st_size > 0 for p in files):
            raise ValueError("Training result requires a nonempty saved artifact")


TrainingRunner = Callable[[TrainingJob], Awaitable[TrainingRunResult]]
MISSING_RUNNER = (
    "No training runner configured. Supply training_runner backed by a real "
    "trainer, or use train-remote. Simulated training is not supported."
)


class _TrainingRuntimeExceeded(TimeoutError):
    """A local runner deadline expired; distinct from a provider timeout."""


class TrainingWorker:
    """Execute an injected trainer without inventing metrics or checkpoints.

    A runner owns checkpoint restoration and retries: blindly retrying an
    optimizer operation could apply it twice. Artifact validation checks local
    completeness, not that arbitrary caller-supplied code learned a useful policy.
    """

    def __init__(self, worker_id: str, training_runner: TrainingRunner | None = None):
        self.worker_id = worker_id
        self.training_runner = training_runner
        self.current_job: TrainingJob | None = None

    @monitor_async_function("training_worker.execute_job")
    async def execute_job(
        self,
        job: TrainingJob,
        experiment_tracker: ExperimentTracker | None,
        checkpoint_callback: Callable | None = None,
    ) -> bool:
        """Publish completion only after a runner returns valid saved output."""
        self.current_job = job
        experiment_id = None
        try:
            if self.training_runner is None:
                raise RuntimeError(MISSING_RUNNER)
            if checkpoint_callback is not None:
                raise ValueError(
                    "Configure checkpoint callbacks in the training runner"
                )
            runtime_limit = validate_runtime_limit(job.config.max_runtime)
            if job.status == TrainingStatus.CANCELLED:
                return False
            job.status = TrainingStatus.RUNNING
            job.started_at = time.time()
            if experiment_tracker is not None:
                experiment_id = await experiment_tracker.start_experiment(job)
            result = await self._run_training(job, runtime_limit)
            if job.status == TrainingStatus.CANCELLED:
                return False
            if not isinstance(result, TrainingRunResult):
                raise ValueError("Runner must return TrainingRunResult")
            result.validate()
            job.current_step = result.steps
            job.current_epoch = result.epochs
            job.metrics = {
                name: [float(value)] for name, value in result.metrics.items()
            }
            job.checkpoint_path = str(result.artifact_path)
            job.completed_at = time.time()
            if experiment_tracker is not None and experiment_id is not None:
                await experiment_tracker.log_metrics(
                    experiment_id, result.metrics, result.steps
                )
                await experiment_tracker.log_artifact(
                    experiment_id, job.checkpoint_path, "final_model"
                )
            # Tracking awaits can change status after the earlier cancellation check.
            if TrainingStatus(job.status) == TrainingStatus.CANCELLED:
                return False
            job.status = TrainingStatus.COMPLETED
        except _TrainingRuntimeExceeded as exc:
            job.status = TrainingStatus.TIMED_OUT
            job.completed_at = time.time()
            job.last_error = str(exc)
            return False
        except asyncio.CancelledError:
            job.status = TrainingStatus.CANCELLED
            job.completed_at = time.time()
            raise
        except Exception as exc:  # A runner is an external integration boundary.
            job.status = TrainingStatus.FAILED
            job.completed_at = time.time()
            job.last_error = str(exc)
            logger.exception("Training job %s failed", job.job_id)
            return False
        finally:
            self.current_job = None
            if job.status == TrainingStatus.CANCELLED and job.completed_at is None:
                job.completed_at = time.time()
            if experiment_tracker is not None and experiment_id is not None:
                try:
                    await experiment_tracker.finish_experiment(
                        experiment_id,
                        (
                            {name: history[-1] for name, history in job.metrics.items()}
                            if job.status == TrainingStatus.COMPLETED
                            else None
                        ),
                        status=job.status,
                    )
                except Exception:
                    logger.exception("Could not finalize experiment %s", experiment_id)
        return job.status == TrainingStatus.COMPLETED

    async def _run_training(
        self, job: TrainingJob, runtime_limit: float | None
    ) -> TrainingRunResult:
        """Request cancellation at the deadline and drain cleanup before returning."""
        assert self.training_runner is not None
        if runtime_limit is None:
            return await self.training_runner(job)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + runtime_limit
        finished_at: float | None = None
        implementation = self.training_runner
        message = f"Training runner exceeded max_runtime of {runtime_limit:g} seconds"

        async def invoke() -> TrainingRunResult:
            nonlocal finished_at
            try:
                if loop.time() >= deadline:
                    raise _TrainingRuntimeExceeded(message)
                return await implementation(job)
            finally:
                finished_at = loop.time()

        runner = asyncio.create_task(invoke())
        cancelled = False
        try:
            await asyncio.wait({runner}, timeout=max(0, deadline - loop.time()))
            # Measure when the runner finished, not when this waiter resumed:
            # unrelated event-loop stalls must not turn timely results into timeouts.
            if runner.done() and finished_at is not None and finished_at <= deadline:
                return runner.result()
        except asyncio.CancelledError:
            cancelled = True
        if not runner.done():
            runner.cancel()
        while not runner.done():
            try:
                # wait() does not forward outer cancellation into runner cleanup.
                await asyncio.wait({runner})
            except asyncio.CancelledError:
                cancelled = True
        if not runner.cancelled():
            cleanup_error = runner.exception()
            if cleanup_error is not None and not isinstance(
                cleanup_error, _TrainingRuntimeExceeded
            ):
                message += f"; runner exited with {type(cleanup_error).__name__}"
        if cancelled:
            raise asyncio.CancelledError
        raise _TrainingRuntimeExceeded(message)

    def _compute_final_metrics(self, job: TrainingJob) -> dict[str, float]:
        """Report observed progress only; never supply invented loss or reward."""
        return {
            **{name: history[-1] for name, history in job.metrics.items() if history},
            "final_epoch": job.current_epoch,
            "total_steps": job.current_step,
            "training_time": job.runtime or 0,
        }


class AdvancedTrainingOrchestrator:
    """Main training orchestrator service"""

    def __init__(
        self,
        max_concurrent_jobs: int = 4,
        scheduling_strategy: (
            SchedulingStrategy | str
        ) = SchedulingStrategy.RESOURCE_AWARE,
        enable_experiment_tracking: bool = True,
        start_background_tasks: bool = True,
        training_runner: TrainingRunner | None = None,
        resource_manager: ResourceManager | None = None,
        journal_dir: str | Path | None = None,
    ) -> None:
        """Initialize scheduling with detected capacity or an explicit resource manager."""
        if type(max_concurrent_jobs) is not int or max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be a positive integer")
        scheduling_strategy = SchedulingStrategy(scheduling_strategy)
        self.training_runner = training_runner
        self.max_concurrent_jobs = max_concurrent_jobs
        self.resource_manager = (
            resource_manager if resource_manager is not None else ResourceManager()
        )
        self.scheduler = JobScheduler(scheduling_strategy)
        self.experiment_tracker = (
            ExperimentTracker() if enable_experiment_tracking else None
        )
        self._background_tasks_enabled = start_background_tasks
        self._closing = False
        self._cleanup_lock = asyncio.Lock()
        self._admission_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._state_write_locks: dict[str, asyncio.Lock] = {}
        self._pending_state_jobs: dict[str, TrainingJob] = {}

        # Worker management
        self.workers: dict[str, TrainingWorker] = {}
        self.worker_tasks: dict[str, asyncio.Task] = {}
        self.worker_jobs: dict[str, TrainingJob] = {}
        self._cancelling_workers: set[str] = set()

        # State management
        self.state_service = get_state_service()
        self.monitoring = get_monitoring_service()

        # Background tasks
        self._orchestration_task: asyncio.Task[None] | None = None
        self._monitoring_task: asyncio.Task[None] | None = None

        self.journal = JobJournal(journal_dir) if journal_dir is not None else None
        if self.journal is not None:
            try:
                self._restore_journal()
            except BaseException:
                self.journal.close()
                raise

        self._start_background_tasks()

    def _restore_journal(self) -> None:
        """Restore queued work; never repeat an optimizer operation after a crash."""
        assert self.journal is not None
        jobs = self.journal.load()
        for job in jobs:
            ResourceManager.normalize_requirements(job.config.resource_requirements)
            validate_runtime_limit(job.config.max_runtime)
            validate_scheduling_inputs(job.config, job.priority, job.user_id)
            if (
                job.status == TrainingStatus.QUEUED
                and job.started_at is None
                and self.training_runner is None
            ):
                raise RuntimeError(MISSING_RUNNER)
        for job in jobs:
            if job.status == TrainingStatus.QUEUED and job.started_at is None:
                self.scheduler.job_queue.append(job)
            else:
                if job.status not in (
                    TrainingStatus.COMPLETED,
                    TrainingStatus.FAILED,
                    TrainingStatus.INTERRUPTED,
                    TrainingStatus.TIMED_OUT,
                ) and not (
                    job.status == TrainingStatus.CANCELLED
                    and job.completed_at is not None
                ):
                    job.status = TrainingStatus.INTERRUPTED
                    job.completed_at = time.time()
                    job.last_error = (
                        "Orchestrator ownership was lost before a terminal outcome was "
                        "recorded. External training may still be active. Reconcile "
                        "provider/process state and checkpoints before submitting a new job."
                    )
                    self.journal.save(serialize_training_job(job))
                self.scheduler.completed_jobs[job.job_id] = job
            self._pending_state_jobs[job.job_id] = job
        self.scheduler._sort_queue()

    def _start_background_tasks(self):
        """Start background orchestration tasks"""
        if not self._background_tasks_enabled or self._closing:
            return
        if self._orchestration_task or self._monitoring_task:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover
            # Constructed outside an event loop (tests or import-time); start later.
            logger.debug("No running event loop; skipping background task startup.")
            return

        self._orchestration_task = loop.create_task(self._orchestration_loop())
        self._monitoring_task = loop.create_task(self._monitoring_loop())

    async def submit_training_job(
        self,
        config: TrainingJobSpec,
        priority: int = 1,
        user_id: str | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> str:
        """Submit a configuration snapshot, or return the same keyed submission.

        Keys are scoped to user_id and this orchestrator/journal. Reusing a key
        with changed configuration or priority fails; terminal jobs are not rerun.
        """
        config = copy.deepcopy(config)
        identity: tuple[str, str] | None = None
        if idempotency_key is not None:
            if (
                not isinstance(idempotency_key, str)
                or not idempotency_key.strip()
                or len(idempotency_key) > 256
            ):
                raise ValueError("idempotency_key must contain 1 to 256 characters")
            if user_id is not None and not isinstance(user_id, str):
                raise ValueError("user_id must be text or None")
            payload = {
                "config": serialize_training_config(config),
                "priority": priority,
                "user_id": user_id,
            }
            try:
                canonical = json.dumps(payload, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Idempotent requests must be JSON-serializable"
                ) from exc
            if json.loads(canonical) != payload:
                raise ValueError(
                    "Idempotent requests must round-trip through JSON unchanged"
                )
            fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            key_hash = hashlib.sha256(
                json.dumps([user_id, idempotency_key]).encode("utf-8")
            ).hexdigest()
            identity = (f"idem_{key_hash}", fingerprint)
        async with self._admission_lock:
            return await self._submit_training_job(config, priority, user_id, identity)

    async def _submit_training_job(
        self,
        config: TrainingJobSpec,
        priority: int,
        user_id: str | None,
        identity: tuple[str, str] | None,
    ) -> str:
        """Persist and enqueue under the admission lock shared with shutdown."""
        ResourceManager.normalize_requirements(config.resource_requirements)
        validate_runtime_limit(config.max_runtime)
        validate_scheduling_inputs(config, priority, user_id)
        job_id, fingerprint = (
            identity if identity is not None else (str(uuid.uuid4()), None)
        )
        if identity is not None:
            existing = (
                self.scheduler.running_jobs.get(job_id)
                or self.scheduler.completed_jobs.get(job_id)
                or next(
                    (job for job in self.scheduler.job_queue if job.job_id == job_id),
                    None,
                )
            )
            if existing is not None:
                if existing.submission_fingerprint != fingerprint:
                    raise ValueError(
                        "Idempotency key was already used for a different request"
                    )
                if job_id in self._pending_state_jobs:
                    await self._persist_job(existing)
                return job_id
        if self._closing:
            raise RuntimeError("Training orchestrator is shutting down")
        if self.training_runner is None:
            raise RuntimeError(MISSING_RUNNER)
        self._start_background_tasks()

        job = TrainingJob(
            job_id=job_id,
            config=config,
            priority=priority,
            user_id=user_id,
            submission_fingerprint=fingerprint,
        )

        try:
            # Persist the admitted status before making the job runnable.
            job.status = TrainingStatus.QUEUED
            await self._persist_job(job)
            if self._closing:
                raise RuntimeError("Training orchestrator is shutting down")
            job.status = TrainingStatus.PENDING
            await self.scheduler.submit_job(job)
        except (Exception, asyncio.CancelledError):
            job.status = TrainingStatus.CANCELLED
            job.completed_at = time.time()
            self.scheduler.completed_jobs[job_id] = job
            self._pending_state_jobs[job_id] = job
            try:
                await self._persist_job(job)
            except Exception:
                logger.exception("Could not persist rejected submission %s", job_id)
            raise

        logger.info(f"Training job submitted: {job_id}")
        return job_id

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get training job status"""
        self._start_background_tasks()
        job_data = (
            self.journal.get(job_id)
            if self.journal is not None
            else await self.state_service.state_manager.get(f"training_job:{job_id}")
        )
        if not job_data:
            return None

        # Convert back to TrainingJob object for runtime calculation
        job = deserialize_training_job(job_data)

        return {
            "job_id": job_id,
            "status": job.status.value,
            "progress": {
                "current_epoch": job.current_epoch,
                "total_epochs": job.config.num_epochs,
                "current_step": job.current_step,
                "progress_percent": (job.current_epoch / job.config.num_epochs) * 100,
            },
            "runtime": job.runtime,
            "estimated_completion": job.estimated_completion,
            "assigned_resources": job.assigned_resources,
            "metrics": job.metrics,
            "last_error": job.last_error,
        }

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        self._start_background_tasks()
        success = await self.scheduler.cancel_job(job_id)

        if success:
            # Stop execution before any state-store I/O. Keep reservations until
            # the task has finished its cancellation cleanup.
            self._cancel_worker(f"worker_{job_id}")
            job = self.scheduler.running_jobs.get(job_id)
            if job is None:
                job = self.scheduler.completed_jobs[job_id]
                self._pending_state_jobs[job_id] = job
            await self._persist_job(job)

        return success

    async def _persist_job(self, job: TrainingJob) -> None:
        """Serialize writes per identity; clear retries only after a matching write."""
        lock = self._state_write_locks.setdefault(job.job_id, asyncio.Lock())
        async with lock:
            snapshot = copy.deepcopy(serialize_training_job(job))
            if self.journal is not None:
                self.journal.save(snapshot)
            try:
                await self.state_service.state_manager.set(
                    f"training_job:{job.job_id}", snapshot
                )
            except Exception:
                if self.journal is None:
                    raise
                # The journal owns this transition. A failed cache projection
                # must not turn an accepted durable job into a rejected one.
                self._pending_state_jobs[job.job_id] = job
                logger.warning("Cache update deferred for training job %s", job.job_id)
                return
            if snapshot == serialize_training_job(job):
                self._pending_state_jobs.pop(job.job_id, None)

    def _cancel_worker(self, worker_id: str) -> None:
        """Request cancellation once so repeated requests do not interrupt cleanup."""
        task = self.worker_tasks.get(worker_id)
        if (
            task is not None
            and not task.done()
            and worker_id not in self._cancelling_workers
        ):
            self._cancelling_workers.add(worker_id)
            task.cancel()

    async def get_system_status(self) -> dict[str, Any]:
        """Get orchestrator system status"""
        self._start_background_tasks()
        resource_utilization = self.resource_manager.get_resource_utilization()
        queue_status = self.scheduler.get_queue_status()

        return {
            "resource_utilization": {
                rt.value: util for rt, util in resource_utilization.items()
            },
            "queue_status": queue_status,
            "active_workers": len(self.workers),
            "running_tasks": len(self.worker_tasks),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "available_resources": {
                rt.value: amount
                for rt, amount in self.resource_manager.available_resources.items()
            },
            "resource_sources": {
                rt.value: source
                for rt, source in self.resource_manager.resource_sources.items()
            },
            "resource_detection_issues": list(self.resource_manager.detection_issues),
            "durable_journal_enabled": self.journal is not None,
        }

    async def _orchestration_loop(self):
        """Main orchestration loop"""
        while True:
            try:
                await self._cleanup_completed_tasks()
                active_jobs = len(self.worker_tasks)

                if active_jobs < self.max_concurrent_jobs:
                    # Get next job from scheduler
                    next_job = await self.scheduler.get_next_job(self.resource_manager)

                    if next_job:
                        await self._start_job(next_job)

                await asyncio.sleep(5)  # Check every 5 seconds

            except TRAINING_ORCH_EXCEPTIONS as e:
                logger.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(30)

    async def _start_job(self, job: TrainingJob):
        """Transfer a reserved job to a worker or release failed startup."""
        worker_id = f"worker_{job.job_id}"
        worker = TrainingWorker(worker_id, training_runner=self.training_runner)
        if job.status != TrainingStatus.CANCELLED:
            job.status = TrainingStatus.RUNNING
        if job.started_at is None:
            job.started_at = time.time()

        self.workers[worker_id] = worker
        self.worker_jobs[worker_id] = job
        self.scheduler.running_jobs[job.job_id] = job

        try:
            await self._persist_job(job)
            if job.status == TrainingStatus.CANCELLED or self._closing:
                job.status = TrainingStatus.CANCELLED
                return
            task = asyncio.create_task(worker.execute_job(job, self.experiment_tracker))
            self.worker_tasks[worker_id] = task
        except asyncio.CancelledError:
            job.status = TrainingStatus.CANCELLED
            raise
        except Exception as exc:
            job.status = TrainingStatus.FAILED
            job.last_error = str(exc)
            raise
        finally:
            if worker_id not in self.worker_tasks:
                job.completed_at = time.time()
                await self.resource_manager.deallocate_resources(job.job_id)
                self.workers.pop(worker_id, None)
                self.worker_jobs.pop(worker_id, None)
                self.scheduler.running_jobs.pop(job.job_id, None)
                self.scheduler.completed_jobs[job.job_id] = job
                self._pending_state_jobs[job.job_id] = job
                try:
                    await self._persist_job(job)
                except Exception:
                    logger.exception("Could not persist failed startup %s", job.job_id)

        logger.info(f"Started training job {job.job_id} on worker {worker_id}")

    async def _cleanup_completed_tasks(self):
        """Clean up completed worker tasks"""
        async with self._cleanup_lock:
            errors: list[Exception] = []
            try:
                await self._reap_completed_tasks()
            except Exception as exc:
                errors.append(exc)
            for job in list(self._pending_state_jobs.values()):
                try:
                    await self._persist_job(job)
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]

    async def _reap_completed_tasks(self) -> None:
        """Reap a snapshot while holding the cleanup lock; retry failed writes."""
        completed_workers = []
        persistence_errors: list[Exception] = []

        for worker_id, task in list(self.worker_tasks.items()):
            if task.done():
                # Get job and deallocate resources
                job = self.worker_jobs.get(worker_id)
                if job is not None:
                    if task.cancelled():
                        # Cancellation during final tracker closure cannot undo
                        # an already established training outcome.
                        if job.status not in (
                            TrainingStatus.COMPLETED,
                            TrainingStatus.FAILED,
                            TrainingStatus.TIMED_OUT,
                            TrainingStatus.INTERRUPTED,
                        ):
                            job.status = TrainingStatus.CANCELLED
                            job.completed_at = job.completed_at or time.time()
                    elif task.exception() is not None:
                        job.status = TrainingStatus.FAILED
                        job.last_error = str(task.exception())
                        job.completed_at = job.completed_at or time.time()
                    await self.resource_manager.deallocate_resources(job.job_id)

                    # Move job to completed
                    if job.job_id in self.scheduler.running_jobs:
                        completed_job = self.scheduler.running_jobs.pop(job.job_id)
                        self.scheduler.completed_jobs[job.job_id] = completed_job

                    # Update job in state
                    try:
                        await self._persist_job(job)
                    except Exception as exc:
                        # One failed write must not strand other finished jobs'
                        # reservations. Keep this task for a later write retry.
                        persistence_errors.append(exc)
                        continue
                completed_workers.append(worker_id)

        # Remove completed workers
        for worker_id in completed_workers:
            del self.workers[worker_id]
            del self.worker_tasks[worker_id]
            self.worker_jobs.pop(worker_id, None)
            self._cancelling_workers.discard(worker_id)
        if persistence_errors:
            raise persistence_errors[0]

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                # Record system metrics
                resource_util = self.resource_manager.get_resource_utilization()

                for resource_type, utilization in resource_util.items():
                    self.monitoring.metrics_collector.record_metric(
                        f"orchestrator.resource_utilization.{resource_type.value}",
                        utilization,
                    )

                # Record queue metrics
                queue_status = self.scheduler.get_queue_status()
                for metric_name, value in queue_status.items():
                    if isinstance(value, (int, float)):
                        self.monitoring.metrics_collector.record_metric(
                            f"orchestrator.queue.{metric_name}", value
                        )

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except TRAINING_ORCH_EXCEPTIONS as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def shutdown(self) -> None:
        """Close admission, cancel accepted work, and persist terminal states."""
        self._closing = True
        async with self._shutdown_lock:
            await self._shutdown()

    async def _shutdown(self) -> None:
        """Drain submissions and workers; a later call retries failed writes."""
        logger.info("Shutting down training orchestrator...")

        # Cancel all background tasks
        if self._orchestration_task:
            self._orchestration_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()
        background = [
            task
            for task in (self._orchestration_task, self._monitoring_task)
            if task is not None
        ]
        await asyncio.gather(*background, return_exceptions=True)

        # Signal every worker before awaiting persistence or runner cleanup.
        for job in list(self.scheduler.running_jobs.values()):
            if job.status not in (
                TrainingStatus.COMPLETED,
                TrainingStatus.FAILED,
                TrainingStatus.TIMED_OUT,
            ):
                job.status = TrainingStatus.CANCELLED
        for worker_id in self.worker_tasks:
            self._cancel_worker(worker_id)

        # Wait for any in-flight submission to finish or compensate its write.
        # No new orchestrator submission can pass the closing check.
        async with self._admission_lock:
            for job in await self.scheduler.close():
                self._pending_state_jobs[job.job_id] = job

        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks.values(), return_exceptions=True)
        await self._cleanup_completed_tasks()

        if self.journal is not None:
            self.journal.close()
            # Durable records are authoritative; a future owner reconstructs
            # cache projections. Do not write to this journal after releasing it.
            self._pending_state_jobs.clear()

        logger.info("Training orchestrator shutdown complete")


# Global orchestrator instance
_orchestrator: AdvancedTrainingOrchestrator | None = None


def get_training_orchestrator() -> AdvancedTrainingOrchestrator:
    """Get or create global training orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AdvancedTrainingOrchestrator()
    return _orchestrator


if __name__ == "__main__":
    raise SystemExit(MISSING_RUNNER)
