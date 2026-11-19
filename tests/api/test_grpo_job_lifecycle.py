"""Legacy GRPO job ownership and terminal-state regression coverage."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from stateset_agents.api.grpo import handlers
from stateset_agents.api.grpo import state as state_module
from stateset_agents.api.grpo.metrics import GRPOMetrics
from stateset_agents.api.grpo.models import GRPOTrainingRequest
from stateset_agents.api.grpo.state import StateManager
from tests.api.test_grpo_handlers import _grpo_service_test_app


def create_job(state, job_id="job", user_id="owner"):
    return state.create_job(job_id, "computational", user_id, "request")


@pytest.fixture
def handler(monkeypatch):
    state = StateManager()
    metrics = GRPOMetrics()
    monkeypatch.setattr(handlers, "get_state_manager", lambda: state)
    monkeypatch.setattr(handlers, "get_grpo_metrics", lambda: metrics)
    return handlers.TrainingHandler({})


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize(
    "late_status", [None, "starting", "running", "completed", "failed", "cancelled"]
)
def test_terminal_updates_preserve_all_completion_evidence(terminal, late_status):
    state = StateManager()
    job = create_job(state)
    state.update_job(job.job_id, status="running")
    state.update_job(
        job.job_id,
        status=terminal,
        iterations=2,
        trajectories=4,
        result={"average_reward": 0.5},
        error="original error" if terminal == "failed" else None,
    )
    snapshot = asdict(job)
    timestamp = state.jobs._timestamps[job.job_id]
    returned = state.update_job(
        job.job_id,
        status=late_status,
        iterations=99,
        trajectories=100,
        error="late error",
        result={"average_reward": 999},
    )
    assert returned is job
    assert asdict(job) == snapshot
    assert state.jobs._timestamps[job.job_id] == timestamp


@pytest.mark.parametrize("late_status", ["starting", "unknown"])
def test_invalid_transition_cannot_partially_publish_progress(late_status):
    state = StateManager()
    job = create_job(state)
    state.update_job(job.job_id, status="running", iterations=1)
    snapshot = asdict(job)
    state.update_job(
        job.job_id, status=late_status, iterations=99, result={"late": True}
    )
    assert asdict(job) == snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["computational", "distributed"])
async def test_cancel_before_worker_start_never_calls_engine(handler, strategy):
    calls = []

    async def train_iteration(prompts):
        calls.append(prompts)
        return {"trajectories_generated": 1}

    handler.services["demo_engine"] = SimpleNamespace(train_iteration=train_iteration)
    job = create_job(handler.state)
    assert handler.cancel_job(job.job_id, "owner")
    snapshot = asdict(job)
    if strategy == "distributed":
        await handler.run_distributed_training(job.job_id, ["prompt"], 2, {})
    else:
        await handler.run_computational_training(job.job_id, ["prompt"], 2)
    assert not calls
    assert asdict(job) == snapshot
    assert handler.metrics.training_jobs_completed == 0
    assert handler.metrics.training_jobs_failed == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "malformed"])
async def test_cancel_during_iteration_discards_late_result_and_stops_worker(
    handler, outcome
):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def train_iteration(prompts):
        calls.append(prompts)
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise RuntimeError("late engine failure")
        if outcome == "malformed":
            return None
        return {"trajectories_generated": 4, "average_reward": 0.8}

    handler.services["demo_engine"] = SimpleNamespace(train_iteration=train_iteration)
    job = create_job(handler.state)
    worker = asyncio.create_task(
        handler.run_computational_training(job.job_id, ["prompt"], 3)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert handler.cancel_job(job.job_id, "owner")
        snapshot = asdict(job)
        release.set()
        await asyncio.wait_for(worker, timeout=1)
        assert len(calls) == 1
        assert asdict(job) == snapshot
        assert handler.metrics.training_jobs_completed == 0
        assert handler.metrics.training_jobs_failed == 0
    finally:
        release.set()
        await worker


@pytest.mark.asyncio
async def test_worker_task_cancellation_records_cancelled_and_propagates(handler):
    entered = asyncio.Event()

    async def train_iteration(prompts):
        entered.set()
        await asyncio.Future()

    handler.services["demo_engine"] = SimpleNamespace(train_iteration=train_iteration)
    job = create_job(handler.state)
    worker = asyncio.create_task(
        handler.run_computational_training(job.job_id, ["prompt"], 3)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(worker, timeout=1)
        assert job.status == "cancelled"
        assert job.completed_at is not None
        assert handler.metrics.training_jobs_failed == 0
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_unexpected_worker_error_records_failure_and_still_propagates(handler):
    error = KeyError("bad engine result")

    async def train_iteration(prompts):
        raise error

    handler.services["demo_engine"] = SimpleNamespace(train_iteration=train_iteration)
    job = create_job(handler.state)
    with pytest.raises(KeyError) as caught:
        await handler.run_computational_training(job.job_id, ["prompt"], 1)
    assert caught.value is error
    assert job.status == "failed"
    assert job.completed_at is not None
    assert handler.metrics.training_jobs_failed == 1


@pytest.mark.asyncio
async def test_duplicate_worker_does_not_execute_or_record_completion_twice(handler):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def train_iteration(prompts):
        calls.append(prompts)
        entered.set()
        await release.wait()
        return {"trajectories_generated": 2, "total_computation_used": 1.5}

    handler.services["demo_engine"] = SimpleNamespace(train_iteration=train_iteration)
    job = create_job(handler.state)
    first = asyncio.create_task(
        handler.run_computational_training(job.job_id, ["prompt"], 1)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.wait_for(
            handler.run_distributed_training(job.job_id, ["prompt"], 1, {}), timeout=0.2
        )
        assert len(calls) == 1
        release.set()
        await first
        assert job.status == "completed"
        assert handler.metrics.training_jobs_completed == 1
        assert handler.metrics.total_trajectories == 2
        assert handler.metrics.total_computation == 1.5
    finally:
        release.set()
        await first


@pytest.mark.asyncio
async def test_old_worker_cannot_update_replacement_job_with_same_id(handler):
    entered, release = asyncio.Event(), asyncio.Event()

    async def train_iteration(prompts):
        entered.set()
        await release.wait()
        return {"trajectories_generated": 2}

    handler.services["demo_engine"] = SimpleNamespace(train_iteration=train_iteration)
    job = create_job(handler.state)
    worker = asyncio.create_task(
        handler.run_computational_training(job.job_id, ["prompt"], 1)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        handler.state.jobs.pop(job.job_id)
        replacement = create_job(handler.state, job.job_id, user_id="other-owner")
        snapshot = asdict(replacement)
        release.set()
        await worker
        assert asdict(replacement) == snapshot
        assert handler.metrics.training_jobs_completed == 0
    finally:
        release.set()
        await worker


@pytest.mark.asyncio
async def test_successful_worker_keeps_existing_response_and_metrics_contract(handler):
    async def train_iteration(prompts):
        return {
            "trajectories_generated": 2,
            "average_reward": 0.8,
            "total_computation_used": 1.5,
        }

    handler.services["demo_engine"] = SimpleNamespace(train_iteration=train_iteration)
    response = await handler.start_training(
        GRPOTrainingRequest(prompts=["prompt"], num_iterations=2), "owner", "request"
    )
    assert response.status == "started"
    await handler.run_computational_training(response.job_id, ["prompt"], 2)
    status = handler.get_job_status(response.job_id, "owner")
    assert status.status == "completed"
    assert status.iterations_completed == 2
    assert status.total_trajectories == 4
    assert status.average_reward == 0.8
    assert status.computation_used == 3.0
    assert status.started_at is not None
    assert status.completed_at is not None
    assert handler.metrics.training_jobs_started == 1
    assert handler.metrics.training_jobs_completed == 1
    assert not handler.cancel_job(response.job_id, "owner")


@pytest.mark.parametrize("first", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("second", ["completed", "failed", "cancelled"])
def test_competing_terminal_transitions_commit_once(first, second):
    state = StateManager()
    job = create_job(state)
    assert state.transition_job(job.job_id, "running", expected_job=job)
    barrier = threading.Barrier(2)

    def finish(status):
        barrier.wait(timeout=2)
        return state.transition_job(job.job_id, status, expected_job=job)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_result = executor.submit(finish, first)
        second_result = executor.submit(finish, second)
        committed = [first_result.result(timeout=2), second_result.result(timeout=2)]
    assert committed.count(True) == 1
    assert job.status == (first if committed[0] else second)
    assert job.is_terminal
    assert job.completed_at is not None


def test_live_job_id_cannot_be_recreated_or_reopened():
    state = StateManager()
    job = create_job(state)
    with pytest.raises(ValueError, match="already exists"):
        create_job(state)
    state.update_job(job.job_id, status="cancelled")
    with pytest.raises(ValueError, match="already exists"):
        create_job(state)
    assert state.get_job(job.job_id) is job
    assert job.status == "cancelled"


def test_cancellation_does_not_claim_success_when_completion_wins(handler, monkeypatch):
    job = create_job(handler.state)

    def authorized_after_completion(*args):
        handler.state.update_job(job.job_id, status="completed")
        return True

    monkeypatch.setattr(handler, "_can_access_job", authorized_after_completion)
    assert not handler.cancel_job(job.job_id, "owner")
    assert job.status == "completed"


def test_authorized_cancellation_cannot_cancel_a_replacement_owner(
    handler, monkeypatch
):
    job = create_job(handler.state)

    def authorize_old_owner(*args):
        handler.state.jobs.pop(job.job_id)
        create_job(handler.state, job.job_id, user_id="other-owner")
        return True

    monkeypatch.setattr(handler, "_can_access_job", authorize_old_owner)
    assert not handler.cancel_job(job.job_id, "owner")
    replacement = handler.state.get_job(job.job_id)
    assert replacement.user_id == "other-owner"
    assert replacement.status == "starting"


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["/api", "/v1"])
@pytest.mark.parametrize("outcome", ["completed", "expired"])
async def test_batch_cancel_reports_committed_outcome(monkeypatch, prefix, outcome):
    with _grpo_service_test_app() as fixture:
        training = fixture["training_handler"]
        job = create_job(training.state)

        def race_cancel(*args):
            if outcome == "completed":
                training.state.update_job(job.job_id, status="completed")
            else:
                training.state.jobs.pop(job.job_id)
            return False

        monkeypatch.setattr(training, "cancel_job", race_cancel)
        async with AsyncClient(
            transport=ASGITransport(app=fixture["app"]), base_url="http://testserver"
        ) as client:
            response = await client.post(
                f"{prefix}/batch/cancel",
                headers={"x-user-id": "owner"},
                json={"job_ids": [job.job_id]},
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["cancelled"] == []
        key = "already_completed" if outcome == "completed" else "not_found"
        assert payload[key] == [job.job_id]


@pytest.mark.asyncio
async def test_expired_job_cannot_be_resurrected_by_worker_results(
    handler, monkeypatch
):
    now = [100.0]
    monkeypatch.setattr(state_module, "time", SimpleNamespace(time=lambda: now[0]))
    handler.state.jobs.ttl_seconds = 1
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def train_iteration(prompts):
        calls.append(prompts)
        entered.set()
        await release.wait()
        return {"trajectories_generated": 2}

    handler.services["demo_engine"] = SimpleNamespace(train_iteration=train_iteration)
    job = create_job(handler.state)
    worker = asyncio.create_task(
        handler.run_computational_training(job.job_id, ["prompt"], 2)
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        now[0] += 2
        release.set()
        await worker
        assert handler.state.get_job(job.job_id) is None
        assert len(calls) == 1
        assert handler.metrics.training_jobs_completed == 0
        assert handler.metrics.training_jobs_failed == 0
    finally:
        release.set()
        await worker


@pytest.mark.asyncio
async def test_missing_job_never_starts_engine_work(handler):
    async def train_iteration(prompts):
        pytest.fail("A missing job cannot own engine work")

    handler.services["demo_engine"] = SimpleNamespace(train_iteration=train_iteration)
    await handler.run_computational_training("missing", ["prompt"], 1)
    assert handler.metrics.training_jobs_completed == 0
    assert handler.metrics.training_jobs_failed == 0


@pytest.mark.asyncio
async def test_missing_engine_commits_failure_once(handler):
    job = create_job(handler.state)
    await handler.run_computational_training(job.job_id, ["prompt"], 1)
    snapshot = asdict(job)
    await handler.run_distributed_training(job.job_id, ["prompt"], 1, {})
    assert job.status == "failed"
    assert job.error == "No training engine available"
    assert asdict(job) == snapshot
    assert handler.metrics.training_jobs_failed == 1


@pytest.mark.asyncio
async def test_shutdown_cleanup_preserves_a_completion_that_wins_the_race(
    handler, monkeypatch, caplog
):
    from stateset_agents.api.grpo import service
    from stateset_agents.core import multiturn_agent
    from stateset_agents.utils import cache, monitoring

    callbacks = {}

    async def execute_shutdown():
        await callbacks["cleanup_jobs"]()

    shutdown = SimpleNamespace(
        is_shutting_down=False,
        register_task=lambda name, phase, callback, **kwargs: callbacks.update(
            {name: callback}
        ),
        execute_shutdown=execute_shutdown,
    )
    monkeypatch.setattr(service, "get_shutdown_manager", lambda: shutdown)
    monkeypatch.setattr(service, "get_state_manager", lambda: handler.state)
    monkeypatch.setattr(
        service,
        "get_grpo_config",
        lambda: SimpleNamespace(
            validate=lambda: [], enable_prometheus=False, use_lightweight_engine=True
        ),
    )
    monkeypatch.setattr(service, "_services", {})
    monkeypatch.setattr(service, "_get_training_handler", lambda: handler)
    monkeypatch.setattr(service, "_get_conversation_handler", lambda: None)
    monkeypatch.setattr(service, "_get_websocket_handler", lambda: None)
    monkeypatch.setattr(cache, "CacheService", lambda: object())
    monkeypatch.setattr(monitoring, "MonitoringService", lambda **kwargs: object())
    monkeypatch.setattr(multiturn_agent, "MultiTurnAgent", lambda *a, **kw: object())
    active = create_job(handler.state, "active")
    raced = create_job(handler.state, "raced")
    completed = create_job(handler.state, "completed")
    handler.state.update_job(completed.job_id, status="completed")
    snapshot = asdict(completed)
    transition = handler.state.transition_job

    def complete_before_cancellation(job_id, status, **kwargs):
        if job_id == raced.job_id:
            assert transition(job_id, "completed", expected_job=raced)
        return transition(job_id, status, **kwargs)

    monkeypatch.setattr(handler.state, "transition_job", complete_before_cancellation)
    async with service.lifespan(SimpleNamespace()):
        pass
    assert active.status == "cancelled"
    assert raced.status == "completed"
    assert asdict(completed) == snapshot
    assert "Cancelled training job: raced" not in caplog.text
