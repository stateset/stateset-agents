"""Tests for training-lab in-memory state bounds and task cancellation."""

import httpx
import pytest

from stateset_agents.api import config as api_config
from stateset_agents.api.routers import training_lab


@pytest.fixture
def preserve_api_config():
    prev = api_config._config
    yield
    api_config._config = prev


@pytest.fixture
def clean_lab_state():
    """Ensure module-level lab state is empty before/after each test."""
    training_lab._experiments.clear()
    training_lab._episodes.clear()
    training_lab._logs.clear()
    training_lab._simulators.clear()
    training_lab._running_tasks.clear()
    training_lab._metrics_subscribers.clear()
    yield
    for task in list(training_lab._running_tasks.values()):
        if not task.done():
            task.cancel()
    training_lab._experiments.clear()
    training_lab._episodes.clear()
    training_lab._logs.clear()
    training_lab._simulators.clear()
    training_lab._running_tasks.clear()
    training_lab._metrics_subscribers.clear()


def _client_for_app(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def _enable_lab_no_auth(monkeypatch):
    monkeypatch.setenv("API_REQUIRE_AUTH", "false")
    monkeypatch.setenv("API_ENABLE_TRAINING_LAB", "true")
    api_config.reload_config()


async def _create_experiment(client, name="exp"):
    payload = {
        "name": name,
        "description": "",
        "environment": {},
        "agent": {},
        "training": {},
    }
    return await client.post("/api/lab/experiments", json=payload)


async def test_experiment_creation_evicts_oldest_finished_when_at_cap(
    monkeypatch,
    preserve_api_config,
    clean_lab_state,
):
    _enable_lab_no_auth(monkeypatch)
    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        # Fill to MAX_EXPERIMENTS, all left in the default CREATED state
        # (CREATED counts as finished/evictable — never started).
        for i in range(training_lab.MAX_EXPERIMENTS):
            resp = await _create_experiment(client, name=f"exp-{i}")
            assert resp.status_code == 201

        assert len(training_lab._experiments) == training_lab.MAX_EXPERIMENTS
        oldest_id = next(iter(training_lab._experiments))

        resp = await _create_experiment(client, name="overflow")
        assert resp.status_code == 201
        assert len(training_lab._experiments) == training_lab.MAX_EXPERIMENTS
        assert oldest_id not in training_lab._experiments


async def test_experiment_creation_429_when_all_running(
    monkeypatch,
    preserve_api_config,
    clean_lab_state,
):
    _enable_lab_no_auth(monkeypatch)
    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        for i in range(training_lab.MAX_EXPERIMENTS):
            resp = await _create_experiment(client, name=f"exp-{i}")
            assert resp.status_code == 201

        # Mark every experiment as RUNNING so none are evictable.
        for exp in training_lab._experiments.values():
            exp["status"] = training_lab.ExperimentStatus.RUNNING.value

        resp = await _create_experiment(client, name="overflow")
        assert resp.status_code == 429
        assert len(training_lab._experiments) == training_lab.MAX_EXPERIMENTS


async def test_logs_trim_to_maxlen(monkeypatch, preserve_api_config, clean_lab_state):
    _enable_lab_no_auth(monkeypatch)
    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        resp = await _create_experiment(client, name="log-exp")
        exp_id = resp.json()["id"]

        for i in range(training_lab.MAX_LOGS_PER_EXPERIMENT + 50):
            training_lab._add_log(exp_id, "info", f"msg-{i}")

        assert len(training_lab._logs[exp_id]) == training_lab.MAX_LOGS_PER_EXPERIMENT


async def test_episodes_trim_to_maxlen(
    monkeypatch, preserve_api_config, clean_lab_state
):
    _enable_lab_no_auth(monkeypatch)
    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        resp = await _create_experiment(client, name="ep-exp")
        exp_id = resp.json()["id"]

        for i in range(training_lab.MAX_EPISODES_PER_EXPERIMENT + 50):
            training_lab._episodes[exp_id].append({"episode_num": i})

        assert (
            len(training_lab._episodes[exp_id])
            == training_lab.MAX_EPISODES_PER_EXPERIMENT
        )


async def test_stop_experiment_cancels_background_task(
    monkeypatch,
    preserve_api_config,
    clean_lab_state,
):
    _enable_lab_no_auth(monkeypatch)
    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        resp = await _create_experiment(client, name="stop-exp")
        exp_id = resp.json()["id"]

        start_resp = await client.post(f"/api/lab/experiments/{exp_id}/start")
        assert start_resp.status_code == 200

        task = training_lab._running_tasks[exp_id]
        assert not task.done()

        stop_resp = await client.post(f"/api/lab/experiments/{exp_id}/stop")
        assert stop_resp.status_code == 200

        # Give the event loop a beat to process the cancellation.
        import asyncio

        for _ in range(10):
            if task.done():
                break
            await asyncio.sleep(0.01)

        assert task.done()


async def test_delete_experiment_cancels_background_task(
    monkeypatch,
    preserve_api_config,
    clean_lab_state,
):
    _enable_lab_no_auth(monkeypatch)
    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        resp = await _create_experiment(client, name="del-exp")
        exp_id = resp.json()["id"]

        start_resp = await client.post(f"/api/lab/experiments/{exp_id}/start")
        assert start_resp.status_code == 200

        task = training_lab._running_tasks[exp_id]
        assert not task.done()

        del_resp = await client.delete(f"/api/lab/experiments/{exp_id}")
        assert del_resp.status_code == 200

        import asyncio

        for _ in range(10):
            if task.done():
                break
            await asyncio.sleep(0.01)

        assert task.done()


async def test_delete_experiment_removes_metrics_subscribers_entry(
    monkeypatch,
    preserve_api_config,
    clean_lab_state,
):
    _enable_lab_no_auth(monkeypatch)
    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        resp = await _create_experiment(client, name="subs-exp")
        exp_id = resp.json()["id"]
        training_lab._metrics_subscribers[exp_id] = []

        del_resp = await client.delete(f"/api/lab/experiments/{exp_id}")
        assert del_resp.status_code == 200

        assert exp_id not in training_lab._metrics_subscribers


async def test_eviction_removes_metrics_subscribers_entry(
    monkeypatch,
    preserve_api_config,
    clean_lab_state,
):
    _enable_lab_no_auth(monkeypatch)
    from stateset_agents.api.main import create_app

    app = create_app()
    async with _client_for_app(app) as client:
        for i in range(training_lab.MAX_EXPERIMENTS):
            resp = await _create_experiment(client, name=f"exp-{i}")
            assert resp.status_code == 201

        oldest_id = next(iter(training_lab._experiments))
        training_lab._metrics_subscribers[oldest_id] = []

        resp = await _create_experiment(client, name="overflow")
        assert resp.status_code == 201

        assert oldest_id not in training_lab._experiments
        assert oldest_id not in training_lab._metrics_subscribers
