"""Health results reflect measured checks and are isolated per application."""

import asyncio
import time

import pytest
from fastapi import FastAPI

from stateset_agents.api.resilience import HealthChecker, HealthStatus
from stateset_agents.api.routers import metrics
from tests.api.asgi_client import SyncASGIClient


def application(checker=None):
    app = FastAPI()
    if checker is not None:
        app.state.health_checker = checker
    app.include_router(metrics.router)
    return app


def test_outage_after_success_is_visible_and_application_scoped():
    checker = HealthChecker()
    state = {"healthy": True}
    checker.add_check("inference", lambda: state["healthy"])
    with SyncASGIClient(application(checker)) as client:
        assert client.get("/health").status_code == 200
        state["healthy"] = False
        failure = client.get("/health")
        assert failure.status_code == 503
        assert failure.json()["components"] == {"inference": "unhealthy"}
        with SyncASGIClient(application()) as other:
            assert other.get("/health").status_code == 503
        state["healthy"] = True
        assert client.get("/health").status_code == 200


@pytest.mark.parametrize("checker", [None, HealthChecker()])
def test_missing_checks_are_not_healthy_but_liveness_is_available(checker):
    with SyncASGIClient(application(checker)) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/healthz").status_code == 200


def test_probe_exception_and_timeout_report_unhealthy(monkeypatch):
    monkeypatch.setattr(metrics, "HEALTH_TIMEOUT_SECONDS", 0.01)
    checker = HealthChecker()

    async def slow():
        await asyncio.sleep(10)
        return True

    def broken():
        raise RuntimeError("dependency unavailable")

    checker.add_check("broken", broken)
    with SyncASGIClient(application(checker)) as client:
        assert client.get("/health").status_code == 503
        checker.remove_check("broken")
        checker.add_check("slow", slow)
        assert client.get("/health").status_code == 503


def test_blocking_probe_does_not_block_response_deadline(monkeypatch):
    monkeypatch.setattr(metrics, "HEALTH_TIMEOUT_SECONDS", 0.01)
    checker = HealthChecker()

    def slow_sync():
        time.sleep(0.2)
        return True

    checker.add_check("blocking", slow_sync)
    with SyncASGIClient(application(checker)) as client:
        assert client.get("/health").status_code == 503


@pytest.mark.asyncio
async def test_async_callable_check_is_awaited():
    class Probe:
        async def __call__(self):
            return False

    checker = HealthChecker()
    checker.add_check("async_object", Probe())
    assert (await checker.check("async_object")).status is HealthStatus.UNHEALTHY
    assert HealthChecker().overall_status is HealthStatus.UNHEALTHY
