"""
Tests for TrainingService, JobProgressCallback, and training DI.

Covers:
- JobProgressCallback episode/progress/metrics updates
- JobProgressCallback cancellation via asyncio.Event
- get_training_service dependency injection
- TrainingService.cancel_training lifecycle
- train() callbacks parameter wiring
"""

import asyncio
from typing import Any

from fastapi import Depends, FastAPI

from stateset_agents.api.dependencies import get_training_service
from stateset_agents.api.services.training_service import (
    JobProgressCallback,
    TrainingService,
)
from tests.api.asgi_client import SyncASGIClient


# ============================================================================
# JobProgressCallback tests
# ============================================================================


class TestJobProgressCallback:
    """Tests for the training progress callback."""

    def _make_job(self, total: int = 10) -> tuple[dict[str, Any], asyncio.Event]:
        job: dict[str, Any] = {
            "current_episode": 0,
            "progress": 0.0,
            "metrics": {},
        }
        return job, asyncio.Event()

    def test_episode_updates_progress(self):
        job, event = self._make_job(total=10)
        cb = JobProgressCallback(job=job, total_episodes=10, cancel_event=event)

        cb.on_episode_end(0, {"loss": 0.5})
        assert job["current_episode"] == 1
        assert job["progress"] == 10.0

        cb.on_episode_end(4, {"loss": 0.3})
        assert job["current_episode"] == 5
        assert job["progress"] == 50.0

        cb.on_episode_end(9, {"loss": 0.1})
        assert job["current_episode"] == 10
        assert job["progress"] == 100.0

    def test_metrics_filters_non_scalar(self):
        job, event = self._make_job()
        cb = JobProgressCallback(job=job, total_episodes=5, cancel_event=event)

        cb.on_episode_end(
            0,
            {
                "loss": 0.5,
                "reward": 1.2,
                "name": "episode-0",
                "converged": True,
                "tensor": object(),  # non-scalar — should be filtered
                "nested": {"a": 1},  # non-scalar — should be filtered
            },
        )

        assert job["metrics"] == {
            "loss": 0.5,
            "reward": 1.2,
            "name": "episode-0",
            "converged": True,
        }

    def test_cancel_event_raises(self):
        job, event = self._make_job()
        cb = JobProgressCallback(job=job, total_episodes=5, cancel_event=event)

        # First call is fine
        cb.on_episode_end(0, {})
        assert job["current_episode"] == 1

        # Set cancel event
        event.set()

        # Next call should raise CancelledError
        try:
            cb.on_episode_end(1, {})
            assert False, "Expected CancelledError"
        except asyncio.CancelledError:
            pass

        # Episode was still updated before the raise
        assert job["current_episode"] == 2

    def test_progress_with_single_episode(self):
        job, event = self._make_job()
        cb = JobProgressCallback(job=job, total_episodes=1, cancel_event=event)

        cb.on_episode_end(0, {"loss": 0.1})
        assert job["progress"] == 100.0
        assert job["current_episode"] == 1


# ============================================================================
# get_training_service DI tests
# ============================================================================


class TestGetTrainingService:
    """Tests for the training service dependency injection."""

    def test_returns_service_from_app_state(self):
        app = FastAPI()
        expected = TrainingService()
        app.state.training_service = expected

        @app.get("/test")
        def endpoint(svc=Depends(get_training_service)):
            return {"same": svc is expected}

        with SyncASGIClient(app) as client:
            resp = client.get("/test")
            assert resp.status_code == 200
            assert resp.json()["same"] is True

    def test_creates_service_lazily_if_missing(self):
        app = FastAPI()

        @app.get("/test")
        def endpoint(svc=Depends(get_training_service)):
            return {"has_jobs": isinstance(svc.jobs, dict)}

        with SyncASGIClient(app) as client:
            resp = client.get("/test")
            assert resp.status_code == 200
            assert resp.json()["has_jobs"] is True


# ============================================================================
# TrainingService lifecycle tests
# ============================================================================


class TestTrainingServiceLifecycle:
    """Tests for TrainingService cancel and status methods."""

    def test_cancel_unknown_job_is_noop(self):
        svc = TrainingService()
        # Should not raise
        svc.cancel_training("nonexistent")

    def test_cancel_sets_status(self):
        svc = TrainingService()
        svc.training_jobs["job-1"] = {
            "status": "running",
            "completed_at": None,
        }
        event = asyncio.Event()
        svc._cancel_events["job-1"] = event

        svc.cancel_training("job-1")

        assert svc.training_jobs["job-1"]["status"] == "cancelled"
        assert svc.training_jobs["job-1"]["completed_at"] is not None
        assert event.is_set()

    def test_get_training_status_returns_none_for_missing(self):
        svc = TrainingService()
        assert svc.get_training_status("missing") is None

    def test_get_training_status_returns_job(self):
        svc = TrainingService()
        svc.training_jobs["job-1"] = {"status": "running"}
        result = svc.get_training_status("job-1")
        assert result is not None
        assert result["status"] == "running"

    def test_jobs_alias(self):
        svc = TrainingService()
        assert svc.jobs is svc.training_jobs


# ============================================================================
# train() callbacks parameter tests
# ============================================================================


class TestTrainCallbacksParam:
    """Tests that train() accepts and wires the callbacks parameter."""

    def test_train_signature_accepts_callbacks(self):
        """Verify the callbacks parameter exists in train()."""
        import inspect

        from stateset_agents.training.train import train

        sig = inspect.signature(train)
        assert "callbacks" in sig.parameters
        param = sig.parameters["callbacks"]
        assert param.default is None
