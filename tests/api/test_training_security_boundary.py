"""Model and path policy survives durable admission and worker execution."""

import asyncio
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import HTTPException

from stateset_agents.api import config as api_config
from stateset_agents.api.schemas import TrainingRequest
from stateset_agents.api.services.training_service import TrainingService
from stateset_agents.api.training_worker import work_once


@pytest.fixture
def policy(monkeypatch, tmp_path):
    config = api_config.APIConfig()
    config.environment = api_config.Environment.DEVELOPMENT
    config.security.require_auth = True
    config.allow_external_models = False
    config.training_output_root = str(tmp_path / "training")
    monkeypatch.setattr(api_config, "_config", config)
    return config


def request(**changes):
    return TrainingRequest(
        agent_config={"model_name": "approved/model"},
        environment_scenarios=[{"id": "case"}],
        reward_config={},
        num_episodes=1,
        **changes,
    )


def capture_worker(monkeypatch):
    calls = []

    async def run(service, job_id, agent, environment, reward, accepted, cancel):
        calls.append((agent.config, accepted.model_copy(deep=True)))
        service.training_jobs[job_id].update(
            status="completed", completed_at=datetime.utcnow()
        )

    monkeypatch.setattr(TrainingService, "_run_training", run)
    return calls


@pytest.mark.parametrize("durable", [False, True])
@pytest.mark.parametrize("refusal", ["disabled", "unlisted", "role"])
async def test_refused_model_never_admits_or_constructs_a_job(
    policy, tmp_path, durable, refusal, monkeypatch
):
    policy.environment = api_config.Environment.PRODUCTION
    policy.allow_external_models = refusal != "disabled"
    policy.model_allowlist = ["different/model"] if refusal == "unlisted" else []
    calls = capture_worker(monkeypatch)
    import stateset_agents.api.services.training_service as service_module

    def forbidden_constructor(config):
        raise AssertionError("A refused model reached agent construction")

    monkeypatch.setattr(service_module, "MultiTurnAgent", forbidden_constructor)
    api = TrainingService(queue_path=tmp_path / "jobs.sqlite3" if durable else None)
    with pytest.raises(HTTPException) as error:
        await api.start_training(
            request(), "alice", roles=["user"] if refusal == "role" else ["trainer"]
        )
    assert error.value.status_code == 403
    assert not api.jobs and not api._tasks and not calls
    await api.aclose()


@pytest.mark.parametrize("durable", [False, True])
async def test_accepted_stub_revision_and_server_output_survive_execution(
    policy, tmp_path, durable, monkeypatch
):
    calls = capture_worker(monkeypatch)
    api = TrainingService(queue_path=tmp_path / "jobs.sqlite3" if durable else None)
    revision = "a" * 40
    original = request(training_config_overrides={"model_revision": revision})
    frozen = original.model_dump()
    job_id = await api.start_training(original, "alice", roles=["trainer"])
    original.agent_config.model_name = "/caller/mutation"
    if durable:
        assert await work_once(api.queue)
    else:
        await asyncio.gather(*tuple(api._tasks.values()))
    assert len(calls) == 1
    agent_config, effective = calls[0]
    assert agent_config.use_stub_model is True
    assert agent_config.model_name == "approved/model"
    assert agent_config.model_revision == revision
    destination = Path(effective.training_config_overrides["output_dir"])
    expected = Path(policy.training_output_root) / job_id
    assert destination.parent == expected if durable else destination == expected
    job = api.get_training_status(job_id, "alice")
    assert job["status"] == "completed" and job["config"] == frozen
    assert job["output_dir"] == str(expected)
    await api.aclose()


async def test_worker_rechecks_current_model_policy(policy, tmp_path, monkeypatch):
    calls = capture_worker(monkeypatch)
    api = TrainingService(queue_path=tmp_path / "jobs.sqlite3")
    policy.environment = api_config.Environment.PRODUCTION
    policy.allow_external_models = True
    policy.model_allowlist = ["approved/model"]
    job_id = await api.start_training(request(), "alice", roles=["trainer"])
    policy.allow_external_models = False
    assert await work_once(api.queue)
    assert api.get_training_status(job_id)["status"] == "failed"
    assert not calls
    await api.aclose()


@pytest.mark.parametrize("durable", [False, True])
async def test_resume_paths_enforce_ownership_and_keep_admin_access(
    policy, tmp_path, durable, monkeypatch
):
    calls = capture_worker(monkeypatch)
    api = TrainingService(queue_path=tmp_path / "jobs.sqlite3" if durable else None)
    first = await api.start_training(request(), "alice", roles=["trainer"])
    if durable:
        await work_once(api.queue)
    else:
        await asyncio.gather(*tuple(api._tasks.values()))
    for value in (
        first,
        f"{first}/checkpoint-1",
        str(Path(policy.training_output_root) / first),
    ):
        with pytest.raises(ValueError, match="accessible"):
            await api.start_training(
                request(resume_from_checkpoint=value), "bob", roles=["trainer"]
            )
    nested = f"{first}/checkpoint-1"
    (Path(policy.training_output_root) / nested).mkdir(parents=True)
    second = await api.start_training(
        request(resume_from_checkpoint=nested), "alice", roles=["trainer"]
    )
    if durable:
        await work_once(api.queue)
    else:
        await asyncio.gather(*tuple(api._tasks.values()))
    assert api.get_training_status(second)["status"] == "completed"
    third = await api.start_training(
        request(resume_from_checkpoint=nested), "admin", roles=["admin"]
    )
    if durable:
        await work_once(api.queue)
    else:
        await asyncio.gather(*tuple(api._tasks.values()))
    assert api.get_training_status(third)["status"] == "completed"
    expected = str(Path(policy.training_output_root) / nested)
    assert [accepted.resume_from_checkpoint for _, accepted in calls[1:]] == [
        expected,
        expected,
    ]
    await api.aclose()


async def test_worker_refuses_different_output_root(policy, tmp_path, monkeypatch):
    calls = capture_worker(monkeypatch)
    api = TrainingService(queue_path=tmp_path / "jobs.sqlite3")
    job_id = api.queue.submit(
        request().model_dump(), "alice", output_root=str(tmp_path / "escape")
    )
    assert await work_once(api.queue)
    job = api.get_training_status(job_id)
    assert job["status"] == "failed"
    assert "configured root" in job["error"] and not calls
    assert not (tmp_path / "escape").exists()
    await api.aclose()


async def test_worker_rechecks_resume_symlinks_after_admission(
    policy, tmp_path, monkeypatch
):
    calls = capture_worker(monkeypatch)
    api = TrainingService(queue_path=tmp_path / "jobs.sqlite3")
    first = await api.start_training(request(), "alice", roles=["trainer"])
    await work_once(api.queue)
    checkpoint = Path(policy.training_output_root) / first / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    second = await api.start_training(
        request(resume_from_checkpoint=str(checkpoint)), "alice", roles=["trainer"]
    )
    checkpoint.rmdir()
    outside = tmp_path / "outside-checkpoint"
    outside.mkdir()
    try:
        checkpoint.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    assert await work_once(api.queue)
    job = api.get_training_status(second)
    assert job["status"] == "failed" and "output root" in job["error"]
    assert len(calls) == 1
    await api.aclose()


@pytest.mark.parametrize("durable", [False, True])
async def test_resume_symlink_cannot_escape_training_root(
    policy, tmp_path, durable, monkeypatch
):
    capture_worker(monkeypatch)
    api = TrainingService(queue_path=tmp_path / "jobs.sqlite3" if durable else None)
    job_id = await api.start_training(request(), "alice", roles=["trainer"])
    if durable:
        await work_once(api.queue)
    else:
        await asyncio.gather(*tuple(api._tasks.values()))
    directory = Path(policy.training_output_root) / job_id
    directory.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = directory / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    with pytest.raises(ValueError, match="output root"):
        await api.start_training(
            request(resume_from_checkpoint=str(link)), "alice", roles=["trainer"]
        )
    await api.aclose()
