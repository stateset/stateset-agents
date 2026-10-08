"""River executor ownership, cleanup ordering, and submission isolation."""

from __future__ import annotations

import sys
import threading
from asyncio import CancelledError
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import JobHandle, JobStatus
from stateset_agents.remote.river import RiverExecutor
from tests.unit.test_river_submit_golden import stub_river_renderers  # noqa: F401
from tests.unit.test_river_submit_golden import (
    FakeTokenizer,
    RecordingClient,
    scenario_episode_harvest,
    scenario_episode_rl,
    scenario_harvest,
    scenario_rl,
    scenario_sft,
)

SCENARIOS = (
    scenario_sft,
    scenario_harvest,
    scenario_episode_harvest,
    scenario_rl,
    scenario_episode_rl,
)


@pytest.fixture
def install_client(monkeypatch):
    """Record construction and close without replacing the training loop."""
    monkeypatch.setenv("RIVER_API_KEY", "rv_test")

    def install(client, *, close_error=None):
        def close():
            client.calls.append({"call": "client_close"})
            if close_error is not None:
                raise close_error

        def construct(**kwargs):
            client.calls.append({"call": "client_construct"})
            return client

        monkeypatch.setattr(client, "close", close, raising=False)
        monkeypatch.setattr(
            sys.modules["river_client"], "Client", construct, raising=False
        )
        return client

    return install


def executor_for(tmp_path, **kwargs):
    return RiverExecutor(
        tokenizer=FakeTokenizer(), ledger_path=tmp_path / "ledger.jsonl", **kwargs
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("injected", [False, True])
def test_client_ownership_and_session_cleanup_order(
    scenario, injected, install_client, tmp_path
):
    spec, client = scenario(tmp_path)
    install_client(client)
    executor = executor_for(tmp_path, **({"client": client} if injected else {}))
    handle = executor.submit(spec)
    assert executor.status(handle) is JobStatus.SUCCEEDED
    calls = [call["call"] for call in client.calls]
    assert "session_close" in calls
    if injected:
        assert "client_construct" not in calls
        assert "client_close" not in calls
        assert executor._client is client
    else:
        assert calls[0] == "client_construct"
        assert calls[-2:] == ["session_close", "client_close"]
        assert calls.count("client_close") == 1
        assert executor._client is None
    # Polling and artifact publication must remain usable after client close.
    assert executor.wait(handle).status is JobStatus.SUCCEEDED
    assert spec.output_dir.exists()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_dry_run_never_constructs_client(scenario, install_client, tmp_path):
    spec, client = scenario(tmp_path)
    install_client(client)
    executor = executor_for(tmp_path)
    handle = executor.submit(replace(spec, dry_run=True))
    assert executor.status(handle) is JobStatus.SUCCEEDED
    assert client.calls == []


@pytest.mark.parametrize("stage", ["capabilities", "tokenizer", "dataset", "training"])
def test_failure_closes_owned_client_and_leaves_terminal_status(
    stage, install_client, tmp_path, monkeypatch
):
    spec, client = scenario_sft(tmp_path)
    install_client(client)
    executor = executor_for(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("original failure")

    if stage == "capabilities":
        monkeypatch.setattr(client, "get_capabilities", fail)
    elif stage == "tokenizer":
        monkeypatch.setattr(executor, "_get_tokenizer", fail)
    elif stage == "dataset":
        # Simulate input disappearing after RemoteJobSpec's existence check.
        spec.dataset.unlink()
    else:
        monkeypatch.setattr(client._model_cls, "train_step", fail)
    expected = {
        "capabilities": RemoteExecutionError,
        "tokenizer": RuntimeError,
        "dataset": FileNotFoundError,
        "training": RemoteExecutionError,
    }
    with pytest.raises(expected[stage]):
        executor.submit(spec)
    assert executor.status(JobHandle("river", "river-1")) is JobStatus.FAILED
    assert client.calls[-1] == {"call": "client_close"}
    assert executor._client is None
    if stage == "training":
        assert client.calls[-2] == {"call": "session_close"}


def test_empty_dataset_closes_client_without_opening_session(install_client, tmp_path):
    spec, client = scenario_sft(tmp_path)
    spec.dataset.write_text("")
    install_client(client)
    executor = executor_for(tmp_path)
    handle = executor.submit(spec)
    assert executor.status(handle) is JobStatus.FAILED
    assert client.calls == [
        {"call": "client_construct"},
        {"call": "get_capabilities"},
        {"call": "client_close"},
    ]


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, CancelledError])
def test_interruption_closes_session_then_client(
    scenario, error_type, install_client, tmp_path, monkeypatch
):
    spec, client = scenario(tmp_path)
    install_client(client)

    def fail(*args, **kwargs):
        raise error_type()

    monkeypatch.setattr(client._model_cls, "train_step", fail)
    monkeypatch.setattr(client._model_cls, "sample", fail)
    executor = executor_for(tmp_path)
    with pytest.raises(error_type):
        executor.submit(spec)
    assert executor.status(JobHandle("river", "river-1")) is JobStatus.CANCELLED
    assert client.calls[-2:] == [{"call": "session_close"}, {"call": "client_close"}]
    assert executor._client is None


def test_cleanup_failure_preserves_training_error(
    install_client, tmp_path, monkeypatch, caplog
):
    spec, client = scenario_sft(tmp_path)
    install_client(client, close_error=RuntimeError("Authorization: rv_secret"))
    original = ValueError("tokenizer failed")

    def fail(*args):
        raise original

    executor = executor_for(tmp_path)
    monkeypatch.setattr(executor, "_get_tokenizer", fail)
    with pytest.raises(ValueError) as caught:
        executor.submit(spec)
    assert caught.value is original
    assert executor.status(JobHandle("river", "river-1")) is JobStatus.FAILED
    assert "client cleanup failed" in caplog.text
    assert "rv_secret" not in caplog.text


def test_cleanup_failure_preserves_completed_checkpoint(
    install_client, tmp_path, caplog
):
    spec, client = scenario_sft(tmp_path)
    install_client(client, close_error=RuntimeError("Authorization: rv_secret"))
    executor = executor_for(tmp_path)
    with pytest.raises(RemoteExecutionError, match="client cleanup failed") as caught:
        executor.submit(spec)
    assert caught.value.context.details["stage"] == "client_cleanup"
    handle = JobHandle("river", caught.value.context.details["job_id"])
    assert executor.status(handle) is JobStatus.SUCCEEDED
    assert (executor.fetch(handle) / "river_checkpoint.json").exists()
    assert executor._client is None
    assert "rv_secret" not in str(caught.value)
    assert "rv_secret" not in caplog.text
    assert "rv_secret" not in "\n".join(executor.logs(handle))


def test_subsequent_submission_constructs_a_new_client(install_client, tmp_path):
    spec, first = scenario_sft(tmp_path)
    install_client(first)
    executor = executor_for(tmp_path)
    executor.submit(spec)
    calls_after_close = list(first.calls)
    second = install_client(RecordingClient())
    handle = executor.submit(replace(spec, output_dir=tmp_path / "second"))
    assert executor.status(handle) is JobStatus.SUCCEEDED
    assert first.calls == calls_after_close
    assert second.calls[0] == {"call": "client_construct"}
    assert second.calls[-1] == {"call": "client_close"}


def test_submission_after_cleanup_failure_does_not_reuse_broken_client(
    install_client, tmp_path
):
    spec, first = scenario_sft(tmp_path)
    install_client(first, close_error=RuntimeError("cannot close"))
    executor = executor_for(tmp_path)
    with pytest.raises(RemoteExecutionError, match="client cleanup failed"):
        executor.submit(spec)
    calls_after_close = list(first.calls)
    second = install_client(RecordingClient())
    handle = executor.submit(replace(spec, output_dir=tmp_path / "second"))
    assert executor.status(handle) is JobStatus.SUCCEEDED
    assert first.calls == calls_after_close
    assert second.calls[-1] == {"call": "client_close"}


def test_completed_rl_resume_does_not_reopen_client(install_client, tmp_path):
    spec, client = scenario_rl(tmp_path)
    install_client(client)
    executor = executor_for(tmp_path)
    executor.submit(spec)
    calls_after_close = list(client.calls)
    handle = executor.submit(replace(spec, resume=True))
    assert executor.status(handle) is JobStatus.SUCCEEDED
    assert client.calls == calls_after_close


def test_concurrent_submission_cannot_close_another_jobs_client(
    install_client, tmp_path, monkeypatch
):
    spec, client = scenario_sft(tmp_path)
    install_client(client)
    executor = executor_for(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = client.get_capabilities

    def wait_for_release():
        entered.set()
        assert release.wait(10), "test did not release the active submission"
        return original()

    monkeypatch.setattr(client, "get_capabilities", wait_for_release)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(executor.submit, spec)
        try:
            assert entered.wait(10), "submission did not reach capability check"
            with pytest.raises(RemoteExecutionError, match="active submission"):
                executor.submit(spec)
            assert executor._counter == 1
            assert {"call": "client_close"} not in client.calls
        finally:
            release.set()
        handle = future.result(timeout=10)
    assert executor.status(handle) is JobStatus.SUCCEEDED
    assert client.calls[-1] == {"call": "client_close"}
