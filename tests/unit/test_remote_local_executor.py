"""Tests for ``LocalExecutor`` — the reference implementation of the contract.

These run the real ``scripts/sft_from_curated.py`` in a subprocess. Because
that script prints its plan and exits 0 when no GPU is present, the full
submit -> poll -> fetch path is exercisable on CPU-only CI.
"""

from __future__ import annotations

import json

import pytest

from stateset_agents.core.errors import StateSetError
from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec
from stateset_agents.remote.local import LocalExecutor


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "curated.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"question {i}"},
                        {"role": "assistant", "content": f"answer {i}"},
                    ]
                }
            )
            for i in range(3)
        )
        + "\n"
    )
    return path


@pytest.fixture
def spec(dataset, tmp_path):
    return RemoteJobSpec(
        dataset=dataset,
        base_model="Qwen/Qwen3.5-0.8B",
        output_dir=tmp_path / "out",
        num_epochs=1,
        dry_run=True,
    )


class TestRemoteExecutionError:
    def test_is_a_stateset_error(self):
        """Consistent with the unified hierarchy — callers catch StateSetError."""
        assert issubclass(RemoteExecutionError, StateSetError)

    def test_preserves_the_wrapped_cause(self):
        original = RuntimeError("provider exploded")

        err = RemoteExecutionError.wrap(original, "submit failed", provider="local")

        assert err.cause is original
        assert "submit failed" in str(err)


class TestLocalExecutorContract:
    def test_implements_the_executor_interface(self):
        assert issubclass(LocalExecutor, RemoteExecutor)

    def test_handles_carry_the_provider_name(self, spec):
        handle = LocalExecutor().submit(spec)

        assert handle.provider == "local"


class TestLocalExecutorRun:
    def test_dry_run_job_succeeds_and_reports_the_training_plan(self, spec):
        executor = LocalExecutor()

        handle = executor.submit(spec)
        result = executor.wait(handle)

        assert result.status is JobStatus.SUCCEEDED
        assert result.succeeded
        combined = "\n".join(result.logs)
        assert "Qwen/Qwen3.5-0.8B" in combined

    def test_status_is_terminal_after_completion(self, spec):
        executor = LocalExecutor()

        handle = executor.submit(spec)
        executor.wait(handle)

        assert executor.status(handle).is_terminal

    def test_failed_job_reports_failed_status_and_keeps_logs(self, dataset, tmp_path):
        """An empty dataset makes sft_from_curated exit non-zero."""
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        spec = RemoteJobSpec(
            dataset=empty, base_model="Qwen/Qwen3.5-0.8B", dry_run=True
        )
        executor = LocalExecutor()

        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.FAILED
        assert not result.succeeded
        assert result.logs

    def test_fetch_before_completion_is_an_error(self, spec):
        executor = LocalExecutor()

        with pytest.raises(RemoteExecutionError, match="not finished"):
            executor.fetch(JobHandle(provider="local", job_id="never-submitted"))

    def test_unknown_handle_is_an_error(self):
        executor = LocalExecutor()

        with pytest.raises(RemoteExecutionError, match="unknown job"):
            executor.status(JobHandle(provider="local", job_id="bogus"))

    def test_timeout_marks_the_job_failed(self, spec, monkeypatch):
        spec.timeout_s = 1
        executor = LocalExecutor()
        monkeypatch.setattr(
            LocalExecutor, "_entrypoint_args", lambda self: ["-c", "import time;time.sleep(30)"]
        )

        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.FAILED
        assert any("timed out" in line for line in result.logs)
