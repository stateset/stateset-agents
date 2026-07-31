"""Tests for ``ModalExecutor`` against a mocked ``modal`` SDK.

Modal's real network path is NOT covered here — it is manually verified. What
these tests pin is everything we control: the guard when the SDK is missing,
the image the job runs in (pinned package, not a working-tree sync), the GPU
and timeout wiring, and the status/log/fetch contract.
"""

from __future__ import annotations

import json
import sys
import types
from unittest import mock

import pytest

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "curated.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
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
        gpu="A100",
        timeout_s=120,
        package_version="0.19.0",
    )


@pytest.fixture
def fake_modal(monkeypatch):
    """A stand-in ``modal`` module recording how it was driven."""
    module = types.ModuleType("modal")

    image = mock.MagicMock(name="Image")
    image.debian_slim.return_value = image
    image.pip_install.return_value = image
    module.Image = image

    module.App = mock.MagicMock(name="App")
    module.Volume = mock.MagicMock(name="Volume")

    monkeypatch.setitem(sys.modules, "modal", module)
    return module


@pytest.fixture
def executor(fake_modal):
    from stateset_agents.remote import modal as modal_mod

    monkey = mock.patch.object(modal_mod, "MODAL_AVAILABLE", True)
    monkey.start()
    try:
        yield modal_mod.ModalExecutor()
    finally:
        monkey.stop()


class TestMissingSdk:
    def test_submit_without_the_sdk_names_the_extra(self, spec, monkeypatch):
        from stateset_agents.remote import modal as modal_mod

        monkeypatch.setattr(modal_mod, "MODAL_AVAILABLE", False)

        with pytest.raises(RemoteExecutionError, match=r"\[modal\]"):
            modal_mod.ModalExecutor().submit(spec)


class TestImageConstruction:
    def test_installs_the_pinned_published_package(self, executor, spec):
        """Artifacts ship, not code — the remote env is a pinned PyPI install."""
        image = executor.build_image(spec)

        installed = [
            call.args[0]
            for call in image.pip_install.call_args_list
            if call.args
        ]
        assert any("stateset-agents[training]==0.19.0" == pkg for pkg in installed)

    def test_falls_back_to_the_running_version_when_unpinned(
        self, executor, dataset, tmp_path
    ):
        spec = RemoteJobSpec(
            dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", package_version=None
        )

        image = executor.build_image(spec)

        installed = [
            call.args[0] for call in image.pip_install.call_args_list if call.args
        ]
        assert any(pkg.startswith("stateset-agents[training]==") for pkg in installed)

    def test_does_not_sync_the_working_tree(self, executor, spec):
        """Working-tree sync is the main reason executors of this kind rot."""
        image = executor.build_image(spec)

        assert not image.add_local_dir.called
        assert not image.copy_local_dir.called


class TestSubmit:
    def test_handle_carries_the_provider_and_a_job_id(self, executor, spec):
        handle = executor.submit(spec)

        assert handle.provider == "modal"
        assert handle.job_id

    def test_requests_the_configured_gpu_and_timeout(self, executor, spec, fake_modal):
        executor.submit(spec)

        kwargs = executor.last_function_kwargs
        assert kwargs["gpu"] == "A100"
        assert kwargs["timeout"] == 120

    def test_provider_errors_are_wrapped_with_the_cause(
        self, executor, spec, monkeypatch
    ):
        boom = RuntimeError("modal is down")
        monkeypatch.setattr(
            executor, "_spawn", mock.Mock(side_effect=boom)
        )

        with pytest.raises(RemoteExecutionError) as excinfo:
            executor.submit(spec)

        assert excinfo.value.cause is boom


class TestStatusAndFetch:
    def test_unknown_handle_is_an_error(self, executor):
        with pytest.raises(RemoteExecutionError, match="unknown job"):
            executor.status(JobHandle(provider="modal", job_id="nope"))

    def test_successful_job_reports_succeeded(self, executor, spec):
        handle = executor.submit(spec)

        assert executor.status(handle) is JobStatus.SUCCEEDED

    def test_fetch_returns_the_local_output_dir(self, executor, spec):
        handle = executor.submit(spec)

        assert executor.fetch(handle) == spec.output_dir

    def test_fetch_before_success_is_an_error(self, executor, spec):
        handle = executor.submit(spec)
        executor.cancel(handle)

        with pytest.raises(RemoteExecutionError, match="not finished"):
            executor.fetch(handle)
