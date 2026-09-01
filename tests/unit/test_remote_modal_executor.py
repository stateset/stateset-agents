"""Tests for ``ModalExecutor``, driven by a behavioural fake of the SDK.

The fake (``fake_modal``) really executes the registered function and really
stores what it writes, so these assert on effects — an adapter arriving on
local disk, a failure being reported as a failure — not on recorded call
kwargs. An earlier version of this executor passed a suite of kwargs
assertions while reporting SUCCEEDED without running anything; that is the
regression this file exists to prevent.

Modal's real network transport is still not exercised here. What is exercised
is every decision the executor makes around it.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec
from tests.unit import fake_modal


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "curated.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"q{i}"},
                        {"role": "assistant", "content": f"a{i}"},
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
        output_dir=tmp_path / "local_out",
        gpu="A100",
        timeout_s=120,
        package_version="0.19.0",
    )


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    module = fake_modal.build(tmp_path / "volumes")
    monkeypatch.setitem(sys.modules, "modal", module)
    return module


@pytest.fixture
def executor(sdk, monkeypatch, tmp_path):
    from stateset_agents.remote import modal as modal_mod

    monkeypatch.setattr(modal_mod, "MODAL_AVAILABLE", True)
    return modal_mod.ModalExecutor(remote_mount="/outputs-test")


@pytest.fixture
def trains_for_real(monkeypatch):
    """Make the job behave as if a GPU were present, writing a real adapter.

    ``run_sft`` itself needs a GPU and is out of scope; what matters here is
    that whatever the job produces actually reaches the caller.
    """
    from stateset_agents.training import sft

    def fake_run_sft(*, output_dir, base_model, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": base_model, "r": kwargs["lora_r"]})
        )
        (output_dir / "adapter_model.safetensors").write_bytes(b"WEIGHTS")
        return output_dir

    monkeypatch.setattr(sft, "gpu_available", lambda: True)
    monkeypatch.setattr(sft, "run_sft", fake_run_sft)


class TestMissingSdk:
    def test_submit_without_the_sdk_names_the_extra(self, spec, monkeypatch):
        from stateset_agents.remote import modal as modal_mod

        monkeypatch.setattr(modal_mod, "MODAL_AVAILABLE", False)

        with pytest.raises(RemoteExecutionError, match=r"\[modal\]"):
            modal_mod.ModalExecutor().submit(spec)


class TestImage:
    def test_installs_the_pinned_published_package(self, executor, spec):
        executor.submit(spec)

        assert (
            "stateset-agents[training]==0.19.0" in executor.build_image(spec).installed
        )

    def test_falls_back_to_the_running_version_when_unpinned(
        self, executor, dataset, tmp_path
    ):
        spec = RemoteJobSpec(
            dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", package_version=None
        )

        image = executor.build_image(spec)

        assert any(
            pkg.startswith("stateset-agents[training]==") for pkg in image.installed
        )

    def test_does_not_sync_the_working_tree(self, executor, spec):
        executor.submit(spec)

        assert executor.build_image(spec).local_dirs_added == []


class TestResourceWiring:
    def test_requests_the_configured_gpu_and_timeout(self, executor, spec, sdk):
        executor.submit(spec)

        registered = sdk.apps[-1].functions[-1]
        assert registered.kwargs["gpu"] == "A100"
        assert registered.kwargs["timeout"] == 120

    def test_mounts_a_volume_for_the_adapter(self, executor, spec, sdk):
        executor.submit(spec)

        registered = sdk.apps[-1].functions[-1]
        assert registered.kwargs["volumes"]
        assert registered.kwargs["serialized"] is True

    def test_uses_current_modal_default_gpu(self, executor, dataset, tmp_path, sdk):
        default_spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            output_dir=tmp_path / "out",
            dry_run=True,
        )

        executor.submit(default_spec)

        assert sdk.apps[-1].functions[-1].kwargs["gpu"] == "A10"

    def test_renders_multi_gpu_request(self, executor, dataset, tmp_path, sdk):
        multi_gpu = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            output_dir=tmp_path / "out",
            gpu="H100",
            gpu_count=2,
            dry_run=True,
        )

        executor.submit(multi_gpu)

        assert sdk.apps[-1].functions[-1].kwargs["gpu"] == "H100:2"

    def test_named_secrets_and_region_are_provider_side_references(
        self, executor, sdk, dataset, tmp_path
    ):
        configured = type(executor)(
            remote_mount="/outputs-test",
            secret_names=["huggingface", "weights-and-biases"],
            region="us-east",
        )
        configured.submit(
            RemoteJobSpec(
                dataset=dataset,
                base_model="Qwen/Qwen3.5-0.8B",
                output_dir=tmp_path / "out",
                dry_run=True,
            )
        )

        options = sdk.apps[-1].functions[-1].kwargs
        assert [secret.name for secret in options["secrets"]] == [
            "huggingface",
            "weights-and-biases",
        ]
        assert options["region"] == "us-east"

    def test_provider_configuration_can_come_from_environment(
        self, executor, tmp_path, monkeypatch
    ):
        monkeypatch.setenv(
            "STATESET_MODAL_SECRET_NAMES", "huggingface, weights-and-biases"
        )
        monkeypatch.setenv("STATESET_MODAL_REGION", "eu-west")

        configured = type(executor)(remote_mount="/outputs-test")

        assert configured.secret_names == ("huggingface", "weights-and-biases")
        assert configured.region == "eu-west"

    def test_dataset_is_uploaded_and_payload_uses_the_mounted_path(
        self, executor, spec, sdk, trains_for_real
    ):
        result = executor.wait(executor.submit(spec))

        payload = sdk.apps[-1].calls[-1][0][0]
        assert payload["dataset"].startswith(f"{executor.remote_mount}/inputs/")
        assert payload["dataset"].endswith("/curated.jsonl")
        assert payload["dataset"] != str(spec.dataset)
        assert result.status is JobStatus.SUCCEEDED
        assert any(payload["dataset"] in line for line in result.logs)

    def test_remote_function_commits_the_output_volume(
        self, executor, spec, sdk, trains_for_real
    ):
        executor.submit(spec)

        volume = next(iter(sdk.apps[-1].functions[-1].kwargs["volumes"].values()))
        assert volume.commit_count == 1

    def test_rejects_gpu_count_encoded_twice(self, executor, dataset, tmp_path):
        ambiguous = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            output_dir=tmp_path / "out",
            gpu="H100:2",
            gpu_count=2,
            dry_run=True,
        )

        with pytest.raises(RemoteExecutionError, match="not both"):
            executor.submit(ambiguous)

        assert not fake_modal.FakeVolume._instances

    def test_download_rejects_artifact_path_traversal(self, executor, tmp_path):
        class UnsafeVolume:
            def reload(self):
                return None

            def iterdir(self, *_args, **_kwargs):
                return [SimpleNamespace(path="outputs/job/../../escape")]

        with pytest.raises(RemoteExecutionError, match="unsafe artifact path"):
            executor._download(UnsafeVolume(), "outputs/job", tmp_path / "out")


class TestSuccessfulJob:
    def test_the_adapter_actually_arrives_on_local_disk(
        self, executor, spec, trains_for_real
    ):
        """The whole point of the feature. Not a kwargs assertion."""
        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.SUCCEEDED
        assert (spec.output_dir / "adapter_config.json").exists()
        assert (
            spec.output_dir / "adapter_model.safetensors"
        ).read_bytes() == b"WEIGHTS"
        assert not (spec.output_dir / "curated.jsonl").exists()

    def test_adapter_contents_survive_the_round_trip(
        self, executor, spec, trains_for_real
    ):
        executor.wait(executor.submit(spec))

        config = json.loads((spec.output_dir / "adapter_config.json").read_text())
        assert config["base_model_name_or_path"] == "Qwen/Qwen3.5-0.8B"
        assert config["r"] == 16

    def test_remote_logs_are_returned(self, executor, spec, trains_for_real):
        result = executor.wait(executor.submit(spec))

        assert any("Loaded 3 examples" in line for line in result.logs)

    def test_executor_owned_volume_is_deleted(self, executor, spec, trains_for_real):
        executor.wait(executor.submit(spec))

        assert fake_modal.FakeVolume.deleted_names
        assert not fake_modal.FakeVolume._instances

    def test_eval_prompts_reach_the_training_job(
        self, executor, spec, trains_for_real, monkeypatch
    ):
        """Modal ships the whole spec dict as the payload, so a job-level
        field must arrive at run_sft unmodified."""
        from stateset_agents.training import sft

        received = {}
        original = sft.run_sft

        def spy(**kwargs):
            received["eval_prompts"] = kwargs.get("eval_prompts")
            return original(**kwargs)

        monkeypatch.setattr(sft, "run_sft", spy)
        spec.eval_prompts = ["what's the return policy?"]

        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.SUCCEEDED
        assert received["eval_prompts"] == ["what's the return policy?"]


class TestFailingJob:
    def test_directory_dataset_is_rejected_before_allocation(self, executor, tmp_path):
        with pytest.raises(RemoteExecutionError, match="regular file"):
            executor.submit(
                RemoteJobSpec(
                    dataset=tmp_path,
                    base_model="Qwen/Qwen3.5-0.8B",
                    output_dir=tmp_path / "out",
                )
            )

        assert not fake_modal.FakeVolume._instances

    def test_empty_dataset_reports_failure_not_success(self, executor, tmp_path):
        """The regression guard: no SUCCEEDED without work."""
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        spec = RemoteJobSpec(
            dataset=empty,
            base_model="Qwen/Qwen3.5-0.8B",
            output_dir=tmp_path / "out",
        )

        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.FAILED
        assert not result.succeeded
        assert result.output_dir is None
        assert not fake_modal.FakeVolume._instances

    def test_fetch_after_failure_is_an_error(self, executor, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        spec = RemoteJobSpec(
            dataset=empty,
            base_model="Qwen/Qwen3.5-0.8B",
            output_dir=tmp_path / "out",
        )
        handle = executor.submit(spec)

        with pytest.raises(RemoteExecutionError, match="not finished"):
            executor.fetch(handle)

    def test_failure_artifacts_are_recovered_before_cleanup(
        self, executor, spec, trains_for_real, monkeypatch
    ):
        from stateset_agents.training import sft

        monkeypatch.setattr(sft, "_apply_eval_gate", lambda *_args, **_kwargs: 1)

        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.FAILED
        assert result.output_dir == spec.output_dir
        assert (spec.output_dir / "adapter_model.safetensors").exists()
        assert not fake_modal.FakeVolume._instances

    def test_cleanup_failure_fails_job_without_hiding_downloaded_adapter(
        self, executor, spec, trains_for_real, monkeypatch
    ):
        monkeypatch.setattr(
            executor,
            "_delete_volume",
            lambda _name: (_ for _ in ()).throw(RuntimeError("cleanup denied")),
        )

        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.FAILED
        assert result.output_dir == spec.output_dir
        assert (spec.output_dir / "adapter_model.safetensors").exists()
        assert any("cleanup denied" in line for line in result.logs)

    def test_a_job_producing_no_artifacts_does_not_report_success(
        self, executor, spec, monkeypatch
    ):
        """A silent empty result is the failure mode that shipped once already."""
        from stateset_agents.training import sft

        monkeypatch.setattr(sft, "gpu_available", lambda: True)
        monkeypatch.setattr(sft, "run_sft", lambda **kwargs: kwargs["output_dir"])

        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.FAILED
        assert any("no artifacts" in line.lower() for line in result.logs)

    def test_provider_errors_are_wrapped_with_the_cause(
        self, executor, spec, monkeypatch
    ):
        boom = RuntimeError("modal is down")
        monkeypatch.setattr(
            executor, "_run_remote", lambda *a, **k: (_ for _ in ()).throw(boom)
        )

        with pytest.raises(RemoteExecutionError) as excinfo:
            executor.submit(spec)

        assert excinfo.value.cause is boom


class TestDryRun:
    def test_dry_run_succeeds_without_artifacts(self, executor, dataset, tmp_path):
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            output_dir=tmp_path / "out",
            dry_run=True,
        )

        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.SUCCEEDED
        assert any("SFT Training Plan" in line for line in result.logs)
        assert not fake_modal.FakeVolume._instances


class TestHandles:
    def test_unknown_handle_is_an_error(self, executor):
        with pytest.raises(RemoteExecutionError, match="unknown job"):
            executor.status(JobHandle(provider="modal", job_id="nope"))

    def test_handle_carries_the_provider(self, executor, spec, trains_for_real):
        assert executor.submit(spec).provider == "modal"
