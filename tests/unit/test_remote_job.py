"""Tests for ``stateset_agents.remote.job`` — the provider-agnostic job contract.

The spec mirrors ``scripts/sft_from_curated.py``'s argparse surface exactly.
These tests pin that correspondence so the two cannot silently drift.
"""

from __future__ import annotations

import json

import pytest

from stateset_agents.remote.job import (
    JobHandle,
    JobStatus,
    RemoteJobResult,
    RemoteJobSpec,
)


@pytest.fixture
def dataset(tmp_path):
    """A minimal chat-format JSONL, as ``improve`` would emit."""
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


class TestRemoteJobSpecDefaults:
    def test_defaults_match_sft_from_curated_argparse(self, dataset):
        spec = RemoteJobSpec(dataset=dataset, base_model="Qwen/Qwen3.5-0.8B")

        assert spec.num_epochs == 3
        assert spec.lora_r == 16
        assert spec.lora_alpha == 32
        assert spec.learning_rate == pytest.approx(2e-5)
        assert spec.max_length == 1024
        assert spec.per_device_batch_size == 2
        assert spec.gradient_accumulation_steps == 4
        assert spec.dry_run is False


class TestRemoteJobSpecValidation:
    def test_rejects_missing_dataset(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            RemoteJobSpec(
                dataset=tmp_path / "nope.jsonl", base_model="Qwen/Qwen3.5-0.8B"
            )

    def test_rejects_empty_base_model(self, dataset):
        with pytest.raises(ValueError, match="base_model"):
            RemoteJobSpec(dataset=dataset, base_model="  ")

    @pytest.mark.parametrize(
        "field,value",
        [
            ("num_epochs", 0),
            ("lora_r", 0),
            ("lora_alpha", 0),
            ("learning_rate", 0.0),
            ("max_length", 0),
            ("per_device_batch_size", 0),
            ("gradient_accumulation_steps", 0),
            ("timeout_s", 0),
        ],
    )
    def test_rejects_non_positive_hyperparameters(self, dataset, field, value):
        with pytest.raises(ValueError, match=field):
            RemoteJobSpec(
                dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", **{field: value}
            )


class TestRemoteJobSpecSerialization:
    def test_round_trips_through_json(self, dataset):
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            num_epochs=5,
            lora_r=8,
            gpu="A100",
        )

        restored = RemoteJobSpec.from_dict(json.loads(json.dumps(spec.to_dict())))

        assert restored == spec

    def test_serialized_spec_contains_no_secrets(self, dataset, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_supersecret")
        monkeypatch.setenv("MODAL_TOKEN_SECRET", "modal_supersecret")

        spec = RemoteJobSpec(dataset=dataset, base_model="Qwen/Qwen3.5-0.8B")

        blob = json.dumps(spec.to_dict())
        assert "supersecret" not in blob

    def test_to_cli_args_are_consumable_by_sft_from_curated(self, dataset):
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            num_epochs=1,
            dry_run=True,
        )

        args = spec.to_cli_args()

        assert "--dataset" in args
        assert str(dataset) in args
        assert "--base-model" in args
        assert "Qwen/Qwen3.5-0.8B" in args
        assert "--num-epochs" in args
        assert "1" in args
        assert "--dry-run" in args

    def test_dry_run_flag_omitted_when_false(self, dataset):
        spec = RemoteJobSpec(dataset=dataset, base_model="Qwen/Qwen3.5-0.8B")

        assert "--dry-run" not in spec.to_cli_args()

    def test_resource_fields_are_not_passed_to_the_training_script(self, dataset):
        """gpu/timeout/package_version configure the provider, not the job."""
        spec = RemoteJobSpec(
            dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", gpu="A100"
        )

        args = spec.to_cli_args()

        assert "--gpu" not in args
        assert "A100" not in args


class TestJobHandle:
    def test_round_trips_through_json(self):
        handle = JobHandle(provider="modal", job_id="fc-123")

        assert JobHandle.from_dict(json.loads(json.dumps(handle.to_dict()))) == handle


class TestJobStatus:
    def test_terminal_states_are_terminal(self):
        assert JobStatus.SUCCEEDED.is_terminal
        assert JobStatus.FAILED.is_terminal
        assert JobStatus.CANCELLED.is_terminal

    def test_pending_and_running_are_not_terminal(self):
        assert not JobStatus.PENDING.is_terminal
        assert not JobStatus.RUNNING.is_terminal


class TestRemoteJobResult:
    def test_succeeded_result_is_successful(self, tmp_path):
        result = RemoteJobResult(
            handle=JobHandle(provider="local", job_id="1"),
            status=JobStatus.SUCCEEDED,
            output_dir=tmp_path,
            logs=["done"],
        )

        assert result.succeeded is True

    def test_failed_result_is_not_successful(self):
        result = RemoteJobResult(
            handle=JobHandle(provider="local", job_id="1"),
            status=JobStatus.FAILED,
            output_dir=None,
            logs=["boom"],
        )

        assert result.succeeded is False


class TestGpuDefaultIsProviderSpecific:
    """GPU names are provider vocabulary, not portable values.

    "A10G" is Modal's name; RunPod calls its hardware "NVIDIA RTX A4000".
    A single shared default silently sends an invalid id to whichever
    provider did not coin it, so the spec carries no default at all and each
    executor supplies its own.
    """

    def test_spec_has_no_baked_in_gpu_default(self, dataset):
        spec = RemoteJobSpec(dataset=dataset, base_model="Qwen/Qwen3.5-0.8B")

        assert spec.gpu is None

    def test_explicit_gpu_is_preserved(self, dataset):
        spec = RemoteJobSpec(
            dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", gpu="H100"
        )

        assert spec.gpu == "H100"

    def test_each_executor_declares_its_own_default(self):
        from stateset_agents.remote.modal import ModalExecutor
        from stateset_agents.remote.runpod import RunPodExecutor

        assert ModalExecutor.DEFAULT_GPU
        assert RunPodExecutor.DEFAULT_GPU
        assert ModalExecutor.DEFAULT_GPU != RunPodExecutor.DEFAULT_GPU
