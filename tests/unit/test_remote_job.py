"""Tests for ``stateset_agents.remote.job`` — the provider-agnostic job contract.

The spec mirrors ``scripts/sft_from_curated.py``'s argparse surface exactly.
These tests pin that correspondence so the two cannot silently drift.
"""

from __future__ import annotations

import json

import pytest

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.fireworks import FireworksExecutor
from stateset_agents.remote.job import (
    JobHandle,
    JobStatus,
    RemoteJobResult,
    RemoteJobSpec,
)
from stateset_agents.remote.local import LocalExecutor
from stateset_agents.remote.modal import ModalExecutor
from stateset_agents.remote.river import RiverExecutor
from stateset_agents.remote.runpod import RunPodExecutor


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

    def test_new_optional_fields_default_to_none(self, dataset):
        spec = RemoteJobSpec(dataset=dataset, base_model="Qwen/Qwen3.5-0.8B")

        assert spec.container_disk_gb is None
        assert spec.eval_prompts is None
        assert spec.network_volume_id is None


class TestRemoteJobSpecValidation:
    def test_rejects_unknown_job_kind(self, dataset):
        with pytest.raises(ValueError, match="job_kind"):
            RemoteJobSpec(
                dataset=dataset,
                base_model="Qwen/Qwen3.5-0.8B",
                job_kind="grpo_typo",
            )

    def test_normalizes_job_kind(self, dataset):
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            job_kind=" HARVEST ",
        )
        assert spec.job_kind == "harvest"

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

    def test_rejects_non_positive_container_disk(self, dataset):
        """None means "executor default"; zero is always a mistake."""
        with pytest.raises(ValueError, match="container_disk_gb"):
            RemoteJobSpec(
                dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", container_disk_gb=0
            )

    def test_rejects_non_positive_gpu_count(self, dataset):
        """None is not allowed here — one GPU is the smallest sane request."""
        with pytest.raises(ValueError, match="gpu_count"):
            RemoteJobSpec(dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", gpu_count=0)

    def test_gpu_count_defaults_to_one(self, dataset):
        spec = RemoteJobSpec(dataset=dataset, base_model="Qwen/Qwen3.5-0.8B")
        assert spec.gpu_count == 1

    def test_gpu_count_never_reaches_the_training_script(self, dataset):
        """It is a provider resource — the job discovers its GPUs itself."""
        spec = RemoteJobSpec(
            dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", gpu_count=2
        )
        assert "--gpu-count" not in spec.to_cli_args()

    def test_rejects_blank_network_volume_id(self, dataset):
        """None means "no volume"; a blank string is always a mistake."""
        with pytest.raises(ValueError, match="network_volume_id"):
            RemoteJobSpec(
                dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", network_volume_id="  "
            )

    def test_rejects_non_positive_cost_ceiling(self, dataset):
        with pytest.raises(ValueError, match="max_cost_usd"):
            RemoteJobSpec(
                dataset=dataset,
                base_model="Qwen/Qwen3.5-0.8B",
                max_cost_usd=0,
            )

    def test_network_volume_id_never_reaches_the_training_script(self, dataset):
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            network_volume_id="vol-123",
        )
        assert "vol-123" not in spec.to_cli_args()

    def test_rejects_unknown_cloud_type(self, dataset):
        with pytest.raises(ValueError, match="cloud_type"):
            RemoteJobSpec(
                dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", cloud_type="SPOT"
            )

    def test_cloud_type_is_normalized_to_upper_case(self, dataset):
        spec = RemoteJobSpec(
            dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", cloud_type="community"
        )
        assert spec.cloud_type == "COMMUNITY"

    def test_cloud_type_defaults_to_secure(self, dataset):
        spec = RemoteJobSpec(dataset=dataset, base_model="Qwen/Qwen3.5-0.8B")
        assert spec.cloud_type == "SECURE"
        assert spec.resume is False


class TestProviderJobKindCapabilities:
    @pytest.mark.parametrize(
        "executor_cls",
        [LocalExecutor, RunPodExecutor],
    )
    def test_machine_executors_reject_remote_autograd_rl(self, dataset, executor_cls):
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            job_kind="rl",
        )
        with pytest.raises(RemoteExecutionError, match="does not support.*rl"):
            executor_cls().submit(spec)

    @pytest.mark.parametrize("executor_cls", [ModalExecutor, FireworksExecutor])
    def test_sft_only_executors_reject_harvest(self, dataset, executor_cls):
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            job_kind="harvest",
        )
        with pytest.raises(RemoteExecutionError, match="does not support.*harvest"):
            executor_cls().submit(spec)

    @pytest.mark.parametrize("job_kind", ["sft", "harvest", "rl"])
    def test_river_declares_every_remote_mode(self, dataset, job_kind):
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            job_kind=job_kind,
        )
        RiverExecutor().validate_spec(spec)

    def test_capabilities_are_machine_readable(self):
        capabilities = FireworksExecutor().capabilities()

        assert capabilities == {
            "provider": "fireworks",
            "job_kinds": ["sft"],
            "durable_handles": True,
            "managed_deployments": True,
            "result_kind": "hosted_pointer_or_local_artifacts",
            "compute_model": "managed-finetuning-and-serving",
            "verification_status": "code-complete-live-lifecycle-pending",
        }


class TestResumeAndCloudTypeCliArgs:
    def test_resume_lands_in_cli_args(self, dataset):
        spec = RemoteJobSpec(
            dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", resume=True
        )
        assert "--resume" in spec.to_cli_args()

    def test_no_resume_flag_by_default(self, dataset):
        spec = RemoteJobSpec(dataset=dataset, base_model="Qwen/Qwen3.5-0.8B")
        assert "--resume" not in spec.to_cli_args()

    def test_cloud_type_is_a_provider_field_and_never_reaches_the_script(self, dataset):
        spec = RemoteJobSpec(
            dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", cloud_type="COMMUNITY"
        )
        # (Exact flag matches: tmp_path embeds the test name, which would
        # trip a substring search.)
        args = spec.to_cli_args()
        assert "--cloud-type" not in args
        assert "COMMUNITY" not in args

    def test_round_trips_through_json(self, dataset):
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            cloud_type="COMMUNITY",
            resume=True,
        )
        restored = RemoteJobSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert restored == spec


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
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            gpu="A100",
            container_disk_gb=160,
        )

        args = spec.to_cli_args()

        assert "--gpu" not in args
        assert "A100" not in args
        assert "--container-disk-gb" not in args
        assert "160" not in args

    def test_eval_prompts_are_a_job_field_and_travel_as_json(self, dataset):
        """Unlike resource fields, eval prompts must reach the training script."""
        prompts = ["what's the return policy?", "hello there"]
        spec = RemoteJobSpec(
            dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", eval_prompts=prompts
        )

        args = spec.to_cli_args()

        assert "--eval-prompts-json" in args
        blob = args[args.index("--eval-prompts-json") + 1]
        assert json.loads(blob) == prompts

    def test_harvest_eval_prompts_travel_as_json(self, dataset):
        prompts = [{"prompt": "hello", "expect": ["hi"]}]
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            job_kind="harvest",
            eval_prompts=prompts,
        )

        args = spec.harvest_cli_args()
        blob = args[args.index("--eval-prompts-json") + 1]
        assert json.loads(blob) == prompts

    def test_parent_adapter_reaches_sft_cli(self, dataset, tmp_path):
        adapter = tmp_path / "adapter"
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            parent_adapter=adapter,
        )

        args = spec.to_cli_args()
        assert args[args.index("--parent-adapter") + 1] == str(adapter)

    def test_eval_spec_dicts_travel_as_json_too(self, dataset):
        """Prompt-spec entries (expect/forbid/judge) ride the same JSON blob."""
        prompts = [
            "plain prompt",
            {"prompt": "Say 41.", "expect": ["41"], "forbid": ["sorry"]},
        ]
        spec = RemoteJobSpec(
            dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", eval_prompts=prompts
        )

        args = spec.to_cli_args()

        blob = args[args.index("--eval-prompts-json") + 1]
        assert json.loads(blob) == prompts

    def test_malformed_eval_spec_is_rejected_at_construction(self, dataset):
        """Validated on this machine, before a GPU is rented."""
        with pytest.raises(ValueError, match="prompt"):
            RemoteJobSpec(
                dataset=dataset,
                base_model="Qwen/Qwen3.5-0.8B",
                eval_prompts=[{"expect": ["no prompt key"]}],
            )

    def test_eval_prompts_flag_omitted_when_unset(self, dataset):
        spec = RemoteJobSpec(dataset=dataset, base_model="Qwen/Qwen3.5-0.8B")

        assert "--eval-prompts-json" not in spec.to_cli_args()

    def test_eval_max_new_tokens_travels_with_the_prompts(self, dataset):
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            eval_prompts=["hi"],
            eval_max_new_tokens=300,
        )

        args = spec.to_cli_args()

        assert args[args.index("--eval-max-new-tokens") + 1] == "300"

    def test_eval_max_new_tokens_flag_omitted_without_prompts(self, dataset):
        """The budget only means something when there are prompts to answer."""
        spec = RemoteJobSpec(
            dataset=dataset, base_model="Qwen/Qwen3.5-0.8B", eval_max_new_tokens=300
        )

        assert "--eval-max-new-tokens" not in spec.to_cli_args()

    def test_eval_max_new_tokens_must_be_positive(self, dataset):
        with pytest.raises(ValueError, match="eval_max_new_tokens"):
            RemoteJobSpec(
                dataset=dataset,
                base_model="Qwen/Qwen3.5-0.8B",
                eval_max_new_tokens=0,
            )

    def test_eval_prompts_round_trip_through_json(self, dataset):
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-0.8B",
            eval_prompts=["a", "b"],
            container_disk_gb=160,
        )

        restored = RemoteJobSpec.from_dict(json.loads(json.dumps(spec.to_dict())))

        assert restored == spec


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
