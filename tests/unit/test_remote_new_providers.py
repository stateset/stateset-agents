"""Contract tests for Tinker, Prime, Hugging Face, and Together providers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from stateset_agents.remote.deployment import DeploymentSpec
from stateset_agents.remote.huggingface import (
    HuggingFaceEndpointProvider,
    HuggingFaceJobsExecutor,
)
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec
from stateset_agents.remote.prime import prime_lab_config
from stateset_agents.remote.registry import available_providers
from stateset_agents.remote.tinker import TinkerExecutor
from stateset_agents.remote.together import TogetherExecutor


def _dataset(tmp_path: Path) -> Path:
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"hi"},'
        '{"role":"assistant","content":"hello"}]}\n',
        encoding="utf-8",
    )
    return path


def test_registry_exposes_new_providers() -> None:
    providers = available_providers()
    assert {"huggingface", "prime", "tinker", "together"} <= set(providers)


def test_prime_config_maps_openenv_options(tmp_path: Path) -> None:
    spec = RemoteJobSpec(
        dataset=_dataset(tmp_path),
        base_model="Qwen/Qwen3.5-9B",
        job_kind="rl",
        harvest={
            "environment": "stateset/support",
            "harness": "openenv",
            "runtime": "uv",
            "max_steps": 12,
            "rollouts_per_example": 4,
        },
    )
    config = prime_lab_config(spec)
    assert 'id = "stateset/support"' in config
    assert 'harness = "openenv"' in config
    assert "max_steps = 12" in config
    assert "rollouts_per_example = 4" in config


class _HfApi:
    def __init__(self) -> None:
        self.cancelled = False
        self.endpoint = SimpleNamespace(
            name="support", url="https://endpoint.test", raw={"status": "running"}
        )
        self.endpoint.delete = lambda: setattr(self, "deleted", True)

    def sync_job_volume(self, **kwargs):
        return "hf://buckets/acme/input:/workspace/input"

    def run_job(self, **kwargs):
        self.run_kwargs = kwargs
        return SimpleNamespace(id="job-1", url="https://hf.test/jobs/job-1")

    def inspect_job(self, *args, **kwargs):
        return SimpleNamespace(status=SimpleNamespace(stage="COMPLETED"))

    def fetch_job_logs(self, *args, **kwargs):
        return iter(["trained"])

    def cancel_job(self, *args, **kwargs):
        self.cancelled = True

    def create_inference_endpoint(self, **kwargs):
        self.endpoint_kwargs = kwargs
        return self.endpoint

    def get_inference_endpoint(self, *args, **kwargs):
        return self.endpoint


def test_huggingface_jobs_lifecycle(tmp_path: Path) -> None:
    api = _HfApi()
    executor = HuggingFaceJobsExecutor(api, bucket="acme/jobs")
    spec = RemoteJobSpec(dataset=_dataset(tmp_path), base_model="Qwen/model")
    handle = executor.submit(spec)
    assert executor.status(handle) is JobStatus.SUCCEEDED
    assert list(executor.logs(handle)) == ["trained"]
    output = executor.fetch(handle, tmp_path / "out")
    assert (output / "huggingface_job.json").exists()
    assert api.run_kwargs["flavor"] == "a10g-large"
    executor.cancel(handle)
    assert api.cancelled


def test_huggingface_endpoint_lifecycle() -> None:
    api = _HfApi()
    provider = HuggingFaceEndpointProvider(api)
    spec = DeploymentSpec(
        name="support",
        model_name="support",
        weights_uri="acme/support-model",
        gpu="nvidia-a10g",
    )
    handle = provider.deploy(spec)
    assert handle.endpoint == "https://endpoint.test"
    assert api.endpoint_kwargs["instance_size"] == "x1"
    assert api.endpoint_kwargs["framework"] == "pytorch"
    assert provider.status(handle) == {"status": "running"}
    provider.delete(handle)
    assert api.deleted


class _FineTuning:
    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(id="ft-1")

    def retrieve(self, *, id):
        return SimpleNamespace(
            id=id,
            status="completed",
            output_name="acme/tuned",
            events=[SimpleNamespace(message="done")],
        )

    def cancel(self, *, id):
        self.cancelled = id


class _Together:
    def __init__(self) -> None:
        self.fine_tuning = _FineTuning()
        self.files = SimpleNamespace(
            upload=lambda **kwargs: SimpleNamespace(id="file-1")
        )


def test_together_lifecycle(tmp_path: Path) -> None:
    client = _Together()
    executor = TogetherExecutor(client)
    handle = executor.submit(
        RemoteJobSpec(dataset=_dataset(tmp_path), base_model="Qwen/model")
    )
    assert executor.status(handle) is JobStatus.SUCCEEDED
    assert list(executor.logs(handle)) == ["done"]
    assert (
        executor.fetch(handle, tmp_path / "together") / "together_checkpoint.json"
    ).exists()
    executor.cancel(JobHandle("together", "ft-1"))
    assert client.fine_tuning.cancelled == "ft-1"


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return "".join(f"<{item['role']}>{item['content']}" for item in messages)

    def encode(self, text, add_special_tokens=False):
        return list(range(1, len(text) + 1))


class _Immediate:
    def __init__(self, value=None) -> None:
        self.value = value

    def result(self):
        return self.value


class _TinkerTraining:
    def __init__(self) -> None:
        self.steps = 0

    def get_tokenizer(self):
        return _Tokenizer()

    def forward_backward(self, batch, loss_fn):
        assert batch and loss_fn == "cross_entropy"
        return _Immediate()

    def optim_step(self, params):
        self.steps += 1
        return _Immediate()

    def save_weights_for_sampler(self, name):
        return _Immediate(SimpleNamespace(path="tinker://sampler"))

    def save_state(self, name):
        return _Immediate(SimpleNamespace(path="tinker://state"))


class _TinkerModule:
    class ModelInput:
        from_ints = staticmethod(lambda values: values)

    class TensorData:
        from_torch = staticmethod(lambda value: value.tolist())

    class Datum:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class AdamParams:
        def __init__(self, learning_rate) -> None:
            self.learning_rate = learning_rate


def test_tinker_remote_autograd_lifecycle(tmp_path: Path) -> None:
    training = _TinkerTraining()
    service = SimpleNamespace(create_lora_training_client=lambda **kwargs: training)
    executor = TinkerExecutor(service, tinker_module=_TinkerModule())
    handle = executor.submit(
        RemoteJobSpec(
            dataset=_dataset(tmp_path),
            base_model="thinkingmachines/Inkling-Small",
            num_epochs=1,
            gradient_accumulation_steps=1,
        )
    )
    result = executor.wait(handle, poll_interval_s=0.001)
    assert result.succeeded
    assert training.steps == 1
    assert (result.output_dir / "tinker_checkpoint.json").exists()
