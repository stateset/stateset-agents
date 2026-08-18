"""Tests for ``RiverExecutor`` — entirely against fakes.

River could not be live-verified (no API key, ``river-client`` unavailable),
so the fakes here record the exact call sequence we believe River's docs
describe. If the real service disagrees, these tests are what will need
changing — which is the point of writing them as a recording rather than as
assertions about internals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec
from stateset_agents.remote.registry import available_providers, get_executor
from stateset_agents.remote.river import (
    CHECKPOINT_POINTER_NAME,
    RIVER_API_KEY_ENV,
    RiverExecutor,
)
from stateset_agents.training.lineage import MANIFEST_NAME
from tests.unit.test_river_batches import FakeTokenizer


@dataclass
class FakeLoraConfig:
    rank: int
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = False
    seed: int | None = None


@dataclass
class FakeModel:
    """Records the training loop River is asked to run."""

    base_model: str
    lora: FakeLoraConfig
    forward_backward_calls: list[tuple[int, str]] = field(default_factory=list)
    optim_steps: list[dict[str, Any]] = field(default_factory=list)
    saved: list[tuple[str, str]] = field(default_factory=list)

    def forward_backward(self, batch, loss_fn="cross_entropy"):
        self.forward_backward_calls.append((len(batch), loss_fn))
        return {"loss": 0.5, "num_tokens": sum(len(d["input_ids"]) for d in batch)}

    def optim_step(self, **kwargs):
        self.optim_steps.append(kwargs)
        return {"ok": True}

    def save_weights(self, name, mode="inference"):
        self.saved.append((name, mode))
        return f"river://checkpoints/{name}"


@dataclass
class FakeSession:
    models: list[FakeModel] = field(default_factory=list)

    def create_model(self, base_model, lora):
        model = FakeModel(base_model=base_model, lora=lora)
        self.models.append(model)
        return model


class FakeRiverClient:
    """Stand-in for ``river_client.Client``."""

    LoraConfig = FakeLoraConfig

    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def create_session(self) -> FakeSession:
        session = FakeSession()
        self.sessions.append(session)
        return session

    @property
    def model(self) -> FakeModel:
        return self.sessions[0].models[0]


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
            for i in range(4)
        )
        + "\n"
    )
    return path


@pytest.fixture
def spec(dataset, tmp_path):
    return RemoteJobSpec(
        dataset=dataset,
        base_model="Qwen/Qwen3.5-9B",
        output_dir=tmp_path / "out",
        num_epochs=2,
        lora_r=16,
        per_device_batch_size=2,
    )


@pytest.fixture
def client():
    return FakeRiverClient()


@pytest.fixture
def executor(client, tmp_path):
    return RiverExecutor(
        client=client,
        tokenizer=FakeTokenizer(),
        ledger_path=tmp_path / "ledger.jsonl",
    )


class TestRegistry:
    def test_river_is_a_registered_provider(self):
        assert "river" in available_providers()

    def test_provider_resolves_without_the_sdk(self):
        """Listing and constructing must not require ``river-client``."""
        assert isinstance(get_executor("river"), RiverExecutor)

    def test_implements_the_executor_contract(self):
        assert isinstance(get_executor("river"), RemoteExecutor)


class TestSubmit:
    def test_creates_one_model_with_the_requested_rank(self, executor, spec, client):
        executor.submit(spec)
        assert client.model.lora.rank == 16
        assert client.model.base_model == "Qwen/Qwen3.5-9B"

    def test_trains_attention_and_mlp_but_not_unembed(self, executor, spec, client):
        executor.submit(spec)
        assert client.model.lora.train_attn is True
        assert client.model.lora.train_mlp is True
        assert client.model.lora.train_unembed is False

    def test_runs_forward_backward_for_every_batch_of_every_epoch(
        self, executor, spec, client
    ):
        executor.submit(spec)
        # 4 rows / batch size 2 = 2 batches, x 2 epochs.
        assert len(client.model.forward_backward_calls) == 4

    def test_uses_cross_entropy_for_supervised_data(self, executor, spec, client):
        executor.submit(spec)
        assert {loss for _, loss in client.model.forward_backward_calls} == {
            "cross_entropy"
        }

    def test_one_optim_step_per_forward_backward(self, executor, spec, client):
        executor.submit(spec)
        assert len(client.model.optim_steps) == len(client.model.forward_backward_calls)

    def test_optim_step_gets_the_specs_learning_rate(self, executor, spec, client):
        executor.submit(spec)
        assert all(
            step["lr"] == spec.learning_rate for step in client.model.optim_steps
        )

    def test_save_weights_is_called_exactly_once(self, executor, spec, client):
        executor.submit(spec)
        assert len(client.model.saved) == 1

    def test_saves_in_inference_mode(self, executor, spec, client):
        executor.submit(spec)
        assert client.model.saved[0][1] == "inference"

    def test_job_succeeds(self, executor, spec):
        handle = executor.submit(spec)
        assert executor.status(handle) is JobStatus.SUCCEEDED

    def test_checkpoint_uri_lands_in_the_logs(self, executor, spec):
        handle = executor.submit(spec)
        assert any("river://" in line for line in executor.logs(handle))

    def test_dry_run_trains_nothing(self, dataset, tmp_path, client):
        executor = RiverExecutor(
            client=client, tokenizer=FakeTokenizer(), ledger_path=tmp_path / "l.jsonl"
        )
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-9B",
            output_dir=tmp_path / "out",
            dry_run=True,
        )
        handle = executor.submit(spec)
        assert executor.status(handle) is JobStatus.SUCCEEDED
        assert client.sessions == []
        assert any("dry run" in line for line in executor.logs(handle))

    def test_empty_dataset_fails_without_calling_river(self, tmp_path, client):
        empty = tmp_path / "empty.jsonl"
        empty.write_text('{"messages": []}\n')
        executor = RiverExecutor(
            client=client, tokenizer=FakeTokenizer(), ledger_path=tmp_path / "l.jsonl"
        )
        handle = executor.submit(
            RemoteJobSpec(
                dataset=empty,
                base_model="Qwen/Qwen3.5-9B",
                output_dir=tmp_path / "out",
            )
        )
        assert executor.status(handle) is JobStatus.FAILED
        assert client.sessions == []


class TestSpecValidation:
    def test_rejects_a_rank_river_cannot_serve(self, executor, spec):
        spec.lora_r = 64
        with pytest.raises(ValueError, match="1-32"):
            executor.submit(spec)

    def test_unknown_base_model_still_runs(self, executor, spec):
        spec.base_model = "acme/private-model"
        assert executor.status(executor.submit(spec)) is JobStatus.SUCCEEDED

    def test_machine_shaped_fields_are_ignored_not_fatal(self, executor, spec):
        """A spec written for RunPod must still be submittable to River."""
        spec.gpu = "NVIDIA H100 80GB HBM3"
        spec.gpu_count = 4
        spec.container_disk_gb = 200
        spec.cloud_type = "COMMUNITY"
        spec.network_volume_id = "vol-123"
        handle = executor.submit(spec)
        assert executor.status(handle) is JobStatus.SUCCEEDED
        ignored = next(line for line in executor.logs(handle) if "ignoring" in line)
        for name in ("gpu", "gpu_count", "container_disk_gb", "cloud_type"):
            assert name in ignored


class TestClientConstruction:
    def test_missing_sdk_names_the_pip_install(self, monkeypatch, spec):
        import builtins

        real_import = builtins.__import__

        def no_river(name, *args, **kwargs):
            if name == "river_client":
                raise ImportError("No module named 'river_client'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_river)
        monkeypatch.setenv(RIVER_API_KEY_ENV, "rv_test")
        with pytest.raises(RemoteExecutionError, match="pip install river-client"):
            RiverExecutor(tokenizer=FakeTokenizer()).submit(spec)

    def test_missing_api_key_names_the_env_var(self, monkeypatch, spec):
        fake_module = type("M", (), {"Client": lambda **kw: object()})
        monkeypatch.setitem(__import__("sys").modules, "river_client", fake_module)
        monkeypatch.delenv(RIVER_API_KEY_ENV, raising=False)
        with pytest.raises(RemoteExecutionError, match=RIVER_API_KEY_ENV):
            RiverExecutor(tokenizer=FakeTokenizer()).submit(spec)

    def test_injected_client_needs_neither(self, monkeypatch, executor, spec):
        monkeypatch.delenv(RIVER_API_KEY_ENV, raising=False)
        assert executor.status(executor.submit(spec)) is JobStatus.SUCCEEDED


class TestFetch:
    def test_writes_a_checkpoint_pointer(self, executor, spec):
        out = executor.fetch(executor.submit(spec))
        assert (out / CHECKPOINT_POINTER_NAME).exists()

    def test_pointer_carries_the_river_uri(self, executor, spec):
        out = executor.fetch(executor.submit(spec))
        pointer = json.loads((out / CHECKPOINT_POINTER_NAME).read_text())
        assert pointer["checkpoint"].startswith("river://")

    def test_pointer_records_the_lora_config_and_base_model(self, executor, spec):
        out = executor.fetch(executor.submit(spec))
        pointer = json.loads((out / CHECKPOINT_POINTER_NAME).read_text())
        assert pointer["base_model"] == "Qwen/Qwen3.5-9B"
        assert pointer["lora"]["rank"] == 16
        assert pointer["steps"] == 4
        assert pointer["final_loss"] == 0.5

    def test_pointer_says_it_is_not_an_adapter(self, executor, spec):
        out = executor.fetch(executor.submit(spec))
        pointer = json.loads((out / CHECKPOINT_POINTER_NAME).read_text())
        assert "pointer" in pointer["note"]

    def test_does_not_fabricate_adapter_weights(self, executor, spec):
        out = executor.fetch(executor.submit(spec))
        assert not (out / "adapter_model.safetensors").exists()
        assert not (out / "adapter_config.json").exists()

    def test_writes_a_lineage_manifest(self, executor, spec):
        out = executor.fetch(executor.submit(spec))
        manifest = json.loads((out / MANIFEST_NAME).read_text())
        assert manifest["base_model"] == "Qwen/Qwen3.5-9B"
        assert manifest["dataset_sha256"]
        assert manifest["hyperparameters"]["provider"] == "river"
        assert manifest["hyperparameters"]["river_checkpoint"].startswith("river://")

    def test_manifest_records_the_parent_adapter(self, dataset, tmp_path, executor):
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-9B",
            output_dir=tmp_path / "gen2",
            parent_adapter="outputs/gen1",
        )
        out = executor.fetch(executor.submit(spec))
        manifest = json.loads((out / MANIFEST_NAME).read_text())
        assert manifest["parent_adapter"] == "outputs/gen1"

    def test_honours_an_explicit_destination(self, executor, spec, tmp_path):
        dest = tmp_path / "elsewhere"
        assert executor.fetch(executor.submit(spec), dest=dest) == dest
        assert (dest / CHECKPOINT_POINTER_NAME).exists()

    def test_refuses_to_fetch_an_unfinished_job(self, executor):
        handle = JobHandle(provider="river", job_id="nope")
        with pytest.raises(RemoteExecutionError, match="unknown job"):
            executor.fetch(handle)


class TestWaitAndCost:
    def test_wait_returns_the_output_dir(self, executor, spec):
        result = executor.wait(executor.submit(spec), poll_interval_s=0)
        assert result.succeeded
        assert result.output_dir == spec.output_dir

    def test_cost_is_unknown_never_zero(self, executor, spec):
        """River bills per token and quotes no price to the SDK."""
        result = executor.wait(executor.submit(spec), poll_interval_s=0)
        assert result.cost_usd is None
        assert result.duration_s is not None

    def test_ledger_records_the_run_with_an_unknown_cost(
        self, executor, spec, tmp_path
    ):
        executor.submit(spec)
        entries = [
            json.loads(line)
            for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
        ]
        assert len(entries) == 1
        assert entries[0]["provider"] == "river"
        assert entries[0]["cost_usd"] is None
        assert entries[0]["status"] == "succeeded"


class TestSessionResolution:
    """The docs' canonical form is ``with client.session(project=...) as s``;
    older shapes (``create_session()``, or the client doubling as the
    session) must keep working, and a context-managed session must be
    closed."""

    def test_docs_canonical_context_manager_session_is_used_and_closed(
        self, spec, tmp_path
    ):
        events: list[str] = []

        class CmSession(FakeSession):
            def __enter__(self):
                events.append("enter")
                return self

            def __exit__(self, *exc):
                events.append("exit")
                return False

        class ModernClient(FakeRiverClient):
            def __init__(self):
                super().__init__()
                self.projects: list[str | None] = []

            create_session = None  # the modern SDK shape has session() only

            def session(self, project=None):
                self.projects.append(project)
                cm = CmSession()
                self.sessions.append(cm)
                return cm

        client = ModernClient()
        executor = RiverExecutor(
            client=client, tokenizer=FakeTokenizer(), ledger_path=tmp_path / "l.jsonl"
        )
        handle = executor.submit(spec)

        assert executor.status(handle) is JobStatus.SUCCEEDED
        assert events == ["enter", "exit"]
        assert client.projects == [spec.output_dir.name]
        assert client.model.saved  # trained through the cm session

    def test_plain_session_method_without_cm_protocol_still_works(self, spec, tmp_path):
        class PlainClient(FakeRiverClient):
            create_session = None

            def session(self, project=None):
                s = FakeSession()
                self.sessions.append(s)
                return s

        executor = RiverExecutor(
            client=PlainClient(),
            tokenizer=FakeTokenizer(),
            ledger_path=tmp_path / "l.jsonl",
        )
        assert executor.status(executor.submit(spec)) is JobStatus.SUCCEEDED

    def test_session_not_accepting_project_kwarg_degrades_gracefully(
        self, spec, tmp_path
    ):
        class OldSignature(FakeRiverClient):
            create_session = None

            def session(self):  # no project kwarg at all
                s = FakeSession()
                self.sessions.append(s)
                return s

        executor = RiverExecutor(
            client=OldSignature(),
            tokenizer=FakeTokenizer(),
            ledger_path=tmp_path / "l.jsonl",
        )
        assert executor.status(executor.submit(spec)) is JobStatus.SUCCEEDED


class TestFailures:
    def test_sdk_exceptions_become_remote_execution_errors(self, spec, tmp_path):
        class Boom(FakeRiverClient):
            def create_session(self):
                raise RuntimeError("river is down")

        executor = RiverExecutor(
            client=Boom(), tokenizer=FakeTokenizer(), ledger_path=tmp_path / "l.jsonl"
        )
        with pytest.raises(RemoteExecutionError, match="River training run failed"):
            executor.submit(spec)

    def test_a_failed_run_is_still_ledgered(self, spec, tmp_path):
        class Boom(FakeRiverClient):
            def create_session(self):
                raise RuntimeError("river is down")

        ledger = tmp_path / "l.jsonl"
        executor = RiverExecutor(
            client=Boom(), tokenizer=FakeTokenizer(), ledger_path=ledger
        )
        with pytest.raises(RemoteExecutionError):
            executor.submit(spec)
        assert '"failed"' in ledger.read_text()

    def test_cancel_on_a_finished_job_is_a_no_op(self, executor, spec):
        handle = executor.submit(spec)
        executor.cancel(handle)
        assert executor.status(handle) is JobStatus.SUCCEEDED

    def test_unknown_handle_is_reported_clearly(self, executor):
        with pytest.raises(RemoteExecutionError, match="unknown job"):
            executor.status(JobHandle(provider="river", job_id="ghost"))


class TestAccountStateErrors:
    """River answers account problems in an OpenAI-shaped envelope; those are
    states the user must act on, not generic training failures.

    Observed live against api.river.ai: an unfunded account answers 402 with
    "Billing: insufficient_funds"; a missing key answers 401.
    """

    def _failing_executor(self, message, tmp_path):
        class Boom(FakeRiverClient):
            def create_session(self):
                raise RuntimeError(message)

        return RiverExecutor(
            client=Boom(),
            tokenizer=FakeTokenizer(),
            ledger_path=tmp_path / "ledger.jsonl",
        )

    def test_insufficient_funds_names_the_fix(self, spec, tmp_path):
        ex = self._failing_executor(
            '{"error":{"message":"Billing: insufficient_funds",'
            '"type":"invalid_request_error"}}',
            tmp_path,
        )
        with pytest.raises(RemoteExecutionError, match="no credits"):
            ex.submit(spec)

    def test_rejected_key_points_at_the_env_var(self, spec, tmp_path):
        ex = self._failing_executor("401 unauthorized", tmp_path)
        with pytest.raises(RemoteExecutionError, match="RIVER_API_KEY"):
            ex.submit(spec)

    def test_ordinary_training_failure_is_not_mislabelled(self, spec, tmp_path):
        ex = self._failing_executor("CUDA kernel exploded", tmp_path)
        with pytest.raises(RemoteExecutionError, match="River training run failed"):
            ex.submit(spec)


class TestTransientRecovery:
    """The SDK's taxonomy: RiverConnectionError/RiverTimeoutError mean
    'back off, rebuild the session, retry'; observed live when a slow
    create_model timed out client-side and the retry raced the server-side
    create into ALREADY_EXISTS."""

    class FakeRiverConnectionError(Exception):
        pass

    def _flaky_client(self, failures):
        outer = self

        class FlakyClient(FakeRiverClient):
            RiverConnectionError = outer.FakeRiverConnectionError
            attempts = 0

            def create_session(self):
                type(self).attempts += 1
                if type(self).attempts <= failures:
                    raise outer.FakeRiverConnectionError("Resource already exists")
                return super().create_session()

        return FlakyClient()

    def _executor(self, client, tmp_path):
        executor = RiverExecutor(
            client=client, tokenizer=FakeTokenizer(), ledger_path=tmp_path / "l.jsonl"
        )
        executor._sleep = lambda s: None  # no real backoff in tests
        # The fake SDK's transient types live on the client, not a module.
        executor._transient_exceptions = lambda c: (self.FakeRiverConnectionError,)
        return executor

    def test_transient_failures_are_retried_and_the_run_succeeds(self, spec, tmp_path):
        client = self._flaky_client(failures=2)
        executor = self._executor(client, tmp_path)

        handle = executor.submit(spec)

        assert executor.status(handle) is JobStatus.SUCCEEDED
        assert type(client).attempts == 3
        assert any("retrying" in line for line in executor.logs(handle))

    def test_persistent_transient_failure_gives_up_after_the_cap(self, spec, tmp_path):
        client = self._flaky_client(failures=99)
        executor = self._executor(client, tmp_path)

        with pytest.raises(RemoteExecutionError):
            executor.submit(spec)
        assert type(client).attempts == RiverExecutor.MAX_TRANSIENT_ATTEMPTS


class TestTrainStepPreference:
    def test_train_step_is_used_when_the_model_offers_it(self, spec, tmp_path):
        class StepModel(FakeModel):
            train_steps: list[dict] = []

            def train_step(self, data, lr, loss_fn="cross_entropy", **kw):
                type(self).train_steps.append({"n": len(data), "lr": lr})
                return {"loss_mean": 0.25}, {"ok": True}

        class StepSession(FakeSession):
            def create_model(self, base_model, lora):
                model = StepModel(base_model=base_model, lora=lora)
                self.models.append(model)
                return model

        class StepClient(FakeRiverClient):
            def create_session(self):
                session = StepSession()
                self.sessions.append(session)
                return session

        StepModel.train_steps = []
        client = StepClient()
        executor = RiverExecutor(
            client=client, tokenizer=FakeTokenizer(), ledger_path=tmp_path / "l.jsonl"
        )

        handle = executor.submit(spec)

        assert executor.status(handle) is JobStatus.SUCCEEDED
        assert StepModel.train_steps, "train_step was never called"
        assert client.model.forward_backward_calls == []
        # loss_mean (train_step's metric name) reaches the job record.
        assert any("loss 0.25" in line for line in executor.logs(handle))


class SamplingModel(FakeModel):
    """FakeModel that also answers ``sample`` — the harvest surface."""

    #: prompt-text -> list of sample texts (greedy calls get [0]).
    canned: dict[str, list[str]] = {}
    sample_calls: list[dict] = []

    def sample(self, prompts=None, *, num_samples=1, temperature=1.0, **kw):
        type(self).sample_calls.append({"n": num_samples, "temperature": temperature})

        class _S:
            def __init__(self, text):
                self.text = text

        groups = []
        for prompt in prompts:
            texts = type(self).canned.get(prompt, ["nothing"] * num_samples)
            groups.append([_S(t) for t in texts[:num_samples]])
        return groups


class SamplingClient(FakeRiverClient):
    def __init__(self):
        super().__init__()
        self.created_with: list[dict] = []

    def create_session(self):
        outer = self

        class _Session(FakeSession):
            def create_model(self, base_model, lora=None, checkpoint=None):
                outer.created_with.append(
                    {"base_model": base_model, "checkpoint": checkpoint}
                )
                model = SamplingModel(base_model=base_model, lora=lora)
                self.models.append(model)
                return model

        session = _Session()
        self.sessions.append(session)
        return session


class TestRiverHarvest:
    """The zero-infrastructure flywheel step: best-of-N through River's
    sampling API, writing the same artifacts as the pod-based harvest so
    run_flywheel cannot tell the difference."""

    def _spec(self, tmp_path, **overrides):
        prompts = tmp_path / "prompts.json"
        prompts.write_text(
            json.dumps([{"prompt": "fix my vpn", "expect": ["vpn profile"]}])
        )
        defaults = {
            "dataset": prompts,
            "base_model": "Qwen/Qwen3.5-9B",
            "output_dir": tmp_path / "harvest",
            "job_kind": "harvest",
            "harvest": {"adapter_dir": None, "best_of": 4, "temperature": 0.9},
        }
        defaults.update(overrides)
        return RemoteJobSpec(**defaults)

    def _executor(self, client, tmp_path):
        return RiverExecutor(
            client=client,
            tokenizer=FakeTokenizer(),
            ledger_path=tmp_path / "l.jsonl",
        )

    def test_harvest_writes_the_pod_contract_artifacts(self, tmp_path):
        SamplingModel.canned = {
            "fix my vpn": [
                "re-provisioned your vpn profile",
                "no idea",
                "vpn profile reset",
                "sorry",
            ]
        }
        SamplingModel.sample_calls = []
        client = SamplingClient()
        executor = self._executor(client, tmp_path)

        result = executor.wait(executor.submit(self._spec(tmp_path)))

        assert result.status is JobStatus.SUCCEEDED
        summary = json.loads(
            (tmp_path / "harvest" / "harvest_summary.json").read_text()
        )
        assert summary["kept"] == 2 and summary["samples"] == 4
        rows = [
            json.loads(line)
            for line in (tmp_path / "harvest" / "harvest.jsonl")
            .read_text()
            .splitlines()
        ]
        assert all("vpn profile" in r["messages"][1]["content"] for r in rows)

    def test_harvest_resolves_the_previous_generation_pointer(self, tmp_path):
        pointer_dir = tmp_path / "gen1"
        pointer_dir.mkdir()
        (pointer_dir / CHECKPOINT_POINTER_NAME).write_text(
            json.dumps({"checkpoint": "river://abc/sampler_weights/gen1"})
        )
        SamplingModel.canned = {"fix my vpn": ["vpn profile ok"] * 4}
        client = SamplingClient()
        executor = self._executor(client, tmp_path)

        executor.wait(
            executor.submit(
                self._spec(
                    tmp_path,
                    harvest={"adapter_dir": str(pointer_dir), "best_of": 4},
                )
            )
        )

        assert client.created_with[0]["checkpoint"] == (
            "river://abc/sampler_weights/gen1"
        )

    def test_bogus_pointer_dir_fails_loudly(self, tmp_path):
        client = SamplingClient()
        executor = self._executor(client, tmp_path)

        with pytest.raises(RemoteExecutionError, match="checkpoint pointer"):
            executor.submit(
                self._spec(
                    tmp_path,
                    harvest={"adapter_dir": str(tmp_path / "nope"), "best_of": 4},
                )
            )

    def test_eval_prompts_are_scored_greedily_first(self, tmp_path):
        SamplingModel.canned = {
            "fix my vpn": ["vpn profile"] * 4,
            "eval me": ["contains token"],
        }
        SamplingModel.sample_calls = []
        client = SamplingClient()
        executor = self._executor(client, tmp_path)

        executor.wait(
            executor.submit(
                self._spec(
                    tmp_path,
                    eval_prompts=[{"prompt": "eval me", "expect": ["token"]}],
                )
            )
        )

        summary = json.loads(
            (tmp_path / "harvest" / "harvest_summary.json").read_text()
        )
        assert summary["eval"] == {
            "passed": 1,
            "total": 1,
            "results": summary["eval"]["results"],
        }
        # First sample call is the greedy eval; second is the harvest.
        assert SamplingModel.sample_calls[0]["temperature"] == 0.0
        assert SamplingModel.sample_calls[1]["n"] == 4

    def test_dry_run_writes_only_the_summary(self, tmp_path):
        client = SamplingClient()
        executor = self._executor(client, tmp_path)

        result = executor.wait(executor.submit(self._spec(tmp_path, dry_run=True)))

        assert result.status is JobStatus.SUCCEEDED
        assert (tmp_path / "harvest" / "harvest_summary.json").exists()
        assert not (tmp_path / "harvest" / "harvest.jsonl").exists()
        assert client.sessions == []


class TestRiverPostTrainEval:
    def test_training_with_eval_prompts_writes_eval_results(self, dataset, tmp_path):
        SamplingModel.canned = {"eval me": ["the token appears"]}
        client = SamplingClient()
        executor = RiverExecutor(
            client=client,
            tokenizer=FakeTokenizer(),
            ledger_path=tmp_path / "l.jsonl",
        )
        spec = RemoteJobSpec(
            dataset=dataset,
            base_model="Qwen/Qwen3.5-9B",
            output_dir=tmp_path / "out",
            num_epochs=1,
            eval_prompts=[{"prompt": "eval me", "expect": ["token"]}],
        )

        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.SUCCEEDED
        rows = json.loads((tmp_path / "out" / "eval_results.json").read_text())
        assert rows[0]["checks"]["passed"] is True
        # The flywheel's reader consumes this file unchanged.
        from stateset_agents.flywheel import _eval_score

        assert _eval_score(tmp_path / "out") == (1, 1)


class RlModel(SamplingModel):
    """SamplingModel that also trains — the RL surface."""

    train_steps: list[dict] = []
    canned_rl: dict[str, list[dict]] = {}  # prompt -> samples w/ text/tokens/logprobs

    def sample(self, prompts=None, *, num_samples=1, temperature=1.0, **kw):
        class _S:
            def __init__(self, spec):
                self.text = spec["text"]
                self.tokens = spec.get("tokens", [1, 2])
                self.logprobs = spec.get("logprobs", [-0.5] * len(self.tokens))
                self.prompt_token_ids = spec.get("prompt_token_ids", [7, 8, 9])

        groups = []
        for prompt in prompts:
            canned = type(self).canned_rl.get(
                prompt, [{"text": "nothing"}] * num_samples
            )
            groups.append([_S(c) for c in canned[:num_samples]])
        return groups

    def train_step(self, data, lr, loss_fn="cross_entropy", **kw):
        type(self).train_steps.append({"n": len(data), "loss_fn": loss_fn, "lr": lr})
        return {"loss_mean": 0.1}, {"ok": True}


class RlClient(FakeRiverClient):
    def create_session(self):
        class _Session(FakeSession):
            def create_model(self, base_model, lora=None, checkpoint=None):
                model = RlModel(base_model=base_model, lora=lora)
                self.models.append(model)
                return model

        session = _Session()
        self.sessions.append(session)
        return session


class TestRiverRl:
    def _spec(self, tmp_path, **overrides):
        prompts = tmp_path / "prompts.json"
        prompts.write_text(
            json.dumps([{"prompt": "fix it", "expect": ["done"], "forbid": ["oops"]}])
        )
        defaults = {
            "dataset": prompts,
            "base_model": "Qwen/Qwen3.6-35B-A3B-FP8",
            "output_dir": tmp_path / "rl",
            "job_kind": "rl",
            "harvest": {"best_of": 2, "rounds": 2, "loss_fn": "cispo"},
        }
        defaults.update(overrides)
        return RemoteJobSpec(**defaults)

    def test_rounds_of_grouped_training_with_graded_rewards(self, tmp_path):
        RlModel.train_steps = []
        RlModel.canned_rl = {
            "fix it": [
                {"text": "done, no problem"},
                {"text": "oops I also did the forbidden thing done"},
            ]
        }
        client = RlClient()
        executor = RiverExecutor(
            client=client, tokenizer=FakeTokenizer(), ledger_path=tmp_path / "l.jsonl"
        )

        result = executor.wait(executor.submit(self._spec(tmp_path)))

        assert result.status is JobStatus.SUCCEEDED, "\n".join(result.logs)
        # 2 rounds x (2 samples w/ differing rewards -> 2 datums each).
        assert [s["loss_fn"] for s in RlModel.train_steps] == ["cispo", "cispo"]
        assert all(s["n"] == 2 for s in RlModel.train_steps)
        report = json.loads((tmp_path / "rl" / "rl_report.json").read_text())
        assert len(report["rounds"]) == 2
        # Graded reward: pass=1.0, forbid-hit = 1.0(expect) - 1.0 = 0.0 ->
        # group mean 0.5.
        assert report["rounds"][0]["mean_reward"] == 0.5
        assert client.model.saved  # checkpoint saved at the end

    def test_zero_variance_rounds_train_nothing(self, tmp_path):
        RlModel.train_steps = []
        RlModel.canned_rl = {"fix it": [{"text": "done"}, {"text": "done"}]}
        client = RlClient()
        executor = RiverExecutor(
            client=client, tokenizer=FakeTokenizer(), ledger_path=tmp_path / "l.jsonl"
        )

        result = executor.wait(executor.submit(self._spec(tmp_path)))

        assert result.status is JobStatus.SUCCEEDED
        assert RlModel.train_steps == []
        assert any("zero-variance" in line for line in result.logs)

    def test_eval_trajectory_is_recorded_per_round(self, tmp_path):
        RlModel.train_steps = []
        RlModel.canned_rl = {
            "fix it": [{"text": "done"}, {"text": "nope"}],
            "check me": [{"text": "the token is here"}],
        }
        client = RlClient()
        executor = RiverExecutor(
            client=client, tokenizer=FakeTokenizer(), ledger_path=tmp_path / "l.jsonl"
        )

        result = executor.wait(
            executor.submit(
                self._spec(
                    tmp_path,
                    eval_prompts=[{"prompt": "check me", "expect": ["token"]}],
                )
            )
        )

        assert result.status is JobStatus.SUCCEEDED
        report = json.loads((tmp_path / "rl" / "rl_report.json").read_text())
        rounds = report["rounds"]
        assert rounds[0]["round"] == 0 and rounds[0]["passed"] == 1
        assert all("passed" in r for r in rounds[1:])
        # Final greedy eval lands in the sft-shaped file the flywheel reads.
        assert (tmp_path / "rl" / "eval_results.json").exists()


class TestHarvestTransientRecovery:
    def test_harvest_retries_transients_with_a_fresh_session(self, tmp_path):
        """Observed live: a gen-2 harvest died on 'Server unavailable' and
        took a finished generation's momentum with it. Same retry policy as
        training."""

        class Flaky(Exception):
            pass

        class FlakyClient(SamplingClient):
            attempts = 0

            def create_session(self):
                type(self).attempts += 1
                if type(self).attempts == 1:
                    raise Flaky("upstream connect error")
                return super().create_session()

        FlakyClient.attempts = 0
        SamplingModel.canned = {"fix my vpn": ["vpn profile ok"] * 4}
        prompts = tmp_path / "p.json"
        prompts.write_text(
            json.dumps([{"prompt": "fix my vpn", "expect": ["vpn profile"]}])
        )
        executor = RiverExecutor(
            client=FlakyClient(),
            tokenizer=FakeTokenizer(),
            ledger_path=tmp_path / "l.jsonl",
        )
        executor._sleep = lambda s: None
        executor._transient_exceptions = lambda c: (Flaky,)

        spec = RemoteJobSpec(
            dataset=prompts,
            base_model="Qwen/Qwen3.5-9B",
            output_dir=tmp_path / "h",
            job_kind="harvest",
            harvest={"best_of": 4},
        )
        result = executor.wait(executor.submit(spec))

        assert result.status is JobStatus.SUCCEEDED, "\n".join(result.logs)
        assert FlakyClient.attempts == 2
        assert any("retrying harvest" in line for line in result.logs)
        summary = json.loads((tmp_path / "h" / "harvest_summary.json").read_text())
        assert summary["samples"] == 4  # counters reset across the retry
