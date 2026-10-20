"""Reject incompatible packaged training before transport or paid allocation."""

import subprocess
import sys
from pathlib import Path

import pytest

from stateset_agents.core.errors import ErrorCode
from stateset_agents.core.frontier_models import get_frontier_model
from stateset_agents.remote.coreweave import CoreWeaveExecutor
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.fireworks import FireworksExecutor
from stateset_agents.remote.huggingface import HuggingFaceJobsExecutor
from stateset_agents.remote.job import JobStatus, RemoteJobSpec
from stateset_agents.remote.local import LocalExecutor
from stateset_agents.remote.modal import ModalExecutor
from stateset_agents.remote.nebius import NebiusExecutor
from stateset_agents.remote.prime import PrimeLabExecutor
from stateset_agents.remote.river import RiverExecutor
from stateset_agents.remote.runpod import RunPodExecutor
from stateset_agents.remote.tinker import TinkerExecutor
from stateset_agents.remote.together import TogetherExecutor

DATED = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
ORIGINAL = "deepseek-ai/DeepSeek-V4.1-Flash"
PACKAGED = ("local", "runpod", "modal", "coreweave", "nebius", "huggingface")
MANAGED = ("river", "tinker", "fireworks", "together", "prime")


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(
        '{"messages":[{"role":"user","content":"hello"},'
        '{"role":"assistant","content":"hi"}]}\n'
    )
    return path


def executor_for(provider, tmp_path):
    """Inject inert clients so construction cannot reach external SDKs."""
    if provider == "local":
        return LocalExecutor()
    if provider == "runpod":
        return RunPodExecutor(
            api=object(),
            ssh=object(),
            public_key="ssh-ed25519 TEST-ONLY",
            lease_dir=tmp_path / "leases",
            ledger_path=tmp_path / "ledger.jsonl",
        )
    if provider == "modal":
        return ModalExecutor(secret_names=())
    if provider == "coreweave":
        return CoreWeaveExecutor(
            kube=object(),
            artifact_store=object(),
            state_dir=tmp_path / "coreweave",
        )
    if provider == "nebius":
        return NebiusExecutor(
            cli=object(),
            artifact_store=object(),
            state_dir=tmp_path / "nebius",
        )
    if provider == "huggingface":
        return HuggingFaceJobsExecutor(api=object())
    if provider == "river":
        return RiverExecutor(client=object(), ledger_path=tmp_path / "ledger.jsonl")
    if provider == "tinker":
        return TinkerExecutor(service_client=object(), tinker_module=object())
    if provider == "fireworks":
        return FireworksExecutor(client=object(), state_dir=tmp_path / "fireworks")
    if provider == "together":
        return TogetherExecutor(client=object())
    if provider == "prime":
        return PrimeLabExecutor(command=("never-execute",))
    raise AssertionError(provider)


def make_spec(dataset, tmp_path, model=DATED, **kwargs):
    return RemoteJobSpec(
        dataset=dataset,
        base_model=model,
        output_dir=tmp_path / "output",
        num_epochs=1,
        **kwargs,
    )


def block_submission_effects(executor, patch):
    """Fail loudly if validation reaches allocation, uploads or local writes."""
    calls = []

    def forbid(name):
        def unexpected(*args, **kwargs):
            calls.append(name)
            pytest.fail(f"Rejected submission reached {name}")

        return unexpected

    for name in (
        "_require_api",
        "_require_public_key",
        "_require_ssh",
        "_run_attempt",
        "_require_sdk",
        "build_image",
        "_run_remote",
        "_upload_dataset",
        "_download",
        "_delete_volume",
        "_entrypoint_args",
        "_client",
        "_require_configuration",
    ):
        if callable(getattr(executor, name, None)):
            patch.setattr(executor, name, forbid(name))
    patch.setattr("subprocess.run", forbid("subprocess.run"))
    patch.setattr("subprocess.Popen", forbid("subprocess.Popen"))
    patch.setattr(Path, "mkdir", forbid("Path.mkdir"))
    return calls


@pytest.mark.parametrize("provider", ["runpod", "modal"])
@pytest.mark.parametrize(
    "model,dry_run",
    [
        (DATED, False),
        ("deepseek-v4.1-flash-1017", False),
        (ORIGINAL, False),
        ("deepseek-v4.1-flash", False),
        (DATED.upper(), False),
        (f"  {DATED}  ", False),
        (DATED, True),
        ("deepseek-v4.1-flash", True),
    ],
)
def test_blocked_frontier_submit_precedes_clients_uploads_and_local_writes(
    dataset, tmp_path, monkeypatch, provider, model, dry_run
):
    executor = executor_for(provider, tmp_path)
    spec = make_spec(dataset, tmp_path, model, dry_run=dry_run)
    frontier = get_frontier_model(model.strip())
    before = set(tmp_path.rglob("*"))
    with monkeypatch.context() as patch:
        calls = block_submission_effects(executor, patch)
        with pytest.raises(
            RemoteExecutionError, match="training is blocked"
        ) as rejected:
            executor.submit(spec)
    assert rejected.value.code is ErrorCode.CFG_INVALID
    details = rejected.value.context.details
    assert details["provider"] == provider
    assert details["model"] == frontier.model_id
    assert details["model_revision"] == frontier.revision
    assert details["blockers"] == list(frontier.blockers)
    assert calls == [] and executor._jobs == {}
    assert set(tmp_path.rglob("*")) == before and not spec.output_dir.exists()


@pytest.mark.parametrize("provider", PACKAGED)
def test_packaged_worker_readiness_and_admission_are_explicit(
    dataset, tmp_path, monkeypatch, provider
):
    executor = executor_for(provider, tmp_path)
    assert executor.uses_packaged_training is True
    assert executor.capabilities()["uses_packaged_training"] is True
    before = set(tmp_path.rglob("*"))
    with monkeypatch.context() as patch:
        calls = block_submission_effects(executor, patch)
        with pytest.raises(
            RemoteExecutionError, match="training is blocked"
        ) as rejected:
            executor.submit(make_spec(dataset, tmp_path))
    assert rejected.value.code is ErrorCode.CFG_INVALID
    assert rejected.value.context.details["provider"] == provider
    assert calls == [] and set(tmp_path.rglob("*")) == before
    assert getattr(executor, "_jobs", {}) == {}
    assert getattr(executor, "_specs", {}) == {}

    # A native FP8 implementation can train through frontier-sft's distributed
    # dequantization recipe, but not this generic packaged SFT worker.
    fp8 = get_frontier_model("glm5.3-flash")
    assert not fp8.blockers
    with pytest.raises(RemoteExecutionError, match="frontier-sft") as rejected:
        executor.validate_spec(make_spec(dataset, tmp_path, fp8.model_id))
    assert rejected.value.code is ErrorCode.CFG_INVALID
    assert rejected.value.context.details["model"] == fp8.model_id
    assert rejected.value.context.details["model_revision"] == fp8.revision
    assert rejected.value.context.details["blockers"]

    # Admission is not a global model allowlist or proof that an admitted
    # released checkpoint fits hardware or has completed training.
    for name in ("acme/private-model", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.8-Flash-Next"):
        executor.validate_spec(make_spec(dataset, tmp_path, name))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("provider", MANAGED)
def test_managed_loaders_keep_their_independent_model_capability_contract(
    dataset, tmp_path, monkeypatch, provider
):
    executor = executor_for(provider, tmp_path)
    assert executor.uses_packaged_training is False
    assert executor.capabilities()["uses_packaged_training"] is False
    for kind in executor.supported_job_kinds:
        for model in (DATED, ORIGINAL):
            executor.validate_spec(make_spec(dataset, tmp_path, model, job_kind=kind))

    if provider == "river":
        from tests.unit.test_remote_river_executor import FakeRiverClient, FakeTokenizer

        def unavailable(*args, **kwargs):
            raise ValueError("recording fake uses raw-text sampling")

        monkeypatch.setattr(
            "stateset_agents.remote.river_renderers.get_river_renderer", unavailable
        )
        client = FakeRiverClient()
        client.get_capabilities = lambda: [DATED]
        executor = RiverExecutor(
            client=client,
            tokenizer=FakeTokenizer(),
            ledger_path=tmp_path / "ledger.jsonl",
        )
        handle = executor.submit(make_spec(dataset, tmp_path))
        assert executor.status(handle) is JobStatus.SUCCEEDED
        assert client.model.base_model == DATED and client.model.optim_steps
        assert client.model.saved

        # The local-loader exemption does not bypass account access. This is a
        # fake transport/codec contract, not evidence of live River 1017 support.
        client.get_capabilities = lambda: []
        monkeypatch.setattr(
            executor, "_get_tokenizer", lambda *_: pytest.fail("denied model tokenized")
        )
        with pytest.raises(RemoteExecutionError, match="does not advertise"):
            executor.submit(make_spec(dataset, tmp_path))
        assert len(client.sessions) == 1


@pytest.mark.parametrize("provider", ["local", "runpod", "huggingface"])
def test_constructor_normalizes_modes_and_keeps_harvest_inference_admission(
    dataset, tmp_path, provider
):
    executor = executor_for(provider, tmp_path)
    normal = make_spec(dataset, tmp_path, "acme/private-model", job_kind=" SFT ")
    assert normal.job_kind == "sft"
    executor.validate_spec(normal)
    harvest = make_spec(dataset, tmp_path, job_kind=" HaRvEsT ")
    assert harvest.job_kind == "harvest"
    executor.validate_spec(harvest)
    # Only the training blocker is bypassed: native inference compatibility and
    # provider access still require their separate checks before real execution.
    assert not harvest.output_dir.exists()


@pytest.mark.parametrize("provider", ["local", "runpod", "modal"])
@pytest.mark.parametrize(
    "mutated",
    ["SFT", "HARVEST", ["sft"], None],
    ids=["upper-sft", "upper-harvest", "list", "none"],
)
def test_mutated_job_modes_cannot_fall_through_into_packaged_sft(
    dataset, tmp_path, monkeypatch, provider, mutated
):
    executor = executor_for(provider, tmp_path)
    spec = make_spec(dataset, tmp_path, "acme/private-model")
    spec.job_kind = mutated
    before = set(tmp_path.rglob("*"))
    with monkeypatch.context() as patch:
        calls = block_submission_effects(executor, patch)
        with pytest.raises(RemoteExecutionError, match="canonical mode") as rejected:
            executor.submit(spec)
    assert rejected.value.code is ErrorCode.CFG_INVALID
    assert rejected.value.context.details["provider"] == provider
    assert calls == [] and executor._jobs == {}
    assert getattr(executor, "_counter", 0) == 0
    assert set(tmp_path.rglob("*")) == before and not spec.output_dir.exists()


def test_blocked_native_admission_is_cold_and_does_not_import_optional_ml(
    dataset, tmp_path
):
    code = """
import sys
from pathlib import Path
from stateset_agents.core.errors import ErrorCode
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import RemoteJobSpec
from stateset_agents.remote.local import LocalExecutor

optional = {'torch', 'transformers', 'trl', 'peft', 'accelerate', 'datasets'}
assert not optional.intersection(sys.modules)
spec = RemoteJobSpec(
    dataset=Path(sys.argv[1]),
    base_model='deepseek-ai/DeepSeek-V4.1-Flash-1017',
    output_dir=Path(sys.argv[2]),
)
executor = LocalExecutor()
for operation in (executor.validate_spec, executor.submit):
    try:
        operation(spec)
    except RemoteExecutionError as error:
        assert error.code is ErrorCode.CFG_INVALID
        assert error.context.details['model_revision'] == '1bae0e1088993b9b82de08f59de5be5a34e42170'
    else:
        raise AssertionError('Blocked model passed packaged training admission')
assert executor._counter == 0 and executor._jobs == {}
assert not spec.output_dir.exists()
assert not optional.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(dataset), str(tmp_path / "cold-output")],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
