"""Tests for River AI 0.11 A+ integration: serving, deployments, MoE, telemetry & bug fixes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.remote.canary import run_provider_canary
from stateset_agents.remote.deployment import DeploymentSpec
from stateset_agents.remote.deployment_registry import (
    available_deployment_providers,
    get_deployment_provider,
)
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import JobStatus, RemoteJobSpec
from stateset_agents.remote.river import RiverExecutor
from stateset_agents.remote.river_rl import RiverRLConfig
from stateset_agents.remote.river_runtime import (
    inspect_server_capabilities,
    river_preflight,
)
from stateset_agents.remote.river_serving import (
    RiverDeploymentManager,
    RiverDeploymentProvider,
    RiverInferenceClient,
)
from stateset_agents.training.rl_blueprint import (
    RLBlueprintConfig,
    build_cloud_launch_plan,
)
from stateset_agents.training.studio_callback import StudioTelemetryCallback
from stateset_agents.weight_studio.finetune_engine import (
    PROVIDER_CATALOG,
    estimate_finetune_plan,
)
from stateset_agents.weight_studio.inspector import WeightInspector
from tests.unit.test_river_submit_golden import (
    RecordingClient,
    RecordingModel,
    _StubRenderer,
)


class _FakeAuthError(Exception):
    pass


_FakeAuthError.__name__ = "AuthenticationError"


class _FakeModelNotFoundError(Exception):
    pass


_FakeModelNotFoundError.__name__ = "ModelNotFoundError"


class _FakeCapacityError(Exception):
    pass


_FakeCapacityError.__name__ = "CapacityError"


@dataclass
class _FakeDeployment:
    id: str = "dep-river-1"
    model: str = "river-deployed-qwen"
    base_url: str = "https://api.river.ai/v1/deployments/dep-river-1"
    checkpoint: str = "river://checkpoints/sft-v1"
    base_model: str = "Qwen/Qwen3.5-9B"
    topology: str = "unified"
    phase: str = "ready"


@dataclass
class _FakeReplica:
    checkpoint: str = "river://checkpoints/sft-v1"
    status: str = "ready"
    base_url: str = "https://api.river.ai/v1/streaming"
    replica_id: str = "rep-1"
    model: str = "stateset-refund-live"
    base_model: str = "Qwen/Qwen3.5-9B"
    updated_at: str = "2026-11-23T00:00:00Z"
    status_reason: str | None = None


class _FakeServingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.closed = False

    def create_deployment(self, checkpoint: Any, **kwargs: Any) -> _FakeDeployment:
        self.calls.append(("create_deployment", (checkpoint, kwargs)))
        return _FakeDeployment()

    def list_deployments(self, **kwargs: Any) -> list[_FakeDeployment]:
        self.calls.append(("list_deployments", kwargs))
        return [_FakeDeployment()]

    def get_deployment(self, deployment_id: str, **kwargs: Any) -> _FakeDeployment:
        self.calls.append(("get_deployment", deployment_id))
        return _FakeDeployment(id=deployment_id)

    def wait_for_deployment(self, deployment_id: str, **kwargs: Any) -> _FakeDeployment:
        self.calls.append(("wait_for_deployment", (deployment_id, kwargs)))
        return _FakeDeployment(id=deployment_id, phase="ready")

    def scale_on_target(self, deployment_id: str, **kwargs: Any) -> _FakeDeployment:
        self.calls.append(("scale_on_target", (deployment_id, kwargs)))
        return _FakeDeployment(id=deployment_id)

    def delete_deployment(self, deployment_id: str, **kwargs: Any) -> _FakeDeployment:
        self.calls.append(("delete_deployment", (deployment_id, kwargs)))
        return _FakeDeployment(id=deployment_id, phase="deleted")

    def get_deployment_usage(self, deployment_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_deployment_usage", deployment_id))
        return {"deployment_id": deployment_id, "output_tokens": 4200, "requests": 19}

    def promote_streaming_replica(
        self, checkpoint: Any, model: str, **kwargs: Any
    ) -> _FakeReplica:
        self.calls.append(("promote_streaming_replica", (checkpoint, model)))
        return _FakeReplica(model=model)

    def get_streaming_replica(self, model: str, **kwargs: Any) -> _FakeReplica:
        self.calls.append(("get_streaming_replica", model))
        return _FakeReplica(model=model)

    def chat_complete_from_checkpoint(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        self.calls.append(("chat_complete_from_checkpoint", (messages, kwargs)))
        return SimpleNamespace(
            status_code=200,
            response_json=json.dumps(
                {
                    "id": "chatcmpl-river-1",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Refund ORD-77701 approved.",
                            },
                        }
                    ],
                }
            ),
        )

    def chat_complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.calls.append(("chat_complete", (messages, kwargs)))
        return SimpleNamespace(
            status_code=200,
            response_json=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Base model reply.",
                            }
                        }
                    ]
                }
            ),
        )

    def chat_complete_stream(self, messages: list[dict[str, Any]], **kwargs: Any):
        self.calls.append(("chat_complete_stream", (messages, kwargs)))
        yield {"choices": [{"delta": {"content": "Hello "}}]}
        yield {"choices": [{"delta": {"content": "River!"}}]}

    def close(self) -> None:
        self.closed = True


def _write_pointer_dir(tmp_path: Path) -> Path:
    ckpt_dir = tmp_path / "ckpt_ptr"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "river_checkpoint.json").write_text(
        json.dumps(
            {
                "provider": "river",
                "checkpoint": "river://checkpoints/sft-v1",
                "base_model": "Qwen/Qwen3.5-9B",
                "lora": {"rank": 16},
            }
        ),
        encoding="utf-8",
    )
    return ckpt_dir


def test_river_canary_closes_owned_client_and_detects_cleanup_failure() -> None:
    events: list[str] = []

    class _OwnedClient:
        def health_check(self) -> bool:
            return True

        def get_capabilities(self) -> list[str]:
            return ["Qwen/Qwen3.5-9B"]

        def get_server_capabilities(self) -> Any:
            return SimpleNamespace(
                supported_models=("Qwen/Qwen3.5-9B",),
                features=frozenset({"routing_replay", "deployments"}),
                model_features={"Qwen/Qwen3.5-9B": frozenset({"cispo"})},
            )

        def close(self) -> None:
            events.append("closed")

    owned_exec = RiverExecutor()
    owned_exec._get_client = lambda: _OwnedClient()  # type: ignore[method-assign]
    res = run_provider_canary("river", executor=owned_exec)
    assert res.ok and res.cleanup_verified is True
    assert events == ["closed"]
    assert res.checks["server_features"] == ["deployments", "routing_replay"]

    class _BrokenCloseClient(_OwnedClient):
        def close(self) -> None:
            raise RuntimeError("close failed")

    broken_exec = RiverExecutor()
    broken_exec._get_client = lambda: _BrokenCloseClient()  # type: ignore[method-assign]
    res_fail = run_provider_canary("river", executor=broken_exec)
    assert res_fail.status == "failed" and res_fail.cleanup_verified is False


def test_inspect_server_capabilities_and_preflight(monkeypatch) -> None:
    scaps_client = SimpleNamespace(
        get_capabilities=lambda: ["Qwen/Qwen3.6-35B-A3B-FP8"],
        get_server_capabilities=lambda **_: SimpleNamespace(
            supported_models=("Qwen/Qwen3.6-35B-A3B-FP8",),
            features=frozenset({"dro", "routing_replay"}),
            model_features={"Qwen/Qwen3.6-35B-A3B-FP8": frozenset({"cispo", "moe"})},
        ),
        close=lambda: None,
    )
    parsed = inspect_server_capabilities(scaps_client)
    assert parsed["supported_models"] == ["Qwen/Qwen3.6-35B-A3B-FP8"]
    assert parsed["features"] == ["dro", "routing_replay"]
    assert parsed["model_features"]["Qwen/Qwen3.6-35B-A3B-FP8"] == ["cispo", "moe"]

    # Fallback client without get_server_capabilities
    legacy = SimpleNamespace(get_capabilities=lambda: ["Qwen/Qwen3.5-9B"])
    assert inspect_server_capabilities(legacy)["supported_models"] == [
        "Qwen/Qwen3.5-9B"
    ]

    monkeypatch.setenv("RIVER_API_KEY", "rv_live_secret")
    monkeypatch.setattr(
        "stateset_agents.remote.river_runtime.inspect_native_runtime",
        lambda: {
            "python": {"version": "3.12.3", "passed": True},
            "credentials": {"configured": True},
            "river_sdk": {"version": "0.11.0", "passed": True},
        },
    )
    monkeypatch.setattr(
        "stateset_agents.remote.river_runtime.importlib.import_module",
        lambda name: SimpleNamespace(Client=lambda **_: scaps_client),
    )
    rep = river_preflight(["Qwen/Qwen3.6-35B-A3B-FP8"], live=True)
    assert rep["passed"] is True
    assert rep["account"]["server_features"] == ["dro", "routing_replay"]


def test_river_typed_exceptions_capacity_retry_and_tokenizer_alias(monkeypatch) -> None:
    ex = RiverExecutor()
    auth_err = ex._account_error(_FakeAuthError("bad key"))
    assert auth_err is not None and "AuthenticationError" in str(auth_err)
    nf_err = ex._account_error(_FakeModelNotFoundError("nope"))
    assert nf_err is not None and "ModelNotFoundError" in str(nf_err)

    fake_mod = SimpleNamespace(
        RiverConnectionError=ConnectionError,
        CapacityError=_FakeCapacityError,
    )
    trans = ex._transient_exceptions(fake_mod)
    assert any(c.__name__ == "CapacityError" for c in trans)

    loaded_names: list[str] = []
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda name: loaded_names.append(name) or SimpleNamespace(name=name),
    )
    ex._get_tokenizer("Qwen/Qwen3.6-35B-A3B-FP8")
    assert loaded_names == ["Qwen/Qwen3.6-35B-A3B"]


def test_river_rl_dro_moe_routing_idempotency_and_studio_telemetry(
    tmp_path: Path, monkeypatch
) -> None:
    try:
        monkeypatch.setattr(
            "stateset_agents.remote.river_renderers.get_river_renderer",
            lambda *_a, **_kw: _StubRenderer(),
        )
    except ImportError:
        monkeypatch.setattr(
            "river_client.renderers.get_renderer",
            lambda *_a, **_kw: _StubRenderer(),
        )
    assert RiverRLConfig(loss_fn="dro").loss_kwargs() == {}
    assert RiverRLConfig(loss_fn="decoupled_ppo").loss_kwargs() == {
        "clip_low": 0.2,
        "clip_high": 0.2,
    }

    fb_calls: list[dict[str, Any]] = []
    opt_calls: list[dict[str, Any]] = []

    class _MoERecordingModel(RecordingModel):
        def forward_backward(
            self, data: Any, loss_fn: str = "cross_entropy", **kwargs: Any
        ) -> Any:
            fb_calls.append({"loss_fn": loss_fn, **kwargs})
            return SimpleNamespace(
                metrics={
                    "loss": 0.25,
                    "loss_mean": 0.125,
                    "moe/expert_flip_rate": 0.04,
                }
            )

        def optim_step(self, **kwargs: Any) -> Any:
            opt_calls.append(dict(kwargs))
            return super().optim_step(**kwargs)

    prompts_path = tmp_path / "rl_prompts.json"
    prompts_path.write_text(
        json.dumps([{"prompt": "fix vpn", "expect": ["done", "vpn"]}]),
        encoding="utf-8",
    )
    client = RecordingClient(
        texts=("done vpn profile", "oops nope"),
        model_cls=_MoERecordingModel,
    )
    cb = StudioTelemetryCallback(run_name="river-moe-rl", workspace_root=tmp_path)
    spec = RemoteJobSpec(
        dataset=prompts_path,
        base_model="Qwen/Qwen3.5-9B",
        output_dir=tmp_path / "rl_out",
        job_kind="rl",
        lora_r=8,
        harvest={
            "rounds": 1,
            "best_of": 2,
            "loss_fn": "dro",
            "microbatch_size": 8,
        },
    )
    ex = RiverExecutor(
        client=client,
        studio_callback=cb,
        routing_replay=True,
        compute_expert_flip_metric=True,
        use_idempotency_keys=True,
    )
    handle = ex.submit(spec)
    assert ex.status(handle) is JobStatus.SUCCEEDED
    assert fb_calls and fb_calls[0]["force_routing_replay"] is True
    assert fb_calls[0]["compute_expert_flip_metric"] is True
    assert fb_calls[0]["loss_fn"] == "dro"
    assert opt_calls and str(opt_calls[0]["idempotency_key"]).startswith("stateset-")
    assert cb.steps_logged == 1
    report = json.loads((Path(spec.output_dir) / "rl_report.json").read_text())
    assert report["rounds"][-1]["expert_flip_rate"] == pytest.approx(0.04)

    # decoupled_ppo + routing_replay must fail closed matching River 0.11
    spec_bad = RemoteJobSpec(
        dataset=prompts_path,
        base_model="Qwen/Qwen3.5-9B",
        output_dir=tmp_path / "rl_bad_decoupled",
        job_kind="rl",
        lora_r=8,
        harvest={"rounds": 1, "best_of": 2, "loss_fn": "decoupled_ppo"},
    )
    ex_bad = RiverExecutor(
        client=RecordingClient(
            texts=("done vpn profile", "oops nope"), model_cls=_MoERecordingModel
        ),
        routing_replay=True,
    )
    with pytest.raises(RemoteExecutionError, match="force_routing_replay"):
        ex_bad.submit(spec_bad)


def test_river_sft_attest_training_data(tmp_path: Path) -> None:
    from tests.unit.test_river_submit_golden import RecordingSession

    ds = tmp_path / "sft.jsonl"
    ds.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Order ORD-1 status?"},
                    {"role": "assistant", "content": "Order ORD-1 is delivered."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    attested: list[Any] = []

    class _AttestSession(RecordingSession):
        def attest_training_data(self, artifacts: list[Any]) -> Any:
            attested.extend(artifacts)
            return SimpleNamespace(training_data_attestation_id="att-river-999")

    class _AttestClient(RecordingClient):
        @dataclass
        class TrainingDataArtifact:
            name: str
            expected_sha256: str
            content: bytes

        def session(self, project: str | None = None) -> Any:
            self.calls.append({"call": "session", "project": project})
            return _AttestSession(self.calls, self._model_cls, self._texts)

    class _MiniTok:
        def apply_chat_template(
            self, messages: list[dict[str, Any]], tokenize: bool = False, **kwargs: Any
        ) -> str:
            out = "".join(
                f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages
            )
            if kwargs.get("add_generation_prompt"):
                out += "<assistant>"
            return out

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [ord(ch) for ch in text]

    client = _AttestClient()
    spec = RemoteJobSpec(
        dataset=ds,
        base_model="Qwen/Qwen3.5-9B",
        output_dir=tmp_path / "sft_out",
        num_epochs=1,
        lora_r=8,
    )
    ex = RiverExecutor(client=client, tokenizer=_MiniTok(), attest_training_data=True)
    handle = ex.submit(spec)
    ex.fetch(handle)
    assert len(attested) == 1 and attested[0].name == "sft.jsonl"
    ptr = json.loads((tmp_path / "sft_out" / "river_checkpoint.json").read_text())
    assert ptr["training_data_attestation_id"] == "att-river-999"


def test_river_serving_deployments_streaming_and_cli(
    tmp_path: Path, monkeypatch
) -> None:
    ckpt_dir = _write_pointer_dir(tmp_path)
    fake_client = _FakeServingClient()

    assert "river" in available_deployment_providers()
    assert isinstance(get_deployment_provider("river"), RiverDeploymentProvider)

    with RiverDeploymentManager(client=fake_client) as mgr:
        dep = mgr.create_deployment(
            ckpt_dir, unified_replicas=2, idempotency_key="idem-1", wait=True
        )
        assert dep["id"] == "dep-river-1"
        assert mgr.list_deployments()[0]["id"] == "dep-river-1"
        assert mgr.get_deployment("dep-river-1")["phase"] == "ready"
        assert mgr.wait_for_deployment("dep-river-1")["phase"] == "ready"
        assert (
            mgr.scale_deployment("dep-river-1", unified_replicas=3)["id"]
            == "dep-river-1"
        )
        assert mgr.get_deployment_usage("dep-river-1")["output_tokens"] == 4200
        rep = mgr.promote_streaming_replica(ckpt_dir, "stateset-refund-live")
        assert rep["model"] == "stateset-refund-live"
        assert mgr.get_streaming_replica("stateset-refund-live")["status"] == "ready"
        chat = mgr.chat_complete(
            [{"role": "user", "content": "Refund?"}], checkpoint=ckpt_dir
        )
        assert chat["text"] == "Refund ORD-77701 approved."
        assert chat["base_model"] == "Qwen/Qwen3.5-9B"
        chunks = list(
            mgr.chat_complete_stream(
                [{"role": "user", "content": "Hi"}], model="stateset-refund-live"
            )
        )
        assert len(chunks) == 2
        assert mgr.delete_deployment("dep-river-1")["phase"] == "deleted"
    # Injected client stays open
    assert fake_client.closed is False

    # RiverDeploymentProvider (InferenceDeploymentProvider contract)
    provider = RiverDeploymentProvider(client=fake_client)
    handle = provider.deploy(
        DeploymentSpec(
            name="refund-prod",
            model_name="Qwen/Qwen3.5-9B",
            weights_uri="river://checkpoints/sft-v1",
            gpu="river-managed",
            min_replicas=1,
            max_replicas=2,
        )
    )
    assert handle.provider == "river" and handle.deployment_id == "dep-river-1"
    st = provider.status(handle)
    assert st["usage"]["output_tokens"] == 4200
    provider.delete(handle)

    # RiverInferenceClient convenience wrapper
    with RiverInferenceClient(
        ckpt_dir, streaming_model="stateset-refund-live", client=fake_client
    ) as inf:
        assert inf.base_model == "Qwen/Qwen3.5-9B"
        assert inf.generate("Refund ORD-77701?") == "Refund ORD-77701 approved."
        assert len(list(inf.stream_chat([{"role": "user", "content": "Hi"}]))) == 2

    # CLI commands: river-chat, river-deploy, river-deployments
    monkeypatch.setattr(
        "stateset_agents.remote.river_serving.RiverDeploymentManager._get_client",
        lambda self: self._client if self._client is not None else fake_client,
    )
    runner = CliRunner()
    r_chat = runner.invoke(
        app, ["river-chat", "--checkpoint", str(ckpt_dir), "--prompt", "Refund?"]
    )
    assert r_chat.exit_code == 0 and "Refund ORD-77701 approved." in r_chat.stdout

    r_dep = runner.invoke(
        app,
        ["river-deploy", "--checkpoint", str(ckpt_dir), "--unified-replicas", "1"],
    )
    assert r_dep.exit_code == 0 and "dep-river-1" in r_dep.stdout

    r_prom = runner.invoke(
        app,
        [
            "river-deploy",
            "--checkpoint",
            str(ckpt_dir),
            "--promote-streaming-model",
            "stateset-live",
        ],
    )
    assert r_prom.exit_code == 0 and "stateset-live" in r_prom.stdout

    r_usage = runner.invoke(
        app, ["river-deployments", "--deployment-id", "dep-river-1", "--usage"]
    )
    assert r_usage.exit_code == 0 and "4200" in r_usage.stdout

    # Additional serving error/validation paths & secret redaction
    base_chat = RiverDeploymentManager(client=fake_client).chat_complete(
        [{"role": "user", "content": "Hi"}], base_model="Qwen/Qwen3.5-9B"
    )
    assert base_chat["text"] == "Base model reply."
    monkeypatch.setenv("RIVER_API_KEY", "rv_super_secret_token")

    class _ExplodingServingClient(_FakeServingClient):
        def list_deployments(self, **kwargs: Any) -> list[_FakeDeployment]:
            raise RuntimeError("auth error with rv_super_secret_token")

    with pytest.raises(RemoteExecutionError) as exc_info:
        RiverDeploymentManager(client=_ExplodingServingClient()).list_deployments()
    assert "rv_super_secret_token" not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)

    with pytest.raises(ValueError):
        RiverDeploymentManager(client=fake_client).scale_deployment("dep-1")
    with pytest.raises(ValueError):
        RiverDeploymentManager(client=fake_client).chat_complete(
            [{"role": "user", "content": "Hi"}]
        )


def test_river_blueprint_studio_plan_and_improvement_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    cfg = RLBlueprintConfig(
        name="river-cispo-v1",
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        algorithm="cispo",
        cloud="river",
        lora_rank=64,
        max_steps=10,
        group_size=8,
    )
    plan = build_cloud_launch_plan(cfg, workspace_root=tmp_path)
    assert plan["dispatch"]["provider"] == "river"
    assert plan["dispatch"]["lora_rank"] == 32  # clamped to River max rank 32
    assert plan["dispatch"]["pricing_model"] == "token_metered_unknown_usd"
    assert "flywheel --provider river --algorithm cispo" in plan["dispatch"]["command"]

    # Finetune studio honors River token-metered pricing & emits real River CLI
    assert PROVIDER_CATALOG["river"]["hourly_usd"] == 0.0
    assert PROVIDER_CATALOG["river"]["cost_unknown"] is True
    inspector = WeightInspector(workspace_root=tmp_path)
    sft_plan = estimate_finetune_plan(
        inspector,
        "preset:deepseek-r1-1.5b",
        {"provider": "river", "algorithm": "sft", "lora_r": 16},
    )
    assert "train-remote --provider river" in sft_plan["cli_command"]
    rl_plan = estimate_finetune_plan(
        inspector,
        "preset:deepseek-r1-1.5b",
        {"provider": "river", "algorithm": "cispo", "lora_r": 16},
    )
    assert "flywheel --provider river --algorithm cispo" in rl_plan["cli_command"]

    # Resilient verify_sampler_runtime behavioral fallback (when improvement module is present)
    try:
        from stateset_agents.improvement.river import verify_sampler_runtime
    except ImportError:
        return

    exact = verify_sampler_runtime()
    assert exact["mode"] == "exact_sha256" and exact["verified"] is True
    monkeypatch.setattr(
        "stateset_agents.improvement.river.AUDITED_ENGINE_SHA256S",
        frozenset({"0" * 64}),
    )
    with pytest.raises(ValueError, match="River rollout engine changed"):
        verify_sampler_runtime()
    behav = verify_sampler_runtime(allow_behavioral_fallback=True)
    assert behav["mode"] == "behavioral_signature" and behav["verified"] is True
