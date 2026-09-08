"""Engine rollouts stay on-policy: after every optimizer step the trainers
push the policy's weights into the attached rollout backend, every turn
records which policy version sampled it, and an engine that cannot take
weights is reported as stale instead of silently sampling from an old policy.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent


class _SyncBackend:
    def __init__(self) -> None:
        self.synced: list = []

    async def generate_with_logprobs(self, prompts, **kwargs):
        return [
            SimpleNamespace(
                response="r",
                prompt_token_ids=[1, 2],
                response_token_ids=[3],
                token_logprobs=[-0.1],
            )
            for _ in prompts
        ]

    def sync_weights(self, model) -> None:
        self.synced.append(model)


class _NoSyncBackend:
    async def generate_with_logprobs(self, prompts, **kwargs):
        return await _SyncBackend().generate_with_logprobs(prompts, **kwargs)


class _BrokenSyncBackend(_SyncBackend):
    def sync_weights(self, model) -> None:
        raise RuntimeError("engine busy")


async def _agent() -> MultiTurnAgent:
    agent = MultiTurnAgent(
        AgentConfig(model_name="stub://x", use_stub_model=True, stub_responses=["ok"])
    )
    await agent.initialize()
    return agent


# --- agent-level contract --------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_pushes_model_and_bumps_version():
    agent = await _agent()
    backend = _SyncBackend()
    agent.set_rollout_backend(backend)
    assert agent.rollout_backend_version == 0
    assert agent.sync_rollout_backend() is True
    assert backend.synced == [agent.model]
    assert agent.rollout_backend_version == 1
    assert agent.rollout_backend_stale is False


@pytest.mark.asyncio
async def test_turns_record_the_policy_version_that_sampled_them():
    agent = await _agent()
    agent.set_rollout_backend(_SyncBackend())
    before = await agent.generate_turn("hi")
    agent.sync_rollout_backend()
    after = await agent.generate_turn("hi")
    assert before.metadata["rollout_backend_version"] == 0
    assert after.metadata["rollout_backend_version"] == 1


@pytest.mark.asyncio
async def test_backend_without_sync_is_reported_stale(caplog):
    agent = await _agent()
    agent.set_rollout_backend(_NoSyncBackend())
    with caplog.at_level("WARNING"):
        assert agent.sync_rollout_backend() is False
        assert agent.sync_rollout_backend() is False
    assert agent.rollout_backend_stale is True
    assert agent.rollout_backend_version == 0
    warnings = [r for r in caplog.records if "sync_weights" in r.getMessage()]
    assert len(warnings) == 1  # warned once, not per step
    turn = await agent.generate_turn("hi")
    assert turn.metadata["rollout_backend_stale"] is True


@pytest.mark.asyncio
async def test_failed_sync_marks_stale_and_keeps_error():
    agent = await _agent()
    agent.set_rollout_backend(_BrokenSyncBackend())
    assert agent.sync_rollout_backend() is False
    assert agent.rollout_backend_stale is True
    assert agent.rollout_backend_error.startswith("RuntimeError")
    # a later successful sync clears the flag
    agent.set_rollout_backend(_SyncBackend())
    assert agent.sync_rollout_backend() is True
    assert agent.rollout_backend_stale is False
    assert agent.rollout_backend_error is None


@pytest.mark.asyncio
async def test_sync_without_backend_is_a_noop():
    agent = await _agent()
    assert agent.sync_rollout_backend() is True
    assert agent.rollout_backend_version == 0


@pytest.mark.asyncio
async def test_reattaching_a_backend_resets_version_and_staleness():
    agent = await _agent()
    agent.set_rollout_backend(_NoSyncBackend())
    agent.sync_rollout_backend()
    assert agent.rollout_backend_stale is True
    agent.set_rollout_backend(_SyncBackend())
    assert agent.rollout_backend_stale is False
    assert agent.rollout_backend_version == 0


# --- trainer-level wiring --------------------------------------------------------

torch = pytest.importorskip("torch")


def _model():
    return torch.nn.Linear(2, 2)


def _config(**overrides):
    base = {
        "seed": 0,
        "bf16": False,
        "fp16": False,
        "use_reference_model": False,
        "report_to": None,
        "learning_rate": 0.1,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "num_generations": 2,
        "continual_strategy": "none",
        "gradient_accumulation_steps": 1,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _SyncingAgent:
    def __init__(self, model):
        self.model = model
        self.tokenizer = None
        self.initialize = AsyncMock()
        self.sync_calls = 0

    def sync_rollout_backend(self) -> bool:
        self.sync_calls += 1
        return True


def _multi_turn_trainer(agent, **cfg):
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    trainer = MultiTurnGRPOTrainer(
        agent=agent,
        environment=MagicMock(),
        reward_fn=MagicMock(),
        config=_config(**cfg),
    )
    trainer.optimizer = torch.optim.SGD(agent.model.parameters(), lr=0.1)
    trainer.scaler = None
    return trainer


def _single_turn_trainer(agent, **cfg):
    from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer

    trainer = object.__new__(SingleTurnGRPOTrainer)
    trainer.agent = agent
    trainer.config = _config(**cfg)
    trainer.optimizer = torch.optim.SGD(agent.model.parameters(), lr=0.1)
    trainer.scaler = None
    trainer.lr_scheduler = None
    trainer.global_step = 0
    return trainer


def _fake_grad(model):
    for p in model.parameters():
        p.grad = torch.ones_like(p)


def test_multi_turn_optimizer_step_syncs_rollout_backend():
    agent = _SyncingAgent(_model())
    trainer = _multi_turn_trainer(agent)
    _fake_grad(agent.model)
    trainer._apply_optimizer_step(torch)
    assert agent.sync_calls == 1
    assert trainer.last_rollout_sync is True


def test_single_turn_optimizer_step_syncs_rollout_backend():
    agent = _SyncingAgent(_model())
    trainer = _single_turn_trainer(agent)
    _fake_grad(agent.model)
    trainer._apply_optimizer_step(torch, False, 1.0)
    assert agent.sync_calls == 1
    assert trainer.last_rollout_sync is True


def test_rollout_sync_can_be_disabled():
    agent = _SyncingAgent(_model())
    trainer = _multi_turn_trainer(agent, rollout_sync=False)
    _fake_grad(agent.model)
    trainer._apply_optimizer_step(torch)
    assert agent.sync_calls == 0
    assert trainer.last_rollout_sync is None


def test_agents_without_sync_hook_are_fine():
    agent = SimpleNamespace(model=_model(), tokenizer=None, initialize=AsyncMock())
    trainer = _multi_turn_trainer(agent)
    _fake_grad(agent.model)
    trainer._apply_optimizer_step(torch)
    assert trainer.last_rollout_sync is None


def test_stale_sync_result_surfaces_in_trainer_state():
    agent = _SyncingAgent(_model())
    agent.sync_rollout_backend = lambda: False  # type: ignore[assignment]
    trainer = _multi_turn_trainer(agent)
    _fake_grad(agent.model)
    trainer._apply_optimizer_step(torch)
    assert trainer.last_rollout_sync is False


# --- VLLMGenerator.sync_weights ----------------------------------------------------


def test_vllm_generator_sync_weights_streams_named_parameters():
    from stateset_agents.training import vllm_backend

    gen = object.__new__(vllm_backend.VLLMGenerator)
    received: list[tuple[str, object]] = []
    gen.weight_loader = lambda weights: received.extend(
        (name, tensor.clone()) for name, tensor in weights
    )
    model = _model()
    gen.sync_weights(model)
    names = [n for n, _ in received]
    assert names == ["weight", "bias"]
    assert torch.equal(received[0][1], model.weight.detach())
    assert gen.weight_sync_count == 1


def test_vllm_generator_sync_weights_merges_lora_for_the_engine():
    from stateset_agents.training import vllm_backend

    class _PeftLike:
        """Looks like a PeftModel: merge/unmerge around the base model."""

        peft_config = {"default": object()}

        def __init__(self):
            self.base = _model()
            self.merged = False
            self.seen_merged: list[bool] = []

        def merge_adapter(self):
            self.merged = True

        def unmerge_adapter(self):
            self.merged = False

        def get_base_model(self):
            return self.base

    gen = object.__new__(vllm_backend.VLLMGenerator)
    peft = _PeftLike()
    seen: list[str] = []

    def loader(weights):
        for name, _ in weights:
            seen.append(name)
            peft.seen_merged.append(peft.merged)

    gen.weight_loader = loader
    gen.sync_weights(peft)
    assert seen == ["weight", "bias"]
    assert all(peft.seen_merged)  # weights were read while the adapter was merged
    assert peft.merged is False  # and the adapter is unmerged again for training


def test_vllm_generator_sync_weights_strips_wrapper_prefixes():
    from stateset_agents.training import vllm_backend

    names = vllm_backend._engine_weight_name("base_model.model.model.layers.0.q.weight")
    assert names == "model.layers.0.q.weight"
    assert vllm_backend._engine_weight_name("module.lm_head.weight") == "lm_head.weight"
    assert vllm_backend._engine_weight_name("lm_head.weight") == "lm_head.weight"
    # PEFT wraps each adapted Linear: <module>.base_layer.weight is the merged weight
    assert (
        vllm_backend._engine_weight_name(
            "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
        )
        == "model.layers.0.self_attn.q_proj.weight"
    )


def test_vllm_generator_sync_weights_without_engine_is_loud():
    from stateset_agents.training import vllm_backend

    gen = object.__new__(vllm_backend.VLLMGenerator)
    gen.engine = None
    gen.weight_loader = None
    with pytest.raises(RuntimeError, match="not initialized"):
        gen.sync_weights(_model())


def test_vllm_generator_resolves_engine_loader_from_known_paths():
    from stateset_agents.training import vllm_backend

    loaded: list[str] = []
    model_runner = SimpleNamespace(
        model=SimpleNamespace(load_weights=lambda ws: loaded.extend(n for n, _ in ws))
    )
    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(
            model_executor=SimpleNamespace(
                driver_worker=SimpleNamespace(model_runner=model_runner)
            )
        )
    )
    gen = object.__new__(vllm_backend.VLLMGenerator)
    gen.engine = engine
    gen.weight_loader = None
    gen.sync_weights(_model())
    assert loaded == ["weight", "bias"]


def test_vllm_generator_finds_the_model_through_v1_worker_nesting():
    """vLLM V1 keeps the model under engine_core → ... → worker → model_runner."""
    from stateset_agents.training import vllm_backend

    loaded: list[str] = []
    model = SimpleNamespace(load_weights=lambda ws: loaded.extend(n for n, _ in ws))
    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(
            engine_core=SimpleNamespace(
                engine_core=SimpleNamespace(
                    model_executor=SimpleNamespace(
                        driver_worker=SimpleNamespace(
                            worker=SimpleNamespace(
                                model_runner=SimpleNamespace(model=model)
                            )
                        )
                    )
                )
            )
        )
    )
    gen = object.__new__(vllm_backend.VLLMGenerator)
    gen.engine = engine
    gen.weight_loader = None
    gen.sync_weights(_model())
    assert loaded == ["weight", "bias"]
    assert gen.engine_model_path == "SimpleNamespace"


def test_vllm_config_keeps_engine_core_in_process_for_weight_sync(monkeypatch):
    from stateset_agents.training import vllm_backend

    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    created: dict = {}

    class _LLM:
        def __init__(self, **kwargs):
            created["env"] = dict(__import__("os").environ).get(
                "VLLM_ENABLE_V1_MULTIPROCESSING"
            )
            created["kwargs"] = kwargs

        def get_tokenizer(self):
            return object()

    monkeypatch.setattr(vllm_backend, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(vllm_backend, "LLM", _LLM)
    gen = vllm_backend.VLLMGenerator(vllm_backend.VLLMConfig(model_name="m"))
    assert __import__("asyncio").run(gen.initialize()) is True
    assert created["env"] == "0"

    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    gen = vllm_backend.VLLMGenerator(
        vllm_backend.VLLMConfig(model_name="m", in_process_weight_sync=False)
    )
    assert __import__("asyncio").run(gen.initialize()) is True
    assert created["env"] is None


def test_vllm_generator_unknown_engine_layout_names_the_problem():
    from stateset_agents.training import vllm_backend

    gen = object.__new__(vllm_backend.VLLMGenerator)
    gen.engine = SimpleNamespace()
    gen.weight_loader = None
    with pytest.raises(RuntimeError, match="weight_loader"):
        gen.sync_weights(_model())
