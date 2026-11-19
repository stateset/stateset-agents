"""Backend initialization preserves completed work and fences uncertain adapters."""

import asyncio
import copy
import inspect
import pickle
import threading
from types import SimpleNamespace

import pytest

from stateset_agents.core import BackendInitializationState
from stateset_agents.core import agent as agents


def configured_agent(monkeypatch, attach):
    monkeypatch.setattr(agents, "_load_transformers_agent", lambda: True)
    monkeypatch.setattr(agents, "_load_peft", lambda: True)
    monkeypatch.setattr(agents, "LoraConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(agents, "get_peft_model", attach)
    config = agents.AgentConfig(
        model_name="local-test",
        use_peft=True,
        peft_config={"r": 4},
        use_chat_template=False,
    )
    return agents.MultiTurnAgent(
        config,
        model_loader=lambda config: SimpleNamespace(attached=False),
        tokenizer_loader=lambda config: SimpleNamespace(
            pad_token="pad", eos_token="eos"
        ),
        generation_config_factory=lambda *args: SimpleNamespace(max_new_tokens=10),
    )


@pytest.mark.asyncio
async def test_repeated_stub_initialization_preserves_model_identity_and_progress():
    agent = agents.MultiTurnAgent(
        agents.AgentConfig(
            model_name="stub://repeat",
            stub_responses=["first response", "second response"],
        )
    )
    await agent.initialize()
    model = agent.model
    assert "first response" in await agent.generate_response("Hello")
    await agent.initialize()
    assert agent.model is model
    assert "second response" in await agent.generate_response("Hello again")


@pytest.mark.asyncio
async def test_repeated_initialization_attaches_adapter_once(monkeypatch):
    attached = []

    def attach(model, config):
        attached.append(model)
        return SimpleNamespace(base_model=model)

    agent = configured_agent(monkeypatch, attach)
    await agent.initialize()
    wrapped = agent.model
    await agent.initialize()
    assert agent.model is wrapped
    assert len(attached) == 1
    status = agent.get_initialization_status()
    status["adapter_request"]["config"]["r"] = 100
    assert agent.get_initialization_status()["adapter_request"]["config"]["r"] == 4


@pytest.mark.asyncio
async def test_uncertain_adapter_attachment_is_not_replayed(monkeypatch):
    attempts = []

    def attach(model, config):
        attempts.append(model)
        model.attached = True
        raise RuntimeError("Adapter attachment failed after mutation")

    agent = configured_agent(monkeypatch, attach)
    with pytest.raises(RuntimeError, match="after mutation"):
        await agent.initialize()
    with pytest.raises(RuntimeError):
        await agent.initialize()
    assert len(attempts) == 1
    assert agent.model is attempts[0]


@pytest.mark.asyncio
async def test_failed_adapter_initialization_cannot_generate(monkeypatch):
    def attach(model, config):
        raise RuntimeError("Adapter failed")

    agent = configured_agent(monkeypatch, attach)
    with pytest.raises(RuntimeError, match="Adapter failed"):
        await agent.initialize()

    def must_not_tokenize(prompt):
        pytest.fail("A failed adapter must not reach generation")

    monkeypatch.setattr(agent, "_tokenize_prompt", must_not_tokenize)
    with pytest.raises(RuntimeError, match="initialization"):
        await agent.generate_response("Hello")


@pytest.mark.asyncio
async def test_status_observation_is_passive_and_describes_current_handle_binding():
    agent = agents.Agent()
    assert (
        agent.get_initialization_status()["state"]
        == BackendInitializationState.NEW.value
    )
    assert agent.model is None
    await agent.initialize()
    status = agent.get_initialization_status()
    assert status["state"] == "ready" and status["backend_matches"]
    status["state"] = "modified"
    assert agent.get_initialization_status()["state"] == "ready"
    agent.model = object()
    assert not agent.get_initialization_status()["backend_matches"]


@pytest.mark.asyncio
async def test_injected_backend_is_not_rebound_over_its_attached_adapter(monkeypatch):
    calls = []

    def attach(model, config):
        calls.append(model)
        return SimpleNamespace(base_model=model)

    agent = configured_agent(monkeypatch, attach)
    backend = SimpleNamespace(
        model=object(), tokenizer=object(), generation_config=object()
    )
    agent._injected_backend = backend
    await agent.initialize()
    wrapped = agent.model
    await agent.initialize()
    assert agent.model is wrapped and wrapped.base_model is backend.model
    assert calls == [backend.model]


@pytest.mark.asyncio
async def test_injected_property_failure_cannot_publish_partial_components():
    class Backend:
        tokenizer = object()
        generation_config = object()
        ready = False

        @property
        def model(self):
            if not self.ready:
                raise RuntimeError("Backend unavailable")
            return self

    backend = Backend()
    agent = agents.Agent(backend=backend)
    with pytest.raises(RuntimeError, match="Backend unavailable"):
        await agent.initialize()
    assert (agent.model, agent.tokenizer, agent.generation_config) == (None, None, None)
    assert agent.get_initialization_status()["state"] == "failed"
    backend.ready = True
    await agent.initialize()
    assert (
        agent.model is backend and agent.get_initialization_status()["state"] == "ready"
    )


@pytest.mark.asyncio
async def test_generation_config_retry_reuses_completed_loads(monkeypatch):
    counts = {"tokenizer": 0, "model": 0, "config": 0}
    agent = configured_agent(monkeypatch, lambda model, config: model)
    agent.config.use_peft = False

    def tokenizer(config):
        counts["tokenizer"] += 1
        return SimpleNamespace(pad_token="pad")

    def model(config):
        counts["model"] += 1
        return object()

    def generation_config(*args):
        counts["config"] += 1
        if counts["config"] == 1:
            raise ValueError("Configuration failed")
        return object()

    agent._model_loader, agent._tokenizer_loader = model, tokenizer
    agent._generation_config_factory = generation_config
    with pytest.raises(ValueError, match="Configuration failed"):
        await agent.initialize()
    assert agent.get_initialization_status()["phase"] == "generation_config"
    await agent.initialize()
    assert counts == {"tokenizer": 1, "model": 1, "config": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("component", ["tokenizer", "model", "generation_config"])
async def test_missing_components_fail_before_adapter_attachment(
    monkeypatch, component
):
    calls = []
    agent = configured_agent(monkeypatch, lambda *args: calls.append(args))
    attribute = (
        "_generation_config_factory"
        if component == "generation_config"
        else f"_{component}_loader"
    )
    setattr(agent, attribute, lambda *args: None)
    with pytest.raises(RuntimeError, match="None"):
        await agent.initialize()
    assert calls == []
    assert agent.get_initialization_status()["phase"] == component
    assert agent.get_initialization_status()["state"] == "failed"


@pytest.mark.asyncio
async def test_adapter_validation_can_retry_without_reloading_the_base(monkeypatch):
    attached = []
    agent = configured_agent(
        monkeypatch, lambda model, config: attached.append(model) or model
    )

    def invalid(**kwargs):
        raise ValueError("Invalid adapter configuration")

    monkeypatch.setattr(agents, "LoraConfig", invalid)
    with pytest.raises(ValueError, match="Invalid adapter"):
        await agent.initialize()
    model = agent.model
    assert not agent.get_initialization_status()["retry_requires_new_model"]
    monkeypatch.setattr(agents, "LoraConfig", lambda **kwargs: kwargs)
    await agent.initialize()
    assert attached == [model]


@pytest.mark.asyncio
async def test_missing_adapter_path_can_be_repaired_and_is_loaded_only_once(
    monkeypatch, tmp_path
):
    calls = []
    agent = configured_agent(
        monkeypatch, lambda *args: pytest.fail("Wrong adapter path")
    )
    adapter = tmp_path / "adapter"
    agent.config.peft_path = str(adapter)
    monkeypatch.setattr(
        agents,
        "PeftModel",
        SimpleNamespace(
            from_pretrained=lambda model, path: calls.append((model, path))
            or SimpleNamespace(base=model)
        ),
    )
    with pytest.raises(FileNotFoundError):
        await agent.initialize()
    model = agent.model
    adapter.mkdir()
    await agent.initialize()
    await agent.initialize()
    assert calls == [(model, str(adapter))]


@pytest.mark.asyncio
async def test_adapter_request_snapshot_is_not_changed_by_the_sdk(monkeypatch):
    calls = []

    def attach(model, config):
        config["target_modules"].append("sdk mutation")
        calls.append(model)
        return model

    agent = configured_agent(monkeypatch, attach)
    agent.config.peft_config["target_modules"] = ["q_proj"]
    await agent.initialize()
    await agent.initialize()
    assert len(calls) == 1
    assert agent.config.peft_config["target_modules"] == ["q_proj"]


@pytest.mark.asyncio
async def test_changed_adapter_request_requires_a_fresh_model(monkeypatch):
    calls = []
    agent = configured_agent(
        monkeypatch, lambda model, config: calls.append(model) or model
    )
    await agent.initialize()
    original = agent.model
    agent.config.peft_config["r"] = 8
    with pytest.raises(ValueError, match="configuration changed"):
        await agent.initialize()
    assert calls == [original]
    agent.model = SimpleNamespace(attached=False)
    await agent.initialize()
    assert len(calls) == 2 and calls[1] is agent.model
    assert agent.get_initialization_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_adapter_returning_none_is_an_uncertain_effect(monkeypatch):
    calls = []
    agent = configured_agent(monkeypatch, lambda model, config: calls.append(model))
    with pytest.raises(RuntimeError, match="no model"):
        await agent.initialize()
    with pytest.raises(RuntimeError, match="fresh model"):
        await agent.initialize()
    assert len(calls) == 1
    assert agent.get_initialization_status()["retry_requires_new_model"]


@pytest.mark.asyncio
async def test_fresh_model_can_recover_from_an_uncertain_attachment(monkeypatch):
    calls = []

    def attach(model, config):
        calls.append(model)
        if len(calls) == 1:
            raise RuntimeError("Attachment failed")
        return model

    agent = configured_agent(monkeypatch, attach)
    with pytest.raises(RuntimeError):
        await agent.initialize()
    agent.model = SimpleNamespace(attached=False)
    await agent.initialize()
    await agent.initialize()
    assert len(calls) == 2 and calls[0] is not calls[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["details", "batch", "stream"])
async def test_all_generation_routes_reject_failed_initialization_before_backend_calls(
    monkeypatch, method
):
    agent = configured_agent(monkeypatch, lambda *args: None)
    with pytest.raises(RuntimeError):
        await agent.initialize()

    async def must_not_generate(*args, **kwargs):
        pytest.fail("Failed initialization must stop rollout calls")

    agent.set_rollout_backend(SimpleNamespace(generate_with_logprobs=must_not_generate))
    with pytest.raises(RuntimeError, match="initialization"):
        if method == "details":
            await agent._generate_with_model_details("Hello")
        elif method == "batch":
            await agent._generate_with_model_details_batch("Hello", 2)
        else:
            async for _ in agent.generate_response_stream("Hello"):
                pytest.fail("Failed initialization cannot publish output")


@pytest.mark.asyncio
async def test_manual_backend_assignment_remains_supported():
    prepared = agents.Agent()
    await prepared.initialize()
    agent = agents.MultiTurnAgent(agents.AgentConfig(model_name="stub://manual"))
    agent.model, agent.tokenizer, agent.generation_config = (
        prepared.model,
        prepared.tokenizer,
        prepared.generation_config,
    )
    assert agent.get_initialization_status()["state"] == "new"
    assert isinstance(await agent.generate_response("Hello"), str)


@pytest.mark.asyncio
async def test_initialized_stub_agents_remain_copyable_and_pickleable():
    original = agents.MultiTurnAgent(
        agents.AgentConfig(
            model_name="stub://copy",
            stub_responses=["first response", "second response"],
        )
    )
    await original.initialize()
    await original.generate_response("Hello")
    for clone in (copy.deepcopy(original), pickle.loads(pickle.dumps(original))):
        model = clone.model
        await clone.initialize()
        assert clone.model is model and clone.model is not original.model
        assert clone.get_initialization_status()["backend_matches"]
        assert "second response" in await clone.generate_response("Hello again")


@pytest.mark.asyncio
async def test_parallel_thread_initializers_share_one_adapter_and_fence_generation(
    monkeypatch,
):
    entered, release, second_started = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    calls, errors = [], []

    def attach(model, config):
        calls.append(model)
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(base=model)

    agent = configured_agent(monkeypatch, attach)

    def run(second=False):
        if second:
            second_started.set()
        try:
            asyncio.run(agent.initialize())
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=run)
    second = threading.Thread(target=run, args=(True,))
    first.start()
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        second.start()
        assert await asyncio.to_thread(second_started.wait, 5)
        assert agent.get_initialization_status()["state"] == "initializing"
        with pytest.raises(RuntimeError, match="initialization"):
            await agent.generate_response("Hello")
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert errors == [] and len(calls) == 1


@pytest.mark.asyncio
async def test_async_loader_result_is_closed_and_cannot_become_a_ready_component(
    monkeypatch,
):
    agent = configured_agent(monkeypatch, lambda model, config: model)
    pending = []

    async def build_model():
        return object()

    def invalid(config):
        coroutine = build_model()
        pending.append(coroutine)
        return coroutine

    agent._model_loader = invalid
    with pytest.raises(TypeError, match="synchronous"):
        await agent.initialize()
    assert agent.model is None
    assert inspect.getcoroutinestate(pending[0]) == inspect.CORO_CLOSED
    assert agent.get_initialization_status()["state"] == "failed"


@pytest.mark.asyncio
async def test_reentrant_initialization_cannot_replace_the_owners_state(monkeypatch):
    agent = configured_agent(monkeypatch, lambda model, config: model)

    def load(config):
        recursive = agent.initialize()
        try:
            with pytest.raises(RuntimeError, match="reenter"):
                recursive.send(None)
        finally:
            recursive.close()
        assert agent.get_initialization_status()["state"] == "initializing"
        return object()

    agent._model_loader = load
    await agent.initialize()
    assert agent.get_initialization_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_adapter_cancellation_is_retained_as_an_uncertain_effect(monkeypatch):
    calls = []

    def attach(model, config):
        calls.append(model)
        raise asyncio.CancelledError

    agent = configured_agent(monkeypatch, attach)
    with pytest.raises(asyncio.CancelledError):
        await agent.initialize()
    assert agent.get_initialization_status()["error_type"] == "CancelledError"
    with pytest.raises(RuntimeError, match="fresh model"):
        await agent.initialize()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_real_lora_attachment_preserves_the_trained_parameter_objects_on_repeat():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    tiny = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
    )
    agent = agents.MultiTurnAgent(
        agents.AgentConfig(
            model_name="local-tiny-gpt2",
            use_peft=True,
            peft_config={
                "r": 2,
                "target_modules": ["c_attn"],
                "task_type": "CAUSAL_LM",
                "fan_in_fan_out": True,
            },
        ),
        model_loader=lambda config: tiny,
        tokenizer_loader=lambda config: SimpleNamespace(
            pad_token="pad", pad_token_id=0, eos_token_id=1
        ),
    )
    await agent.initialize()
    wrapped = agent.model
    parameters = dict(wrapped.named_parameters())
    assert isinstance(wrapped, peft.PeftModel)
    assert any("lora_A" in name for name in parameters)
    await agent.initialize()
    assert agent.model is wrapped
    assert all(parameters[name] is value for name, value in wrapped.named_parameters())
