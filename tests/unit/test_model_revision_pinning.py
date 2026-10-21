"""Regression tests for immutable model revision propagation."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stateset_agents.core.agent import Agent
from stateset_agents.core.agent_config import AgentConfig, ConfigValidationError
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.gspo_config import GSPOConfig
from stateset_agents.training.gspo_trainer import GSPOModelManager

REVISION = "a" * 40


@pytest.mark.parametrize("mode", ["single_turn", "multi_turn", "offline", "hybrid"])
@pytest.mark.parametrize("source", ["inherited", "keyword", "override"])
async def test_train_keeps_agent_identity_in_every_mode(monkeypatch, mode, source):
    module = importlib.import_module("stateset_agents.training.train")
    agent = Agent(AgentConfig(model_name="example/model", model_revision=REVISION))
    observed = []

    class Trainer:
        def __init__(self, *, config, **kwargs):
            observed.append(config)

        def add_callback(self, callback):
            pass

        async def initialize(self):
            pass

        async def train(self):
            return agent

    async def offline(agent, config, *args, **kwargs):
        observed.append(config)
        return agent

    monkeypatch.setattr(module, "MultiTurnGRPOTrainer", Trainer)
    monkeypatch.setattr(
        "stateset_agents.training.trainer.SingleTurnGRPOTrainer", Trainer
    )
    monkeypatch.setattr(module, "_train_offline", offline)
    settings = {"model_name": "example/model", "model_revision": REVISION}
    kwargs = (
        {"config_overrides": settings}
        if source == "override"
        else (settings if source == "keyword" else {})
    )
    assert (
        await module.train(
            agent,
            SimpleNamespace(reward_fn=None),
            training_mode=mode,
            dataset=[{}],
            num_episodes=1,
            **kwargs,
        )
        is agent
    )
    assert observed and all(config.model_name == "example/model" for config in observed)
    assert all(config.model_revision == REVISION for config in observed)


@pytest.mark.parametrize("source", ["keyword", "override"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("model_name", "other/model"),
        ("model_revision", "b" * 40),
        ("model_revision", None),
    ],
)
async def test_train_rejects_mislabelled_agent_before_training(source, field, value):
    module = importlib.import_module("stateset_agents.training.train")
    agent = Agent(AgentConfig(model_name="example/model", model_revision=REVISION))
    settings = {field: value}
    kwargs = {"config_overrides": settings} if source == "override" else settings
    with pytest.raises(ValueError, match=field):
        await module.train(agent, object(), **kwargs)
    assert agent.model is None


async def test_train_cannot_retroactively_pin_an_unpinned_agent():
    module = importlib.import_module("stateset_agents.training.train")
    agent = Agent(AgentConfig(model_name="example/model"))
    with pytest.raises(ValueError, match="model_revision"):
        await module.train(agent, object(), model_revision=REVISION)


@pytest.mark.parametrize("revision", ["", "  ", True, 42])
def test_native_training_config_rejects_invalid_model_revision(revision):
    with pytest.raises(ValueError, match="model_revision"):
        TrainingConfig(model_revision=revision).validate_for_training()


async def test_offline_config_retains_agent_model_revision(monkeypatch):
    module = importlib.import_module("stateset_agents.training.train")
    trainer = MagicMock()
    trainer.pretrain_value_functions = AsyncMock(return_value={})
    trainer.train = AsyncMock()
    factory = MagicMock(return_value=trainer)
    monkeypatch.setattr(
        "stateset_agents.experimental.offline_rl.offline_grpo.OfflineGRPOTrainer",
        factory,
    )
    agent = SimpleNamespace(model=object(), tokenizer=None)
    config = TrainingConfig(model_name="example/model", model_revision=REVISION)
    await module._train_offline(agent, config, None, [{}], None, None)
    assert factory.call_args.kwargs["config"].model_revision == REVISION


def test_training_configs_retain_model_revision() -> None:
    assert (
        TrainingConfig(model_revision=REVISION).to_dict()["model_revision"] == REVISION
    )
    assert GSPOConfig(model_revision=REVISION).model_revision == REVISION


def test_agent_config_rejects_empty_revision() -> None:
    try:
        AgentConfig(model_name="gpt2", model_revision="")
    except ConfigValidationError as exc:
        assert exc.field == "model_revision"
    else:  # pragma: no cover - makes the intended failure explicit
        raise AssertionError("empty model revision should be rejected")


def test_agent_passes_revision_to_tokenizer_and_model() -> None:
    config = AgentConfig(
        model_name="example/model",
        model_revision=REVISION,
        attn_implementation=None,
        device_map=None,
    )
    agent = Agent(config)
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    model = MagicMock()
    with (
        patch("stateset_agents.core.agent._load_transformers_agent", return_value=True),
        patch("stateset_agents.core.agent.AutoTokenizer") as tokenizer_cls,
        patch("stateset_agents.core.agent.AutoModelForCausalLM") as model_cls,
        patch(
            "stateset_agents.core.agent.load_generation_model",
            return_value=(model, model_cls),
        ) as loader,
    ):
        tokenizer_cls.from_pretrained.return_value = tokenizer
        import asyncio

        asyncio.run(agent.initialize())

    assert tokenizer_cls.from_pretrained.call_args.kwargs["revision"] == REVISION
    assert loader.call_args.args[2]["revision"] == REVISION


def test_shared_flagship_loader_pins_tokenizer_and_model() -> None:
    config = GSPOConfig(
        model_name="example/model",
        model_revision=REVISION,
        use_lora=False,
        gradient_checkpointing=False,
        use_reference_model=False,
    )
    manager = GSPOModelManager(config)
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    model = MagicMock()
    with (
        patch(
            "stateset_agents.training.gspo_trainer._load_transformers",
            return_value=True,
        ),
        patch("stateset_agents.training.gspo_trainer.AutoTokenizer") as tokenizer_cls,
        patch(
            "stateset_agents.training.gspo_trainer.AutoModelForCausalLM"
        ) as model_cls,
        patch(
            "stateset_agents.training.trainer_runtime.load_generation_model",
            return_value=(model, model_cls),
        ) as loader,
    ):
        tokenizer_cls.from_pretrained.return_value = tokenizer
        manager.load_model_and_tokenizer()

    assert tokenizer_cls.from_pretrained.call_args.kwargs["revision"] == REVISION
    assert loader.call_args.args[2]["revision"] == REVISION


def test_gspo_vllm_rollout_uses_same_revision() -> None:
    from stateset_agents.training import gspo_generation

    config = GSPOConfig(
        model_name="example/model", model_revision=REVISION, use_vllm=False
    )
    generator = object.__new__(gspo_generation.GSPOTrajectoryGenerator)
    generator.config = config
    generator.vllm_generator = None
    config_cls = MagicMock()
    generator_cls = MagicMock()
    with (
        patch.object(gspo_generation, "_load_vllm_backend", return_value=True),
        patch.object(gspo_generation, "VLLMConfig", config_cls),
        patch.object(gspo_generation, "VLLMGenerator", generator_cls),
    ):
        generator._setup_vllm_generator()

    assert config_cls.call_args.kwargs["revision"] == REVISION
