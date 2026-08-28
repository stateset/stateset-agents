"""Regression tests for immutable model revision propagation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from stateset_agents.core.agent import Agent
from stateset_agents.core.agent_config import AgentConfig, ConfigValidationError
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.gspo_config import GSPOConfig
from stateset_agents.training.gspo_trainer import GSPOModelManager

REVISION = "a" * 40


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
