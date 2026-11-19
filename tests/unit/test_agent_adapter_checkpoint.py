"""Named LoRA snapshots preserve weights and execution state across restoration."""

import asyncio
import json
import shutil

import pytest

from stateset_agents.core.agent import load_agent_from_checkpoint, save_agent_checkpoint

from .test_agent_checkpoint_roundtrip import real_agent
from .test_agent_checkpoint_roundtrip import tiny_base as tiny_base


async def adapted_agent(base, names, **adapter_options):
    import torch

    peft = pytest.importorskip("peft")
    agent = await real_agent(base)

    def config():
        return peft.LoraConfig(
            r=2,
            target_modules=["c_attn"],
            task_type="CAUSAL_LM",
            fan_in_fan_out=True,
            **adapter_options,
        )

    agent.model = peft.get_peft_model(agent.model, config(), adapter_name=names[0])
    for name in names[1:]:
        agent.model.add_adapter(name, config())
    generator = torch.Generator().manual_seed(31)
    with torch.no_grad():
        for name, parameter in agent.model.named_parameters():
            if "lora_" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    agent.model.set_adapter(names[-1])
    agent.model.eval()
    return agent


def logits(model):
    import torch

    with torch.no_grad():
        return model(torch.tensor([[3, 2, 3]])).logits.detach().clone()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "names", [("support",), ("default", "support"), ("sales", "support")]
)
async def test_named_adapters_roundtrip_all_weights_and_active_selection(
    tiny_base, tmp_path, names
):
    import torch

    agent = await adapted_agent(tiny_base, names)
    from peft import get_peft_model_state_dict

    expected = {
        name: {
            key: value.detach().clone()
            for key, value in get_peft_model_state_dict(
                agent.model, adapter_name=name
            ).items()
        }
        for name in names
    }
    expected_logits = logits(agent.model)
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    restored = await load_agent_from_checkpoint(str(path))
    assert list(restored.model.peft_config) == list(names)
    assert restored.model.active_adapter == names[-1]
    assert restored.model.active_adapters == [names[-1]]
    for name in names:
        actual = get_peft_model_state_dict(restored.model, adapter_name=name)
        assert set(actual) == set(expected[name])
        assert all(
            torch.equal(value, actual[key]) for key, value in expected[name].items()
        )
    torch.testing.assert_close(logits(restored.model), expected_logits)
    assert not any(p.requires_grad for p in restored.model.parameters())
    model = restored.model
    restored.model.set_adapter(names[0])
    await restored.initialize()
    assert restored.model is model
    assert restored.model.active_adapter == names[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["multiple_active", "none_active", "disabled", "merged", "multiple_merged"]
)
async def test_adapter_execution_state_roundtrips(tiny_base, tmp_path, mode):
    import torch

    agent = await adapted_agent(tiny_base, ("default", "support"))
    if mode == "multiple_active":
        agent.model.base_model.set_adapter(["support", "default"])
    elif mode == "none_active":
        agent.model.base_model.set_adapter([])
    elif mode == "disabled":
        agent.model.base_model.disable_adapter_layers()
    elif mode == "merged":
        agent.model.base_model.merge_adapter(adapter_names=["support"])
    else:
        agent.model.base_model.set_adapter(["support", "default"])
        agent.model.base_model.merge_adapter(adapter_names=["support", "default"])
    expected_logits = logits(agent.model)
    expected = agent.model.get_model_status()
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    restored = await load_agent_from_checkpoint(str(path))
    actual = restored.model.get_model_status()
    for field in (
        "active_adapters",
        "available_adapters",
        "enabled",
        "merged_adapters",
    ):
        assert getattr(actual, field) == getattr(expected, field)
    torch.testing.assert_close(logits(restored.model), expected_logits)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_partial_multi_adapter_load_is_not_replayed(
    tiny_base, tmp_path, monkeypatch, error_type
):
    from transformers import AutoModelForCausalLM

    agent = await adapted_agent(tiny_base, ("default", "support"))
    from peft import PeftModel

    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    pending = await load_agent_from_checkpoint(str(path), load_model=False)
    original = PeftModel.load_adapter
    calls = []

    def fail_after_effect(self, model_id, adapter_name, *args, **kwargs):
        calls.append(adapter_name)
        result = original(self, model_id, adapter_name, *args, **kwargs)
        if adapter_name == "support":
            raise error_type("injected after attachment")
        return result

    monkeypatch.setattr(PeftModel, "load_adapter", fail_after_effect)
    with pytest.raises(error_type):
        await pending.initialize()
    count = len(calls)
    assert pending.get_initialization_status()["retry_requires_new_model"]
    with pytest.raises(RuntimeError, match="initialization is failed"):
        await pending.generate_response("hello")
    with pytest.raises(RuntimeError, match="fresh model"):
        await pending.initialize()
    assert len(calls) == count
    monkeypatch.setattr(PeftModel, "load_adapter", original)
    pending.model = AutoModelForCausalLM.from_pretrained(
        tiny_base, local_files_only=True
    )
    await pending.initialize()
    assert pending.get_initialization_status()["state"] == "ready"
    assert set(pending.model.peft_config) == {"default", "support"}


def republish(path, edit):
    """Exercise semantic validation independently of publication hash checks."""
    from stateset_agents.training.checkpoint_publication import (
        COMPLETION_FILE,
        checkpoint_write,
    )

    with checkpoint_write(path, kind="agent") as staging:
        shutil.copytree(
            path,
            staging,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(COMPLETION_FILE),
        )
        edit(staging)


@pytest.mark.asyncio
@pytest.mark.parametrize("published", [True, False])
async def test_older_default_adapter_checkpoints_remain_loadable(
    tiny_base, tmp_path, published
):
    import torch

    from stateset_agents.training.checkpoint_publication import COMPLETION_FILE

    agent = await adapted_agent(tiny_base, ("default",))
    expected = logits(agent.model)
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))

    def downgrade(staging):
        descriptor = staging / "agent_checkpoint.json"
        state = json.loads(descriptor.read_text())
        state["schema_version"] = 1
        del state["adapters"]
        descriptor.write_text(json.dumps(state))

    republish(path, downgrade)
    if not published:
        (path / COMPLETION_FILE).unlink()
        (path / "agent_checkpoint.json").unlink()
    restored = await load_agent_from_checkpoint(str(path))
    torch.testing.assert_close(logits(restored.model), expected)


@pytest.mark.asyncio
async def test_disabled_wrapper_and_loaded_model_survive_checkpoint_removal(
    tiny_base, tmp_path
):
    import torch

    agent = await adapted_agent(tiny_base, ("support",))
    path = tmp_path / "saved"
    with agent.model.disable_adapter():
        expected = logits(agent.model)
        await save_agent_checkpoint(agent, str(path))
    restored = await load_agent_from_checkpoint(str(path))
    assert restored.model._adapters_disabled
    assert restored.model.get_model_status().enabled is False
    torch.testing.assert_close(logits(restored.model), expected)
    model = restored.model
    shutil.rmtree(path)
    await restored.initialize()
    assert restored.model is model
    torch.testing.assert_close(logits(restored.model), expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "unsafe_name",
        "unknown_active",
        "irregular",
        "missing_weights",
        "extra_adapter",
        "different_base",
    ],
)
async def test_invalid_adapter_collection_is_rejected_before_loading(
    tiny_base, tmp_path, mutation, monkeypatch
):
    agent = await adapted_agent(tiny_base, ("default", "support"))
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))

    def mutate(staging):
        descriptor = staging / "agent_checkpoint.json"
        state = json.loads(descriptor.read_text())
        adapters = state["adapters"]
        if mutation == "unsafe_name":
            adapters["names"] = ["default", "../support"]
        elif mutation == "unknown_active":
            adapters["active"] = ["absent"]
        elif mutation == "irregular":
            adapters["enabled"] = "irregular"
        elif mutation == "missing_weights":
            (staging / "support" / "adapter_model.safetensors").unlink()
        elif mutation == "extra_adapter":
            shutil.copytree(staging / "support", staging / "undeclared")
        else:
            config_path = staging / "support" / "adapter_config.json"
            config = json.loads(config_path.read_text())
            config["base_model_name_or_path"] = "different/base"
            config_path.write_text(json.dumps(config))
        descriptor.write_text(json.dumps(state))

    republish(path, mutate)
    monkeypatch.setattr(
        "stateset_agents.core.agent._load_peft",
        lambda: pytest.fail("loaded invalid adapters"),
    )
    with pytest.raises(ValueError):
        await load_agent_from_checkpoint(str(path), load_model=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", "folder/name", "SUPPORT"])
async def test_unsafe_names_fail_before_serialization(
    tiny_base, tmp_path, monkeypatch, unsafe
):
    agent = await adapted_agent(tiny_base, ("default", "support"))
    agent.model.peft_config[unsafe] = agent.model.peft_config["support"]
    monkeypatch.setattr(
        agent.model,
        "save_pretrained",
        lambda *args, **kwargs: pytest.fail("unsafe model save"),
    )
    with pytest.raises(ValueError, match="Adapter names"):
        await save_agent_checkpoint(agent, str(tmp_path / "saved"))
    assert not (tmp_path / "saved").exists()


@pytest.mark.asyncio
async def test_partial_collection_save_retains_previous_checkpoint(
    tiny_base, tmp_path, monkeypatch
):
    import torch

    agent = await adapted_agent(tiny_base, ("default", "support"))
    path = tmp_path / "saved"
    expected = logits(agent.model)
    await save_agent_checkpoint(agent, str(path))
    before = {
        item.relative_to(path): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }
    original = agent.model.save_pretrained

    def fail_after_first(staging):
        original(staging, selected_adapters=["default"])
        raise OSError("injected second adapter write failure")

    monkeypatch.setattr(agent.model, "save_pretrained", fail_after_first)
    with pytest.raises(OSError, match="second adapter"):
        await save_agent_checkpoint(agent, str(path))
    assert {
        item.relative_to(path): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    } == before
    restored = await load_agent_from_checkpoint(str(path))
    torch.testing.assert_close(logits(restored.model), expected)


@pytest.mark.asyncio
async def test_checkpoint_base_revision_overrides_stale_agent_config(
    tiny_base, tmp_path
):
    agent = await adapted_agent(tiny_base, ("support",))
    agent.config.model_revision = "stale"
    agent.config.model_kwargs = {"revision": "also-stale"}
    agent.model.peft_config["support"].revision = "adapter-revision"
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    restored = await load_agent_from_checkpoint(str(path), load_model=False)
    assert restored.config.model_revision == "adapter-revision"
    assert "revision" not in restored.config.model_kwargs
    await restored.initialize()


@pytest.mark.asyncio
async def test_dora_merge_order_survives_roundtrip(tiny_base, tmp_path):
    import torch

    agent = await adapted_agent(tiny_base, ("default", "support"), use_dora=True)
    agent.model.base_model.set_adapter(["support", "default"])
    agent.model.base_model.merge_adapter(adapter_names=["support", "default"])
    expected = logits(agent.model)
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    restored = await load_agent_from_checkpoint(str(path))
    assert [layer.merged_adapters for layer in restored.model.get_layer_status()] == [
        layer.merged_adapters for layer in agent.model.get_layer_status()
    ]
    torch.testing.assert_close(logits(restored.model), expected)


@pytest.mark.asyncio
async def test_failed_merge_cannot_replay_on_mutated_model(
    tiny_base, tmp_path, monkeypatch
):
    agent = await adapted_agent(tiny_base, ("support",))
    agent.model.base_model.merge_adapter(adapter_names=["support"])
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    pending = await load_agent_from_checkpoint(str(path), load_model=False)
    tuner = type(agent.model.base_model)
    original = tuner.merge_adapter
    calls = []

    def fail_after_merge(self, *args, **kwargs):
        calls.append(self)
        original(self, *args, **kwargs)
        raise RuntimeError("injected after merge")

    monkeypatch.setattr(tuner, "merge_adapter", fail_after_merge)
    with pytest.raises(RuntimeError, match="after merge"):
        await pending.initialize()
    assert pending.get_initialization_status()["retry_requires_new_model"]
    with pytest.raises(RuntimeError, match="fresh model"):
        await pending.initialize()
    assert len(calls) == 1
