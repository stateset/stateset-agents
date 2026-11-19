"""Agent snapshots restore saved artifacts rather than reopening their source."""

import json
import shutil
import subprocess
import sys

import pytest

from stateset_agents.core.agent import (
    AgentConfig,
    MultiTurnAgent,
    load_agent_from_checkpoint,
    save_agent_checkpoint,
)


async def stub_agent():
    agent = MultiTurnAgent(
        AgentConfig(model_name="stub://saved", stub_responses=["first", "second"])
    )
    await agent.initialize()
    return agent


@pytest.mark.asyncio
async def test_stub_restores_actual_responses_cursor_and_generation_settings(tmp_path):
    agent = await stub_agent()
    agent.model._responses = ["trained first", "trained second"]
    assert await agent.generate_response("hello") == "trained first"
    agent.generation_config.temperature = 0.25
    agent.tokenizer.model_max_length = 1234
    await save_agent_checkpoint(agent, str(tmp_path / "saved"))
    restored = await load_agent_from_checkpoint(str(tmp_path / "saved"))
    assert await restored.generate_response("hello") == "trained second"
    assert restored.generation_config.temperature == 0.25
    assert restored.tokenizer.model_max_length == 1234
    await restored.initialize()
    assert await restored.generate_response("hello") == "trained first"


@pytest.fixture
def tiny_base(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    base = tmp_path / "base"
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2, "hello": 3})
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    wrapped = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    wrapped.save_pretrained(base)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=4, n_embd=8, n_layer=1, n_head=1, n_positions=32
        )
    )
    model.save_pretrained(base)
    return base


async def real_agent(base, **kwargs):
    agent = MultiTurnAgent(
        AgentConfig(
            model_name=str(base),
            torch_dtype="float32",
            device_map=None,
            attn_implementation="eager",
            **kwargs,
        )
    )
    await agent.initialize()
    return agent


@pytest.mark.asyncio
async def test_real_full_checkpoint_loads_saved_weights_without_original_source(
    tiny_base, tmp_path
):
    import torch

    agent = await real_agent(tiny_base)
    with torch.no_grad():
        agent.model.transformer.wte.weight.fill_(0.314)
    expected = {
        name: value.detach().clone() for name, value in agent.model.state_dict().items()
    }
    agent.generation_config.max_new_tokens = 17
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    shutil.rmtree(tiny_base)
    restored = await load_agent_from_checkpoint(str(path))
    assert all(
        torch.equal(value, restored.model.state_dict()[name])
        for name, value in expected.items()
    )
    assert restored.generation_config.max_new_tokens == 17
    assert restored.config.model_name == str(path)
    assert restored.get_initialization_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_failed_replacement_keeps_previous_snapshot(tmp_path, monkeypatch):
    agent = await stub_agent()
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    dump = json.dump

    def fail(value, stream, *args, **kwargs):
        stream.write("partial")
        raise OSError("disk full")

    monkeypatch.setattr(json, "dump", fail)
    with pytest.raises(OSError, match="disk full"):
        await save_agent_checkpoint(agent, str(path))
    monkeypatch.setattr(json, "dump", dump)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before


@pytest.mark.asyncio
async def test_real_adapter_restores_trained_lora_and_saved_tokenizer(
    tiny_base, tmp_path
):
    import torch

    peft = pytest.importorskip("peft")
    agent = await real_agent(
        tiny_base,
        use_peft=True,
        peft_config={
            "r": 2,
            "target_modules": ["c_attn"],
            "task_type": "CAUSAL_LM",
            "fan_in_fan_out": True,
        },
    )
    with torch.no_grad():
        for name, value in agent.model.named_parameters():
            if "lora_" in name:
                value.fill_(0.7)
    expected = {
        name: value.detach().clone()
        for name, value in peft.get_peft_model_state_dict(agent.model).items()
    }
    agent.tokenizer.chat_template = "saved tokenizer template"
    path = tmp_path / "adapter"
    await save_agent_checkpoint(agent, str(path))
    restored = await load_agent_from_checkpoint(str(path))
    assert isinstance(restored.model, peft.PeftModel)
    actual = peft.get_peft_model_state_dict(restored.model)
    assert set(actual) == set(expected)
    assert all(torch.equal(value, actual[name]) for name, value in expected.items())
    assert restored.tokenizer.chat_template == "saved tokenizer template"
    assert restored.config.peft_path == str(path)
    assert restored.config.peft_config is None
    loaded = restored.model
    await restored.initialize()
    assert restored.model is loaded
    assert set(restored.model.peft_config) == {"default"}


@pytest.mark.asyncio
async def test_deferred_load_is_lightweight_then_restores_local_weights(
    tiny_base, tmp_path, monkeypatch
):
    import stateset_agents.core.agent as agents

    agent = await real_agent(tiny_base)
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    shutil.rmtree(tiny_base)
    with monkeypatch.context() as patch:
        patch.setattr(
            agents,
            "_load_transformers_agent",
            lambda: pytest.fail("eager model loading"),
        )
        restored = await load_agent_from_checkpoint(str(path), load_model=False)
    assert restored.model is None and restored.tokenizer is None
    assert restored.get_initialization_status()["state"] == "new"
    await restored.initialize()
    assert restored.get_initialization_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_deferred_load_rejects_another_published_generation(tmp_path):
    agent = await stub_agent()
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    pending = await load_agent_from_checkpoint(str(path), load_model=False)
    agent.model._index = 1
    await save_agent_checkpoint(agent, str(path))
    with pytest.raises(ValueError, match="changed since load"):
        await pending.initialize()
    assert pending.model is None
    assert pending.get_initialization_status()["state"] == "failed"
    restored = await load_agent_from_checkpoint(str(path))
    assert await restored.generate_response("hello") == "second"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["changed", "extra", "missing", "symlink", "marker"]
)
async def test_corruption_rejected_even_for_config_only_load(tmp_path, mutation):
    agent = await stub_agent()
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    descriptor = path / "agent_checkpoint.json"
    if mutation == "changed":
        descriptor.write_text("{}")
    elif mutation == "extra":
        (path / "stale.bin").write_bytes(b"stale")
    elif mutation == "missing":
        descriptor.unlink()
    elif mutation == "symlink":
        descriptor.unlink()
        descriptor.symlink_to(path / "agent_config.json")
    else:
        (path / ".stateset-checkpoint.json").unlink()
    with pytest.raises(ValueError):
        await load_agent_from_checkpoint(str(path), load_model=False)


@pytest.mark.asyncio
async def test_configuration_only_is_explicit_and_cannot_erase_weights(tmp_path):
    agent = await stub_agent()
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(ValueError, match="configuration-only"):
        await save_agent_checkpoint(agent, str(path), save_model=False)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before
    config_path = tmp_path / "configuration"
    config_path.mkdir()
    await save_agent_checkpoint(agent, str(config_path), save_model=False)
    descriptor = json.loads((config_path / "agent_checkpoint.json").read_text())
    assert descriptor["kind"] == "configuration"
    loaded = await load_agent_from_checkpoint(str(config_path), load_model=False)
    assert loaded.config == agent.config and loaded.model is None
    await loaded.initialize()
    assert await loaded.generate_response("hello") == "first"


@pytest.mark.asyncio
async def test_uninitialized_model_save_cannot_claim_a_complete_snapshot(tmp_path):
    agent = MultiTurnAgent(AgentConfig(model_name="stub://uninitialized"))
    with pytest.raises(ValueError, match="Initialize"):
        await save_agent_checkpoint(agent, str(tmp_path / "saved"))
    assert not (tmp_path / "saved").exists()


@pytest.mark.asyncio
async def test_legacy_model_checkpoint_uses_local_artifacts(tiny_base, tmp_path):
    from dataclasses import asdict

    agent = await real_agent(tiny_base)
    path = tmp_path / "legacy"
    agent.model.save_pretrained(path)
    agent.tokenizer.save_pretrained(path)
    (path / "agent_config.json").write_text(json.dumps(asdict(agent.config)))
    shutil.rmtree(tiny_base)
    loaded = await load_agent_from_checkpoint(str(path))
    assert loaded.config.model_name == str(path)
    assert loaded.get_initialization_status()["state"] == "ready"


@pytest.mark.asyncio
async def test_legacy_missing_weights_cannot_silently_reopen_base(tiny_base, tmp_path):
    from dataclasses import asdict

    agent = await real_agent(tiny_base)
    path = tmp_path / "partial"
    path.mkdir()
    agent.model.config.save_pretrained(path)
    (path / "agent_config.json").write_text(json.dumps(asdict(agent.config)))
    with pytest.raises(ValueError, match="weights are missing"):
        await load_agent_from_checkpoint(str(path))


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["old_moved", "new_published"])
async def test_process_exit_recovers_complete_agent_checkpoint(tmp_path, phase):
    agent = await stub_agent()
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    script = """
import asyncio
import os
import sys
from pathlib import Path
from stateset_agents.core.agent import load_agent_from_checkpoint, save_agent_checkpoint
from stateset_agents.training import checkpoint_publication as publication
path, phase = Path(sys.argv[1]), sys.argv[2]
replace = os.replace
def crash(source, target):
    replace(source, target)
    if ((phase == "old_moved" and Path(source) == path)
        or (phase == "new_published" and Path(target) == path)):
        os._exit(72)
async def main():
    agent = await load_agent_from_checkpoint(str(path))
    agent.model._responses = ["replacement"]
    publication.os.replace = crash
    await save_agent_checkpoint(agent, str(path))
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), phase],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 72, result.stderr.decode()
    restored = await load_agent_from_checkpoint(str(path))
    assert await restored.generate_response("hello") == (
        "first" if phase == "old_moved" else "replacement"
    )
    assert not (tmp_path / ".saved.stateset-publication.json").exists()


@pytest.mark.asyncio
async def test_agent_and_training_publications_keep_distinct_contracts(tmp_path):
    from stateset_agents.training.checkpoint_publication import (
        checkpoint_write,
        validate_checkpoint_publication,
    )

    agent = await stub_agent()
    path = tmp_path / "agent"
    await save_agent_checkpoint(agent, str(path))
    assert validate_checkpoint_publication(path, required=True, kind="agent")
    with pytest.raises(ValueError):
        validate_checkpoint_publication(path, required=True)
    with pytest.raises(ValueError, match="not a checkpoint"):
        with checkpoint_write(path):
            pytest.fail("Training writer must not erase an agent checkpoint")
    training = tmp_path / "training"
    with checkpoint_write(training) as staging:
        (staging / "training_state.pt").write_bytes(b"training")
        (staging / "agent_config.json").write_text("{}")
    with pytest.raises(ValueError, match="training checkpoint"):
        await save_agent_checkpoint(agent, str(training))
    assert validate_checkpoint_publication(training, required=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["model", "tokenizer"])
async def test_component_save_failure_retains_complete_full_model(
    tiny_base, tmp_path, monkeypatch, phase
):
    agent = await real_agent(tiny_base)
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    before = {
        p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }

    def fail(staging):
        (staging / "partial.bin").write_bytes(b"partial")
        raise OSError("component save failed")

    monkeypatch.setattr(getattr(agent, phase), "save_pretrained", fail)
    with pytest.raises(OSError, match="component save failed"):
        await save_agent_checkpoint(agent, str(path))
    assert {
        p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()
    } == before
    assert not list(tmp_path.glob(".saved.stateset-stage-*"))


@pytest.mark.asyncio
async def test_deferred_initialization_holds_publication_lock(tmp_path, monkeypatch):
    from stateset_agents.training.checkpoint_publication import checkpoint_write

    agent = await stub_agent()
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    pending = await load_agent_from_checkpoint(str(path), load_model=False)
    initialize = pending._initialize_stub_backend
    attempted = []

    def initialize_while_contended():
        with pytest.raises(ValueError, match="busy"):
            with checkpoint_write(path, kind="agent"):
                pytest.fail("Cannot publish over an active restore")
        attempted.append(True)
        initialize()

    monkeypatch.setattr(pending, "_initialize_stub_backend", initialize_while_contended)
    await pending.initialize()
    assert attempted == [True]


@pytest.mark.asyncio
async def test_non_json_config_fails_without_stringifying_runtime_objects(tmp_path):
    agent = await stub_agent()
    agent.config.model_kwargs = {"custom_object": object()}
    with pytest.raises(TypeError, match="JSON serializable"):
        await save_agent_checkpoint(agent, str(tmp_path / "saved"))
    assert not (tmp_path / "saved").exists()


@pytest.mark.asyncio
async def test_deferred_metadata_read_postpones_weight_verification(
    tiny_base, tmp_path, monkeypatch
):
    from stateset_agents.training import checkpoint_publication as publication

    agent = await real_agent(tiny_base)
    path = tmp_path / "saved"
    await save_agent_checkpoint(agent, str(path))
    weights = path / "model.safetensors"
    weights.write_bytes(b"corrupt weight bytes")
    # Metadata remains inspectable without opening any weight payload.
    with monkeypatch.context() as patch:
        patch.setattr(
            publication,
            "validate_checkpoint_publication",
            lambda *args, **kwargs: pytest.fail("eager weight verification"),
        )
        restored = await load_agent_from_checkpoint(str(path), load_model=False)
    assert restored.model is None
    with pytest.raises(ValueError, match="differ"):
        await restored.initialize()
    assert restored.model is None
    assert restored.get_initialization_status()["state"] == "failed"


@pytest.mark.asyncio
async def test_unrelated_destination_is_preserved(tmp_path):
    agent = await stub_agent()
    path = tmp_path / "notes"
    path.mkdir()
    (path / "notes.txt").write_text("keep")
    with pytest.raises(ValueError, match="not a checkpoint"):
        await save_agent_checkpoint(agent, str(path))
    assert (path / "notes.txt").read_text() == "keep"
