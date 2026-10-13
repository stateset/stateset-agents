"""Small Gemma configuration, training wiring, and local architecture checks."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from examples import finetune_gspo
from stateset_agents.training import gemma4_small_starter as starter


@pytest.mark.parametrize("size", ["e2b", "e4b"])
def test_installed_cli_roundtrip_and_training_dispatch(size, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from stateset_agents.cli import app

    runner = CliRunner()
    command = f"gemma-4-{size}"
    path = tmp_path / "config.json"
    result = runner.invoke(
        app,
        [command, "--starter-profile", "memory", "--write-config", str(path), "--json"],
    )
    assert result.exit_code == 0, result.output
    config = json.loads(result.output)["config"]
    assert config["model_name"] == f"google/gemma-4-{size.upper()}-it"
    assert config["output_dir"] == f"./outputs/gemma4_{size}_gspo"
    result = runner.invoke(app, [command, "--config", str(path), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["config"] == config
    train = AsyncMock(return_value="trained")
    monkeypatch.setattr(starter, "run_gemma4_small_config", train)
    result = runner.invoke(app, [command, "--config", str(path), "--no-dry-run"])
    assert result.exit_code == 0, result.output
    train.assert_awaited_once()
    assert train.call_args.args[0].model_name == config["model_name"]
    assert train.call_args.kwargs == {"dry_run": False}


def test_installed_cli_honors_output_override(tmp_path):
    from typer.testing import CliRunner

    from stateset_agents.cli import app

    result = CliRunner().invoke(
        app, ["gemma-4-e4b", "--output-dir", str(tmp_path), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["config"]["output_dir"] == str(tmp_path)


@pytest.mark.parametrize("size", ["e2b", "e4b"])
def test_cli_profile_roundtrip(size, tmp_path, capsys):
    path = tmp_path / "config.json"
    assert (
        finetune_gspo.main(
            [
                "--model",
                f"gemma4-{size}",
                "--starter-profile",
                "memory",
                "--write-config",
                str(path),
            ]
        )
        == 0
    )
    config = starter.load_gemma4_small_config_file(path)
    assert config.model_name == f"google/gemma-4-{size.upper()}-it"
    assert config.use_lora and config.use_4bit
    assert config.num_generations == 2
    assert config.per_device_train_batch_size == 1
    assert (
        finetune_gspo.main(
            ["--model", f"gemma4-{size}", "--config", str(path), "--dry-run"]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"] == config.to_dict()
    assert payload["gspo_overrides"]["use_4bit"] is True
    assert payload["agent_config"]["tokenizer_kwargs"]["padding_side"] == "left"


def test_overrides_and_dependency_guidance(monkeypatch):
    config = starter.get_gemma4_small_config(starter_profile="memory", use_4bit=False)
    assert not config.use_4bit
    monkeypatch.setattr(
        starter.starter_common, "get_transformers_version", lambda: (4, 57, 1)
    )
    assert any("transformers>=5.5.0" in warning for warning in config.validate())
    config.use_4bit = True
    config.use_lora = False
    with pytest.raises(ValueError, match="requires use_lora"):
        config.validate()


@pytest.mark.asyncio
@pytest.mark.parametrize("stub", [False, True])
async def test_starter_leaves_real_model_loading_to_trainer(monkeypatch, stub):
    from stateset_agents import MultiTurnAgent
    from stateset_agents.training import gspo_trainer

    initialize = AsyncMock()
    train = AsyncMock(return_value="trained")
    monkeypatch.setattr(MultiTurnAgent, "initialize", initialize)
    monkeypatch.setattr(gspo_trainer, "train_with_gspo", train)
    config = starter.get_gemma4_small_config(
        model_name="stub://gemma" if stub else starter.GEMMA4_SMALL_MODELS[0],
        starter_profile="memory",
    )
    assert await starter.run_gemma4_small_config(config) == "trained"
    assert initialize.await_count == int(stub)
    kwargs = train.call_args.kwargs
    assert kwargs["config"].use_4bit
    assert kwargs["config"].model_name == config.model_name
    assert kwargs["agent"].model is None


@pytest.mark.parametrize("bits", [4, 8])
def test_gspo_uses_modern_quantization_config(monkeypatch, bits):
    import torch
    import transformers

    from stateset_agents.training import gspo_trainer
    from stateset_agents.training.gspo_config import GSPOConfig

    monkeypatch.setattr(gspo_trainer, "_require_peft", lambda: None)
    monkeypatch.setattr(gspo_trainer, "_require_bitsandbytes", lambda: None)
    # Capture constructor inputs without requiring a CUDA bitsandbytes install.
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", lambda **kwargs: kwargs)
    config = GSPOConfig(use_4bit=bits == 4, use_8bit=bits == 8)
    manager = gspo_trainer.GSPOModelManager(config)
    kwargs = {"torch_dtype": torch.bfloat16}
    manager._prepare_model_kwargs(kwargs)
    quant = kwargs["quantization_config"]
    assert quant[f"load_in_{bits}bit"] is True
    assert quant["bnb_4bit_compute_dtype"] == torch.bfloat16
    assert "load_in_4bit" not in kwargs and "load_in_8bit" not in kwargs


@pytest.mark.asyncio
async def test_tiny_gemma4_checkpoint_loads_and_updates_lora(tmp_path, monkeypatch):
    """Exercise actual Transformers + PEFT on CPU, without downloading weights."""
    import torch

    transformers = pytest.importorskip("transformers", minversion="5.5.0")
    peft = pytest.importorskip("peft")
    from stateset_agents.core.transformers_compat import load_generation_model
    from stateset_agents.training.gspo_trainer import GSPOModelManager, GSPOTrainer
    from tests._tiny_tokenizer import tiny_tokenizer

    text_config = transformers.Gemma4TextConfig(
        vocab_size=256,
        vocab_size_per_layer_input=256,
        hidden_size=32,
        hidden_size_per_layer_input=8,
        intermediate_size=64,
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        layer_types=["sliding_attention", "full_attention"] * 2,
        max_position_embeddings=128,
        sliding_window=16,
    )
    config = transformers.Gemma4Config(
        text_config=text_config, vision_config=None, audio_config=None
    )
    base = transformers.Gemma4ForConditionalGeneration(config)
    base_dir = tmp_path / "base"
    base.save_pretrained(base_dir)
    del base
    model, _ = load_generation_model(
        transformers.AutoModelForCausalLM, str(base_dir), {"local_files_only": True}
    )
    training_config = starter.get_gemma4_small_gspo_config(
        starter.get_gemma4_small_config(
            model_name=str(base_dir),
            lora_r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bf16=False,
            learning_rate=1e-3,
            num_generations=2,
            max_prompt_length=16,
            max_completion_length=16,
        )
    )
    training_config.warmup_ratio = 0.0
    manager = GSPOModelManager(training_config)
    model = manager._apply_lora(manager._prepare_base_model(model))
    assert isinstance(model, peft.PeftModel)
    model.train()
    trainable = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    tokens = torch.tensor([[2, 3, 4, 5]])
    loss = model(input_ids=tokens, labels=tokens).loss
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(before, dict(model.named_parameters())[name])
        for name, before in trainable.items()
    )

    class Reward:
        async def compute_reward(self, turns, context):
            return SimpleNamespace(total_reward=float(turns[0].content == "ok"))

    trainer = GSPOTrainer(
        config=training_config,
        model=model,
        tokenizer=tiny_tokenizer(),
        agent=None,
        environment=None,
        reward_model=Reward(),
        ref_model=None,
    )

    async def generate(prompt, count):
        # Controlled rewards isolate the optimizer from random tiny-model text.
        return [("ok", -5.0), ("no", -5.0)]

    monkeypatch.setattr(trainer.generator, "generate_group_responses", generate)
    before_rl = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    await trainer.train_step(["help"], num_groups=1)
    assert trainer.optimizer.state
    assert any(
        not torch.equal(before, dict(model.named_parameters())[name])
        for name, before in before_rl.items()
    )

    adapter_dir = tmp_path / "adapter"
    trainer.save_model(str(adapter_dir))
    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "tokenizer.json").is_file()
    assert (adapter_dir / "training_metrics.json").is_file()
    base, _ = load_generation_model(
        transformers.AutoModelForCausalLM, str(base_dir), {"local_files_only": True}
    )
    reloaded = peft.PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            model(input_ids=tokens).logits, reloaded(input_ids=tokens).logits
        )
