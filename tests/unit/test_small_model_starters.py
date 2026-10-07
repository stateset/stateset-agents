"""Small-model CLI contracts and real, local architecture training checks."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from stateset_agents.core.model_presets import PRESETS

NAMES = [
    "granite4-micro",
    "llama3.2-1b",
    "llama3.2-3b",
    "qwen3-4b-instruct",
    "olmo3-7b",
    "deepseek-r1-1.5b",
    "deepseek-r1-7b",
    "qwen3.5-2b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "smollm3-3b",
    "phi4-mini",
    "ministral3-3b",
    "ministral3-8b",
    "lfm2.5-2.6b",
]

ARCHITECTURES = [
    "qwen3.5-2b",
    "smollm3-3b",
    "phi4-mini",
    "ministral3-3b",
    "lfm2.5-2.6b",
    "granite4-micro",
    "llama3.2-1b",
    "qwen3-4b-instruct",
    "olmo3-7b",
    "deepseek-r1-1.5b",
]


@pytest.mark.parametrize("name", NAMES)
def test_cli_profiles_config_and_dispatch(name, tmp_path, monkeypatch):
    from stateset_agents.cli import app

    preset = PRESETS[name]
    runner = CliRunner()
    result = runner.invoke(app, [preset.cli_command, "--list-profiles", "--json"])
    assert result.exit_code == 0, result.output
    profiles = json.loads(result.output)["profiles"]
    assert set(profiles) == {"balanced", "memory", "quality"}
    assert profiles["memory"]["config"]["use_4bit"]
    assert profiles["balanced"]["config"]["model_name"] == preset.model_id
    path = tmp_path / "config.json"
    result = runner.invoke(
        app,
        [
            preset.cli_command,
            "--starter-profile",
            "memory",
            "--write-config",
            str(path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["config"]["output_dir"] == preset.cli_default_output_dir
    assert payload["gspo_overrides"]["trust_remote_code"] is False
    assert payload["gspo_overrides"]["lora_target_modules"] == list(
        preset.lora_target_modules
    )
    module = importlib.import_module(
        f"stateset_agents.training.{preset.starter_module}"
    )
    train = AsyncMock(return_value="trained")
    monkeypatch.setattr(module, f"run_{preset.cli_symbol_infix}_config", train)
    result = runner.invoke(
        app, [preset.cli_command, "--config", str(path), "--no-dry-run"]
    )
    assert result.exit_code == 0, result.output
    train.assert_awaited_once()
    assert train.call_args.args[0].model_name == preset.model_id


@pytest.mark.parametrize("name", NAMES)
def test_native_sft_and_quantization_validation(name):
    from stateset_agents.training.sft import _trust_remote_code

    preset = PRESETS[name]
    assert _trust_remote_code(preset.model_id) is False
    module = importlib.import_module(
        f"stateset_agents.training.{preset.starter_module}"
    )
    config = getattr(module, f"get_{preset.cli_symbol_infix}_config")(
        model_name=preset.model_id, starter_profile="memory", use_lora=False
    )
    with pytest.raises(ValueError, match="requires use_lora"):
        config.validate()


def test_adapters_never_target_vision_or_audio():
    import torch

    from stateset_agents.core.lora_targets import text_lora_targets

    model = torch.nn.Module()
    model.language_model = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(4, 4)})
    model.vision_tower = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(4, 4)})
    model.audio_encoder = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(4, 4)})
    assert text_lora_targets(model, ["q_proj"]) == ["language_model.q_proj"]
    del model.language_model
    with pytest.raises(ValueError, match="only non-text"):
        text_lora_targets(model, ["q_proj"])


def _tiny_model(name, transformers):
    common = {
        "vocab_size": 256,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 1,
    }
    if name.startswith("granite"):
        return transformers.GraniteMoeHybridForCausalLM(
            transformers.GraniteMoeHybridConfig(
                **common,
                num_local_experts=0,
                num_experts_per_tok=0,
                shared_intermediate_size=64,
                layer_types=["attention", "attention"],
                position_embedding_type="rope",
                mamba_n_heads=4,
                mamba_d_head=16,
            )
        )
    if name.startswith("llama"):
        return transformers.LlamaForCausalLM(transformers.LlamaConfig(**common))
    if name.startswith("olmo"):
        return transformers.Olmo3ForCausalLM(transformers.Olmo3Config(**common))
    if name.startswith("deepseek"):
        return transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**common))
    if name == "qwen3-4b-instruct":
        return transformers.Qwen3ForCausalLM(
            transformers.Qwen3Config(**common, head_dim=16)
        )
    if name.startswith("functiongemma"):
        return transformers.Gemma3ForCausalLM(
            transformers.Gemma3TextConfig(
                **common, head_dim=16, query_pre_attn_scalar=16
            )
        )
    if name.startswith("qwen3.5"):
        config = transformers.Qwen3_5TextConfig(
            **common,
            head_dim=16,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [2, 3, 3],
            },
        )
        return transformers.Qwen3_5ForCausalLM(config)
    if name.startswith("smollm"):
        return transformers.SmolLM3ForCausalLM(transformers.SmolLM3Config(**common))
    if name.startswith("phi"):
        return transformers.Phi3ForCausalLM(
            transformers.Phi3Config(**common, original_max_position_embeddings=128)
        )
    if name.startswith("ministral"):
        return transformers.Ministral3ForCausalLM(
            transformers.Ministral3Config(**common, head_dim=16)
        )
    return transformers.Lfm2ForCausalLM(
        transformers.Lfm2Config(**common, full_attn_idxs=[1], block_multiple_of=16)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ARCHITECTURES)
async def test_native_architecture_gspo_update_and_adapter_reload(
    name, tmp_path, monkeypatch
):
    """Use tiny random models, real PEFT/GSPO, and deterministic reward groups."""
    import torch

    transformers = pytest.importorskip("transformers", minversion="5.5.0")
    peft = pytest.importorskip("peft")
    from stateset_agents.core.transformers_compat import load_generation_model
    from stateset_agents.training.gspo_trainer import GSPOModelManager, GSPOTrainer
    from stateset_agents.training.sft import infer_lora_target_modules
    from tests._tiny_tokenizer import tiny_tokenizer

    torch.manual_seed(7)
    preset = PRESETS[name]
    module = importlib.import_module(
        f"stateset_agents.training.{preset.starter_module}"
    )
    base_dir = tmp_path / "base"
    _tiny_model(name, transformers).save_pretrained(base_dir)
    tiny_tokenizer().save_pretrained(base_dir)
    config = getattr(module, f"get_{preset.cli_symbol_infix}_config")(
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
    training = getattr(module, f"get_{preset.cli_symbol_infix}_gspo_config")(config)
    training.warmup_ratio = 0.0
    model, _ = load_generation_model(
        transformers.AutoModelForCausalLM, str(base_dir), {"local_files_only": True}
    )
    assert set(preset.lora_target_modules) <= set(infer_lora_target_modules(model))
    del model
    manager = GSPOModelManager(training)
    model, tokenizer = manager.load_model_and_tokenizer()
    # Every configured projection family must receive adapters, including hybrid layers.
    adapted = {
        n.split(".lora_A")[0].rsplit(".", 1)[-1]
        for n, _ in model.named_parameters()
        if ".lora_A" in n
    }
    assert set(preset.lora_target_modules) == adapted

    class Reward:
        async def compute_reward(self, turns, context):
            return SimpleNamespace(total_reward=float(turns[0].content == "ok"))

    trainer = GSPOTrainer(
        config=training,
        model=model,
        tokenizer=tokenizer,
        agent=None,
        environment=None,
        reward_model=Reward(),
        ref_model=None,
    )

    async def generate(prompt, count):
        return [("ok", -5.0), ("no", -5.0)]

    monkeypatch.setattr(trainer.generator, "generate_group_responses", generate)
    before = {
        n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
    }
    await trainer.train_step(["help"], num_groups=1)
    assert trainer.optimizer.state
    assert any(
        not torch.equal(old, dict(model.named_parameters())[n])
        for n, old in before.items()
    )
    adapter_dir = tmp_path / "adapter"
    trainer.save_model(str(adapter_dir))
    base, _ = load_generation_model(
        transformers.AutoModelForCausalLM, str(base_dir), {"local_files_only": True}
    )
    reloaded = peft.PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    reloaded.eval()
    tokens = torch.tensor([[2, 3, 4, 5]])
    with torch.no_grad():
        torch.testing.assert_close(
            model(input_ids=tokens).logits, reloaded(input_ids=tokens).logits
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [*ARCHITECTURES, "functiongemma-270m"])
async def test_native_architecture_sft_chat_data_and_agent_reload(
    name, tmp_path, monkeypatch
):
    """Train real adapters from JSONL, preserving chat boundaries and EOS labels."""
    import torch

    transformers = pytest.importorskip("transformers", minversion="5.5.0")
    pytest.importorskip("peft")
    pytest.importorskip("datasets")
    from tokenizers.processors import TemplateProcessing

    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
    from stateset_agents.training import sft
    from tests._tiny_tokenizer import tiny_tokenizer

    torch.manual_seed(7)
    base_dir = tmp_path / "base"
    _tiny_model(name, transformers).save_pretrained(base_dir)
    tokenizer = tiny_tokenizer()
    tokenizer.bos_token = "<pad>"
    tokenizer.pad_token = None  # Exercise the SFT fallback to EOS for padding.
    tokenizer.padding_side = "left" if name.startswith("lfm") else "right"
    tokenizer.init_kwargs["padding_side"] = tokenizer.padding_side
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="<pad> $A", special_tokens=[("<pad>", 0)]
    )
    tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}"
        "{{ message['role'] }}:{{ message['content'] }}{{ eos_token }}"
        "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
    )
    tokenizer.save_pretrained(base_dir)
    rows = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": answer},
            ]
        }
        for answer in ["hello", "hello there"]
    ]
    dataset_path = tmp_path / "chat.jsonl"
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    # Keep the production Trainer/PEFT path; only replace GPU precision/placement.
    monkeypatch.setattr(sft, "model_load_kwargs", lambda: {"local_files_only": True})
    build_arguments = sft.build_training_arguments

    def cpu_arguments(cls, **kwargs):
        kwargs.update(
            bf16=False,
            use_cpu=True,
            disable_tqdm=True,
            dataloader_pin_memory=False,
            warmup_ratio=0.0,
        )
        return build_arguments(cls, **kwargs)

    monkeypatch.setattr(sft, "build_training_arguments", cpu_arguments)
    trained = []

    trainer_init = transformers.Trainer.__init__

    def inspecting_init(self, **kwargs):
        trainer_init(self, **kwargs)
        assert self.data_collator.tokenizer.padding_side == tokenizer.padding_side
        features = [self.train_dataset[i] for i in range(len(rows))]
        for feature in features:
            assert feature["input_ids"].count(tokenizer.bos_token_id) == 1
            assert feature["input_ids"][-1] == tokenizer.eos_token_id
        batch = self.data_collator(features)
        visible = batch["attention_mask"].bool()
        assert (~visible).any(), "Unequal-length conversations must exercise padding"
        assert (batch["labels"][~visible] == -100).all()
        torch.testing.assert_close(
            batch["labels"][visible], batch["input_ids"][visible]
        )
        self.before = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        trained.append(self)

    monkeypatch.setattr(transformers.Trainer, "__init__", inspecting_init)
    output_dir = tmp_path / "adapter"
    sft.run_sft(
        sft.load_chat_dataset(dataset_path),
        str(base_dir),
        output_dir,
        num_epochs=1,
        lora_r=2,
        lora_alpha=4,
        learning_rate=1e-3,
        max_length=64,
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
        dataset_path=dataset_path,
        eval_prompts=["hi"],
        eval_max_new_tokens=2,
    )
    trainer = trained[0]
    assert trainer.state.global_step == 1
    assert any(
        not torch.equal(old, dict(trainer.model.named_parameters())[n])
        for n, old in trainer.before.items()
    )
    assert (output_dir / "adapter_config.json").is_file()
    assert (output_dir / "eval_results.json").is_file()

    agent = MultiTurnAgent(
        AgentConfig(
            model_name=str(base_dir),
            peft_path=str(output_dir),
            torch_dtype="float32",
            device_map="cpu",
            trust_remote_code=False,
            max_new_tokens=2,
            do_sample=False,
        )
    )
    await agent.initialize()
    trainer.model.eval()
    agent.model.eval()
    tokens = torch.tensor([[0, 3, 4, 1]])
    with torch.no_grad():
        torch.testing.assert_close(
            trainer.model(input_ids=tokens).logits,
            agent.model(input_ids=tokens).logits,
        )
    assert isinstance(await agent.generate_response("hi"), str)
