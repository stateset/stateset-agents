"""Real tiny Muse Spark contracts; no released checkpoint weights or GPU use.

Run in the optional Muse runtime. An optional STATESET_MUSE_TEMPLATE_PATH points
to a separately retrieved native template; its tokenizer here remains synthetic.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

if os.environ.get("STATESET_REQUIRE_MUSE_RUNTIME") == "1":
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
else:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "MuseSparkForConditionalGeneration"):
    if os.environ.get("STATESET_REQUIRE_MUSE_RUNTIME") == "1":
        raise RuntimeError("Required native Muse Spark runtime is missing")
    pytest.skip(
        "requires transformers>=5.20 with native Muse Spark", allow_module_level=True
    )

from stateset_agents.training.muse_spark import (  # noqa: E402
    MUSE_SPARK_LORA_TARGETS,
    MuseSparkSFTConfig,
    _prepare_tokenizer,
    _validate_encoded_rows,
)


def _record(name, report):
    destination = os.environ.get("STATESET_MUSE_CONTRACT_REPORT_DIR")
    if destination:
        output = Path(destination)
        output.mkdir(parents=True, exist_ok=True)
        (output / name).write_text(json.dumps(report, indent=2) + "\n")


def _tiny_model():
    from transformers import MuseSparkConfig, MuseSparkForConditionalGeneration

    config = MuseSparkConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "moe_hidden_size": 16,
            "max_position_embeddings": 2048,
            "sliding_window": 32,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "use_cache": False,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "patch_size": 2,
            "pos_emb_height": 4,
            "pos_emb_width": 4,
            "patch_temporal": 1,
            "merge_size": 2,
        },
        audio_config={
            "patch_sizes": [2],
            "hidden_sizes": [16],
            "depths": [1],
            "num_hidden_layers": 1,
            "out_hidden_size": 32,
        },
        out_hidden_size=128,
        projector_hidden_size=32,
        image_token_id=61,
        video_token_id=62,
        audio_token_id=63,
    )
    return MuseSparkForConditionalGeneration(config)


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "[PAD]": 0,
        "<|begin_of_text|>": 1,
        "<|end_of_text|>": 2,
        "[UNK]": 3,
        "question": 4,
        "answer": 5,
        "<|start|>": 6,
        "<|message|>": 7,
        "<|eot|>": 8,
        "<|eom|>": 9,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="<|begin_of_text|>",
        eos_token="<|eot|>",
        additional_special_tokens=["<|start|>", "<|message|>", "<|eot|>", "<|eom|>"],
    )
    native = os.environ.get("STATESET_MUSE_TEMPLATE_PATH")
    tokenizer.chat_template = (
        Path(native).read_text()
        if native
        else "{{bos_token}}{% for m in messages %}"
        "{% if m.role == 'assistant' %}{% generation %}{{m.content}} {{eos_token}}{% endgeneration %}"
        "{% else %}{{m.content}} {% endif %}{% endfor %}"
    )
    _prepare_tokenizer(tokenizer, MuseSparkSFTConfig())
    return tokenizer


def test_real_moe_adapter_update_frozen_encoders_and_reload(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model

    torch.manual_seed(42)
    base = _tiny_model()
    assert sum(p.numel() for p in base.parameters()) < 1_000_000
    model = get_peft_model(
        base, LoraConfig(r=2, lora_alpha=4, target_modules=MUSE_SPARK_LORA_TARGETS)
    )
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert names and all(
        "language_model.layers." in name and "self_attn" in name for name in names
    )
    vision = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if "vision_tower" in name or "audio_tower" in name
    }
    before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    ids = torch.tensor([[4, 5, 6, 7]])
    labels = torch.tensor([[-100, -100, 6, 7]])
    model.train()
    loss = model(input_ids=ids, labels=labels, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.parameters()
        if p.requires_grad
    )
    torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    ).step()
    assert any(
        not torch.equal(before[name], p)
        for name, p in model.named_parameters()
        if name in before
    )
    assert all(
        torch.equal(vision[name], p)
        for name, p in model.named_parameters()
        if name in vision
    )
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=ids, use_cache=False).logits
    model.save_pretrained(tmp_path / "adapter")
    torch.manual_seed(42)
    restored = PeftModel.from_pretrained(_tiny_model(), tmp_path / "adapter")
    restored.eval()
    with torch.no_grad():
        actual = restored(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
    _record(
        "native_adapter.json",
        {
            "architecture": type(base).__name__,
            "parameters": sum(p.numel() for p in base.parameters()),
            "random_initialization": True,
            "released_weights_loaded": False,
            "device": "cpu",
            "optimizer_updates": 1,
            "loss": float(loss.detach()),
            "nonzero_gradients_verified": True,
            "language_attention_adapters_only": True,
            "frozen_vision_audio_verified": True,
            "adapter_reload_verified": True,
        },
    )


def test_real_sft_trainer_masks_users_and_commits_update(tmp_path):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = _tokenizer()
    rows = [
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    ]
    _validate_encoded_rows(tokenizer, rows, 1024)
    expected_ids = tokenizer.apply_chat_template(
        rows[0]["messages"],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )
    trainer = SFTTrainer(
        model=_tiny_model(),
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(rows),
        peft_config=LoraConfig(
            r=2, lora_alpha=4, target_modules=MUSE_SPARK_LORA_TARGETS
        ),
        args=SFTConfig(
            output_dir=str(tmp_path),
            use_cpu=True,
            bf16=False,
            max_steps=1,
            max_length=1024,
            assistant_only_loss=True,
            eos_token="<|eot|>",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            gradient_checkpointing=False,
            save_strategy="no",
            report_to=[],
            dataloader_pin_memory=False,
        ),
    )
    assert trainer.train_dataset[0]["input_ids"] == expected_ids
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0]
    ids = batch["input_ids"][0]
    assert torch.all(labels[ids == 4] == -100)
    assert torch.any(labels[ids == 5] == 5)
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    }
    result = trainer.train()
    assert trainer.state.global_step == 1
    assert result.metrics["train_loss"] >= 0
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in trainer.model.named_parameters()
        if name in before
    )
    assert all(
        torch.isfinite(parameter).all()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    trainer.save_model(str(tmp_path / "export"))
    assert (tmp_path / "export" / "adapter_config.json").is_file()
    _record(
        "native_sft_trainer.json",
        {
            "architecture": "MuseSparkForConditionalGeneration",
            "random_initialization": True,
            "released_weights_loaded": False,
            "device": "cpu",
            "optimizer_updates": trainer.state.global_step,
            "loss": result.metrics["train_loss"],
            "synthetic_tokenizer": True,
            "template_from_public_model_metadata": bool(
                os.environ.get("STATESET_MUSE_TEMPLATE_PATH")
            ),
            "user_tokens_masked": True,
            "assistant_targets_verified": True,
            "preflight_ids_equal_trainer_ids": True,
            "adapter_export_verified": True,
            "adapter_weights_changed": True,
        },
    )
