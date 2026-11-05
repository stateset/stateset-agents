"""Real tiny native architecture updates; no published weights, GPUs or sharding.

These tests certify CPU autograd, adapter targets, native SFT label handling
and adapter reload only. Distributed and released-weight runs remain unverified.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path

import pytest

if os.environ.get("STATESET_REQUIRE_FRONTIER_RUNTIME") == "1":
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
else:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

from stateset_agents.core.frontier_models import get_frontier_model  # noqa: E402
from stateset_agents.training.frontier_sft import (  # noqa: E402
    encode_frontier_rows,
    prepare_frontier_config,
)

NATIVE_KEYS = ["glm5.3", "glm5.3-flash", "qwen3.8-2.4t-a95b", "qwen3.8-flash-next"]


def _tiny_model(spec):
    from transformers import AutoConfig

    text = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "pad_token_id": 1,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "use_cache": False,
        "moe_intermediate_size": 8,
        "tie_word_embeddings": False,
    }
    if spec.key.startswith("glm"):
        text.update(
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            kv_lora_rank=4,
            q_lora_rank=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=2,
            v_head_dim=4,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=4,
            mlp_layer_types=["sparse", "sparse"],
        )
        if spec.key == "glm5.3-flash":
            text.update(
                qk_rope_head_dim=0,
                layer_types=["linear_attention", "indexed_attention"],
                linear_head_dim=4,
                linear_num_heads=2,
                linear_conv_kernel_dim=2,
                hc_mult=2,
                hc_sinkhorn_iters=2,
                index_kpool=2,
            )
    else:
        text.update(
            num_experts=4,
            num_experts_per_tok=2,
            shared_expert_intermediate_size=8,
            head_dim=8,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_conv_kernel_dim=2,
            partial_rotary_factor=0.5,
            layer_types=["linear_attention", "full_attention"],
        )
        if spec.key == "qwen3.8-flash-next":
            text.update(
                hc_count=2,
                hc_lowrank=4,
                ple_layer_ids=[1],
                ple_embed_dim=16,
                ple_conv_kernel_size=2,
                ngram_size=2,
                heads_per_ngram=2,
                ngram_vocab_size_base=17,
                make_ngram_vocab_size_divisible_by=2,
                split_ngram_parts=1,
                indexer_n_heads=2,
                indexer_kv_heads=1,
                indexer_head_dim=4,
                indexer_budget=4,
                indexer_compress_ratio=2,
            )
    if spec.key in ("glm5.3-flash", "qwen3.8-flash-next"):
        vision = {
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "out_hidden_size": 16,
        }
        if spec.key == "glm5.3-flash":
            vision.update(image_size=8, projection_intermediate_size=32)
        else:
            vision.update(num_position_embeddings=16)
        config = AutoConfig.for_model(
            spec.model_type,
            text_config=text,
            vision_config=vision,
            image_token_id=30,
            video_token_id=31,
        )
    else:
        config = AutoConfig.for_model(spec.model_type, **text)
    config._attn_implementation = "eager"
    prepare_frontier_config(spec, config)
    return getattr(transformers, spec.architecture)(config)


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[EOS]": 2,
                "user:": 3,
                "question": 4,
                "assistant:": 5,
                "answer": 6,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = "{% for m in messages %}{{m.role + ': ' + m.content + ' [EOS] '}}{% endfor %}{% if add_generation_prompt %}{{'assistant: '}}{% endif %}"
    return tokenizer


@pytest.mark.parametrize("name", NATIVE_KEYS)
def test_native_attention_sft_update_frozen_experts_and_adapter_roundtrip(
    name, tmp_path
):
    spec = get_frontier_model(name)
    if not hasattr(transformers, spec.architecture):
        if os.environ.get("STATESET_REQUIRE_FRONTIER_RUNTIME") == "1":
            pytest.fail(f"Missing required native {spec.architecture}")
        pytest.skip("requires the optional [frontier] runtime")
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(42)
    base = _tiny_model(spec)
    assert sum(p.numel() for p in base.parameters()) < 1_000_000
    modules = dict(base.named_modules())
    targets = [n for n in modules if re.fullmatch(spec.lora_regex, n)]
    assert targets
    assert all(isinstance(modules[n], torch.nn.Linear) for n in targets)
    for target in spec.attention_targets:
        assert any(n.endswith("." + target) for n in targets), target
    assert any("layers.*" in key for key in base._fsdp_plan)
    tokenizer = _tokenizer()
    rows = [
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    ]
    encoded = encode_frontier_rows(tokenizer, rows, 32)
    assert encoded[0]["labels"][encoded[0]["input_ids"].index(4)] == -100
    assert encoded[0]["labels"][encoded[0]["input_ids"].index(6)] == 6
    args = SFTConfig(
        output_dir=str(tmp_path / "run"),
        max_steps=1,
        max_length=32,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=0.01,
        use_cpu=True,
        bf16=False,
        fp16=False,
        dataset_num_proc=1,
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,
        padding_free=False,
        router_aux_loss_coef=0.0,
        assistant_only_loss=False,
        completion_only_loss=False,
        save_strategy="no",
        report_to=[],
        logging_steps=1,
        seed=42,
        data_seed=42,
    )
    trainer = SFTTrainer(
        model=base,
        args=args,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(encoded),
        peft_config=LoraConfig(
            r=2, lora_alpha=4, target_modules=spec.lora_regex, lora_dropout=0
        ),
    )
    assert trainer.train_dataset[0]["labels"] == encoded[0]["labels"]
    before = {
        n: p.detach().clone()
        for n, p in trainer.model.named_parameters()
        if p.requires_grad
    }
    assert before and all("lora_" in n for n in before)
    frozen = {
        n: p.detach().clone()
        for n, p in trainer.model.named_parameters()
        if not p.requires_grad
    }
    result = trainer.train()
    assert trainer.state.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert any(
        not torch.equal(before[n], p)
        for n, p in trainer.model.named_parameters()
        if n in before
    )
    assert all(
        torch.equal(frozen[n], p)
        for n, p in trainer.model.named_parameters()
        if n in frozen
    )
    model = trainer.model.eval()
    ids = torch.tensor([encoded[0]["input_ids"]])
    with torch.no_grad():
        expected = model(input_ids=ids, use_cache=False).logits
    model.save_pretrained(tmp_path / "adapter")
    torch.manual_seed(42)
    reloaded = PeftModel.from_pretrained(_tiny_model(spec), tmp_path / "adapter").eval()
    with torch.no_grad():
        actual = reloaded(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(expected, actual)
    report_dir = os.environ.get("STATESET_FRONTIER_CONTRACT_REPORT_DIR")
    if report_dir:
        report = {
            "model": spec.model_id,
            "revision": spec.revision,
            "architecture": spec.architecture,
            "loss": result.training_loss,
            "global_step": 1,
            "attention_modules": targets,
            "frozen_parameters_unchanged": True,
            "adapter_reload_equal": True,
            "scope": "random tiny native CPU; synthetic tokenizer; no quantized checkpoint or distributed execution",
            "full_size_training_verified": False,
        }
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        (Path(report_dir) / (spec.key + ".json")).write_text(
            json.dumps(report, indent=2) + "\n"
        )
