"""Pinned model readiness, fail-closed loaders and exact text SFT contracts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from stateset_agents.core.frontier_models import (
    FRONTIER_MODELS,
    get_frontier_model,
    require_frontier_training,
)
from stateset_agents.training.frontier_sft import (
    FrontierSFTConfig,
    describe_frontier_sft,
    encode_frontier_rows,
    prepare_frontier_config,
    train_frontier_sft,
)

NATIVE_KEYS = ["glm5.3", "glm5.3-flash", "qwen3.8-2.4t-a95b", "qwen3.8-flash-next"]
BLOCKED_KEYS = [
    "mimo-v2.6-pro",
    "mimo-v2.6-pro-mopd",
    "kimi-k3",
    "deepseek-v4.1-flash",
    "deepseek-v4.1-flash-1017",
]


def _data(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        )
        + "\n"
    )
    return str(path)


@pytest.mark.parametrize("spec", FRONTIER_MODELS, ids=lambda spec: spec.key)
def test_pinned_roster_and_actual_published_attention_targets(spec):
    evidence_root = Path(__file__).parents[2] / "benchmark_results/training_contracts"
    audits = ("20261015/frontier-models", "20261017/deepseek-v4.1-flash-1017")
    row = next(
        item
        for audit in audits
        for item in json.loads((evidence_root / audit / "metadata.json").read_text())
        if item["key"] == spec.key
    )
    assert row["audit_date"] == spec.audit_date
    assert row["revision"] == spec.revision
    assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)
    assert row["checkpoint_bytes"] > 100_000_000_000
    samples = row["representative_attention_modules"]
    assert samples
    for target in spec.attention_targets:
        assert any(name.endswith("." + target) for name in samples), target
    assert all(re.fullmatch(spec.lora_regex, name) for name in samples)
    assert not any(
        re.fullmatch(spec.lora_regex, name)
        for name in (
            "visual.blocks.0.attn.q_proj",
            "model.visual.blocks.0.attn.q_proj",
            "audio_encoder.layers.0.self_attn.o_proj",
            "mtp.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.experts.0.q_proj",
        )
    )
    assert get_frontier_model(spec.model_id.upper()) is spec
    assert spec.describe()["full_size_training_verified"] is False


@pytest.mark.parametrize("name", BLOCKED_KEYS)
def test_unsupported_layout_fails_before_data_or_optional_dependencies(
    name, monkeypatch
):
    from stateset_agents.training import frontier_sft

    monkeypatch.setattr(
        frontier_sft, "version", lambda _: pytest.fail("runtime touched")
    )
    with pytest.raises(RuntimeError, match="training is blocked"):
        train_frontier_sft(
            FrontierSFTConfig(model=name, train_data="nonexistent.jsonl")
        )
    assert get_frontier_model(name).describe()["training_path_available"] is False


@pytest.mark.parametrize("name", BLOCKED_KEYS)
def test_generic_loader_cannot_bypass_packed_layout_block(name):
    from stateset_agents.core.transformers_compat import load_generation_model

    loader = MagicMock()
    with pytest.raises(RuntimeError, match="training is blocked"):
        load_generation_model(
            loader, get_frontier_model(name).model_id, {}, for_training=True
        )
    loader.from_pretrained.assert_not_called()


@pytest.mark.parametrize("name", NATIVE_KEYS)
def test_native_path_reports_explicit_dtype_and_unverified_scope(name):
    spec = get_frontier_model(name)
    require_frontier_training(spec)
    report = describe_frontier_sft(FrontierSFTConfig(model=name))
    assert report["training_path_available"] is True
    assert report["optimizer_resume_supported"] is False
    assert report["full_size_training_verified"] is False
    assert report["dequantize_to_bf16"] is name.startswith("glm")
    assert report["license"] != "apache-2.0"


@pytest.mark.parametrize(
    "values",
    [
        {"model": "unknown"},
        {"max_steps": 0},
        {"max_steps": True},
        {"max_length": 1},
        {"learning_rate": float("nan")},
        {"gradient_accumulation_steps": 0},
        {"train_data": " "},
        {"output_dir": ""},
        {"resume_from_checkpoint": "old"},
        {"use_4bit": True},
        {"revision": "main"},
    ],
)
def test_invalid_config_rejected(values):
    with pytest.raises(ValueError):
        FrontierSFTConfig.model_validate({"model": "glm5.3", **values})


def test_data_preview_hashes_and_rejects_overlap(tmp_path):
    path = _data(tmp_path)
    report = describe_frontier_sft(
        FrontierSFTConfig(model="glm5.3", train_data=path), check_data=True
    )
    assert report["train_examples"] == 1
    assert len(report["dataset_sha256"]["train"]) == 64
    with pytest.raises(ValueError, match="overlap"):
        describe_frontier_sft(
            FrontierSFTConfig(model="glm5.3", train_data=path, validation_data=path),
            check_data=True,
        )


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    engine = Tokenizer(
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
    engine.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=engine, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tok.chat_template = "{% for message in messages %}{{ message.role + ': ' + message.content + ' [EOS] ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    return tok


def test_real_tokenizer_labels_preserve_native_ids_and_mask_user(tmp_path):
    from stateset_agents.data.text_sft import load_text_sft_data

    rows, _ = load_text_sft_data(_data(tmp_path))
    tokenizer = _tokenizer()
    result = encode_frontier_rows(tokenizer, rows, 32)[0]
    assert result["input_ids"] == tokenizer.apply_chat_template(
        rows[0]["messages"], tokenize=True, return_dict=False
    )
    assert result["labels"][result["input_ids"].index(4)] == -100
    assert result["labels"][result["input_ids"].index(6)] == 6
    with pytest.raises(ValueError, match="no truncation"):
        encode_frontier_rows(tokenizer, rows, 2)
    tokenizer.apply_chat_template = MagicMock(return_value=[999])
    with pytest.raises(ValueError, match="cannot locate|differ"):
        encode_frontier_rows(tokenizer, rows, 32)


def test_fsdp_plan_wraps_layers_and_preserves_existing_rules():
    config = SimpleNamespace(model_type="glm_moe_dsa")
    text = SimpleNamespace(base_model_fsdp_plan={"embed_tokens": "keep_full_weight"})
    config.get_text_config = lambda: text
    assert prepare_frontier_config(get_frontier_model("glm5.3"), config) is config
    assert text.base_model_fsdp_plan == {
        "embed_tokens": "keep_full_weight",
        "layers.*": "free_full_weight",
        "norm": "keep_full_weight",
    }
    with pytest.raises(ValueError, match="model_type"):
        prepare_frontier_config(get_frontier_model("glm5.3-flash"), config)


@pytest.mark.parametrize("name", NATIVE_KEYS)
def test_distributed_recipe_pins_config_tokenizer_weights_and_exports(
    name, tmp_path, monkeypatch
):
    import torch
    import transformers
    import trl

    from stateset_agents.training import frontier_sft

    spec = get_frontier_model(name)
    monkeypatch.setattr(frontier_sft, "version", lambda _: "99.0.0")
    monkeypatch.setattr(transformers, spec.architecture, MagicMock(), raising=False)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    tokenizer = _tokenizer()
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", MagicMock(return_value=tokenizer)
    )
    tokenizer.save_pretrained = MagicMock()
    cfg = SimpleNamespace(model_type=spec.model_type)
    cfg.get_text_config = lambda: cfg
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", MagicMock(return_value=cfg)
    )
    mesh = MagicMock()
    monkeypatch.setitem(
        sys.modules, "transformers.distributed", SimpleNamespace(DistributedConfig=mesh)
    )
    args = MagicMock()
    monkeypatch.setattr(trl, "SFTConfig", args)
    trainer = MagicMock()
    trainer.model.named_parameters.return_value = [
        (
            spec.attention_prefix.replace("\\", "")
            + ".0.self_attn.o_proj.lora_B.default.weight",
            torch.nn.Parameter(torch.ones(1)),
        )
    ]
    trainer.state.global_step = 2
    trainer.train.return_value.metrics = {"train_loss": 0.5}
    trainer.is_world_process_zero.return_value = True

    def agree_on_identity(factory, *, world_size):
        assert world_size == 8
        assert factory()["dataset"]["train_examples"] == 1
        return {"verified": True, "world_size": world_size}

    consensus = MagicMock(side_effect=agree_on_identity)
    monkeypatch.setattr(frontier_sft, "verify_sft_rank_consensus", consensus)

    def construct_after_consensus(**kwargs):
        assert consensus.called
        return trainer

    factory = MagicMock(side_effect=construct_after_consensus)
    monkeypatch.setattr(trl, "SFTTrainer", factory)
    settings = FrontierSFTConfig(
        model=name, train_data=_data(tmp_path), output_dir=str(tmp_path / "adapter")
    )
    captured_source = Path(settings.train_data).read_bytes()

    def change_source_during_tokenizer_load(*args, **kwargs):
        Path(settings.train_data).write_text("not valid JSON anymore")
        return tokenizer

    transformers.AutoTokenizer.from_pretrained.side_effect = (
        change_source_during_tokenizer_load
    )
    report = train_frontier_sft(settings)
    assert (
        report["dataset_sha256"]["train"] == hashlib.sha256(captured_source).hexdigest()
    )
    Path(settings.train_data).write_bytes(captured_source)
    assert report["global_step"] == 2
    assert report["rank_consensus"]["verified"] is True
    consensus.assert_called_once()
    assert consensus.call_args.kwargs["world_size"] == 8
    identity = consensus.call_args.args[0]()
    assert identity["model"] == spec.model_id
    assert identity["dataset"]["dataset_sha256"] == report["dataset_sha256"]
    assert "train_data" not in identity["training_config"]
    assert "output_dir" not in identity["training_config"]
    assert identity["encoded_data"]["train"]
    assert report["full_size_training_verified"] is False
    mesh.assert_called_once_with(fsdp_size=8, ep_size=1)
    kwargs = args.call_args.kwargs
    assert kwargs["dataset_kwargs"] == {"skip_prepare_dataset": True}
    assert kwargs["save_only_model"] is True
    assert kwargs["router_aux_loss_coef"] == 0.0
    assert kwargs["model_init_kwargs"]["revision"] == spec.revision
    assert "device_map" not in kwargs["model_init_kwargs"]
    assert (
        "quantization_config" in kwargs["model_init_kwargs"]
        if name.startswith("glm")
        else "quantization_config" not in kwargs["model_init_kwargs"]
    )
    transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
        spec.model_id, revision=spec.revision, trust_remote_code=False
    )
    assert factory.call_args.kwargs["peft_config"].target_modules == spec.lora_regex
    assert (tmp_path / "adapter/stateset_training.json").is_file()
    assert factory.call_args.kwargs["train_dataset"][0]["labels"].count(-100) > 0
    tokenizer.save_pretrained.assert_called_once()
    trainer.state.global_step = 0
    with pytest.raises(RuntimeError, match="without optimizer steps"):
        train_frontier_sft(settings)
    Path(settings.train_data).write_bytes(captured_source)
    trainer.state.global_step = 2
    trainer.model.named_parameters.return_value = [
        (
            "model.visual.blocks.0.q_proj.lora_B.default.weight",
            torch.nn.Parameter(torch.ones(1)),
        )
    ]
    with pytest.raises(RuntimeError, match="only language attention"):
        train_frontier_sft(settings)
    Path(settings.train_data).write_bytes(captured_source)
    consensus.side_effect = RuntimeError(
        "SFT preflight disagreement across ranks: dataset"
    )
    factory.reset_mock()
    with pytest.raises(RuntimeError, match="preflight disagreement"):
        train_frontier_sft(settings)
    factory.assert_not_called()


def test_cli_roster_and_config_are_lightweight_and_report_blockers(tmp_path):
    from stateset_agents.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["frontier-models"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert len(rows) == 9
    assert sum(row["training_path_available"] for row in rows) == 4
    config = tmp_path / "sft.json"
    result = runner.invoke(
        app,
        ["frontier-sft", "--model", "qwen3.8-2.4t-a95b", "--write-config", str(config)],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(config.read_text())["model"] == "qwen3.8-2.4t-a95b"
    result = runner.invoke(app, ["frontier-sft", "--model", "kimi-k3", "--no-dry-run"])
    assert result.exit_code != 0
    assert "training is blocked" in result.output
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from stateset_agents.cli import app; from typer.testing import CliRunner; r=CliRunner().invoke(app,['frontier-models']); assert r.exit_code==0, r.output; assert not {'torch','transformers','trl','peft'} & set(sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def test_kimi_starter_uses_published_hybrid_targets_and_pin():
    from stateset_agents.training.kimi_k3_starter import (
        create_kimi_k3_agent_config,
        get_kimi_k3_config,
        get_kimi_k3_gspo_config,
    )

    spec = get_frontier_model("kimi-k3")
    cfg = get_kimi_k3_config()
    assert cfg.use_4bit is False
    assert tuple(cfg.lora_target_modules) == spec.attention_targets
    assert create_kimi_k3_agent_config(cfg).model_revision == spec.revision
    assert get_kimi_k3_gspo_config(cfg).model_revision == spec.revision
    assert any("MXFP4" in warning for warning in cfg.validate())


def test_native_qwen_template_options_respect_thinking_only_checkpoint():
    from stateset_agents.training.frontier_sft import prepare_frontier_tokenizer

    thinking = _tokenizer()
    thinking.chat_template = "{% if enable_thinking is defined and enable_thinking is false %}{{ raise_exception('Disabling thinking is not supported.') }}{% endif %}{{ reasoning_effort }}"
    prepare_frontier_tokenizer(thinking, get_frontier_model("qwen3.8-2.4t-a95b"))
    assert (
        thinking.apply_chat_template([{"role": "user", "content": "q"}], tokenize=False)
        == "low"
    )
    switchable = _tokenizer()
    switchable.chat_template = "{{ enable_thinking is false }}"
    prepare_frontier_tokenizer(switchable, get_frontier_model("qwen3.8-flash-next"))
    assert (
        switchable.apply_chat_template(
            [{"role": "user", "content": "q"}], tokenize=False
        )
        == "True"
    )


def test_frontier_rejects_old_torch_before_optional_runtime_imports(
    tmp_path, monkeypatch
):
    from stateset_agents.training import frontier_sft

    monkeypatch.setattr(
        frontier_sft, "version", lambda name: "2.6.0" if name == "torch" else "99.0.0"
    )
    with pytest.raises(RuntimeError, match=r"torch>=2.7.0"):
        train_frontier_sft(
            FrontierSFTConfig(model="glm5.3", train_data=_data(tmp_path))
        )
