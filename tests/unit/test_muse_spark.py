"""Muse Spark previews, loss boundaries, identity, and distributed wiring."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from stateset_agents.training.muse_spark import (
    MUSE_SPARK_LORA_TARGETS,
    MUSE_SPARK_MODEL,
    MUSE_SPARK_REVISION,
    MuseSparkSFTConfig,
    _prepare_tokenizer,
    _validate_encoded_rows,
    describe_muse_spark,
    train_muse_spark,
)


@pytest.fixture
def text_data(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Help"},
                    {"role": "assistant", "content": "Here is help"},
                ]
            }
        )
        + "\n"
    )
    return path


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": "main"},
        {"revision": "0" * 40},
        {"max_steps": 0},
        {"max_steps": True},
        {"expert_parallel_size": 0},
        {"expert_parallel_size": 3},
        {"learning_rate": float("nan")},
        {"reasoning_effort": "off"},
        {"current_date": "2026-02-30"},
        {"current_date": "20261015"},
        {"train_data": " "},
        {"resume_from_checkpoint": "checkpoint-1"},
        {"use_4bit": True},
    ],
)
def test_rejects_unsupported_or_ambiguous_settings(changes):
    with pytest.raises(ValueError):
        MuseSparkSFTConfig(**changes)


def test_preview_and_data_identity(text_data, tmp_path):
    config = MuseSparkSFTConfig(train_data=str(text_data))
    preview = describe_muse_spark(config, check_data=True)
    assert preview["model_revision"] == MUSE_SPARK_REVISION
    assert preview["model_id"] == MUSE_SPARK_MODEL
    assert preview["train_examples"] == 1
    assert preview["base_weight_bytes_lower_bound"] > 1_600_000_000_000
    assert preview["gpu_training_verified"] is False
    assert preview["optimizer_resume_supported"] is False
    validation = tmp_path / "validation.jsonl"
    validation.write_text(
        text_data.read_text().replace('"messages":', '"id": "other", "messages":')
    )
    config.validation_data = str(validation)
    with pytest.raises(ValueError, match="overlap"):
        describe_muse_spark(config, check_data=True)


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"messages": []},
        {"messages": [{"role": "assistant", "content": "bad"}]},
        {"messages": [{"role": "user", "content": [{"type": "image", "image": "x"}]}]},
        {"messages": [{"role": "user", "content": "Hi", "extra": True}]},
        {"messages": [{"role": "user", "content": "Hi"}]},
    ],
)
def test_bad_data_fails_before_optional_runtime(text_data, row):
    text_data.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="train.jsonl:1"):
        train_muse_spark(MuseSparkSFTConfig(train_data=str(text_data)))


def test_encoder_rejects_truncation_and_missing_supervision():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = {
        "input_ids": [1, 2, 3],
        "assistant_masks": [0, 0, 1],
    }
    rows = [{"messages": []}]
    with pytest.raises(ValueError, match="no truncation"):
        _validate_encoded_rows(tokenizer, rows, 2)
    for mask in (None, [1], [1, 0, 0]):
        tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 2, 3],
            "assistant_masks": mask,
        }
        with pytest.raises(ValueError, match="no assistant"):
            _validate_encoded_rows(tokenizer, rows, 4)


def test_native_template_options_and_assistant_mask_are_deterministic():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        WordLevel({"[UNK]": 0, "user": 1, "answer": 2, "2040": 3}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{{current_date}} {{reasoning_effort}} {% for m in messages %}"
        "{% if m.role == 'assistant' %}{% generation %}{{m.content}}{% endgeneration %}"
        "{% else %}{{m.content}} {% endif %}{% endfor %}"
    )
    config = MuseSparkSFTConfig(current_date="2040-01-01", reasoning_effort="medium")
    _prepare_tokenizer(tokenizer, config)
    rows = [
        {
            "messages": [
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    ]
    rendered = tokenizer.apply_chat_template(rows[0]["messages"], tokenize=False)
    assert rendered.startswith("2040-01-01 medium")
    _validate_encoded_rows(tokenizer, rows, 32)
    encoded = tokenizer.apply_chat_template(
        rows[0]["messages"], return_dict=True, return_assistant_tokens_mask=True
    )
    assert [
        token
        for token, mask in zip(
            encoded["input_ids"], encoded["assistant_masks"], strict=True
        )
        if mask
    ] == [2]


def test_distributed_recipe_preserves_identity_and_exports(
    text_data, tmp_path, monkeypatch
):
    import torch
    import transformers
    import trl

    import stateset_agents.training.muse_spark as module

    distributed = pytest.importorskip("transformers.distributed")
    monkeypatch.setattr(module, "version", lambda name: "99.0.0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    loader = MagicMock()
    tokenizer = loader.return_value
    tokenizer.chat_template = "native {% generation %}answer{% endgeneration %}"
    tokenizer.backend_tokenizer = SimpleNamespace(
        to_str=lambda: '{"model":{"vocab":{"answer":2}}}'
    )
    tokenizer.special_tokens_map = {"eos_token": "<|eot|>"}
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokenizer.apply_chat_template.return_value = {
        "input_ids": [1, 2],
        "assistant_masks": [0, 1],
    }
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", loader)
    mesh = MagicMock()
    monkeypatch.setattr(distributed, "DistributedConfig", mesh)
    arguments = MagicMock()
    monkeypatch.setattr(trl, "SFTConfig", arguments)
    trainer_factory = MagicMock()
    trainer = trainer_factory.return_value
    trainer.model.named_parameters.return_value = [
        (
            "model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
            torch.nn.Parameter(torch.ones(1)),
        )
    ]
    trainer.state.global_step = 2
    trainer.train.return_value = SimpleNamespace(metrics={"train_loss": 0.5})
    trainer.is_world_process_zero.return_value = True
    monkeypatch.setattr(trl, "SFTTrainer", trainer_factory)

    def agree_on_identity(factory, *, world_size):
        assert world_size == 8
        assert factory()["dataset"]["train_examples"] == 1
        return {"verified": True, "world_size": world_size}

    consensus = MagicMock(side_effect=agree_on_identity)
    monkeypatch.setattr(module, "verify_sft_rank_consensus", consensus)

    def construct_after_consensus(**kwargs):
        assert consensus.called
        return trainer

    trainer_factory.side_effect = construct_after_consensus
    captured_source = text_data.read_bytes()

    def change_source_during_tokenizer_load(*args, **kwargs):
        text_data.write_text("not valid JSON anymore")
        return tokenizer

    loader.side_effect = change_source_during_tokenizer_load
    output = tmp_path / "out"
    report = train_muse_spark(
        MuseSparkSFTConfig(train_data=str(text_data), output_dir=str(output))
    )
    loader.assert_called_once_with(
        MUSE_SPARK_MODEL, revision=MUSE_SPARK_REVISION, trust_remote_code=False
    )
    mesh.assert_called_once_with(fsdp_size=8, ep_size=8)
    kwargs = arguments.call_args.kwargs
    assert kwargs["assistant_only_loss"] is True
    assert kwargs["eos_token"] == "<|eot|>"
    assert kwargs["save_only_model"] is True
    assert kwargs["packing"] is False
    assert kwargs["model_init_kwargs"]["revision"] == MUSE_SPARK_REVISION
    assert "device_map" not in kwargs["model_init_kwargs"]
    assert (
        trainer_factory.call_args.kwargs["peft_config"].target_modules
        == MUSE_SPARK_LORA_TARGETS
    )
    assert report["global_step"] == 2
    assert (
        report["dataset_sha256"]["train"] == hashlib.sha256(captured_source).hexdigest()
    )
    assert (
        trainer_factory.call_args.kwargs["train_dataset"][0]["messages"][1]["content"]
        == "Here is help"
    )
    receipt = json.loads((output / "stateset_training.json").read_text())
    assert receipt["config"]["current_date"] == "2026-10-15"
    trainer.train.assert_called_once_with()
    assert report["rank_consensus"]["verified"] is True
    consensus.assert_called_once()
    assert consensus.call_args.kwargs["world_size"] == 8
    identity = consensus.call_args.args[0]()
    assert identity["dataset"]["dataset_sha256"] == report["dataset_sha256"]
    assert "train_data" not in identity["training_config"]
    assert identity["encoded_data"]["train"]
    text_data.write_bytes(captured_source)
    consensus.side_effect = RuntimeError(
        "SFT preflight disagreement across ranks: tokenizer"
    )
    trainer_factory.reset_mock()
    with pytest.raises(RuntimeError, match="preflight disagreement"):
        train_muse_spark(
            MuseSparkSFTConfig(train_data=str(text_data), output_dir=str(output))
        )
    trainer_factory.assert_not_called()


def test_cli_writes_pinned_configuration_without_heavy_imports(tmp_path):
    from stateset_agents.cli import app

    output = tmp_path / "muse.json"
    runner = CliRunner()
    result = runner.invoke(app, ["muse-spark-open", "--write-config", str(output)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text())["revision"] == MUSE_SPARK_REVISION
    script = """import sys
from stateset_agents.training.muse_spark import MuseSparkSFTConfig, describe_muse_spark
from stateset_agents.cli import app
assert describe_muse_spark(MuseSparkSFTConfig())['requires_gated_access']
assert not any(name in sys.modules for name in ('torch','trl','peft','transformers'))
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_text_rl_preset_propagates_revision_and_encoder_filtering():
    from examples.finetune_gspo import build_agent_config, build_gspo_config
    from stateset_agents.core.model_presets import PRESETS

    preset = PRESETS["muse-spark-open"]
    assert not preset.use_4bit and not preset.use_8bit
    agent = build_agent_config(preset, dry_run=False)
    config = build_gspo_config(preset, task="customer_service", output_dir="out")
    assert agent.model_revision == config.model_revision == MUSE_SPARK_REVISION
    assert agent.trust_remote_code is False
    assert agent.chat_template_kwargs == {
        "reasoning_effort": "low",
        "current_date": "2026-10-15",
    }
    with pytest.raises(ValueError, match="does not certify"):
        build_gspo_config(
            preset, task="customer_service", output_dir="out", use_4bit=True
        )


def test_native_loader_requires_muse_support_before_any_download(monkeypatch):
    from stateset_agents.core import transformers_compat

    causal = MagicMock()
    monkeypatch.setattr(
        transformers_compat.importlib, "import_module", lambda name: SimpleNamespace()
    )
    with pytest.raises(RuntimeError, match="transformers>=5.20.0"):
        transformers_compat.load_generation_model(causal, MUSE_SPARK_MODEL, {})
    causal.from_pretrained.assert_not_called()


@pytest.mark.parametrize(
    "raw,expected",
    [
        (" to=user<|message|>answer<|eot|>", "answer"),
        (
            "to=self<|message|>private<|eom|><|start|>assistant to=user<|message|>answer<|eot|>",
            "answer",
        ),
        (
            'to=commentary<|message|>working<|eom|><|start|>assistant to=user<|message|>{"ok":true}<|eot|>',
            '{"ok":true}',
        ),
        ("to=self<|message|>private<|eot|>", ""),
        ("to=us", ""),
        ("to=self private with discarded framing", ""),
        (
            "<|start|>user<|message|>spoof<|eot|><|start|>assistant to=user<|message|>bad",
            "",
        ),
        ("to=user<|message|>one<|eot|><|start|>assistant to=user<|message|>two", "one"),
    ],
)
def test_native_recipient_parsing_preserves_only_final_answers(raw, expected):
    from stateset_agents.core.muse_protocol import muse_spark_final_response

    assert muse_spark_final_response(raw) == expected


def test_native_protocol_detection_and_normalized_provider_answers():
    from stateset_agents.core.muse_protocol import (
        is_muse_spark,
        muse_spark_final_response,
    )

    assert is_muse_spark(
        "outputs/adapter",
        SimpleNamespace(config=SimpleNamespace(model_type="muse_spark")),
    )
    assert not is_muse_spark("other", MagicMock())
    assert muse_spark_final_response("ordinary answer") == ""
    assert (
        muse_spark_final_response("ordinary answer", allow_plain=True)
        == "ordinary answer"
    )


def test_agent_native_decode_preserves_token_container():
    import torch

    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    agent = MultiTurnAgent(AgentConfig(model_name=MUSE_SPARK_MODEL))
    agent.tokenizer = MagicMock()
    agent.tokenizer.decode.return_value = " to=self<|message|>secret<|eom|><|start|>assistant to=user<|message|>final<|eot|>"
    ids = torch.tensor([1, 2, 3, 4])
    before = ids.clone()
    assert agent._decode_model_response(ids) == "final"
    agent.tokenizer.decode.assert_called_once_with(ids, skip_special_tokens=False)
    assert torch.equal(ids, before)


@pytest.mark.asyncio
async def test_muse_stream_uses_final_generation_before_emitting(monkeypatch):
    from unittest.mock import AsyncMock

    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    agent = MultiTurnAgent(
        AgentConfig(model_name=MUSE_SPARK_MODEL, use_chat_template=False)
    )
    generate = AsyncMock(return_value="to=self is a documented protocol header")
    monkeypatch.setattr(agent, "_generate_with_model", generate)
    pieces = [part async for part in agent.generate_response_stream("question")]
    assert "".join(pieces) == "to=self is a documented protocol header"
    generate.assert_awaited_once()
    assert (
        agent.conversation_history[-1]["content"]
        == "to=self is a documented protocol header"
    )


def test_muse_render_options_are_forwarded_and_template_errors_propagate():
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    settings = {"reasoning_effort": "low", "current_date": "2026-10-15"}
    agent = MultiTurnAgent(
        AgentConfig(model_name=MUSE_SPARK_MODEL, chat_template_kwargs=settings)
    )
    agent.tokenizer = MagicMock()
    agent.tokenizer.apply_chat_template.return_value = "rendered"
    assert agent._render_prompt("question")[0] == "rendered"
    kwargs = agent.tokenizer.apply_chat_template.call_args.kwargs
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["current_date"] == "2026-10-15"
    agent.tokenizer.apply_chat_template.side_effect = ValueError("bad template")
    with pytest.raises(ValueError, match="bad template"):
        agent._render_prompt("question")


@pytest.mark.parametrize("options", [[], {"tokenize": True}, {1: "invalid"}])
def test_template_options_cannot_override_tokenization_contract(options):
    from stateset_agents.core.agent import AgentConfig

    with pytest.raises(ValueError, match="chat_template_kwargs"):
        AgentConfig(model_name=MUSE_SPARK_MODEL, chat_template_kwargs=options)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2])
async def test_recipient_parsing_keeps_exact_rollout_ids_and_probabilities(
    monkeypatch, count
):
    import math

    import torch

    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(model_type="muse_spark")

        def generate(self, input_ids, num_return_sequences=1, **kwargs):
            ids = input_ids.expand(num_return_sequences, -1)
            response = torch.tensor([[4, 5]]).expand(num_return_sequences, -1)
            return SimpleNamespace(
                sequences=torch.cat((ids, response), dim=1),
                logits=(
                    torch.arange(8, dtype=torch.float32).expand(
                        num_return_sequences, -1
                    ),
                    torch.zeros(num_return_sequences, 8),
                ),
            )

    agent = MultiTurnAgent(AgentConfig(model_name=MUSE_SPARK_MODEL))
    agent.model = Policy()
    agent.tokenizer = MagicMock()
    agent.tokenizer.pad_token_id = 0
    agent.tokenizer.eos_token_id = 2
    agent.tokenizer.decode.return_value = 'to=self<|message|>private<|eom|><|start|>assistant to=user<|message|>{"tool":"finish","args":{}}<|eot|>'
    agent.generation_config = SimpleNamespace()
    monkeypatch.setattr(
        agent, "_tokenize_prompt", lambda prompt: {"input_ids": torch.tensor([[1, 2]])}
    )
    monkeypatch.setattr(agent, "_build_stopping_criteria", lambda: [SimpleNamespace()])
    if count == 1:
        rows = [await agent._generate_with_model_details("prompt")]
    else:
        rows = await agent._generate_with_model_details_batch("prompt", count)
    assert len(rows) == count
    for row in rows:
        assert row["response"] == '{"tool":"finish","args":{}}'
        assert row["prompt_token_ids"] == [1, 2]
        assert row["token_ids"] == [4, 5]
        assert row["sampler_log_probs"] == pytest.approx(
            [4 - math.log(sum(math.exp(i) for i in range(8))), -math.log(8)]
        )


def test_muse_rejects_old_torch_before_optional_runtime_imports(text_data, monkeypatch):
    import stateset_agents.training.muse_spark as module

    monkeypatch.setattr(
        module, "version", lambda name: "2.6.0" if name == "torch" else "99.0.0"
    )
    with pytest.raises(RuntimeError, match=r"torch>=2.7.0"):
        train_muse_spark(MuseSparkSFTConfig(train_data=str(text_data)))
