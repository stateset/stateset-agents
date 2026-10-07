"""Tool schemas and specialized small-model training contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateset_agents.core.model_presets import PRESETS
from stateset_agents.core.transformers_compat import generation_compat_kwargs
from stateset_agents.training.sft import _render_sft_row, load_chat_dataset


def test_tool_schemas_and_typed_arguments_survive_rendering():
    """Different tools must not acquire each other's properties or null fields."""
    pytest.importorskip("transformers")
    Dataset = pytest.importorskip("datasets").Dataset

    from tests._tiny_tokenizer import tiny_tokenizer

    path = (
        Path(__file__).resolve().parents[2]
        / "examples/data/functiongemma_commerce.jsonl"
    )
    rows = load_chat_dataset(path)
    tokenizer = tiny_tokenizer()
    tokenizer.chat_template = (
        '{"tools": {{ tools | tojson }}, "messages": {{ messages | tojson }}}'
    )
    dataset = Dataset.from_list([_render_sft_row(tokenizer, row) for row in rows])
    assert dataset.column_names == ["text"]
    assert [json.loads(row["text"]) for row in dataset] == rows
    assert (
        rows[1]["messages"][-1]["tool_calls"][0]["function"]["arguments"][
            "warehouse_id"
        ]
        == 3
    )


@pytest.mark.parametrize("tools", ["bad schema", {}, [None]])
def test_invalid_tool_schemas_fail_before_template_rendering(tools):
    with pytest.raises(ValueError, match="list of tool schema objects"):
        _render_sft_row(None, {"messages": [], "tools": tools})


def test_functiongemma_is_explicitly_sft_only():
    from examples.finetune_gspo import build_gspo_config

    preset = PRESETS["functiongemma-270m"]
    assert preset.trust_remote_code is False
    assert preset.cli_command is None
    with pytest.raises(ValueError, match="SFT only"):
        build_gspo_config(preset, task="customer_service", output_dir="unused")


@pytest.mark.parametrize(
    "model_type,layers,expected",
    [
        (
            "granitemoehybrid",
            ["full_attention", "full_attention"],
            {"use_cache": False},
        ),
        ("granitemoehybrid", ["attention", "attention"], {"use_cache": False}),
        ("granitemoehybrid", ["full_attention", "linear_attention"], {}),
        ("llama", ["full_attention"], {}),
        ("granitemoehybrid", [], {}),
    ],
)
def test_cache_workaround_only_applies_to_dense_granite(model_type, layers, expected):
    model = SimpleNamespace(
        config=SimpleNamespace(model_type=model_type, layer_types=layers)
    )
    assert generation_compat_kwargs(model) == expected


def test_deepseek_memory_profile_keeps_reasoning_budget():
    from stateset_agents.training.deepseek_r1_small_starter import (
        get_deepseek_r1_small_config,
    )

    config = get_deepseek_r1_small_config(starter_profile="memory")
    assert config.max_completion_length == config.max_new_tokens == 2048
    assert any("reasoning" in warning for warning in config.validate())
