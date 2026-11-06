"""Offline contracts for native Inkling 2 rendering and model discovery."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from stateset_agents.remote.model_catalog import get_model_support
from stateset_agents.remote.model_evidence import model_provider_evidence
from stateset_agents.remote.tinker import INKLING_2_SMALL
from stateset_agents.remote.tinker_batches import build_inkling2_sft_batch


@pytest.fixture
def native_renderer(monkeypatch):
    calls = []
    examples = [([10, 20, 30, 40, 50], [0.0, 0.0, 1.0, 1.0, 1.0])]

    def convert(messages, tools=None):
        calls.append((messages, tools))
        return messages

    def adapt(example):
        ids, weights = example
        return SimpleNamespace(to_ints=lambda: ids), weights

    module = ModuleType("tml_renderers")
    module.chat = SimpleNamespace(
        OpenAIMessage=SimpleNamespace(from_openai_chat_dicts=convert)
    )
    module.renderers = SimpleNamespace(
        Inkling2=lambda: SimpleNamespace(render_for_sft=lambda messages: examples)
    )
    adapter = ModuleType("tml_renderers.tinker")
    adapter.training_example_to_tinker_model_input_and_weights = adapt
    monkeypatch.setitem(sys.modules, "tml_renderers", module)
    monkeypatch.setitem(sys.modules, "tml_renderers.tinker", adapter)
    return calls, examples


def test_native_loss_mask_is_shifted_once_and_truncated(native_renderer):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    batch = build_inkling2_sft_batch([{"messages": messages}], max_length=4)
    assert batch == [
        {
            "input_ids": [10, 20, 30],
            "target_tokens": [20, 30, 40],
            "weights": [0.0, 1.0, 1.0],
        }
    ]
    assert native_renderer[0] == [(messages, None)]


def test_native_converter_receives_tools_and_tool_calls(native_renderer):
    messages = [{"role": "assistant", "content": None, "tool_calls": []}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    build_inkling2_sft_batch([{"messages": messages, "tools": tools}])
    assert native_renderer[0] == [(messages, tools)]


def test_multiple_rendered_examples_and_zero_loss(native_renderer):
    native_renderer[1].extend([([1, 2, 3], [0.0, 1.0, 1.0]), ([1, 2], [0.0, 0.0])])
    rows = [{"messages": [{"role": "assistant", "content": "ok"}]}]
    assert len(build_inkling2_sft_batch(rows)) == 2
    assert build_inkling2_sft_batch(rows, max_length=2) == [
        {"input_ids": [1], "target_tokens": [2], "weights": [1.0]}
    ]


def test_renderer_alignment_is_checked(native_renderer):
    native_renderer[1][:] = [([1, 2], [1.0])]
    with pytest.raises(ValueError, match="misaligned"):
        build_inkling2_sft_batch(
            [{"messages": [{"role": "assistant", "content": "ok"}]}]
        )


def test_media_is_rejected_explicitly(native_renderer):
    with pytest.raises(ValueError, match="media inputs are unsupported"):
        build_inkling2_sft_batch(
            [{"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}]
        )


def test_missing_optional_renderer_has_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "tml_renderers", None)
    with pytest.raises(RuntimeError, match=r"stateset-agents\[tinker\]"):
        build_inkling2_sft_batch([])


def test_python310_has_actionable_error(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(sys, "version_info", (3, 10))
        with pytest.raises(RuntimeError, match="Python 3.11"):
            build_inkling2_sft_batch([])


def test_inkling2_catalog_keeps_live_verification_pending():
    support = get_model_support(INKLING_2_SMALL.upper())
    assert support is not None
    assert support.model == INKLING_2_SMALL
    assert support.recommended_provider == "tinker"
    assert support.tier == "frontier-preview"
    assert support.certification == "configured"
    assert support.runpod is None
    evidence = [r for r in model_provider_evidence() if r["model"] == INKLING_2_SMALL]
    assert len(evidence) == 1
    assert evidence[0]["level"] == "unit-tested"
