"""Exercise the published Inkling renderer and Tinker types without RPCs."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from stateset_agents.remote.job import RemoteJobSpec
from stateset_agents.remote.tinker import INKLING_2_SMALL, TinkerExecutor
from stateset_agents.remote.tinker_batches import build_inkling2_sft_batch


@pytest.fixture
def inkling_sdk():
    pytest.importorskip("torch")  # Load libtorch before the native renderer.
    sdk = pytest.importorskip("tinker")
    native = pytest.importorskip("tml_renderers")
    if not hasattr(native.renderers, "Inkling2"):
        pytest.skip("tml-renderers with Inkling2 support is required")
    return sdk, native.renderers.Inkling2()


@pytest.mark.parametrize("multi_turn", [False, True])
def test_real_renderer_masks_users_and_preserves_assistant_turns(
    inkling_sdk, multi_turn
):
    sdk, renderer = inkling_sdk
    messages = [
        {"role": "user", "content": "USER_ONLY_CACTUS"},
        {"role": "assistant", "content": "ASSISTANT_ONLY_ORCHID"},
    ]
    if multi_turn:
        messages.extend(
            [
                {"role": "user", "content": "USER_ONLY_JUNIPER"},
                {"role": "assistant", "content": "ASSISTANT_ONLY_VIOLET"},
            ]
        )
    batch = build_inkling2_sft_batch([{"messages": messages}])
    assert batch
    learned_text = ""
    for item in batch:
        datum = TinkerExecutor._datum(sdk, item)
        assert isinstance(datum, sdk.Datum)
        assert datum.loss_fn_inputs["target_tokens"].dtype == "int64"
        assert datum.loss_fn_inputs["weights"].dtype == "float32"
        assert datum.model_input.length == len(item["target_tokens"])
        assert item["input_ids"][1:] == item["target_tokens"][:-1]
        learned_text += renderer.tokenizer.decode(
            [
                t
                for t, w in zip(item["target_tokens"], item["weights"], strict=True)
                if w > 0
            ]
        )
    assert "ASSISTANT_ONLY_ORCHID" in learned_text
    assert "USER_ONLY_CACTUS" not in learned_text
    if multi_turn:
        assert "ASSISTANT_ONLY_VIOLET" in learned_text
        assert "USER_ONLY_JUNIPER" not in learned_text


def test_real_renderer_trains_tool_calls_and_masks_tool_results(inkling_sdk):
    sdk, renderer = inkling_sdk
    row = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "description": "Look up an order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Where is order 123?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_order",
                            "arguments": '{"order_id":"123"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "lookup_order",
                "content": "TOOL_ONLY_AWAITING_PICKUP",
            },
            {"role": "assistant", "content": "Your order is awaiting pickup."},
        ],
    }
    batch = build_inkling2_sft_batch([row])
    learned_text = ""
    for item in batch:
        TinkerExecutor._datum(sdk, item)
        learned_text += renderer.tokenizer.decode(
            [
                t
                for t, w in zip(item["target_tokens"], item["weights"], strict=True)
                if w > 0
            ]
        )
    assert "lookup_order" in learned_text
    assert "awaiting pickup" in learned_text
    assert "TOOL_ONLY_AWAITING_PICKUP" not in learned_text


@pytest.mark.parametrize("target", ["calls", "reply"])
def test_native_weighted_actions_keep_only_complete_targets(inkling_sdk, target):
    sdk, renderer = inkling_sdk
    calls = [
        {
            "id": key,
            "type": "function",
            "function": {"name": "lookup", "arguments": {"order": key}},
        }
        for key in ("ORDER_CACTUS", "ORDER_ORCHID")
    ]
    messages = [
        {"role": "user", "content": "Check two orders."},
        {
            "role": "assistant",
            "content": "Looking up both orders.",
            "tool_calls": calls,
            "weight": int(target == "calls"),
        },
    ]
    if target == "reply":
        messages.extend(
            [
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": "PRIVATE_RECEIPT",
                }
                for call in reversed(calls)
            ]
        )
        messages.append(
            {"role": "assistant", "content": "Both orders shipped.", "weight": 1}
        )
    row = {
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Find an order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order": {"type": "string"}},
                    },
                },
            }
        ],
    }
    data = build_inkling2_sft_batch([row])
    assert data

    def supervised(batch):
        return [
            token
            for datum in batch
            for token, weight in zip(
                datum["target_tokens"], datum["weights"], strict=True
            )
            if weight > 0
        ]

    learned = renderer.tokenizer.decode(supervised(data))
    for item in data:
        assert isinstance(TinkerExecutor._datum(sdk, item), sdk.Datum)
    assert "PRIVATE_RECEIPT" not in learned
    if target == "calls":
        assert "ORDER_CACTUS" in learned and "ORDER_ORCHID" in learned
    else:
        assert "Both orders shipped." in learned
        assert all(
            marker not in learned
            for marker in ("Looking up", "ORDER_CACTUS", "ORDER_ORCHID")
        )
        context = renderer.tokenizer.decode(data[-1]["input_ids"])
        assert "ORDER_CACTUS" in context and "ORDER_ORCHID" in context
    # A shifted weight at index i supervises the unshifted token at i + 1.
    end = max(i + 2 for item in data for i, w in enumerate(item["weights"]) if w > 0)
    assert supervised(build_inkling2_sft_batch([row], end)) == supervised(data)
    assert build_inkling2_sft_batch([row], end - 1) == []


def test_real_sdk_training_handoff_and_checkpoint_fetch(inkling_sdk, tmp_path):
    sdk, _ = inkling_sdk
    calls = []

    def forward_backward(batch, loss_fn):
        assert loss_fn == "cross_entropy"
        assert all(isinstance(datum, sdk.Datum) for datum in batch)
        calls.append("forward_backward")

    def optim_step(params):
        assert isinstance(params, sdk.AdamParams)
        calls.append("optim_step")

    training = SimpleNamespace(
        forward_backward=forward_backward,
        optim_step=optim_step,
        save_weights_for_sampler=lambda name: SimpleNamespace(path="tinker://sampler"),
        save_state=lambda name: SimpleNamespace(path="tinker://state"),
    )

    def create(base_model, rank):
        assert base_model == INKLING_2_SMALL
        assert rank == 16
        return training

    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    executor = TinkerExecutor(
        SimpleNamespace(create_lora_training_client=create), tinker_module=sdk
    )
    handle = executor.submit(
        RemoteJobSpec(
            dataset=dataset,
            base_model=INKLING_2_SMALL,
            output_dir=tmp_path / "checkpoint",
            num_epochs=1,
            gradient_accumulation_steps=1,
        )
    )
    result = executor.wait(handle, poll_interval_s=0.001)
    assert result.succeeded, result.logs
    assert calls == ["forward_backward", "optim_step"]
    pointer = json.loads((result.output_dir / "tinker_checkpoint.json").read_text())
    assert pointer["base_model"] == INKLING_2_SMALL
    assert pointer["sampler_uri"] == "tinker://sampler"
    assert pointer["state_uri"] == "tinker://state"
