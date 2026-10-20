"""Opt-in contracts against pinned NVIDIA processors; never loads model weights.

Run with STATESET_TEST_HF_PROCESSORS=1 and the nemotron-vl extra installed.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("STATESET_TEST_HF_PROCESSORS") != "1",
        reason="opt-in processor/tokenizer downloads; no model weights",
    ),
]


@pytest.fixture(scope="module", params=["super-vl", "bolt"])
def processor(request):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    from stateset_agents.training.nemotron_vl import (
        NEMOTRON_VL_MODELS,
        NEMOTRON_VL_REVISIONS,
    )

    return transformers.AutoProcessor.from_pretrained(
        NEMOTRON_VL_MODELS[request.param],
        revision=NEMOTRON_VL_REVISIONS[request.param],
        trust_remote_code=True,
    )


@pytest.mark.parametrize("layout", ["single", "multiturn", "multiimage"])
def test_pinned_processor_preserves_pixels_and_assistant_targets(processor, layout):
    import torch
    from PIL import Image

    from stateset_agents.data.nemotron_vl import NemotronVLCollator

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (512, 512), "white")},
                {"type": "text", "text": "Identify the color."},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "White."}]},
    ]
    if layout == "multiimage":
        messages[0]["content"].insert(
            1,
            {
                "type": "image",
                "image": Image.new("RGB", (1024, 512), "white"),
            },
        )
    elif layout == "multiturn":
        messages.extend(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": Image.new("RGB", (512, 256), "black"),
                        },
                        {"type": "text", "text": "Identify the other color."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "Black."}]},
            ]
        )
    batch = NemotronVLCollator(processor, max_length=8192)([{"conversation": messages}])
    labels = batch["labels"][0]
    answer = processor.tokenizer.decode(labels[labels != -100].tolist())
    assert "White." in answer
    assert "Identify" not in answer
    if layout == "multiturn":
        assert "Black." in answer
    assert not torch.any(labels == processor.image_token_id)
    assert batch["pixel_values"].numel() > 0
    assert batch["pixel_values"].dtype == torch.bfloat16
    assert batch["input_ids"].shape == batch["labels"].shape


def test_river_text_renderer_matches_the_checkpoint_template(processor):
    pytest.importorskip("river_client")
    from stateset_agents.remote.river_renderers import get_river_renderer

    model = processor.tokenizer.name_or_path
    renderer = get_river_renderer(model, thinking=False)
    messages = [{"role": "user", "content": "Reply with OK."}]
    expected = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        truncate_history_thinking=True,
    )
    assert renderer.build_sample_prompt(messages).prompt == expected
    from river_client.client import _resolve_model_input

    ids = renderer.tokenizer.encode(expected, add_special_tokens=False)
    resolved = _resolve_model_input(
        [[{"type": "text", "tokens": ids}]],
        prompts=None,
        prompt_token_ids=None,
        images=None,
        tokenizer=renderer.tokenizer,
    )
    assert resolved == (None, [ids], None)
    assert ids == processor.tokenizer.encode(expected, add_special_tokens=False)
    assert renderer.tokenizer.get_vocab() == processor.tokenizer.get_vocab()
