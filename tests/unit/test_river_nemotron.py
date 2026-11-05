"""Pinned text renderer routing is independent of River account model access."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from stateset_agents.remote.river_renderers import get_river_renderer
from stateset_agents.training.nemotron_vl import (
    NEMOTRON_VL_MODELS,
    NEMOTRON_VL_REVISIONS,
)

pytest.importorskip("river_client", reason="install the optional River SDK")


@pytest.mark.parametrize(
    "family,model",
    [
        ("bolt", NEMOTRON_VL_MODELS["bolt"]),
        ("bolt", "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"),
        ("super-vl", NEMOTRON_VL_MODELS["super-vl"]),
        ("super-vl", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"),
    ],
)
def test_new_nemotron_models_use_their_own_pinned_tokenizer(monkeypatch, family, model):
    from river_client.renderers.nemotron import Nemotron35Renderer
    from transformers import AutoTokenizer

    loader = MagicMock(return_value=MagicMock())
    loader.return_value.convert_tokens_to_ids.return_value = 18
    loader.return_value.unk_token_id = 0
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", loader)
    renderer = get_river_renderer(model, thinking=False)
    assert isinstance(renderer, Nemotron35Renderer)
    loader.assert_called_once_with(
        NEMOTRON_VL_MODELS[family],
        revision=NEMOTRON_VL_REVISIONS[family],
        trust_remote_code=True,
    )
    assert renderer.tokenizer._tokenizer is loader.return_value
    assert renderer.get_stop_strings() == ["<|im_end|>", "</s>"]


def test_existing_sdk_renderer_routing_is_preserved(monkeypatch):
    import river_client.renderers

    delegate = MagicMock()
    monkeypatch.setattr(river_client.renderers, "get_renderer", delegate)
    assert get_river_renderer("Qwen/Qwen3.5-9B") is delegate.return_value
    delegate.assert_called_once_with("Qwen/Qwen3.5-9B", thinking=False)


def test_sdk_text_chunks_lower_without_changing_token_ids():
    from river_client.client import _resolve_model_input

    from stateset_agents.remote.river_renderers import _NemotronRiverTokenizer

    tokenizer = MagicMock()
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.side_effect = lambda token: (
        18 if token == "<image>" else 0
    )
    adapted = _NemotronRiverTokenizer(tokenizer)
    chunks = [[{"type": "text", "tokens": [3, 4, 5]}]]
    with pytest.raises(ValueError, match="tokenizer has none"):
        _resolve_model_input(
            chunks,
            prompts=None,
            prompt_token_ids=None,
            images=None,
            tokenizer=tokenizer,
        )
    assert _resolve_model_input(
        chunks, prompts=None, prompt_token_ids=None, images=None, tokenizer=adapted
    ) == (None, [[3, 4, 5]], None)
    assert adapted.convert_tokens_to_ids("ordinary text") == 0
    assert adapted.encode("text") is tokenizer.encode.return_value
    assert adapted.decode([3]) is tokenizer.decode.return_value


def test_invalid_nemotron_image_metadata_fails_closed():
    from stateset_agents.remote.river_renderers import _NemotronRiverTokenizer

    tokenizer = MagicMock()
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.return_value = 0
    with pytest.raises(ValueError, match="valid <image>"):
        _NemotronRiverTokenizer(tokenizer)


@pytest.mark.parametrize("operation", ["_rendered_prompts", "_rl_sample_groups"])
def test_known_nemotron_renderer_failure_never_falls_back_to_raw_text(
    monkeypatch, operation
):
    import stateset_agents.remote.river as backend

    monkeypatch.setattr(
        "stateset_agents.remote.river_renderers.get_river_renderer",
        MagicMock(side_effect=ValueError("tokenizer unavailable")),
    )
    if operation == "_rendered_prompts":
        args = (NEMOTRON_VL_MODELS["bolt"], ["Test prompt"])
        kwargs = {}
    else:
        args = (MagicMock(), NEMOTRON_VL_MODELS["bolt"], ["Test prompt"])
        kwargs = {"num_samples": 2, "max_tokens": 32, "temperature": 1.0, "top_p": 1.0}
    with pytest.raises(ValueError, match="tokenizer unavailable"):
        getattr(backend, operation)(*args, **kwargs)
