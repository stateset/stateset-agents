"""Generation prompts and exact-token sampling for the on-policy trainers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.training.preference.encoding import (  # noqa: E402
    PreferenceEncoder,
)
from stateset_agents.training.preference.sampling import (  # noqa: E402
    sample_completions,
    stop_token_ids,
)
from tests.learning._toy import make_model, make_tokenizer  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}<eos>{% endfor %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    return make_tokenizer()


@pytest.mark.parametrize(
    "prompt", ["raw prompt: ", [{"role": "user", "content": "plain prompt"}]]
)
def test_generation_prompt_is_the_encoded_prompt(tokenizer: Any, prompt: Any) -> None:
    encoder = PreferenceEncoder(tokenizer, max_length=64)
    assert encoder.encode_prompt(prompt) == encoder.encode(prompt, "12").prompt_ids


def test_chat_template_generation_prompt(tokenizer: Any) -> None:
    chat = make_tokenizer()
    chat.chat_template = CHAT_TEMPLATE
    encoder = PreferenceEncoder(chat, max_length=128)
    messages = [{"role": "user", "content": "hi"}]
    prompt_ids = encoder.encode_prompt(messages)
    text = chat.decode(list(prompt_ids))
    assert text.startswith("<user>hi") and text.endswith("<assistant>")


def test_generation_prompt_keeps_the_end_when_truncated(tokenizer: Any) -> None:
    encoder = PreferenceEncoder(tokenizer, max_length=16, max_prompt_length=5)
    ids = encoder.encode_prompt("a long raw prompt that overflows")
    assert len(ids) == 5 and tokenizer.decode(list(ids)) == "flows"


def test_stop_ids_merge_tokenizer_and_generation_config(tokenizer: Any) -> None:
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[5, 50256]),
        config=SimpleNamespace(vocab_size=103),
    )
    assert stop_token_ids(model, tokenizer) == sorted({tokenizer.eos_token_id, 5})


def _sample(model: Any, tokenizer: Any, **overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "num_samples": 6,
        "max_new_tokens": 5,
        "temperature": 1.0,
        "top_p": 1.0,
        "pad_token_id": tokenizer.pad_token_id,
        "stop_ids": [tokenizer.eos_token_id],
        "batch_size": 4,
    }
    settings.update(overrides)
    return sample_completions(model, [(10, 11, 12), (13,)], **settings)


def test_sampling_is_reproducible_and_bounded(tokenizer: Any) -> None:
    model = make_model(0, n_embd=16)
    torch.manual_seed(1)
    samples = _sample(model, tokenizer)
    torch.manual_seed(1)
    assert _sample(model, tokenizer) == samples  # torch's global RNG
    assert [len(group) for group in samples] == [6, 6]
    assert all(1 <= len(c) <= 5 for group in samples for c in group)
    with pytest.raises(ValueError, match="num_samples"):
        _sample(model, tokenizer, num_samples=0)


def test_completions_end_at_the_first_stop_token(tokenizer: Any) -> None:
    model = make_model(0, n_embd=16)
    eos = tokenizer.eos_token_id
    with torch.no_grad():  # every position now strongly predicts EOS
        model.transformer.ln_f.weight.zero_()
        model.transformer.ln_f.bias.fill_(1.0)
        model.transformer.wte.weight[eos] = 5.0
    samples = _sample(model, tokenizer)
    assert all(c == (eos,) for group in samples for c in group)
    # Without EOS as a stop token the completion runs to max_new_tokens.
    longer = _sample(model, tokenizer, stop_ids=[])
    assert all(c == (eos,) * 5 for group in longer for c in group)
