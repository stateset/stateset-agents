"""Assistant spans end where the turn's own render ends, not at the first
place the reply text happens to occur.

``_reply_end`` searched for the reply from the turn's start, so a reply that
also occurs earlier in the turn — inside the reasoning ("18" in "9*2=18
dollars") or in template text ("think" in ``<think>``) — was cut short: the
rest of the reasoning, ``</think>``, the answer and the end-of-turn token went
unlabeled while the row still counted as fully supervised.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from stateset_agents.core.chat_masking import (  # noqa: E402
    assistant_token_mask,
    supervised_turn_sequences,
)
from tests.unit.test_loss_computation_real_tokenizer_masks import (  # noqa: E402
    CHATML,
    _byte_tokenizer,
)

#: Qwen3-like: reasoning rendered before the answer, an empty think block on
#: the newest turn only.
REASONING = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{% if m['role'] == 'assistant' %}"
    "{% if m.get('reasoning_content') %}<think>\n{{ m['reasoning_content'] }}"
    "\n</think>\n\n{% elif loop.last %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _supervised(tok, masked) -> str:
    ids, mask = masked
    return tok.decode([t for t, keep in zip(ids, mask, strict=True) if keep])


@pytest.mark.parametrize(
    "reply,reasoning",
    [
        ("18", "16-3-4=9 eggs; 9*2=18 dollars."),
        ("think", ""),
        ("in", ""),
        ("assistant", ""),
        ("user", "the user wants a user"),
    ],
)
def test_reply_that_also_occurs_earlier_in_the_turn_is_labeled_in_full(
    reply, reasoning
):
    tok = _byte_tokenizer(REASONING)
    message = {"role": "assistant", "content": reply}
    if reasoning:
        message["reasoning_content"] = reasoning
    masked = assistant_token_mask(tok, [{"role": "user", "content": "hello"}, message])
    assert masked is not None
    expected = (
        f"<think>\n{reasoning}\n</think>\n\n"
        if reasoning
        else "<think>\n\n</think>\n\n"
    ) + f"{reply}<|im_end|>\n"
    assert _supervised(tok, masked) == expected


def test_history_turn_is_located_by_its_own_content_not_a_repeat():
    """An earlier turn whose reply repeats inside its reasoning ends after
    the reply, not inside the reasoning."""
    tok = _byte_tokenizer(REASONING)
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "18", "reasoning_content": "9*2=18"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "ok"},
    ]
    supervised = _supervised(tok, assistant_token_mask(tok, messages))
    assert supervised.startswith("<think>\n9*2=18\n</think>\n\n18<|im_end|>\n")
    assert "q2" not in supervised


def test_weight_zero_turns_are_not_marked_and_need_not_be_located():
    tok = _byte_tokenizer(CHATML)
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "bad", "weight": 0},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "good"},
    ]
    assert _supervised(tok, assistant_token_mask(tok, messages)) == "good<|im_end|>\n"
    mode, sequences = supervised_turn_sequences(tok, messages)
    assert mode == "all_assistant_turns" and len(sequences) == 1


def test_unstable_turns_split_into_sequences_rendered_as_generated():
    tok = _byte_tokenizer(REASONING)
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    mode, sequences = supervised_turn_sequences(tok, messages)
    assert mode == "split_per_turn"
    assert [_supervised(tok, s) for s in sequences] == [
        "<think>\n\n</think>\n\na1<|im_end|>\n",
        "<think>\n\n</think>\n\na2<|im_end|>\n",
    ]
    assert assistant_token_mask(tok, messages, require_stable=True) is None
