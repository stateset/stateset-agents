"""Assistant-only loss masking against a real (offline-built) HF fast tokenizer.

Mocked tokenizers previously hid two defects: real ``apply_chat_template``
returns a ``BatchEncoding`` (not a ``dict``) whose mask key is
``assistant_masks``, so the sequence-path loss silently supervised prompt tokens.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from stateset_agents.training.loss_computation import (  # noqa: E402
    _prepare_inputs_and_labels,
)
from stateset_agents.training.rollout_contract import (  # noqa: E402
    RolloutContractError,
)

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
CHATML_GENERATION = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{% if m['role'] == 'assistant' %}{% generation %}"
    "{{ m['content'] }}<|im_end|>{% endgeneration %}\n"
    "{% else %}{{ m['content'] }}<|im_end|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _byte_tokenizer(chat_template: str | None):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    for special in ("<|im_start|>", "<|im_end|>", "<pad>"):
        vocab[special] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tok.chat_template = chat_template
    return tok


def _supervised_text(
    chat_template: str | None, turns: list[dict[str, str]], **settings: object
) -> str:
    tok = _byte_tokenizer(chat_template)
    agent = SimpleNamespace(
        tokenizer=tok, model=SimpleNamespace(device=torch.device("cpu"))
    )
    config = SimpleNamespace(
        max_prompt_length=512, max_completion_length=512, **settings
    )
    inputs, labels = _prepare_inputs_and_labels(
        SimpleNamespace(turns=turns), agent, config
    )
    ids = inputs["input_ids"][0]
    assert labels.shape == inputs["input_ids"].shape
    kept = ids[labels[0] != -100].tolist()
    return tok.decode(kept)


TURNS = [
    {"role": "system", "content": "Be terse."},
    {"role": "user", "content": "Where is my order?"},
    {"role": "assistant", "content": "It ships today."},
    {"role": "user", "content": "Thanks"},
    {"role": "assistant", "content": "Welcome!"},
]


@pytest.mark.parametrize("template", [CHATML, CHATML_GENERATION])
def test_only_assistant_tokens_are_supervised(template):
    supervised = _supervised_text(template, TURNS)
    assert "Where is my order" not in supervised
    assert "Be terse" not in supervised
    assert "Thanks" not in supervised
    assert "It ships today." in supervised
    assert "Welcome!" in supervised
    # The end-of-turn marker is part of what the policy must learn to emit.
    assert supervised.count("<|im_end|>") == 2


def test_template_without_prefix_property_falls_back_to_full_stream():
    # Reverses history, so rendering a prefix is not a prefix of the full render.
    reversing = (
        "{% for m in messages|reverse %}<|im_start|>{{ m['role'] }}\n"
        "{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    # TURNS has a user turn after the first reply: the full stream would train
    # the policy on it, so it needs the explicit opt-in (test_unmasked_label_gate).
    with pytest.raises(RolloutContractError, match="allow_unmasked_labels"):
        _supervised_text(reversing, TURNS)
    supervised = _supervised_text(reversing, TURNS, allow_unmasked_labels=True)
    # Never all -100 (that would make the loss NaN): full stream is supervised.
    assert "Where is my order" in supervised
    assert "Welcome!" in supervised


def test_tokenizer_without_chat_template_uses_role_spans():
    # HF tokenizers always expose apply_chat_template; with no template set it
    # raises, which previously crashed the loss instead of falling back.
    supervised = _supervised_text(None, TURNS)
    assert "Where is my order" not in supervised
    assert "It ships today." in supervised
    assert "Welcome!" in supervised


@pytest.mark.parametrize("template", [CHATML, CHATML_GENERATION])
def test_truncated_away_reply_supervises_nothing(template):
    """If truncation drops every assistant token, nothing is supervised -- the
    full-stream fallback would train on the (long) user turn instead."""
    tok = _byte_tokenizer(template)
    agent = SimpleNamespace(
        tokenizer=tok, model=SimpleNamespace(device=torch.device("cpu"))
    )
    config = SimpleNamespace(max_prompt_length=20, max_completion_length=20)
    turns = [
        {"role": "user", "content": "x" * 200},
        {"role": "assistant", "content": "Done."},
    ]
    _, labels = _prepare_inputs_and_labels(SimpleNamespace(turns=turns), agent, config)
    assert labels.shape[1] == 40
    assert not labels.ne(-100).any()


def test_group_loss_skips_rows_without_supervised_tokens(monkeypatch):
    from stateset_agents.training import loss_computation as lc

    forwards = []

    class Model:
        device = torch.device("cpu")

        def __call__(self, input_ids, labels=None, **kwargs):
            forwards.append(labels.clone())
            logits = torch.zeros(1, input_ids.shape[1], 4, requires_grad=True)
            return SimpleNamespace(loss=logits.sum() + 1.0, logits=logits)

    rows = iter(
        [
            (
                {"input_ids": torch.ones(1, 3, dtype=torch.long)},
                torch.full((1, 3), -100),
            ),
            (
                {"input_ids": torch.ones(1, 3, dtype=torch.long)},
                torch.tensor([[-100, 5, 6]]),
            ),
        ]
    )
    monkeypatch.setattr(lc, "_prepare_inputs_and_labels", lambda *a: next(rows))
    group = SimpleNamespace(trajectories=[SimpleNamespace(), SimpleNamespace()])
    config = SimpleNamespace(clip_ratio=0.0)
    loss, _ = lc._compute_group_policy_loss(
        group, torch.tensor([1.0, 1.0]), config, SimpleNamespace(model=Model())
    )
    assert len(forwards) == 1
    assert loss.item() == pytest.approx(1.0)


def test_entropy_bonus_uses_the_positions_that_predict_supervised_tokens(monkeypatch):
    from stateset_agents.training import loss_computation as lc

    seen = {}

    def spy(logits, mask):
        seen["shape"], seen["mask"] = logits.shape, mask.tolist()
        return logits.sum() * 0

    monkeypatch.setattr(lc, "compute_entropy_bonus", spy)

    class Model:
        device = torch.device("cpu")

        def __call__(self, input_ids, labels=None, **kwargs):
            logits = torch.zeros(1, 4, 3, requires_grad=True)
            return SimpleNamespace(loss=logits.sum() + 1.0, logits=logits)

    labels = torch.tensor([[-100, -100, 7, 8]])
    monkeypatch.setattr(
        lc,
        "_prepare_inputs_and_labels",
        lambda *a: ({"input_ids": torch.ones(1, 4, dtype=torch.long)}, labels),
    )
    group = SimpleNamespace(trajectories=[SimpleNamespace()])
    lc._compute_group_policy_loss(
        group,
        torch.tensor([1.0]),
        SimpleNamespace(clip_ratio=0.0),
        SimpleNamespace(model=Model()),
        entropy_coef=0.1,
    )
    # Logits at positions 1 and 2 predict the supervised tokens 2 and 3.
    assert seen["shape"] == (1, 3, 3)
    assert seen["mask"] == [[0.0, 1.0, 1.0]]
