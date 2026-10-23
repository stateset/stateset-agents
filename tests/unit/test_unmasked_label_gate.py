"""The sequence path's full-stream fallback needs ``allow_unmasked_labels``
whenever it would train on user turns after the first reply.

When assistant spans cannot be located (a chat template that reorders or
rewrites messages, a tokenizer without offsets), the sequence path
supervised the whole token stream. The offset-mapping path already required
``allow_unmasked_labels`` for that, but the template-spans-not-found path did
not: a multi-turn conversation then trained the policy to produce the user's
follow-up turns, silently.

Gating every full-stream fallback would break stub-agent training (the stub
tokenizer has neither a chat template nor offsets, so every stub sequence
row takes this path; ``tests/integration/test_stub_training_loop.py`` and the
GRPO pipeline tests run it). So the gate covers exactly the harmful case: a
conversation with a user (or any non-assistant) turn after the first
assistant reply raises ``RolloutContractError`` unless opted in. A prompt
followed by one reply keeps the documented fallback (the prompt is
supervised too, with the existing warning).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from stateset_agents.core.agent_backends import StubTokenizer  # noqa: E402
from stateset_agents.core.trajectory import ConversationTurn  # noqa: E402
from stateset_agents.training.loss_computation import (  # noqa: E402
    _prepare_inputs_and_labels,
)
from stateset_agents.training.rollout_contract import (  # noqa: E402
    RolloutContractError,
)
from tests._tool_use_helpers import make_tokenizer  # noqa: E402

#: Renders messages in reverse, so no assistant span can be located.
REVERSING = (
    "{% for m in messages|reverse %}<|im_start|>{{ m['role'] }}\n"
    "{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _trajectory(*roles_and_text: tuple[str, str]) -> Any:
    return SimpleNamespace(
        turns=[ConversationTurn(role=r, content=t) for r, t in roles_and_text]
    )


MULTI_TURN = _trajectory(
    ("user", "first question"),
    ("assistant", "first answer"),
    ("user", "follow up please"),
    ("assistant", "second answer"),
)
SINGLE_REPLY = _trajectory(("user", "first question"), ("assistant", "first answer"))


def _labels(tokenizer: Any, trajectory: Any, **config: Any) -> tuple[Any, Any]:
    agent = SimpleNamespace(
        tokenizer=tokenizer, model=SimpleNamespace(device=torch.device("cpu"))
    )
    settings = SimpleNamespace(
        max_prompt_length=1024, max_completion_length=512, **config
    )
    return _prepare_inputs_and_labels(trajectory, agent, settings)


def _chat_tokenizer() -> Any:
    tokenizer = make_tokenizer()
    tokenizer.chat_template = REVERSING
    return tokenizer


@pytest.mark.parametrize(
    "tokenizer", [_chat_tokenizer, StubTokenizer], ids=["template", "no_offsets"]
)
def test_later_user_turns_need_the_opt_in(tokenizer: Any) -> None:
    with pytest.raises(RolloutContractError, match="allow_unmasked_labels"):
        _labels(tokenizer(), MULTI_TURN)


@pytest.mark.parametrize(
    "tokenizer", [_chat_tokenizer, StubTokenizer], ids=["template", "no_offsets"]
)
def test_opting_in_supervises_the_full_stream(tokenizer: Any) -> None:
    inputs, labels = _labels(tokenizer(), MULTI_TURN, allow_unmasked_labels=True)
    attended = inputs["attention_mask"][0].bool()
    assert bool(labels[0][attended].ne(-100).all())


@pytest.mark.parametrize(
    "tokenizer", [_chat_tokenizer, StubTokenizer], ids=["template", "no_offsets"]
)
def test_a_single_reply_keeps_the_documented_fallback(tokenizer: Any) -> None:
    """Stub-agent training (no template, no offsets) relies on this."""
    _inputs, labels = _labels(tokenizer(), SINGLE_REPLY)
    assert bool(labels.ne(-100).any())
