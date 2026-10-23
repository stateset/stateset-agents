"""Tokenize preference contexts and responses with exact response masks.

Every method scores only the response tokens. A sequence is stored as its
prompt ids followed by its response ids, so the response mask is exact and a
prompt can be re-paired with another response (KTO's KL estimate) without
re-tokenizing.

Three renderings, chosen by ``prompt_format``:

``chat_template``
    The tokenizer's chat template. The response is the final assistant turn,
    located with :func:`stateset_agents.core.chat_masking.final_turn_token_mask`
    (assistant-only labels, including the template's end-of-turn tokens).
``plain``
    ``MultiTurnAgent``'s text format for models without a chat template
    (``"System: ...\\nUser: ...\\nAssistant: "``) followed by the response and
    the EOS token. Only system/user/assistant turns render.
``raw``
    A raw prompt string followed directly by the response and EOS (TRL's
    "standard" dataset format, e.g. pre-formatted prompts).

``auto`` (the default) renders message prompts with the chat template when the
tokenizer has one (``plain`` otherwise) and raw prompt strings as ``raw``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from stateset_agents.core.chat_masking import final_turn_token_mask

from .data import Message

__all__ = [
    "PROMPT_FORMATS",
    "EncodedSequence",
    "PreferenceEncoder",
    "render_plain_prompt",
]

PROMPT_FORMATS = ("auto", "chat_template", "plain", "raw")
_PLAIN_ROLES = {"system": "System", "user": "User", "assistant": "Assistant"}


@dataclass(frozen=True)
class EncodedSequence:
    """Prompt token ids followed by response token ids."""

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.prompt_ids) + len(self.response_ids)

    @property
    def input_ids(self) -> list[int]:
        return [*self.prompt_ids, *self.response_ids]

    @property
    def response_mask(self) -> list[int]:
        return [0] * len(self.prompt_ids) + [1] * len(self.response_ids)

    def with_response(self, other: EncodedSequence) -> EncodedSequence:
        """This prompt followed by ``other``'s response (KTO's KL pairs)."""
        return EncodedSequence(self.prompt_ids, other.response_ids, False)


def render_plain_prompt(messages: Sequence[Mapping[str, Any]]) -> str:
    """``MultiTurnAgent``'s prompt for models without a chat template."""
    parts = []
    for message in messages:
        role = message.get("role")
        if role not in _PLAIN_ROLES:
            raise ValueError(
                f"the plain prompt format renders system/user/assistant turns "
                f"only, not {role!r}; use a tokenizer with a chat template"
            )
        parts.append(f"{_PLAIN_ROLES[str(role)]}: {message.get('content', '')}\n")
    return "".join(parts) + "Assistant: "


def _as_messages(prompt: str | Sequence[Message]) -> list[Message]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [dict(message) for message in prompt]


def _assistant(response: str | Message) -> Message:
    if isinstance(response, str):
        return {"role": "assistant", "content": response}
    return dict(response)


def _response_text(response: str | Message) -> str:
    if isinstance(response, str):
        return response
    if response.get("tool_calls"):
        raise ValueError("tool-call responses need the chat_template prompt format")
    content = response.get("content")
    return content if isinstance(content, str) else ""


def _positive_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class PreferenceEncoder:
    """Turn ``(prompt, response)`` into an :class:`EncodedSequence`.

    ``max_prompt_length`` keeps the *end* of an overlong prompt; a sequence
    still longer than ``max_length`` keeps the end of the prompt (leaving room
    for at least one response token) and the start of the response.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        prompt_format: str = "auto",
        max_length: int = 1024,
        max_prompt_length: int | None = None,
        append_eos: bool = True,
    ) -> None:
        if prompt_format not in PROMPT_FORMATS:
            raise ValueError(
                f"prompt_format must be one of {PROMPT_FORMATS}, got {prompt_format!r}"
            )
        self.tokenizer = tokenizer
        self.prompt_format = prompt_format
        self.max_length = _positive_int(max_length, "max_length", minimum=2)
        if max_prompt_length is not None:
            _positive_int(max_prompt_length, "max_prompt_length")
            if max_prompt_length >= max_length:
                raise ValueError("max_prompt_length must be smaller than max_length")
        self.max_prompt_length = max_prompt_length
        self.append_eos = bool(append_eos)
        if prompt_format == "chat_template" and not self.has_chat_template:
            raise ValueError("prompt_format='chat_template' needs a chat template")

    @property
    def has_chat_template(self) -> bool:
        return bool(getattr(self.tokenizer, "chat_template", None)) and callable(
            getattr(self.tokenizer, "apply_chat_template", None)
        )

    @property
    def pad_token_id(self) -> int:
        for name in ("pad_token_id", "eos_token_id"):
            value = getattr(self.tokenizer, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return 0

    def settings(self) -> dict[str, Any]:
        """Encoder settings to persist with a checkpoint."""
        return {
            "prompt_format": self.prompt_format,
            "max_length": self.max_length,
            "max_prompt_length": self.max_prompt_length,
            "append_eos": self.append_eos,
        }

    def resolve_format(self, prompt: str | Sequence[Message]) -> str:
        """The rendering used for ``prompt``."""
        if self.prompt_format != "auto":
            if self.prompt_format == "raw" and not isinstance(prompt, str):
                raise ValueError("prompt_format='raw' needs raw prompt strings")
            return self.prompt_format
        if isinstance(prompt, str):
            return "raw"
        return "chat_template" if self.has_chat_template else "plain"

    def encode(
        self, prompt: str | Sequence[Message], response: str | Message
    ) -> EncodedSequence:
        rendering = self.resolve_format(prompt)
        if rendering == "chat_template":
            prompt_ids, response_ids = self._encode_chat(prompt, response)
        else:
            text = (
                prompt
                if rendering == "raw" and isinstance(prompt, str)
                else render_plain_prompt(_as_messages(prompt))
            )
            prompt_ids = self._tokenize(text, special=True)
            response_ids = self._tokenize(_response_text(response), special=False)
            eos = getattr(self.tokenizer, "eos_token_id", None)
            if self.append_eos and isinstance(eos, int) and not isinstance(eos, bool):
                response_ids.append(eos)
        if not prompt_ids:
            bos = getattr(self.tokenizer, "bos_token_id", None)
            if not isinstance(bos, int) or isinstance(bos, bool):
                raise ValueError(
                    "the prompt encodes to no tokens and the tokenizer has no BOS "
                    "token; the first response token needs a preceding position"
                )
            prompt_ids = [bos]
        if not response_ids:
            raise ValueError("the response encodes to no tokens")
        return self._truncate(prompt_ids, response_ids)

    def _encode_chat(
        self, prompt: str | Sequence[Message], response: str | Message
    ) -> tuple[list[int], list[int]]:
        messages = [*_as_messages(prompt), _assistant(response)]
        located = final_turn_token_mask(self.tokenizer, messages)
        if located is None:
            raise ValueError("the chat template could not locate the response turn")
        ids, mask = located
        if True not in mask:
            raise ValueError("the response renders to no tokens")
        start = mask.index(True)
        if not all(mask[start:]):
            raise ValueError("the response tokens are not contiguous")
        return list(ids[:start]), list(ids[start:])

    def _tokenize(self, text: str, *, special: bool) -> list[int]:
        if not text:
            return []
        try:
            encoded = self.tokenizer(text, add_special_tokens=special)
            ids = encoded["input_ids"]
        except (TypeError, KeyError):
            ids = self.tokenizer.encode(text)
        return [int(token) for token in ids]

    def _truncate(
        self, prompt_ids: list[int], response_ids: list[int]
    ) -> EncodedSequence:
        truncated = False
        if (
            self.max_prompt_length is not None
            and len(prompt_ids) > self.max_prompt_length
        ):
            prompt_ids = prompt_ids[-self.max_prompt_length :]
            truncated = True
        if len(prompt_ids) + len(response_ids) > self.max_length:
            keep = min(len(prompt_ids), self.max_length - 1)
            prompt_ids = prompt_ids[len(prompt_ids) - keep :]
            response_ids = response_ids[: self.max_length - len(prompt_ids)]
            truncated = True
        return EncodedSequence(tuple(prompt_ids), tuple(response_ids), truncated)
