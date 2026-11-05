"""One identity for "the same request", shared by every split and eval check.

Train/validation splitting (``split_finetuning_data``,
``ConversationDataset.split``), the text SFT and held-out evaluation leakage
checks, and the eval ladder / flywheel prompt-set checks all decide whether
two records ask the same thing. They must agree, or data one boundary keeps
apart another lets through. A request's identity is its text under NFKC,
casefold and collapsed whitespace; generated prompt sets, whose only
difference between two renderings is a reference number, additionally mask
that reference (or every digit run when it is unknown).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "first_user_text",
    "normalize_identity_text",
    "prompt_identity",
]

_DIGITS = re.compile(r"\d+")
_REFERENCE = "\x00ref\x00"


def normalize_identity_text(text: str) -> str:
    """NFKC, casefold and collapse whitespace.

    Two requests that differ only in Unicode compatibility forms (full-width
    or non-breaking characters), letter case or whitespace are the same
    request for train/validation/test separation.
    """
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prompt_identity(
    text: str, *, mask_digits: bool = False, ref: str | None = None
) -> str:
    """The identity of one request's text.

    ``ref`` masks exactly that reference string; ``mask_digits`` masks every
    digit run (for generated prompts whose references are unknown). Without
    either, numbers are part of the request ("order 41" is not "order 42").
    """
    normalized = normalize_identity_text(text)
    if ref:
        return normalized.replace(normalize_identity_text(ref), _REFERENCE)
    if mask_digits:
        return _DIGITS.sub("#", normalized)
    return normalized


def first_user_text(messages: Sequence[Mapping[str, Any]]) -> str | None:
    """Content of the first user message (the request), or None."""
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    return None
