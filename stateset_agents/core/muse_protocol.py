"""Extract public final answers from Muse Spark's native recipient protocol."""

from __future__ import annotations

import re
from typing import Any

_START = "<|start|>"
_MESSAGE = "<|message|>"
_HEADER = re.compile(r"assistant\s+to=([^\s]+)\s*\Z")
_END = re.compile(r"<\|(?:eot|eom|end_of_text)\|>")


def is_muse_spark(model_name: str, model: Any = None) -> bool:
    """Identify official checkpoints and reloaded models with native Muse config."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return (
        isinstance(model_name, str)
        and model_name.startswith("meta-models/Muse-Spark-Open-")
    ) or (
        isinstance(model_type, str) and model_type in {"muse_spark", "muse_spark_text"}
    )


def muse_spark_final_response(text: str, *, allow_plain: bool = False) -> str:
    """Return only explicit ``to=user`` bodies, retaining no private channels.

    Local generation continues a prompt ending in ``<|start|>assistant``.
    Normalized provider responses may opt into plain text. Reasoning-only,
    incomplete headers and foreign-role continuations never become answers.
    """
    text = text.strip()
    if text.startswith("to=") or text.startswith(_MESSAGE):
        text = _START + "assistant " + text
    elif text.startswith("assistant to="):
        text = _START + text
    elif not text.startswith(_START):
        return text if allow_plain else ""
    answers = []
    for frame in text.split(_START)[1:]:
        if _MESSAGE not in frame:
            break
        header, body = frame.split(_MESSAGE, 1)
        match = _HEADER.fullmatch(header.strip())
        if match is None:
            break
        if match.group(1) == "user":
            answers.append(_END.split(body, maxsplit=1)[0])
        end = _END.search(body)
        if end is not None and end.group() != "<|eom|>":
            break
    return "".join(answers).strip()


__all__ = ["is_muse_spark", "muse_spark_final_response"]
