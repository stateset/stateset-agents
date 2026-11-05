"""
Server-side limits and an allowlist for generation parameters.

The request models for ``/v1/messages`` and ``/v1/chat/completions`` accept
extra fields so that OpenAI/Anthropic clients keep working as their schemas
evolve. Extra fields must never be forwarded blindly to the inference backend,
though: vLLM honours many knobs (``n``, ``best_of``, ``ignore_eos``,
``guided_*``...) that let a single request fan out or run unbounded.

This module enforces a ``max_tokens`` ceiling and forwards only a known-safe
set of sampling parameters, each validated and bounded. Unknown fields are
dropped (logged at debug level). Violations raise :class:`ValueError`, which
the routers translate into HTTP 422, matching the existing validation
convention for message counts and lengths.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import ValidationConfig
    from .messages_models import MessagesRequest

logger = logging.getLogger(__name__)

# Fields that are always handled explicitly by the gateway (transformed or
# gateway-only) and therefore never part of the passthrough set.
GATEWAY_MANAGED_FIELDS: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "system",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "stop_sequences",
        "stream",
        "stream_options",
        "tools",
        "tool_choice",
        "metadata",
        "response_format",
    }
)

MAX_TOP_LOGPROBS = 20
MAX_USER_LENGTH = 256


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _float_in_range(name: str, low: float, high: float) -> Callable[[Any], Any]:
    def _check(value: Any) -> Any:
        if not _is_number(value) or not (low <= float(value) <= high):
            raise ValueError(f"'{name}' must be a number between {low} and {high}")
        return value

    return _check


def _positive_float(name: str, high: float) -> Callable[[Any], Any]:
    def _check(value: Any) -> Any:
        if not _is_number(value) or not (0.0 < float(value) <= high):
            raise ValueError(f"'{name}' must be a number in (0, {high}]")
        return value

    return _check


def _bool(name: str) -> Callable[[Any], Any]:
    def _check(value: Any) -> Any:
        if not isinstance(value, bool):
            raise ValueError(f"'{name}' must be a boolean")
        return value

    return _check


def _validate_seed(value: Any) -> Any:
    if not _is_int(value) or not (-(2**63) <= value < 2**64):
        raise ValueError("'seed' must be a 64-bit integer")
    return value


def _validate_top_logprobs(value: Any) -> Any:
    if not _is_int(value) or not (0 <= value <= MAX_TOP_LOGPROBS):
        raise ValueError(
            f"'top_logprobs' must be an integer between 0 and {MAX_TOP_LOGPROBS}"
        )
    return value


def _validate_user(value: Any) -> Any:
    if not isinstance(value, str) or len(value) > MAX_USER_LENGTH:
        raise ValueError(f"'user' must be a string of at most {MAX_USER_LENGTH} chars")
    return value


_SIMPLE_VALIDATORS: dict[str, Callable[[Any], Any]] = {
    "presence_penalty": _float_in_range("presence_penalty", -2.0, 2.0),
    "frequency_penalty": _float_in_range("frequency_penalty", -2.0, 2.0),
    "repetition_penalty": _positive_float("repetition_penalty", 2.0),
    "min_p": _float_in_range("min_p", 0.0, 1.0),
    "seed": _validate_seed,
    "logprobs": _bool("logprobs"),
    "top_logprobs": _validate_top_logprobs,
    "parallel_tool_calls": _bool("parallel_tool_calls"),
    "user": _validate_user,
}

# Public view of every passthrough field accepted by the gateway.
ALLOWED_PASSTHROUGH_FIELDS: frozenset[str] = frozenset(
    set(_SIMPLE_VALIDATORS) | {"logit_bias", "n", "best_of", "min_tokens"}
)


def _validate_logit_bias(value: Any, limits: ValidationConfig) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("'logit_bias' must be an object of token id -> bias")
    if len(value) > limits.max_logit_bias_entries:
        raise ValueError(
            f"'logit_bias' has too many entries (max {limits.max_logit_bias_entries})"
        )
    cleaned: dict[str, Any] = {}
    for token, bias in value.items():
        token_str = str(token)
        if not token_str.isdigit():
            raise ValueError("'logit_bias' keys must be integer token ids")
        if not _is_number(bias) or not (-100.0 <= float(bias) <= 100.0):
            raise ValueError("'logit_bias' values must be numbers between -100 and 100")
        cleaned[token_str] = bias
    return cleaned


def _validate_fan_out(name: str, value: Any, limits: ValidationConfig) -> int:
    ceiling = limits.max_completions_per_request
    if not _is_int(value) or not (1 <= value <= ceiling):
        raise ValueError(f"'{name}' must be an integer between 1 and {ceiling}")
    return int(value)


def validate_max_tokens(max_tokens: int | None, limits: ValidationConfig) -> None:
    """Reject ``max_tokens`` values above the configured server ceiling."""
    if max_tokens is None:
        return
    ceiling = limits.max_generation_tokens
    if max_tokens > ceiling:
        raise ValueError(f"'max_tokens' must be <= {ceiling} (got {max_tokens})")


def sanitize_passthrough_params(
    extra: Mapping[str, Any] | None,
    limits: ValidationConfig,
    *,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Return the validated subset of extra request fields safe to forward.

    Args:
        extra: Extra (non-modelled) request fields supplied by the client.
        limits: Validation limits from the API config.
        max_tokens: Effective ``max_tokens`` for the request; ``min_tokens``
            may not exceed it.

    Returns:
        A new dict containing only allowlisted, validated fields.

    Raises:
        ValueError: If an allowlisted field is present with an invalid or
            out-of-bounds value.
    """
    if not extra:
        return {}

    forwarded: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in extra.items():
        if value is None or key in GATEWAY_MANAGED_FIELDS:
            continue
        if key in _SIMPLE_VALIDATORS:
            forwarded[key] = _SIMPLE_VALIDATORS[key](value)
        elif key == "logit_bias":
            forwarded[key] = _validate_logit_bias(value, limits)
        elif key in ("n", "best_of"):
            forwarded[key] = _validate_fan_out(key, value, limits)
        elif key == "min_tokens":
            ceiling = max_tokens or limits.max_generation_tokens
            if not _is_int(value) or not (0 <= value <= ceiling):
                raise ValueError(
                    f"'min_tokens' must be an integer between 0 and {ceiling}"
                )
            forwarded[key] = value
        else:
            dropped.append(key)

    if "n" in forwarded and "best_of" in forwarded:
        if forwarded["best_of"] < forwarded["n"]:
            raise ValueError("'best_of' must be greater than or equal to 'n'")

    if dropped:
        logger.debug("Dropping unsupported generation parameters: %s", sorted(dropped))
    return forwarded


def validate_generation_request(
    request: MessagesRequest, limits: ValidationConfig
) -> dict[str, Any]:
    """Validate generation limits on a request and return safe passthrough fields.

    Raises:
        ValueError: If ``max_tokens``, stop sequences, or any allowlisted
            passthrough field violates the configured limits.
    """
    validate_max_tokens(request.max_tokens, limits)
    if (
        request.stop_sequences
        and len(request.stop_sequences) > limits.max_stop_sequences
    ):
        raise ValueError(f"Too many stop sequences (max {limits.max_stop_sequences})")
    return sanitize_passthrough_params(
        request.model_extra,
        limits,
        max_tokens=request.max_tokens,
    )
