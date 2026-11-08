"""Bounded greedy generation for measuring a weight edit's actual answers."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ...core.errors import ConfigurationError

if TYPE_CHECKING:
    from .session import WeightSpaceSession

__all__ = ["GreedyGeneration", "matches_answer_prefix"]


@dataclass(frozen=True)
class GreedyGeneration:
    """Generated IDs include a stopping EOS token; no decoded text is inferred."""

    prompt_sha256: str
    token_ids: tuple[int, ...]
    finish_reason: Literal["eos", "length"]
    model_forwards: int


def matches_answer_prefix(text: str, answers: Sequence[str]) -> bool:
    """Match a declared answer at the start, with a complete answer boundary.

    Normalize Unicode, whitespace, case and opening quotation marks only. This
    is a completion-prefix metric, not a semantic judge or substring search.
    Reasoning, negation and words merely containing an answer do not match.
    """

    def normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    value = normalize(text).lstrip(" \"'“‘")
    if not answers or any(not normalize(a) for a in answers):
        raise ConfigurationError("answer aliases must be nonempty")
    for alias in answers:
        wanted = normalize(alias)
        if value.startswith(wanted) and (
            len(value) == len(wanted) or value[len(wanted)] in " \t\n.,:;!?\"'”’)]}"
        ):
            return True
    return False


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _generate(
    session: WeightSpaceSession,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    eos_token_ids: Sequence[int] | None,
    use_cache: bool,
    max_context_tokens: int,
    max_total_tokens: int,
    max_logit_elements: int,
    reason: str,
) -> tuple[GreedyGeneration, ...]:
    import torch

    for name, limit in (
        ("max_new_tokens", max_new_tokens),
        ("max_context_tokens", max_context_tokens),
        ("max_total_tokens", max_total_tokens),
        ("max_logit_elements", max_logit_elements),
    ):
        if type(limit) is not int or limit < 1:
            raise ConfigurationError(f"{name} must be a positive integer")
    if type(use_cache) is not bool or not prompts:
        raise ConfigurationError("generation needs prompts and a boolean use_cache")
    if any(module.training for module in session.model.modules()):
        raise ConfigurationError("generation requires every module in eval mode")
    encoded: list[tuple[int, ...]] = []
    for prompt in prompts:
        if not isinstance(prompt, str) or not prompt:
            raise ConfigurationError("generation prompts must be nonempty strings")
        tokens = tuple(session.encode(prompt))
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ConfigurationError(
                "generation prompts must encode to nonnegative token ids"
            )
        encoded.append(tokens)
    if max(map(len, encoded)) + max_new_tokens > max_context_tokens:
        raise ConfigurationError("generation exceeds max_context_tokens")
    if sum(map(len, encoded)) + len(encoded) * max_new_tokens > max_total_tokens:
        raise ConfigurationError("generation exceeds max_total_tokens")
    if eos_token_ids is None:
        eos = getattr(
            getattr(session.model, "generation_config", None), "eos_token_id", None
        )
        if eos is None:
            eos = getattr(getattr(session.model, "config", None), "eos_token_id", None)
        eos_token_ids = () if eos is None else ((eos,) if type(eos) is int else eos)
    try:
        raw_eos = tuple(eos_token_ids)
    except TypeError as exc:
        raise ConfigurationError("EOS ids must be a sequence of integers") from exc
    if any(type(t) is not int or t < 0 for t in raw_eos):
        raise ConfigurationError("EOS ids must be nonnegative integers")
    eos_ids = tuple(sorted(set(raw_eos)))
    vocab = getattr(getattr(session.model, "config", None), "vocab_size", None)
    if type(vocab) is int and vocab > 0:
        if any(t >= vocab for seq in (*encoded, eos_ids) for t in seq):
            raise ConfigurationError("generation token id exceeds vocabulary")
        if vocab > max_logit_elements:
            raise ConfigurationError("generation logits exceed max_logit_elements")
    for name in session._weight_hashes:
        session._check_weight(name)
    session.budget.spend(reason)
    results: list[GreedyGeneration] = []
    forwards = 0
    policy = {
        "max_new_tokens": max_new_tokens,
        "eos_token_ids": eos_ids,
        "use_cache": use_cache,
    }
    record = {
        "cases_sha256": _digest(encoded),
        "policy": policy,
        "active_edit_ids": session.active_edit_ids,
    }
    device = next(session.model.parameters()).device
    try:
        for prompt_ids in encoded:
            generated: list[int] = []
            past: Any = None
            finish: Literal["eos", "length"] = "length"
            for _ in range(max_new_tokens):
                full = [*prompt_ids, *generated]
                ids = torch.tensor(
                    [full[-1:] if past is not None else full], device=device
                )
                mask = torch.ones((1, len(full)), dtype=torch.long, device=device)
                kwargs: dict[str, Any] = (
                    {"past_key_values": past} if past is not None else {}
                )
                forwards += 1
                with torch.no_grad():
                    output = session.model(
                        input_ids=ids,
                        attention_mask=mask,
                        logits_to_keep=1,
                        use_cache=use_cache,
                        **kwargs,
                    )
                logits = output.logits
                if (
                    logits.ndim != 3
                    or logits.shape[0] != 1
                    or logits.shape[1] < 1
                    or logits.shape[2] < 1
                ):
                    raise ConfigurationError(
                        "generation requires nonempty causal LM logits"
                    )
                if logits.numel() > max_logit_elements:
                    raise ConfigurationError(
                        "generation logits exceed max_logit_elements"
                    )
                last = logits[0, -1]
                if not bool(torch.isfinite(last).all()):
                    raise ConfigurationError("generation produced nonfinite logits")
                token = int(last.argmax())
                generated.append(token)
                past = getattr(output, "past_key_values", None) if use_cache else None
                del output, logits, last
                if token in eos_ids:
                    finish = "eos"
                    break
            results.append(
                GreedyGeneration(
                    _digest(prompt_ids), tuple(generated), finish, len(generated)
                )
            )
            del past
        for name in session._weight_hashes:
            session._check_weight(name)
    except BaseException as exc:
        session.generations.append(
            {
                **record,
                "status": "failed",
                "error_type": type(exc).__name__,
                "model_forwards": forwards,
                "completed_prompts": len(results),
            }
        )
        raise
    session.generations.append(
        {
            **record,
            "status": "completed",
            "model_forwards": forwards,
            "output_tokens": sum(len(r.token_ids) for r in results),
            "outputs_sha256": _digest([r.token_ids for r in results]),
            "finish_reasons": [r.finish_reason for r in results],
        }
    )
    return tuple(results)
