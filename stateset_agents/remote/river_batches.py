"""Batch construction for the River AI training service — pure, SDK-free.

River (https://docs.river.ai) is a *remote autograd service*, not a job
submitter: you keep the training loop on your machine and hand River
token-aligned batches, one ``forward_backward`` call at a time. So the part
that has to be right is not the RPC plumbing — it is the shape of the batch.
This module is that part, deliberately isolated: no ``river_client`` import,
no network, no state. Everything here is testable on a laptop.

.. warning::

   **UNVERIFIED AGAINST THE LIVE SERVICE.** Everything below was derived from
   River's public documentation. We hold no River API key and ``river-client``
   is not installable from PyPI here, so not one byte of this has been checked
   against a real ``forward_backward`` call.

   The single most consequential guess is the **target shift**. River's SFT
   datum is documented as ``{"input_ids", "target_tokens", "weights"}`` without
   stating who performs the causal shift. We assume **the caller does**, and
   emit, for a tokenized conversation ``t[0..n-1]``::

       input_ids     = t[0 : n-1]     # every token but the last
       target_tokens = t[1 : n]       # each position's next-token target
       weights       = w[1 : n]       # weight of the *target*, not the input

   All three lists therefore have length ``n - 1`` and are index-aligned:
   ``target_tokens[i]`` is what the model should predict after consuming
   ``input_ids[0..i]``, and ``weights[i]`` scales that position's loss.

   If River instead shifts internally, the symptom is unmistakable and cheap
   to detect on the first real run: loss will be high and flat, and sampled
   continuations will look off-by-one (the model predicting the token it was
   just given). The fix is one line — pass ``t[0:n]`` as both ``input_ids``
   and, shifted, as targets — so this assumption is isolated in
   :func:`_shift_for_causal_lm` and nowhere else.

The second assumption is **prefix-stable tokenization**: we locate the
assistant spans by tokenizing successively longer chat-template renderings and
diffing their lengths. That is exact for BPE/SentencePiece tokenizers in
practice, but a tokenizer that re-segments across a boundary would shift a span
by a token or two. Weights, not correctness of the ids, are what would suffer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DOCUMENTED_BASE_MODELS",
    "MAX_LORA_RANK",
    "MIN_LORA_RANK",
    "build_rl_batch",
    "build_sft_batch",
    "validate_base_model",
    "validate_lora_rank",
]

#: River's documented LoRA rank window. Outside it the service rejects the
#: ``create_model`` call, so we refuse locally with a clearer message.
MIN_LORA_RANK = 1
MAX_LORA_RANK = 32

#: Base models named in River's docs. This is *not* an allowlist: River scopes
#: model access per account via ``client.get_capabilities()``, so a name absent
#: here may still be perfectly valid for the caller — see
#: :func:`validate_base_model`.
DOCUMENTED_BASE_MODELS: tuple[str, ...] = (
    "nvidia/GLM-5.2-NVFP4",
    "nvidia/GLM-5.2-NVFP4-262K",
    "nvidia/Kimi-K2.6-NVFP4",
    "nvidia/Kimi-K2.6-NVFP4-262K",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-35B-A3B-FP8",
    "Qwen/Qwen3.5-122B-A10B-FP8",
    "Qwen/Qwen3.5-397B-A17B-FP8",
    "Qwen/Qwen3.6-35B-A3B-FP8",
)

#: Weight given to tokens the model is being taught to produce.
_COMPLETION_WEIGHT = 1.0
#: Weight given to prompt/user/system tokens. Zero means "do not train on
#: this position" — River documents 0.0 as excluding a token from the loss.
_PROMPT_WEIGHT = 0.0


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def validate_lora_rank(rank: int) -> int:
    """Return ``rank`` if River would accept it, else raise ``ValueError``.

    River documents LoRA ranks of 1..32. Checking here means a bad
    ``--lora-r`` fails before a dataset is tokenized, with a message that
    names whose limit it is.
    """
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise ValueError(f"LoRA rank must be an int, got {rank!r}")
    if rank < MIN_LORA_RANK or rank > MAX_LORA_RANK:
        raise ValueError(
            f"LoRA rank {rank} is outside River's supported range "
            f"{MIN_LORA_RANK}-{MAX_LORA_RANK}. River's service caps LoRA rank "
            f"at {MAX_LORA_RANK}; pick a rank in that window "
            f"(--lora-r {MAX_LORA_RANK} is the largest River will accept)."
        )
    return rank


def validate_base_model(name: str, allowed: Sequence[str] | None = None) -> str:
    """Return ``name``, warning (never failing) when it looks unfamiliar.

    Deliberately warn-not-fail: River authorizes base models *per account*
    (``client.get_capabilities()``), so this process cannot know the true set.
    Refusing an unlisted-but-authorized model would be worse than a warning
    followed by River's own, authoritative error.

    Pass ``allowed`` — e.g. the list from ``get_capabilities()`` — to check
    against the account's real entitlements instead of the doc list.
    """
    if not name or not name.strip():
        raise ValueError("base_model must be a non-empty model name")
    name = name.strip()
    known = tuple(allowed) if allowed is not None else DOCUMENTED_BASE_MODELS
    if name not in known:
        source = (
            "your account's capabilities"
            if allowed is not None
            else "River's documented base models"
        )
        logger.warning(
            "base model %r is not in %s (%s). Proceeding anyway — River "
            "authorizes models per account, so this may still be valid; if it "
            "is not, River will say so.",
            name,
            source,
            ", ".join(known) if len(known) <= 12 else f"{len(known)} models",
        )
    return name


# --------------------------------------------------------------------------
# SFT batches
# --------------------------------------------------------------------------


def _shift_for_causal_lm(
    ids: list[int], weights: list[float]
) -> tuple[list[int], list[int], list[float]]:
    """Apply the causal shift documented at the top of this module.

    The one place the UNVERIFIED target-shift assumption is implemented.
    """
    return ids[:-1], ids[1:], weights[1:]


def _render(tokenizer: Any, messages: list[dict[str, Any]], **kwargs: Any) -> str:
    """Chat-template a message list to text."""
    return str(tokenizer.apply_chat_template(messages, tokenize=False, **kwargs))


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Tokenize without adding a second set of special tokens.

    The chat template already emits BOS/EOS-style markers, so
    ``add_special_tokens=False`` avoids duplicating them. Tokenizers that do
    not accept the kwarg fall back to a plain call.
    """
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:  # pragma: no cover - simple tokenizers
        ids = tokenizer.encode(text)
    return [int(i) for i in ids]


def _valid_messages(row: Any) -> list[dict[str, Any]] | None:
    """Extract a usable ``messages`` list from a dataset row, or None."""
    if not isinstance(row, dict):
        return None
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    clean: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        clean.append({"role": role, "content": content})
    if not any(m["role"] == "assistant" and m["content"] for m in clean):
        # Nothing to train on: every token would have weight 0.
        return None
    return clean


def build_sft_batch(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    max_length: int = 2048,
    *,
    shift_targets: bool = True,
) -> list[dict[str, Any]]:
    """Turn chat rows into River SFT data — one datum per usable row.

    ``rows`` are in the shape ``training.sft.load_chat_dataset`` returns:
    ``{"messages": [{"role", "content"}, ...]}``. Rows that are malformed,
    empty, or carry no assistant content are skipped with a warning rather
    than failing the run — a single bad line should not cost a training job.

    Prompt tokens (system/user/tool, plus the generation prefix that precedes
    each assistant reply) get weight ``0.0``; assistant tokens get ``1.0``.
    Loss is therefore computed only on what the model should say — the whole
    point of masking. Multi-turn conversations get *every* assistant turn
    weighted, not just the last.

    Truncation keeps the first ``max_length`` tokens; a row left with no
    weighted token after truncation is dropped, since it would contribute
    nothing but compute.

    ``shift_targets`` is the flip for the UNVERIFIED assumption documented at
    the top of this module. ``True`` (default): the caller performs the causal
    shift and each datum carries ``input_ids``/``target_tokens``/``weights``.
    ``False``: River shifts server-side — as its docs' loss table
    (``cross_entropy``: ``input_ids``, ``weights``) hints — and each datum
    carries only ``input_ids``/``weights``, unshifted. One argument, because
    getting this wrong off-by-ones every label in the batch.
    """
    if max_length <= 1:
        raise ValueError(f"max_length must be > 1, got {max_length}")

    batch: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        messages = _valid_messages(row)
        if messages is None:
            logger.warning("river: skipping row %d — no usable messages", index)
            continue
        try:
            datum = _build_sft_datum(
                messages, tokenizer, max_length, shift_targets=shift_targets
            )
        except Exception as exc:  # noqa: BLE001 - one bad row must not fail the job
            logger.warning("river: skipping row %d — %s", index, exc)
            continue
        if datum is not None:
            batch.append(datum)
    return batch


def _build_sft_datum(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    *,
    shift_targets: bool = True,
) -> dict[str, Any] | None:
    """One row -> one River SFT datum, or None when nothing is trainable."""
    ids: list[int] = []
    weights: list[float] = []

    for i, message in enumerate(messages):
        if message["role"] != "assistant" or not message["content"]:
            continue

        # Everything up to and including the generation prefix is prompt.
        prefix_ids = _encode(
            tokenizer,
            _render(tokenizer, messages[:i], add_generation_prompt=True),
        )
        through_ids = _encode(tokenizer, _render(tokenizer, messages[: i + 1]))

        if len(through_ids) <= len(prefix_ids):
            # Tokenizer did not grow across the assistant turn — nothing we
            # can honestly attribute to the completion.
            continue

        # Extend the running sequence to the prompt prefix (weight 0) …
        while len(ids) < len(prefix_ids):
            ids.append(prefix_ids[len(ids)])
            weights.append(_PROMPT_WEIGHT)
        # … then to the end of the assistant turn (weight 1).
        while len(ids) < len(through_ids):
            ids.append(through_ids[len(ids)])
            weights.append(_COMPLETION_WEIGHT)

    if not ids:
        return None

    # Any trailing messages after the final assistant turn are prompt-only and
    # deliberately dropped: they carry no loss and only consume context.
    ids = ids[:max_length]
    weights = weights[:max_length]

    if not shift_targets:
        # River shifts server-side: send the sequence unshifted, labels
        # implied. Weight layout matches the docs' cross_entropy fields.
        if len(ids) < 2 or not any(w > 0.0 for w in weights):
            return None
        return {"input_ids": ids, "weights": weights}

    input_ids, target_tokens, target_weights = _shift_for_causal_lm(ids, weights)
    if not input_ids or not any(w > 0.0 for w in target_weights):
        return None

    return {
        "input_ids": input_ids,
        "target_tokens": target_tokens,
        "weights": target_weights,
    }


# --------------------------------------------------------------------------
# RL batches
# --------------------------------------------------------------------------


def build_rl_batch(
    trajectories: Sequence[Any],
    tokenizer: Any,
    old_logprobs: Sequence[Sequence[float]],
    advantages: Sequence[float] | Sequence[Sequence[float]] | None = None,
    max_length: int = 2048,
) -> list[dict[str, Any]]:
    """Turn trajectories into River RL data (importance-sampling/PPO/CISPO).

    ``old_logprobs`` is a **required explicit argument**, one sequence of
    per-token logprobs per trajectory, covering the *completion* tokens only.
    That is deliberate. Our :class:`~stateset_agents.core.trajectory.Trajectory`
    carries a ``log_probs`` field, but nothing in the type guarantees it is
    per-token, aligned to this tokenizer, or produced by the policy that River
    now holds — and an RL batch built from misaligned logprobs trains silently
    in the wrong direction. So the caller must supply them from whatever
    actually generated the samples; we refuse to invent them.

    ``advantages`` may be one scalar per trajectory (broadcast across its
    completion tokens, the usual GRPO-style whole-sequence credit) or an
    explicit per-token sequence. When omitted, each trajectory's ``reward``
    is used as its scalar advantage.

    Prompt tokens get advantage ``0.0`` — River documents 0.0 as excluding a
    token from the loss — and ``attention_mask`` is all ones, since we emit
    unpadded per-datum sequences.
    """
    if len(old_logprobs) != len(trajectories):
        raise ValueError(
            f"old_logprobs has {len(old_logprobs)} entries but there are "
            f"{len(trajectories)} trajectories; supply one per trajectory"
        )
    if advantages is not None and len(advantages) != len(trajectories):
        raise ValueError(
            f"advantages has {len(advantages)} entries but there are "
            f"{len(trajectories)} trajectories"
        )

    batch: list[dict[str, Any]] = []
    for index, trajectory in enumerate(trajectories):
        prompt = str(getattr(trajectory, "prompt", "") or "")
        response = str(getattr(trajectory, "response", "") or "")
        if not response:
            logger.warning("river: skipping trajectory %d — empty response", index)
            continue

        prompt_ids = _encode(tokenizer, prompt)
        completion_ids = _encode(tokenizer, response)
        if not completion_ids:
            logger.warning("river: skipping trajectory %d — empty completion", index)
            continue

        logprobs = [float(x) for x in old_logprobs[index]]
        if len(logprobs) != len(completion_ids):
            raise ValueError(
                f"trajectory {index}: got {len(logprobs)} old_logprobs for "
                f"{len(completion_ids)} completion tokens. River aligns "
                f"logprobs per token, so the counts must match exactly — "
                f"they usually diverge because the logprobs came from a "
                f"different tokenizer than the one passed here."
            )

        raw_advantage = (
            advantages[index]
            if advantages is not None
            else float(getattr(trajectory, "reward", 0.0) or 0.0)
        )
        if isinstance(raw_advantage, (int, float)):
            completion_advantages = [float(raw_advantage)] * len(completion_ids)
        else:
            completion_advantages = [float(a) for a in raw_advantage]
            if len(completion_advantages) != len(completion_ids):
                raise ValueError(
                    f"trajectory {index}: {len(completion_advantages)} per-token "
                    f"advantages for {len(completion_ids)} completion tokens"
                )

        ids = (prompt_ids + completion_ids)[:max_length]
        # Prompt positions carry no credit; completion positions carry theirs.
        per_token_lp = [0.0] * len(prompt_ids) + logprobs
        per_token_adv = [0.0] * len(prompt_ids) + completion_advantages

        datum: dict[str, Any] = {
            "input_ids": ids,
            "old_logprobs": per_token_lp[: len(ids)],
            "advantages": per_token_adv[: len(ids)],
            "attention_mask": [1] * len(ids),
        }
        if not any(a != 0.0 for a in datum["advantages"]):
            logger.warning(
                "river: trajectory %d has zero advantage everywhere (likely "
                "truncated at max_length=%d); skipping",
                index,
                max_length,
            )
            continue
        batch.append(datum)
    return batch
