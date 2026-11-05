"""Sample completions from a causal LM as exact token ids (on-policy trainers).

Online DPO and on-policy distillation train on what the policy actually
sampled, so completions are returned as token ids (cut after the first stop
token) rather than text that would be re-tokenized. Sampling is plain
temperature / nucleus sampling from an explicit ``GenerationConfig`` (no
top-k, repetition penalty or other defaults inherited from the checkpoint)
and draws from torch's global RNG, so a seeded trainer resumes exactly.

torch is fetched lazily, so importing this module never requires torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..trainer_utils import require_torch
from . import _loop

__all__ = ["stop_token_ids", "sample_completions"]


def _int_ids(value: Any) -> list[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if isinstance(value, list | tuple):
        return [
            int(item)
            for item in value
            if isinstance(item, int) and not isinstance(item, bool)
        ]
    return []


def stop_token_ids(model: Any, tokenizer: Any) -> list[int]:
    """The tokenizer's EOS plus any end-of-turn ids in the model's generation
    config (chat models stop on e.g. ``<|im_end|>``), excluding ids the model
    cannot emit (a config default beyond its vocabulary)."""
    generation_config = getattr(model, "generation_config", None)
    ids = _int_ids(getattr(tokenizer, "eos_token_id", None))
    ids += _int_ids(getattr(generation_config, "eos_token_id", None))
    vocab = getattr(getattr(model, "config", None), "vocab_size", None)
    if isinstance(vocab, int) and not isinstance(vocab, bool) and vocab > 0:
        ids = [token for token in ids if 0 <= token < vocab]
    return sorted(set(ids))


def _cut(tokens: list[int], stops: set[int]) -> tuple[int, ...]:
    for index, token in enumerate(tokens):
        if token in stops:
            return tuple(tokens[: index + 1])
    return tuple(tokens)


def sample_completions(
    model: Any,
    prompts: Sequence[Sequence[int]],
    *,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pad_token_id: int,
    stop_ids: Sequence[int],
    batch_size: int,
    suppress_tokens: Sequence[int] | None = None,
) -> list[list[tuple[int, ...]]]:
    """``num_samples`` sampled completions (token ids) for each prompt.

    Prompts are left-padded into batches of ``batch_size`` rows. Each
    completion ends at (and includes) its first stop token, or after
    ``max_new_tokens``. ``suppress_tokens`` can never be sampled. The model
    is in eval mode without gradients while sampling.
    """
    torch = require_torch()
    from transformers import GenerationConfig

    if num_samples < 1 or max_new_tokens < 1 or batch_size < 1:
        raise ValueError("num_samples, max_new_tokens and batch_size must be >= 1")
    stops = {int(token) for token in stop_ids}
    settings: dict[str, Any] = {
        "do_sample": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": 0,
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": int(pad_token_id),
        "eos_token_id": sorted(stops) or None,
    }
    if suppress_tokens:
        settings["suppress_tokens"] = [int(token) for token in suppress_tokens]
    generation_config = GenerationConfig(**settings)
    requests = [index for index in range(len(prompts)) for _ in range(num_samples)]
    results: list[list[tuple[int, ...]]] = [[] for _ in prompts]
    device = next(model.parameters()).device
    with _loop.evaluation_mode(model):
        for start in range(0, len(requests), batch_size):
            chunk = requests[start : start + batch_size]
            width = max(len(prompts[index]) for index in chunk)
            ids = torch.full((len(chunk), width), int(pad_token_id), dtype=torch.long)
            attention = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, index in enumerate(chunk):
                tokens = list(prompts[index])
                if not tokens:
                    raise ValueError("cannot sample from an empty prompt")
                ids[row, width - len(tokens) :] = torch.tensor(tokens)
                attention[row, width - len(tokens) :] = 1
            output = model.generate(
                input_ids=ids.to(device),
                attention_mask=attention.to(device),
                generation_config=generation_config,
            )
            sequences = output.sequences if hasattr(output, "sequences") else output
            for row, index in enumerate(chunk):
                completion = [int(t) for t in sequences[row, width:].tolist()]
                results[index].append(_cut(completion, stops))
    return results
