"""Shared, framework-neutral evaluation helpers for measured shootouts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def evaluate_causal_lm(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Any],
    *,
    format_prompt: Callable[[Any], str],
    score_response: Callable[[Any, str], tuple[float, bool]],
    max_tokens: int,
) -> dict[str, float]:
    """Evaluate one model state with the shootout's deterministic protocol."""
    import torch

    model.eval()
    total_score = 0.0
    parseable = 0
    for example in examples:
        encoded = tokenizer(format_prompt(example), return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[-1]
        response = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
        score, parsed = score_response(example, response)
        total_score += float(score)
        parseable += int(parsed)
    count = max(len(examples), 1)
    return {
        "pass_at_1": total_score / count,
        "parse_rate": parseable / count,
        "n": float(count),
    }
