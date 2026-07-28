"""
GSM8K dataset loader and verifier reward for math-reasoning benchmarks.

GSM8K is a 8.5K-problem grade-school math dataset where each problem has a
numeric ground-truth answer. It's the canonical small-but-non-trivial
verifiable-reward task — the right shape for an RL fine-tuning benchmark
because correctness is mechanical to check.

This module provides:

* :class:`GSM8KExample` — a parsed problem with extracted gold answer.
* :func:`load_gsm8k` — Hugging Face dataset loader with answer pre-extraction.
* :class:`GSM8KReward` — a ``RewardFunction`` that returns 1.0 for correct
  numeric answers and 0.0 otherwise. The reward parses common answer formats
  ("The answer is 42", "#### 42", "42", "$42.00") robustly.
* :func:`make_gsm8k_scenarios` — converts examples to ``ConversationEnvironment``
  scenario dicts.

Usage:

    from stateset_agents.data.gsm8k import load_gsm8k, GSM8KReward
    from stateset_agents.core import ConversationEnvironment

    train, test = load_gsm8k()
    env = ConversationEnvironment(
        scenarios=make_gsm8k_scenarios(train),
        reward_fn=GSM8KReward(),
        max_turns=1,
    )
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..core.reward_base import RewardFunction, RewardResult, RewardType
from ..core.trajectory import ConversationTurn

# The canonical GSM8K answer format ends with "#### <number>" on its own line.
_GOLD_ANSWER_RE = re.compile(r"####\s*([\-+]?[\d,\.]+)")

# Generated answer extraction — try several patterns in priority order.
_ANSWER_PATTERNS = [
    # "The answer is 42" or "answer: 42" — most common in instruction-tuned outputs.
    re.compile(
        r"(?:the\s+)?answer\s+is\s*[:=]?\s*\$?([\-+]?[\d,]+(?:\.\d+)?)", re.IGNORECASE
    ),
    re.compile(r"answer\s*[:=]\s*\$?([\-+]?[\d,]+(?:\.\d+)?)", re.IGNORECASE),
    # "#### 42" — GSM8K's native format if the model imitates it.
    re.compile(r"####\s*\$?([\-+]?[\d,]+(?:\.\d+)?)"),
    # "\boxed{42}" — LaTeX format from math-tuned models.
    re.compile(r"\\boxed\{\$?([\-+]?[\d,]+(?:\.\d+)?)\}"),
    # Fallback: last number anywhere in the text.
    re.compile(r"\$?([\-+]?[\d,]+(?:\.\d+)?)(?!.*\d)", re.DOTALL),
]


@dataclass
class GSM8KExample:
    """A single GSM8K problem with extracted gold answer."""

    question: str
    answer_text: str  # full chain-of-thought + final answer
    gold_answer: float  # parsed numeric answer

    def to_scenario(self) -> dict[str, Any]:
        return {
            "user_query": self.question,
            "gold_answer": self.gold_answer,
            "answer_text": self.answer_text,
        }


def _parse_number(s: str | None) -> float | None:
    """Parse a number string, tolerating commas and currency symbols."""
    if s is None:
        return None
    cleaned = s.strip().replace(",", "").replace("$", "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def extract_gold_answer(answer_text: str) -> float | None:
    """Extract the numeric gold answer from GSM8K's "#### N" format."""
    match = _GOLD_ANSWER_RE.search(answer_text)
    if not match:
        return None
    return _parse_number(match.group(1))


def extract_predicted_answer(response: str) -> float | None:
    """Extract a numeric answer from a free-form model response.

    Tries several patterns in priority order: "the answer is N", "answer: N",
    "#### N", "\\boxed{N}", and finally the last number in the text.
    Returns None if no number can be parsed.
    """
    if not response:
        return None
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(response)
        if match:
            value = _parse_number(match.group(1))
            if value is not None:
                return value
    return None


def load_gsm8k(
    split: str | None = None,
    cache_dir: str | None = None,
    limit: int | None = None,
) -> list[GSM8KExample] | tuple[list[GSM8KExample], list[GSM8KExample]]:
    """Load GSM8K from Hugging Face.

    Args:
        split: "train", "test", or None for both (returned as a tuple).
        cache_dir: Optional Hugging Face cache directory.
        limit: If set, return at most this many examples per split. Useful for
            smoke tests on Colab.

    Returns:
        A list of ``GSM8KExample`` if ``split`` is named, otherwise a
        ``(train, test)`` tuple.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "GSM8K requires the `datasets` library. "
            "Install it with `pip install datasets`."
        ) from e

    def _to_examples(hf_split: Any) -> list[GSM8KExample]:
        examples: list[GSM8KExample] = []
        for row in hf_split:
            gold = extract_gold_answer(row["answer"])
            if gold is None:
                continue  # skip malformed rows
            examples.append(
                GSM8KExample(
                    question=row["question"],
                    answer_text=row["answer"],
                    gold_answer=gold,
                )
            )
            if limit is not None and len(examples) >= limit:
                break
        return examples

    # Dataset repo id ("openai/gsm8k") is a fixed, well-known public
    # benchmark name, not attacker-controlled input; pinning a revision
    # would require this module to track upstream commit hashes for a
    # dataset it doesn't own. The bare "gsm8k" repo id (no namespace) that
    # used to resolve here no longer does -- recent `datasets`/
    # `huggingface_hub` releases validate repo ids as "namespace/name"
    # before ever hitting the network, so "gsm8k" alone fails locally with
    # `HFValidationError: Repository id must be 'namespace/name', got
    # 'gsm8k'` regardless of connectivity. "openai/gsm8k" is the correct,
    # currently-resolving repo id with the same "main" config/schema.
    if split is not None:
        ds = load_dataset(
            "openai/gsm8k", "main", split=split, cache_dir=cache_dir
        )  # nosec: B615
        return _to_examples(ds)

    train = load_dataset(
        "openai/gsm8k", "main", split="train", cache_dir=cache_dir
    )  # nosec: B615
    test = load_dataset(
        "openai/gsm8k", "main", split="test", cache_dir=cache_dir
    )  # nosec: B615
    return _to_examples(train), _to_examples(test)


def make_gsm8k_scenarios(examples: Sequence[GSM8KExample]) -> list[dict[str, Any]]:
    """Convert a sequence of examples into ``ConversationEnvironment`` scenarios."""
    return [ex.to_scenario() for ex in examples]


class GSM8KReward(RewardFunction):
    """Reward function for GSM8K problems.

    Returns 1.0 if the response's extracted numeric answer matches the gold
    answer (within ``tolerance``), 0.0 otherwise. The ``breakdown`` dict
    surfaces the parsed values so training dashboards can diagnose failures.

    The reward expects ``context["gold_answer"]`` to contain the ground-truth
    numeric answer (populated by ``make_gsm8k_scenarios``).
    """

    name = "gsm8k"

    def __init__(self, weight: float = 1.0, tolerance: float = 1e-3) -> None:
        super().__init__(weight=weight, reward_type=RewardType.SPARSE, name=self.name)
        self.tolerance = tolerance

    async def compute_reward(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        if not turns:
            return RewardResult(score=0.0, breakdown={"no_response": 1.0})

        # Final assistant turn is the answer.
        response = turns[-1].content or ""
        gold = (context or {}).get("gold_answer")

        if gold is None:
            return RewardResult(
                score=0.0,
                breakdown={"no_gold": 1.0},
                explanation="No gold_answer in context",
            )

        predicted = extract_predicted_answer(response)
        if predicted is None:
            return RewardResult(
                score=0.0,
                breakdown={"unparseable": 1.0, "gold": float(gold)},
                explanation=f"Could not parse a numeric answer from response of length {len(response)}",
            )

        correct = abs(predicted - float(gold)) <= self.tolerance
        return RewardResult(
            score=1.0 if correct else 0.0,
            breakdown={
                "correct": 1.0 if correct else 0.0,
                "predicted": float(predicted),
                "gold": float(gold),
                "abs_error": abs(predicted - float(gold)),
            },
            explanation=(
                "Correct" if correct else f"Predicted {predicted}, expected {gold}"
            ),
        )


class PartialCreditGSM8KReward(RewardFunction):
    """Dense-reward variant of GSM8KReward.

    Gives partial credit so the gradient signal doesn't collapse when most
    rollouts are incorrect — which is the dominant regime for weak base
    models on GSM8K. Reward schedule:

    - ``0.0`` if the response has no parseable numeric answer.
    - ``parseable_weight`` (default 0.2) if a number was parsed but is wrong
      and not "close" to gold.
    - ``close_weight`` (default 0.5) if the parsed answer is within
      ``close_relative_tolerance`` of the gold value (default 10%).
    - ``1.0`` if exactly correct within ``tolerance`` (default 1e-3).

    The motivation is to keep within-group variance above zero on early
    training when the model can produce numbers but rarely the right one.
    With a binary 0/1 reward, those groups are all-zero and contribute no
    gradient. With this shaped reward, "parseable but wrong" sits at 0.2,
    "close" at 0.5, "correct" at 1.0 — so even a group of (unparseable,
    wrong, wrong, close) has spread (0.0, 0.2, 0.2, 0.5) and a non-zero
    advantage signal.
    """

    name = "gsm8k_partial"

    def __init__(
        self,
        weight: float = 1.0,
        tolerance: float = 1e-3,
        close_relative_tolerance: float = 0.1,
        parseable_weight: float = 0.2,
        close_weight: float = 0.5,
    ) -> None:
        super().__init__(weight=weight, reward_type=RewardType.SPARSE, name=self.name)
        self.tolerance = tolerance
        self.close_relative_tolerance = close_relative_tolerance
        self.parseable_weight = parseable_weight
        self.close_weight = close_weight

    async def compute_reward(
        self,
        turns: list[ConversationTurn],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        if not turns:
            return RewardResult(score=0.0, breakdown={"no_response": 1.0})

        response = turns[-1].content or ""
        gold = (context or {}).get("gold_answer")

        if gold is None:
            return RewardResult(
                score=0.0,
                breakdown={"no_gold": 1.0},
                explanation="No gold_answer in context",
            )

        predicted = extract_predicted_answer(response)
        if predicted is None:
            return RewardResult(
                score=0.0,
                breakdown={"unparseable": 1.0, "gold": float(gold)},
                explanation=f"Could not parse a numeric answer from response of length {len(response)}",
            )

        gold_f = float(gold)
        abs_error = abs(predicted - gold_f)

        if abs_error <= self.tolerance:
            score = 1.0
            tier = "correct"
        else:
            # Relative tolerance — for gold == 0 fall back to absolute.
            denom = abs(gold_f) if abs(gold_f) > self.tolerance else 1.0
            relative_error = abs_error / denom
            if relative_error <= self.close_relative_tolerance:
                score = self.close_weight
                tier = "close"
            else:
                score = self.parseable_weight
                tier = "parseable_wrong"

        return RewardResult(
            score=score,
            breakdown={
                "tier_" + tier: 1.0,
                "predicted": float(predicted),
                "gold": gold_f,
                "abs_error": abs_error,
            },
            explanation=f"{tier}: predicted {predicted}, expected {gold_f}",
        )


__all__ = [
    "GSM8KExample",
    "GSM8KReward",
    "PartialCreditGSM8KReward",
    "extract_gold_answer",
    "extract_predicted_answer",
    "load_gsm8k",
    "make_gsm8k_scenarios",
]
