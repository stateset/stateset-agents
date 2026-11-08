"""Declared generated-answer and preservation gates for atomic edit trials."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ...core.errors import ConfigurationError
from .generation import (
    GreedyGeneration,
    _digest,
    _GenerationPlan,
    _prepare_generation,
    _run_generation,
    matches_answer_prefix,
)

if TYPE_CHECKING:
    from .session import WeightSpaceSession

__all__ = [
    "AnswerCase",
    "AnswerLimits",
    "AnswerCheck",
    "AnswerCaseScore",
    "AnswerMetrics",
    "answer_occurrences",
]


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _aliases(answers: Sequence[str]) -> tuple[str, ...]:
    if (
        isinstance(answers, str)
        or not answers
        or any(not isinstance(a, str) or not _normalize(a) for a in answers)
    ):
        raise ConfigurationError(
            "answers must be a nonempty sequence of nonempty strings"
        )
    return tuple(answers)


def answer_occurrences(text: str, answers: Sequence[str]) -> int:
    """Count nonoverlapping whole-answer aliases, longest first, after normalization.

    This is a lexical repetition check, not a semantic quality judgment. Nested
    aliases such as 'Dickens' and 'Charles Dickens' count once at the same span.
    """
    aliases = sorted(
        {_normalize(a) for a in _aliases(answers)}, key=lambda a: (-len(a), a)
    )
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(a) for a in aliases) + r")(?!\w)"
    return len(re.findall(pattern, _normalize(text)))


@dataclass(frozen=True)
class AnswerCase:
    """An expected completion prefix; locality retention uses base-correct cases."""

    prompt: str
    answers: tuple[str, ...]
    kind: Literal["target", "locality"] = "target"

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ConfigurationError("answer-case prompt must be a nonempty string")
        if self.kind not in ("target", "locality"):
            raise ConfigurationError("answer case kind must be target or locality")
        object.__setattr__(self, "answers", _aliases(self.answers))


@dataclass(frozen=True)
class AnswerLimits:
    """Predeclared lexical generation gates and resource caps.

    The default requires a correct target prefix on every target case, retains
    every base-correct locality answer, and needs at least one such locality
    case. EOS follows the model's configured IDs; prompts and EOS are frozen
    before measurement. Length-limited outputs can pass a prefix check.
    """

    min_target_accuracy: float = 1.0
    min_locality_retention: float = 1.0
    min_baseline_locality_correct: int = 1
    max_answer_occurrences: int | None = 2
    max_new_tokens: int = 32
    use_cache: bool = True
    max_context_tokens: int = 4096
    max_total_tokens: int = 16384
    max_logit_elements: int = 2_000_000
    max_decoded_chars: int = 65536

    def __post_init__(self) -> None:
        for name in ("min_target_accuracy", "min_locality_retention"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ConfigurationError(f"{name} must be finite and in [0, 1]")
        for name in (
            "max_new_tokens",
            "max_context_tokens",
            "max_total_tokens",
            "max_logit_elements",
            "max_decoded_chars",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ConfigurationError(f"{name} must be a positive integer")
        if (
            type(self.min_baseline_locality_correct) is not int
            or self.min_baseline_locality_correct < 0
        ):
            raise ConfigurationError(
                "min_baseline_locality_correct must be a nonnegative integer"
            )
        if self.max_answer_occurrences is not None and (
            type(self.max_answer_occurrences) is not int
            or self.max_answer_occurrences < 1
        ):
            raise ConfigurationError("max_answer_occurrences must be positive or None")
        if type(self.use_cache) is not bool:
            raise ConfigurationError("use_cache must be boolean")


@dataclass(frozen=True)
class AnswerCaseScore:
    """Case-order diagnostics; neither prompts nor decoded text are retained."""

    kind: Literal["target", "locality"]
    prompt_sha256: str
    baseline_correct: bool
    edited_correct: bool
    edited_answer_occurrences: int
    exact_generation_agreement: bool


@dataclass(frozen=True)
class AnswerMetrics:
    target_cases: int
    target_correct: int
    target_accuracy: float
    locality_cases: int
    baseline_locality_correct: int
    locality_retained: int
    locality_retention: float
    max_target_answer_occurrences: int


@dataclass(frozen=True)
class AnswerCheck:
    """Generated-answer evidence bound to cases and frozen decoding policy."""

    cases_sha256: str
    limits: AnswerLimits
    metrics: AnswerMetrics
    scores: tuple[AnswerCaseScore, ...]
    violations: tuple[str, ...]
    eos_token_ids: tuple[int, ...]
    baseline_outputs_sha256: str
    edited_outputs_sha256: str


@dataclass(frozen=True)
class _AnswerPlan:
    cases: tuple[AnswerCase, ...]
    generation: _GenerationPlan
    limits: AnswerLimits
    decode: Callable[[Sequence[int]], str]
    cases_sha256: str


@dataclass(frozen=True)
class _AnswerSnapshot:
    outputs: tuple[GreedyGeneration, ...]
    correct: tuple[bool, ...]
    occurrences: tuple[int, ...]


def _prepare_answers(
    session: WeightSpaceSession,
    cases: Sequence[AnswerCase],
    limits: AnswerLimits | None,
) -> _AnswerPlan | None:
    if limits is None:
        if cases:
            raise ConfigurationError("answer_limits are required with answer_cases")
        return None
    if not isinstance(limits, AnswerLimits) or not callable(session.decode):
        raise ConfigurationError(
            "answer checks require AnswerLimits and a session decoder"
        )
    if not cases or any(not isinstance(c, AnswerCase) for c in cases):
        raise ConfigurationError("answer_cases must contain AnswerCase values")
    cases = tuple(cases)
    if not any(c.kind == "target" for c in cases):
        raise ConfigurationError("answer checks require at least one target case")
    if sum(c.kind == "locality" for c in cases) < limits.min_baseline_locality_correct:
        raise ConfigurationError("too few locality cases for the baseline requirement")
    plan = _prepare_generation(
        session,
        [c.prompt for c in cases],
        max_new_tokens=limits.max_new_tokens,
        eos_token_ids=None,
        use_cache=limits.use_cache,
        max_context_tokens=limits.max_context_tokens,
        max_total_tokens=limits.max_total_tokens,
        max_logit_elements=limits.max_logit_elements,
    )
    if len(set(plan.encoded)) != len(plan.encoded):
        raise ConfigurationError("answer cases must have distinct tokenized prompts")
    return _AnswerPlan(
        cases,
        plan,
        limits,
        session.decode,
        _digest(
            [
                (ids, c.kind, c.answers)
                for ids, c in zip(plan.encoded, cases, strict=True)
            ]
        ),
    )


def _answer_snapshot(
    session: WeightSpaceSession, plan: _AnswerPlan, reason: str
) -> _AnswerSnapshot:
    outputs = _run_generation(session, plan.generation, reason)
    correct, occurrences = [], []
    for case, output in zip(plan.cases, outputs, strict=True):
        text = plan.decode(output.token_ids)
        if not isinstance(text, str) or len(text) > plan.limits.max_decoded_chars:
            raise ConfigurationError(
                "decoder must return a string within max_decoded_chars"
            )
        correct.append(matches_answer_prefix(text, case.answers))
        occurrences.append(answer_occurrences(text, case.answers))
    return _AnswerSnapshot(outputs, tuple(correct), tuple(occurrences))


def _compare_answers(
    plan: _AnswerPlan, before: _AnswerSnapshot, after: _AnswerSnapshot
) -> AnswerCheck:
    scores = tuple(
        AnswerCaseScore(
            c.kind,
            a.prompt_sha256,
            b_correct,
            a_correct,
            count,
            b.token_ids == a.token_ids,
        )
        for c, b, a, b_correct, a_correct, count in zip(
            plan.cases,
            before.outputs,
            after.outputs,
            before.correct,
            after.correct,
            after.occurrences,
            strict=True,
        )
    )
    targets = [s for s in scores if s.kind == "target"]
    locality = [s for s in scores if s.kind == "locality"]
    base_correct = sum(s.baseline_correct for s in locality)
    retained = sum(s.baseline_correct and s.edited_correct for s in locality)
    correct = sum(s.edited_correct for s in targets)
    metrics = AnswerMetrics(
        len(targets),
        correct,
        correct / len(targets),
        len(locality),
        base_correct,
        retained,
        retained / base_correct if base_correct else 1.0,
        max(s.edited_answer_occurrences for s in targets),
    )
    failures = []
    if metrics.target_accuracy < plan.limits.min_target_accuracy:
        failures.append("target_accuracy")
    if base_correct < plan.limits.min_baseline_locality_correct:
        failures.append("baseline_locality_correct")
    if metrics.locality_retention < plan.limits.min_locality_retention:
        failures.append("locality_retention")
    if (
        plan.limits.max_answer_occurrences is not None
        and metrics.max_target_answer_occurrences > plan.limits.max_answer_occurrences
    ):
        failures.append("answer_repetition")
    return AnswerCheck(
        plan.cases_sha256,
        plan.limits,
        metrics,
        scores,
        tuple(failures),
        plan.generation.eos_ids,
        _digest([o.token_ids for o in before.outputs]),
        _digest([o.token_ids for o in after.outputs]),
    )
