"""Verifiable math reward: does the reply commit to the reference answer?

:class:`MathAnswerVerifier` scores 1.0 when the newest assistant reply
commits to exactly one final answer equivalent to the reference, and 0.0
otherwise. The final answer is, in order of authority:

1. every ``\\boxed{...}`` (or ``\\fbox{...}``) in the reply;
2. otherwise every explicit statement: "the answer is X", "answer: X",
   "final answer = X", "#### X";
3. otherwise the last number, but only when the reply offers exactly one
   number beyond those quoted from the question (the candidate rule of
   :class:`~stateset_agents.core.basic_rewards.CorrectnessReward`), so worked
   arithmetic must state its answer explicitly.

Tool-call markup is removed first, so the arguments of a tool call are never
read as an answer. There is no partial credit for hedging, as in
``CorrectnessReward`` (whose helpers this reuses): several different
committed answers, a number offered as an alternative ("42, or maybe 41",
"41 or 42"), an admission of not knowing ("not sure", "no idea") and a
negated answer ("It is not 42.", "42 is not the answer") all score 0.

Equivalence (:mod:`stateset_agents.core.math_expressions`, which never calls
``eval``): a numeric reference must be matched by a number *written out*
(``84``, ``-3/4``, ``0.75``, ``\\frac{3}{4}``), compared exactly as a
fraction, or within ``abs_tolerance``/``rel_tolerance`` when set; restating
the question's arithmetic (``\\boxed{12*7}``) earns nothing. A symbolic
reference (``2x+2``) is matched by any equivalent expression (exact
evaluation at fixed random rational points, or ``sympy.simplify`` for
functions such as ``sqrt`` when sympy is installed); this cannot tell an
expanded form from an unexpanded one, so pass ``symbolic=False`` for
form-sensitive tasks (expand, factor, simplify) to require the same
canonical text. A reference that is not math is compared as normalized text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from stateset_agents.core.basic_rewards import (
    _NUMBER,
    _UNCERTAINTY_PATTERN,
    _answer_negated,
    _ground_truth_mapping,
    _normalize_text,
    _text_answer_credit,
)
from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.math_expressions import (
    ExpressionError,
    Node,
    equivalent,
    free_symbols,
    is_numeric_literal,
    normalize_answer_text,
    parse_expression,
    rational_value,
)
from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.reward_context import assistant_turns_in_scope
from stateset_agents.core.tool_calls import strip_tool_calls

__all__ = [
    "MathAnswerVerifier",
    "MathVerdict",
    "extract_final_answers",
    "verify_math_answer",
]

#: A number as a reply writes it, fractions included (``-3/4``, ``1,299``).
_LITERAL = re.compile(r"(?<![\w.])" + _NUMBER + r"(?:\s*/\s*\d+)?(?![\w]|\.\d)")
_STATED = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=|was)\s*:?\s*(.+?)"
    r"(?=\.\s|\.$|[\n;]|\s+(?:because|since|as)\b|$)",
    re.IGNORECASE,
)
_HASHES = re.compile(r"####\s*([^\n]+)")
_BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")
#: Connectives that offer a number as one of several answers.
#: A number, or arithmetic over numbers ("1/2 + 1"), offered as a value.
_VALUE = _LITERAL.pattern + r"(?:\s*[-+*/^]\s*" + _LITERAL.pattern + r")*"
_HEDGE = re.compile(
    r"\b(?P<word>or|maybe|perhaps|possibly|probably|alternatively|otherwise)\b"
    r"(?:\s+(?:it|it's|its|is|could|might|may|be|was|the|answer|even))*"
    r"\s*(?:\\boxed\{\s*)?\$?(?P<num>" + _VALUE + r")",
    re.IGNORECASE,
)
_LEADING_OR = re.compile(r"(?P<num>" + _VALUE + r")\s*\}?\s*,?\s+or\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[!?\n]|\.(?!\d)")
#: An answer as committed: its text and where it starts and ends in the reply.
Answer = tuple[str, int, int]
_MAX_BOX_CHARS = 1000


@dataclass(frozen=True)
class MathVerdict:
    """The outcome of checking one reply against one reference."""

    correct: bool
    status: str
    answer: str | None = None


def _boxed(text: str) -> list[Answer]:
    """Contents of every ``\\boxed{...}`` (balanced braces) and their span."""
    found = []
    for match in _BOX.finditer(text):
        depth, start = 1, match.end()
        # A box longer than this is not an answer (and bounds the scan).
        for index in range(start, min(len(text), start + _MAX_BOX_CHARS)):
            depth += {"{": 1, "}": -1}.get(text[index], 0)
            if depth == 0:
                found.append((text[start:index], start, index + 1))
                break
    return found


def _literal_values(text: str) -> list[tuple[Fraction, str, int, int]]:
    values = []
    for match in _LITERAL.finditer(text):
        value = _literal_value(match.group(0))
        if value is not None:
            values.append((value, match.group(0), match.start(), match.end()))
    return values


def _literal_value(text: str) -> Fraction | None:
    try:
        node = parse_expression(normalize_answer_text(text))
    except ExpressionError:
        return None
    return rational_value(node) if is_numeric_literal(node) else None


def extract_final_answers(text: str, question: str = "") -> tuple[str, list[Answer]]:
    """``(source, [(answer text, start, end), ...])`` committed to by ``text``.

    ``source`` is ``"boxed"``, ``"stated"``, ``"last_number"``, or ``"none"``
    with no answers. See the module docs for the order of authority.
    """
    boxed = _boxed(text)
    if boxed:
        return "boxed", boxed
    stated = [
        (m.group(1).strip(), m.start(1), m.end(1))
        for pattern in (_STATED, _HASHES)
        for m in pattern.finditer(text)
    ]
    if stated:
        return "stated", sorted(stated, key=lambda item: item[1])
    quoted = {item[0] for item in _literal_values(question)}
    offered = [item for item in _literal_values(text) if item[0] not in quoted]
    if len({item[0] for item in offered}) == 1:
        _, literal, start, end = offered[-1]
        return "last_number", [(literal, start, end)]
    return "none", []


def _parse_answer(text: str) -> Node | None:
    """An answer's expression, or its leading number ("42 dollars" -> 42)."""
    normalized = normalize_answer_text(text)
    try:
        return parse_expression(normalized)
    except ExpressionError:
        pass
    leading = _LITERAL.match(normalized)
    if leading is None:
        return None
    try:
        return parse_expression(normalize_answer_text(leading.group(0)))
    except ExpressionError:
        return None


def _text_key(text: str) -> str:
    return " ".join(_normalize_text(normalize_answer_text(text)).strip(" .!").split())


def _hedged(text: str, answer: Fraction | None) -> bool:
    """Whether ``text`` offers a number as an alternative or tentatively.

    "41 or 42" and "42, or maybe 41" offer an alternative; "maybe 42" and
    "probably 42" hedge the answer itself. "84 (or 84.0)" restates it.
    """
    for match in _HEDGE.finditer(text):
        value = _expression_value(match.group("num"))
        if value is None:
            continue
        if value != answer or match.group("word").lower() != "or":
            return True
    return any(
        (value := _expression_value(match.group("num"))) is not None and value != answer
        for match in _LEADING_OR.finditer(text)
    )


def _expression_value(text: str) -> Fraction | None:
    try:
        return rational_value(parse_expression(normalize_answer_text(text)))
    except ExpressionError:
        return None


def _sentence_start(text: str, position: int) -> int:
    """Where the sentence holding ``text[position]`` starts."""
    start = 0
    for match in _SENTENCE_END.finditer(text, 0, position):
        start = match.end()
    return start


class _Reference:
    """The reference answer, parsed once."""

    def __init__(self, text: str, symbolic: bool) -> None:
        self.text = text
        self.symbolic = symbolic
        try:
            self.node: Node | None = parse_expression(normalize_answer_text(text))
        except ExpressionError:
            self.node = None
        self.value = rational_value(self.node) if self.node is not None else None
        self.numeric = (
            self.node is not None
            and not free_symbols(self.node)
            and (self.value is not None)
        )

    def matches(self, answer: str, abs_tol: float, rel_tol: float) -> bool:
        if self.node is None:
            return _text_key(answer) == _text_key(self.text)
        node = _parse_answer(answer)
        if node is None:
            return False
        if self.numeric:
            value = rational_value(node) if is_numeric_literal(node) else None
            return value is not None and _close(value, self.value, abs_tol, rel_tol)
        if not self.symbolic:
            return _form_key(answer) == _form_key(self.text)
        return equivalent(node, self.node) is True


def _form_key(text: str) -> str:
    """An expression's written form, ignoring spacing and case."""
    return re.sub(r"\s+", "", normalize_answer_text(text)).lower()


def _close(
    value: Fraction, expected: Fraction | None, abs_tol: float, rel_tol: float
) -> bool:
    if expected is None:
        return False
    if value == expected:
        return True
    difference = abs(float(value - expected))
    return difference <= max(abs_tol, rel_tol * abs(float(expected)))


def verify_math_answer(
    reply: str,
    reference: str,
    *,
    question: str = "",
    abs_tolerance: float = 0.0,
    rel_tolerance: float = 0.0,
    symbolic: bool = True,
) -> MathVerdict:
    """Check one reply against one reference answer (see the module docs)."""
    text = strip_tool_calls(reply)
    if _UNCERTAINTY_PATTERN.search(_normalize_text(text)):
        return MathVerdict(False, "hedged")
    source, answers = extract_final_answers(text, question)
    if not answers:
        return MathVerdict(False, "no_answer")
    if len({_answer_key(answer) for answer, _, _ in answers}) != 1:
        return MathVerdict(False, "multiple_answers", answers[-1][0])
    answer = answers[-1][0]
    # Without an explicit commitment the whole reply is the answer's context.
    region = (
        text
        if source == "last_number"
        else text[_sentence_start(text, answers[0][1]) :]
    )
    if _hedged(region, _committed_value(answer)) or (
        # CorrectnessReward's candidate rule: an unstated answer listed among
        # other candidates ("Paris, 1/2, Glacier") is not committed to.
        source == "last_number"
        and _text_answer_credit(text, answer, question) < 1.0
    ):
        return MathVerdict(False, "hedged", answer)
    if _answer_negated(region, answer):
        return MathVerdict(False, "negated", answer)
    ref = _Reference(str(reference), symbolic)
    correct = ref.matches(answer, abs_tolerance, rel_tolerance)
    return MathVerdict(correct, "correct" if correct else "incorrect", answer)


def _committed_value(answer: str) -> Fraction | None:
    node = _parse_answer(answer)
    return rational_value(node) if node is not None else None


def _answer_key(answer: str) -> Any:
    """Two committed answers agree when their values (or texts) do."""
    node = _parse_answer(answer)
    value = rational_value(node) if node is not None else None
    return value if value is not None else _text_key(answer)


class MathAnswerVerifier(RewardFunction):
    """1.0 when the reply commits to an answer equal to the reference, else 0.0.

    A SPARSE reward: applied once to the finished conversation, it reads the
    newest assistant turn (``turns[-1]`` for a single reply), so the tool
    calls of a tool-use episode are not answers. The reference is the
    context's ``ground_truth`` (a bare value or ``{"answer": ...}``, as for
    ``CorrectnessReward``) or ``answer=``; without one it raises
    ``ConfigurationError``.
    """

    def __init__(
        self,
        weight: float = 1.0,
        *,
        answer: Any = None,
        abs_tolerance: float = 0.0,
        rel_tolerance: float = 0.0,
        symbolic: bool = True,
        name: str = "MathAnswerVerifier",
    ) -> None:
        super().__init__(weight, RewardType.SPARSE, name)
        self.answer = None if answer is None else str(answer)
        self.abs_tolerance = float(abs_tolerance)
        self.rel_tolerance = float(rel_tolerance)
        self.symbolic = bool(symbolic)

    def _reference(self, context: Mapping[str, Any] | None) -> str:
        truth = _ground_truth_mapping((context or {}).get("ground_truth"))
        reference = truth.get("answer", self.answer)
        if reference is None or not str(reference).strip():
            raise ConfigurationError(
                "MathAnswerVerifier needs a reference answer "
                "(context ground_truth or answer=...)"
            )
        return str(reference)

    async def compute_reward(
        self, turns: list[Any], context: dict[str, Any] | None = None
    ) -> RewardResult:
        reference = self._reference(context)
        replies = assistant_turns_in_scope(turns, context)
        if not replies:
            return RewardResult(score=0.0, metadata={"status": "no_reply"})
        reply = replies[-1].content
        question = " ".join(
            str(turn.content)
            for turn in turns
            if getattr(turn, "role", None) == "user" and isinstance(turn.content, str)
        )
        verdict = verify_math_answer(
            reply if isinstance(reply, str) else "",
            reference,
            question=question,
            abs_tolerance=self.abs_tolerance,
            rel_tolerance=self.rel_tolerance,
            symbolic=self.symbolic,
        )
        score = 1.0 if verdict.correct else 0.0
        return RewardResult(
            score=score,
            breakdown={"correct": score},
            metadata={"status": verdict.status, "answer": verdict.answer},
        )
