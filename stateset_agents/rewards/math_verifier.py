"""Verifiable math reward: does the reply commit to the reference answer?

:class:`MathAnswerVerifier` scores 1.0 when the newest assistant reply
commits to a final answer equivalent to the reference, and 0.0 otherwise. A
chain of thought is judged by its **final answer** only:

1. the last ``\\boxed{...}`` (or ``\\fbox{...}``) in the reply;
2. otherwise the last explicit statement: "the answer is X", "answer: X",
   "final answer = X", "the solutions are X", "#### X" (a statement that is
   no value at all, "the answer is consistent", is not one);
3. otherwise the closing sentence, when it concludes with one value and no
   number or question follows it ("So the bottle holds 2500 mL.",
   "Therefore f'(x) = 3x^2 + 2.");
4. otherwise a lone number, only when the reply offers exactly one number
   beyond those quoted from the question (by written form: "2/6" is a step,
   not the question's "1/3"), so worked arithmetic must state its answer
   (for a reference that is a set of solutions, the last sentence may list
   them: "x = 3 or x = -3"). A reply that stops mid-step ("... = 3/6, so")
   has no closing or lone-number answer.

After a ``<think>`` block, the text that follows it is read first. Everything
before the final answer's sentence is reasoning: an admission ("we don't
know the total yet"), a check, or an earlier answer the reply corrects
("\\boxed{160}. Wait... \\boxed{168}") costs nothing. An earlier *different*
answer with no correction cue between ("\\boxed{41} \\boxed{84}") is two
answers and scores 0. A statement corrected afterwards and concluded again
("So the answer is 3x^2 + 2x. Hmm, ... not 2x. So f'(x) = 3x^2 + 2.") is
replaced by the new conclusion; a box followed by a correction to another
value is contradicted. Tool-call markup is removed first, so the arguments
of a tool call are never read as an answer.

**No credit for hedging the final answer**, checked on the answer itself,
its own sentence and the text after it: an answer that hedges inside
(``\\boxed{84, 85}``, ``\\boxed{84 \\pm 1}``, ``\\boxed{84 \\text{ or } 85}``,
``\\approx``, "about", "84 or so"), an alternative offered ("85 or
\\boxed{84}", "84, or maybe 85", "85 is also possible"), a tentative,
approximate or doubted answer ("Maybe 84", "\\boxed{84} (approximately)",
"I'm not sure, but the answer is 84", "... but I might be wrong"), an answer
posed as a question ("Or is it \\boxed{84}?"), a negated one ("It is not
84", "\\boxed{84} is wrong"), and an answer taken back or replaced
afterwards ("I made a mistake", "On second thought, 83.", "it could also be
85", "some textbooks say 85") all score 0. A restatement or a
check after it ("It is 7 * 12.", "84 / 7 = 12", "I hope it is correct")
keeps the credit, and so does a unit or a restatement inside it ("84
dollars", ``\\boxed{168\\,\\text{pens}}``, "84 (or 84.0)").

In a tool-use episode the reply must also *be* the final answer: when the
context records how the episode ended (``termination``, set by
``ToolUseEnvironment``), anything but ``"final_answer"`` (a spent budget,
``max_turns``) scores 0, whatever the last turn says.

Equivalence (:mod:`stateset_agents.core.math_expressions`, which never calls
``eval``; LaTeX and markdown such as ``\\dfrac``, ``\\frac12``, ``\\left(``,
``\\,``, ``\\text{}``, ``^{...}``, ``**bold**`` and ``$...$`` are read): a
numeric reference must be matched by a number *written out* (``84``,
``-3/4``, ``0.75``, ``\\frac{3}{4}``; a percentage ``50\\%`` reads as 50 or
0.5), compared exactly as a fraction, or within
``abs_tolerance``/``rel_tolerance`` when set; restating the question's
arithmetic (``\\boxed{12*7}``) earns nothing, unless the reference is itself
written that way (``2^{10}``). A reference with ``\\pm`` or a list of
solutions (``\\pm 3``, ``1, 2``) is matched by the same set in any order and
notation (``\\pm 3``, ``3, -3``, ``x = 3 or x = -3``). A symbolic reference
(``2x+2``) is matched by any equivalent expression (exact evaluation at
pseudo-random rational points seeded from both expressions, or
``sympy.simplify`` for functions such as ``sqrt`` when sympy is installed,
in a worker process killed after ``sympy_timeout`` seconds; a timeout scores
0); this cannot tell an expanded form from an unexpanded one, so pass
``symbolic=False`` for form-sensitive tasks (expand, factor, simplify) to
require the same canonical text. A reference that is not math is compared as
normalized text.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from stateset_agents.core.basic_rewards import _ground_truth_mapping
from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.math_expressions import (
    SYMPY_TIMEOUT_SECONDS,
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
from stateset_agents.core.reward_context import (
    assistant_turns_in_scope,
    unanswered_termination,
)
from stateset_agents.core.tool_calls import strip_tool_calls

from ._math_reading import (
    CORRECTION,
    LITERAL,
    OFFER,
    Answer,
    answer_core,
    boxed_answers,
    closing_statements,
    corrected_value_text,
    hedged_after,
    negated_after,
    negated_before,
    retracted,
    sentence_start,
    stated_answers,
    tentative_before,
    uncertain,
)

__all__ = [
    "MathAnswerVerifier",
    "MathVerdict",
    "extract_final_answers",
    "verify_math_answer",
]

#: A number, or arithmetic over numbers ("1/2 + 1"), offered as a value.
_VALUE = LITERAL.pattern + r"(?:\s*[-+*/^]\s*" + LITERAL.pattern + r")*"
#: Connectives that offer a number as one of several answers.
_HEDGE = re.compile(
    r"\b(?P<word>or|maybe|perhaps|possibly|probably|alternatively|otherwise)\b"
    r"(?:\s+(?:it|it's|its|is|could|might|may|be|was|the|answer|even))*"
    r"\s*(?:\\boxed\{\s*)?\$?(?P<num>" + _VALUE + r")",
    re.IGNORECASE,
)
_LEADING_OR = re.compile(r"(?P<num>" + _VALUE + r")\s*\}?\s*,?\s+or\b", re.IGNORECASE)
_TRAILING_OR = re.compile(
    r"(?P<num>" + _VALUE + r")\s*\$?\s*,?\s+or\s*$", re.IGNORECASE
)
#: Marks an answer as a range: ``84 \\pm 1``.
_PLUS_MINUS = re.compile(r"\\pm|\\mp|\u00b1|\u2213|\+/?-")
#: Marks an answer as approximate: ``\\approx 84``, ``~84``.
_APPROXIMATE = re.compile(
    r"\\approx|\u2248|\\simeq|\\sim(?![a-z])|(?:^|[\s(=$])~\s*[-+]?\d"
)
#: Words that make an answer tentative or approximate.
_HEDGE_WORD = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|either|approximately|approx|about"
    r"|around|roughly|nearly|almost|circa|ish"
    r"|or\s+(?:so|thereabouts|more|less|fewer|greater|higher|lower))\b",
    re.IGNORECASE,
)
_OR = re.compile(r"\bor\b", re.IGNORECASE)
_LIST_SEPARATOR = re.compile(r";|,(?!\d{3}(?!\d))")
_OFFER_END = re.compile(r"[;!?\n,]|\.(?!\d)|\s\(")
_OFFER_FILLER = re.compile(
    r"^(?:\s*(?:actually|really|instead|in\s+fact|also|then|just|now|:))*\s*"
    r"|(?:\s*(?:instead|then|actually|after\s+all|really|though|too))+\s*$",
    re.IGNORECASE,
)
_BOX_OPENING = re.compile(r"\\(?:boxed|fbox)\s*\{\s*$")
#: A labelled solution: ``x = 3``, ``x_1 = 3``.
_SOLUTION_LABEL = re.compile(r"^\s*[A-Za-z](?:_\{?\w+\}?)?\s*=\s*")
_SET_SPLIT = re.compile(r"\s*(?:;|,(?!\d{3}(?!\d))|\bor\b|\band\b)\s*", re.IGNORECASE)
_SET_BRACES = re.compile(r"^\s*(?:\\\{|\{)\s*(.*?)\s*(?:\\\}|\})\s*$")
_TUPLE = re.compile(r"^\s*[(\[].*[)\]]\s*$")
_CONCLUSION_LEAD = re.compile(
    r"^\s*(?:(?:so|therefore|thus|hence|finally|the\s+solutions?\s+(?:are|is))"
    r"\b[\s,:]*)+",
    re.IGNORECASE,
)
_UNITS = re.compile(r"[\sA-Za-z().=/^]*")
#: ``x = 4 \text{ or } x = -4``: a connective written as text.
_TEXT_CONNECTIVE = re.compile(
    r"\\(?:text|textrm|mathrm|mbox)\s*\{\s*(or|and)\s*\}", re.IGNORECASE
)
#: The answer's sentence is a question.
_ASKED = re.compile(r"(?:[^.!?\n]|\.\d)*\?")
#: A reply cut off mid-step: it ends on a connective, an operator or a comma.
_UNFINISHED = re.compile(
    r"(?:\b(?:so|then|thus|therefore|hence|and|or|is|are|equals|of|by|to|the)"
    r"|[=+\-*/\u00d7,:(])\s*$",
    re.IGNORECASE,
)
#: After a concluding statement, a reply may close without another value.
_CLOSING_REMAINDER = re.compile(r"[^\d?]*")
#: Words around a short list item that do not make it a candidate answer.
_ITEM_FILLER = frozenset(
    "so then thus therefore it is it's its could might may be maybe perhaps "
    "possibly probably the a an answer i think guess or and either well ok okay "
    "yes no".split()
)
_ITEM_SPLIT = re.compile(r"[,;|\n]|\.\s|\s+(?:or|and)\s+", re.IGNORECASE)
#: After an answer: a word offering what follows instead of it, or as well.
_ALTERNATIVE_WORD = re.compile(
    r"\b(?P<word>or|maybe|perhaps|possibly|probably|alternatively)\b", re.IGNORECASE
)
_ALTERNATIVE_FILLER = re.compile(
    r"^(?:\s|\b(?:it|it's|its|is|could|might|may|be|was|the|answer|even|maybe"
    r"|perhaps|possibly|probably|also)\b)*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MathVerdict:
    """The outcome of checking one reply against one reference."""

    correct: bool
    status: str
    answer: str | None = None


# --- values of an answer --------------------------------------------------------


def _literal_values(text: str) -> list[tuple[Fraction, str, int, int]]:
    values = []
    for match in LITERAL.finditer(text):
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


def _plain(text: str) -> str:
    """An answer's text as the parser reads it, without a percent sign."""
    return normalize_answer_text(answer_core(text)).rstrip().rstrip("%").strip()


def _is_percent(text: str) -> bool:
    return normalize_answer_text(answer_core(text)).rstrip().endswith("%")


def _parse_answer(text: str) -> Node | None:
    """An answer's expression, or its leading number when the rest is only a
    unit or a restatement ("42 dollars", "84 (or 84.0)" -> the number; "84,
    85" and "84 \\pm 1" -> no answer)."""
    normalized = _plain(text)
    try:
        return parse_expression(normalized)
    except ExpressionError:
        pass
    leading = LITERAL.match(normalized)
    if leading is None:
        return None
    try:
        node = parse_expression(normalize_answer_text(leading.group(0)))
    except ExpressionError:
        return None
    if not _restates_only(normalized[leading.end() :], rational_value(node)):
        return None
    return node


def _restates_only(rest: str, value: Fraction | None) -> bool:
    """Whether ``rest`` adds only words (a unit) or the same number again."""
    if _PLUS_MINUS.search(rest) or _APPROXIMATE.search(rest):
        return False
    if _HEDGE_WORD.search(rest) or _LIST_SEPARATOR.search(rest):
        return False
    if any(other != value for other, _, _, _ in _literal_values(rest)):
        return False
    return _UNITS.fullmatch(LITERAL.sub(" ", rest)) is not None


def _value_forms(text: str, node: Node) -> list[Fraction]:
    """The values an answer may mean: ``50%`` is 50 or 0.5."""
    value = rational_value(node)
    if value is None:
        return []
    return [value, value / 100] if _is_percent(text) else [value]


def _same_value(first: Node, second: Node, sympy_timeout: float | None) -> bool:
    a, b = rational_value(first), rational_value(second)
    if a is not None and b is not None:
        return a == b
    return equivalent(first, second, timeout=sympy_timeout) is True


def _text_key(text: str) -> str:
    plain = normalize_answer_text(answer_core(text)).lower()
    return re.sub(r"\s+", "", plain.replace("\u2019", "'")).strip(".!")


def _form_key(text: str) -> str:
    """An expression's written form, ignoring spacing and case."""
    return re.sub(r"\s+", "", _plain(text)).lower()


# --- solution sets ---------------------------------------------------------------


def _is_solution_set(text: str) -> bool:
    """A reference listing solutions: ``\\pm 3``, ``1, 2``, ``2 or 3``; not a
    tuple or an interval (``(1, 2)``, ``[1, 2)``)."""
    core = answer_core(text)
    if _PLUS_MINUS.search(core):
        return True
    if _TUPLE.match(core):
        return False
    body = _SET_BRACES.sub(r"\1", core)
    return len([part for part in _SET_SPLIT.split(body) if part.strip()]) > 1


def _solutions(text: str) -> list[Node] | None:
    """The solutions a reply lists (``\\pm`` expanded), or ``None``."""
    body = _TEXT_CONNECTIVE.sub(r" \1 ", answer_core(text).replace("$", " "))
    body = _SET_BRACES.sub(r"\1", body)
    nodes: list[Node] = []
    for part in _SET_SPLIT.split(body):
        part = _SOLUTION_LABEL.sub("", part.strip())
        if not part:
            continue
        sign = _PLUS_MINUS.search(part)
        if sign is None:
            node = _parse_answer(part)
            if node is None:
                return None
            nodes.append(node)
            continue
        base = (
            _parse_answer(part[: sign.start()])
            if part[: sign.start()].strip()
            else None
        )
        spread = _parse_answer(part[sign.end() :])
        if spread is None:
            return None
        origin = base if base is not None else ("num", Fraction(0))
        nodes.extend([("add", origin, spread), ("sub", origin, spread)])
    return nodes or None


def _same_solutions(
    answer: list[Node], reference: list[Node], sympy_timeout: float | None
) -> bool:
    """The same solutions, in any order (duplicates ignored)."""
    unmatched = _distinct(reference, sympy_timeout)
    for node in _distinct(answer, sympy_timeout):
        found = next(
            (
                i
                for i, ref in enumerate(unmatched)
                if _same_value(node, ref, sympy_timeout)
            ),
            None,
        )
        if found is None:
            return False
        unmatched.pop(found)
    return not unmatched


def _distinct(nodes: list[Node], sympy_timeout: float | None) -> list[Node]:
    kept: list[Node] = []
    for node in nodes:
        if not any(_same_value(node, other, sympy_timeout) for other in kept):
            kept.append(node)
    return kept


# --- the reference ---------------------------------------------------------------


class _Reference:
    """The reference answer, parsed once."""

    def __init__(self, text: str, symbolic: bool) -> None:
        self.text = text
        self.symbolic = symbolic
        self.solutions = _solutions(text) if _is_solution_set(text) else None
        self.node = _parse_answer(text)
        self.value = rational_value(self.node) if self.node is not None else None
        self.numeric = (
            self.node is not None
            and not free_symbols(self.node)
            and (self.value is not None)
        )
        self.literal = (
            self.numeric and self.node is not None and is_numeric_literal(self.node)
        )
        self.values = (
            _value_forms(text, self.node)
            if self.numeric and self.node is not None
            else []
        )
        self.plain = _plain(text)

    @property
    def is_set(self) -> bool:
        return self.solutions is not None

    @property
    def is_math(self) -> bool:
        return self.node is not None or self.solutions is not None

    def allows(self, pattern: re.Pattern[str]) -> bool:
        """Whether the reference itself contains ``pattern`` (a list, "or")."""
        return pattern.search(self.plain) is not None or self.is_set

    def answers(self, text: str) -> bool:
        """Whether a statement offers a value at all: for a math reference,
        "the answer is consistent" (no number, no expression) does not."""
        if not self.is_math:
            return bool(_text_key(text))
        if _parse_answer(text) is not None or re.search(r"\d", text):
            return True
        return self.is_set and _solutions(text) is not None

    def matches(
        self,
        answer: str,
        abs_tol: float,
        rel_tol: float,
        sympy_timeout: float | None = None,
    ) -> bool:
        if self.solutions is not None:
            listed = _solutions(answer)
            return listed is not None and _same_solutions(
                listed, self.solutions, sympy_timeout
            )
        if self.node is None:
            return _text_key(answer) == _text_key(self.text)
        node = _parse_answer(answer)
        if node is None:
            return False
        if self.numeric:
            if is_numeric_literal(node):
                return any(
                    _close(value, expected, abs_tol, rel_tol)
                    for value in _value_forms(answer, node)
                    for expected in self.values
                )
            # A reference written as an expression (2^{10}) may be restated.
            return not self.literal and _form_key(answer) == _form_key(self.text)
        if not self.symbolic:
            return _form_key(answer) == _form_key(self.text)
        return equivalent(node, self.node, timeout=sympy_timeout) is True


def _close(value: Fraction, expected: Fraction, abs_tol: float, rel_tol: float) -> bool:
    if value == expected:
        return True
    difference = abs(float(value - expected))
    return difference <= max(abs_tol, rel_tol * abs(float(expected)))


# --- finding the final answer ------------------------------------------------------


def extract_final_answers(text: str, question: str = "") -> tuple[str, list[Answer]]:
    """``(source, [(answer text, start, end), ...])`` committed to by ``text``.

    ``source`` is ``"boxed"``, ``"stated"``, ``"concluded"``,
    ``"last_number"``, or ``"none"`` with no answers; the final answer is
    the last one. See the module docs for the order of authority.
    """
    return _final_answers(text, question, None)


_THINK_END = "</think>"


def _read_reply(
    text: str, question: str, reference: _Reference
) -> tuple[str, str, list[Answer]]:
    """``(answer region, source, answers)``: after a ``<think>`` block, the
    text that follows it when it holds an answer, else the whole reply."""
    cut = text.rfind(_THINK_END)
    if cut >= 0:
        region = text[cut + len(_THINK_END) :]
        source, answers = _final_answers(region, question, reference)
        if answers:
            return region, source, answers
    source, answers = _final_answers(text, question, reference)
    return text, source, answers


def _final_answers(
    text: str, question: str, reference: _Reference | None
) -> tuple[str, list[Answer]]:
    boxed = boxed_answers(text)
    if boxed:
        return "boxed", boxed
    stated = [
        item
        for item in stated_answers(text)
        if reference is None or reference.answers(item[0])
    ]
    if stated:
        return "stated", stated
    concluded = _concluded(text, reference)
    if concluded is not None:
        return "concluded", [concluded]
    if reference is not None and reference.is_set:
        listed = _listed_solutions(text)
        if listed is not None:
            return "solutions", [listed]
    if _unfinished(text):
        return "none", []  # cut off mid-step ("... = 3/6, so")
    # Quoted by its written form: "2/6" is a step, not the question's "1/3".
    quoted = {_written(item[1]) for item in _literal_values(question)}
    offered = [
        item for item in _literal_values(text) if _written(item[1]) not in quoted
    ]
    if len({item[0] for item in offered}) == 1:
        _, literal, start, end = offered[-1]
        return "last_number", [(literal, start, end)]
    return "none", []


def _unfinished(text: str) -> bool:
    """Whether the reply stops mid-step (trailing markdown aside)."""
    return _UNFINISHED.search(text.rstrip().rstrip("*_`").rstrip()) is not None


def _written(literal: str) -> str:
    return re.sub(r"[\s,]", "", literal)


def _closing_value(
    body: str, start: int, reference: _Reference | None
) -> Answer | None:
    """The value a concluding clause gives: the clause itself when it is an
    answer ("f'(x) = 3x^2 + 2", "2.5 * 1000 = 2500 mL", "x = 3 or x = -3"),
    else its only number ("the bottle holds 2500 mL")."""
    end = start + len(body)
    if _parse_answer(body) is not None:
        return body, start, end
    if reference is not None and reference.is_set and _solutions(body) is not None:
        return body, start, end
    values = _literal_values(body)
    if len({item[0] for item in values}) != 1:
        return None
    _, literal, lit_start, lit_end = values[-1]
    return literal, start + lit_start, start + lit_end


def _concluded(text: str, reference: _Reference | None) -> Answer | None:
    """The reply's closing sentence, when it concludes with a value and
    nothing after it offers another number or asks a question."""
    closings = closing_statements(text)
    if not closings or _unfinished(text):
        return None
    body, start, end = closings[-1]
    if _CLOSING_REMAINDER.fullmatch(text[end:]) is None:
        return None
    return _closing_value(body, start, reference)


def _replacement(
    text: str, final: Answer, reference: _Reference | None
) -> Answer | None:
    """A value the reply concludes with after correcting its final answer
    ("So the answer is 3x^2 + 2x. Hmm, ... not 2x. So f'(x) = 3x^2 + 2.")."""
    cue = CORRECTION.search(text, final[2])
    if cue is None:
        return None
    for body, start, _ in reversed(closing_statements(text)):
        if start < cue.end() or "=" not in body:
            continue
        value = _closing_value(body, start, reference)
        if value is not None:
            return value
    return None


def _listed_solutions(text: str) -> Answer | None:
    """The last sentence, when it lists solutions ("So x = 3 or x = -3.")."""
    body = text.rstrip().rstrip(".").rstrip()
    start = sentence_start(body, len(body))
    sentence = body[start:]
    lead = _CONCLUSION_LEAD.match(sentence)
    offset = lead.end() if lead else 0
    candidate = sentence[offset:]
    if _solutions(candidate) is None:
        return None
    return candidate, start + offset, start + len(sentence)


# --- is the final answer committed to? ---------------------------------------------


def _answer_hedged(answer: str, reference: _Reference, boxed: bool) -> bool:
    """Whether a box or stated answer hedges inside itself.

    ``\\approx`` and words such as "about" or "or so" make it approximate;
    ``\\pm``, a comma or semicolon (a list) and "or" (in a box; in a
    statement, "or" with another value) offer several answers, unless the
    reference is itself such a set (``\\pm 3``, ``1, 2``).
    """
    core = answer_core(answer)
    plain = normalize_answer_text(core)
    if _APPROXIMATE.search(core) or _APPROXIMATE.search(plain):
        return True
    if not reference.is_set and (_PLUS_MINUS.search(core) or _PLUS_MINUS.search(plain)):
        return True
    if not reference.allows(_LIST_SEPARATOR) and _LIST_SEPARATOR.search(plain):
        return True
    if _HEDGE_WORD.search(plain) and not _HEDGE_WORD.search(reference.plain):
        return True
    if reference.allows(_OR) or not _OR.search(plain):
        return False
    # A box holds the answer alone. In a statement, "84 (or 84.0)" restates
    # the answer and "84 or 85" offers another.
    return boxed or _alternative_offered(plain, _committed_value(answer))


def _committed_value(answer: str) -> Fraction | None:
    node = _parse_answer(answer)
    return rational_value(node) if node is not None else None


def _expression_value(text: str) -> Fraction | None:
    try:
        return rational_value(parse_expression(normalize_answer_text(text)))
    except ExpressionError:
        return None


def _alternative_offered(text: str, answer: Fraction | None) -> bool:
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


def _conflicting_earlier_answer(
    text: str, answers: list[Answer], sympy_timeout: float | None
) -> bool:
    """An earlier answer different from the final one, with no correction
    between them ("\\boxed{41} \\boxed{84}"); a corrected one is reasoning."""
    final, final_start, _ = answers[-1]
    final_node = _parse_answer(final)
    for earlier, _, end in answers[:-1]:
        node = _parse_answer(earlier)
        if node is not None and final_node is not None:
            same = _same_value(node, final_node, sympy_timeout)
        else:
            same = _text_key(earlier) == _text_key(final)
        if not same and not CORRECTION.search(text[end:final_start]):
            return True
    return False


def _before_problem(before: str, answer: str) -> str | None:
    """Doubt, an alternative or a denial in the answer's own sentence."""
    if uncertain(before):
        return "hedged"
    if tentative_before(before):
        return "hedged"
    trailing = _TRAILING_OR.search(before)
    if trailing is not None and _expression_value(trailing.group("num")) != (
        _committed_value(answer)
    ):
        return "hedged"
    if negated_before(before):
        return "negated"
    return None


def _after_problem(after: str, answer: str, sympy_timeout: float | None) -> str | None:
    """Doubt, an alternative, a denial or a replacement after the answer."""
    if uncertain(after) or hedged_after(after):
        return "hedged"
    if _alternative_offered(after, _committed_value(answer)):
        return "hedged"
    if _alternative_expression(after, answer, sympy_timeout):
        return "hedged"
    if negated_after(after):
        return "negated"
    if retracted(after):
        return "contradicted"
    committed = _parse_answer(answer)
    if committed is None:
        return None  # a text answer: values cannot be compared
    offered = [_parse_answer(clause) for clause in corrected_value_text(after)]
    offered += [_offered_value(after[match.end() :]) for match in OFFER.finditer(after)]
    replaced = _replacement(after, (answer, 0, 0), None)
    if replaced is not None:
        offered.append(_parse_answer(replaced[0]))
    if any(
        node is not None and not _same_value(node, committed, sympy_timeout)
        for node in offered
    ):
        return "contradicted"
    return None


def _alternative_expression(
    after: str, answer: str, sympy_timeout: float | None
) -> bool:
    """ "... or maybe 3x^2 + 12" right after an answer: another expression
    offered (or the answer itself offered tentatively)."""
    committed = _parse_answer(answer)
    for match in _ALTERNATIVE_WORD.finditer(after):
        rest = after[match.end() :]
        end = _OFFER_END.search(rest)
        clause = _ALTERNATIVE_FILLER.sub("", rest[: end.start()] if end else rest)
        node = _parse_answer(clause.strip()) if clause.strip() else None
        if node is None:
            continue
        if match.group("word").lower() != "or" or committed is None:
            return True
        if not _same_value(node, committed, sympy_timeout):
            return True
    return False


def _offered_value(text: str) -> Node | None:
    """The expression a phrase offers, when the rest of its clause is one."""
    end = _OFFER_END.search(text)
    clause = _OFFER_FILLER.sub("", text[: end.start()] if end else text)
    clause = clause.strip().lstrip("$").rstrip("$").strip()
    while clause.endswith((")", "]", "}")) and clause.count("(") < clause.count(")"):
        clause = clause[:-1].rstrip()
    if not clause:
        return None
    return _parse_answer(clause)


def _listed_among_candidates(text: str, answer: str) -> bool:
    """A lone number listed among other short candidate items ("Parsnip,
    84, Glacier", "It could be Parsnip or 84 or Glacier.")."""
    competing = 0
    for item in _ITEM_SPLIT.split(text):
        if not item or answer in item:
            continue
        words = [
            word
            for word in re.findall(r"[a-z0-9']+", item.lower())
            if word not in _ITEM_FILLER
        ]
        if 0 < len(words) <= 3:
            competing += 1
    return competing >= 2


def _commitment_problem(
    text: str,
    source: str,
    answers: list[Answer],
    reference: _Reference,
    sympy_timeout: float | None,
) -> str | None:
    """Why the final answer is not committed to, or ``None`` when it is."""
    answer, start, end = answers[-1]
    if source in ("boxed", "stated") and _answer_hedged(
        answer, reference, source == "boxed"
    ):
        return "hedged"
    if _conflicting_earlier_answer(text, answers, sympy_timeout):
        return "multiple_answers"
    if source == "last_number":
        if _listed_among_candidates(text, answer):
            return "hedged"
        before = text[:start]
    else:
        before = _BOX_OPENING.sub("", text[sentence_start(text, start) : start])
    if _ASKED.match(text, end):
        return "hedged"  # "Or is it \\boxed{85}?" asks; it does not commit
    return _before_problem(before, answer) or _after_problem(
        text[end:], answer, sympy_timeout
    )


def verify_math_answer(
    reply: str,
    reference: str,
    *,
    question: str = "",
    abs_tolerance: float = 0.0,
    rel_tolerance: float = 0.0,
    symbolic: bool = True,
    sympy_timeout: float | None = None,
) -> MathVerdict:
    """Check one reply against one reference answer (see the module docs).

    ``sympy_timeout`` bounds the seconds sympy may spend on an equivalence
    (default ``math_expressions.SYMPY_TIMEOUT_SECONDS``); a timeout is
    incorrect.
    """
    ref = _Reference(str(reference), symbolic)
    text, source, answers = _read_reply(strip_tool_calls(reply), question, ref)
    if not answers:
        return MathVerdict(False, "no_answer")
    replacement = _replacement(text, answers[-1], ref)
    if replacement is not None and source != "boxed":
        # The last statement was corrected: the corrected value is final.
        answers = [*answers, replacement]
    answer = answers[-1][0]
    problem = _commitment_problem(text, source, answers, ref, sympy_timeout)
    if problem is not None:
        return MathVerdict(False, problem, answer)
    correct = ref.matches(answer, abs_tolerance, rel_tolerance, sympy_timeout)
    return MathVerdict(correct, "correct" if correct else "incorrect", answer)


class MathAnswerVerifier(RewardFunction):
    """1.0 when the reply commits to an answer equal to the reference, else 0.0.

    A SPARSE reward: applied once to the finished conversation, it reads the
    newest assistant turn (``turns[-1]`` for a single reply), so the tool
    calls of a tool-use episode are not answers. The reference is the
    context's ``ground_truth`` (a bare value or ``{"answer": ...}``, as for
    ``CorrectnessReward``) or ``answer=``; without one it raises
    ``ConfigurationError``. An episode whose context records that it ended
    without a final answer (``termination`` other than ``"final_answer"``)
    scores 0. Verification runs in a worker thread, so a slow check never
    blocks the event loop, and sympy work is cut off after ``sympy_timeout``
    seconds (scored 0).
    """

    score_range = (0.0, 1.0)

    def __init__(
        self,
        weight: float = 1.0,
        *,
        answer: Any = None,
        abs_tolerance: float = 0.0,
        rel_tolerance: float = 0.0,
        symbolic: bool = True,
        sympy_timeout: float = SYMPY_TIMEOUT_SECONDS,
        name: str = "MathAnswerVerifier",
    ) -> None:
        super().__init__(weight, RewardType.SPARSE, name)
        self.answer = None if answer is None else str(answer)
        self.abs_tolerance = float(abs_tolerance)
        self.rel_tolerance = float(rel_tolerance)
        self.symbolic = bool(symbolic)
        self.sympy_timeout = float(sympy_timeout)
        if not self.sympy_timeout > 0:
            raise ConfigurationError("sympy_timeout must be positive")

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
        termination = unanswered_termination(context)
        if termination is not None:
            return RewardResult(
                score=0.0,
                breakdown={"correct": 0.0},
                metadata={"status": "no_final_answer", "termination": termination},
            )
        replies = assistant_turns_in_scope(turns, context)
        if not replies:
            return RewardResult(score=0.0, metadata={"status": "no_reply"})
        reply = replies[-1].content
        question = " ".join(
            str(turn.content)
            for turn in turns
            if getattr(turn, "role", None) == "user" and isinstance(turn.content, str)
        )
        # Off the event loop: parsing, exact checks and the sympy round trip.
        verdict = await asyncio.to_thread(
            verify_math_answer,
            reply if isinstance(reply, str) else "",
            reference,
            question=question,
            abs_tolerance=self.abs_tolerance,
            rel_tolerance=self.rel_tolerance,
            symbolic=self.symbolic,
            sympy_timeout=self.sympy_timeout,
        )
        score = 1.0 if verdict.correct else 0.0
        return RewardResult(
            score=score,
            breakdown={"correct": score},
            metadata={"status": verdict.status, "answer": verdict.answer},
        )
