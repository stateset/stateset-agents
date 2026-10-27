"""Reading the final answer of a chain-of-thought reply (for the math verifier).

Self-contained text rules, deliberately independent of
:class:`~stateset_agents.core.basic_rewards.CorrectnessReward`'s lexical
helpers: a chain of thought is judged by its *final answer region* only.

* **Where answers are.** Every ``\\boxed{...}``/``\\fbox{...}`` (balanced
  braces); otherwise every explicit statement ("the answer is X", "answer:
  X", "final answer = X", "#### X"); otherwise a lone number (see
  :mod:`~stateset_agents.rewards.math_verifier`).
* **What an answer says.** :func:`answer_core` peels markup a reply puts
  around a value: markdown bold, a nested "The final answer is", filler such
  as "therefore", and an equation chain over numbers ("7 * 24 = 168" says
  168).
* **Cues around it.** :data:`CORRECTION` marks a self-correction (an earlier
  answer followed by "wait", "actually", "I made a mistake" is reasoning,
  not a second answer); :func:`retracted`, :func:`uncertain`,
  :func:`negated_after`, :func:`corrected_value` and :data:`OFFER` find a
  final answer taken back, doubted, denied or replaced *after* it is given.
"""

from __future__ import annotations

import re

__all__ = [
    "Answer",
    "CORRECTION",
    "LITERAL",
    "OFFER",
    "answer_core",
    "boxed_answers",
    "closing_statements",
    "corrected_value_text",
    "hedged_after",
    "negated_after",
    "negated_before",
    "retracted",
    "sentence_start",
    "stated_answers",
    "tentative_before",
    "uncertain",
]

#: A number as a reply writes it (``-3``, ``1,299``, ``0.75``).
NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
#: A number standing alone, fractions included (``-3/4``).
LITERAL = re.compile(r"(?<![\w.])" + NUMBER + r"(?:\s*/\s*\d+)?(?![\w]|\.\d)")
#: An answer as committed: its text and where it starts and ends in the reply.
Answer = tuple[str, int, int]

_BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")
_MAX_BOX_CHARS = 1000
_STATED = re.compile(
    r"\b(?:final\s+answer|answer|solutions?|roots?)(?:\s*\*\*)?\s*"
    r"(?:is|are|:|=|was|were|would\s+be|will\s+be|should\s+be)\s*(?:\*\*)?\s*:?"
    r"\s*(.+?)"
    r"(?=\.\s|\.\**\s*$|\.\*\*|[\n;]|\s+(?:because|since|as)\b"
    r"|,\s*(?:which|that|because|since|so|as|where|i\.e\.)\b|$)",
    re.IGNORECASE,
)
_HASHES = re.compile(r"####\s*([^\n]+)")
_SENTENCE_END = re.compile(r"[!?\n]|\.(?!\d)")

# --- answer markup --------------------------------------------------------------

_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_NESTED_STATEMENT = re.compile(
    r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*:?\s*", re.IGNORECASE
)
_LEADING_FILLER = re.compile(
    r"^(?:(?:therefore|thus|so|hence|then|simply|just|clearly|exactly|indeed"
    r"|now|that|actually|finally)\b[\s,:]*)+",
    re.IGNORECASE,
)
#: A relation that is not an equation chain's ``=``.
_RELATION = re.compile(r"[<>!]=|==|\\neq|\\le|\\ge|\\ne\b")
#: ``x``, ``x_1``, ``f(3)``: the left side of an answer's ``lhs = value``.
_VARIABLE_VALUE = re.compile(r"[A-Za-z]'*(?:_\{?\w+\}?)?'*(?:\([^()]*\))?")
_LATEX_WORD = re.compile(r"\\[A-Za-z]+")


def _strip_markup(text: str) -> str:
    text = _BOLD.sub(r"\2", text).replace("`", " ")
    return text.strip().strip("*_").strip()


def answer_core(text: str) -> str:
    """The value an answer states, without the markup around it.

    ``**168**`` -> ``168``; "The final answer is $168$" -> ``$168$``;
    "therefore 168" -> ``168``; "7 * 24 = 168" -> ``168``. An equation whose
    left side is an expression in variables (``x^2 + y^2 = 25``) is kept.
    """
    text = _strip_markup(text)
    for _ in range(3):
        stripped = _LEADING_FILLER.sub("", _NESTED_STATEMENT.sub("", text))
        stripped = _strip_markup(stripped)
        if stripped == text:
            break
        text = stripped
    if "=" in text and not _RELATION.search(text):
        left, _, right = text.rpartition("=")
        if right.strip() and _names_a_value(left):
            return right.strip()
    return text


def _names_a_value(left: str) -> bool:
    """Whether ``left = value`` gives a value: the left side is a variable or
    function value (``x``, ``f(3)``), or arithmetic over numbers, possibly
    with unit words (``7 boxes * 24``), but no other variable (``2x + 3``)."""
    side = _LATEX_WORD.sub(" ", left.split("=")[-1]).replace("$", " ").strip()
    if not side:
        return False
    if _VARIABLE_VALUE.fullmatch(side):
        return True
    words = re.findall(r"[A-Za-z]+", side)
    return bool(re.search(r"\d", side)) and all(len(word) > 1 for word in words)


# --- where answers are ----------------------------------------------------------


def boxed_answers(text: str) -> list[Answer]:
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


def stated_answers(text: str) -> list[Answer]:
    """Every explicit statement of an answer, in reply order."""
    stated = [
        (m.group(1).strip(), m.start(1), m.end(1))
        for pattern in (_STATED, _HASHES)
        for m in pattern.finditer(text)
        if m.group(1).strip()
    ]
    return sorted(stated, key=lambda item: item[1])


#: Clause boundaries for concluding statements (not a decimal point, not a
#: thousands comma); a comma only before "so"/"therefore"/"thus"/"hence".
_CLAUSE_SPLIT = re.compile(
    r"[!?\n;\u2014]|\.(?!\d)|,(?=\s*(?:so|therefore|thus|hence)\b)", re.IGNORECASE
)
_CLOSING_LEAD = re.compile(
    r"[\s>*_]*(?:so|therefore|thus|hence|in\s+total|altogether|in\s+conclusion"
    r"|overall|in\s+all)\b[\s,:*]*",
    re.IGNORECASE,
)
_CLOSING_TAIL = re.compile(
    r",\s*(?:which|that|because|since|so|as|where|i\.e\.)\b|\s+(?:because|since)\b",
    re.IGNORECASE,
)


def closing_statements(text: str) -> list[Answer]:
    """Concluding clauses and their spans: "So the bottle holds 2500 mL",
    ", so x = 4", "Therefore f'(x) = 3x^2 + 2" (the text after the lead)."""
    found = []
    start = 0
    bounds = [(m.start(), m.end()) for m in _CLAUSE_SPLIT.finditer(text)]
    for stop, resume in [*bounds, (len(text), len(text))]:
        lead = _CLOSING_LEAD.match(text, start, stop)
        if lead is not None:
            body_end = stop
            tail = _CLOSING_TAIL.search(text, lead.end(), stop)
            if tail is not None:
                body_end = tail.start()
            body = text[lead.end() : body_end]
            if body.strip():
                offset = len(body) - len(body.lstrip())
                found.append(
                    (body.strip(), lead.end() + offset, lead.end() + len(body.rstrip()))
                )
        start = resume
    return found


def sentence_start(text: str, position: int) -> int:
    """Where the sentence holding ``text[position]`` starts."""
    start = 0
    for match in _SENTENCE_END.finditer(text, 0, position):
        start = match.end()
    return start


# --- cues -----------------------------------------------------------------------

#: A self-correction between two answers: the earlier one was reasoning.
CORRECTION = re.compile(
    r"\b(?:wait|actually|hmm+|oops|correction|on\s+second\s+thought"
    r"|scratch\s+that|i\s+mean|i\s+meant|instead|rather|mistake|mistaken|error"
    r"|wrong|incorrect|not\s+(?:right|correct)|miscalculat\w*|recalculat\w*"
    r"|recomput\w*|re-?(?:check|consider|examine|evaluate|do|count)\w*"
    r"|double[-\s]?check\w*|let\s+me\s+(?:check|verify|try|redo|fix|see))\b"
    r"|\bno\s*[,.!:]",
    re.IGNORECASE,
)
#: After the final answer: a cue that the next value replaces it.
_REPLACING_CUE = re.compile(
    r"\b(?:wait|actually|hmm+|oops|correction|on\s+second\s+thought"
    r"|scratch\s+that|i\s+mean|i\s+meant|make\s+that|rather|instead)\b"
    r"|\bno\s*[,.!:]",
    re.IGNORECASE,
)
_CUE_FILLER = re.compile(
    r"^(?:[\s,:;.!-]|\b(?:it|it's|its|is|was|should|must|would|be|the|answer"
    r"|final|correct|right|real|result|value|that|this|actually|really|wait"
    r"|no|hmm+|oops|i|think|mean|meant|make|so|then|instead|rather|um+)\b)*",
    re.IGNORECASE,
)
_CLAUSE_END = re.compile(r"[;!?\n]|\.(?!\d)|,\s+(?!\d{3}\b)")
#: After the final answer: it is taken back.
_RETRACTION = re.compile(
    r"\b(?:(?:i|we)\s+(?:have\s+)?(?:made|make)\s+(?:an?\s+)?"
    r"(?:error|mistake|miscalculation)"
    r"|there\s+(?:is|was)\s+(?:an?\s+)?(?:error|mistake|miscalculation)"
    r"|(?:i|we)\s+(?:was|were)\s+wrong|i(?:'m|\s+am)\s+wrong|scratch\s+that"
    r"|(?:that|this|it)(?:['\u2019]s|\s+is|\s+was)\s+(?:wrong|incorrect"
    r"|not\s+(?:right|correct)))\b",
    re.IGNORECASE,
)
_CONDITIONAL = re.compile(r"\b(?:if|whether|no|not|never)\s+\S*\s*$", re.IGNORECASE)
#: First-person doubt about the answer: "I'm not sure", "just a guess".
_UNCERTAIN = re.compile(
    r"\b(?:i(?:'m|\s+am)\s+(?:not\s+(?:\w+\s+)?(?:sure|certain|confident)"
    r"|unsure|uncertain)|not\s+(?:\w+\s+)?sure\s+(?:if|whether|about|but)"
    r"|i\s+(?:don't|do\s+not)\s+know|no\s+idea|no\s+clue|could\s+be\s+anything"
    r"|(?:just|only|merely)\s+(?:a\s+|an\s+)?(?:educated\s+)?guess(?:ing)?"
    r"|my\s+(?:best\s+)?guess|i\s+guess|can(?:'t|not)\s+be\s+(?:sure|certain)"
    r"|i\s+(?:might|may|could)\s+be\s+wrong|(?:might|may)\s+not\s+be\s+(?:right"
    r"|correct)|take\s+(?:this|it)\s+with\s+a\s+grain\s+of\s+salt)\b",
    re.IGNORECASE,
)
#: Words that make the answer they frame tentative ("Maybe 84").
_TENTATIVE = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|likely|presumably|roughly"
    r"|approximately|about|around|nearly|almost)\b|\\approx|\u2248|~\s*$",
    re.IGNORECASE,
)
_NEGATION_BEFORE = re.compile(
    r"(?:\b(?:not|never|isn't|isn\u2019t|wasn't|aren't|cannot|can't)\b|\bno\b)"
    r"[\s,:$*]*(?:\S+[\s,:$*]+)?$",
    re.IGNORECASE,
)
_NEGATION_AFTER = re.compile(
    r"^[\s.)}$*]*(?:,\s*)?(?:is|was)?\s*(?:not\s+(?:the\s+)?(?:right\s+|correct"
    r"\s+|final\s+)?(?:answer|right|correct)|wrong|incorrect|isn't|isn\u2019t"
    r"|wasn't)\b",
    re.IGNORECASE,
)
#: After the answer: a phrase offering a value as the answer ("it is 85",
#: "it could also be 85", "some textbooks say 85").
OFFER = re.compile(
    r"\b(?:it|this|that|the\s+(?:final\s+|correct\s+|real\s+|true\s+)?"
    r"(?:answer|result|value))(?:['\u2019]s|\s+(?:is|was|equals|should\s+be"
    r"|must\s+be|would\s+be|(?:could|might|may|can)\s+(?:also\s+)?be))\b"
    r"|\b(?:says?|said|claims?|suggests?|states?|gives?|gets?)\b",
    re.IGNORECASE,
)


#: Right after an answer: it is approximate ("(approximately)", ", roughly").
_APPROXIMATION_AFTER = re.compile(
    r"^[\s,(\[*_-]*(?:approximately|approx|roughly|about|around|ish|or\s+so"
    r"|give\s+or\s+take|more\s+or\s+less|at\s+(?:least|most)"
    r"|or\s+(?:more|less|fewer|higher|lower))\b",
    re.IGNORECASE,
)
#: After an answer: another one offered beside it ("170 is also possible").
_ANOTHER_ANSWER = re.compile(
    r"\b(?:is|are|would\s+be|could\s+be|might\s+be)\s+(?:also|equally)\s+"
    r"(?:possible|plausible|valid|correct|acceptable|right)\b"
    r"|(?<!\bno\s)(?<!\bnot\s)\b(?:another|alternative|other)\s+"
    r"(?:possible\s+|valid\s+|plausible\s+)?(?:possibility|option|answer"
    r"|candidate|interpretation)\b",
    re.IGNORECASE,
)


def hedged_after(after: str) -> bool:
    """Whether the text after an answer makes it approximate or offers
    another answer beside it ("(approximately)", "170 is also possible")."""
    return bool(_APPROXIMATION_AFTER.search(after) or _ANOTHER_ANSWER.search(after))


def retracted(after: str) -> bool:
    """Whether the text after an answer takes it back ("I made a mistake"),
    not counting a conditional check ("Let me check if I made a mistake")."""
    return any(
        not _CONDITIONAL.search(after[max(0, match.start() - 24) : match.start()])
        for match in _RETRACTION.finditer(after)
    )


def uncertain(text: str) -> bool:
    """Whether ``text`` admits doubt about the answer ("I'm not sure")."""
    return _UNCERTAIN.search(text) is not None


def tentative_before(before: str) -> bool:
    """Whether the words right before an answer make it tentative."""
    clause = re.split(r"[,;:]\s|\.\s", before)[-1]
    words = clause.split()
    return _TENTATIVE.search(" ".join(words[-5:])) is not None


def negated_before(before: str) -> bool:
    """Whether the words right before an answer deny it ("It is not 84")."""
    return _NEGATION_BEFORE.search(before[-40:]) is not None


def negated_after(after: str) -> bool:
    """Whether the words right after an answer deny it ("84 is wrong")."""
    return _NEGATION_AFTER.search(after[:60]) is not None


def corrected_value_text(after: str) -> list[str]:
    """Values the text after an answer offers in its place, right after a
    replacing cue ("On second thought, 83.", "Wait, it's 85")."""
    found = []
    for match in _REPLACING_CUE.finditer(after):
        rest = after[match.end() :].lstrip(" \t,:;.!-")
        end = _CLAUSE_END.search(rest)
        clause = _CUE_FILLER.sub("", rest[: end.start()] if end else rest).strip()
        if clause:
            found.append(clause)
    return found
