"""Where a reasoning trace first writes the gold answer as a result.

The conclude neuron is designed from line breaks right after the model has
computed the answer. For GSM8K the gold answer is a number; for MATH it is a
LaTeX expression that the model may write as ``\\dfrac{3}{4}``, ``\\frac{3}{4}``
or ``3/4``. :func:`answer_end` finds the first place the thinking part of a
trace writes any such form right after ``=``, ``is``, ``answer ...`` or
``\\boxed{`` and returns the character offset where it ends.

Torch-free so it can be unit tested on its own.
"""

from __future__ import annotations

import re

__all__ = ["answer_end", "answer_forms", "gold_in_question"]

_FRAC = re.compile(r"^\\[dt]?frac\{(-?\d+)\}\{(\d+)\}$")
_RESULT = r"(?:=|\bis\b|answer[^\n]{0,15}?|boxed\{)\s*\\?\$?\**\s*"
_AFTER = r"(?![0-9A-Za-z])(?!\.\d)"


def answer_forms(gold: str) -> list[str]:
    """Surface forms of a MATH/GSM8K gold answer, most specific first."""
    g = gold.strip().strip("$").strip()
    forms = [g, g.replace(" ", "")]
    for a, b in (("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\frac", "\\dfrac")):
        if a in g:
            forms.append(g.replace(a, b))
    m = _FRAC.match(g.replace(" ", ""))
    if m:
        forms.append(f"{m.group(1)}/{m.group(2)}")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", g):
        forms.append(g.replace(",", ""))
    if "^\\circ" in g or "^{\\circ}" in g:
        forms.append(g.replace("^\\circ", "").replace("^{\\circ}", "").strip())
    if "\\text{" in g:
        forms.append(re.sub(r"\\text\{[^}]*\}", "", g).strip())
    seen: list[str] = []
    for f in forms:
        if f and f not in seen:
            seen.append(f)
    return seen


def answer_end(text: str, gold: str) -> int | None:
    """Character offset where the thinking part first writes ``gold`` as a result.

    Only text before the first ``</think>`` is searched (the answer section is
    not thinking). Returns ``None`` when no form of ``gold`` appears there.
    """
    thinking = text.split("</think>", 1)[0]
    best: tuple[int, int] | None = None
    for form in answer_forms(gold):
        m = re.search(_RESULT + re.escape(form) + _AFTER, thinking)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), m.end())
    return best[1] if best else None


def gold_in_question(question: str, gold: str) -> bool:
    """True if the question already states the answer (such problems are skipped)."""
    for form in answer_forms(gold):
        if re.fullmatch(r"-?\d+(\.\d+)?", form):
            if re.search(r"(?<![\d.])" + re.escape(form) + r"(?![\d])", question):
                return True
        elif form in question:
            return True
    return False
