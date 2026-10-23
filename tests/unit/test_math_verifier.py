"""MathAnswerVerifier and the safe expression engine behind it."""

from __future__ import annotations

import asyncio
from fractions import Fraction

import pytest

from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.math_expressions import (
    ExpressionError,
    equivalent,
    is_numeric_literal,
    normalize_answer_text,
    parse_expression,
    rational_value,
)
from stateset_agents.core.reward_context import score_response
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.math_verifier import (
    MathAnswerVerifier,
    extract_final_answers,
    verify_math_answer,
)

QUESTION = "What is 12 * 7?"


# --- expression engine -------------------------------------------------------


@pytest.mark.parametrize(
    "text,value",
    [
        ("84", Fraction(84)),
        ("-3/4", Fraction(-3, 4)),
        ("2^-1", Fraction(1, 2)),
        ("2**10", Fraction(1024)),
        ("(1+2)(3+4)", Fraction(21)),
        ("1e3", Fraction(1000)),
        ("0.1 + 0.2", Fraction(3, 10)),
    ],
)
def test_rational_values_are_exact(text, value):
    assert rational_value(parse_expression(text)) == value


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('true')",
        "lambda: 1",
        "a.b",
        "x[0]",
        "foo(1)",
        "(" * 100 + "1" + ")" * 100,
        "1" * 300,
        "",
    ],
)
def test_parser_rejects_anything_outside_the_grammar(text):
    with pytest.raises(ExpressionError):
        parse_expression(text)


def test_oversized_powers_are_refused_not_computed():
    assert rational_value(parse_expression("9^9^9")) is None
    assert rational_value(parse_expression("2^100000")) is None
    assert rational_value(parse_expression("1/0")) is None


@pytest.mark.parametrize(
    "a,b,same",
    [
        ("x^2 + 2x + 1", "(x+1)^2", True),
        ("2(x+1)", "2x + 2", True),
        ("(x^2-1)/(x-1)", "x + 1", True),
        ("2x + 3", "2x + 2", False),
        ("x*y", "y*x", True),
        ("1/2", "0.5", True),
        ("3x^2 + 2", "3x^2 + 2 + 10", False),
    ],
)
def test_symbolic_equivalence_without_sympy(a, b, same, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_sympy(name, *args, **kwargs):
        if name == "sympy":
            raise ImportError("no sympy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sympy)
    assert equivalent(parse_expression(a), parse_expression(b)) is same


def test_functions_use_sympy_when_installed():
    pytest.importorskip("sympy")
    assert equivalent(parse_expression("sqrt(8)"), parse_expression("2sqrt(2)"))
    assert equivalent(parse_expression("sin(x)^2 + cos(x)^2"), parse_expression("1"))
    assert not equivalent(parse_expression("sqrt(8)"), parse_expression("3"))


@pytest.mark.parametrize(
    "latex,plain",
    [
        (r"\frac{3}{4}", "((3)/(4))"),
        (r"\dfrac{1}{2}", "((1)/(2))"),
        (r"$1,299$", "1299"),
        (r"x = 5.", "5"),
        (r"\text{84}", "84"),
        (r"2 \cdot 3", "2 * 3"),
        (r"\sqrt{2}", "sqrt(2)"),
    ],
)
def test_latex_normalization(latex, plain):
    assert normalize_answer_text(latex) == plain


def test_numeric_literals_exclude_unevaluated_arithmetic():
    assert is_numeric_literal(parse_expression("-3/4"))
    assert is_numeric_literal(parse_expression("0.75"))
    assert not is_numeric_literal(parse_expression("12*7"))
    assert not is_numeric_literal(parse_expression("1/2 + 1"))


# --- verdicts ----------------------------------------------------------------

CORRECT = [
    ("12 * 7 = 84, so the answer is \\boxed{84}.", "84"),
    ("The answer is 84.", "84"),
    ("#### 84", "84"),
    ("84", "84"),
    ("12 * 7 = 84", "84"),
    ("The answer is 84 (or 84.0).", "84"),
    ("\\boxed{\\frac{3}{4}}", "0.75"),
    ("\\boxed{0.75}", "3/4"),
    ("\\boxed{6/8}", "3/4"),
    ("The answer is -3/4.", "-0.75"),
    ("The answer is 1,299.", "1299"),
    ("The answer is 42 dollars.", "42"),
    ("\\boxed{2(x+1)}", "2x+2"),
    ("The answer is Paris.", "paris"),
]
WRONG = [
    ("First 3*4 = 12, then 12*7 = 84.", "84", "no_answer"),
    ("\\boxed{12*7}", "84", "incorrect"),
    ("\\boxed{41} \\boxed{84}", "84", "multiple_answers"),
    ("\\boxed{84} or maybe 85", "84", "hedged"),
    ("The answer is 84, or maybe 85.", "84", "hedged"),
    ("The answer is 84 or 85.", "84", "hedged"),
    ("It is 85 or \\boxed{84}.", "84", "hedged"),
    ("Maybe 84.", "84", "hedged"),
    ("I'm not sure, but the answer is 84.", "84", "hedged"),
    ("Parsnip, Saffron, 84, Glacier", "84", "hedged"),
    ("\\boxed{\\text{84 or 84 + 1}}", "84", "hedged"),
    ("It is not 84.", "84", "negated"),
    ("84 is not the answer.", "84", "negated"),
    ("\\boxed{84} is wrong", "84", "negated"),
    ("41, 42, 84, 85", "84", "no_answer"),
    ("\\boxed{84.0001}", "84", "incorrect"),
    ("\\boxed{}", "84", "incorrect"),
    ("\\boxed{__import__('os').getpid()}", "84", "incorrect"),
    ("\\boxed{x^2+1}", "2x+2", "incorrect"),
    (
        '<tool_call>{"name": "calculator", "arguments": {"expression": "84"}}</tool_call>',
        "84",
        "no_answer",
    ),
    ('{"tool": "calculator", "args": {"expression": "84"}}', "84", "no_answer"),
]


@pytest.mark.parametrize("reply,reference", CORRECT)
def test_committed_correct_answers_score(reply, reference):
    verdict = verify_math_answer(reply, reference, question=QUESTION)
    assert verdict.correct, verdict


@pytest.mark.parametrize("reply,reference,status", WRONG)
def test_hedged_wrong_or_missing_answers_get_no_credit(reply, reference, status):
    verdict = verify_math_answer(reply, reference, question=QUESTION)
    assert not verdict.correct
    assert verdict.status == status, verdict


def test_tolerance_is_opt_in():
    assert not verify_math_answer("\\boxed{0.333}", "1/3").correct
    assert verify_math_answer("\\boxed{0.333}", "1/3", rel_tolerance=1e-2).correct
    assert verify_math_answer("\\boxed{0.333}", "1/3", abs_tolerance=1e-3).correct


def test_symbolic_false_requires_the_canonical_form():
    assert verify_math_answer("\\boxed{(x+1)^2}", "x^2+2x+1").correct
    assert not verify_math_answer(
        "\\boxed{(x+1)^2}", "x^2+2x+1", symbolic=False
    ).correct
    assert verify_math_answer(
        "\\boxed{x^2 + 2x + 1}", "x^2+2x+1", symbolic=False
    ).correct


def test_extraction_order_of_authority():
    assert extract_final_answers("The answer is 3. \\boxed{4}")[0] == "boxed"
    assert extract_final_answers("x is 3. The answer is 4.")[0] == "stated"
    assert extract_final_answers("It is 4.", "What is 2+2?")[0] == "last_number"


def _turns(reply: str) -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user", content=QUESTION),
        ConversationTurn(role="assistant", content=reply),
    ]


def test_reward_reads_reference_from_context_or_constructor():
    verifier = MathAnswerVerifier()
    result = asyncio.run(
        verifier.compute_reward(
            _turns("\\boxed{84}"), {"ground_truth": {"answer": "84"}}
        )
    )
    assert result.score == 1.0 and result.metadata["status"] == "correct"
    assert (
        asyncio.run(MathAnswerVerifier(answer=84).compute_reward(_turns("84"))).score
        == 1.0
    )
    with pytest.raises(ConfigurationError):
        asyncio.run(verifier.compute_reward(_turns("84"), {}))


def test_score_response_path_matches_an_episode():
    score = asyncio.run(
        score_response(
            MathAnswerVerifier(), QUESTION, "\\boxed{84}", {"ground_truth": "84"}
        )
    )
    assert score == 1.0
    question_numbers = asyncio.run(
        score_response(MathAnswerVerifier(), QUESTION, "12 * 7", {"ground_truth": "84"})
    )
    assert question_numbers == 0.0


def test_pathological_answers_are_bounded():
    """Model output cannot make the parser build huge numbers or scan forever."""
    import time

    started = time.monotonic()
    assert rational_value(parse_expression("1e300")) == Fraction(10) ** 300
    with pytest.raises(ExpressionError):
        parse_expression("1e99999999")
    assert not verify_math_answer("\\boxed{1e99999999}", "84").correct
    assert not verify_math_answer("\\boxed{" * 5000, "84").correct
    long_reply = "\\frac{1}{2} " * 5000 + "\\boxed{84}"
    assert verify_math_answer(long_reply, "84").correct
    assert time.monotonic() - started < 20


# --- bounded sympy work (round 5) ---------------------------------------------


@pytest.mark.parametrize("reply", [r"\boxed{sqrt(x)+9^9^8}", r"\boxed{sqrt(x)+9^9^7}"])
def test_power_towers_beside_a_function_are_refused_quickly(reply):
    """The exact path's size limits also bound the sympy path: a tower next to
    ``sqrt`` used to build a multi-million-digit power (60 s and more)."""
    import time

    started = time.monotonic()
    verdict = verify_math_answer(reply, "3x^2")
    assert not verdict.correct
    assert time.monotonic() - started < 2.0


def test_sympy_conversion_applies_the_exact_path_limits():
    sympy = pytest.importorskip("sympy")
    from stateset_agents.core.math_expressions import _to_sympy

    for text in ("9^9^8", "x^(9^9)", "(2^300)^300", "x^1000", "sqrt(x)^(10^5)"):
        with pytest.raises(ExpressionError):
            _to_sympy(parse_expression(text), sympy)
    x = sympy.Symbol("x")
    assert _to_sympy(parse_expression("x^256 + 2^10"), sympy) == x**256 + 1024


def test_slow_simplification_hits_a_hard_timeout_and_scores_zero():
    """A bounded but slow simplification (a minute or more in sympy) is cut off by
    the worker's hard timeout; a timeout decides nothing, so it scores 0."""
    import time

    pytest.importorskip("sympy")
    slow = "tan(x)^30 - sin(x)^30/cos(x)^30"  # over 60 s in sympy.simplify
    # Start the worker first so the timing below measures only the timeout.
    assert equivalent(parse_expression("sqrt(8)"), parse_expression("2sqrt(2)"))
    started = time.monotonic()
    assert (
        equivalent(parse_expression(slow), parse_expression("1"), timeout=1.0) is None
    )
    verdict = verify_math_answer(rf"\boxed{{{slow} + x}}", "x + 1", sympy_timeout=1.0)
    assert not verdict.correct
    assert time.monotonic() - started < 20
    # A fresh worker takes over after a timeout.
    assert equivalent(parse_expression("sin(x)^2 + cos(x)^2"), parse_expression("1"))


def test_compute_reward_does_not_block_the_event_loop(monkeypatch):
    """Verification runs off the event loop, so a slow check never stalls the
    trainer's other coroutines."""
    import time

    from stateset_agents.rewards import math_verifier

    def slow_verify(*args, **kwargs):
        time.sleep(0.5)
        return math_verifier.MathVerdict(True, "correct", "84")

    monkeypatch.setattr(math_verifier, "verify_math_answer", slow_verify)

    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        result = await MathAnswerVerifier(answer=84).compute_reward(
            _turns("\\boxed{84}")
        )
        task.cancel()
        return result, ticks

    result, ticks = asyncio.run(main())
    assert result.score == 1.0
    assert ticks >= 10, f"the event loop ran only {ticks} ticks in 0.5 s"


# --- hedges inside a box and later contradictions (round 5) -------------------

HEDGED_OR_CONTRADICTED = [
    r"\boxed{84, 85}",
    r"\boxed{84 \pm 1}",
    r"\boxed{84 \text{ or } 85}",
    r"\boxed{84}. Actually, I think I made an error; it is 85.",
    r"\boxed{84} (it could also be 85)",
    "The answer is 84. Some textbooks say 85.",
    r"\boxed{84 +/- 1}",
    r"\boxed{\approx 84}",
    r"\boxed{84;85}",
    r"\boxed{84 \text{ or so}}",
    r"\boxed{84}. Wait, I made a mistake.",
    "The answer is 84. Actually it is 85.",
    "The answer is 84, 85.",
    "The answer is 84 or so.",
    r"\boxed{3x^2 + 2}. Some books say 3x^2 + 3.",
]


@pytest.mark.parametrize("reply", HEDGED_OR_CONTRADICTED)
def test_hedged_boxes_and_later_contradictions_score_zero(reply):
    reference = "3x^2 + 2" if "x" in reply else "84"
    verdict = verify_math_answer(reply, reference, question=QUESTION)
    assert not verdict.correct, verdict
    assert verdict.status in {"hedged", "contradicted"}, verdict


COMMITTED_DESPITE_EXTRA_TEXT = [
    (r"\boxed{1,299}", "1299"),
    (r"\boxed{84 \text{ dollars}}", "84"),
    (r"\boxed{84}. Check: 84 / 7 = 12.", "84"),
    ("The answer is 84, which is 12 more than 72.", "84"),
    (r"\boxed{84}. It is 7 * 12.", "84"),
    (r"\boxed{84}, since the answer is 12 * 7.", "84"),
    (r"\boxed{1, 2}", "1, 2"),
]


@pytest.mark.parametrize("reply,reference", COMMITTED_DESPITE_EXTRA_TEXT)
def test_units_restatements_and_checks_keep_their_credit(reply, reference):
    verdict = verify_math_answer(reply, reference, question=QUESTION)
    assert verdict.correct, verdict
