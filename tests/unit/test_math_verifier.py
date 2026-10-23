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
