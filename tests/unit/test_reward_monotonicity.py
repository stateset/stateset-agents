"""Property-based checks: reply mutations that add nothing never raise a score.

Hypothesis fuzzes mutations of the exploit-search references (and of
generated replies) and checks monotonicity properties of task grounding, the
shipped heuristic factories and ``CorrectnessReward``:

* adding unsupported specifics -- numbers and actions in a sentence or a
  semicolon clause that names nothing in the request -- never raises the score;
* saying a sentence again, appending the request, or appending request terms
  never raises it;
* inserting words of no recognized kind never raises grounding;
* a parenthetical, slash or "or" alternative beside a correct answer, or an
  appended "Or possibly X.", leaves no answer credit.
"""

from __future__ import annotations

import asyncio
import re

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from stateset_agents.core.basic_rewards import CorrectnessReward  # noqa: E402
from stateset_agents.core.grounding import content_terms, grounding_score  # noqa: E402
from stateset_agents.core.reward_factories import (  # noqa: E402
    create_helpful_agent_reward,
)
from stateset_agents.core.trajectory import ConversationTurn  # noqa: E402
from stateset_agents.evaluation.reward_exploits import DEFAULT_CASES  # noqa: E402
from stateset_agents.rewards.multi_objective_reward import (  # noqa: E402
    create_customer_service_reward,
)

CASES = [case for case in DEFAULT_CASES if len(case.reference) == 1]
NOUNS = (
    "card fee dispute voucher parcel invoice subscription claim booking coupon "
    "locker ticket"
).split()
VERBS = "blocked waived opened issued shipped escalated refunded cancelled".split()
SALAD = "zeppelin quokka marmot tulip lantern meadow walrus origami saffron".split()
EPS = 1e-9
PROPERTIES = settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)


def _score(case, reply: str) -> float:
    labels = list(case.labels)
    return grounding_score(reply, case.requests, labels, question=case.requests[-1])[0]


def _offtopic(case, draw_nouns, numbers, verbs, joiner: str) -> str:
    request = set(content_terms(" ".join(case.requests + list(case.reference))))
    nouns = [n for n in draw_nouns if n not in request] or ["locker"]
    taken = set(re.findall(r"\d+", " ".join(case.requests + list(case.reference))))
    numbers = [n for n in numbers if str(n) not in taken] or [98765]
    clauses = [
        f"{nouns[i % len(nouns)]} {numbers[i % len(numbers)]} was {verbs[i % len(verbs)]}"
        for i in range(max(len(numbers), 1))
    ]
    text = joiner.join(clauses)
    return text[0].upper() + text[1:] + "."


offtopic_args = st.tuples(
    st.lists(st.sampled_from(NOUNS), min_size=1, max_size=4),
    st.lists(st.integers(1000, 99999), min_size=1, max_size=4),
    st.lists(st.sampled_from(VERBS), min_size=1, max_size=4),
    st.sampled_from(["; ", ". "]),
)


@PROPERTIES
@given(case=st.sampled_from(CASES), args=offtopic_args)
def test_unsupported_specifics_never_raise_grounding(case, args):
    reply = case.reference[0]
    mutated = f"{reply} {_offtopic(case, *args)}"
    assert _score(case, mutated) <= _score(case, reply) + EPS


@PROPERTIES
@given(case=st.sampled_from(CASES), args=offtopic_args, front=st.booleans())
def test_unsupported_specifics_never_raise_a_factory_score(case, args, front):
    reply = case.reference[0]
    extra = _offtopic(case, *args)
    mutated = f"{extra} {reply}" if front else f"{reply} {extra}"
    for make in (create_helpful_agent_reward, create_customer_service_reward):
        assert _factory(make, case, mutated) <= _factory(make, case, reply) + EPS


def _factory(make, case, reply: str) -> float:
    turns = [
        ConversationTurn(role="user", content=case.requests[0]),
        ConversationTurn(role="assistant", content=reply),
    ]
    context = {**case.scenario, "scenario": dict(case.scenario)}
    return asyncio.run(make().compute_reward(turns, context)).score


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


@PROPERTIES
@given(case=st.sampled_from(CASES), index=st.integers(0, 10), where=st.integers(0, 10))
def test_saying_a_sentence_again_never_raises_grounding(case, index, where):
    sentences = _sentences(case.reference[0])
    repeated = sentences[index % len(sentences)]
    position = where % (len(sentences) + 1)
    mutated = " ".join([*sentences[:position], repeated, *sentences[position:]])
    assert _score(case, mutated) < _score(case, case.reference[0]) or (
        _score(case, case.reference[0]) == 0.0
    )


@PROPERTIES
@given(case=st.sampled_from(CASES), times=st.integers(1, 3))
def test_appending_the_request_or_its_terms_never_raises_grounding(case, times):
    reply = case.reference[0]
    base = _score(case, reply)
    assert _score(case, f"{reply} {case.requests[0]}") <= base + EPS
    terms = " ".join(sorted(content_terms(case.requests[0])) * times)
    assert _score(case, f"{reply} {terms}") <= base + EPS


@PROPERTIES
@given(
    case=st.sampled_from(CASES),
    words=st.lists(st.sampled_from(SALAD), min_size=1, max_size=8),
    positions=st.lists(st.integers(0, 1000), min_size=8, max_size=8),
)
def test_inserting_unrecognized_words_never_raises_grounding(case, words, positions):
    tokens = case.reference[0].split()
    for word, position in zip(words, positions, strict=False):
        tokens.insert(position % (len(tokens) + 1), word)
    assert _score(case, " ".join(tokens)) <= _score(case, case.reference[0]) + EPS


LABELED = [
    ("What is the capital of Australia?", "Canberra", "Sydney"),
    ("Who wrote Hamlet?", "William Shakespeare", "Christopher Marlowe"),
    ("What is 12 + 30?", "42", "43"),
]
HEDGES = (
    "{a} (or {b})",
    "{b} (or {a})",
    "{a} (possibly {b})",
    "{b} (maybe {a})",
    "{a}/{b}",
    "{b},{a}",
    "{a}|{b}",
    "{a} or {b}",
    "either {b} or {a}",
    "{a}. Or possibly {b}",
)


@PROPERTIES
@given(
    item=st.sampled_from(LABELED),
    hedge=st.sampled_from(HEDGES),
    lead=st.sampled_from(["The answer is", "It is", "I'd say", ""]),
)
def test_any_hedge_beside_a_correct_answer_earns_no_credit(item, hedge, lead):
    question, answer, wrong = item
    reward = CorrectnessReward(ground_truth={"answer": answer})
    reply = f"{lead} {hedge.format(a=answer, b=wrong)}.".strip()
    turns = [
        ConversationTurn(role="user", content=question),
        ConversationTurn(role="assistant", content=reply),
    ]
    committed = [
        ConversationTurn(role="user", content=question),
        ConversationTurn(role="assistant", content=f"{lead} {answer}.".strip()),
    ]
    assert asyncio.run(reward.compute_reward(committed)).score == 1.0
    assert asyncio.run(reward.compute_reward(turns)).score == 0.0
