"""Task grounding: does a reply engage with the request and add specifics?

Heuristic style rewards (helpfulness phrases, empathy and action keywords,
length, engagement) can be maximized without answering the request: a canned
"I understand, happy to help, let me check..." template, keyword stuffing, an
echo of the request, or word salad. The shipped heuristic factories therefore
gate their style terms on *grounding*, a request-specific score:

* **information**: the specifics a reply adds that the request did not
  contain: numbers and amounts (two or more characters, so a "step 1" list
  marker is not one) and affirmed action words (``ACTION_TERMS``: "issued",
  "blocked", "rejoin", "multiply"...; "not issued" does not count), each
  only in a clause that also names something else, so a bare list of
  numbers or verbs states nothing. Information grows with every distinct
  specific, ``n / (n + INFO_HALF)``, so removing any of them lowers it. Words that
  are none of these (including salad, real or invented) are not
  information. With reference labels (``ground_truth``), a short label
  earns the answer credit ``CorrectnessReward`` gives it (hedged lists share
  it, negations get none) and numbers other than a numeric label are
  competing answers, not information; other specifics then count at most
  half;
* **relevance**: content terms shared with the request (user messages,
  ``user_query``/``prompt``/``context``/``user_goal``/``topic``/task
  description and labels), saturating at two, halved for a near-verbatim
  echo (at least 80% of a request of four or more terms);
* **quality**: ``(1 - boilerplate) * (1 - filler ** 3)``, where
  *boilerplate* is the share of the reply's terms that are request-independent
  phrasing (``GENERIC_TERMS``, which covers every lexicon the heuristics
  reward) and *filler* adds the share of words of no recognized kind.
  Ordinary prose keeps some such words at little cost; padding a reply with
  boilerplate or salad is penalized steeply.

``grounding = information * (0.5 + 0.5 * relevance) * quality``

Templates, keyword stuffing, echoes, salad, terse replies and blanket
refusals therefore have no information and score 0. Declining is grounded
only when the request asks for help with something unsafe
(``basic_rewards.harmful_request``: "how do I hack...", "where can I buy
drugs"), not when it merely mentions an unsafe topic ("my card was stolen").

Grounding is lexical: it cannot tell a fabricated number or action from a
real one, and it does not read word order. It is a shaping signal like the
heuristics it gates; for a training objective prefer outcome, verifier or
judge rewards (see ``docs/REWARD_CONTRACT.md``, which lists exactly what the
exploit search in ``stateset_agents.evaluation.reward_exploits`` verifies).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any

from .basic_rewards import (
    _REFUSAL_PATTERN,
    _answer_credit,
    _as_number,
    _normalize_text,
    harmful_request,
)
from .reward_base import RewardFunction, RewardResult, RewardType
from .reward_context import scores_latest_turn_only

#: Function words: never evidence of content.
STOPWORDS = frozenset("""
    a about above after again against all also am an and any are aren't as at
    be because been before being below between both but by can can't cannot
    could couldn't did didn't do does doesn't doing don't down during each few
    for from further get got had hadn't has hasn't have haven't having he he'd
    he'll he's her here here's hers herself him himself his how how's i i'd i'll
    i'm i've if in into is isn't it it's its itself just let let's me might
    more most must mustn't my myself no nor not now of off on once only or
    other ought our ours ourselves out over own same shan't she she'd she'll
    she's should shouldn't so some such than that that's the their theirs them
    themselves then there there's these they they'd they'll they're they've
    this those through to too under until up us very was wasn't we we'd we'll
    we're we've were weren't what what's when when's where where's which while
    who who's whom why why's will with won't would wouldn't yes yet you you'd
    you'll you're you've your yours yourself yourselves
    """.split())

#: Request-independent phrasing: service boilerplate and every keyword the
#: built-in heuristics reward (helpfulness phrases, empathy, action,
#: professionalism, engagement and domain lexicons). Saying these is not
#: information; ``tests/unit/test_reward_gaming.py`` checks the heuristic
#: lexicons stay covered so keyword stuffing cannot earn grounding.
GENERIC_TERMS = frozenset("""
    absolutely account action actually advantage again amazing angry annoying
    believe since sorry wtf artistic beautiful buy character choice creative
    descriptive discount exclusive ideal imagine innovative inspiring narrative
    order original purchase save scene story unique vivid
    answer anything apologize appreciate approach assist assistance assistant
    away awesome based benefit benefits best better caused certainly check
    checking click completed concern concerns configuration configure contact
    course customer deal definitely detail details diagnose diagnostic due
    dumb dunno else email empathy everything exciting experience explanation
    fantastic fascinating feature features feel finally first fix fixed
    follow found frustrat frustrated frustrating glad good great guarantee
    happy hate help helpful helping hope idea idk important incredible
    information inspect interesting issue issues kindly know limited look
    looking lol matter may method need needs next note offer okay omg
    opinion option options perfect personalization personally please pleased
    possible problem problems procedure process processed professional
    professionalism provide quality question questions quick quickly reach
    really reason recommend reinstall request resolution resolve resolved
    restart return right satisfaction second see service settings situation
    software solution solutions solve something special step steps stupid
    sucks suggest support sure system team test thank thanks think time today
    totally trust trusted try understand update updated urgency value verify
    visit want warranty way welcome wonderful worry
    """.split())

#: Context fields that describe what the user asked for.
_REQUEST_FIELDS = (
    "user_query",
    "prompt",
    "query",
    "question",
    "context",
    "user_goal",
    "goal",
    "topic",
    "description",
    "instructions",
)
_TOKEN = re.compile(r"[a-z0-9]+(?:['./:-][a-z0-9]+)*")


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


_GENERIC = frozenset(_stem(term) for term in GENERIC_TERMS)


#: Verbs of doing: what was done or is to be done for the user ("issued",
#: "blocked", "forget ... and rejoin"). With numbers and reference labels
#: they are the specifics grounding counts as information; any other word a
#: reply adds might be salad. Disjoint from ``GENERIC_TERMS`` (pinned by a
#: test), so stuffing the heuristics' lexicons earns none.
_ACTION_VERBS = """
    issue refund credit reverse block unblock cancel replace ship deliver open
    reopen close reset forget rejoin reconnect disconnect disable enable turn
    switch toggle install uninstall reboot dispute schedule reschedule book
    transfer waive approve deny post escalate unlock lock renew extend upgrade
    downgrade activate deactivate adjust increase decrease reduce raise lower
    multiply divide subtract add differentiate integrate simplify substitute
    factor expand apply set change assign send track charge pay remove delete
    rename move restore recover file submit confirm
    """.split()
_IRREGULAR = {
    "forget": ("forgot", "forgotten"),
    "send": ("sent",),
    "pay": ("paid",),
    "deny": ("denied",),
    "set": (),
    "lock": (),
}


def _inflections(verb: str) -> set[str]:
    forms = {verb, *_IRREGULAR.get(verb, ())}
    if verb.endswith("e"):
        forms |= {verb + "d", verb[:-1] + "ing"}
    elif verb.endswith("y") and verb[-2] not in "aeiou":
        forms |= {verb[:-1] + "ied", verb[:-1] + "ies", verb + "ing"}
    else:
        forms |= {verb + "ed", verb + "ing"}
        if (
            len(verb) <= 4
            and verb[-1] not in "aeiouwxy"
            and verb[-2] in "aeiou"
            and verb[-3] not in "aeiou"
        ):
            # ship -> shipped, shipping
            forms |= {verb + verb[-1] + "ed", verb + verb[-1] + "ing"}
    if verb == "cancel":
        forms |= {"cancelled", "cancelling"}
    return forms


ACTION_TERMS = frozenset(
    _stem(form)
    for verb in _ACTION_VERBS
    for form in _inflections(verb)
    if _stem(form) not in _GENERIC
)


_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _plain(text: str) -> str:
    """Normalized text with "$"/"#" dropped and "1,299" read as "1299"."""
    text = _THOUSANDS.sub("", _normalize_text(text))
    return text.replace("$", " ").replace("#", " ")


def content_terms(text: str) -> set[str]:
    """Distinct content terms of ``text`` (stemmed; no function words)."""
    terms = set()
    for token in _TOKEN.findall(_plain(text)):
        if token in STOPWORDS:
            continue
        if len(token) < 2 and not token.isdigit():
            continue
        terms.add(_stem(token))
    return terms


def _is_specific(term: str, labels: Collection[str] = ()) -> bool:
    """Whether ``term`` is a specific: a reference label term, a number or
    amount of two or more characters (a lone digit is a list marker), or an
    action word. Any other word, known or invented, is not."""
    if term in labels:
        return True
    if any(ch.isdigit() for ch in term):
        return len(term) > 1
    return term in ACTION_TERMS


#: Share of a gated style reward paid for grounding alone: a grounded reply
#: earns ``grounding * (STYLE_SHARE + (1 - STYLE_SHARE) * style)``, so style
#: adjusts a grounded reply's reward by at most a quarter.
STYLE_SHARE = 0.75
#: Specifics at which information reaches one half: ``n / (n + INFO_HALF)``.
#: Information keeps growing with every specific, so a reply that drops some
#: of an answer's specifics scores lower than the answer.
INFO_HALF = 6.0

#: Labels of at most this many words are answers (scored with answer
#: credit); longer ones are facts whose terms count as specifics.
_ANSWER_WORDS = 3


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield str(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _request_texts(context: Mapping[str, Any] | None) -> tuple[list[str], list[str]]:
    """Request descriptions and ground-truth labels found in a reward context."""
    if not isinstance(context, Mapping):
        return [], []
    sources = [context]
    for nested in ("scenario", "task"):
        if isinstance(context.get(nested), Mapping):
            sources.append(context[nested])
    texts: list[str] = []
    labels: list[str] = []
    for source in sources:
        for field in _REQUEST_FIELDS:
            value = source.get(field)
            if isinstance(value, str):
                texts.append(value)
        labels.extend(_strings(source.get("ground_truth")))
        labels.extend(_strings(source.get("correct_facts")))
    return texts, labels


def _turn_field(turn: Any, name: str) -> Any:
    return turn.get(name) if isinstance(turn, Mapping) else getattr(turn, name, None)


_NEGATIONS = frozenset(
    "not no never cannot can't won't didn't don't doesn't isn't wasn't haven't "
    "hasn't hadn't unable".split()
)


#: Clause boundaries; a period inside a number ("42.10") is not one.
_CLAUSE = re.compile(r"[,;:!?()\n]|\.(?!\d)")


def _stated_specifics(text: str) -> set[str]:
    """Numbers and action words ``text`` states in context.

    A number of two or more characters, or an action word, counts only in a
    clause that also has a word of another kind (an object or subject such as
    "refund", "card", "battery"); a bare list of numbers or of verbs
    ("4093 8812 7741", "issued, blocked, reversed") states nothing. An action
    is also not stated when a negation precedes it ("not issued", "never
    blocked").
    """
    stated: set[str] = set()
    for clause in _CLAUSE.split(_plain(text)):
        tokens = [_stem(token) for token in _TOKEN.findall(clause)]
        anchored = any(
            token not in STOPWORDS
            and token not in ACTION_TERMS
            and token not in _GENERIC
            and token not in _NEGATIONS
            and not any(ch.isdigit() for ch in token)
            for token in tokens
        )
        if not anchored:
            continue
        for index, token in enumerate(tokens):
            if any(ch.isdigit() for ch in token):
                if len(token) > 1:
                    stated.add(token)
            elif token in ACTION_TERMS:
                before = tokens[max(0, index - 2) : index]
                if not any(w in _NEGATIONS or w.endswith("n't") for w in before):
                    stated.add(token)
    return stated


def grounding_score(
    reply: str, request: str, labels: Sequence[str] = ()
) -> tuple[float, dict[str, float]]:
    """Grounding of one ``reply`` in a ``request`` (see the module docs).

    ``labels`` are reference facts (ground-truth answers). A short label is
    an answer: the reply earns ``CorrectnessReward``'s answer credit for it
    (shared among hedged candidates, none when negated). A longer label is a
    fact whose terms count as specifics.
    """
    reply_terms = content_terms(reply)
    request_terms = content_terms(request)
    answers = [label for label in labels if len(label.split()) <= _ANSWER_WORDS]
    facts = [label for label in labels if len(label.split()) > _ANSWER_WORDS]
    label_terms = set().union(*(content_terms(label) for label in labels))
    fact_terms = set().union(*(content_terms(fact) for fact in facts))
    details = {"information": 0.0, "relevance": 0.0, "filler": 0.0}
    if not reply_terms:
        return 0.0, details
    if _REFUSAL_PATTERN.search(_normalize_text(reply)) and harmful_request(request):
        # Declining a harmful request answers it.
        return 1.0, {"information": 1.0, "relevance": 1.0, "filler": 0.0}

    numeric_answer = any(
        _as_number(answer.strip().lstrip("$")) is not None for answer in answers
    )
    specifics = set()
    stated = _stated_specifics(reply)
    for term in reply_terms - request_terms:
        if term in fact_terms:
            specifics.add(term)
        elif term in label_terms:
            continue  # an answer label: scored by answer credit below
        elif any(ch.isdigit() for ch in term):
            # Beside a numeric answer, other numbers are competing answers.
            if term in stated and not numeric_answer:
                specifics.add(term)
        elif term in stated:
            specifics.add(term)  # an action done (not denied), in context
    breadth = len(specifics) / (len(specifics) + INFO_HALF)
    if answers:
        credit = max(_answer_credit(reply, answer, request) for answer in answers)
        information = max(credit, 0.5 * breadth)
    else:
        information = breadth

    known = request_terms | label_terms
    relevant = reply_terms & known
    relevance = 1.0 if not known else min(1.0, len(relevant) / 2.0)
    if len(request_terms) >= 4 and (
        len(reply_terms & request_terms) / len(request_terms) >= 0.8
    ):
        relevance *= 0.5  # a near-verbatim echo of the request
    grounded = relevant | specifics | (reply_terms & label_terms)
    ungrounded = reply_terms - grounded
    boilerplate = len({t for t in ungrounded if t in _GENERIC}) / len(reply_terms)
    unknown = len({t for t in ungrounded if t not in _GENERIC}) / len(reply_terms)
    # Ordinary prose leaves some words of no recognized kind; padding with
    # boilerplate or salad pushes that share up. The cubic tolerates the
    # former and penalizes the latter steeply; boilerplate also costs its
    # share outright.
    quality = (1.0 - boilerplate) * (1.0 - (boilerplate + unknown) ** 3)
    details = {
        "information": information,
        "relevance": relevance,
        "boilerplate": boilerplate,
        "unknown": unknown,
    }
    return information * (0.5 + 0.5 * relevance) * quality, details


def task_grounding(
    turns: Sequence[Any], context: Mapping[str, Any] | None = None
) -> float:
    """Mean grounding of the assistant replies in scope (dict or object turns).

    Each reply is judged against the user messages before it plus the
    request fields and labels of ``context``; with ``reward_scope`` set to
    the latest turn only the newest reply counts.
    """
    texts, labels = _request_texts(context)
    asked: list[str] = []
    scores: list[float] = []
    for turn in turns:
        role, content = _turn_field(turn, "role"), _turn_field(turn, "content")
        text = content if isinstance(content, str) else ""
        if role == "user":
            asked.append(text)
        elif role == "assistant":
            score, _ = grounding_score(text, " ".join([*texts, *asked]), labels)
            scores.append(score)
    if not scores:
        return 0.0
    if scores_latest_turn_only(context):
        return scores[-1]
    return sum(scores) / len(scores)


class TaskGroundingReward(RewardFunction):
    """Reward for replies that address the request with specific content."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight, RewardType.IMMEDIATE, "TaskGroundingReward")

    async def compute_reward(
        self, turns: list[Any], context: dict[str, Any] | None = None
    ) -> RewardResult:
        score = task_grounding(turns, context)
        return RewardResult(score=score, breakdown={"grounding": score})


class GroundedReward(RewardFunction):
    """A style (shaping) reward that only pays for grounded replies.

    ``score = grounding * (share + (1 - share) * base_score)``, with
    ``share`` defaulting to ``STYLE_SHARE``: a canned or keyword-stuffed reply
    (grounding 0) earns nothing however well it matches the style heuristics,
    while a grounded reply keeps their ordering. With
    ``gate_if`` the gate applies only to results it returns True for (e.g.
    a task reward scored by keywords alone). Name, weight and reward type are
    the base reward's, so composites treat the wrapper as the base.
    """

    def __init__(
        self,
        base: RewardFunction,
        *,
        share: float | None = None,
        gate_if: Callable[[RewardResult], bool] | None = None,
    ) -> None:
        super().__init__(base.weight, base.reward_type, base.name)
        self.base = base
        self.share = float(STYLE_SHARE if share is None else share)
        self.gate_if = gate_if

    @property
    def reward_type(self) -> RewardType:
        return self.base.reward_type

    @reward_type.setter
    def reward_type(self, value: RewardType) -> None:
        if hasattr(self, "base"):
            self.base.reward_type = value

    @property
    def fail_on_error(self) -> bool:
        return getattr(self.base, "fail_on_error", False) is True

    @property
    def requires_environment_context(self) -> bool:
        return getattr(self.base, "requires_environment_context", False) is True

    async def compute_reward(
        self, turns: list[Any], context: dict[str, Any] | None = None
    ) -> RewardResult:
        result = await self.base.compute_reward(turns, context)
        if self.gate_if is not None and not self.gate_if(result):
            return result
        grounding = task_grounding(turns, context)
        score = grounding * (self.share + (1.0 - self.share) * result.score)
        return RewardResult(
            score=score,
            breakdown={
                **result.breakdown,
                "grounding": grounding,
                "ungrounded_score": result.score,
            },
            metadata=dict(result.metadata),
            explanation=result.explanation,
        )


#: Logged by starters and presets that train on a heuristic factory reward.
HEURISTIC_REWARD_NOTICE = (
    "%s trains on the heuristic %s reward, a grounded shaping signal rather "
    "than a training objective. For production training use an outcome, "
    "verifier or judge reward (e.g. TaskOutcomeReward, NSRVerifierReward, "
    "LLMJudgeReward); see docs/REWARD_CONTRACT.md."
)


def log_heuristic_reward_notice(logger: Any, trainer: str, reward: str) -> None:
    """Recommend an outcome/verifier/judge reward over a heuristic factory."""
    logger.warning(HEURISTIC_REWARD_NOTICE, trainer, reward)


def ground_shaping_rewards(
    rewards: Sequence[RewardFunction], *, share: float | None = None
) -> list[RewardFunction]:
    """Wrap every reward marked ``shaping = True`` in :class:`GroundedReward`."""
    return [
        (
            GroundedReward(reward, share=share)
            if getattr(reward, "shaping", False) is True
            else reward
        )
        for reward in rewards
    ]


__all__ = [
    "ACTION_TERMS",
    "GENERIC_TERMS",
    "HEURISTIC_REWARD_NOTICE",
    "STOPWORDS",
    "GroundedReward",
    "TaskGroundingReward",
    "content_terms",
    "ground_shaping_rewards",
    "grounding_score",
    "log_heuristic_reward_notice",
    "task_grounding",
]
