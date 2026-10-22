"""Task grounding: does a reply engage with the request and add specifics?

Heuristic style rewards (helpfulness phrases, empathy and action keywords,
length, engagement) can be maximized without answering the request: a canned
"I understand, happy to help, let me check..." template, or a reply stuffed
with the rewarded keywords. The shipped heuristic factories therefore gate
their style terms on *grounding*, a request-specific score:

* **information**: content terms the reply adds that are neither function
  words nor request-independent boilerplate (the service phrasing and keyword
  lexicons the heuristics reward), saturating at three new terms;
* **relevance**: content terms shared with the request (user messages,
  ``user_query``/``prompt``/``context``/``user_goal``/``topic``/task
  description and ground-truth labels), saturating at two;
* a penalty for the share of the reply that is boilerplate.

``grounding = information * (0.5 + 0.5 * relevance) * (1 - 0.5 * boilerplate)``

A reply that only echoes the request adds no information and scores 0; so do
canned templates and keyword stuffing. Declining an unsafe request is treated
as fully grounded, so a refusal never loses to unsafe compliance on grounding.

Grounding is a shaping signal like the heuristics it gates. For a training
objective prefer outcome, verifier or judge rewards (see
``docs/REWARD_CONTRACT.md``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any

from .basic_rewards import _REFUSAL_PATTERN, SafetyReward, _normalize_text
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
_UNSAFE_PATTERNS = SafetyReward().unsafe_patterns


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


_GENERIC = frozenset(_stem(term) for term in GENERIC_TERMS)


def content_terms(text: str) -> set[str]:
    """Distinct content terms of ``text`` (stemmed; no function words)."""
    terms = set()
    for token in _TOKEN.findall(
        _normalize_text(text).replace("$", " ").replace("#", " ")
    ):
        if token in STOPWORDS:
            continue
        if len(token) < 2 and not token.isdigit():
            continue
        terms.add(_stem(token))
    return terms


def _is_specific(term: str, labels: Collection[str] = ()) -> bool:
    """Whether ``term`` carries information: a reference label, a multi-digit
    number or amount, or a word that is not boilerplate. A lone digit ("step
    1") is a list marker unless it is a label."""
    if term in labels:
        return True
    if any(ch.isdigit() for ch in term):
        return len(term) > 1
    return term not in _GENERIC


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


def grounding_score(
    reply: str, request: str, labels: Sequence[str] = ()
) -> tuple[float, dict[str, float]]:
    """Grounding of one ``reply`` in a ``request`` (see the module docs).

    ``labels`` are reference facts (ground-truth answers): mentioning them is
    relevant *and* new information, since the user did not say them.
    """
    reply_terms = content_terms(reply)
    request_terms = content_terms(request)
    label_terms = set().union(*(content_terms(label) for label in labels))
    details = {"information": 0.0, "relevance": 0.0, "boilerplate": 0.0}
    if not reply_terms:
        return 0.0, details
    request_lower = _normalize_text(request)
    if _REFUSAL_PATTERN.search(_normalize_text(reply)) and any(
        re.search(pattern, request_lower) for pattern in _UNSAFE_PATTERNS
    ):
        # Declining an unsafe request answers it.
        return 1.0, {"information": 1.0, "relevance": 1.0, "boilerplate": 0.0}
    known = request_terms | label_terms
    relevant = reply_terms & known
    new = {t for t in reply_terms - request_terms if _is_specific(t, label_terms)}
    boilerplate = {t for t in reply_terms - known if not _is_specific(t, label_terms)}
    information = min(1.0, len(new) / 3.0)
    relevance = 1.0 if not known else min(1.0, len(relevant) / 2.0)
    share = len(boilerplate) / len(reply_terms)
    details = {"information": information, "relevance": relevance, "boilerplate": share}
    return information * (0.5 + 0.5 * relevance) * (1.0 - 0.5 * share), details


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

    ``score = grounding * (share + (1 - share) * base_score)``: a canned or
    keyword-stuffed reply (grounding 0) earns nothing however well it matches
    the style heuristics, while a grounded reply keeps their ordering. With
    ``gate_if`` the gate applies only to results it returns True for (e.g.
    a task reward scored by keywords alone). Name, weight and reward type are
    the base reward's, so composites treat the wrapper as the base.
    """

    def __init__(
        self,
        base: RewardFunction,
        *,
        share: float = 0.5,
        gate_if: Callable[[RewardResult], bool] | None = None,
    ) -> None:
        super().__init__(base.weight, base.reward_type, base.name)
        self.base = base
        self.share = float(share)
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
    rewards: Sequence[RewardFunction], *, share: float = 0.5
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
