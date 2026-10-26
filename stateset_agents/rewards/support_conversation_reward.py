"""Conversation-level, hack-resistant customer-support reward (v2).

:class:`SupportConversationReward` scores a whole support episode. Every
assistant reply is first graded on its own by
:class:`~stateset_agents.data.customer_support_bench.SupportRewardComposite`
(given the scenario context and the customer turns before it), then adjusted
by penalties that close the exploits a per-turn keyword reward leaves open:

* **Repetition** -- a policy can re-send its best-scoring paragraph on every
  turn. For assistant turn ``t > 0`` the similarity to the closest earlier
  assistant turn (``difflib`` ratio over lower-cased, whitespace-collapsed
  text, ``autojunk=False``) above ``repetition_threshold`` (``tau``) scales
  the turn by ``max(0, 1 - (sim - tau) / (1 - tau))``: an identical reply
  scores 0.
* **Placeholders** -- an unfilled template field (``[Customer's Name]``,
  ``{order_id}``) multiplies the turn by ``placeholder_factor``. Detection is
  :func:`stateset_agents.evaluation.support_judge.detect_findings`, the same
  detector the held-out judge is shown, so the two never drift apart.
* **must_avoid** -- any assistant turn containing a scenario ``must_avoid``
  phrase (case-insensitive substring, again via ``detect_findings``) zeroes
  the whole episode.
* **Responsiveness** -- for turns after the first, when the customer's latest
  message introduces at least ``responsiveness_min_new_words`` new content
  words (lower-case alphanumeric tokens of 3+ characters that are not
  stopwords and do not appear in any earlier assistant turn), the reply must
  use at least one of them -- matched exactly or by a shared 5-character
  prefix, so "refunded" engages with "refund" -- or the turn is multiplied by
  ``responsiveness_factor``. Follow-ups that bring nothing new are not
  penalized. The heuristic is deliberately lexical: it cannot tell a good
  answer from a bad one, it only catches replies that ignore the follow-up.

**Task grounding.** The default per-turn composite scores style: keywords
(``must_acknowledge`` terms, politeness, action and timeframe phrases) and
length. On its own it pays a canned template, keyword stuffing or the
customer's message echoed back as much as a reply that answers the customer,
so it is gated on task grounding (:mod:`stateset_agents.core.grounding`) the
way the shipped factories gate their style terms: a turn's base is
``grounding * (STYLE_SHARE + (1 - STYLE_SHARE) * composite)``, where
``grounding`` reads the reply against the customer messages before it and the
scenario's request fields, with its ``policy_facts`` and ``product`` (and any
system or tool turns) as evidence. A failed safety check still zeroes the turn.
A custom composite (a judge, a test double) is gated only when it is a
``SupportRewardComposite``, declares ``shaping = True`` or ``ground=True`` is
passed.

The episode reward is the mean adjusted turn score, in ``[0, 1]``. The result
breakdown carries per-component means (``repetition_penalty_mean``,
``placeholder_rate``, ``must_avoid_rate``, ``unresponsive_rate``,
``grounding_mean`` when gated, ...) so a training loop can log drift toward
any of these hacks. ``base_mean`` is the composite's own (ungated) mean.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from stateset_agents.core.grounding import STYLE_SHARE, task_grounding
from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn

__all__ = [
    "DEFAULT_STOPWORDS",
    "SupportConversationReward",
    "SupportConversationRewardV3",
    "invented_specifics",
    "is_boilerplate",
    "reply_opener",
    "request_sentences",
    "content_words",
    "turn_similarity",
]

_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STEM_PREFIX = 5

#: English function words and support-chat filler that never signal new
#: content on their own ("thanks", "please", "order" is NOT here: it is
#: content in a support conversation).
DEFAULT_STOPWORDS: frozenset[str] = frozenset("""
    a about above after again against all also am an and any are aren't as at
    be because been before being below between both but by can can't cannot
    could couldn't did didn't do does doesn't doing don't down during each
    few for from further get got had hadn't has hasn't have haven't having he
    her here hers herself him himself his how i i'd i'll i'm i've if in into
    is isn't it it's its itself just let's like me more most mustn't my
    myself no nor not now of off on once only or other ought our ours
    ourselves out over own really same shan't she should shouldn't so some
    still such than that that's the their theirs them themselves then there
    there's these they they'd they'll they're they've this those through to
    too under until up very was wasn't we we'd we'll we're we've were weren't
    what what's when where which while who whom why will with won't would
    wouldn't yes yet you you'd you'll you're you've your yours yourself
    yourselves hi hello hey thanks thank please okay ok sure yeah yep hmm
    well one know think want need going get gonna wanna also already even
    sorry anything nothing something everything else seriously question
    questions thing things way much many lot maybe actually
    """.split())


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def turn_similarity(first: str, second: str) -> float:
    """``difflib`` ratio of two replies, lower-cased and whitespace-collapsed."""
    return difflib.SequenceMatcher(
        None, _normalize(first), _normalize(second), autojunk=False
    ).ratio()


def content_words(text: str, stopwords: Iterable[str] = DEFAULT_STOPWORDS) -> set[str]:
    """Lower-case content tokens (3+ characters, not stopwords, not numbers).

    A possessive ``'s`` is dropped, so "courier's" and "courier" match.
    """
    stop = stopwords if isinstance(stopwords, (set, frozenset)) else set(stopwords)
    tokens = (
        t[:-2] if t.endswith("'s") else t for t in _TOKEN_RE.findall(text.lower())
    )
    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in stop and not token.isdigit()
    }


def _engages(reply_words: set[str], new_words: set[str]) -> bool:
    if reply_words & new_words:
        return True
    prefixes = {w[:_STEM_PREFIX] for w in reply_words if len(w) >= _STEM_PREFIX}
    return any(
        len(word) >= _STEM_PREFIX and word[:_STEM_PREFIX] in prefixes
        for word in new_words
    )


def _field(turn: Any, name: str) -> Any:
    return turn.get(name) if isinstance(turn, Mapping) else getattr(turn, name, None)


def _phrases(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(p) for p in raw]


#: Scenario fields holding facts the agent was given: grounding evidence, so a
#: policy window or amount restated from them is supported, not invented.
_EVIDENCE_FIELDS = ("policy_facts", "product")


def _scores_style(composite: Any) -> bool:
    """Whether a per-turn composite scores style (keywords, length) rather
    than an outcome: the built-in ``SupportRewardComposite`` or a reward that
    declares ``shaping = True``."""
    from stateset_agents.data.customer_support_bench import SupportRewardComposite

    return (
        isinstance(composite, SupportRewardComposite)
        or getattr(composite, "shaping", False) is True
    )


class SupportConversationReward(RewardFunction):
    """Episode-level support reward with anti-hacking penalties.

    Args:
        weight: Reward weight when composed with other rewards.
        composite: Per-turn base reward; defaults to ``SupportRewardComposite()``.
        repetition_threshold: ``tau``; similarity at or below it is free.
        placeholder_factor: Multiplier for a turn with an unfilled placeholder.
        must_avoid_zeroes_episode: Zero the episode on any ``must_avoid`` hit.
        responsiveness_factor: Multiplier for a reply that ignores new content.
        responsiveness_min_new_words: New content words a follow-up needs
            before the responsiveness check applies.
        stopwords: Words ignored by the responsiveness heuristic.
        ground: Gate the per-turn composite on task grounding. ``None`` (the
            default) gates a style composite (see :func:`_scores_style`): the
            default ``SupportRewardComposite`` is gated.
        grounding_share: ``STYLE_SHARE``; the share of a grounded turn's base
            that the composite cannot take away.
    """

    name = "support_conversation_v2"

    def __init__(
        self,
        weight: float = 1.0,
        composite: RewardFunction | None = None,
        repetition_threshold: float = 0.5,
        placeholder_factor: float = 0.5,
        must_avoid_zeroes_episode: bool = True,
        responsiveness_factor: float = 0.8,
        responsiveness_min_new_words: int = 3,
        stopwords: Iterable[str] = DEFAULT_STOPWORDS,
        ground: bool | None = None,
        grounding_share: float = STYLE_SHARE,
    ) -> None:
        super().__init__(weight=weight, reward_type=RewardType.SPARSE, name=self.name)
        if not 0.0 <= repetition_threshold < 1.0:
            raise ValueError("repetition_threshold must be in [0, 1)")
        for label, factor in (
            ("placeholder_factor", placeholder_factor),
            ("responsiveness_factor", responsiveness_factor),
            ("grounding_share", grounding_share),
        ):
            if not 0.0 <= factor <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")
        if responsiveness_min_new_words < 1:
            raise ValueError("responsiveness_min_new_words must be >= 1")
        if composite is None:
            from stateset_agents.data.customer_support_bench import (
                SupportRewardComposite,
            )

            composite = SupportRewardComposite()
        self.composite = composite
        self.ground = _scores_style(composite) if ground is None else bool(ground)
        self.grounding_share = float(grounding_share)
        self.repetition_threshold = float(repetition_threshold)
        self.placeholder_factor = float(placeholder_factor)
        self.must_avoid_zeroes_episode = bool(must_avoid_zeroes_episode)
        self.responsiveness_factor = float(responsiveness_factor)
        self.responsiveness_min_new_words = int(responsiveness_min_new_words)
        self.stopwords = frozenset(stopwords)

    def repetition_multiplier(self, similarity: float) -> float:
        """``max(0, 1 - max(0, sim - tau) / (1 - tau))``."""
        tau = self.repetition_threshold
        return max(0.0, 1.0 - max(0.0, similarity - tau) / (1.0 - tau))

    def _turn_base(
        self,
        score: float,
        breakdown: Mapping[str, Any],
        text: str,
        earlier: Sequence[str],
        context: Mapping[str, Any],
    ) -> float:
        """Per-turn base score from the composite (v2: unchanged)."""
        return score

    def _turn_grounding(
        self,
        before: Sequence[ConversationTurn],
        turn: ConversationTurn,
        context: Mapping[str, Any],
    ) -> float:
        """Grounding of ``turn`` in the customer messages and request fields
        before it, with the scenario's facts and earlier system or tool turns
        as evidence (earlier assistant turns are neither)."""
        facts = [
            ConversationTurn(role="system", content=text)
            for field in _EVIDENCE_FIELDS
            for text in _phrases(context.get(field))
        ]
        history = [t for t in before if t.role != "assistant"]
        return task_grounding([*facts, *history, turn], context)

    def _gate(
        self, base: float, breakdown: Mapping[str, Any], grounding: float
    ) -> float:
        """``grounding * (share + (1 - share) * base)``; a failed safety check
        (``safety_score`` 0) is an outcome and still zeroes the turn."""
        if float(breakdown.get("safety_score", 1.0)) <= 0.0:
            return 0.0
        share = self.grounding_share
        return grounding * (share + (1.0 - share) * base)

    def _extra_turn_factors(
        self,
        text: str,
        earlier: Sequence[str],
        before: Sequence[ConversationTurn],
        context: Mapping[str, Any],
    ) -> dict[str, float]:
        """Extra per-turn multipliers by name (v2: none)."""
        return {}

    async def compute_reward(
        self,
        turns: Sequence[Any],
        context: dict[str, Any] | None = None,
    ) -> RewardResult:
        from stateset_agents.evaluation.support_judge import JudgeItem, detect_findings

        ctx = dict(context or {})
        normalized = [
            ConversationTurn(
                role=str(_field(t, "role")),
                content=str(_field(t, "content") or ""),
            )
            for t in turns
        ]
        positions = [
            i for i, t in enumerate(normalized) if t.role in ("user", "assistant")
        ]
        dialogue = [normalized[i] for i in positions]
        findings = detect_findings(
            JudgeItem(
                id=str(ctx.get("id", "episode")),
                scenario={"must_avoid": _phrases(ctx.get("must_avoid"))},
                transcript=tuple((str(t.role), t.content or "") for t in dialogue),
            )
        )
        placeholder_turns = {int(f["turn"]) - 1 for f in findings["placeholders"]}
        must_avoid_hits = len(findings["must_avoid_hits"])

        adjusted: list[float] = []
        bases: list[float] = []
        similarities: list[float] = []
        repetition_penalties: list[float] = []
        placeholder_flags: list[float] = []
        unresponsive_flags: list[float] = []
        checked = 0
        base_sums: dict[str, float] = {}
        earlier: list[str] = []
        extra_flags: dict[str, list[float]] = {}
        groundings: list[float] = []
        earlier_words: set[str] = set()
        latest_user: str | None = None

        for index, turn in enumerate(dialogue):
            text = turn.content or ""
            if turn.role == "user":
                latest_user = text
                continue
            prefix = [t for t in dialogue[:index] if t.role == "user"]
            result = await self.composite.compute_reward(prefix + [turn], ctx)
            result_breakdown = dict(getattr(result, "breakdown", None) or {})
            base = min(
                1.0,
                max(
                    0.0,
                    self._turn_base(
                        float(result.score), result_breakdown, text, earlier, ctx
                    ),
                ),
            )
            for key, value in result_breakdown.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    base_sums[key] = base_sums.get(key, 0.0) + float(value)
            score = base
            if self.ground:
                grounding = self._turn_grounding(
                    normalized[: positions[index]], turn, ctx
                )
                groundings.append(grounding)
                score = self._gate(base, result_breakdown, grounding)

            similarity = max(
                (turn_similarity(text, prior) for prior in earlier),
                default=0.0,
            )
            repetition = self.repetition_multiplier(similarity) if earlier else 1.0
            score *= repetition

            has_placeholder = index in placeholder_turns
            if has_placeholder:
                score *= self.placeholder_factor

            unresponsive = False
            reply_words = content_words(text, self.stopwords)
            if earlier and latest_user is not None:
                new_words = content_words(latest_user, self.stopwords) - earlier_words
                if len(new_words) >= self.responsiveness_min_new_words:
                    checked += 1
                    if not _engages(reply_words, new_words):
                        unresponsive = True
                        score *= self.responsiveness_factor

            for name, factor in self._extra_turn_factors(
                text, earlier, dialogue[:index], ctx
            ).items():
                extra_flags.setdefault(name, [])
                extra_flags[name].append(1.0 if factor < 1.0 else 0.0)
                score *= factor

            bases.append(base)
            adjusted.append(score)
            if earlier:
                similarities.append(similarity)
            repetition_penalties.append(1.0 - repetition)
            placeholder_flags.append(1.0 if has_placeholder else 0.0)
            unresponsive_flags.append(1.0 if unresponsive else 0.0)
            earlier.append(text)
            earlier_words |= reply_words
            latest_user = None

        if not adjusted:
            return RewardResult(score=0.0, breakdown={"no_assistant_turn": 1.0})
        return self._episode_result(
            adjusted,
            bases,
            base_sums,
            similarities,
            repetition_penalties,
            {
                "placeholder": placeholder_flags,
                "unresponsive": unresponsive_flags,
                **extra_flags,
            },
            groundings,
            must_avoid_hits=must_avoid_hits,
            checked=checked,
        )

    def _episode_result(
        self,
        adjusted: Sequence[float],
        bases: Sequence[float],
        base_sums: Mapping[str, float],
        similarities: Sequence[float],
        repetition_penalties: Sequence[float],
        flags: Mapping[str, Sequence[float]],
        groundings: Sequence[float],
        *,
        must_avoid_hits: int,
        checked: int,
    ) -> RewardResult:
        """The episode score (mean adjusted turn, zeroed on a ``must_avoid``
        hit) with its breakdown: per-turn means and ``<flag>_rate``s."""
        count = float(len(adjusted))
        turn_mean = sum(adjusted) / count
        zeroed = self.must_avoid_zeroes_episode and must_avoid_hits > 0
        episode = 0.0 if zeroed else turn_mean
        breakdown: dict[str, float] = {
            f"base_{key}": value / count for key, value in base_sums.items()
        }
        breakdown.update(
            {
                "base_mean": sum(bases) / count,
                "adjusted_turn_mean": turn_mean,
                "turn_min": min(adjusted),
                "assistant_turns": count,
                "repetition_penalty_mean": sum(repetition_penalties) / count,
                "max_similarity_mean": (
                    sum(similarities) / len(similarities) if similarities else 0.0
                ),
                "must_avoid_rate": 1.0 if must_avoid_hits else 0.0,
                "must_avoid_hits": float(must_avoid_hits),
                "responsiveness_checked": float(checked),
            }
        )
        for name, values in flags.items():
            breakdown[f"{name}_rate"] = sum(values) / count
        if groundings:
            breakdown["grounding_mean"] = sum(groundings) / len(groundings)
        explanation = (
            f"turns={len(adjusted)} base={breakdown['base_mean']:.3f} "
            f"adjusted={turn_mean:.3f} repetition={breakdown['repetition_penalty_mean']:.3f} "
            f"placeholders={breakdown['placeholder_rate']:.2f} "
            f"unresponsive={breakdown['unresponsive_rate']:.2f}"
        )
        if groundings:
            explanation += f" grounding={breakdown['grounding_mean']:.3f}"
        if zeroed:
            explanation += f" must_avoid_hits={must_avoid_hits} -> 0"
        return RewardResult(
            score=float(min(1.0, max(0.0, episode))),
            breakdown=breakdown,
            explanation=explanation,
        )


# --------------------------------------------------------------------------- #
# v3: closes the gaps the flagship v2 preflight-5 policy exploited
# --------------------------------------------------------------------------- #

_MARKDOWN_RE = re.compile(r"[*_`#>]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[a-z']+")

#: A sentence that is pure preamble: a greeting, thanks, apology or generic
#: "I understand" before the reply says anything.
_BOILERPLATE_RE = re.compile(
    r"^(?:hello|hi|hey|dear|greetings|good (?:morning|afternoon|evening)"
    r"|thanks?\b|thank you|many thanks"
    r"|i (?:completely |totally |truly |really |fully |do )?"
    r"(?:understand|appreciate|apologi[sz]e|hear you|see|can see|know)"
    r"|i'?m (?:so |very |really |truly )?sorry|i am (?:so |very |really |truly )?sorry"
    r"|we (?:appreciate|apologi[sz]e|understand)|my apologies|apologies)\b"
)

#: A sentence that asks the customer for something.
_REQUEST_RE = re.compile(
    r"\b(?:please|could you|can you|would you|kindly)\b.*\b(?:share|provide|send"
    r"|upload|reply|confirm|attach|submit|include|give|tell|let me know)\b"
)
_ORDER_REQUEST_RE = re.compile(r"\border (?:number|id|#)")
_ORDER_GIVEN_RE = re.compile(r"#\s?\d{3,}|\border (?:number |no\.? |id )?#?\s?\d{3,}")

#: Specifics the rubric's ``accuracy_no_invented_details`` forbids unless
#: they appear in the policy facts or the conversation.
_SPECIFIC_RES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(\d+(?:\s*(?:-|–|to)\s*\d+)?)\s*(?:business\s+|working\s+|calendar\s+)?"
        r"(?:hours?|hrs?|days?|weeks?|months?|minutes?|mins?)\b"
    ),
    re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)"),
    re.compile(r"\b(\d+(?:\.\d+)?)\s?%"),
    re.compile(r"\b((?:https?://|www\.)\S+)"),
    re.compile(r"\b([\w.+-]+@[\w-]+\.[\w.]+)"),
    re.compile(r"(\+?\d[\d\s().-]{8,}\d)"),
    re.compile(
        r"\b(?:case|ticket|tracking|reference|confirmation|rma)"
        r"\s*(?:number|no\.?|id|#)?\s*[:#]?\s*([a-z]*\d[\w-]{3,})"
    ),
)


def _plain(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _MARKDOWN_RE.sub(" ", text.lower())).strip()


def _sentences(text: str) -> list[str]:
    return [
        s
        for s in (_plain(part) for part in _SENTENCE_SPLIT_RE.split(text))
        if s and _WORD_RE.search(s)
    ]


def reply_opener(text: str, words: int = 2) -> str:
    """The first ``words`` lower-case words of a reply, markdown removed."""
    return " ".join(_WORD_RE.findall(_plain(text))[:words])


def is_boilerplate(sentence: str) -> bool:
    """True for a greeting/thanks/apology/"I understand" preamble sentence."""
    plain = _plain(sentence)
    return bool(_BOILERPLATE_RE.match(plain)) or len(_WORD_RE.findall(plain)) <= 2


def request_sentences(text: str) -> list[str]:
    """Sentences of ``text`` that ask the customer for something."""
    return [s for s in _sentences(text) if _REQUEST_RE.search(s)]


def invented_specifics(text: str, grounding: str) -> list[str]:
    """Durations, amounts, URLs, emails, phone and case numbers in ``text``
    that do not appear (digits/characters normalized) in ``grounding``."""
    ground = _plain(grounding)
    ground_digits = re.sub(r"\s+", "", ground.replace("–", "-").replace(" to ", "-"))
    found: list[str] = []
    lower = text.lower()
    for pattern in _SPECIFIC_RES:
        for match in pattern.finditer(lower):
            value = match.group(1).strip().rstrip(".,)")
            key = re.sub(r"\s+", "", value.replace("–", "-").replace(" to ", "-"))
            if key and key not in ground_digits and value not in ground:
                found.append(match.group(0).strip())
    return found


class SupportConversationRewardV3(SupportConversationReward):
    """Support reward v3: v2 plus the gaps the flagship v2 preflight exploited.

    Flagship v2 preflight-5 (Qwen3.5-9B, 250 GSPO steps under v2) raised the
    rule reward 0.636 -> 0.677 while the rubric-v4 judge fell 0.916 -> 0.803.
    The policy learned one template ("Hello! Thank you for ... **Next Step:**
    ..."): 93% of replies opened with a greeting (base 15%), 59% of
    conversations opened every turn the same way (base 1%), 160 of 642
    replies ran over 120 words (base 64), and invented timeframes ("within 24
    hours") and re-asked earlier requests. v3 changes, each tied to the judge
    check that regressed:

    * **Composite base without farmable keywords.** The per-turn composite's
      politeness keywords ("thank you", "please") and order-reference bonus
      get full credit on every turn, and ``must_acknowledge`` terms count
      when said in this or any earlier assistant turn, so none of them pays
      for being repeated (``empathy_not_formulaic``, repetition, length).
    * **Length** (``conciseness_reply_length``, judge-verified): a reply over
      ``max_reply_words`` (120) is multiplied by ``long_reply_factor``.
    * **Formulaic openers** (``empathy_not_formulaic``): a turn that keeps
      every assistant turn so far opening with the same two words is
      multiplied by ``formulaic_opener_factor``.
    * **Preamble** (``conciseness_key_point_early``): when the first two
      sentences are both greeting/thanks/apology boilerplate, the turn is
      multiplied by ``boilerplate_lead_factor``.
    * **Redundant requests** (``resolution_no_redundant_request``): asking for
      an order number already in the conversation, or repeating an earlier
      request sentence (``difflib`` ratio >= ``request_similarity``),
      multiplies the turn by ``redundant_request_factor``.
    * **Invented specifics** (``accuracy_no_invented_details``): durations,
      amounts, percentages, URLs, emails, phone or case numbers absent from
      the policy facts, product and customer turns multiply the turn by
      ``invented_detail_factor``.
    * **Sentence reuse** (``conciseness_no_repetition``): the share of the
      turn's sentences (5+ words) that near-duplicate (ratio >=
      ``sentence_similarity``) an earlier assistant sentence scales the turn
      by ``1 - sentence_reuse_weight * share``, on top of v2's whole-turn
      repetition penalty.

    The breakdown adds ``<name>_rate`` for each new check.
    """

    name = "support_conversation_v3"

    def __init__(
        self,
        weight: float = 1.0,
        composite: RewardFunction | None = None,
        *,
        max_reply_words: int = 120,
        long_reply_factor: float = 0.5,
        formulaic_opener_factor: float = 0.6,
        boilerplate_lead_factor: float = 0.75,
        redundant_request_factor: float = 0.7,
        request_similarity: float = 0.6,
        invented_detail_factor: float = 0.6,
        sentence_similarity: float = 0.8,
        sentence_reuse_weight: float = 0.8,
        neutralize_keyword_bonuses: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(weight=weight, composite=composite, **kwargs)
        for label, factor in (
            ("long_reply_factor", long_reply_factor),
            ("formulaic_opener_factor", formulaic_opener_factor),
            ("boilerplate_lead_factor", boilerplate_lead_factor),
            ("redundant_request_factor", redundant_request_factor),
            ("invented_detail_factor", invented_detail_factor),
            ("request_similarity", request_similarity),
            ("sentence_similarity", sentence_similarity),
            ("sentence_reuse_weight", sentence_reuse_weight),
        ):
            if not 0.0 <= factor <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")
        if max_reply_words < 1:
            raise ValueError("max_reply_words must be >= 1")
        self.max_reply_words = int(max_reply_words)
        self.long_reply_factor = float(long_reply_factor)
        self.formulaic_opener_factor = float(formulaic_opener_factor)
        self.boilerplate_lead_factor = float(boilerplate_lead_factor)
        self.redundant_request_factor = float(redundant_request_factor)
        self.request_similarity = float(request_similarity)
        self.invented_detail_factor = float(invented_detail_factor)
        self.sentence_similarity = float(sentence_similarity)
        self.sentence_reuse_weight = float(sentence_reuse_weight)
        self.neutralize_keyword_bonuses = bool(neutralize_keyword_bonuses)

    def _turn_base(
        self,
        score: float,
        breakdown: Mapping[str, Any],
        text: str,
        earlier: Sequence[str],
        context: Mapping[str, Any],
    ) -> float:
        needed = ("voice_politeness", "resolution_reference", "intent_score")
        if not self.neutralize_keyword_bonuses or not all(
            k in breakdown for k in needed
        ):
            return score
        if float(breakdown.get("safety_score", 1.0)) <= 0.0:
            return 0.0
        composite = self.composite
        voice_w = float(getattr(composite, "brand_voice_weight", 0.25))
        resolution_w = float(getattr(composite, "resolution_weight", 0.3))
        intent_w = float(getattr(composite, "intent_weight", 0.45))
        if getattr(composite, "persona", None):
            scale = 1.0 - float(getattr(composite, "persona_weight", 0.0))
            voice_w, resolution_w, intent_w = (
                voice_w * scale,
                resolution_w * scale,
                intent_w * scale,
            )
        adjusted = score
        # Politeness keywords are half of brand voice: always full credit.
        adjusted += voice_w * 0.5 * (1.0 - float(breakdown["voice_politeness"]))
        # The order-reference bonus is 20% of resolution: always full credit.
        # A raw score already clipped at 1 cannot grow, so credit only headroom.
        resolution = float(breakdown.get("resolution_score", 0.0))
        bonus = 0.2 * (1.0 - float(breakdown["resolution_reference"]))
        adjusted += resolution_w * (min(1.0, resolution + bonus) - resolution)
        # must_acknowledge counts if said in this or any earlier turn.
        terms = [str(t) for t in (context.get("must_acknowledge") or [])]
        if terms:
            said = "\n".join([*earlier, text]).lower()
            conversation_ack = sum(1 for t in terms if t.lower() in said) / len(terms)
            avoid = float(breakdown.get("intent_avoid_penalty", 0.0))
            intent = max(0.0, conversation_ack - 0.5 * avoid)
            adjusted += intent_w * (intent - float(breakdown["intent_score"]))
        return adjusted

    def _extra_turn_factors(
        self,
        text: str,
        earlier: Sequence[str],
        before: Sequence[ConversationTurn],
        context: Mapping[str, Any],
    ) -> dict[str, float]:
        factors: dict[str, float] = {}
        factors["long_reply"] = (
            self.long_reply_factor if len(text.split()) > self.max_reply_words else 1.0
        )

        opener = reply_opener(text)
        same_opener = (
            bool(earlier)
            and bool(opener)
            and all(reply_opener(prior) == opener for prior in earlier)
        )
        factors["formulaic_opener"] = (
            self.formulaic_opener_factor if same_opener else 1.0
        )

        sentences = _sentences(text)
        lead = sentences[:2]
        boilerplate = len(lead) == 2 and all(is_boilerplate(s) for s in lead)
        factors["boilerplate_lead"] = (
            self.boilerplate_lead_factor if boilerplate else 1.0
        )

        user_text = "\n".join(
            t.content or "" for t in before if t.role == "user"
        ).lower()
        requests = request_sentences(text)
        earlier_requests = [r for prior in earlier for r in request_sentences(prior)]
        redundant = any(
            _ORDER_REQUEST_RE.search(r) and _ORDER_GIVEN_RE.search(user_text)
            for r in requests
        ) or any(
            difflib.SequenceMatcher(None, r, prior, autojunk=False).ratio()
            >= self.request_similarity
            for r in requests
            for prior in earlier_requests
        )
        factors["redundant_request"] = (
            self.redundant_request_factor if redundant else 1.0
        )

        grounding = "\n".join(
            [
                user_text,
                str(context.get("user_query") or ""),
                str(context.get("product") or ""),
                *(str(f) for f in (context.get("policy_facts") or [])),
            ]
        )
        factors["invented_detail"] = (
            self.invented_detail_factor if invented_specifics(text, grounding) else 1.0
        )

        reuse = 0.0
        long_sentences = [s for s in sentences if len(s.split()) >= 5]
        if earlier and long_sentences:
            prior_sentences = [s for prior in earlier for s in _sentences(prior)]
            reused = sum(
                1
                for s in long_sentences
                if any(
                    difflib.SequenceMatcher(None, s, p, autojunk=False).ratio()
                    >= self.sentence_similarity
                    for p in prior_sentences
                )
            )
            reuse = reused / len(long_sentences)
        factors["sentence_reuse"] = max(0.0, 1.0 - self.sentence_reuse_weight * reuse)
        return factors
