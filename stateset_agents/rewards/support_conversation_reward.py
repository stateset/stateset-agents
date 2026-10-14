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

The episode reward is the mean adjusted turn score, in ``[0, 1]``. The result
breakdown carries per-component means (``repetition_penalty_mean``,
``placeholder_rate``, ``must_avoid_rate``, ``unresponsive_rate``, ...) so a
training loop can log drift toward any of these hacks.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from stateset_agents.core.reward_base import RewardFunction, RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn

__all__ = [
    "DEFAULT_STOPWORDS",
    "SupportConversationReward",
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
    ) -> None:
        super().__init__(weight=weight, reward_type=RewardType.SPARSE, name=self.name)
        if not 0.0 <= repetition_threshold < 1.0:
            raise ValueError("repetition_threshold must be in [0, 1)")
        for label, factor in (
            ("placeholder_factor", placeholder_factor),
            ("responsiveness_factor", responsiveness_factor),
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
        dialogue = [t for t in normalized if t.role in ("user", "assistant")]
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
        earlier_words: set[str] = set()
        latest_user: str | None = None

        for index, turn in enumerate(dialogue):
            text = turn.content or ""
            if turn.role == "user":
                latest_user = text
                continue
            prefix = [t for t in dialogue[:index] if t.role == "user"]
            result = await self.composite.compute_reward(prefix + [turn], ctx)
            base = min(1.0, max(0.0, float(result.score)))
            for key, value in (getattr(result, "breakdown", None) or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    base_sums[key] = base_sums.get(key, 0.0) + float(value)
            score = base

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
                "placeholder_rate": sum(placeholder_flags) / count,
                "must_avoid_rate": 1.0 if must_avoid_hits else 0.0,
                "must_avoid_hits": float(must_avoid_hits),
                "unresponsive_rate": sum(unresponsive_flags) / count,
                "responsiveness_checked": float(checked),
            }
        )
        explanation = (
            f"turns={len(adjusted)} base={breakdown['base_mean']:.3f} "
            f"adjusted={turn_mean:.3f} repetition={breakdown['repetition_penalty_mean']:.3f} "
            f"placeholders={breakdown['placeholder_rate']:.2f} "
            f"unresponsive={breakdown['unresponsive_rate']:.2f}"
        )
        if zeroed:
            explanation += f" must_avoid_hits={must_avoid_hits} -> 0"
        return RewardResult(
            score=float(min(1.0, max(0.0, episode))),
            breakdown=breakdown,
            explanation=explanation,
        )
