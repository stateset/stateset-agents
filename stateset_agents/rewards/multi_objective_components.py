"""Component implementations for multi-objective rewards."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from stateset_agents.core.grounding import task_grounding
from stateset_agents.core.reward_base import RewardUnavailableError

logger = logging.getLogger(__name__)

MULTI_REWARD_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    OSError,
    asyncio.TimeoutError,
)


#: Judge call failures that mean "no verdict available" (network, timeout,
#: provider errors, an unparseable reply), as opposed to programming errors.
_JUDGE_OUTAGES = (RuntimeError, ValueError, OSError, asyncio.TimeoutError)


_MODEL_PROVIDER_ALIASES = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "openai-compatible": "openai",
    "openai_compatible": "openai",
    "responses": "openai",
    "auto": "auto",
}

_DEFAULT_MODEL_NAMES = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
}


def _normalize_model_provider(value: str | None) -> str:
    normalized = (value or "").strip().lower().replace("_", "-")
    if not normalized:
        return "auto"
    return _MODEL_PROVIDER_ALIASES.get(normalized, normalized)


def _resolve_model_provider(provider: str | None, api_client: Any | None = None) -> str:
    normalized = _normalize_model_provider(provider)
    if normalized != "auto":
        return normalized
    if api_client is None:
        return "auto"
    if hasattr(api_client, "messages"):
        return "anthropic"
    if hasattr(api_client, "chat") or hasattr(api_client, "completions"):
        return "openai"
    if hasattr(api_client, "generate"):
        return "generate"
    return "auto"


def _resolve_model_name(provider: str, configured_model_name: str | None = None) -> str:
    if configured_model_name:
        return configured_model_name
    if provider == "anthropic":
        return (
            os.getenv("ANTHROPIC_MODEL")
            or os.getenv("STATESET_JUDGE_MODEL")
            or os.getenv("MODEL_NAME")
            or _DEFAULT_MODEL_NAMES["anthropic"]
        )
    return (
        os.getenv("OPENAI_MODEL")
        or os.getenv("STATESET_JUDGE_MODEL")
        or os.getenv("MODEL_NAME")
        or _DEFAULT_MODEL_NAMES["openai"]
    )


@dataclass
class RewardComponent:
    """Individual reward component with weight and configuration."""

    name: str
    weight: float
    config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")


class BaseRewardComponent(ABC):
    """Base class for individual reward components."""

    def __init__(self, name: str, weight: float = 1.0, **kwargs):
        self.name = name
        self.weight = weight
        self.config = kwargs

    @abstractmethod
    async def compute_score(
        self, turns: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> float:
        """Compute score for this component."""

    def get_info(self) -> dict[str, Any]:
        """Get component information."""
        return {"name": self.name, "weight": self.weight, "config": self.config}


class LengthRewardComponent(BaseRewardComponent):
    """Reward component based on response length."""

    def __init__(
        self,
        name: str = "length",
        weight: float = 0.1,
        min_length: int = 10,
        max_length: int = 200,
        optimal_range: tuple[int, int] = (50, 150),
        **kwargs,
    ):
        super().__init__(name, weight, **kwargs)
        # The optimal range must lie inside [min_length, max_length]; a range
        # reaching past max_length would score its own top end as too long.
        self.min_length = min(min_length, optimal_range[0])
        self.max_length = max(max_length, optimal_range[1])
        self.optimal_range = optimal_range

    async def compute_score(
        self, turns: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> float:
        if not turns:
            return 0.0

        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_responses:
            return 0.0

        last_response = assistant_responses[-1].get("content", "")
        word_count = len(last_response.split())

        if word_count < self.min_length:
            return 0.2
        if word_count > self.max_length:
            return 0.3
        if self.optimal_range[0] <= word_count <= self.optimal_range[1]:
            return 1.0
        if word_count < self.optimal_range[0]:
            return 0.5 + 0.5 * (word_count - self.min_length) / (
                self.optimal_range[0] - self.min_length
            )
        return 0.5 + 0.5 * (self.max_length - word_count) / (
            self.max_length - self.optimal_range[1]
        )


def _keyword_pattern(keywords: list[str]) -> re.Pattern[str]:
    """Whole-word keyword matcher; a keyword may be a stem ("frustrat")."""
    escaped = [re.escape(keyword) for keyword in keywords]
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\w*", re.IGNORECASE)


def _distinct_keyword_score(pattern: re.Pattern[str], texts: list[str]) -> float:
    """Mean over replies of ``1 - 0.5 ** distinct_keywords``.

    Whole-word matches only ("hate" never matches inside "whatever"), each
    distinct keyword counted once per reply, with diminishing returns: one
    keyword is worth 0.5, two 0.75, three 0.875, so repeating or piling up
    keywords never saturates the score.
    """
    if not texts:
        return 0.0
    scores = []
    for text in texts:
        distinct = {match.lower() for match in pattern.findall(text)}
        scores.append(1.0 - 0.5 ** len(distinct))
    return sum(scores) / len(scores)


class EmpathyRewardComponent(BaseRewardComponent):
    """Reward component based on empathy indicators."""

    def __init__(
        self,
        name: str = "empathy",
        weight: float = 0.2,
        empathy_keywords: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(name, weight, **kwargs)
        self.empathy_keywords = empathy_keywords or [
            "understand",
            "sorry",
            "apologize",
            "feel",
            "frustrat",
            "concern",
            "worry",
            "help",
            "support",
            "appreciate",
            "thank",
            "welcome",
            "please",
            "certainly",
            "absolutely",
            "definitely",
            "of course",
        ]
        self._keyword_pattern = _keyword_pattern(self.empathy_keywords)

    async def compute_score(
        self, turns: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> float:
        replies = [
            str(t.get("content") or "") for t in turns if t.get("role") == "assistant"
        ]
        return _distinct_keyword_score(self._keyword_pattern, replies)


class ActionOrientedRewardComponent(BaseRewardComponent):
    """Reward component based on action-oriented language."""

    def __init__(
        self,
        name: str = "action_oriented",
        weight: float = 0.2,
        action_keywords: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(name, weight, **kwargs)
        self.action_keywords = action_keywords or [
            "will",
            "can",
            "let me",
            "i'll",
            "here's",
            "you can",
            "try",
            "suggest",
            "recommend",
            "solution",
            "step",
            "first",
            "then",
            "next",
            "process",
            "procedure",
            "method",
            "approach",
            "way",
        ]
        self._keyword_pattern = _keyword_pattern(self.action_keywords)

    async def compute_score(
        self, turns: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> float:
        replies = [
            str(t.get("content") or "") for t in turns if t.get("role") == "assistant"
        ]
        return _distinct_keyword_score(self._keyword_pattern, replies)


class SimilarityRewardComponent(BaseRewardComponent):
    """Reward component based on similarity to expected responses."""

    def __init__(
        self,
        name: str = "similarity",
        weight: float = 0.3,
        expected_responses: list[str] | None = None,
        similarity_threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(name, weight, **kwargs)
        self.expected_responses = expected_responses or []
        self.similarity_threshold = similarity_threshold

    async def compute_score(
        self, turns: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> float:
        if not turns or not self.expected_responses:
            return 0.5

        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_responses:
            return 0.0

        max_similarity = 0.0
        for response in assistant_responses:
            response_words = set(response.get("content", "").lower().split())
            for expected in self.expected_responses:
                expected_words = set(expected.lower().split())
                if not expected_words:
                    continue
                union = response_words | expected_words
                if union:
                    intersection = response_words & expected_words
                    max_similarity = max(
                        max_similarity,
                        len(intersection) / len(union),
                    )

        if max_similarity >= self.similarity_threshold:
            return min(1.0, max_similarity / self.similarity_threshold)
        return max_similarity / self.similarity_threshold * 0.5


class ProfessionalismRewardComponent(BaseRewardComponent):
    """Reward component based on professionalism indicators."""

    def __init__(
        self,
        name: str = "professionalism",
        weight: float = 0.2,
        professional_indicators: list[str] | None = None,
        unprofessional_indicators: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(name, weight, **kwargs)
        self.professional_indicators = professional_indicators or [
            "please",
            "thank you",
            "may i",
            "would you",
            "could you",
            "i'd be happy",
            "certainly",
            "absolutely",
            "professional",
            "assistance",
            "service",
            "support",
            "help",
            "resolve",
        ]
        # Dismissive phrases rather than the bare word "whatever", which is
        # also how helpful replies offer flexibility ("whatever you need").
        self.unprofessional_indicators = unprofessional_indicators or [
            "not my problem",
            "deal with it",
            "dunno",
            "idk",
            "lol",
            "omg",
            "wtf",
            "stupid",
            "dumb",
            "sucks",
            "hate",
            "annoying",
            "angry",
        ]
        professional = [re.escape(keyword) for keyword in self.professional_indicators]
        self._professional_pattern = re.compile(
            r"(?:" + "|".join(professional) + r")",
            re.IGNORECASE,
        )
        # Whole words only: "hate" must not match inside "whatever". Naming
        # the customer's feelings ("I understand you're frustrated") is
        # empathy, not unprofessional language, so it is not listed.
        unprofessional = [
            re.escape(keyword) for keyword in self.unprofessional_indicators
        ]
        self._unprofessional_pattern = re.compile(
            r"\b(?:" + "|".join(unprofessional) + r")\b",
            re.IGNORECASE,
        )

    async def compute_score(
        self, turns: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> float:
        if not turns:
            return 0.0

        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_responses:
            return 0.0

        professional_count = 0
        unprofessional_count = 0
        for response in assistant_responses:
            content = response.get("content", "")
            professional_count += len(self._professional_pattern.findall(content))
            unprofessional_count += len(self._unprofessional_pattern.findall(content))

        total_responses = len(assistant_responses)
        professional_density = professional_count / max(1, total_responses)
        unprofessional_penalty = unprofessional_count / max(1, total_responses)
        score = min(1.0, professional_density / 2.0) - unprofessional_penalty
        return max(0.0, score)


class ModelBasedRewardComponent(BaseRewardComponent):
    """Reward component that uses a judge model to evaluate response quality."""

    def __init__(
        self,
        name: str = "model_judge",
        weight: float = 0.5,
        judge_prompt_template: str | None = None,
        api_client: Any | None = None,
        **kwargs,
    ):
        provider = kwargs.pop("provider", None)
        model_name = kwargs.pop("model_name", None)
        strict = bool(kwargs.pop("strict", False))
        super().__init__(name, weight, **kwargs)
        #: Raise instead of scoring by reply length when no client is set.
        self.strict = strict
        self._warned_no_client = False
        self.provider = _normalize_model_provider(
            provider or os.getenv("STATESET_JUDGE_PROVIDER")
        )
        self._configured_model_name = model_name
        self.model_name = _resolve_model_name(self.provider, model_name)
        self.judge_prompt_template = judge_prompt_template or (
            "You are an impartial judge evaluating the quality of an AI assistant's response.\n"
            "User Query: {user_query}\n"
            "Assistant Response: {assistant_response}\n\n"
            "Rate the response from 0 to 1, focusing on helpfulness, accuracy, and tone.\n"
            'Output ONLY a JSON object: {{"score": <number from 0 to 1>}}'
        )
        self.api_client = api_client

    async def compute_score(
        self, turns: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> float:
        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_responses:
            return 0.0  # no reply, nothing to credit
        # The judge always sees the reply; the request comes from the last
        # user turn, else the context's user_query. A reply is never scored
        # 0.0 in place of a verdict because the request was not in ``turns``.
        user_turns = [t for t in turns if t.get("role") == "user"]
        if user_turns:
            user_query = str(user_turns[-1].get("content", "") or "")
        else:
            query = (context or {}).get("user_query")
            user_query = query if isinstance(query, str) else ""

        assistant_response = assistant_responses[-1].get("content", "")
        prompt = self.judge_prompt_template.format(
            user_query=user_query,
            assistant_response=assistant_response,
        )

        client = self.api_client
        if client is None:
            if self.strict:
                raise RuntimeError(
                    f"{self.name}: no api_client configured; refusing to "
                    "substitute a length heuristic for the judge (strict=True)"
                )
            if not self._warned_no_client:
                self._warned_no_client = True
                logger.warning(
                    "%s has no api_client; scoring replies by length "
                    "(len/200) instead of a judge. Pass api_client=... or "
                    "strict=True to fail instead.",
                    self.name,
                )
            return max(0.0, min(1.0, len(assistant_response) / 200.0))

        effective_provider = _resolve_model_provider(self.provider, client)
        self.model_name = _resolve_model_name(
            effective_provider, self._configured_model_name
        )
        if effective_provider not in ("openai", "anthropic") and not hasattr(
            client, "generate"
        ):
            # A misconfigured client is a bug, not an outage.
            raise TypeError(
                f"{self.name}: api_client for provider {effective_provider!r} "
                "has no chat/completions/messages/generate interface"
            )
        try:
            if effective_provider == "openai":
                response = await self._call_openai_api(prompt)
            elif effective_provider == "anthropic":
                response = await self._call_anthropic_api(prompt)
            else:
                response = await client.generate(prompt)
            score = self._parse_score_from_response(response)
        except _JUDGE_OUTAGES as exc:
            # Fail closed: the composer leaves this component out and spreads
            # its weight; a neutral 0.5 would be trained on as a verdict.
            raise RewardUnavailableError(
                f"{self.name}: model judge failed: {exc}"
            ) from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RewardUnavailableError(
                f"{self.name}: judge score {score} is outside [0, 1]"
            )
        return score

    async def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API for scoring."""
        client = self.api_client
        if client is None:
            raise RuntimeError("OpenAI client is not configured")
        try:
            if hasattr(client, "chat"):
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=0.0,
                )
                content: object = response.choices[0].message.content
                return content if isinstance(content, str) else str(content)

            response = await client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=20,
                temperature=0.0,
            )
            text: object = response.choices[0].text
            return text if isinstance(text, str) else str(text)
        except MULTI_REWARD_EXCEPTIONS as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise

    async def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API for scoring."""
        client = self.api_client
        if client is None:
            raise RuntimeError("Anthropic client is not configured")
        try:
            response = await client.messages.create(
                model=self.model_name,
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text: object = response.content[0].text
            return text if isinstance(text, str) else str(text)
        except MULTI_REWARD_EXCEPTIONS as exc:
            logger.error("Anthropic API call failed: %s", exc)
            raise

    def _parse_score_from_response(self, response: str) -> float:
        """Read the score from the judge's reply: exactly ``{"score": x}``.

        The reply must be one JSON object whose ``"score"`` is a number in
        [0, 1]; anything else (prose, a bare number, "7/10", "85/100", a
        negative or out-of-range score) raises ``ValueError``, which
        ``compute_score`` reports as ``RewardUnavailableError``. No number is
        salvaged from free text and nothing is rescaled or clamped.
        """
        try:
            parsed = json.loads(str(response).strip())
        except json.JSONDecodeError as error:
            raise ValueError(f"judge reply is not strict JSON: {response!r}") from error
        score = parsed.get("score") if isinstance(parsed, dict) else None
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"judge reply has no numeric score: {response!r}")
        if not (math.isfinite(score) and 0.0 <= score <= 1.0):
            raise ValueError(f"judge score {score} is outside [0, 1]")
        return float(score)


class ReasoningRewardComponent(BaseRewardComponent):
    """Reward component based on reasoning traces in turn metadata."""

    def __init__(
        self,
        name: str = "reasoning",
        weight: float = 0.3,
        min_length: int = 50,
        optimal_length: int = 500,
        **kwargs,
    ):
        super().__init__(name, weight, **kwargs)
        self.min_length = min_length
        self.optimal_length = optimal_length

    async def compute_score(
        self, turns: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> float:
        if not turns:
            return 0.0

        assistant_turns = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_turns:
            return 0.0

        metadata = assistant_turns[-1].get("metadata", {})
        reasoning = metadata.get("reasoning", "") if isinstance(metadata, dict) else ""
        if not reasoning:
            return 0.0

        length = len(reasoning.split())
        if length < self.min_length:
            return 0.2 * (length / self.min_length)
        if length <= self.optimal_length:
            return 0.2 + 0.8 * (
                (length - self.min_length) / (self.optimal_length - self.min_length)
            )
        return 1.0


__all__ = [
    "ActionOrientedRewardComponent",
    "BaseRewardComponent",
    "EmpathyRewardComponent",
    "LengthRewardComponent",
    "MULTI_REWARD_EXCEPTIONS",
    "ModelBasedRewardComponent",
    "ProfessionalismRewardComponent",
    "ReasoningRewardComponent",
    "RewardComponent",
    "SimilarityRewardComponent",
]


class GroundedComponent(BaseRewardComponent):
    """A style component that only pays for grounded replies.

    ``score = grounding * (share + (1 - share) * component_score)`` (see
    ``stateset_agents.core.grounding``): a canned template or keyword stuffing
    has no grounding and scores 0. Name and weight are the wrapped
    component's, so the composite reports it under its own name.
    """

    def __init__(self, component: BaseRewardComponent, share: float = 0.5):
        super().__init__(component.name, component.weight)
        self.component = component
        self.share = float(share)
        self.config = {"grounded": True, **getattr(component, "config", {})}

    async def compute_score(
        self, turns: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> float:
        score = await self.component.compute_score(turns, context)
        grounding = task_grounding(turns, context)
        return grounding * (self.share + (1.0 - self.share) * score)


def ground_components(
    components: list[BaseRewardComponent], share: float = 0.5
) -> list[BaseRewardComponent]:
    """Wrap each component in :class:`GroundedComponent`."""
    return [GroundedComponent(component, share) for component in components]
