"""Component implementations for multi-objective rewards."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

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
        return os.getenv("ANTHROPIC_MODEL") or os.getenv("STATESET_JUDGE_MODEL") or os.getenv(
            "MODEL_NAME"
        ) or _DEFAULT_MODEL_NAMES["anthropic"]
    return os.getenv("OPENAI_MODEL") or os.getenv("STATESET_JUDGE_MODEL") or os.getenv(
        "MODEL_NAME"
    ) or _DEFAULT_MODEL_NAMES["openai"]


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
        self.min_length = min_length
        self.max_length = max_length
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
        escaped = [re.escape(keyword) for keyword in self.empathy_keywords]
        self._keyword_pattern = re.compile(
            r"(?:" + "|".join(escaped) + r")",
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

        total_matches = 0
        total_words = 0
        for response in assistant_responses:
            content = response.get("content", "")
            total_words += len(content.split())
            total_matches += len(self._keyword_pattern.findall(content))

        if total_words == 0:
            return 0.0

        empathy_density = total_matches / max(1, len(assistant_responses))
        return min(1.0, empathy_density / 2.0)


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
        escaped = [re.escape(keyword) for keyword in self.action_keywords]
        self._keyword_pattern = re.compile(
            r"(?:" + "|".join(escaped) + r")",
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

        total_matches = 0
        for response in assistant_responses:
            total_matches += len(
                self._keyword_pattern.findall(response.get("content", ""))
            )

        action_density = total_matches / max(1, len(assistant_responses))
        return min(1.0, action_density / 3.0)


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
        self.unprofessional_indicators = unprofessional_indicators or [
            "whatever",
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
            "frustrated",
            "angry",
        ]
        professional = [re.escape(keyword) for keyword in self.professional_indicators]
        self._professional_pattern = re.compile(
            r"(?:" + "|".join(professional) + r")",
            re.IGNORECASE,
        )
        unprofessional = [
            re.escape(keyword) for keyword in self.unprofessional_indicators
        ]
        self._unprofessional_pattern = re.compile(
            r"(?:" + "|".join(unprofessional) + r")",
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
        super().__init__(name, weight, **kwargs)
        self.provider = _normalize_model_provider(
            provider or os.getenv("STATESET_JUDGE_PROVIDER")
        )
        self._configured_model_name = model_name
        self.model_name = _resolve_model_name(self.provider, model_name)
        self.judge_prompt_template = judge_prompt_template or (
            "You are an impartial judge evaluating the quality of an AI assistant's response.\n"
            "User Query: {user_query}\n"
            "Assistant Response: {assistant_response}\n\n"
            "Rate the response on a scale of 0 to 1 (float), focusing on helpfulness, accuracy, and tone.\n"
            "Output ONLY the score."
        )
        self.api_client = api_client

    async def compute_score(
        self, turns: list[dict[str, Any]], context: dict[str, Any] | None = None
    ) -> float:
        if not turns:
            return 0.0

        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        user_turns = [t for t in turns if t.get("role") == "user"]
        if not assistant_responses or not user_turns:
            return 0.0

        assistant_response = assistant_responses[-1].get("content", "")
        user_query = user_turns[-1].get("content", "")
        prompt = self.judge_prompt_template.format(
            user_query=user_query,
            assistant_response=assistant_response,
        )

        try:
            if self.api_client:
                effective_provider = _resolve_model_provider(
                    self.provider, self.api_client
                )
                self.model_name = _resolve_model_name(
                    effective_provider, self._configured_model_name
                )
                if effective_provider == "openai":
                    response = await self._call_openai_api(prompt)
                    score = self._parse_score_from_response(response)
                elif effective_provider == "anthropic":
                    response = await self._call_anthropic_api(prompt)
                    score = self._parse_score_from_response(response)
                elif hasattr(self.api_client, "generate"):
                    response = await self.api_client.generate(prompt)
                    score = self._parse_score_from_response(response)
                else:
                    logger.warning(
                        "Unknown API client configuration for provider %s, using heuristic",
                        effective_provider,
                    )
                    score = min(1.0, len(assistant_response) / 200.0)
            else:
                score = min(1.0, len(assistant_response) / 200.0)

            return max(0.0, min(1.0, score))
        except MULTI_REWARD_EXCEPTIONS as exc:
            logger.error("Model judge failed: %s", exc)
            return 0.5

    async def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API for scoring."""
        try:
            if hasattr(self.api_client, "chat"):
                response = await self.api_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0,
                )
                return response.choices[0].message.content

            response = await self.api_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=10,
                temperature=0.0,
            )
            return response.choices[0].text
        except MULTI_REWARD_EXCEPTIONS as exc:
            logger.error("OpenAI API call failed: %s", exc)
            raise

    async def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API for scoring."""
        try:
            response = await self.api_client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.content[0].text
        except MULTI_REWARD_EXCEPTIONS as exc:
            logger.error("Anthropic API call failed: %s", exc)
            raise

    def _parse_score_from_response(self, response: str) -> float:
        """Parse a numerical score from a model response."""
        try:
            numbers = re.findall(r"\d+\.?\d*", response)
            if numbers:
                score = float(numbers[0])
                if score > 1.0:
                    score = score / 10.0
                return min(1.0, max(0.0, score))
            return 0.5
        except (IndexError, ValueError) as exc:
            logger.warning(
                "Failed to parse score from response %r: %s",
                response,
                exc,
            )
            return 0.5


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
