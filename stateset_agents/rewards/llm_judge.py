"""
LLM-as-Judge Reward Model for RLAIF

This module provides a comprehensive LLM-as-Judge implementation for computing
rewards using language models as evaluators. This enables RLAIF (Reinforcement
Learning from AI Feedback) training.

Features:
- Multiple API provider support (OpenAI, Anthropic, local models)
- Configurable evaluation criteria and rubrics
- Batch evaluation for efficiency
- Caching to reduce API costs
- Structured output parsing
- Multiple judging strategies (single, pairwise, rubric-based)

Reference: https://arxiv.org/abs/2306.05685 (Judging LLM-as-a-Judge)
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)

LLM_JUDGE_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    OSError,
    asyncio.TimeoutError,
)


_JUDGE_PROVIDER_ALIASES = {
    "openai": "openai",
    "openai-compatible": "openai",
    "openai_compatible": "openai",
    "responses": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "local": "local",
    "vllm": "vllm",
    "together": "together",
    "auto": "auto",
}

_DEFAULT_JUDGE_MODEL_NAMES = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "local": "local-judge",
    "vllm": "gpt-4o",
    "together": "gpt-4o",
}


def _normalize_judge_provider_name(value: "JudgeProvider | str | None") -> str:
    if isinstance(value, JudgeProvider):
        return value.value
    normalized = (value or "").strip().lower().replace("_", "-")
    if not normalized:
        return "auto"
    return _JUDGE_PROVIDER_ALIASES.get(normalized, normalized)


def _infer_judge_provider_from_api_key(api_key: str | None) -> str | None:
    value = (api_key or "").strip()
    if not value:
        return None
    if value.startswith("sk-ant-"):
        return "anthropic"
    if value.startswith("sk-") or value.startswith("sk-proj-"):
        return "openai"
    return None


def _infer_judge_provider_from_model(model_name: str | None) -> str | None:
    value = (model_name or "").strip().lower()
    if not value:
        return None
    if value.startswith("claude"):
        return "anthropic"
    if value.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return None


def _resolve_judge_provider(
    provider: "JudgeProvider | str | None",
    *,
    model_name: str | None = None,
    api_key: str | None = None,
    env: dict[str, str] | None = None,
) -> "JudgeProvider":
    env = env or os.environ
    normalized = _normalize_judge_provider_name(provider)
    if normalized != "auto":
        return JudgeProvider(normalized)

    configured_provider = _normalize_judge_provider_name(
        env.get("STATESET_JUDGE_PROVIDER") or env.get("LLM_JUDGE_PROVIDER")
    )
    if configured_provider != "auto":
        return JudgeProvider(configured_provider)

    for candidate in (
        _infer_judge_provider_from_api_key(api_key),
        _infer_judge_provider_from_model(model_name),
        _infer_judge_provider_from_model(
            env.get("STATESET_JUDGE_MODEL")
            or env.get("LLM_JUDGE_MODEL")
            or env.get("MODEL_NAME")
        ),
        "openai" if env.get("OPENAI_API_KEY") or env.get("OPENAI_TOKEN") else None,
        "anthropic" if env.get("ANTHROPIC_API_KEY") else None,
    ):
        if candidate is not None:
            return JudgeProvider(candidate)

    return JudgeProvider.OPENAI


def _resolve_judge_model_name(
    provider: "JudgeProvider",
    model_name: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> str:
    if model_name:
        return model_name

    env = env or os.environ
    shared_model = env.get("STATESET_JUDGE_MODEL") or env.get("LLM_JUDGE_MODEL") or env.get(
        "MODEL_NAME"
    )
    if shared_model:
        return shared_model

    if provider == JudgeProvider.ANTHROPIC:
        return env.get("ANTHROPIC_MODEL") or _DEFAULT_JUDGE_MODEL_NAMES[provider.value]
    if provider == JudgeProvider.LOCAL:
        return env.get("LOCAL_JUDGE_MODEL") or _DEFAULT_JUDGE_MODEL_NAMES[provider.value]
    return env.get("OPENAI_MODEL") or _DEFAULT_JUDGE_MODEL_NAMES[provider.value]


def _resolve_judge_api_key(
    provider: "JudgeProvider",
    api_key: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> str | None:
    if api_key:
        return api_key

    env = env or os.environ
    generic_key = env.get("STATESET_JUDGE_API_KEY") or env.get("LLM_JUDGE_API_KEY")
    if generic_key:
        return generic_key
    if provider == JudgeProvider.ANTHROPIC:
        return env.get("ANTHROPIC_API_KEY")
    if provider in (JudgeProvider.OPENAI, JudgeProvider.TOGETHER, JudgeProvider.VLLM):
        return env.get("OPENAI_API_KEY") or env.get("OPENAI_TOKEN")
    return None


def _resolve_judge_api_base(
    provider: "JudgeProvider",
    api_base: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> str | None:
    if api_base:
        return api_base

    env = env or os.environ
    generic_base = env.get("STATESET_JUDGE_API_BASE") or env.get("LLM_JUDGE_API_BASE")
    if generic_base:
        return generic_base
    if provider in (JudgeProvider.OPENAI, JudgeProvider.VLLM, JudgeProvider.TOGETHER):
        return env.get("OPENAI_BASE_URL") or env.get("OPENAI_API_BASE")
    return None


class JudgeProvider(Enum):
    """Supported LLM providers for judging."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    VLLM = "vllm"
    TOGETHER = "together"


class EvaluationCriteria(Enum):
    """Standard evaluation criteria for LLM judges."""

    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    HONESTY = "honesty"
    CORRECTNESS = "correctness"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    CONCISENESS = "conciseness"
    CREATIVITY = "creativity"
    ENGAGEMENT = "engagement"
    PROFESSIONALISM = "professionalism"
    CUSTOM = "custom"


@dataclass
class JudgeConfig:
    """Configuration for LLM-as-Judge."""

    # Provider settings
    provider: JudgeProvider | str | None = None
    model_name: str | None = None
    api_key: str | None = None
    api_base: str | None = None

    # Evaluation settings
    criteria: list[EvaluationCriteria] = field(
        default_factory=lambda: [
            EvaluationCriteria.HELPFULNESS,
            EvaluationCriteria.CORRECTNESS,
        ]
    )
    criteria_weights: dict[str, float] | None = None

    # Scoring settings
    score_min: float = 0.0
    score_max: float = 1.0
    normalize_scores: bool = True

    # API settings
    temperature: float = 0.0  # Low temp for consistency
    max_tokens: int = 256
    timeout: float = 30.0

    # Caching
    use_cache: bool = True
    cache_size: int = 10000

    # Batch settings
    batch_size: int = 10
    max_concurrent: int = 5

    # Custom prompts
    system_prompt: str | None = None
    evaluation_template: str | None = None

    def __post_init__(self):
        resolved_provider = _resolve_judge_provider(
            self.provider,
            model_name=self.model_name,
            api_key=self.api_key,
            env=os.environ,
        )
        self.provider = resolved_provider
        self.model_name = _resolve_judge_model_name(
            resolved_provider, self.model_name, env=os.environ
        )
        self.api_key = _resolve_judge_api_key(
            resolved_provider, self.api_key, env=os.environ
        )
        self.api_base = _resolve_judge_api_base(
            resolved_provider, self.api_base, env=os.environ
        )


# Default evaluation prompts
DEFAULT_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of AI assistant responses.
You will evaluate responses based on specific criteria and provide numerical scores.
Be objective, consistent, and fair in your evaluations."""


DEFAULT_SINGLE_EVAL_TEMPLATE = """Evaluate the following response based on the criteria below.

**User Query:**
{query}

**Assistant Response:**
{response}

**Evaluation Criteria:**
{criteria}

**Instructions:**
1. Analyze the response carefully
2. Consider each criterion
3. Provide a score from {score_min} to {score_max}
4. Output ONLY a JSON object with the format: {{"score": <number>, "reasoning": "<brief explanation>"}}

Your evaluation:"""


DEFAULT_PAIRWISE_TEMPLATE = """Compare the following two responses and determine which is better.

**User Query:**
{query}

**Response A:**
{response_a}

**Response B:**
{response_b}

**Evaluation Criteria:**
{criteria}

**Instructions:**
1. Carefully compare both responses
2. Consider the criteria above
3. Output ONLY a JSON object: {{"winner": "A" or "B" or "tie", "reasoning": "<brief explanation>"}}

Your comparison:"""


DEFAULT_RUBRIC_TEMPLATE = """Evaluate the response using the following rubric.

**User Query:**
{query}

**Assistant Response:**
{response}

**Rubric:**
{rubric}

**Instructions:**
Assign a score based on the rubric above.
Output ONLY a JSON object: {{"score": <number>, "level": "<rubric level>", "reasoning": "<explanation>"}}

Your evaluation:"""


# Criteria descriptions for prompts
CRITERIA_DESCRIPTIONS = {
    EvaluationCriteria.HELPFULNESS: "Is the response helpful and addresses the user's needs?",
    EvaluationCriteria.HARMLESSNESS: "Is the response safe and free from harmful content?",
    EvaluationCriteria.HONESTY: "Is the response truthful and acknowledges uncertainty appropriately?",
    EvaluationCriteria.CORRECTNESS: "Is the information accurate and factually correct?",
    EvaluationCriteria.COHERENCE: "Is the response well-organized and logically structured?",
    EvaluationCriteria.RELEVANCE: "Does the response stay on topic and address the query?",
    EvaluationCriteria.CONCISENESS: "Is the response appropriately concise without unnecessary verbosity?",
    EvaluationCriteria.CREATIVITY: "Does the response show creativity or novel thinking?",
    EvaluationCriteria.ENGAGEMENT: "Is the response engaging and interesting to read?",
    EvaluationCriteria.PROFESSIONALISM: "Is the tone professional and appropriate?",
}


class ResultCache:
    """Simple LRU cache for evaluation results."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: dict[str, Any] = {}
        self.access_order: list[str] = []

    def _hash_key(self, query: str, response: str, criteria: str) -> str:
        """Create hash key for cache."""
        content = f"{query}|{response}|{criteria}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, response: str, criteria: str) -> Any | None:
        """Get cached result."""
        key = self._hash_key(query, response, criteria)
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, query: str, response: str, criteria: str, result: Any) -> None:
        """Set cache entry."""
        key = self._hash_key(query, response, criteria)

        # Evict if full
        while len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[key] = result
        self.access_order.append(key)


class LLMJudgeBase(ABC):
    """Abstract base class for LLM judges."""

    def __init__(self, config: JudgeConfig):
        self.config = config
        self.cache = ResultCache(config.cache_size) if config.use_cache else None

    @abstractmethod
    async def _call_api(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM API."""
        pass

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response."""
        # Try to extract JSON from response
        try:
            # Direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: extract score number
        score_match = re.search(r"(\d+\.?\d*)", response)
        if score_match:
            score = float(score_match.group(1))
            # Normalize if needed
            if score > self.config.score_max:
                score = score / 10.0  # Assume 0-10 scale
            return {"score": score, "reasoning": "Extracted from response"}

        # Default
        return {"score": 0.5, "reasoning": "Failed to parse response"}

    def _build_criteria_text(self) -> str:
        """Build criteria description text."""
        criteria_parts = []
        for criterion in self.config.criteria:
            if criterion in CRITERIA_DESCRIPTIONS:
                criteria_parts.append(
                    f"- {criterion.value}: {CRITERIA_DESCRIPTIONS[criterion]}"
                )
        return "\n".join(criteria_parts)


class OpenAIJudge(LLMJudgeBase):
    """LLM Judge using OpenAI API."""

    def __init__(self, config: JudgeConfig):
        super().__init__(config)
        self.client = None

    async def _ensure_client(self):
        """Lazily initialize OpenAI client."""
        if self.client is None:
            try:
                from openai import AsyncOpenAI

                self.client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.api_base,
                )
            except ImportError as exc:
                raise ImportError(
                    "openai package required: pip install openai"
                ) from exc

    async def _call_api(self, messages: list[dict[str, str]]) -> str:
        """Call OpenAI API."""
        await self._ensure_client()

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ),
                timeout=self.config.timeout,
            )
            return response.choices[0].message.content
        except asyncio.TimeoutError:
            logger.warning("OpenAI API call timed out")
            return '{"score": 0.5, "reasoning": "Timeout"}'
        except LLM_JUDGE_EXCEPTIONS as e:
            logger.error(f"OpenAI API error: {e}")
            return '{"score": 0.5, "reasoning": "API error"}'


class AnthropicJudge(LLMJudgeBase):
    """LLM Judge using Anthropic API."""

    def __init__(self, config: JudgeConfig):
        super().__init__(config)
        self.client = None

    async def _ensure_client(self):
        """Lazily initialize Anthropic client."""
        if self.client is None:
            try:
                from anthropic import AsyncAnthropic

                self.client = AsyncAnthropic(api_key=self.config.api_key)
            except ImportError as exc:
                raise ImportError(
                    "anthropic package required: pip install anthropic"
                ) from exc

    async def _call_api(self, messages: list[dict[str, str]]) -> str:
        """Call Anthropic API."""
        await self._ensure_client()

        # Convert messages format
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        try:
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.config.model_name,
                    system=system_msg,
                    messages=user_messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ),
                timeout=self.config.timeout,
            )
            return response.content[0].text
        except asyncio.TimeoutError:
            logger.warning("Anthropic API call timed out")
            return '{"score": 0.5, "reasoning": "Timeout"}'
        except LLM_JUDGE_EXCEPTIONS as e:
            logger.error(f"Anthropic API error: {e}")
            return '{"score": 0.5, "reasoning": "API error"}'


class LocalJudge(LLMJudgeBase):
    """LLM Judge using local model."""

    def __init__(self, config: JudgeConfig, model=None, tokenizer=None):
        super().__init__(config)
        self.model = model
        self.tokenizer = tokenizer

    async def _call_api(self, messages: list[dict[str, str]]) -> str:
        """Generate using local model."""
        if self.model is None or self.tokenizer is None:
            return '{"score": 0.5, "reasoning": "Local model not configured"}'

        try:
            import torch

            # Format prompt
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature or 0.1,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            return response

        except LLM_JUDGE_EXCEPTIONS as e:
            logger.error(f"Local model error: {e}")
            return '{"score": 0.5, "reasoning": "Generation error"}'


class LLMJudge:
    """
    Main LLM-as-Judge reward model.

    This class provides a unified interface for using LLMs to evaluate
    response quality, enabling RLAIF training.

    Example:
        ```python
        config = JudgeConfig(
            provider=JudgeProvider.OPENAI,
            model_name="gpt-4o",
            criteria=[EvaluationCriteria.HELPFULNESS, EvaluationCriteria.CORRECTNESS],
        )
        judge = LLMJudge(config)

        score = await judge.evaluate(
            query="What is machine learning?",
            response="Machine learning is..."
        )
        ```
    """

    def __init__(
        self,
        config: JudgeConfig | None = None,
        provider: JudgeProvider | str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        # Build config from parameters
        if config is None:
            config = JudgeConfig(
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                **kwargs,
            )

        self.config = config

        # Initialize appropriate backend
        if config.provider in (JudgeProvider.OPENAI, JudgeProvider.TOGETHER, JudgeProvider.VLLM):
            self.backend = OpenAIJudge(config)
        elif config.provider == JudgeProvider.ANTHROPIC:
            self.backend = AnthropicJudge(config)
        elif config.provider == JudgeProvider.LOCAL:
            self.backend = LocalJudge(config)
        else:
            self.backend = OpenAIJudge(config)

        # Cache
        self.cache = self.backend.cache

    async def evaluate(
        self,
        query: str,
        response: str,
        reference: str | None = None,
    ) -> float:
        """
        Evaluate a single response.

        Args:
            query: User query/prompt
            response: Assistant response to evaluate
            reference: Optional reference/gold response

        Returns:
            Normalized score in [0, 1]
        """
        # Check cache
        criteria_key = ",".join([c.value for c in self.config.criteria])
        if self.cache:
            cached = self.cache.get(query, response, criteria_key)
            if cached is not None:
                return cached

        # Build evaluation prompt
        system_prompt = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT
        template = self.config.evaluation_template or DEFAULT_SINGLE_EVAL_TEMPLATE

        criteria_text = self.backend._build_criteria_text()

        user_prompt = template.format(
            query=query,
            response=response,
            criteria=criteria_text,
            score_min=self.config.score_min,
            score_max=self.config.score_max,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Call API
        result_text = await self.backend._call_api(messages)

        # Parse result
        result = self.backend._parse_json_response(result_text)
        score = float(result.get("score", 0.5))

        # Normalize
        if self.config.normalize_scores:
            score = (score - self.config.score_min) / (
                self.config.score_max - self.config.score_min
            )
            score = max(0.0, min(1.0, score))

        # Cache
        if self.cache:
            self.cache.set(query, response, criteria_key, score)

        return score

    async def evaluate_batch(
        self,
        queries: list[str],
        responses: list[str],
    ) -> list[float]:
        """
        Evaluate a batch of responses.

        Args:
            queries: List of user queries
            responses: List of responses to evaluate

        Returns:
            List of scores
        """
        # Create tasks
        tasks = [self.evaluate(q, r) for q, r in zip(queries, responses, strict=False)]

        # Run with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def bounded_task(task):
            async with semaphore:
                return await task

        bounded_tasks = [bounded_task(t) for t in tasks]
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

        # Handle errors
        scores = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Evaluation error: {result}")
                scores.append(0.5)  # Default on error
            else:
                scores.append(result)

        return scores

    async def compare_pairwise(
        self,
        query: str,
        response_a: str,
        response_b: str,
    ) -> tuple[str, float, float]:
        """
        Compare two responses pairwise.

        Args:
            query: User query
            response_a: First response
            response_b: Second response

        Returns:
            Tuple of (winner, score_a, score_b)
        """
        system_prompt = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT
        criteria_text = self.backend._build_criteria_text()

        user_prompt = DEFAULT_PAIRWISE_TEMPLATE.format(
            query=query,
            response_a=response_a,
            response_b=response_b,
            criteria=criteria_text,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result_text = await self.backend._call_api(messages)
        result = self.backend._parse_json_response(result_text)

        winner = result.get("winner", "tie").upper()

        # Assign scores based on winner
        if winner == "A":
            return "A", 1.0, 0.0
        elif winner == "B":
            return "B", 0.0, 1.0
        else:
            return "tie", 0.5, 0.5


# Convenience function for creating reward functions
def create_llm_judge_reward(
    provider: str | JudgeProvider | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    criteria: list[str] | None = None,
    **kwargs,
) -> Callable[[str, str], float]:
    """
    Create an LLM judge reward function for use with trainers.

    Args:
        provider: API provider. When omitted, resolves from env/model/api-key hints.
        model_name: Model to use for judging. When omitted, resolves from provider/env.
        api_key: API key (uses env var if not provided)
        criteria: List of evaluation criteria
        **kwargs: Additional config parameters

    Returns:
        Async reward function compatible with trainers
    """
    # Parse criteria
    if criteria:
        eval_criteria = [
            EvaluationCriteria(c) if isinstance(c, str) else c for c in criteria
        ]
    else:
        eval_criteria = [EvaluationCriteria.HELPFULNESS, EvaluationCriteria.CORRECTNESS]

    # Build config
    config = JudgeConfig(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        criteria=eval_criteria,
        **kwargs,
    )

    judge = LLMJudge(config)

    async def reward_fn(prompt: str, completion: str) -> float:
        return await judge.evaluate(prompt, completion)

    return reward_fn
