"""
LLM Judge Reward Function for GRPO Agent Framework

This module provides sophisticated reward computation using external LLM Judge
to evaluate trajectory quality based on customizable rubrics.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

try:
    from litellm import completion
    from litellm.types.utils import ModelResponse
    from openai.types.chat.chat_completion_message_param import (
        ChatCompletionMessageParam,
    )

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    ChatCompletionMessageParam = Dict[str, Any]


from stateset_agents.core import reward as core_reward

RewardFunction = core_reward.RewardFunction
RewardResult = core_reward.RewardResult

import utils.cache

CacheService = utils.cache.CacheService

import utils.monitoring

MonitoringService = utils.monitoring.MonitoringService
logger = logging.getLogger(__name__)


class TrajectoryScore(BaseModel):
    """Score for a single trajectory"""

    trajectory_id: str = Field(description="The id of the trajectory being scored.")
    explanation: str = Field(
        description="A short description of the trajectory's performance."
    )
    score: float = Field(description="A score between 0 and 1.")


class RulerResponse(BaseModel):
    """Response from the ruler LLM judge"""

    scores: List[TrajectoryScore] = Field(description="The scores for each trajectory.")


# Pre-defined rubrics for different domains
RUBRICS = {
    "default": dedent(
        """
        - A trajectory that achieves its goal should always get a significantly higher score than a trajectory that does not achieve its goal.
        - A trajectory that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a trajectory that achieves its goal less efficiently.
        - If one trajectory is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
        - You may give some partial credit for a trajectory that makes progress towards its goal but does not complete it.
    """
    ),
    "customer_service": dedent(
        """
        Score each assistant response based on:
        - Helpfulness: Does it directly address the customer's query?
        - Empathy: Shows understanding of customer's situation?
        - Clarity: Clear and concise communication?
        - Professionalism: Polite and brand-appropriate tone?
        - Completeness: Provides all necessary information/action steps?
        Give higher scores to responses that excel in multiple categories.
    """
    ),
    "technical_support": dedent(
        """
        Evaluate technical support responses on:
        - Accuracy: Is the technical information correct?
        - Clarity: Are complex concepts explained clearly?
        - Step-by-step guidance: Are instructions easy to follow?
        - Problem resolution: Does it solve the user's issue?
        - Alternative solutions: Are fallback options provided?
    """
    ),
    "sales_assistant": dedent(
        """
        Rate sales interactions based on:
        - Product knowledge: Accurate information about products/services?
        - Customer needs identification: Does it understand what the customer wants?
        - Value proposition: Clearly communicates benefits?
        - Objection handling: Addresses concerns effectively?
        - Call to action: Guides customer to next steps?
    """
    ),
    "educational": dedent(
        """
        Score educational responses on:
        - Correctness: Is the information accurate?
        - Pedagogical approach: Is it appropriate for the learner's level?
        - Engagement: Does it maintain interest?
        - Examples: Are concrete examples provided?
        - Learning reinforcement: Does it check understanding?
    """
    ),
    "creative_writing": dedent(
        """
        Evaluate creative writing responses on:
        - Creativity: Is the content original and imaginative?
        - Coherence: Does the narrative flow logically?
        - Engagement: Is it interesting and captivating?
        - Style: Is the writing style appropriate and well-crafted?
        - Completion: Does it fulfill the creative prompt?
    """
    ),
    "code_assistance": dedent(
        """
        Rate code assistance responses on:
        - Correctness: Is the code syntactically and logically correct?
        - Efficiency: Is the solution optimized and well-structured?
        - Clarity: Are explanations clear and educational?
        - Best practices: Does it follow coding standards?
        - Completeness: Does it address all aspects of the problem?
    """
    ),
}


class RulerRewardFunction(RewardFunction):
    """
    LLM Judge reward function using the LLM Judge system.
    This reward function uses external LLMs to evaluate trajectory quality
    based on customizable rubrics, providing sophisticated reward signals
    for GRPO training.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4",
        rubric_type: str = "default",
        custom_rubric: Optional[str] = None,
        temperature: float = 0.0,
        fallback_enabled: bool = True,
        cache_ttl: int = 3600,
        batch_size: int = 10,
        max_retries: int = 3,
        timeout: float = 30.0,
        weight: float = 1.0,
        cache_service: Optional[CacheService] = None,
        monitoring_service: Optional[MonitoringService] = None,
    ):
        """
        Initialize LLM Judge reward function.

        Args:
            model: LLM model to use as judge (e.g., "openai/gpt-4")
            rubric_type: Pre-defined rubric type to use
            custom_rubric: Custom rubric text (overrides rubric_type)
            temperature: Temperature for judge model
            fallback_enabled: Whether to use heuristic fallback on failure
            cache_ttl: Cache TTL in seconds
            batch_size: Number of trajectories to judge in one call
            max_retries: Maximum retries for failed API calls
            timeout: Timeout for API calls
            weight: Weight for this reward function
            cache_service: Optional cache service
            monitoring_service: Optional monitoring service
        """
        super().__init__(weight=weight)

        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is required for LLM Judge reward function. Install with: pip install litellm"
            )

        self.model = model
        self.rubric_type = rubric_type
        self.custom_rubric = custom_rubric
        self.temperature = temperature
        self.fallback_enabled = fallback_enabled
        self.cache_ttl = cache_ttl
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

        # Services
        self.cache = cache_service
        self.monitoring = monitoring_service

        # Get rubric
        self.rubric = custom_rubric or RUBRICS.get(rubric_type, RUBRICS["default"])

        # Metrics
        self.total_calls = 0
        self.cache_hits = 0
        self.fallback_uses = 0

    async def compute_reward(
        self, turns: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """
        Compute reward using LLM Judge.

        Args:
            turns: List of conversation turns
            context: Optional context information

        Returns:
            RewardResult with score and breakdown
        """
        # Convert turns to messages
        messages = self._turns_to_messages(turns)

        # For single trajectory, create a comparison batch
        if len(messages) == 1:
            # Duplicate for comparison (RULER works better with multiple trajectories)
            messages = messages * 2

        try:
            # Compute scores
            scores = await self._compute_scores(messages, context)

            # Return first score (primary trajectory)
            score = scores[0] if scores else 0.0

            return RewardResult(
                score=score,
                breakdown={
                    "ruler_score": score,
                    "model": self.model,
                    "rubric_type": self.rubric_type,
                    "cache_hit": False,  # Will be updated by cache check
                },
            )

        except Exception as e:
            logger.error(f"RULER reward computation failed: {e}")

            if self.fallback_enabled:
                fallback_score = self._heuristic_fallback(turns)
                self.fallback_uses += 1

                return RewardResult(
                    score=fallback_score,
                    breakdown={
                        "ruler_score": fallback_score,
                        "fallback_used": True,
                        "error": str(e),
                    },
                )
            else:
                raise

    async def compute_batch_rewards(
        self,
        batch_turns: List[List[Dict[str, Any]]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[RewardResult]:
        """
        Compute rewards for a batch of trajectory turns.

        Args:
            batch_turns: List of trajectory turns
            context: Optional context information

        Returns:
            List of RewardResult objects
        """
        if not batch_turns:
            return []

        # Convert all turns to messages
        all_messages = [self._turns_to_messages(turns) for turns in batch_turns]

        # Flatten for batch processing
        flat_messages = [msg for msgs in all_messages for msg in msgs]

        # Process in batches
        all_scores = []
        for i in range(0, len(flat_messages), self.batch_size):
            batch = flat_messages[i : i + self.batch_size]
            scores = await self._compute_scores(batch, context)
            all_scores.extend(scores)

        # Create results
        results = []
        for i, score in enumerate(all_scores):
            results.append(
                RewardResult(
                    score=score,
                    breakdown={
                        "ruler_score": score,
                        "model": self.model,
                        "rubric_type": self.rubric_type,
                        "batch_index": i,
                    },
                )
            )

        return results

    async def _compute_scores(
        self,
        messages_list: List[List[ChatCompletionMessageParam]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Compute scores for a list of message sequences"""
        if not messages_list:
            return []

        # Check cache
        cache_key = self._get_cache_key(messages_list)
        if self.cache:
            cached_scores = await self.cache.get(cache_key)
            if cached_scores:
                self.cache_hits += 1
                return cached_scores

        # Compute scores with retries
        for attempt in range(self.max_retries):
            try:
                scores = await self._ruler_judge(messages_list)

                # Cache results
                if self.cache:
                    await self.cache.set(cache_key, scores, ttl=self.cache_ttl)

                # Log metrics
                if self.monitoring:
                    await self.monitoring.log_metric(
                        "ruler_reward.batch_processed",
                        len(messages_list),
                        tags={
                            "model": self.model,
                            "rubric_type": self.rubric_type,
                            "attempt": attempt + 1,
                        },
                    )

                self.total_calls += 1
                return scores

            except Exception as e:
                logger.warning(f"RULER attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise

                # Wait before retry
                await asyncio.sleep(0.5 * (2**attempt))

        return []

    async def _ruler_judge(
        self, message_lists: List[List[ChatCompletionMessageParam]]
    ) -> List[float]:
        """Core LLM Judge implementation"""
        if not message_lists:
            return []

        # Find common prefix
        common_prefix_len = 0
        if len(message_lists) > 1:
            for idx, msg in enumerate(message_lists[0]):
                if all(
                    len(msg_list) > idx and msg_list[idx] == msg
                    for msg_list in message_lists
                ):
                    common_prefix_len += 1
                else:
                    break

        # Build judge prompt
        user_text = ""
        if common_prefix_len > 0:
            common_prefix_messages = message_lists[0][:common_prefix_len]
            user_text += (
                "<context>\n"
                + json.dumps(common_prefix_messages, indent=2)
                + "\n</context>\n\n"
            )

        # Serialize trajectories
        serialized_trajectories = []
        for idx, full_messages in enumerate(message_lists, start=1):
            trimmed_messages = full_messages[common_prefix_len:]
            serialized_trajectories.append(
                f'<trajectory id="{idx}">\n'
                + json.dumps(trimmed_messages, indent=2)
                + "\n</trajectory>"
            )

        user_text += "Trajectories:\n\n" + "\n\n".join(serialized_trajectories)

        judge_prompt = dedent(
            f"""
            All of the trajectories below have been given the same goal. 
            Your job is to consider each of them and give them a score between 0 and 1. 
            Take into consideration your best judgement of the agent's goal.

            Grading standards:
            {self.rubric}

            Please provide scores for each trajectory with explanations.
        """
        )

        messages = [
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": user_text},
        ]

        # Make API call with timeout
        response = await asyncio.wait_for(
            asyncio.create_task(self._make_completion_call(messages)),
            timeout=self.timeout,
        )

        # Parse response
        if len(response.choices) == 0:
            raise ValueError(f"No choices in response: {response}")

        first_choice = response.choices[0]
        content = first_choice.message.content or "{}"

        try:
            parsed = RulerResponse.model_validate_json(content)
        except Exception as e:
            logger.error(f"Failed to parse RULER response: {e}")
            logger.error(f"Raw content: {content}")
            raise

        if len(parsed.scores) != len(message_lists):
            logger.warning(
                f"Expected {len(message_lists)} scores, got {len(parsed.scores)}. "
                f"Padding/truncating to match."
            )

            # Pad or truncate to match expected length
            scores = [s.score for s in parsed.scores]
            while len(scores) < len(message_lists):
                scores.append(0.5)  # Default neutral score
            scores = scores[: len(message_lists)]

            return scores

        return [s.score for s in parsed.scores]

    async def _make_completion_call(self, messages: List[Dict[str, Any]]) -> Any:
        """Make async completion call (wrapper for sync litellm)"""
        # Run in thread pool since litellm is synchronous
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format=RulerResponse,
                caching=False,
            ),
        )

    def _turns_to_messages(
        self, turns: List[Dict[str, Any]]
    ) -> List[ChatCompletionMessageParam]:
        """Convert conversation turns to message format"""
        messages = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")

            if role and content:
                messages.append({"role": role, "content": content})

        return messages

    def _heuristic_fallback(self, turns: List[Dict[str, Any]]) -> float:
        """Simple heuristic scoring as fallback"""
        if not turns:
            return 0.0

        # Get the last assistant message
        assistant_messages = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_messages:
            return 0.0

        last_response = assistant_messages[-1].get("content", "")

        score = 0.5  # Base score

        # Length scoring
        word_count = len(last_response.split())
        if 20 <= word_count <= 150:
            score += 0.2
        elif word_count < 10:
            score -= 0.2

        # Quality indicators
        helpful_phrases = [
            "i can help",
            "let me",
            "would you like",
            "please",
            "here's",
            "you can",
            "try",
            "suggest",
            "recommend",
        ]

        if any(phrase in last_response.lower() for phrase in helpful_phrases):
            score += 0.2

        # Politeness indicators
        polite_phrases = ["thank you", "please", "sorry", "excuse me"]
        if any(phrase in last_response.lower() for phrase in polite_phrases):
            score += 0.1

        # Ensure score is in [0, 1]
        return max(0.0, min(1.0, score))

    def _get_cache_key(
        self, messages_list: List[List[ChatCompletionMessageParam]]
    ) -> str:
        """Generate cache key for LLM Judge results"""
        import hashlib

        content = json.dumps(
            {
                "messages": messages_list,
                "model": self.model,
                "rubric": self.rubric,
                "temperature": self.temperature,
            },
            sort_keys=True,
            default=str,
        )
        return f"ruler_reward:{hashlib.sha256(content.encode()).hexdigest()}"

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "fallback_uses": self.fallback_uses,
            "cache_hit_rate": self.cache_hits / max(1, self.total_calls),
            "fallback_rate": self.fallback_uses / max(1, self.total_calls),
        }

    @staticmethod
    def get_available_rubrics() -> Dict[str, str]:
        """Get all available pre-defined rubrics"""
        return RUBRICS.copy()

    @staticmethod
    def create_custom_rubric(name: str, criteria: List[Dict[str, str]]) -> str:
        """
        Create a custom rubric from criteria.

        Args:
            name: Name of the rubric
            criteria: List of dicts with 'aspect' and 'description' keys

        Returns:
            Formatted rubric text
        """
        rubric_lines = [f"Score {name} based on:"]
        for criterion in criteria:
            aspect = criterion.get("aspect", "")
            description = criterion.get("description", "")
            rubric_lines.append(f"- {aspect}: {description}")

        rubric_lines.append(
            "Give higher scores to responses that excel in multiple categories."
        )
        return "\n".join(rubric_lines)


# Convenience functions for common use cases
def create_customer_service_ruler(
    model: str = "openai/gpt-4", weight: float = 1.0, **kwargs
) -> RulerRewardFunction:
    """Create a LLM Judge reward function for customer service"""
    return RulerRewardFunction(
        model=model, rubric_type="customer_service", weight=weight, **kwargs
    )


def create_technical_support_ruler(
    model: str = "openai/gpt-4", weight: float = 1.0, **kwargs
) -> RulerRewardFunction:
    """Create a LLM Judge reward function for technical support"""
    return RulerRewardFunction(
        model=model, rubric_type="technical_support", weight=weight, **kwargs
    )


def create_custom_ruler(
    rubric: str, model: str = "openai/gpt-4", weight: float = 1.0, **kwargs
) -> RulerRewardFunction:
    """Create a LLM Judge reward function with custom rubric"""
    return RulerRewardFunction(
        model=model, custom_rubric=rubric, weight=weight, **kwargs
    )
