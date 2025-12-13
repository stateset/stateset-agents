"""
Multi-Objective Reward System for GRPO Agent Framework

This module provides sophisticated reward functions that combine multiple
evaluation criteria with weighted scoring, heuristic fallbacks, and 
domain-specific optimizations.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from stateset_agents.core import reward as core_reward

RewardFunction = core_reward.RewardFunction
RewardResult = core_reward.RewardResult
import utils.cache

CacheService = utils.cache.CacheService
import utils.monitoring

MonitoringService = utils.monitoring.MonitoringService

logger = logging.getLogger(__name__)


@dataclass
class RewardComponent:
    """Individual reward component with weight and configuration"""

    name: str
    weight: float
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")


class BaseRewardComponent(ABC):
    """Base class for individual reward components"""

    def __init__(self, name: str, weight: float = 1.0, **kwargs):
        self.name = name
        self.weight = weight
        self.config = kwargs

    @abstractmethod
    async def compute_score(
        self, turns: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute score for this component (0-1 range)"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get component information"""
        return {"name": self.name, "weight": self.weight, "config": self.config}


class LengthRewardComponent(BaseRewardComponent):
    """Reward component based on response length"""

    def __init__(
        self,
        name: str = "length",
        weight: float = 0.1,
        min_length: int = 10,
        max_length: int = 200,
        optimal_range: Tuple[int, int] = (50, 150),
        **kwargs,
    ):
        super().__init__(name, weight, **kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.optimal_range = optimal_range

    async def compute_score(
        self, turns: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score based on response length"""
        if not turns:
            return 0.0

        # Get last assistant response
        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_responses:
            return 0.0

        last_response = assistant_responses[-1].get("content", "")
        word_count = len(last_response.split())

        # Penalty for too short or too long
        if word_count < self.min_length:
            return 0.2
        if word_count > self.max_length:
            return 0.3

        # Reward optimal range
        if self.optimal_range[0] <= word_count <= self.optimal_range[1]:
            return 1.0

        # Gradual penalty outside optimal range
        if word_count < self.optimal_range[0]:
            return 0.5 + 0.5 * (word_count - self.min_length) / (
                self.optimal_range[0] - self.min_length
            )
        else:
            return 0.5 + 0.5 * (self.max_length - word_count) / (
                self.max_length - self.optimal_range[1]
            )


class EmpathyRewardComponent(BaseRewardComponent):
    """Reward component based on empathy indicators"""

    def __init__(
        self,
        name: str = "empathy",
        weight: float = 0.2,
        empathy_keywords: Optional[List[str]] = None,
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

    async def compute_score(
        self, turns: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score based on empathy indicators"""
        if not turns:
            return 0.0

        # Get assistant responses
        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_responses:
            return 0.0

        # Check for empathy keywords
        total_matches = 0
        total_words = 0

        for response in assistant_responses:
            content = response.get("content", "").lower()
            words = content.split()
            total_words += len(words)

            for keyword in self.empathy_keywords:
                if keyword in content:
                    total_matches += 1

        # Normalize by response length
        if total_words == 0:
            return 0.0

        empathy_density = total_matches / max(1, len(assistant_responses))
        return min(1.0, empathy_density / 2.0)  # Scale to 0-1


class ActionOrientedRewardComponent(BaseRewardComponent):
    """Reward component based on action-oriented language"""

    def __init__(
        self,
        name: str = "action_oriented",
        weight: float = 0.2,
        action_keywords: Optional[List[str]] = None,
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

    async def compute_score(
        self, turns: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score based on action-oriented language"""
        if not turns:
            return 0.0

        # Get assistant responses
        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_responses:
            return 0.0

        # Check for action keywords
        total_matches = 0

        for response in assistant_responses:
            content = response.get("content", "").lower()

            for keyword in self.action_keywords:
                if keyword in content:
                    total_matches += 1

        # Normalize by number of responses
        action_density = total_matches / max(1, len(assistant_responses))
        return min(1.0, action_density / 3.0)  # Scale to 0-1


class SimilarityRewardComponent(BaseRewardComponent):
    """Reward component based on similarity to expected responses"""

    def __init__(
        self,
        name: str = "similarity",
        weight: float = 0.3,
        expected_responses: Optional[List[str]] = None,
        similarity_threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(name, weight, **kwargs)
        self.expected_responses = expected_responses or []
        self.similarity_threshold = similarity_threshold

    async def compute_score(
        self, turns: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score based on similarity to expected responses"""
        if not turns or not self.expected_responses:
            return 0.5  # Neutral score if no expected responses

        # Get assistant responses
        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_responses:
            return 0.0

        # Calculate similarity using simple keyword overlap
        max_similarity = 0.0

        for response in assistant_responses:
            content = response.get("content", "").lower()
            response_words = set(content.split())

            for expected in self.expected_responses:
                expected_words = set(expected.lower().split())

                if not expected_words:
                    continue

                # Jaccard similarity
                intersection = response_words & expected_words
                union = response_words | expected_words

                if union:
                    similarity = len(intersection) / len(union)
                    max_similarity = max(max_similarity, similarity)

        # Apply threshold and scale
        if max_similarity >= self.similarity_threshold:
            return min(1.0, max_similarity / self.similarity_threshold)
        else:
            return max_similarity / self.similarity_threshold * 0.5


class ProfessionalismRewardComponent(BaseRewardComponent):
    """Reward component based on professionalism indicators"""

    def __init__(
        self,
        name: str = "professionalism",
        weight: float = 0.2,
        professional_indicators: Optional[List[str]] = None,
        unprofessional_indicators: Optional[List[str]] = None,
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

    async def compute_score(
        self, turns: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score based on professionalism"""
        if not turns:
            return 0.0

        # Get assistant responses
        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_responses:
            return 0.0

        professional_count = 0
        unprofessional_count = 0

        for response in assistant_responses:
            content = response.get("content", "").lower()

            for indicator in self.professional_indicators:
                if indicator in content:
                    professional_count += 1

            for indicator in self.unprofessional_indicators:
                if indicator in content:
                    unprofessional_count += 1

        # Calculate professionalism score
        total_responses = len(assistant_responses)
        professional_density = professional_count / max(1, total_responses)
        unprofessional_penalty = unprofessional_count / max(1, total_responses)

        score = min(1.0, professional_density / 2.0) - unprofessional_penalty
        return max(0.0, score)


class ModelBasedRewardComponent(BaseRewardComponent):
    """
    Reward component that uses an LLM (Judge) to evaluate response quality.
    "LLM-as-a-Judge" implementation.
    """

    def __init__(
        self,
        name: str = "model_judge",
        weight: float = 0.5,
        judge_prompt_template: Optional[str] = None,
        api_client: Optional[Any] = None, # Placeholder for OpenAI/Anthropic/Local client
        **kwargs,
    ):
        super().__init__(name, weight, **kwargs)
        self.judge_prompt_template = judge_prompt_template or (
            "You are an impartial judge evaluating the quality of an AI assistant's response.\n"
            "User Query: {user_query}\n"
            "Assistant Response: {assistant_response}\n\n"
            "Rate the response on a scale of 0 to 1 (float), focusing on helpfulness, accuracy, and tone.\n"
            "Output ONLY the score."
        )
        self.api_client = api_client

    async def compute_score(
        self, turns: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score using Model-as-a-Judge"""
        if not turns:
            return 0.0

        # Get relevant content
        assistant_responses = [t for t in turns if t.get("role") == "assistant"]
        user_turns = [t for t in turns if t.get("role") == "user"]
        
        if not assistant_responses or not user_turns:
            return 0.0

        assistant_response = assistant_responses[-1].get("content", "")
        user_query = user_turns[-1].get("content", "")

        # Format prompt
        prompt = self.judge_prompt_template.format(
            user_query=user_query,
            assistant_response=assistant_response
        )

        try:
            # API client integration for model-based judging
            if self.api_client:
                # Check client type and call appropriate API
                client_type = type(self.api_client).__name__.lower()

                if 'openai' in client_type:
                    # OpenAI API client
                    response = await self._call_openai_api(prompt)
                    score = self._parse_score_from_response(response)
                elif 'anthropic' in client_type:
                    # Anthropic API client
                    response = await self._call_anthropic_api(prompt)
                    score = self._parse_score_from_response(response)
                elif hasattr(self.api_client, 'generate'):
                    # Generic client with generate method (e.g., vLLM, local model)
                    response = await self.api_client.generate(prompt)
                    score = self._parse_score_from_response(response)
                else:
                    logger.warning(f"Unknown API client type: {client_type}, using heuristic")
                    score = min(1.0, len(assistant_response) / 200.0)
            else:
                 # Heuristic fallback (length + keyword mix) for demo/testing without API key
                 score = min(1.0, len(assistant_response) / 200.0)

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.error(f"Model judge failed: {e}")
            return 0.5 # Neutral fallback

    async def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API for scoring"""
        try:
            # Async OpenAI client call
            if hasattr(self.api_client, 'chat'):
                response = await self.api_client.chat.completions.create(
                    model=getattr(self, 'model_name', 'gpt-4'),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0
                )
                return response.choices[0].message.content
            else:
                # Fallback for older API structure
                response = await self.api_client.completions.create(
                    model=getattr(self, 'model_name', 'gpt-3.5-turbo-instruct'),
                    prompt=prompt,
                    max_tokens=10,
                    temperature=0.0
                )
                return response.choices[0].text
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    async def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API for scoring"""
        try:
            response = await self.api_client.messages.create(
                model=getattr(self, 'model_name', 'claude-3-sonnet-20240229'),
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise

    def _parse_score_from_response(self, response: str) -> float:
        """Parse numerical score from API response"""
        try:
            # Try to extract first number from response
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                # Normalize to 0-1 range if needed
                if score > 1.0:
                    score = score / 10.0  # Assume 1-10 scale
                return min(1.0, max(0.0, score))
            return 0.5  # Default if no number found
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse score from response: {response}, error: {e}")
            return 0.5


class ReasoningRewardComponent(BaseRewardComponent):
    """
    Reward component based on the presence and quality of reasoning traces.
    (e.g., <think>...</think> blocks captured in metadata)
    """

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
        self, turns: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Score based on reasoning trace"""
        if not turns:
            return 0.0

        # Get last assistant turn
        assistant_turns = [t for t in turns if t.get("role") == "assistant"]
        if not assistant_turns:
            return 0.0
            
        last_turn = assistant_turns[-1]
        metadata = last_turn.get("metadata", {})
        reasoning = metadata.get("reasoning", "")
        
        if not reasoning:
            return 0.0
            
        # Length-based heuristic
        length = len(reasoning.split())
        
        if length < self.min_length:
            return 0.2 * (length / self.min_length) # Penalty for too short
            
        # Reward curve: linear up to optimal, then plateau or slight decay
        if length <= self.optimal_length:
             # Scale 0.2 -> 1.0
             return 0.2 + 0.8 * ((length - self.min_length) / (self.optimal_length - self.min_length))
        else:
            return 1.0 # Plateau at max score for long reasoning (DeepSeek style)


class MultiObjectiveRewardFunction(RewardFunction):
    """
    Multi-objective reward function that combines multiple reward components
    with weighted scoring and sophisticated evaluation criteria.
    """

    def __init__(
        self,
        components: Optional[List[BaseRewardComponent]] = None,
        weight: float = 1.0,
        normalization_method: str = "weighted_sum",
        cache_service: Optional[CacheService] = None,
        monitoring_service: Optional[MonitoringService] = None,
    ):
        """
        Initialize multi-objective reward function.

        Args:
            components: List of reward components
            weight: Overall weight for this reward function
            normalization_method: How to combine component scores
            cache_service: Optional cache service
            monitoring_service: Optional monitoring service
        """
        super().__init__(weight=weight)

        self.components = components or []
        self.normalization_method = normalization_method
        self.cache = cache_service
        self.monitoring = monitoring_service

        # Validate weights sum to reasonable range
        total_weight = sum(c.weight for c in self.components)
        if total_weight > 0:
            # Normalize weights to sum to 1
            for component in self.components:
                component.weight = component.weight / total_weight

        # Metrics
        self.evaluation_count = 0
        self.component_scores = {}

    def add_component(self, component: BaseRewardComponent):
        """Add a reward component"""
        self.components.append(component)

        # Re-normalize weights
        total_weight = sum(c.weight for c in self.components)
        if total_weight > 0:
            for comp in self.components:
                comp.weight = comp.weight / total_weight

    def remove_component(self, name: str):
        """Remove a reward component by name"""
        self.components = [c for c in self.components if c.name != name]

        # Re-normalize weights
        total_weight = sum(c.weight for c in self.components)
        if total_weight > 0:
            for comp in self.components:
                comp.weight = comp.weight / total_weight

    async def compute_reward(  # type: ignore[override]
        self,
        turns: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        trajectory: Optional[Any] = None,
        turn: Optional[Any] = None,
        **_: Any,
    ) -> RewardResult:
        """
        Compute multi-objective reward.

        Args:
            turns: List of conversation turns (dicts or ConversationTurn objects).
            context: Optional context information

        Returns:
            RewardResult with combined score and component breakdown
        """
        # Support trainer-style calls: compute_reward(trajectory=?, turn=?, context=?)
        if turns is None and turn is not None:
            normalized: List[Dict[str, Any]] = []
            if context and isinstance(context, dict):
                user_query = context.get("user_query") or context.get("prompt")
                if user_query:
                    normalized.append({"role": "user", "content": str(user_query)})
            normalized.append(
                {
                    "role": getattr(turn, "role", "assistant"),
                    "content": getattr(turn, "content", str(turn)),
                }
            )
            turns = normalized

        # Normalize ConversationTurn objects into dicts for component evaluation.
        turns = turns or []
        normalized_turns: List[Dict[str, Any]] = []
        for item in turns:
            if isinstance(item, dict):
                normalized_turns.append(item)
                continue

            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
            if role is not None and content is not None:
                normalized_turns.append({"role": str(role), "content": str(content)})
            else:
                normalized_turns.append({"role": "assistant", "content": str(item)})

        if not self.components:
            return RewardResult(
                score=0.0, breakdown={"error": "No components configured"}
            )

        # Compute scores for each component
        component_scores = {}
        weighted_scores = []

        for component in self.components:
            try:
                score = await component.compute_score(normalized_turns, context)
                component_scores[component.name] = score
                weighted_scores.append(score * component.weight)

                # Track component performance
                if component.name not in self.component_scores:
                    self.component_scores[component.name] = []
                self.component_scores[component.name].append(score)

            except Exception as e:
                logger.error(f"Component {component.name} failed: {e}")
                component_scores[component.name] = 0.0
                weighted_scores.append(0.0)

        # Combine scores
        if self.normalization_method == "weighted_sum":
            final_score = sum(weighted_scores)
        elif self.normalization_method == "weighted_average":
            final_score = sum(weighted_scores) / len(weighted_scores)
        elif self.normalization_method == "geometric_mean":
            # Geometric mean of weighted scores
            if all(s > 0 for s in weighted_scores):
                final_score = np.prod(weighted_scores) ** (1.0 / len(weighted_scores))
            else:
                final_score = 0.0
        else:
            final_score = sum(weighted_scores)

        # Ensure score is in [0, 1]
        final_score = max(0.0, min(1.0, final_score))

        self.evaluation_count += 1

        # Create breakdown
        breakdown = {
            "total_score": final_score,
            "components": component_scores,
            "weights": {c.name: c.weight for c in self.components},
            "normalization_method": self.normalization_method,
            "evaluation_count": self.evaluation_count,
        }

        # Log metrics
        if self.monitoring:
            await self.monitoring.log_metric(
                "multi_objective_reward.evaluation",
                final_score,
                tags={"method": self.normalization_method},
            )

        result = RewardResult(score=final_score, breakdown=breakdown, metadata={})
        # Some trainers expect `total_reward`; keep it as an alias for score.
        setattr(result, "total_reward", float(final_score))
        return result

    def get_component_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each component"""
        stats = {}

        for name, scores in self.component_scores.items():
            if scores:
                stats[name] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "count": len(scores),
                }

        return stats

    def get_info(self) -> Dict[str, Any]:
        """Get reward function information"""
        return {
            "type": "multi_objective",
            "components": [c.get_info() for c in self.components],
            "normalization_method": self.normalization_method,
            "evaluation_count": self.evaluation_count,
            "component_statistics": self.get_component_statistics(),
        }


# Convenience functions for creating common multi-objective rewards
def create_customer_service_reward(
    expected_responses: Optional[List[str]] = None, weight: float = 1.0, **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for customer service"""
    components = [
        EmpathyRewardComponent(weight=0.25),
        ActionOrientedRewardComponent(weight=0.25),
        ProfessionalismRewardComponent(weight=0.25),
        LengthRewardComponent(weight=0.1, min_length=20, optimal_range=(50, 200)),
        SimilarityRewardComponent(weight=0.15, expected_responses=expected_responses),
    ]

    return MultiObjectiveRewardFunction(components=components, weight=weight, **kwargs)


def create_domain_reward(
    domain: str,
    expected_responses: Optional[List[str]] = None,
    weight: float = 1.0,
    **kwargs,
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward function for a given domain.

    This is a small convenience wrapper used by many examples. It maps common
    domain names (e.g., ``"technical_support"``) to the corresponding
    multi-objective reward template.
    """

    domain_lower = domain.strip().lower()

    if domain_lower in {"customer_service", "customer", "cs", "support"}:
        return create_customer_service_reward(
            expected_responses=expected_responses, weight=weight, **kwargs
        )

    if domain_lower in {
        "technical_support",
        "tech_support",
        "technical",
        "it",
        "it_support",
        "helpdesk",
    }:
        return create_technical_support_reward(
            expected_responses=expected_responses, weight=weight, **kwargs
        )

    if domain_lower in {"sales", "marketing", "product", "sales_assistant"}:
        return create_sales_reward(
            expected_responses=expected_responses, weight=weight, **kwargs
        )

    if domain_lower in {"education", "educational", "tutoring", "tutor"}:
        return create_educational_reward(
            expected_responses=expected_responses, weight=weight, **kwargs
        )

    if domain_lower in {"creative", "creative_writing", "writing"}:
        return create_creative_reward(
            expected_responses=expected_responses, weight=weight, **kwargs
        )

    # Default to customer service heuristics for conversational alignment.
    return create_customer_service_reward(
        expected_responses=expected_responses, weight=weight, **kwargs
    )


def create_technical_support_reward(
    expected_responses: Optional[List[str]] = None, weight: float = 1.0, **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for technical support"""
    components = [
        ActionOrientedRewardComponent(weight=0.4),  # High weight for solutions
        ProfessionalismRewardComponent(weight=0.2),
        LengthRewardComponent(weight=0.1, min_length=30, optimal_range=(100, 300)),
        SimilarityRewardComponent(weight=0.3, expected_responses=expected_responses),
    ]

    return MultiObjectiveRewardFunction(components=components, weight=weight, **kwargs)


def create_educational_reward(
    expected_responses: Optional[List[str]] = None, weight: float = 1.0, **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for educational content"""
    components = [
        ActionOrientedRewardComponent(weight=0.3),
        ProfessionalismRewardComponent(weight=0.2),
        LengthRewardComponent(weight=0.1, min_length=50, optimal_range=(150, 400)),
        SimilarityRewardComponent(weight=0.4, expected_responses=expected_responses),
    ]

    return MultiObjectiveRewardFunction(components=components, weight=weight, **kwargs)


def create_sales_reward(
    expected_responses: Optional[List[str]] = None, weight: float = 1.0, **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for sales interactions"""
    # Add sales-specific components
    sales_action_keywords = [
        "buy",
        "purchase",
        "order",
        "discount",
        "offer",
        "deal",
        "save",
        "recommend",
        "suggest",
        "perfect for",
        "ideal",
        "best choice",
        "limited time",
        "special",
        "exclusive",
        "value",
        "benefit",
    ]

    components = [
        ActionOrientedRewardComponent(
            weight=0.35, action_keywords=sales_action_keywords
        ),
        ProfessionalismRewardComponent(weight=0.25),
        EmpathyRewardComponent(weight=0.15),
        LengthRewardComponent(weight=0.1, min_length=30, optimal_range=(80, 250)),
        SimilarityRewardComponent(weight=0.15, expected_responses=expected_responses),
    ]

    return MultiObjectiveRewardFunction(components=components, weight=weight, **kwargs)


def create_creative_reward(
    expected_responses: Optional[List[str]] = None, weight: float = 1.0, **kwargs
) -> MultiObjectiveRewardFunction:
    """Create a multi-objective reward for creative content"""
    creative_keywords = [
        "imagine",
        "creative",
        "unique",
        "original",
        "innovative",
        "artistic",
        "beautiful",
        "inspiring",
        "fascinating",
        "amazing",
        "wonderful",
        "story",
        "narrative",
        "character",
        "scene",
        "vivid",
        "descriptive",
    ]

    components = [
        ActionOrientedRewardComponent(weight=0.2, action_keywords=creative_keywords),
        LengthRewardComponent(weight=0.2, min_length=50, optimal_range=(200, 500)),
        SimilarityRewardComponent(weight=0.6, expected_responses=expected_responses),
    ]

    return MultiObjectiveRewardFunction(components=components, weight=weight, **kwargs)
