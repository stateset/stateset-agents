"""
Custom Reward Functions for StateSet Agents

This example demonstrates how to create and use custom reward functions for training
conversational agents. It covers:

1. Creating simple custom rewards from scratch
2. Composing multiple rewards with weights
3. Using neural reward models
4. Creating domain-specific rewards
5. Implementing reward shaping and curriculum learning
6. Advanced rewards with external APIs (sentiment analysis, toxicity detection)

Requirements:
    - pip install stateset-agents[dev]
    - Optional: pip install transformers  # For neural reward models
    - Optional: pip install textblob      # For sentiment analysis

Usage:
    # Run with default example
    python examples/custom_reward_functions.py

    # Train with a specific custom reward
    python examples/custom_reward_functions.py --reward-type empathy

    # Show all available custom rewards
    python examples/custom_reward_functions.py --list-rewards
"""

import argparse
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

# Framework imports
from stateset_agents import MultiTurnAgent
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import (
    CompositeReward,
    RewardFunction,
    RewardResult,
    RewardType,
)
from stateset_agents.core.trajectory import ConversationTurn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. Simple Custom Rewards
# ============================================================================

class ResponseLengthReward(RewardFunction):
    """
    Simple reward based on response length

    Rewards responses within an optimal length range.
    Too short = unhelpful, too long = verbose
    """

    def __init__(
        self,
        weight: float = 1.0,
        min_length: int = 50,
        max_length: int = 300,
        optimal_min: int = 100,
        optimal_max: int = 200,
    ):
        super().__init__(weight, RewardType.IMMEDIATE, "ResponseLengthReward")
        self.min_length = min_length
        self.max_length = max_length
        self.optimal_min = optimal_min
        self.optimal_max = optimal_max

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute reward based on response length"""

        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            length = len(turn.content)

            # Score based on length
            if self.optimal_min <= length <= self.optimal_max:
                score = 1.0
            elif self.min_length <= length < self.optimal_min:
                # Linear increase from min to optimal
                score = 0.5 + 0.5 * (length - self.min_length) / (self.optimal_min - self.min_length)
            elif self.optimal_max < length <= self.max_length:
                # Linear decrease from optimal to max
                score = 0.5 + 0.5 * (self.max_length - length) / (self.max_length - self.optimal_max)
            elif length < self.min_length:
                # Too short
                score = 0.2 * (length / self.min_length)
            else:
                # Too long
                score = 0.2

            total_score += score
            breakdown[f"turn_{i}_length"] = length
            breakdown[f"turn_{i}_score"] = score

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={
                "num_turns": len(assistant_turns),
                "avg_length": sum(len(t.content) for t in assistant_turns) / len(assistant_turns)
            },
        )


class QuestionAnsweringReward(RewardFunction):
    """
    Rewards responses that answer questions directly

    Penalizes evasive or off-topic responses.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight, RewardType.IMMEDIATE, "QuestionAnsweringReward")
        self.question_patterns = [
            r'\?$',  # Ends with question mark
            r'^(what|when|where|who|why|how|can|could|would|should|is|are|do|does)',
        ]

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute reward for question answering quality"""

        if len(turns) < 2:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}
        question_count = 0

        # Look at user-assistant pairs
        for i in range(len(turns) - 1):
            if turns[i].role == "user" and turns[i + 1].role == "assistant":
                user_content = turns[i].content
                assistant_content = turns[i + 1].content

                # Check if user asked a question
                is_question = any(
                    re.search(pattern, user_content, re.IGNORECASE)
                    for pattern in self.question_patterns
                )

                if is_question:
                    question_count += 1
                    score = self._evaluate_answer_quality(user_content, assistant_content)
                    total_score += score
                    breakdown[f"pair_{i}_answer_quality"] = score

        if question_count == 0:
            return RewardResult(score=0.5, breakdown={}, metadata={"questions_found": 0})

        avg_score = total_score / question_count

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"questions_answered": question_count},
        )

    def _evaluate_answer_quality(self, question: str, answer: str) -> float:
        """Evaluate how well an answer addresses a question"""
        score = 0.5  # Base score

        # Bonus for substantive answers
        if len(answer) > 50:
            score += 0.2

        # Bonus for specific indicators of good answers
        good_indicators = ["because", "specifically", "for example", "in particular"]
        if any(indicator in answer.lower() for indicator in good_indicators):
            score += 0.15

        # Penalty for evasive answers
        evasive_phrases = [
            "i don't know", "not sure", "can't say", "no idea",
            "unable to", "don't have", "cannot provide"
        ]
        if any(phrase in answer.lower() for phrase in evasive_phrases):
            score -= 0.3

        return max(0.0, min(1.0, score))


# ============================================================================
# 2. Advanced Domain-Specific Rewards
# ============================================================================

class EmpathyReward(RewardFunction):
    """
    Rewards empathetic responses in customer service contexts

    Looks for empathy indicators, acknowledgment of customer feelings,
    and supportive language.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight, RewardType.IMMEDIATE, "EmpathyReward")

        self.empathy_phrases = [
            "i understand", "i can see", "i appreciate", "i'm sorry",
            "that must be", "i realize", "i recognize", "thank you for",
            "i apologize", "frustrating", "concerning", "important to you"
        ]

        self.negative_phrases = [
            "you should have", "you need to", "you must", "that's wrong",
            "you're mistaken", "actually", "just", "simply"
        ]

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute empathy score for assistant responses"""

        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            content_lower = turn.content.lower()

            # Count empathy indicators
            empathy_count = sum(
                1 for phrase in self.empathy_phrases if phrase in content_lower
            )

            # Count negative phrases (reduce empathy score)
            negative_count = sum(
                1 for phrase in self.negative_phrases if phrase in content_lower
            )

            # Compute score
            base_score = 0.3  # Base empathy score
            empathy_bonus = min(0.5, empathy_count * 0.15)
            negative_penalty = min(0.4, negative_count * 0.2)

            score = max(0.0, base_score + empathy_bonus - negative_penalty)

            total_score += score
            breakdown[f"turn_{i}_empathy"] = score
            breakdown[f"turn_{i}_empathy_count"] = empathy_count
            breakdown[f"turn_{i}_negative_count"] = negative_count

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_turns": len(assistant_turns)},
        )


class ActionOrientedReward(RewardFunction):
    """
    Rewards responses that provide clear next steps or actions

    Useful for task-oriented agents like customer service or technical support.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight, RewardType.IMMEDIATE, "ActionOrientedReward")

        self.action_indicators = [
            r'I (will|can|\'ll)',
            r'let me',
            r'I\'m going to',
            r'step \d+',
            r'first|next|then|finally',
            r'you (can|could|should|need to)',
            r'please (try|follow|check)',
        ]

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute reward for action-oriented responses"""

        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            content = turn.content

            # Count action indicators
            action_count = sum(
                1 for pattern in self.action_indicators
                if re.search(pattern, content, re.IGNORECASE)
            )

            # Check for numbered lists or bullet points (structured actions)
            has_list = bool(re.search(r'(\d+\.|•|-)\s+', content))

            # Compute score
            base_score = 0.2
            action_bonus = min(0.6, action_count * 0.2)
            structure_bonus = 0.2 if has_list else 0.0

            score = min(1.0, base_score + action_bonus + structure_bonus)

            total_score += score
            breakdown[f"turn_{i}_action_oriented"] = score
            breakdown[f"turn_{i}_action_count"] = action_count
            breakdown[f"turn_{i}_has_structure"] = has_list

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_turns": len(assistant_turns)},
        )


# ============================================================================
# 3. Neural Reward Models (Optional)
# ============================================================================

class SentimentReward(RewardFunction):
    """
    Uses sentiment analysis to reward positive, helpful tone

    Requires: pip install textblob
    Optional: For better results, use transformers-based sentiment models
    """

    def __init__(self, weight: float = 1.0, use_transformer: bool = False):
        super().__init__(weight, RewardType.IMMEDIATE, "SentimentReward")
        self.use_transformer = use_transformer

        if use_transformer:
            try:
                from transformers import pipeline
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
            except ImportError:
                logger.warning("transformers not available, falling back to TextBlob")
                self.use_transformer = False

        if not self.use_transformer:
            try:
                from textblob import TextBlob
                self.textblob = TextBlob
            except ImportError:
                logger.warning("textblob not available, using basic sentiment")
                self.textblob = None

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute reward based on sentiment"""

        assistant_turns = [t for t in turns if t.role == "assistant"]
        if not assistant_turns:
            return RewardResult(score=0.0, breakdown={}, metadata={})

        total_score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            if self.use_transformer:
                result = self.sentiment_analyzer(turn.content[:512])[0]
                # Convert to 0-1 scale
                if result['label'] == 'POSITIVE':
                    sentiment = result['score']
                else:
                    sentiment = 1 - result['score']
            elif self.textblob:
                blob = self.textblob(turn.content)
                # TextBlob polarity is -1 to 1, normalize to 0-1
                sentiment = (blob.sentiment.polarity + 1) / 2
            else:
                # Simple fallback
                sentiment = 0.7  # Neutral-positive default

            total_score += sentiment
            breakdown[f"turn_{i}_sentiment"] = sentiment

        avg_score = total_score / len(assistant_turns)

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_turns": len(assistant_turns), "model": "transformer" if self.use_transformer else "textblob"},
        )


# ============================================================================
# 4. Reward Composition Examples
# ============================================================================

def create_balanced_customer_service_reward() -> CompositeReward:
    """
    Create a well-balanced reward for customer service agents

    Combines multiple aspects:
    - Empathy (30%)
    - Action-oriented responses (25%)
    - Question answering (25%)
    - Appropriate length (10%)
    - Positive sentiment (10%)
    """
    return CompositeReward([
        EmpathyReward(weight=0.30),
        ActionOrientedReward(weight=0.25),
        QuestionAnsweringReward(weight=0.25),
        ResponseLengthReward(weight=0.10),
        SentimentReward(weight=0.10, use_transformer=False),
    ])


def create_technical_support_reward() -> CompositeReward:
    """
    Create reward optimized for technical support

    Emphasizes:
    - Action-oriented (40%)
    - Question answering (35%)
    - Appropriate length (15%)
    - Positive sentiment (10%)
    """
    return CompositeReward([
        ActionOrientedReward(weight=0.40),
        QuestionAnsweringReward(weight=0.35),
        ResponseLengthReward(
            weight=0.15,
            min_length=100,
            max_length=500,
            optimal_min=150,
            optimal_max=350,
        ),
        SentimentReward(weight=0.10),
    ])


def create_empathetic_agent_reward() -> CompositeReward:
    """
    Create reward for highly empathetic agent (e.g., mental health support)

    Prioritizes:
    - Empathy (50%)
    - Sentiment (25%)
    - Question answering (15%)
    - Length (10%)
    """
    return CompositeReward([
        EmpathyReward(weight=0.50),
        SentimentReward(weight=0.25),
        QuestionAnsweringReward(weight=0.15),
        ResponseLengthReward(weight=0.10),
    ])


# ============================================================================
# 5. Curriculum Learning with Adaptive Rewards
# ============================================================================

class CurriculumReward(RewardFunction):
    """
    Adaptive reward that changes difficulty over time

    Starts with simple rewards (length, sentiment) and gradually
    introduces more complex rewards (empathy, action-oriented).
    """

    def __init__(
        self,
        weight: float = 1.0,
        initial_rewards: Optional[List[RewardFunction]] = None,
        advanced_rewards: Optional[List[RewardFunction]] = None,
        transition_episodes: int = 50,
    ):
        super().__init__(weight, RewardType.CUMULATIVE, "CurriculumReward")

        self.initial_rewards = initial_rewards or [
            ResponseLengthReward(weight=0.6),
            SentimentReward(weight=0.4),
        ]

        self.advanced_rewards = advanced_rewards or [
            EmpathyReward(weight=0.3),
            ActionOrientedReward(weight=0.3),
            QuestionAnsweringReward(weight=0.2),
            ResponseLengthReward(weight=0.1),
            SentimentReward(weight=0.1),
        ]

        self.transition_episodes = transition_episodes
        self.episode_count = 0

    async def compute_reward(
        self, turns: List[ConversationTurn], context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute reward with curriculum progression"""

        # Update episode count
        self.episode_count += 1

        # Calculate interpolation factor (0 = initial, 1 = advanced)
        alpha = min(1.0, self.episode_count / self.transition_episodes)

        # Compute rewards from both sets
        initial_results = [
            await reward.compute_reward(turns, context)
            for reward in self.initial_rewards
        ]
        advanced_results = [
            await reward.compute_reward(turns, context)
            for reward in self.advanced_rewards
        ]

        # Weighted combination of initial and advanced
        initial_score = sum(
            r.score * self.initial_rewards[i].weight
            for i, r in enumerate(initial_results)
        ) / sum(r.weight for r in self.initial_rewards)

        advanced_score = sum(
            r.score * self.advanced_rewards[i].weight
            for i, r in enumerate(advanced_results)
        ) / sum(r.weight for r in self.advanced_rewards)

        # Interpolate
        final_score = (1 - alpha) * initial_score + alpha * advanced_score

        breakdown = {
            "initial_score": initial_score,
            "advanced_score": advanced_score,
            "curriculum_alpha": alpha,
            "episode_count": self.episode_count,
        }

        return RewardResult(
            score=final_score,
            breakdown=breakdown,
            metadata={
                "curriculum_stage": "initial" if alpha < 0.5 else "advanced",
                "progression": f"{alpha * 100:.1f}%",
            },
        )


# ============================================================================
# Demo and Testing
# ============================================================================

async def test_reward_function(
    reward_fn: RewardFunction,
    test_conversations: List[List[ConversationTurn]]
):
    """Test a reward function on sample conversations"""

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Testing Reward: {reward_fn.name}")
    logger.info(f"{'=' * 80}\n")

    for i, conversation in enumerate(test_conversations):
        logger.info(f"Conversation {i + 1}:")
        for turn in conversation:
            logger.info(f"  {turn.role.upper()}: {turn.content[:100]}...")

        result = await reward_fn.compute_reward(conversation)

        logger.info(f"\n  Reward Score: {result.score:.3f}")
        logger.info(f"  Breakdown: {result.breakdown}")
        logger.info(f"  Metadata: {result.metadata}\n")


def create_test_conversations() -> List[List[ConversationTurn]]:
    """Create sample conversations for testing"""

    return [
        # Conversation 1: Good customer service
        [
            ConversationTurn(
                role="user",
                content="My order hasn't arrived and I'm really frustrated. It's been 2 weeks!",
            ),
            ConversationTurn(
                role="assistant",
                content="I completely understand your frustration, and I apologize for the delay. Let me look into this right away. I'll check the shipping status and see what we can do to resolve this quickly. Can you provide your order number?",
            ),
        ],

        # Conversation 2: Poor response
        [
            ConversationTurn(
                role="user",
                content="How do I reset my password?",
            ),
            ConversationTurn(
                role="assistant",
                content="You need to reset it.",
            ),
        ],

        # Conversation 3: Technical support
        [
            ConversationTurn(
                role="user",
                content="What causes a memory leak in Python?",
            ),
            ConversationTurn(
                role="assistant",
                content="Memory leaks in Python can occur due to several reasons. First, circular references between objects can prevent garbage collection. Second, keeping references to large objects unnecessarily. Here's what you can do: 1) Use weak references for circular deps, 2) Profile with memory_profiler, 3) Check for unclosed file handles or database connections.",
            ),
        ],
    ]


async def main():
    parser = argparse.ArgumentParser(
        description="Custom Reward Functions Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--reward-type",
        type=str,
        default="all",
        choices=["all", "length", "qa", "empathy", "action", "sentiment", "composite", "curriculum"],
        help="Type of reward to test (default: all)"
    )

    parser.add_argument(
        "--list-rewards",
        action="store_true",
        help="List all available reward types"
    )

    args = parser.parse_args()

    if args.list_rewards:
        print("\nAvailable Custom Reward Functions:")
        print("=" * 60)
        print("1. length       - ResponseLengthReward")
        print("2. qa           - QuestionAnsweringReward")
        print("3. empathy      - EmpathyReward")
        print("4. action       - ActionOrientedReward")
        print("5. sentiment    - SentimentReward")
        print("6. composite    - Composite reward combinations")
        print("7. curriculum   - Adaptive curriculum learning reward")
        print("=" * 60)
        return

    # Create test conversations
    test_convs = create_test_conversations()

    # Test selected reward functions
    if args.reward_type in ["all", "length"]:
        await test_reward_function(ResponseLengthReward(), test_convs)

    if args.reward_type in ["all", "qa"]:
        await test_reward_function(QuestionAnsweringReward(), test_convs)

    if args.reward_type in ["all", "empathy"]:
        await test_reward_function(EmpathyReward(), test_convs)

    if args.reward_type in ["all", "action"]:
        await test_reward_function(ActionOrientedReward(), test_convs)

    if args.reward_type in ["all", "sentiment"]:
        await test_reward_function(SentimentReward(use_transformer=False), test_convs)

    if args.reward_type in ["all", "composite"]:
        logger.info("\n" + "=" * 80)
        logger.info("Testing Composite Rewards")
        logger.info("=" * 80)

        composite = create_balanced_customer_service_reward()
        await test_reward_function(composite, test_convs)

    if args.reward_type in ["all", "curriculum"]:
        curriculum = CurriculumReward()
        await test_reward_function(curriculum, test_convs)

    logger.info("\n" + "=" * 80)
    logger.info("Testing Complete!")
    logger.info("=" * 80)
    logger.info("\nTo use these rewards in training:")
    logger.info("  reward_fn = create_balanced_customer_service_reward()")
    logger.info("  environment = ConversationEnvironment(..., reward_fn=reward_fn)")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
