"""
Data processing utilities for GRPO Agent Framework

This module provides utilities for loading, validating, and preprocessing
conversational data for GRPO training, inspired by real-world implementations.
"""

import json
import importlib.util
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_SKLEARN_LAZY = object()
train_test_split = _SKLEARN_LAZY  # type: ignore[assignment]
_SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

from .trajectory import ConversationTurn, MultiTurnTrajectory

logger = logging.getLogger(__name__)


@dataclass
class ConversationExample:
    """Data structure for conversation training examples"""

    query: str
    expected_response: Optional[str] = None
    task_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[ConversationTurn] = field(default_factory=list)


class DataLoader:
    """Load and preprocess conversational data"""

    def __init__(
        self,
        max_examples: Optional[int] = None,
        validation_split: float = 0.1,
        stratify_by: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.max_examples = max_examples
        self.validation_split = validation_split
        self.stratify_by = stratify_by
        self.random_seed = random_seed
        random.seed(random_seed)

    def load_jsonl(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load conversations from JSONL file"""
        file_path = Path(file_path)
        data = []

        if not file_path.exists():
            logger.warning(f"Data file not found: {file_path}")
            return self._get_fallback_data()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        conv = json.loads(line.strip())
                        if self.validate_conversation(conv):
                            data.append(conv)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {line_num}: {e}")
                        continue

                    if self.max_examples and len(data) >= self.max_examples:
                        break

        except Exception as e:
            logger.error(f"Error reading data file: {e}")
            return self._get_fallback_data()

        if not data:
            logger.warning("No valid data loaded. Using fallback sample data.")
            return self._get_fallback_data()

        logger.info(f"Loaded {len(data)} conversations from {file_path}")
        return data

    def validate_conversation(self, conv: Dict[str, Any]) -> bool:
        """Validate conversation structure"""
        if not isinstance(conv, dict) or "messages" not in conv:
            return False

        messages = conv["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            return False

        # Check message structure
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False
            if msg["role"] not in ["user", "assistant", "system"]:
                return False

        return True

    def extract_examples(
        self, conversations: List[Dict[str, Any]], classify_task: bool = True
    ) -> List[ConversationExample]:
        """Extract training examples from conversations"""
        examples = []

        for conv_idx, conv in enumerate(conversations):
            messages = conv.get("messages", [])

            # Skip system messages and find user-assistant pairs
            user_message = None
            conversation_history = []

            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    continue

                if msg["role"] == "user" and user_message is None:
                    user_message = msg["content"]
                elif msg["role"] == "assistant" and user_message is not None:
                    # We have a user-assistant pair
                    query = user_message
                    response = msg["content"]

                    # Classify task type if requested
                    task_type = None
                    if classify_task:
                        task_type = self.classify_task_type(query, response)

                    example = ConversationExample(
                        query=query,
                        expected_response=response,
                        task_type=task_type,
                        metadata={
                            "conversation_id": conv_idx,
                            "turn_index": i,
                            "source_file": conv.get("source", "unknown"),
                        },
                        conversation_history=conversation_history.copy(),
                    )
                    examples.append(example)

                    # Add to conversation history
                    conversation_history.append(
                        ConversationTurn(
                            role="user",
                            content=user_message,
                            metadata={"turn_index": i - 1},
                        )
                    )
                    conversation_history.append(
                        ConversationTurn(
                            role="assistant",
                            content=response,
                            metadata={"turn_index": i},
                        )
                    )

                    # Reset for next pair
                    user_message = None
                elif msg["role"] == "user":
                    # New user message, update the current one
                    user_message = msg["content"]

        return examples

    def classify_task_type(self, query: str, response: str) -> str:
        """Classify the type of conversational task"""
        query_lower = query.lower()
        response_lower = response.lower()

        # Common task type patterns
        task_patterns = {
            "question_answering": ["what", "when", "where", "who", "why", "how", "?"],
            "instruction_following": [
                "please",
                "can you",
                "could you",
                "i need",
                "help me",
            ],
            "creative_writing": ["write", "story", "poem", "creative", "imagine"],
            "coding": ["code", "function", "program", "debug", "error"],
            "analysis": ["analyze", "explain", "compare", "evaluate", "assess"],
            "conversation": ["hi", "hello", "hey", "thanks", "goodbye"],
            "problem_solving": ["solve", "solution", "problem", "issue", "fix"],
        }

        # Score each task type
        scores = {}
        for task_type, keywords in task_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[task_type] = score

        # Return the task type with highest score
        if scores:
            return max(scores, key=scores.get)
        else:
            return "general"

    def split_train_eval(
        self,
        examples: List[ConversationExample],
        eval_size: Optional[float] = None,
        stratify: bool = True,
    ) -> Tuple[List[ConversationExample], List[ConversationExample]]:
        """Split examples into train and eval sets"""
        eval_size = eval_size or self.validation_split

        if eval_size <= 0 or len(examples) < 10:
            return examples, []

        splitter = _get_train_test_split()
        if splitter is None:
            train_examples, eval_examples = self._split_train_eval_fallback(
                examples=examples, eval_size=eval_size, stratify=stratify
            )
            logger.info(
                "Split data (fallback): %s train, %s eval",
                len(train_examples),
                len(eval_examples),
            )
            return train_examples, eval_examples

        if stratify and self.stratify_by == "task_type":
            # Stratify by task type
            task_types = [e.task_type for e in examples]
            if len(set(task_types)) > 1:
                train_examples, eval_examples = splitter(
                    examples,
                    test_size=eval_size,
                    random_state=self.random_seed,
                    stratify=task_types,
                )
            else:
                # No stratification possible
                train_examples, eval_examples = splitter(
                    examples, test_size=eval_size, random_state=self.random_seed
                )
        else:
            # Simple random split
            train_examples, eval_examples = splitter(
                examples, test_size=eval_size, random_state=self.random_seed
            )

        logger.info(
            f"Split data: {len(train_examples)} train, {len(eval_examples)} eval"
        )
        return train_examples, eval_examples

    def _split_train_eval_fallback(
        self,
        examples: List[ConversationExample],
        eval_size: float,
        stratify: bool,
    ) -> Tuple[List[ConversationExample], List[ConversationExample]]:
        """Split examples without sklearn.

        Uses a deterministic shuffle based on `random_seed` and optionally performs
        simple stratification by `task_type`.
        """
        rng = random.Random(self.random_seed)

        def split_bucket(items: List[ConversationExample]) -> Tuple[List[ConversationExample], List[ConversationExample]]:
            bucket = items.copy()
            rng.shuffle(bucket)
            split_idx = int(round(len(bucket) * (1.0 - float(eval_size))))
            split_idx = max(0, min(len(bucket), split_idx))
            return bucket[:split_idx], bucket[split_idx:]

        if stratify and self.stratify_by == "task_type":
            buckets: Dict[str, List[ConversationExample]] = {}
            for ex in examples:
                key = ex.task_type or "unknown"
                buckets.setdefault(key, []).append(ex)

            train_examples: List[ConversationExample] = []
            eval_examples: List[ConversationExample] = []
            for bucket_examples in buckets.values():
                train_bucket, eval_bucket = split_bucket(bucket_examples)
                train_examples.extend(train_bucket)
                eval_examples.extend(eval_bucket)

            rng.shuffle(train_examples)
            rng.shuffle(eval_examples)
            return train_examples, eval_examples

        return split_bucket(examples)

    def _get_fallback_data(self) -> List[Dict[str, Any]]:
        """Return fallback sample data for testing"""
        return [
            {
                "messages": [
                    {"role": "user", "content": "Hello! How are you today?"},
                    {
                        "role": "assistant",
                        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?",
                    },
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Can you explain what machine learning is?",
                    },
                    {
                        "role": "assistant",
                        "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make decisions or predictions based on those patterns.",
                    },
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Write a short poem about the ocean"},
                    {
                        "role": "assistant",
                        "content": "Waves dance upon the shore so blue,\nWhispering secrets, old yet new.\nThe ocean's heart beats deep and strong,\nAn endless, ancient, timeless song.",
                    },
                ]
            },
        ]


def _get_train_test_split():
    """Return sklearn's train_test_split if available, otherwise None.

    This avoids importing sklearn/scipy at module import time (expensive) and
    preserves the ability for tests to monkeypatch `train_test_split` to None.
    """
    global train_test_split

    if train_test_split is None:
        return None

    if train_test_split is _SKLEARN_LAZY:
        if not _SKLEARN_AVAILABLE:
            train_test_split = None  # type: ignore[assignment]
            return None
        try:
            from sklearn.model_selection import train_test_split as _tts  # type: ignore[import-not-found]

            train_test_split = _tts  # type: ignore[assignment]
        except Exception:  # pragma: no cover - optional dependency
            train_test_split = None  # type: ignore[assignment]

    return train_test_split  # type: ignore[return-value]


class DataProcessor:
    """Process and prepare data for GRPO training"""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def prepare_for_grpo(
        self, examples: List[ConversationExample], include_expected: bool = False
    ) -> List[Dict[str, Any]]:
        """Prepare examples for GRPO training format"""

        formatted_data = []
        for example in examples:
            # Format prompt
            prompt = self.format_prompt(example.query, example.conversation_history)

            data_point = {
                "prompt": prompt,
                "query": example.query,
                "task_type": example.task_type,
                "metadata": example.metadata,
            }

            # Optionally include expected response (for evaluation)
            if include_expected and example.expected_response:
                data_point["expected_response"] = example.expected_response

            formatted_data.append(data_point)

        return formatted_data

    def format_prompt(
        self, query: str, history: Optional[List[ConversationTurn]] = None
    ) -> str:
        """Format query and history into a prompt"""
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            # Use tokenizer's chat template if available
            messages = []

            # Add history
            if history:
                for turn in history:
                    messages.append({"role": turn.role, "content": turn.content})

            # Add current query
            messages.append({"role": "user", "content": query})

            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Simple formatting
            prompt_parts = []

            # Add history
            if history:
                for turn in history:
                    if turn.role == "user":
                        prompt_parts.append(f"User: {turn.content}")
                    elif turn.role == "assistant":
                        prompt_parts.append(f"Assistant: {turn.content}")

            # Add current query
            prompt_parts.append(f"User: {query}")
            prompt_parts.append("Assistant:")

            return "\n".join(prompt_parts)


# Utility functions


def load_and_prepare_data(
    data_path: Union[str, Path],
    max_examples: Optional[int] = None,
    validation_split: float = 0.1,
    tokenizer=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load and prepare data for training"""

    # Initialize data loader
    loader = DataLoader(
        max_examples=max_examples,
        validation_split=validation_split,
        stratify_by="task_type",
    )

    # Load conversations
    conversations = loader.load_jsonl(data_path)

    # Extract examples
    examples = loader.extract_examples(conversations)

    # Split train/eval
    train_examples, eval_examples = loader.split_train_eval(examples)

    # Process for GRPO
    processor = DataProcessor(tokenizer)
    train_data = processor.prepare_for_grpo(train_examples)
    eval_data = processor.prepare_for_grpo(eval_examples, include_expected=True)

    return train_data, eval_data
