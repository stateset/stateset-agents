"""
Test fixtures for StateSet Agents.

Provides reusable fixtures for unit and integration tests.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


class AgentFixture:
    """Fixture for creating test agents."""

    @staticmethod
    def create_mock(
        model_name: str = "gpt2",
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> MagicMock:
        """Create a mock agent for testing."""
        agent = MagicMock()
        agent.config.model_name = model_name
        agent.config.system_prompt = system_prompt or "You are a test agent."
        agent.config.max_new_tokens = kwargs.get("max_new_tokens", 512)
        agent.config.temperature = kwargs.get("temperature", 0.7)
        agent.generate_response = MagicMock(return_value="Test response")
        agent.initialize = MagicMock(return_value=None)
        return agent

    @staticmethod
    def create_stub_config(
        model_name: str = "stub://test",
        **kwargs
    ) -> Dict[str, Any]:
        """Create a stub agent configuration for testing."""
        return {
            "model_name": model_name,
            "system_prompt": kwargs.get("system_prompt", "You are a test agent."),
            "max_new_tokens": kwargs.get("max_new_tokens", 128),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
        }


class EnvironmentFixture:
    """Fixture for creating test environments."""

    @staticmethod
    def create_mock(
        scenarios: Optional[List[Dict]] = None,
        **kwargs
    ) -> MagicMock:
        """Create a mock environment for testing."""
        env = MagicMock()
        env.scenarios = scenarios or EnvironmentFixture.default_scenarios()
        env.max_turns = kwargs.get("max_turns", 5)
        env.reset = MagicMock(return_value={"state": "initial"})
        env.step = MagicMock(return_value=("next_state", 0.5, False, {}))
        return env

    @staticmethod
    def default_scenarios() -> List[Dict[str, Any]]:
        """Return default test scenarios."""
        return [
            {
                "id": "test-1",
                "topic": "test",
                "context": "Test scenario context",
                "user_responses": ["Hello", "Tell me more", "Thanks"],
            },
            {
                "id": "test-2",
                "topic": "test",
                "context": "Another test scenario",
                "user_responses": ["Hi", "What?", "Goodbye"],
            },
        ]

    @staticmethod
    def create_conversation_config(
        max_turns: int = 5,
        num_scenarios: int = 2,
    ) -> Dict[str, Any]:
        """Create a conversation environment configuration."""
        scenarios = []
        for i in range(num_scenarios):
            scenarios.append({
                "id": f"scenario-{i}",
                "topic": "test",
                "context": f"Test scenario {i} context",
                "user_responses": ["Hello", "Tell me more", "Thanks"],
            })

        return {
            "scenarios": scenarios,
            "max_turns": max_turns,
            "reward_on_completion": True,
        }


class RewardFixture:
    """Fixture for creating test reward functions."""

    @staticmethod
    def create_mock(value: float = 0.5) -> MagicMock:
        """Create a mock reward function."""
        reward = MagicMock()
        reward.compute = MagicMock(return_value=value)
        reward.__call__ = MagicMock(return_value=value)
        return reward

    @staticmethod
    def constant(value: float = 1.0):
        """Create a constant reward function."""
        async def _reward(*args, **kwargs):
            return value
        return _reward

    @staticmethod
    def deterministic(mapping: Dict[str, float]):
        """Create a deterministic reward based on input."""
        async def _reward(text: str, **kwargs):
            return mapping.get(text, 0.0)
        return _reward


class ModelFixture:
    """Fixture for creating test model configurations."""

    @staticmethod
    def create_mock_config(
        model_name: str = "gpt2",
        **kwargs
    ) -> Dict[str, Any]:
        """Create a mock model configuration."""
        return {
            "model_name": model_name,
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
        }

    @staticmethod
    def create_tokenizer_mock(vocab_size: int = 50257) -> MagicMock:
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.vocab_size = vocab_size
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.decode = MagicMock(return_value="Test text")
        tokenizer.__len__ = MagicMock(return_value=vocab_size)
        return tokenizer

    @staticmethod
    def small_models() -> List[str]:
        """Return list of small models suitable for testing."""
        return [
            "gpt2",
            "gpt2-medium",
            "distilgpt2",
        ]


@pytest.fixture
def mock_agent():
    """Pytest fixture for mock agent."""
    return AgentFixture.create_mock()


@pytest.fixture
def mock_environment():
    """Pytest fixture for mock environment."""
    return EnvironmentFixture.create_mock()


@pytest.fixture
def mock_reward():
    """Pytest fixture for mock reward function."""
    return RewardFixture.create_mock()


@pytest.fixture
def conversation_config():
    """Pytest fixture for conversation environment config."""
    return EnvironmentFixture.create_conversation_config()
