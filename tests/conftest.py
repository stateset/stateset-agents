"""Shared test configuration and fixtures for all test modules."""

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

# Set up API environment variables BEFORE any imports that might use them
# This ensures API tests can run without manual environment setup
os.environ.setdefault("API_ENVIRONMENT", "development")
os.environ.setdefault("API_JWT_SECRET", "test-secret-key-for-testing-purposes-only-minimum-32-chars")
os.environ.setdefault("API_CORS_ORIGINS", "*")
os.environ.setdefault("API_REQUIRE_AUTH", "false")
os.environ.setdefault("API_RATE_LIMIT_ENABLED", "false")

try:
    import torch
except ImportError:  # pragma: no cover - test suite assumes PyTorch is available
    torch = None  # type: ignore[assignment]

if torch is None:  # pragma: no cover - short-circuit tests when torch missing
    pytest.skip("PyTorch is required for the test suite", allow_module_level=True)


# Ensure repository root is importable for tests that rely on top-level packages.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def event_loop():
    """Create an isolated event loop for the entire test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir():
    """Provide a temporary directory for tests that need filesystem access."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_model_and_tokenizer():
    """Provide a mocked HF model/tokenizer pair used across unit tests."""
    model = MagicMock()
    tokenizer = MagicMock()

    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = 2
    tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = MagicMock(return_value="Mock response")
    tokenizer.encode = MagicMock(return_value=[1, 2, 3])

    model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
    return model, tokenizer


@pytest.fixture
def sample_conversation_messages():
    """Provide sample conversation messages for multi-turn tests."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I learn Python?"},
        {"role": "assistant", "content": "Start with the official tutorial!"},
        {"role": "user", "content": "What about practical projects?"},
    ]


@pytest.fixture
def sample_conversation_scenarios():
    """Provide sample conversation scenarios for environment tests."""
    return [
        {
            "id": "learning_python",
            "topic": "education",
            "context": "User wants to learn Python programming",
            "user_responses": [
                "Hi, I want to learn Python. Where should I start?",
                "That sounds good. What about practical projects?",
                "Great suggestions! How long will it take?",
                "Thank you for all the helpful advice!",
            ],
        },
        {
            "id": "technical_help",
            "topic": "technical_support",
            "context": "User needs technical assistance",
            "user_responses": [
                "I'm having trouble with my code.",
                "I get this error message. What does it mean?",
                "I tried that but it didn't work. Any other suggestions?",
                "That fixed it! Thank you so much.",
            ],
        },
    ]


@pytest.fixture
def mock_reward_function():
    """Provide a mock reward function for tests that need reward callbacks."""
    reward_fn = MagicMock()
    reward_fn.compute_reward = AsyncMock(return_value=0.85)
    reward_fn.weight = 1.0
    return reward_fn


@pytest.fixture
def mock_environment():
    """Provide a mock environment for simulating agent interactions."""
    env = MagicMock()
    env.reset = AsyncMock(return_value={"step": 0, "context": "test"})
    env.step = AsyncMock(
        return_value={"state": {"step": 1}, "reward": 0.8, "done": False}
    )
    return env


class AsyncMockHelper:
    """Helper for creating async mocks with configurable behaviour."""

    @staticmethod
    def create_async_mock(
        return_value: Any = None, side_effect: Optional[Any] = None
    ) -> AsyncMock:
        mock_obj = AsyncMock()
        if return_value is not None:
            mock_obj.return_value = return_value
        if side_effect is not None:
            mock_obj.side_effect = side_effect
        return mock_obj

    @staticmethod
    def mock_agent_response(responses: List[str]):
        """Create a coroutine that cycles through provided responses."""
        response_index = 0

        async def mock_generate_response(*_: Any, **__: Any) -> str:
            nonlocal response_index
            response = responses[response_index % len(responses)]
            response_index += 1
            return response

        return mock_generate_response


@pytest.fixture
def async_mock_helper() -> AsyncMockHelper:
    """Fixture exposing the async mock helper class."""
    return AsyncMockHelper()


@pytest.fixture(autouse=True)
def mock_torch_cuda():
    """Ensure tests run as if no GPU is available."""
    with mock.patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture(autouse=True)
def mock_transformers_logging():
    """Silence noisy transformers logs during the test suite."""
    logging.getLogger("transformers").setLevel(logging.WARNING)
    yield


def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers to keep selection explicit."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line(
        "markers",
        "integration: Integration tests (slower, may need external resources)",
    )
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "e2e: End-to-end scenario tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")


# API Testing Support
@pytest.fixture(scope="session")
def api_test_env():
    """Ensure API environment is configured for testing."""
    original_env = {}
    test_vars = {
        "API_ENVIRONMENT": "development",
        "API_JWT_SECRET": "test-secret-key-for-testing-purposes-only-minimum-32-chars",
        "API_CORS_ORIGINS": "*",
        "API_REQUIRE_AUTH": "false",
        "API_RATE_LIMIT_ENABLED": "false",
    }

    # Store original values and set test values
    for key, value in test_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    yield test_vars

    # Restore original values
    for key, original in original_env.items():
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Automatically add markers based on test location."""
    for item in items:
        # Add markers based on path
        if "/api/" in str(item.fspath):
            item.add_marker(pytest.mark.api)
        elif "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
