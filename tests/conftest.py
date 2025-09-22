"""
import unittest
import unittest
Shared test configuration and fixtures for all test modules.

This file provides common test utilities, fixtures, and configuration
used across different test categories.
"""
import unittest

import pytest
import sys
from pathlib import Path

# Ensure the repository root is at the front of sys.path so that imports like
# `import core` resolve to this repo's top-level packages, not similarly named
# packages from other local projects.
try:
    _repo_root = str(Path(__file__).resolve().parents[1])
    if _repo_root in sys.path:
        sys.path.remove(_repo_root)
    sys.path.insert(0, _repo_root)
except Exception:
    pass
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any, List, Optional

import torch


@pytest.fixture(scope="session")
def event_loop():
    """
import unittestCreate an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir():
    """
import unittestCreate a temporary directory for the test session."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_model_and_tokenizer():
    """
import unittestProvide mock model and tokenizer for testing."""
    model = MagicMock()
    tokenizer = MagicMock()
    
    # Configure common tokenizer attributes
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = 2
    tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = MagicMock(return_value="Mock response")
    tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    
    # Configure model
    model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
    
    return model, tokenizer


@pytest.fixture
def sample_conversation_messages():
    """
import unittestProvide sample conversation messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I learn Python?"},
        {"role": "assistant", "content": "Start with the official tutorial!"},
        {"role": "user", "content": "What about practical projects?"}
    ]


@pytest.fixture
def sample_conversation_scenarios():
    """
import unittestProvide sample conversation scenarios for testing."""
    return [
        {
            "id": "learning_python",
            "topic": "education",
            "context": "User wants to learn Python programming",
            "user_responses": [
                "Hi, I want to learn Python. Where should I start?",
                "That sounds good. What about practical projects?",
                "Great suggestions! How long will it take?",
                "Thank you for all the helpful advice!"
            ]
        },
        {
            "id": "technical_help",
            "topic": "technical_support",
            "context": "User needs technical assistance",
            "user_responses": [
                "I'm having trouble with my code.",
                "I get this error message. What does it mean?",
                "I tried that but it didn't work. Any other suggestions?",
                "That fixed it! Thank you so much."
            ]
        }
    ]


@pytest.fixture
def mock_reward_function():
    """
import unittestProvide a mock reward function for testing."""
    reward_fn = MagicMock()
    reward_fn.compute_reward = AsyncMock(return_value=0.85)
    reward_fn.weight = 1.0
    return reward_fn


@pytest.fixture
def mock_environment():
    """
import unittestProvide a mock environment for testing."""
    env = MagicMock()
    env.reset = AsyncMock(return_value={"step": 0, "context": "test"})
    env.step = AsyncMock(return_value={
        "state": {"step": 1},
        "reward": 0.8,
        "done": False
    })
    return env


class AsyncMockHelper:
    """
import unittestHelper class for creating async mocks with proper behavior."""
    
    @staticmethod
    def create_async_mock(return_value=None, side_effect=None):
        """
import unittestCreate an async mock with specified behavior."""
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock
    
    @staticmethod
    def mock_agent_response(responses: List[str]):
        """
import unittestCreate a mock agent that cycles through responses."""
        response_index = 0
        
        async def mock_generate_response(*args, **kwargs):
            nonlocal response_index
            response = responses[response_index % len(responses)]
            response_index += 1
            return response
        
        return mock_generate_response


@pytest.fixture
def async_mock_helper():
    """
import unittestProvide async mock helper for tests."""
    return AsyncMockHelper()


@pytest.fixture(autouse=True)
def mock_torch_cuda():
    """
import unittestMock torch.cuda to avoid GPU-related issues in tests."""
    with unittest.mock.patch('torch.cuda.is_available', return_value=False):
        yield


@pytest.fixture(autouse=True)
def mock_transformers_logging():
    """
import unittestSuppress transformers logging in tests."""
    import logging
    logging.getLogger("transformers").setLevel(logging.WARNING)
    yield


# Custom markers
def pytest_configure(config):
    """
import unittestConfigure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (slower, may need external resources)")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")


@pytest.fixture
def performance_monitor():
    """
import unittestProvide a simple performance monitor for tests."""
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            if self.start_time is None:
                return 0
            elapsed = time.time() - self.start_time
            self.start_time = None
            return elapsed
        
        def measure_async(self, coro):
            """
import unittestMeasure execution time of an async coroutine."""
            async def wrapper():
                self.start()
                result = await coro
                elapsed = self.stop()
                return result, elapsed
            return wrapper()
    
    return PerformanceMonitor()


# Test data factories
def create_test_trajectory(user_messages: List[str], assistant_responses: List[str], 
                          rewards: Optional[List[float]] = None) -> Dict[str, Any]:
    """
import unittestCreate test trajectory data."""
    if rewards is None:
        rewards = [0.8] * len(user_messages)
    
    turns = []
    for i, (user_msg, assistant_resp) in enumerate(zip(user_messages, assistant_responses)):
        turns.append({
            "user_message": user_msg,
            "assistant_response": assistant_resp,
            "reward": rewards[i] if i < len(rewards) else 0.8,
            "turn_number": i + 1
        })
    
    return {
        "trajectory_id": "test_trajectory_123",
        "turns": turns,
        "total_reward": sum(rewards),
        "average_reward": sum(rewards) / len(rewards) if rewards else 0
    }


@pytest.fixture
def test_trajectory_factory():
    """
import unittestProvide test trajectory factory."""
    return create_test_trajectory
