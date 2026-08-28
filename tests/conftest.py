"""Shared test configuration and fixtures for all test modules."""

import asyncio
import atexit
import inspect
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.agent_backends import create_stub_backend

try:
    from _pytest.fixtures import FixtureDef

    if not hasattr(FixtureDef, "unittest"):
        FixtureDef.unittest = False  # type: ignore[attr-defined]
except Exception:
    pass

if "app" not in inspect.signature(httpx.Client.__init__).parameters and not getattr(
    httpx.Client, "_stateset_agents_app_compat", False
):
    _original_httpx_client_init = httpx.Client.__init__

    def _compat_httpx_client_init(self, *args, **kwargs):
        kwargs.pop("app", None)
        return _original_httpx_client_init(self, *args, **kwargs)

    httpx.Client.__init__ = _compat_httpx_client_init  # type: ignore[assignment]
    httpx.Client._stateset_agents_app_compat = True  # type: ignore[attr-defined]

# Set up API environment variables BEFORE importing API modules that might read
# them at import time. This lets API tests run without manual env setup.
os.environ.setdefault("API_ENVIRONMENT", "development")
os.environ.setdefault(
    "API_JWT_SECRET", "test-secret-key-for-testing-purposes-only-minimum-32-chars"
)
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
        return_value: Any = None, side_effect: Any | None = None
    ) -> AsyncMock:
        mock_obj = AsyncMock()
        if return_value is not None:
            mock_obj.return_value = return_value
        if side_effect is not None:
            mock_obj.side_effect = side_effect
        return mock_obj

    @staticmethod
    def mock_agent_response(responses: list[str]):
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


@pytest.fixture
def force_cpu():
    """Force tests to run as if no GPU is available.

    This is opt-in only — use it in tests that explicitly need CPU-only
    behaviour.  Most tests should use the stub backend instead.
    """
    with mock.patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture(autouse=True)
def mock_transformers_logging():
    """Silence noisy transformers logs during the test suite."""
    logging.getLogger("transformers").setLevel(logging.WARNING)
    yield


@pytest.fixture(autouse=True)
def isolate_cost_ledger(tmp_path_factory, monkeypatch):
    """Keep the suite out of the user's real cost ledger.

    The remote executors append a spend record for every job they run, at a
    per-user path outside the repo. Tests that exercise submit() with fake
    providers were writing zero-cost rows straight into that file — the
    user's own accounting — so every test gets its own ledger instead.
    """
    ledger = tmp_path_factory.mktemp("cost_ledger") / "cost_ledger.jsonl"
    monkeypatch.setattr(
        "stateset_agents.remote.ledger.DEFAULT_LEDGER_PATH", ledger, raising=False
    )
    yield ledger


@pytest.fixture(autouse=True)
def reset_torch_default_dtype():
    """Keep the suite isolated from tests that mutate PyTorch global dtype.

    Guarded: some tests deliberately simulate a torch-less environment (by
    removing it from ``sys.modules`` or making its import fail), and this
    fixture's teardown can run while that simulation is still in place.
    Isolation must not turn such a test into an error.
    """

    def _reset() -> None:
        try:
            torch.set_default_dtype(torch.float32)
        except Exception:  # pragma: no cover - torch simulated away
            pass

    _reset()
    yield
    _reset()


@pytest.fixture(autouse=True)
def restore_stateset_agents_sys_modules():
    """Undo any `sys.modules` surgery a test performs on our own package tree.

    Several tests exercise lazy-import behaviour by doing
    ``sys.modules.pop("stateset_agents...", None)`` (or
    ``del sys.modules[name]``) and then re-importing fresh, without ever
    restoring the original module object afterwards. That leaves a *new*
    module object installed under the parent package's attribute for the
    rest of the process — e.g. popping and reimporting
    ``stateset_agents.api`` drops its ``.services`` attribute, because that
    attribute is only re-populated by Python's import machinery when a
    submodule is *actually loaded*, not when it's already cached elsewhere
    in ``sys.modules``. Any later test that does
    ``import stateset_agents.api.services`` sees the submodule already in
    ``sys.modules`` and skips re-setting the parent's attribute, so the
    stale, attribute-less package object leaks into every subsequent test
    in the same process — reproducing only when the whole suite runs
    together, never in isolation.

    Snapshot every ``sys.modules`` entry under the ``stateset_agents``
    package tree before each test and force it back to the exact same
    object afterwards, regardless of what the test did to it. This is a
    no-op for the overwhelming majority of tests (which don't touch
    ``sys.modules``) and heals the handful of lazy-import tests that do,
    without having to rewrite each of them individually.

    Restoring ``sys.modules`` alone is not enough: tools like
    ``unittest.mock.patch`` resolve dotted targets by attribute-walking
    from the *root* module (``getattr(stateset_agents, "api")``, then
    ``getattr(that, "services")``, ...), not by looking each component up
    in ``sys.modules``. Re-importing a submodule also rebinds it as an
    attribute on its parent package object, so the parent's attribute has
    to be re-pointed at the original child object too, or attribute-walking
    consumers keep seeing the leaked replacement.
    """
    snapshot = {
        name: module
        for name, module in sys.modules.items()
        if name == "stateset_agents" or name.startswith("stateset_agents.")
    }
    yield
    # Restore sys.modules first, then re-link each parent package's
    # attribute to the original child module object, shallowest names
    # first so a parent is back in place before we bind its children onto
    # it.
    for name in sorted(snapshot, key=lambda n: n.count(".")):
        module = snapshot[name]
        sys.modules[name] = module
        if "." in name:
            parent_name, leaf = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, leaf, module)


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


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
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


# ---------------------------------------------------------------------------
# Real stub-backend fixtures (prefer these over MagicMock-based mocks)
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_agent_config():
    """Agent config that uses the real stub backend — no mocking required."""
    return AgentConfig(
        model_name="stub://test",
        use_stub_model=True,
        max_new_tokens=64,
        temperature=0.7,
    )


@pytest.fixture
async def initialized_stub_agent(stub_agent_config):
    """A fully initialized MultiTurnAgent running the real stub backend."""
    agent = MultiTurnAgent(stub_agent_config)
    await agent.initialize()
    return agent


@pytest.fixture
def stub_backend():
    """Directly provide a StubBackend for dependency-injection tests."""
    return create_stub_backend(
        stub_responses=["Test response one.", "Test response two."],
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=0,
        eos_token_id=0,
    )


# ---------------------------------------------------------------------------
# Clean interpreter shutdown (silence litellm/wandb atexit tracebacks)
# ---------------------------------------------------------------------------
#
# At interpreter exit litellm's own atexit hook (registered by
# ``litellm.llms.custom_httpx.async_client_cleanup``) calls
# ``asyncio.get_event_loop()``; if pytest-asyncio has left no usable loop it
# falls back to ``asyncio.new_event_loop()``, which emits a DEBUG record
# ("Using selector: EpollSelector"). Tests that exercise the API logging setup
# leave a root ``StreamHandler`` bound to pytest's captured stdout, which is
# closed by then — and wandb's console capture may still be wrapping the same
# stream — so the record surfaces as a "--- Logging error ---" traceback after
# the test summary.
#
# ``atexit`` runs LIFO, so a hook registered at session finish (i.e. after
# litellm was imported) runs *before* litellm's. We use that window to drain
# litellm's async clients, install a fresh event loop so litellm's hook takes
# the quiet path, restore the real stdio, and drop logging handlers whose
# stream is already closed.


def _quiet_interpreter_shutdown() -> None:
    """Best-effort teardown that runs before litellm's own atexit hook."""
    # 1. Drop logging handlers whose stream is already closed (pytest's
    #    captured stdout), which is what turns a stray DEBUG record into a
    #    "--- Logging error ---" traceback. This must happen first: the steps
    #    below can themselves emit DEBUG records.
    manager = logging.Logger.manager
    loggers: list[logging.Logger] = [logging.getLogger()]
    loggers.extend(
        logger
        for logger in manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )
    for logger in loggers:
        for handler in list(getattr(logger, "handlers", [])):
            stream = getattr(handler, "stream", None)
            if stream is not None and getattr(stream, "closed", False):
                try:
                    logger.removeHandler(handler)
                except Exception:  # pragma: no cover - defensive teardown
                    pass

    # 2. Undo wandb's console capture / any stdio replacement so writes during
    #    shutdown go to the real streams rather than a wrapped, closed one.
    for name, real in (("stdout", sys.__stdout__), ("stderr", sys.__stderr__)):
        try:
            if real is not None and getattr(real, "closed", False) is False:
                setattr(sys, name, real)
        except Exception:  # pragma: no cover - defensive teardown
            pass

    # 3. Drain litellm's cached async clients and leave a usable event loop
    #    behind so litellm's cleanup_wrapper never calls new_event_loop().
    if "litellm" in sys.modules:
        try:
            from litellm.llms.custom_httpx.async_client_cleanup import (
                close_litellm_async_clients,
            )

            asyncio.run(close_litellm_async_clients())
        except Exception:  # pragma: no cover - defensive teardown
            pass
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
        except Exception:  # pragma: no cover - defensive teardown
            pass


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Register the shutdown cleanup so it runs before litellm's atexit hook."""
    atexit.register(_quiet_interpreter_shutdown)
