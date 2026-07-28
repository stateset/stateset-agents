"""Local pytest config for the testing examples — picks `asyncio_mode = auto`."""

# pytest-asyncio reads this from `pytest.ini` / `pyproject.toml` / `setup.cfg`,
# but a local `pytestmark` keeps the examples portable without a config file.
import pytest

pytest_plugins = ("pytest_asyncio",)


def pytest_collection_modifyitems(config, items):
    """Auto-apply the asyncio marker to coroutine tests."""
    import asyncio

    for item in items:
        if asyncio.iscoroutinefunction(getattr(item, "function", None)):
            item.add_marker(pytest.mark.asyncio)
