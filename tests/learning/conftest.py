"""Configuration for the behavioural "does it actually learn" suite."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "learning: behavioural RL tests asserting a trainer measurably improves "
        "a toy policy (and that a sign-flipped reward moves it the other way)",
    )


@pytest.fixture(autouse=True)
def _single_threaded_torch():
    """Tiny models run fastest single-threaded, and -n 8 workers oversubscribe
    the CPU badly otherwise (a ~10x slowdown on a loaded box)."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
