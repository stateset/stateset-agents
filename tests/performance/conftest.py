"""Skip pytest-benchmark-dependent tests when the plugin isn't active.

``tests/performance/test_benchmarks.py`` uses the ``benchmark`` fixture
supplied by the pytest-benchmark plugin. If the plugin package isn't
installed, the module-level ``pytest.importorskip("pytest_benchmark")`` in
that file already skips it cleanly at import time. But the package can be
*installed* while the plugin is explicitly disabled (e.g. ``-p
no:benchmark``, used locally to simulate CI environments that don't ship the
plugin) — in that case the import succeeds but no ``benchmark`` fixture is
registered, so tests would otherwise fail at fixture-resolution time. Catch
that case here.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    if config.pluginmanager.hasplugin("benchmark"):
        return
    skip_no_benchmark_plugin = pytest.mark.skip(
        reason="pytest-benchmark plugin not available/enabled"
    )
    for item in items:
        if "test_benchmarks.py" in str(item.fspath):
            item.add_marker(skip_no_benchmark_plugin)
