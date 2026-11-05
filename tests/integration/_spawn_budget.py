"""Wall-clock budgets for tests that spawn multi-process Gloo workers.

Spawned workers re-import torch per process. On loaded Windows CI runners this
is several times slower than on Linux, so hard deadlines sized for Linux
(observed ~25 s for a 90 s budget) time out there. Budgets scale on Windows only.
"""

from __future__ import annotations

import sys

WINDOWS_SCALE = 2.0 if sys.platform == "win32" else 1.0


def spawn_budget(seconds: float) -> float:
    """Platform-adjusted budget for a multi-process spawn of ``seconds`` on Linux."""
    return seconds * WINDOWS_SCALE
