"""
Deprecated shim for backwards compatibility.

The canonical package is ``stateset_agents.api``.

.. deprecated:: 0.8.0
    This module will be removed in a future version.
    Import from ``stateset_agents.api`` instead.
"""

import warnings
from importlib import import_module
import sys

warnings.warn(
    "Importing from 'api' is deprecated. "
    "Use 'stateset_agents.api' instead. "
    "This shim will be removed in version 1.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = import_module("stateset_agents.api")
globals().update(_canonical.__dict__)
sys.modules[__name__] = _canonical
