"""
Deprecated shim for backwards compatibility.

The canonical package is ``stateset_agents.core``.

.. deprecated:: 0.8.0
    This module will be removed in a future version.
    Import from ``stateset_agents.core`` instead.
"""

import warnings
from importlib import import_module
import sys

warnings.warn(
    "Importing from 'core' is deprecated. "
    "Use 'stateset_agents.core' instead. "
    "This shim will be removed in version 1.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = import_module("stateset_agents.core")
globals().update(_canonical.__dict__)
sys.modules[__name__] = _canonical
