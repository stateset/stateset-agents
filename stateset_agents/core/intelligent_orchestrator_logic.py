"""Deprecated shim: moved to `stateset_agents.experimental.intelligent_orchestrator_logic`.

This module was relocated to the experimental namespace. Importing it from
`stateset_agents.core` is deprecated and will stop working in a future
release.
"""

import sys
import warnings

from stateset_agents.experimental import intelligent_orchestrator_logic as _new

warnings.warn(
    "stateset_agents.core.intelligent_orchestrator_logic has moved to "
    "stateset_agents.experimental.intelligent_orchestrator_logic; the old import path is "
    "deprecated and will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias the module so attribute access, monkeypatching, and `from ... import`
# via the old path all target the relocated module object.
sys.modules[__name__] = _new
