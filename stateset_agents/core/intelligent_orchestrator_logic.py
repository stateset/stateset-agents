"""Deprecated shim: moved to `stateset_agents.experimental.intelligent_orchestrator_logic`.

This module was relocated to the experimental namespace. Importing it from
`stateset_agents.core` has been deprecated since v0.22.0 and is removed in
v0.71.0.
"""

import sys

from stateset_agents._deprecation import warn_deprecated
from stateset_agents.experimental import intelligent_orchestrator_logic as _new

warn_deprecated(
    "stateset_agents.core.intelligent_orchestrator_logic",
    removal="0.71.0",
    replacement="stateset_agents.experimental.intelligent_orchestrator_logic",
)

# Alias the module so attribute access, monkeypatching, and `from ... import`
# via the old path all target the relocated module object.
sys.modules[__name__] = _new
