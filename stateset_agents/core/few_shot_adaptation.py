"""Deprecated shim: moved to `stateset_agents.experimental.few_shot_adaptation`.

This module was relocated to the experimental namespace. Importing it from
`stateset_agents.core` has been deprecated since v0.22.0 and is removed in
v0.71.0.
"""

import sys
import warnings

from stateset_agents.experimental import few_shot_adaptation as _new

warnings.warn(
    "stateset_agents.core.few_shot_adaptation has moved to "
    "stateset_agents.experimental.few_shot_adaptation; the old import path is "
    "deprecated and will be removed in v0.71.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias the module so attribute access, monkeypatching, and `from ... import`
# via the old path all target the relocated module object.
sys.modules[__name__] = _new
