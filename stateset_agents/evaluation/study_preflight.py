"""Deprecated shim: moved to `stateset_agents.verticals.study_preflight`.

The River refund study preflight checks the remote River runtime and the study
plan, so it cannot live in ``evaluation``, a layer below both
(docs/ARCHITECTURE.md, Layering). Importing it from
`stateset_agents.evaluation.study_preflight` is deprecated since v0.70.0 and is
removed in v0.71.0.
"""

import sys

from stateset_agents._deprecation import warn_deprecated
from stateset_agents.verticals import study_preflight as _new

warn_deprecated(
    "stateset_agents.evaluation.study_preflight",
    removal="0.71.0",
    replacement="stateset_agents.verticals.study_preflight",
)

# Alias the module so attribute access, monkeypatching, and `from ... import`
# via the old path all target the relocated module object.
sys.modules[__name__] = _new
