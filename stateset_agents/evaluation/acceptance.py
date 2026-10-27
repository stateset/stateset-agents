"""Deprecated shim: moved to `stateset_agents.verticals.acceptance`.

The A+ acceptance audit revalidates training artifacts (checkpoint publication,
the native refund study), so it cannot live in ``evaluation``, a layer below
training (docs/ARCHITECTURE.md, Layering). Importing it from
`stateset_agents.evaluation.acceptance` is deprecated since v0.70.0 and is
removed in v0.71.0.
"""

import sys

from stateset_agents._deprecation import warn_deprecated
from stateset_agents.verticals import acceptance as _new

warn_deprecated(
    "stateset_agents.evaluation.acceptance",
    removal="0.71.0",
    replacement="stateset_agents.verticals.acceptance",
)

# Alias the module so attribute access, monkeypatching, and `from ... import`
# via the old path all target the relocated module object.
sys.modules[__name__] = _new
