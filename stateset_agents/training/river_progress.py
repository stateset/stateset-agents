"""Deprecated shim: moved to `stateset_agents.evaluation.river_progress`.

River training-progress records are validated by evaluation's checkpoint
selection and trace audits, so they live in ``evaluation``, below the studies
that write them (docs/ARCHITECTURE.md, Layering). Importing them from
`stateset_agents.training.river_progress` is deprecated since v0.70.0 and is
removed in v0.71.0.
"""

import sys

from stateset_agents._deprecation import warn_deprecated
from stateset_agents.evaluation import river_progress as _new

warn_deprecated(
    "stateset_agents.training.river_progress",
    removal="0.71.0",
    replacement="stateset_agents.evaluation.river_progress",
)

# Alias the module so attribute access, monkeypatching, and `from ... import`
# via the old path all target the relocated module object.
sys.modules[__name__] = _new
