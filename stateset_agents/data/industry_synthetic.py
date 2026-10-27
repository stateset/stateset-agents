"""Deprecated shim: moved to `stateset_agents.verticals.industry_synthetic`.

Synthetic industry datasets are built from the ``training.industry`` recipes,
so they cannot live in ``data``, a layer below training (docs/ARCHITECTURE.md,
Layering). Importing it from `stateset_agents.data.industry_synthetic` is
deprecated since v0.70.0 and is removed in v0.71.0.
"""

import sys

from stateset_agents._deprecation import warn_deprecated
from stateset_agents.verticals import industry_synthetic as _new

warn_deprecated(
    "stateset_agents.data.industry_synthetic",
    removal="0.71.0",
    replacement="stateset_agents.verticals.industry_synthetic",
)

# Alias the module so attribute access, monkeypatching, and `from ... import`
# via the old path all target the relocated module object.
sys.modules[__name__] = _new
