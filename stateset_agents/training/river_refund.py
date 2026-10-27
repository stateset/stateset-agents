"""Deprecated shim: moved to `stateset_agents.verticals.river_refund`.

The River-native refund experiment drives the remote River runtime, so it
cannot live in ``training``, a layer below ``remote`` (docs/ARCHITECTURE.md,
Layering). Importing it, or running ``python -m`` on it, from
`stateset_agents.training.river_refund` is deprecated since v0.70.0 and is
removed in v0.71.0.
"""

import sys

from stateset_agents._deprecation import warn_deprecated

warn_deprecated(
    "stateset_agents.training.river_refund",
    removal="0.71.0",
    replacement="stateset_agents.verticals.river_refund",
)

if __name__ == "__main__":
    # `python -m stateset_agents.training.river_refund ...` keeps working.
    import runpy

    runpy.run_module(
        "stateset_agents.verticals.river_refund", run_name="__main__", alter_sys=True
    )
else:
    from stateset_agents.verticals import river_refund as _new

    # Alias the module so attribute access, monkeypatching, and `from ...
    # import` via the old path all target the relocated module object.
    sys.modules[__name__] = _new
