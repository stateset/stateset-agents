"""Deprecated shim: moved to ``stateset_agents.experimental.sim_to_real.domain_randomization``.

``stateset_agents.training.domain_randomization`` was relocated to the
experimental namespace in v0.64.0. This alias keeps the old import path
working, with a ``DeprecationWarning``, until it is removed in v0.65.0.
"""

import importlib
import sys
import warnings

_OLD_MODULE = "stateset_agents.training.domain_randomization"
_NEW_MODULE = "stateset_agents.experimental.sim_to_real.domain_randomization"

warnings.warn(
    f"{_OLD_MODULE} has moved to {_NEW_MODULE}; the old import path is "
    "deprecated and will be removed in v0.65.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias the module so attribute access, monkeypatching, and ``from ... import``
# via the old path all target the relocated module object.
sys.modules[__name__] = importlib.import_module(_NEW_MODULE)
