"""v0.74.0 compatibility alias for ``stateset_agents.experimental.sim_to_real.metrics``."""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stateset_agents.experimental.sim_to_real.metrics import *  # noqa: F403
else:
    from stateset_agents.experimental.sim_to_real import metrics as _implementation

    sys.modules[__name__] = _implementation
