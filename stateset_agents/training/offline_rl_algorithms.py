"""v0.74.0 compatibility alias for ``stateset_agents.experimental.offline_rl.algorithms``."""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stateset_agents.experimental.offline_rl.algorithms import *  # noqa: F403
else:
    from stateset_agents.experimental.offline_rl import algorithms as _implementation

    sys.modules[__name__] = _implementation
