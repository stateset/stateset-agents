"""v0.74.0 compatibility alias for ``stateset_agents.experimental.ppo``."""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stateset_agents.experimental.ppo import *  # noqa: F403
else:
    from stateset_agents.experimental import ppo as _implementation

    sys.modules[__name__] = _implementation
