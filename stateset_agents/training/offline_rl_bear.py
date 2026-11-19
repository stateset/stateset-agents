"""v0.74.0 compatibility alias for ``stateset_agents.experimental.offline_rl.bear``."""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stateset_agents.experimental.offline_rl.bear import *  # noqa: F403
else:
    from stateset_agents.experimental.offline_rl import bear as _implementation

    sys.modules[__name__] = _implementation
