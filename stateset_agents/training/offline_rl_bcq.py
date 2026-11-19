"""v0.74.0 compatibility alias for ``stateset_agents.experimental.offline_rl.bcq``."""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stateset_agents.experimental.offline_rl.bcq import *  # noqa: F403
else:
    from stateset_agents.experimental.offline_rl import bcq as _implementation

    sys.modules[__name__] = _implementation
