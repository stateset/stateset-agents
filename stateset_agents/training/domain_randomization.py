"""v0.74.0 compatibility alias for ``stateset_agents.experimental.sim_to_real.domain_randomization``."""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stateset_agents.experimental.sim_to_real.domain_randomization import *  # noqa: F403
else:
    from stateset_agents.experimental.sim_to_real import (
        domain_randomization as _implementation,
    )

    sys.modules[__name__] = _implementation
