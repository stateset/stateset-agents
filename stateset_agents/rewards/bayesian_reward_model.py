"""v0.74.0 compatibility alias for ``stateset_agents.experimental.bayesian_reward_model``."""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stateset_agents.experimental.bayesian_reward_model import *  # noqa: F403
else:
    from stateset_agents.experimental import bayesian_reward_model as _implementation

    sys.modules[__name__] = _implementation
