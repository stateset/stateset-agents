"""Deprecated shim for ``stateset_agents.training.trainer``."""

from stateset_agents.training.trainer import *  # noqa: F401, F403

# Preserve legacy patching hooks (tests patch training.trainer.torch/np).
import stateset_agents.training.trainer as _trainer  # noqa: E402

torch = getattr(_trainer, "torch", None)
np = getattr(_trainer, "np", None)

