"""Deprecated shim for ``stateset_agents.training.advanced_training_orchestrator``."""

from importlib import import_module
import sys

from stateset_agents.training.advanced_training_orchestrator import *  # noqa: F401, F403

# Preserve legacy patching hooks used by unit tests.
_orchestrator = import_module("stateset_agents.training.advanced_training_orchestrator")
psutil = getattr(_orchestrator, "psutil", None)

# Make legacy module name point to the real implementation so that
# monkeypatching training.advanced_training_orchestrator affects canonical code.
sys.modules[__name__] = _orchestrator
