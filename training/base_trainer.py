"""Deprecated shim for ``stateset_agents.training.base_trainer``."""

from importlib import import_module
import sys

from stateset_agents.training.base_trainer import *  # noqa: F401, F403

_base_trainer = import_module("stateset_agents.training.base_trainer")
sys.modules[__name__] = _base_trainer
