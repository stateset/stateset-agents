"""
Deprecated shim for backwards compatibility.

The canonical package is ``stateset_agents.api``.
"""

from importlib import import_module
import sys

_canonical = import_module("stateset_agents.api")
globals().update(_canonical.__dict__)
sys.modules[__name__] = _canonical
