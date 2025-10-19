"""
Proxy module for `stateset_agents.training`.

Forwards imports to the top-level `training` package shipped with the
distribution so callers can use `stateset_agents.training.*` while we
keep a single source of truth for implementation.
"""

import sys as _sys
from importlib import import_module as _import_module
from pathlib import Path as _Path

# Ensure repository root is preferred on sys.path
try:
    _root_dir = _Path(__file__).resolve().parents[2]
    _root_str = str(_root_dir)
    if _root_str not in _sys.path:
        _sys.path.insert(0, _root_str)
except Exception:
    pass

# Load underlying top-level package
_training_pkg = _import_module("training")

# Re-export common submodules for dotted imports
_submodules = (
    "train",
    "config",
    "trainer",
    "neural_reward_trainer",
    "distributed",
    "distributed_trainer",
    "trl_grpo_trainer",
    "diagnostics",
    "advanced_training_orchestrator",
)

for _name in _submodules:
    try:
        _mod = _import_module(f"training.{_name}")
        _sys.modules[__name__ + f".{_name}"] = _mod
    except Exception:
        # Allow import even if optional submodule missing
        pass

# Alias this package to underlying one for attribute access
_sys.modules[__name__] = _training_pkg
