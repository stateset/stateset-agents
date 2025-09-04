"""
StateSet Agents

Canonical import namespace for the StateSet RL Agent Framework.

This shim proxies subpackages from the repository's top-level modules
to provide a stable `stateset_agents` namespace without immediately
moving files. A subsequent refactor can relocate code under this
package with minimal external changes.
"""

from importlib import import_module
import sys as _sys

# Optionally re-export public API from the root module if defined
try:
    _root = import_module('__init__')  # best-effort; may be absent or minimal
    for _name in getattr(_root, '__all__', []):
        globals()[_name] = getattr(_root, _name)
    __all__ = getattr(_root, '__all__', [])
except Exception:
    __all__ = []

# Proxy common subpackages so users can import like:
#   from stateset_agents.core import agent
#   from stateset_agents.training import trainer
for _subpkg in ("core", "training", "utils", "rewards", "api", "examples", "benchmarks"):
    try:
        _mod = import_module(_subpkg)
        globals()[_subpkg] = _mod
        _sys.modules[__name__ + f'.{_subpkg}'] = _mod
    except Exception:
        # Allow namespace even if optional subpackages are missing
        pass

# Basic metadata (fallback if root __init__ doesn't expose these)
try:
    from __init__ import __version__ as _v  # type: ignore
    __version__ = _v
except Exception:
    __version__ = "0.3.0"

