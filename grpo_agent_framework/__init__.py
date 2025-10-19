# Shim package to expose the existing framework under the canonical package name
# This re-exports the public API from the top-level modules.

import sys as _sys
from importlib import import_module

# Re-export top-level API from root __init__ if available
try:
    # Import the root package (this repository root has an __init__.py)
    _root = import_module("__init__")
    # Copy public attributes
    for _name in getattr(_root, "__all__", []):
        globals()[_name] = getattr(_root, _name)
    __all__ = getattr(_root, "__all__", [])
except Exception:
    __all__ = []

# Provide subpackages by proxying to existing modules
for _subpkg in (
    "core",
    "training",
    "utils",
    "rewards",
    "api",
    "examples",
    "benchmarks",
):
    try:
        _mod = import_module("stateset_agents." + _subpkg)
        globals()[_subpkg] = _mod
        _sys.modules[__name__ + f".{_subpkg}"] = _mod
    except Exception:
        pass
