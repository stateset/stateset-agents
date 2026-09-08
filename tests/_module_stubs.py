"""Import-time module stubs that stay scoped to the test file that installed them.

Several test modules replace optional third-party packages in ``sys.modules``
at import time (a module-object ``vllm`` stub to keep torchvision out of the
import path; ``MagicMock`` stand-ins for ``trl``/``peft``/``vllm`` to test a
trainer without them). Installing them directly leaks the fakes into every
other test collected in the same process. ``install_import_stub`` records the
owning test file so the autouse fixture in ``tests/conftest.py`` can hide each
stub while any *other* test file's tests run, and put it back afterwards.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

#: name -> (stub, owner test file)
STUBS: dict[str, tuple[Any, Path]] = {}


def install_import_stub(name: str, module: Any, *, owner: str | None = None) -> Any:
    """Install ``module`` as ``sys.modules[name]`` for the calling test file.

    Returns the stub so callers can keep using it. ``owner`` defaults to the
    caller's file.
    """
    if owner is None:
        frame = inspect.stack()[1]
        owner = frame.filename
    STUBS[name] = (module, Path(owner).resolve())
    sys.modules[name] = module
    return module


def make_module_stub(name: str, **attrs: Any) -> ModuleType:
    """A plain module object with ``__spec__`` and the given attributes."""
    import importlib.machinery

    stub = ModuleType(name)
    stub.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in attrs.items():
        setattr(stub, key, value)
    return stub


def hide_foreign_stubs(test_file: Path) -> dict[str, Any]:
    """Remove stubs owned by other test files; return what was removed."""
    hidden: dict[str, Any] = {}
    test_file = Path(test_file).resolve()
    for name, (stub, owner) in STUBS.items():
        if owner == test_file:
            continue
        for key in list(sys.modules):
            if (key == name or key.startswith(name + ".")) and (
                sys.modules[key] is stub or key != name
            ):
                if sys.modules.get(key) is stub or key.startswith(name + "."):
                    hidden[key] = sys.modules.pop(key)
    return hidden


def restore_stubs(hidden: dict[str, Any]) -> None:
    for key, module in hidden.items():
        sys.modules[key] = module
