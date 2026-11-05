"""Portable identities for scheduler implementations and callable bindings.

No callable is executed. Paths and line numbers are excluded from Python code
identity, so identical checkouts can live at different locations. Arbitrary
callable equivalence and mutable external services are outside this contract.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import sys
import types
from pathlib import Path
from typing import Any


def _kind(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


@functools.lru_cache(maxsize=256)
def _file_digest(path: str, size: int, modified: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(value: Any) -> str | None:
    try:
        filename = inspect.getsourcefile(value)
    except (TypeError, OSError):
        filename = None
    if filename is None:
        return None
    path = Path(filename)
    stat = path.stat()
    return _file_digest(str(path), stat.st_size, stat.st_mtime_ns)


def callable_identity(value: Any) -> Any:
    """Describe Python callable code, defaults, closures and referenced globals.

    Supports functions, partials, bound methods and callable objects whose
    bindings contain plain data. Unsupported opaque bindings fail closed.
    """
    seen: dict[int, int] = {}
    active: set[int] = set()

    def describe(item: Any) -> Any:
        if item is None or type(item) in (bool, int, str):
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("Scheduler bindings must be finite")
            return {"float": item.hex()}
        if isinstance(item, bytes):
            return {"bytes": item.hex()}
        if item is Ellipsis:
            return {"ellipsis": True}
        if isinstance(item, complex):
            return {"complex": [item.real.hex(), item.imag.hex()]}
        if isinstance(item, types.CodeType):
            return {
                "code": item.co_code.hex(),
                "constants": [describe(child) for child in item.co_consts],
                "names": list(item.co_names),
                "variables": list(item.co_varnames),
                "free": list(item.co_freevars),
                "cells": list(item.co_cellvars),
                "arguments": [
                    item.co_argcount,
                    item.co_posonlyargcount,
                    item.co_kwonlyargcount,
                ],
                "flags": item.co_flags,
                "exceptions": getattr(item, "co_exceptiontable", b"").hex(),
            }
        if isinstance(item, (list, tuple, dict, frozenset)):
            if id(item) in active:
                raise ValueError("Cyclic scheduler bindings are unsupported")
            active.add(id(item))
            try:
                if isinstance(item, (list, tuple)):
                    return {
                        "list" if isinstance(item, list) else "tuple": [
                            describe(v) for v in item
                        ]
                    }
                if isinstance(item, dict):
                    if any(type(key) is not str for key in item):
                        raise ValueError(
                            "Scheduler binding dictionaries require string keys"
                        )
                    return {
                        "mapping": {key: describe(item[key]) for key in sorted(item)}
                    }
                parts = [describe(v) for v in item]
                return {
                    "frozenset": sorted(
                        parts, key=lambda v: json.dumps(v, sort_keys=True)
                    )
                }
            finally:
                active.remove(id(item))
        if isinstance(item, types.ModuleType):
            return {
                "module": item.__name__,
                "version": getattr(item, "__version__", None),
                "source": _source(item),
            }
        if inspect.isbuiltin(item):
            return {
                "builtin": [item.__module__, item.__qualname__],
                "python": list(sys.version_info[:3]),
            }
        if isinstance(item, type):
            return {"class": _kind(item), "source": _source(item)}
        if id(item) in seen:
            return {"reference": seen[id(item)]}
        seen[id(item)] = len(seen)
        if isinstance(item, functools.partial):
            return {
                "partial": describe(item.func),
                "args": describe(item.args),
                "keywords": describe(item.keywords),
                "attributes": describe(vars(item)),
            }
        if inspect.ismethod(item):
            return {
                "method": describe(item.__func__),
                "owner": describe(vars(item.__self__)),
                "type": _kind(item.__self__),
                "class_state": class_state(type(item.__self__)),
            }
        if inspect.isfunction(item):
            closure = [
                describe(cell.cell_contents) for cell in (item.__closure__ or ())
            ]
            globals_used = inspect.getclosurevars(item).globals
            return {
                "function": _code_digest(item.__code__),
                "defaults": describe(item.__defaults__),
                "keyword_defaults": describe(item.__kwdefaults__),
                "closure": closure,
                "globals": {
                    name: describe(globals_used[name]) for name in sorted(globals_used)
                },
            }
        if callable(item) and hasattr(item, "__dict__"):
            method = type(item).__call__
            if not inspect.isfunction(method):
                raise ValueError("Opaque native scheduler callables are unsupported")
            return {
                "callable": _kind(item),
                "code": describe(method),
                "state": describe(vars(item)),
                "class_state": class_state(type(item)),
            }
        raise ValueError("Unsupported scheduler callable binding")

    def class_state(cls: type) -> dict[str, Any]:
        result = {}
        for parent in cls.__mro__:
            for name, binding in sorted(vars(parent).items()):
                if (
                    name.startswith("__")
                    or callable(binding)
                    or isinstance(binding, (property, staticmethod, classmethod))
                ):
                    continue
                result[_kind(parent) + "." + name] = describe(binding)
        return result

    return describe(value)


@functools.lru_cache(maxsize=1024)
def _code_digest(code: types.CodeType) -> str:
    encoded = json.dumps(
        callable_identity(code), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def scheduler_identity(scheduler: Any) -> dict[str, Any] | None:
    """Bind a scheduler's class implementations and supported callable fields.

    Scheduler scalar state is compared separately. Composite schedulers recurse;
    Lambdas/partials retain code and bindings absent from PyTorch state_dict().
    """
    if scheduler is None:
        return None
    implementations = []
    for cls in type(scheduler).__mro__:
        if cls is object:
            continue
        methods = {}
        for name, method in sorted(vars(cls).items()):
            function = (
                method.__func__
                if isinstance(method, (staticmethod, classmethod))
                else method
            )
            if inspect.isfunction(function):
                # Class code is bound without walking optimizer/trainer globals.
                methods[name] = _code_digest(function.__code__)
        implementations.append(
            {"class": _kind(cls), "source": _source(cls), "methods": methods}
        )
    fields = {}
    for name, value in sorted(vars(scheduler).items()):
        if name in {"optimizer", "_schedulers", "after_scheduler"}:
            continue
        if callable(value) or (
            isinstance(value, (tuple, list)) and any(callable(v) for v in value)
        ):
            fields[name] = callable_identity(value)
    children = [
        scheduler_identity(child) for child in getattr(scheduler, "_schedulers", ())
    ]
    after = getattr(scheduler, "after_scheduler", None)
    return {
        "python": list(sys.version_info[:3]),
        "implementations": implementations,
        "callables": fields,
        "children": children,
        "after": scheduler_identity(after) if after is not None else None,
    }
