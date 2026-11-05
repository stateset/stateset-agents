"""Portable implementation revisions for native distributed training.

These identities bind source files, loaded Python method bodies and runtime
versions. They do not attest hardware, native library contents, arbitrary
mutable globals or external services. Use an immutable installation and restart
workers after changing code; source provenance alone cannot reload Python.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from .scheduler_identity import _code_digest

PROTOCOL = "stateset-native-implementation-v1"


def _package_revision(root: str) -> str:
    files = {}

    def visit(directory: str, prefix: str) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                name = prefix + entry.name
                if entry.is_dir(follow_symlinks=False):
                    visit(entry.path, name + "/")
                elif entry.name.endswith(".py"):
                    with open(entry.path, "rb") as stream:
                        files[name] = hashlib.sha256(stream.read()).hexdigest()

    visit(root, "")
    if "__init__.py" not in files:
        raise ValueError("Native implementation identity requires package sources")
    return _digest(
        {"schema_version": 1, "scope": "stateset_agents_python_sources", "files": files}
    )


def package_revision() -> str:
    """Read and hash package sources afresh, without trusting file timestamps."""
    root = str(Path(__file__).resolve().parents[1])
    return _package_revision(root)


def _source(value: Any) -> str | None:
    try:
        filename = inspect.getsourcefile(value)
    except (TypeError, OSError):
        filename = None
    if filename is None:
        return None
    with open(filename, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def class_implementation(cls: type) -> str:
    """Bind class ancestry, source revisions and loaded Python method code."""
    parents = []
    for parent in cls.__mro__:
        if parent is object:
            continue
        methods = {}
        for name, member in sorted(vars(parent).items()):
            functions = (
                [member.fget, member.fset, member.fdel]
                if isinstance(member, property)
                else [
                    (
                        member.__func__
                        if isinstance(member, (staticmethod, classmethod))
                        else member
                    )
                ]
            )
            codes = [
                (
                    _code_digest(function.__code__)
                    if inspect.isfunction(function)
                    else None
                )
                for function in functions
            ]
            if any(codes):
                methods[name] = codes
        source = _source(parent)
        if source is None and parent.__module__ != "builtins":
            raise ValueError("Native implementation identity requires class sources")
        parents.append(
            {
                "class": f"{parent.__module__}.{parent.__qualname__}",
                "source": source,
                "methods": methods,
            }
        )
    return _digest(parents)


def model_implementation(model: Any) -> str | None:
    """Bind every submodule implementation without reading tensor values."""
    if model is None:
        return None
    classes: dict[type, str] = {}
    modules = []
    for name, module in model.named_modules(remove_duplicate=False):
        cls = type(module)
        if cls not in classes:
            classes[cls] = class_implementation(cls)
        overrides = {}
        # Instance methods can shadow a class implementation (including a
        # compiled forward). Bind Python overrides; reject opaque substitutes.
        for field in ("forward", "_call_impl", "_compiled_call_impl"):
            function = vars(module).get(field)
            if function is None:
                continue
            if inspect.ismethod(function):
                function = function.__func__
            if not inspect.isfunction(function):
                raise ValueError("Opaque native model method overrides are unsupported")
            overrides[field] = _code_digest(function.__code__)
        modules.append([name, classes[cls], overrides])
    return _digest(modules)


@lru_cache(maxsize=1)
def _installed_versions() -> tuple[tuple[str, str | None], ...]:
    # Distribution metadata describes the process installation, not loaded code.
    # Changing dependencies requires a worker restart. Loaded version strings
    # and numerical controls are still inspected at every admission.
    versions: dict[str, str | None] = {}
    for name in (
        "numpy",
        "transformers",
        "tokenizers",
        "peft",
        "accelerate",
        "safetensors",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return tuple(versions.items())


def runtime_implementation() -> dict[str, Any]:
    """Describe numerical runtime versions and process-wide precision controls."""
    import torch

    versions = dict(_installed_versions())
    for name in versions:
        loaded = getattr(sys.modules.get(name), "__version__", None)
        if loaded is not None:
            versions[name] = str(loaded)
    return {
        "python": [sys.implementation.name, *sys.version_info[:3]],
        "torch": str(torch.__version__),
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "dependencies": versions,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_version": torch.backends.cudnn.version(),
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }


def learner_implementation(trainer: Any) -> dict[str, Any]:
    """Bind package and learner implementation revisions at each admission."""
    return {
        "protocol": PROTOCOL,
        "package_sha256": package_revision(),
        "runtime": runtime_implementation(),
        "components": {
            name: None if value is None else class_implementation(type(value))
            for name in (
                "trainer",
                "optimizer",
                "lr_scheduler",
                "scaler",
                "continual_manager",
            )
            for value in [
                trainer if name == "trainer" else getattr(trainer, name, None)
            ]
        },
    }
