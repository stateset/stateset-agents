"""Portable source identity for experiment protocols, without importing trainers."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any


def package_implementation() -> dict[str, Any]:
    """Fingerprint installed package Python sources, including uncommitted edits.

    Paths are relative to the package, so a checkout and an unpacked wheel with
    identical sources have the same identity. All package sources are included
    conservatively to cover indirect scoring and training dependencies. This is
    local file provenance, not dependency locking or provider attestation. Run
    experiments from an immutable installation and restart after editing code.
    """
    root = Path(__file__).resolve().parents[1]
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    identity = {
        "schema_version": 1,
        "scope": "stateset_agents_python_sources",
        "files": files,
    }
    validate_implementation(identity)
    return identity


def validate_implementation(identity: Any) -> None:
    """Reject malformed source identities without consulting current code."""
    if (
        not isinstance(identity, dict)
        or set(identity) != {"schema_version", "scope", "files"}
        or type(identity["schema_version"]) is not int
        or identity["schema_version"] != 1
        or identity["scope"] != "stateset_agents_python_sources"
    ):
        raise ValueError("Invalid implementation identity")
    files = identity["files"]
    if not isinstance(files, dict) or not files or "__init__.py" not in files:
        raise ValueError("Implementation identity requires package source hashes")
    for name, digest in files.items():
        if (
            not isinstance(name, str)
            or not name.endswith(".py")
            or "\\" in name
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or PurePosixPath(name).as_posix() != name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(
                "Implementation identity requires relative .py paths and SHA-256 hashes"
            )
