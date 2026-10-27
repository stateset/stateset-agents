"""Portable identities for experiment artifacts and the code that made them.

* ``content_hash`` -- SHA-256 of canonical JSON, rejecting non-finite numbers;
  ``json_sha256`` / ``canonical_json`` are the shared JSON-digest
  implementation (with per-call-site legacy settings);
* ``package_implementation`` / ``validate_implementation`` -- a fingerprint of
  the installed ``stateset_agents`` Python sources, without importing trainers.

Every layer that writes or checks an artifact needs these, so they live at the
bottom of the stack. They moved here from ``stateset_agents.evaluation``
(``agent_runs.content_hash`` and ``evaluation.implementation``), which keep
re-exporting them, so ``data`` no longer depends on ``evaluation``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

__all__ = [
    "canonical_json",
    "content_hash",
    "json_sha256",
    "package_implementation",
    "validate_implementation",
]


def canonical_json(
    value: Any,
    *,
    sort_keys: bool = True,
    compact: bool = True,
    allow_nan: bool = False,
    ensure_ascii: bool = True,
) -> str:
    """Serialize ``value`` for hashing; the defaults are the canonical form.

    ``compact`` selects ``(",", ":")`` separators; ``compact=False`` keeps
    ``json.dumps``'s default ``", "``/``": "``. The other flags are passed to
    ``json.dumps`` unchanged.
    """
    return json.dumps(
        value,
        sort_keys=sort_keys,
        separators=(",", ":") if compact else None,
        allow_nan=allow_nan,
        ensure_ascii=ensure_ascii,
    )


def json_sha256(
    value: Any,
    *,
    sort_keys: bool = True,
    compact: bool = True,
    allow_nan: bool = False,
    ensure_ascii: bool = True,
) -> str:
    """SHA-256 hex digest of :func:`canonical_json` encoded as UTF-8.

    New code should use the defaults (:func:`content_hash`). The keyword
    settings reproduce the serializations older identities were hashed with
    -- manifests, dataset and case fingerprints, experiment digests -- so a
    call site's stored hashes stay byte-identical; never change a call site's
    settings once its hashes are persisted or compared.
    ``tests/unit/test_content_hashing.py`` pins every setting in use.
    """
    data = canonical_json(
        value,
        sort_keys=sort_keys,
        compact=compact,
        allow_nan=allow_nan,
        ensure_ascii=ensure_ascii,
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def content_hash(value: Any) -> str:
    """Hash canonical JSON (sorted keys, compact), rejecting nonfinite numbers."""
    return json_sha256(value)


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
