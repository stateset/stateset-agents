"""Portable source identity for experiment protocols, without importing trainers.

The implementation moved to :mod:`stateset_agents.utils.provenance` so layers
below ``evaluation`` (``data``) can fingerprint their artifacts; this module
re-exports it for existing imports.
"""

from __future__ import annotations

from stateset_agents.utils.provenance import (
    package_implementation,
    validate_implementation,
)

__all__ = ["package_implementation", "validate_implementation"]
