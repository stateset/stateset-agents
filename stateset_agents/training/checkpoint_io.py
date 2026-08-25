"""Backwards-compatible re-export of the checkpoint loader.

The implementation lives in :mod:`stateset_agents.core.checkpoint_io`.
"""

from __future__ import annotations

from ..core.checkpoint_io import load_checkpoint_file

__all__ = ["load_checkpoint_file"]
