"""Backwards-compatible re-export of the packaged model preset registry.

The registry now lives in :mod:`stateset_agents.core.model_presets` so that
package code (``stateset_agents/mcp_server.py``, ``stateset_agents/cli_train.py``)
can import it without depending on the ``examples/`` directory. This module is
kept so existing scripts and docs that do ``from examples.model_presets import
...`` keep working.
"""

from __future__ import annotations

from stateset_agents.core.model_presets import (
    PRESETS,
    ModelPreset,
    get_preset,
    list_preset_names,
)

__all__ = ["ModelPreset", "PRESETS", "get_preset", "list_preset_names"]
