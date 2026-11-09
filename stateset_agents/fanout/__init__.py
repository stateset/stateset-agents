"""Fan RL workloads out into independent, parallel runs.

One manifest expands into arms times seeds. Each run gets its own isolated
compute (a StateSet Sandbox, or a local process), its own budget, and its own
journal entry, so a crash, a preemption, or one bad seed stays contained. Results
come back to a run directory and roll up into a per-arm report.

See ``docs/RL_FANOUT.md``. Importing this package needs neither torch nor the
sandbox SDK; both load only when a run actually needs them.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FanoutManifest",
    "FanoutRunner",
    "FanoutSummary",
    "LocalBackend",
    "ManifestError",
    "SandboxBackend",
    "build_report",
]

_EXPORTS = {
    "FanoutManifest": "stateset_agents.fanout.manifest",
    "ManifestError": "stateset_agents.fanout.manifest",
    "FanoutRunner": "stateset_agents.fanout.orchestrator",
    "FanoutSummary": "stateset_agents.fanout.orchestrator",
    "LocalBackend": "stateset_agents.fanout.local_backend",
    "SandboxBackend": "stateset_agents.fanout.sandbox_backend",
    "build_report": "stateset_agents.fanout.report",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)
