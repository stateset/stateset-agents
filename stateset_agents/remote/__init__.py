"""Remote execution of the fine-tune step on rented compute.

Closes the last gap in the improvement loop: ``ingest`` and ``improve`` are
CPU-only and cheap, but the SFT that consumes ``curated.jsonl`` needs a GPU.
This package runs that job — unchanged — somewhere else.

See ``docs/superpowers/specs/2026-07-30-remote-executor-design.md``.
"""

from __future__ import annotations

from stateset_agents.remote.deployment import (
    DeploymentHandle,
    DeploymentSpec,
    InferenceDeploymentProvider,
)
from stateset_agents.remote.job import (
    JobHandle,
    JobStatus,
    RemoteJobResult,
    RemoteJobSpec,
)

__all__ = [
    "DeploymentHandle",
    "DeploymentSpec",
    "InferenceDeploymentProvider",
    "JobHandle",
    "JobStatus",
    "RemoteJobResult",
    "RemoteJobSpec",
]
