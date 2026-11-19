"""Curated public synthetic evidence; never discover or export private workspaces."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any


def evidence_catalog(*, repository: Path | None = None) -> dict[str, Any]:
    """Read the packaged catalog; optionally verify its exact archived source bytes.

    Catalog entries summarize historical runs, not the current implementation's
    capabilities. Audit success and experiment acceptance are separate fields.
    No private directory scanning or customer-data export is performed.
    """
    data = json.loads(
        files("stateset_agents.improvement")
        .joinpath("evidence-catalog.json")
        .read_text()
    )
    if repository is not None:
        root = repository.resolve()
        for entry in data["entries"]:
            for relative, expected in entry["artifact_sha256"].items():
                path = root / relative
                if (
                    path.is_symlink()
                    or not path.resolve().is_relative_to(root)
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected
                ):
                    raise ValueError(
                        "Historical evidence artifact changed or is missing"
                    )
    return dict(data)
