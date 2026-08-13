"""Provenance for trained adapters: what made this thing, and from what.

A LoRA adapter directory is otherwise anonymous — a few hundred megabytes of
tensors with no record of which base model it modifies, which data taught it,
or which earlier adapter's conversations produced that data. That matters
most exactly when it is hardest to reconstruct: months later, deciding
whether the adapter in production is the one an eval blessed.

Every training run writes ``stateset_manifest.json`` beside its adapter.
``stateset-agents adapters`` reads them back and reconstructs the family
tree: generation 2 knows it came from generation 1 when the run was told so
(``--parent-adapter``), which is how the improvement loop's generations stay
distinguishable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Filename written into every adapter directory.
MANIFEST_NAME = "stateset_manifest.json"


@dataclass
class AdapterManifest:
    """What produced one adapter."""

    base_model: str
    #: Absolute or user-supplied path of the training dataset, plus a content
    #: hash — the path alone is not evidence, the same name is reused freely.
    dataset_path: str | None = None
    dataset_sha256: str | None = None
    dataset_rows: int | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    #: Adapter this one descends from (the flywheel's previous generation),
    #: as given by the caller. None for a first generation.
    parent_adapter: str | None = None
    #: Summary of the run's own eval, when it ran one.
    eval_passed: int | None = None
    eval_total: int | None = None
    package_version: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hash_dataset(path: Path) -> tuple[str | None, int | None]:
    """Return (sha256, row count) for a JSONL dataset, or (None, None).

    Hashing the content rather than trusting the filename is the point: two
    runs that claim the same dataset are only comparable if the bytes match.
    """
    try:
        data = Path(path).read_bytes()
    except OSError:
        return (None, None)
    digest = hashlib.sha256(data).hexdigest()
    rows = sum(
        1 for line in data.decode("utf-8", "replace").splitlines() if line.strip()
    )
    return (digest, rows)


def write_manifest(output_dir: Path, manifest: AdapterManifest) -> Path:
    """Write ``manifest`` into ``output_dir``.

    Never raises: provenance is valuable but an adapter that trained
    successfully must not be reported as a failure because bookkeeping
    could not be written.
    """
    target = Path(output_dir) / MANIFEST_NAME
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:  # pragma: no cover - defensive
        pass
    return target


def read_manifest(adapter_dir: Path) -> dict[str, Any] | None:
    """Read one adapter's manifest, or None when it has none."""
    path = Path(adapter_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def discover_adapters(root: Path) -> list[dict[str, Any]]:
    """Find every adapter under ``root``, manifest or not.

    Adapters predating manifests still show up (that is the point of an
    audit) — they simply carry no provenance.
    """
    found: list[dict[str, Any]] = []
    root = Path(root)
    if not root.exists():
        return found
    for config in sorted(root.rglob("adapter_config.json")):
        adapter_dir = config.parent
        entry: dict[str, Any] = {
            "path": str(adapter_dir),
            "name": adapter_dir.name,
            "manifest": read_manifest(adapter_dir),
        }
        eval_path = adapter_dir / "eval_results.json"
        entry["has_eval"] = eval_path.exists()
        found.append(entry)
    return found


def build_lineage(adapters: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map parent adapter path -> child adapter paths.

    Parents are matched on the recorded string first and on directory name
    second, so a manifest written on a pod (absolute remote path) still links
    up after the adapter is fetched to a different local directory.
    """
    by_name = {Path(a["path"]).name: a["path"] for a in adapters}
    children: dict[str, list[str]] = {}
    for adapter in adapters:
        manifest = adapter.get("manifest") or {}
        parent = manifest.get("parent_adapter")
        if not parent:
            continue
        resolved = (
            parent
            if parent in {a["path"] for a in adapters}
            else by_name.get(Path(str(parent)).name, str(parent))
        )
        children.setdefault(resolved, []).append(adapter["path"])
    return children


__all__ = [
    "MANIFEST_NAME",
    "AdapterManifest",
    "build_lineage",
    "discover_adapters",
    "hash_dataset",
    "read_manifest",
    "write_manifest",
]
