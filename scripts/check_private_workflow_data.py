#!/usr/bin/env python3
"""Reject tracked JSON artifacts explicitly labeled as customer/private workflow data.

This is a narrow accidental-commit guard, not a PII detector. Customer workspaces
must stay outside Git; passing this check does not authorize publishing data.
Only paths are reported, never conversation contents.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def contains_private_marker(value: Any) -> bool:
    """Find explicit workflow/customer markers without reporting their payload."""
    if isinstance(value, dict):
        if (
            value.get("source_kind") == "customer"
            or value.get("kind") == "stateset-private-conversation-dataset"
        ):
            return True
        return any(contains_private_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_private_marker(item) for item in value)
    return False


def check_paths(root: Path, paths: list[str]) -> list[str]:
    """Check tracked JSON/JSONL inputs; reject unreadable or oversized artifacts."""
    rejected = []
    for relative in paths:
        path = root / relative
        if path.suffix not in (".json", ".jsonl") or not path.exists():
            continue
        if path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
            rejected.append(relative)
            continue
        try:
            text = path.read_text()
            records = (
                [json.loads(line) for line in text.splitlines() if line.strip()]
                if path.suffix == ".jsonl"
                else [json.loads(text)]
            )
            if any(contains_private_marker(record) for record in records):
                rejected.append(relative)
        except (OSError, ValueError, UnicodeError):
            rejected.append(relative)
    return rejected


def main() -> int:
    """Audit the tracked public tree, leaving all private workspaces untouched."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    )
    rejected = check_paths(root, result.stdout.decode().strip("\0").split("\0"))
    if rejected:
        print("Private-data or unreadable JSON artifacts cannot be published:")
        for path in rejected:
            print(path)
        return 1
    print(
        "Tracked JSON artifacts contain no explicit customer/private-workflow markers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
