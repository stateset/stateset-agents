"""Shipped modules must not reach into a developer's home directory."""

from __future__ import annotations

import re
from pathlib import Path

from stateset_agents import finetune_workspace as workspace

PACKAGE = Path(workspace.__file__).resolve().parent
HOME_PATH = re.compile(r"(?<![\w.~])/(?:home|Users)/[A-Za-z0-9_.-]+/")


def test_no_packaged_module_contains_an_absolute_home_path():
    offenders = [
        f"{path.relative_to(PACKAGE.parent)}:{number}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if HOME_PATH.search(line)
    ]
    assert not offenders, f"Developer-specific absolute paths: {offenders}"


def test_queue_is_only_found_through_configuration_or_path(monkeypatch):
    """A fixed fallback would run another user's binary with this environment."""
    monkeypatch.delenv("STATESET_BUILDR", raising=False)
    monkeypatch.setattr(workspace.shutil, "which", lambda name: None)
    monkeypatch.setattr(workspace.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(workspace.os, "access", lambda path, mode: True)
    assert workspace.queue_executable() is None

    monkeypatch.setattr(workspace.shutil, "which", lambda name: "/opt/bin/" + name)
    assert workspace.queue_executable() == "/opt/bin/buildr"

    monkeypatch.setenv("STATESET_BUILDR", "/srv/queue/buildr")
    monkeypatch.setattr(workspace.Path, "is_file", lambda self: True)
    assert workspace.queue_executable() == "/srv/queue/buildr"
