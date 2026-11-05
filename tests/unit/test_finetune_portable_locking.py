"""Studio file locking is portable and independent of faked platform checks."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateset_agents import finetune_workspace as workspace

PACKAGE = Path(workspace.__file__).resolve().parent


@pytest.fixture
def run(tmp_path):
    source = tmp_path / "chat.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        )
        + "\n"
    )
    config = workspace.FinetuneConfig(dataset=str(source))
    return workspace.create_run(tmp_path / "workspace", workspace.plan_run(config))


@pytest.mark.parametrize("faked", ["linux", "win32", "darwin"])
def test_state_transitions_lock_with_the_real_platform_primitive(
    run, monkeypatch, faked
):
    """Windows CI failed when a test faked ``sys.platform = "linux"``."""
    monkeypatch.setattr(workspace, "sys", SimpleNamespace(platform=faked))
    for module in ("fcntl", "msvcrt"):
        monkeypatch.setitem(sys.modules, module, None)  # no call-time imports
    assert workspace.transition_run(run, status="queued") is True
    assert workspace.read_json(run / "state.json")["status"] == "queued"


def test_private_lock_reports_a_busy_holder_with_the_callers_message(tmp_path):
    path = tmp_path / ".submit.lock"
    with workspace.private_file_lock(path, timeout_s=1, unavailable="x", busy="b"):
        with pytest.raises(ValueError, match="^busy-now$"):
            with workspace.private_file_lock(
                path, timeout_s=0, unavailable="x", busy="busy-now"
            ):
                pass
    with workspace.private_file_lock(path, timeout_s=0, unavailable="x", busy="b"):
        pass


def test_only_the_workspace_selects_studio_lock_primitives():
    """Studio modules share one lock helper instead of importing fcntl/msvcrt."""
    pattern = re.compile(r"^\s*import (fcntl|msvcrt)\b", re.MULTILINE)
    offenders = [
        path.name
        for path in sorted(PACKAGE.glob("finetune_*.py"))
        if path.name != "finetune_workspace.py" and pattern.search(path.read_text())
    ]
    assert offenders == []
