"""Review an agent-prepared run in a terminal without cloning or double submission."""

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as api
from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace
from stateset_agents.finetune_workspace import FinetuneConfig, read_json, write_json
from stateset_agents.tui import Studio


@pytest.fixture
def handoff(tmp_path, monkeypatch, settle_preview):
    dataset = tmp_path / "source chat.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Return policy?"},
                    {
                        "role": "assistant",
                        "content": "Returns are allowed for 30 days.",
                    },
                ]
            }
        )
        + "\n"
    )
    config = FinetuneConfig(
        dataset=str(dataset), cpu_slots=3, memory_gb=7, scratch_gb=2
    )
    config_path = tmp_path / "agent.json"
    write_json(config_path, asdict(config))
    root = tmp_path / "workspace"
    result = CliRunner().invoke(
        api.app, ["prepare", "--config", str(config_path), "--workspace", str(root)]
    )
    assert result.exit_code == 0, result.stdout
    prepared = json.loads(result.stdout)["result"]["run"]
    run = Path(prepared["path"])
    calls = []
    monkeypatch.setattr(
        workspace.subprocess,
        "Popen",
        lambda *args, **kwargs: calls.append((args, kwargs))
        or SimpleNamespace(pid=12345),
    )
    studio = Studio(root)
    studio.tab = "runs"
    return SimpleNamespace(
        root=root,
        run=run,
        config=config,
        studio=studio,
        calls=calls,
        prepared=prepared,
        settle=settle_preview,
    )


def test_agent_preparation_review_escape_and_launch_preserve_identity_and_draft(
    handoff,
):
    h = handoff
    studio = h.studio
    values = dict(studio.values)
    draft_plan = workspace.plan_run(replace(h.config, epochs=2, cpu_slots=1))
    studio.plan = draft_plan
    original = (h.run / "dataset").read_bytes()
    Path(h.config.dataset).unlink()
    assert studio.handle("r")
    handoff.settle(studio)
    assert studio.overlay == "prepared-review"
    assert studio.prepared_review.run == h.run
    text = "\n".join(studio.lines(110, 60))
    assert h.run.name in text and h.prepared["plan_sha256"] in text
    assert "3 CPUs / 7 GiB RAM" in text and "2 GiB" in text
    assert "3 epochs" in text and "2 epochs" not in text
    studio.handle("ESC")
    assert studio.prepared_review is None and not h.calls
    assert not (h.run / "launch.lock").exists()
    studio.handle("r")
    handoff.settle(studio)
    studio.handle("ENTER")
    handoff.settle(studio)
    assert len(h.calls) == 1 and h.calls[0][0][0][-1] == str(h.run)
    assert studio.runs[studio.run_index]["id"] == h.run.name
    assert (h.run / "dataset").read_bytes() == original
    assert studio.values == values and studio.plan is draft_plan
    assert len(list(h.root.glob("*/plan.json"))) == 1
    assert read_json(h.run / "launch.lock")["plan_sha256"] == h.prepared["plan_sha256"]
    replay = CliRunner().invoke(
        api.app,
        [
            "launch",
            h.run.name,
            "--workspace",
            str(h.root),
            "--expected-plan-sha256",
            h.prepared["plan_sha256"],
        ],
    )
    assert replay.exit_code == 0
    assert json.loads(replay.stdout)["result"]["already_submitted"]
    assert len(h.calls) == 1


@pytest.mark.parametrize("change", ["plan", "snapshot", "state", "submitted"])
def test_confirmation_rejects_changes_after_opening_review(handoff, change):
    h = handoff
    h.studio.handle("r")
    h.settle(h.studio)
    if change == "plan":
        record = read_json(h.run / "plan.json")
        record["config"]["epochs"] += 1
        write_json(h.run / "plan.json", record)
    elif change == "snapshot":
        (h.run / "dataset").write_text("changed")
    elif change == "state":
        write_json(h.run / "state.json", {"status": "failed"})
    else:
        workspace.launch_run(h.run)
    count = len(h.calls)
    h.studio.handle("ENTER")
    h.settle(h.studio)
    assert len(h.calls) == count and "Launch failed:" in h.studio.message
    assert h.studio.overlay is None and h.studio.prepared_review is None
    assert len(list(h.root.glob("*/plan.json"))) == 1


def test_review_stays_bound_to_original_selection_during_refresh(handoff):
    h = handoff
    h.studio.handle("ENTER")
    h.studio.handle("r")
    h.settle(h.studio)
    assert h.studio.overlay == "prepared-review"
    other = workspace.create_run(h.root, workspace.plan_run(h.config))
    h.studio.refresh()
    h.studio.run_index = next(
        index for index, row in enumerate(h.studio.runs) if row["id"] == other.name
    )
    h.studio.handle("ENTER")
    h.settle(h.studio)
    assert h.calls[0][0][0][-1] == str(h.run)
    assert not (other / "launch.lock").exists()


@pytest.mark.parametrize(
    "change", ["code", "version", "schema", "notes", "resource", "data", "symlink"]
)
def test_invalid_prepared_records_cannot_open_launch_review(handoff, change):
    h = handoff
    record = read_json(h.run / "plan.json")
    if change == "code":
        record["package_implementation_sha256"] = "0" * 64
    elif change == "version":
        record["framework_version"] = "old"
    elif change == "schema":
        record["schema_version"] = True
    elif change == "notes":
        record["notes"] = [42]
    elif change == "resource":
        record["resources"] = []
    elif change == "data":
        (h.run / "dataset").write_text("changed")
    else:
        retained = h.run / "retained-data"
        (h.run / "dataset").rename(retained)
        (h.run / "dataset").symlink_to(retained)
    write_json(h.run / "plan.json", record)
    h.studio.handle("r")
    h.settle(h.studio)
    assert h.studio.prepared_review is None and not h.calls
    assert "Cannot review prepared run:" in h.studio.message


def test_worker_rejects_config_only_plan_changes_during_queue_wait(handoff):
    h = handoff
    workspace.launch_run(h.run)
    assert worker._settings(h.run)[1] == h.config
    record = read_json(h.run / "plan.json")
    record["config"]["epochs"] += 1
    write_json(h.run / "plan.json", record)
    with pytest.raises(ValueError, match="plan bytes changed"):
        worker._settings(h.run)
    assert worker.execute(h.run) == 1
    assert read_json(h.run / "failure.json")["code"] == "review_invalid"
    assert read_json(h.run / "state.json")["status"] == "failed"


def test_remote_review_uses_saved_resources_and_requires_environment_key(
    handoff, monkeypatch
):
    h = handoff
    config = replace(h.config, provider="runpod", max_cost_usd=6.5)
    run = workspace.create_run(h.root, workspace.plan_run(config))
    h.studio.refresh()
    h.studio.run_index = next(
        index for index, row in enumerate(h.studio.runs) if row["id"] == run.name
    )
    h.studio.handle("r")
    h.settle(h.studio)
    text = "\n".join(h.studio.lines(110, 65))
    assert "RunPod pod ceiling: $6.50" in text
    assert "GPU:" in text and "Local queue: not held" in text
    assert "Local reservation:" not in text
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    h.studio.handle("ENTER")
    h.settle(h.studio)
    assert "RUNPOD_API_KEY is missing" in h.studio.message
    assert not h.calls and not (run / "launch.lock").exists()


def test_malformed_remote_resource_fields_are_rejected_before_render(handoff):
    h = handoff
    run = workspace.create_run(
        h.root, workspace.plan_run(replace(h.config, provider="runpod"))
    )
    record = read_json(run / "plan.json")
    del record["resources"]["gpu_count"]
    write_json(run / "plan.json", record)
    h.studio.refresh()
    h.studio.run_index = next(
        index for index, row in enumerate(h.studio.runs) if row["id"] == run.name
    )
    h.studio.handle("r")
    h.settle(h.studio)
    assert "Malformed retained resource metadata" in h.studio.message
    assert h.studio.prepared_review is None
    h.studio.lines(74, 44)


def test_receipt_write_failure_marks_run_failed_and_never_launches(
    handoff, monkeypatch
):
    original = json.dump

    def fail_receipt(record, *args, **kwargs):
        if "plan_sha256" in record:
            raise OSError("Synthetic receipt write failure")
        return original(record, *args, **kwargs)

    monkeypatch.setattr(json, "dump", fail_receipt)
    with pytest.raises(OSError, match="Synthetic receipt"):
        workspace.launch_run(handoff.run)
    state = read_json(handoff.run / "state.json")
    assert state["status"] == "failed" and "created_at" in state
    assert (handoff.run / "launch.lock").exists() and not handoff.calls


@pytest.mark.parametrize("receipt", ["legacy", "invalid", "oversized", "symlink"])
def test_launch_receipt_validation_and_legacy_compatibility(handoff, receipt):
    h = handoff
    lock = h.run / "launch.lock"
    if receipt == "legacy":
        lock.touch()
        assert worker._settings(h.run)[1] == h.config
    elif receipt == "invalid":
        write_json(
            lock, {"schema_version": True, "plan_sha256": h.prepared["plan_sha256"]}
        )
    elif receipt == "oversized":
        lock.write_text(" " * 4097)
    else:
        lock.symlink_to(h.run / "plan.json")
    if receipt != "legacy":
        with pytest.raises(ValueError):
            worker._settings(h.run)
    h.studio.handle("r")
    h.settle(h.studio)
    assert h.studio.prepared_review is None and not h.calls


@pytest.mark.parametrize("width,height", [(74, 44), (110, 32)])
def test_review_scrolls_full_plan_and_keeps_confirmation_visible(
    handoff, width, height
):
    studio = handoff.studio
    studio.handle("r")
    handoff.settle(studio)
    seen = []
    for _ in range(8):
        frame = studio.lines(width, height)
        assert len(frame) == height
        assert "Enter launch" in "\n".join(frame[-5:])
        seen.extend(frame)
        studio.handle("PAGEDOWN")
    text = "\n".join(seen)
    assert "Closing this terminal" in text
    assert handoff.prepared["plan_sha256"] in text
    assert "3 CPUs / 7 GiB RAM" in text
    assert not handoff.calls


@pytest.mark.skipif(
    sys.platform != "linux", reason="Real curses terminal uses Linux PTY"
)
@pytest.mark.parametrize("width,height", [(74, 44), (110, 32)])
def test_real_terminal_prepared_review_and_escape_do_not_launch(
    tmp_path, width, height
):
    import fcntl
    import pty
    import select
    import struct
    import subprocess
    import termios

    dataset = tmp_path / "chat.jsonl"
    dataset.write_text(
        '{"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]}\n'
    )
    root = tmp_path / "studio"
    run = workspace.create_run(
        root, workspace.plan_run(FinetuneConfig(dataset=str(dataset)))
    )
    digest = hashlib.sha256((run / "plan.json").read_bytes()).hexdigest()
    (run / "run.log").write_text("synthetic training log\n")
    (run / "supervisor.log").write_text("SUPERVISOR-ONLY-DIAGNOSIS\n")
    write_json(root / "draft.json", asdict(FinetuneConfig(dataset=str(dataset))))
    terminal, slave = pty.openpty()
    fcntl.ioctl(terminal, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
    process = subprocess.Popen(
        [sys.executable, "-m", "stateset_agents.cli", "tui", "--workspace", str(root)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
        env=dict(os.environ, TERM="xterm-256color"),
    )
    os.close(slave)
    captured = bytearray()

    def wait_for(text):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if text.encode() in captured:
                return
            ready, _, _ = select.select([terminal], [], [], 0.1)
            if ready:
                try:
                    captured.extend(os.read(terminal, 65536))
                except OSError:
                    break
        pytest.fail(f"Missing terminal output {text!r}: {bytes(captured)[-1500:]!r}")

    try:
        wait_for("FINE-TUNING STUDIO")
        os.write(terminal, b"w")
        wait_for("GUIDED SETUP")
        wait_for("snapshot ready; Enter opens")
        captured.clear()
        os.write(terminal, b"\x1b")
        wait_for("files | u data")
        os.write(terminal, b"u")
        wait_for("INSPECTOR / statistics")
        wait_for("Examples: 1")
        os.write(terminal, b"e")
        wait_for("SAMPLE 1/1")
        wait_for("assistant: Hello")
        captured.clear()
        os.write(terminal, b"\x1b")
        wait_for("files | u data")
        os.write(terminal, b"\t")
        wait_for("RUNS 1/1")
        os.write(terminal, b"e")
        wait_for("recorded base-versus-trained outcomes")
        wait_for("No verifiable quality evidence")
        captured.clear()
        os.write(terminal, b"\x1b")
        wait_for("r review prepared")
        os.write(terminal, b"k")
        wait_for("LOCAL STOP REQUEST")
        captured.clear()
        os.write(terminal, b"\x1b")
        wait_for("r review prepared")
        os.write(terminal, b"l")
        # Curses can preserve the RUN prefix from the previous RUNS heading.
        wait_for("LOGS / long lines wrap")
        wait_for("synthetic training log")
        os.write(terminal, b" ")
        wait_for("PAUSED")
        os.write(terminal, b"/missing-pattern\n")
        wait_for("No matching lines")
        os.write(terminal, b"\x7fss")
        wait_for("SUPERVISOR-ONLY-DIAGNOSIS")
        captured.clear()
        os.write(terminal, b"\x1b")
        wait_for("r review prepared")
        os.write(terminal, b"r")
        wait_for("REVIEW PREPARED RUN")
        wait_for(digest)
        captured.clear()
        os.write(terminal, b"\x1b")
        # Curses uses a short timeout to distinguish Esc from an arrow sequence.
        wait_for("r review prepared")
        os.write(terminal, b"a")
        # Curses can retain the 'Re' prefix of the previous Review message.
        wait_for("pinned. Select another run")
        candidate = workspace.create_run(
            root,
            workspace.plan_run(
                FinetuneConfig(dataset=str(dataset), learning_rate=5e-6)
            ),
        )
        captured.clear()
        wait_for(candidate.name[-8:])
        # History uses descending run IDs. Runs prepared in the same second
        # have random suffixes, so creation order does not establish row order.
        # Refresh preserves the pinned reference selection in either position.
        os.write(terminal, b"\x1bOA" if candidate.name > run.name else b"\x1bOB")
        os.write(terminal, b"x")
        wait_for("COMPARISON / candidate minus reference")
        wait_for("Training snapshots do not establish quality")
        captured.clear()
        os.write(terminal, b"\x1b")
        wait_for("r review prepared")
        os.write(terminal, b"q")
        assert process.wait(timeout=10) == 0
        assert not (run / "launch.lock").exists()
        assert not (run / "stop.json").exists()
        assert read_json(run / "state.json")["status"] == "prepared"
        assert not (candidate / "launch.lock").exists()
        assert len(list(root.glob("*/plan.json"))) == 2
    finally:
        os.close(terminal)
        if process.poll() is None:
            process.kill()
            process.wait()
