"""Evaluation case browsing preserves training state and explicit review."""

import copy
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.cells import cell_len

from stateset_agents.finetune_evaluation_tui import EvaluationPanel
from stateset_agents.tui import Studio


def editor(tmp_path):
    studio = Studio(tmp_path / "workspace")
    studio.values["dataset"] = str(tmp_path / "training.jsonl")
    studio.handle("s")
    panel = EvaluationPanel(studio.workspace / "synthetic-run")
    panel.mode = "form"
    panel.values["cases"] = str(tmp_path / "selected.json")
    studio.evaluation_panel, studio.overlay = panel, "evaluation"
    return studio, panel


def filter_name(studio, name):
    for char in name:
        studio.handle(char)


@pytest.mark.parametrize("size", [(72, 22), (74, 44), (110, 32)])
def test_picker_selects_only_cases_without_reads_launches_or_draft_changes(
    tmp_path, monkeypatch, size
):
    chosen = tmp_path / "held-out café.json"
    chosen.write_text("Invalid until review; private case contents.")
    studio, panel = editor(tmp_path)
    values = copy.deepcopy(studio.values)
    draft = (studio.workspace / "draft.json").read_bytes()
    panel.review, panel.prepared_path, panel.result = object(), tmp_path / "old", {}

    def forbidden(*args, **kwargs):
        pytest.fail("Browsing must not read contents, review inputs or submit work")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_bytes", forbidden)
        scoped.setattr(Path, "read_text", forbidden)
        for name in ("plan_evaluation", "prepare_evaluation", "launch_evaluation"):
            scoped.setattr("stateset_agents.finetune_evaluation_tui." + name, forbidden)
        studio.handle("f")
        assert studio.overlay == "files" and studio.file_evaluation is panel
        filter_name(studio, chosen.name)
        screen = studio.lines(*size)
        assert len(screen) == size[1]
        assert all(cell_len(line) <= size[0] for line in screen)
        assert "HELD-OUT CASE PICKER" in "\n".join(screen)
        studio.handle("ENTER")
        assert studio.overlay == "evaluation" and panel.mode == "form"
        assert panel.values["cases"] == str(chosen)
        assert panel.field == 0 and panel.review is None
        assert panel.prepared_path is None and panel.result is None
        assert "Press r" in panel.message
        assert studio.values == values and not panel.busy
    assert (studio.workspace / "draft.json").read_bytes() == draft
    assert not (panel.run / "evaluations").exists()


def test_escape_restores_the_captured_form_and_preserves_its_values(tmp_path):
    directory = tmp_path / "case directory"
    directory.mkdir()
    (directory / "test.json").write_text("{}")
    studio, panel = editor(tmp_path)
    before = copy.deepcopy(panel.values)
    panel.field = 5
    review = panel.review = object()
    studio.handle("f")
    filter_name(studio, directory.name)
    studio.handle("ENTER")
    assert studio.file_directory == directory
    studio.handle("BACKSPACE")
    assert studio.file_directory == tmp_path
    studio.handle("ESC")
    assert studio.overlay == "evaluation" and studio.file_evaluation is None
    assert panel.mode == "form" and panel.field == 5
    assert panel.values == before and panel.review is review


@pytest.mark.skipif(
    os.name != "posix", reason="POSIX symlink and FIFO directory entries"
)
def test_case_picker_lists_supported_regular_json_and_directories(tmp_path):
    (tmp_path / "test.JSON").write_text("{}")
    (tmp_path / "training.jsonl").write_text("{}")
    (tmp_path / "notes.txt").write_text("private")
    (tmp_path / ".hidden.json").write_text("private")
    (tmp_path / "linked.json").symlink_to(tmp_path / "test.JSON")
    os.mkfifo(tmp_path / "pipe.json")
    studio, _ = editor(tmp_path)
    studio.handle("f")
    names = {name for name, _, directory in studio.file_entries if not directory}
    assert names == {"test.JSON"}
    studio.handle("ESC")


@pytest.mark.parametrize("change", ["busy", "mode", "owner", "closed"])
def test_late_selection_cannot_mutate_a_changed_evaluation(tmp_path, change):
    chosen = tmp_path / "test.json"
    chosen.write_text("{}")
    studio, original = editor(tmp_path)
    before = copy.deepcopy(original.values)
    studio.handle("f")
    filter_name(studio, chosen.name)
    if change == "busy":
        original.busy = True
    elif change == "mode":
        original.mode = "review"
    elif change == "owner":
        studio.evaluation_panel = EvaluationPanel(studio.workspace / "different-run")
    else:
        studio.evaluation_panel = None
    current = studio.evaluation_panel
    current_values = copy.deepcopy(current.values) if current else None
    studio.handle("ENTER")
    assert original.values == before
    assert studio.file_evaluation is None
    if current:
        assert current.values == current_values
        assert "discarded" in current.message
        assert studio.overlay == "evaluation"
    else:
        assert studio.overlay is None


def test_directory_permission_failure_can_return_without_losing_editor(
    tmp_path, monkeypatch
):
    studio, panel = editor(tmp_path)
    before = dict(panel.values)

    def denied(*args, **kwargs):
        raise PermissionError("PRIVATE_DIRECTORY_ERROR")

    monkeypatch.setattr("stateset_agents.tui.os.scandir", denied)
    studio.handle("f")
    assert "PermissionError" in studio.file_error
    assert "PRIVATE_DIRECTORY_ERROR" not in studio.file_error
    assert studio.file_entries[0][0] == ".."
    studio.handle("ESC")
    assert studio.overlay == "evaluation" and panel.values == before


def test_directory_scan_is_bounded_and_pages_keep_selection_visible(
    tmp_path, monkeypatch
):
    studio, _ = editor(tmp_path)
    scanned = []

    @contextmanager
    def entries(*args):
        def rows():
            for index in range(3000):
                scanned.append(index)
                yield SimpleNamespace(
                    name=f"case-{index:04d}.json",
                    path=str(tmp_path / f"case-{index:04d}.json"),
                    is_dir=lambda: False,
                    is_file=lambda **kwargs: True,
                )

        yield rows()

    monkeypatch.setattr("stateset_agents.tui.os.scandir", entries)
    studio.handle("f")
    assert len(scanned) == 2000 and len(studio.file_entries) == 2001
    studio.handle("PAGEDOWN")
    assert studio.file_index == 8
    studio.handle("PAGEUP")
    assert studio.file_index == 0
    studio.handle("END")
    assert "case-1999.json" in "\n".join(studio.lines(74, 44))
    studio.handle("HOME")
    studio.handle("ESC")


def test_browsing_does_not_load_model_libraries(tmp_path):
    code = """from pathlib import Path
import sys
from stateset_agents.tui import Studio
from stateset_agents.finetune_evaluation_tui import EvaluationPanel
s=Studio(Path(sys.argv[1])/'workspace')
p=EvaluationPanel(s.workspace/'synthetic-run'); p.mode='form'
s.evaluation_panel=p; s.overlay='evaluation'; s.handle('f'); s.handle('ESC')
assert s.overlay=='evaluation'
assert not any(name in sys.modules for name in ['torch','transformers','peft'])
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True)
