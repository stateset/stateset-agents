"""Studio overlays extracted as panels keep their own state, keys and frames."""

from __future__ import annotations

import os

import pytest

from stateset_agents.tui_panels import FilePicker, LogViewer


@pytest.fixture
def run(tmp_path):
    (tmp_path / "run.log").write_text(
        "".join(f"step {index}\n" for index in range(40)) + "WARNING late\n"
    )
    (tmp_path / "supervisor.log").write_text("supervisor line\n")
    return tmp_path


def test_log_viewer_follows_pauses_filters_and_closes(run):
    viewer = LogViewer(run)
    assert "FOLLOW" in "\n".join(viewer.lines(60, 20))
    assert viewer.handle(" ") is False and not viewer.follow
    (run / "run.log").write_text("NEW OUTPUT\n")
    assert "NEW OUTPUT" not in "\n".join(viewer.lines(60, 20))  # frozen snapshot
    viewer.handle("/")
    assert viewer.editing
    for key in "warn":
        viewer.handle(key)
    assert viewer.keys(60)[0] == "Filter: warn"
    viewer.handle("ENTER")
    assert viewer.query == "warn" and not viewer.editing
    assert "WARNING late" in "\n".join(viewer.lines(60, 20))
    viewer.handle("BACKSPACE")
    viewer.handle("END")
    assert viewer.follow and "NEW OUTPUT" in "\n".join(viewer.lines(60, 20))
    viewer.handle("s")
    viewer.handle("s")
    assert "supervisor line" in "\n".join(viewer.lines(60, 20))
    assert viewer.handle("ESC") is True


def test_log_viewer_filter_editor_discards_on_escape(run):
    viewer = LogViewer(run)
    viewer.handle("/")
    viewer.handle("x")
    assert viewer.handle("ESC") is False  # leaves the editor, not the viewer
    assert viewer.query == "" and not viewer.editing


def test_file_picker_lists_only_selectable_files_and_returns_a_choice(tmp_path):
    (tmp_path / "train.jsonl").write_text("{}")
    (tmp_path / "cases.json").write_text("[]")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / ".hidden.json").write_text("[]")
    (tmp_path / "nested").mkdir()
    picker = FilePicker(tmp_path)
    picker.read()
    names = [name for name, _, _ in picker.entries]
    assert names == ["..", "nested", "cases.json", "train.jsonl"]
    for key in "train":
        assert picker.handle(key) is None
    assert picker.handle("ENTER") == tmp_path / "train.jsonl"
    assert "DATASET PICKER" in picker.lines(60, 20)[0]


def test_file_picker_navigates_directories_without_choosing(tmp_path):
    (tmp_path / "nested").mkdir()
    picker = FilePicker(tmp_path)
    picker.read()
    picker.handle("DOWN")
    assert picker.handle("ENTER") is None
    assert picker.directory == (tmp_path / "nested").resolve()
    assert picker.handle("BACKSPACE") is None
    assert picker.directory == tmp_path.resolve()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_case_picker_accepts_only_regular_json_files(tmp_path):
    (tmp_path / "cases.json").write_text("[]")
    (tmp_path / "train.jsonl").write_text("{}")
    try:
        (tmp_path / "linked.json").symlink_to(tmp_path / "cases.json")
    except OSError:
        pytest.skip("symlinks unavailable")
    picker = FilePicker(tmp_path, cases=True)
    picker.read()
    assert [name for name, _, _ in picker.entries] == ["..", "cases.json"]
    assert "HELD-OUT CASE PICKER" in picker.lines(60, 20)[0]
