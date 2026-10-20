"""Command search, responsive terminal drawing, and keyboard navigation."""

from types import SimpleNamespace

import pytest
from rich.cells import cell_len

from stateset_agents.tui import Studio
from stateset_agents.tui_actions import ActionMenu
from stateset_agents.tui_render import FrameRenderer, configure_terminal


class Screen:
    def __init__(self):
        self.erases = 0
        self.writes = []
        self.fail = False

    def erase(self):
        self.erases += 1

    def addstr(self, *args):
        if self.fail:
            raise RuntimeError("racing resize")
        self.writes.append(args)


def curses_fake():
    return SimpleNamespace(
        error=RuntimeError,
        A_BOLD=256,
        A_REVERSE=512,
        A_DIM=1024,
        color_pair=lambda pair: pair,
    )


def test_renderer_repaints_only_changed_rows_and_removes_old_tail():
    screen = Screen()
    renderer = FrameRenderer(curses_fake(), colors=True)
    renderer.draw(screen, ["Heading", "long old text"], 30, 3)
    assert screen.erases == 1 and len(screen.writes) == 3
    renderer.draw(screen, ["Heading", "long old text"], 30, 3)
    assert len(screen.writes) == 3
    renderer.draw(screen, ["Heading", "new"], 30, 3)
    assert len(screen.writes) == 4
    assert screen.writes[-1][2] == "new" + " " * 26
    renderer.draw(screen, ["Heading"], 20, 2)
    assert screen.erases == 2
    assert all(cell_len(row[2]) == 19 for row in screen.writes[-2:])


def test_renderer_preserves_unicode_cells_and_retries_failed_draw():
    screen = Screen()
    renderer = FrameRenderer(curses_fake(), colors=False)
    screen.fail = True
    renderer.draw(screen, [" > 退款 e\u0301"], 14, 1)
    assert renderer.previous == [None]
    screen.fail = False
    renderer.draw(screen, [" > 退款 e\u0301"], 14, 1)
    assert cell_len(screen.writes[-1][2]) == 13
    assert "退款 e\u0301" in screen.writes[-1][2]
    assert screen.writes[-1][3] & 512
    assert renderer.style("[FAIL] Invalid configuration", 4) & 256


@pytest.mark.parametrize("explicit", [False, True])
def test_escape_delay_honors_operator_setting(monkeypatch, explicit):
    calls = []
    if explicit:
        monkeypatch.setenv("ESCDELAY", "400")
    else:
        monkeypatch.delenv("ESCDELAY", raising=False)
    screen = SimpleNamespace(
        keypad=lambda value: calls.append(("keypad", value)),
        timeout=lambda value: calls.append(("timeout", value)),
    )
    configure_terminal(
        screen,
        SimpleNamespace(set_escdelay=lambda value: calls.append(("escape", value))),
    )
    assert ("timeout", 100) in calls
    assert (("escape", 35) in calls) is not explicit


def test_palette_search_matches_words_and_never_confirms_without_enter():
    menu = ActionMenu("configure", has_run=False, searchable=True)
    for char in "training launch":
        assert menu.handle(char) is None
    assert len(menu.matches()) == 1
    assert menu.matches()[0].key == "r"
    assert menu.handle("ENTER") == "r"
    menu.handle("CLEAR")
    for char in "NO SUCH COMMAND":
        menu.handle(char)
    assert not menu.matches()
    assert menu.handle("ENTER") is None
    assert any("No matching" in line for line in menu.lines(10))
    menu.handle("CLEAR")
    menu.handle("END")
    assert menu.index == len(menu.actions) - 1
    menu.handle("HOME")
    assert menu.index == 0


@pytest.mark.parametrize("width,height", [(32, 16), (40, 24), (80, 24), (140, 40)])
def test_palette_digits_are_literal_and_frames_fit(tmp_path, width, height):
    studio = Studio(tmp_path / "workspace")
    studio.lines(width, height)
    studio.handle("COMMANDS")
    for char in "28":
        studio.handle(char)
    assert studio.action_menu.query == "28"
    frame = studio.lines(width, height)
    assert len(frame) == height and all(cell_len(line) <= width for line in frame)
    assert not studio.workspace.exists()
    studio.handle("ESC")
    assert studio.overlay is None


def test_palette_launch_opens_existing_review_and_selection_stays_bound(
    tmp_path, monkeypatch
):
    studio = Studio(tmp_path / "workspace")
    calls = []
    monkeypatch.setattr(studio, "preview", lambda **kwargs: calls.append(kwargs))
    studio.handle("COMMANDS")
    for char in "training launch":
        studio.handle(char)
    assert not calls
    studio.handle("ENTER")
    assert calls == [{"review": True}]
    assert not studio.workspace.exists()
    studio.tab = "runs"
    studio.runs = [{"id": "old"}]
    studio.handle("COMMANDS")
    for char in "stop":
        studio.handle(char)
    studio.runs = [{"id": "new"}]
    studio.handle("ENTER")
    assert "Selected run changed" in studio.message


def test_settings_page_and_end_keys_preserve_edit_cursor_semantics(tmp_path):
    studio = Studio(tmp_path / "workspace")
    studio.lines(100, 32)
    studio.handle("PAGEDOWN")
    assert studio.field == min(studio.field_page_size, len(studio.fields()) - 1)
    studio.handle("END")
    assert studio.field == len(studio.fields()) - 1
    studio.handle("HOME")
    assert studio.field == 0
    studio.field = next(
        i for i, field in enumerate(studio.fields()) if field.name == "model"
    )
    studio.handle("ENTER")
    studio.handle("HOME")
    assert studio.edit_cursor == 0
    studio.handle("END")
    assert studio.edit_cursor == len(studio.editing)
