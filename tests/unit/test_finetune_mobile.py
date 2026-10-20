"""Phone-sized terminal layouts, literal numeric editing and resize safety."""

import io
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from rich.cells import cell_len
from typer.testing import CliRunner

from stateset_agents import tui
from stateset_agents.cli import app
from stateset_agents.finetune_evaluation_tui import EvaluationPanel


@pytest.mark.parametrize("size", [(31, 16), (39, 24), (59, 18), (71, 21), (99, 16)])
def test_phone_configuration_keeps_selected_field_and_controls_visible(tmp_path, size):
    studio = tui.Studio(tmp_path)
    for index in range(len(studio.fields())):
        studio.field = index
        frame = studio.lines(*size)
        assert len(frame) == size[1]
        assert all(cell_len(line) <= size[0] for line in frame)
        assert any(
            line.startswith(" > ") and studio.fields()[index].label in line
            for line in frame
        )
        assert any("2/8" in line for line in frame)
        assert any("? help" in line for line in frame)


def test_phone_navigation_digits_do_not_change_values_or_confirm_actions(tmp_path):
    studio = tui.Studio(tmp_path)
    studio.lines(39, 24)
    studio.handle("2")
    assert studio.field == 1
    studio.handle("8")
    assert studio.field == 0
    studio.handle("4")
    assert studio.tab == "runs"
    studio.handle("4")
    assert studio.tab == "configure"
    studio.overlay = "review"
    marker = object()
    studio.plan = marker
    for key in "5239":
        studio.handle(key)
    assert studio.plan is marker and studio.overlay == "review"
    assert studio.dialog_offset > 0
    studio.handle("0")
    assert studio.overlay is None
    assert not tmp_path.joinpath("plan.json").exists()


def test_phone_zero_and_other_navigation_digits_remain_valid_numeric_values(tmp_path):
    studio = tui.Studio(tmp_path)
    studio.lines(39, 24)
    studio.field = next(i for i, f in enumerate(studio.fields()) if f.name == "seed")
    studio.handle("ENTER")
    studio.handle("CLEAR")
    for key in "208394":
        studio.handle(key)
    assert studio.editing == "208394"
    studio.handle("ENTER")
    assert studio.config().seed == 208394
    studio.handle("ENTER")
    studio.handle("CLEAR")
    studio.handle("0")
    studio.handle("ENTER")
    assert studio.config().seed == 0


@pytest.mark.parametrize(
    "overlay", ["models", "files", "run-search", "quality-path", "logs", "evaluation"]
)
def test_phone_filters_paths_and_evaluation_editor_preserve_all_digits(
    tmp_path, overlay
):
    studio = tui.Studio(tmp_path)
    studio.overlay = overlay
    if overlay == "logs":
        studio.log_search_edit = ""
    if overlay == "evaluation":
        studio.evaluation_panel = EvaluationPanel(tmp_path / "absent")
        studio.evaluation_panel.mode = "edit"
        studio.evaluation_panel.edit = ""
    studio.lines(39, 24)
    for key in "208394":
        studio.handle(key)
    if overlay == "files":
        value = studio.file_search
    elif overlay == "logs":
        value = studio.log_search_edit
    elif overlay == "evaluation":
        value = studio.evaluation_panel.edit
    else:
        value = studio.search
    assert value == "208394" and studio.overlay == overlay


@pytest.mark.parametrize("size", [(31, 16), (39, 24), (59, 18)])
def test_phone_backend_selection_follows_focus_and_allows_details_scrolling(
    tmp_path, size
):
    studio = tui.Studio(tmp_path)
    studio.lines(*size)
    studio.handle("b")
    for _ in range(4):
        frame = studio.lines(*size)
        assert any(
            line.lstrip().startswith("> ")
            and tui.PROVIDERS[studio.backend_index] in line
            for line in frame
        )
        studio.handle("2")
    studio.lines(*size)
    studio.handle("3")
    studio.lines(*size)
    assert studio.dialog_offset > 0
    studio.handle("0")
    assert studio.overlay is None
    assert studio.values["provider"] == "local"


def test_desktop_shortcuts_do_not_interpret_numeric_navigation(tmp_path):
    studio = tui.Studio(tmp_path)
    studio.lines(110, 32)
    studio.handle("2")
    assert studio.field == 0
    assert "Arrows/Enter" in "\n".join(studio.lines(110, 32))


@pytest.mark.parametrize("size", [(32, 16), (40, 24)])
def test_phone_snapshot_dimensions_are_explicit_and_create_no_workspace(tmp_path, size):
    workspace = tmp_path / "snapshot-only"
    result = CliRunner().invoke(
        app,
        [
            "tui",
            "--snapshot",
            "--workspace",
            str(workspace),
            "--columns",
            str(size[0]),
            "--rows",
            str(size[1]),
        ],
    )
    assert result.exit_code == 0, result.output
    frame = result.output.splitlines()
    assert len(frame) == size[1] and all(cell_len(line) <= size[0] for line in frame)
    assert "2/8" in result.output and not workspace.exists()


def test_explicit_snapshot_dimensions_are_not_silently_ignored_interactively(tmp_path):
    result = CliRunner().invoke(
        app, ["tui", "--workspace", str(tmp_path / "unused"), "--columns", "40"]
    )
    assert result.exit_code == 2 and "require --snapshot" in result.output
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("size", [(31, 16), (40, 15)])
def test_snapshot_rejects_unusable_dimensions(tmp_path, size):
    result = CliRunner().invoke(
        app,
        [
            "tui",
            "--snapshot",
            "--workspace",
            str(tmp_path / "unused"),
            "--columns",
            str(size[0]),
            "--rows",
            str(size[1]),
        ],
    )
    assert result.exit_code == 2 and not (tmp_path / "unused").exists()


class Screen:
    def __init__(self, sizes, keys):
        self.sizes, self.keys, self.index = sizes, keys, 0
        self.frames = []

    def keypad(self, value):
        pass

    def timeout(self, value):
        pass

    def erase(self):
        self.frames.append([])

    def getmaxyx(self):
        return self.sizes[min(self.index, len(self.sizes) - 1)]

    def addstr(self, row, column, text, style):
        self.frames[-1].append(text)

    def refresh(self):
        pass

    def move(self, row, column):
        height, width = self.getmaxyx()
        assert 0 <= row < height and 0 <= column < width

    def get_wch(self):
        assert self.index < len(self.keys), "Unexpected terminal loop"
        key = self.keys[self.index]
        self.index += 1
        return key


def simulated_terminal(monkeypatch, studio, screen):
    class TerminalIO(io.StringIO):
        def isatty(self):
            return True

    class CursesError(Exception):
        pass

    constants = [
        "KEY_UP",
        "KEY_DOWN",
        "KEY_LEFT",
        "KEY_RIGHT",
        "KEY_NPAGE",
        "KEY_PPAGE",
        "KEY_END",
        "KEY_HOME",
        "KEY_DC",
        "KEY_BACKSPACE",
        "KEY_ENTER",
    ]
    curses = SimpleNamespace(
        **{name: 1000 + i for i, name in enumerate(constants)},
        error=CursesError,
        A_REVERSE=1,
        A_BOLD=2,
        A_DIM=4,
        wrapper=lambda main: main(screen),
        curs_set=lambda value: None,
        has_colors=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "curses", curses)
    monkeypatch.setattr(tui.sys, "stdin", TerminalIO())
    monkeypatch.setattr(tui.sys, "stdout", TerminalIO())
    monkeypatch.setattr(tui, "Studio", lambda *args, **kwargs: studio)
    tui.run_studio(studio.workspace)


@pytest.mark.parametrize("size", [(16, 32), (24, 40)])
def test_interactive_runner_renders_phone_terminal_instead_of_resize_wall(
    tmp_path, monkeypatch, size
):
    studio = tui.Studio(tmp_path)
    screen = Screen([size], ["q"])
    simulated_terminal(monkeypatch, studio, screen)
    assert "[CONFIGURE]" in "\n".join(screen.frames[0])
    assert "Paused:" not in "\n".join(screen.frames[0])


def test_too_small_terminal_never_confirms_a_hidden_launch(tmp_path, monkeypatch):
    studio = tui.Studio(tmp_path)
    studio.overlay = "review"
    marker = object()
    studio.plan = marker

    def forbidden(*args, **kwargs):
        pytest.fail("Hidden launch must not be confirmed")

    monkeypatch.setattr(tui, "create_run", forbidden)
    screen = Screen([(12, 20)], ["\n", "q"])
    simulated_terminal(monkeypatch, studio, screen)
    assert studio.plan is marker and studio.overlay == "review"
    assert "Paused:" in "\n".join(screen.frames[0])


def test_resize_pause_preserves_editor_and_ignores_enter_until_usable(
    tmp_path, monkeypatch
):
    studio = tui.Studio(tmp_path)
    studio.field = next(i for i, f in enumerate(studio.fields()) if f.name == "seed")
    studio.editing = "0"
    studio.edit_cursor = 1
    screen = Screen([(12, 20), (24, 40)], ["\n", "\x03"])
    simulated_terminal(monkeypatch, studio, screen)
    assert studio.editing == "0" and studio.config().seed == 42
    assert "EDIT: 0" in "\n".join(screen.frames[1])


@pytest.mark.parametrize("after_key", [(12, 20), (16, 40)])
def test_resize_after_paint_cannot_confirm_an_outdated_launch_review(
    tmp_path, monkeypatch, after_key
):
    studio = tui.Studio(tmp_path)
    studio.overlay = "review"
    plan = SimpleNamespace(config=SimpleNamespace(provider="local"))
    studio.plan = plan
    monkeypatch.setattr(
        studio, "lines", lambda *args: ["REVIEW / explicit Enter launch"]
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Viewport changed after paint; launch must wait for repaint")

    monkeypatch.setattr(tui, "create_run", forbidden)
    screen = Screen([(24, 80), after_key], ["\n", "\x03"])
    simulated_terminal(monkeypatch, studio, screen)
    assert studio.plan is plan and studio.overlay == "review"


@pytest.mark.skipif(
    sys.platform != "linux", reason="Phone curses sessions use a Linux PTY"
)
@pytest.mark.parametrize("size", [(32, 16), (40, 24)])
def test_real_phone_terminal_selects_backend_edits_zero_and_saves_without_launch(
    tmp_path, size
):
    import fcntl
    import pty
    import select
    import struct
    import termios
    import time

    studio = tui.Studio(tmp_path)
    studio.values["provider"] = "river"
    seed = next(i for i, f in enumerate(studio.fields()) if f.name == "seed")
    terminal, child = pty.openpty()
    fcntl.ioctl(
        terminal, termios.TIOCSWINSZ, struct.pack("HHHH", size[1], size[0], 0, 0)
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "stateset_agents.cli",
            "tui",
            "--workspace",
            str(tmp_path),
        ],
        stdin=child,
        stdout=child,
        stderr=child,
        env=dict(
            os.environ, TERM="xterm-256color", COLUMNS=str(size[0]), LINES=str(size[1])
        ),
    )
    os.close(child)
    captured = bytearray()

    def wait_for(marker):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if marker.encode() in captured:
                return
            ready, _, _ = select.select([terminal], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(terminal, 65536)
                except OSError:
                    break
                captured.extend(chunk)
        pytest.fail(f"Missing phone marker {marker!r}: {bytes(captured)[-1500:]!r}")

    def send(keys, marker):
        captured.clear()
        os.write(terminal, keys)
        wait_for(marker)

    try:
        wait_for("STATESET / FINE-TUNING")
        send(b"b", "TRAINING BACKENDS")
        send(b"222", "river")
        send(b"\r", "m model")
        send(b"2" * seed + b"\r", "EDIT:")
        # Long save paths wrap after the word, without emitting a trailing space.
        send(b"\x150\rs", "Saved")
        os.write(terminal, b"q")
        assert process.wait(timeout=15) == 0
        saved = json.loads((tmp_path / "draft.json").read_text())
        assert saved["seed"] == 0 and saved["provider"] == "river"
        assert not list(tmp_path.glob("*/plan.json"))
    finally:
        os.close(terminal)
        if process.poll() is None:
            process.kill()
            process.wait()


def test_phone_field_editor_keeps_save_and_cancel_visible(tmp_path):
    studio = tui.Studio(tmp_path)
    studio.lines(39, 24)
    studio.field = next(i for i, f in enumerate(studio.fields()) if f.name == "seed")
    studio.handle("ENTER")
    frame = studio.lines(31, 16)
    assert frame[-1] == "Enter save | Esc discard"
