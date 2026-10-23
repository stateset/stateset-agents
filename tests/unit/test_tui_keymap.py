"""The studio's table-driven key bindings cover exactly what the old if-chain did.

``Studio.handle`` used to be one 458-line if/elif chain. The expected key sets
below were transcribed from that chain (stateset_agents/tui.py before the
split into stateset_agents/tui_keys.py), per input mode and per dialog, so a
binding cannot silently disappear or appear when the tables change.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from stateset_agents import tui, tui_keys, tui_screens
from tests.unit.test_complexity_ratchet import oversized_functions

SCROLL = {"UP", "DOWN", "PAGEUP", "PAGEDOWN"}

# Transcribed from the old chain: keys each dialog acted on besides Esc.
# ``True`` marks dialogs whose final branch accepted other keys too (a
# printable-character fallback, or the file picker receiving every key).
EXPECTED_OVERLAY_KEYS: dict[str, tuple[set[str], bool]] = {
    "workflow": ({"w", "UP", "DOWN", "ENTER", "PAGEUP", "PAGEDOWN"}, False),
    "recovery": ({"z", "UP", "DOWN", "ENTER", "PAGEUP", "PAGEDOWN"}, False),
    "recovery-review": (SCROLL | {"ENTER"}, False),
    "stop": (SCROLL | {"ENTER"}, False),
    "quality": (SCROLL | {"e", "j", "f"}, False),
    "files": (set(), True),
    "validation": (SCROLL | {"v", "ENTER"}, False),
    "backends": (SCROLL | {"ENTER"}, False),
    "data": (SCROLL | {"u", "e", "LEFT", "RIGHT"}, False),
    "readiness": (SCROLL | {"i"}, False),
    "run-search": ({"ENTER", "BACKSPACE", "CLEAR"}, True),
    "review": (SCROLL | {"ENTER"}, False),
    "prepared-review": (SCROLL | {"ENTER"}, False),
    "help": (SCROLL, False),
    "metrics": (SCROLL, False),
    "compare": (SCROLL, False),
    "details": (SCROLL | set("crlaxejkz"), False),
    "models": ({"UP", "DOWN", "ENTER", "BACKSPACE"}, True),
}
EXPECTED_EDIT_KEYS = {
    "ESC",
    "ENTER",
    "BACKSPACE",
    "DELETE",
    "LEFT",
    "RIGHT",
    "HOME",
    "END",
    "CLEAR",
}
EXPECTED_QUALITY_PATH_KEYS = {"ESC", "ENTER", "BACKSPACE", "CLEAR"}
EXPECTED_GLOBAL_KEYS = {"TAB", ":", "COMMANDS", "?", "i"}
EXPECTED_CONFIGURE_KEYS = {
    "UP",
    "DOWN",
    "HOME",
    "END",
    "PAGEUP",
    "PAGEDOWN",
    "ENTER",
    "LEFT",
    "RIGHT",
    *"wmuvbfprs",
}
EXPECTED_RUNS_KEYS = {
    "UP",
    "DOWN",
    "PAGEUP",
    "PAGEDOWN",
    "HOME",
    "END",
    "ENTER",
    "BACKSPACE",
    "/",
    *"dcrtglaxejkz",
}


@pytest.mark.parametrize("overlay", sorted(EXPECTED_OVERLAY_KEYS))
def test_each_dialog_binds_the_keys_the_old_chain_handled(overlay: str) -> None:
    keys, has_fallback = EXPECTED_OVERLAY_KEYS[overlay]
    keymap = tui_keys.OVERLAY_KEYS[overlay]
    assert set(keymap.keys) == keys
    assert (keymap.fallback is not None) is has_fallback


def test_no_dialog_gains_or_loses_a_table() -> None:
    # logs, planning, actions, quality-path and evaluation are handled before
    # the dialog tables; unknown dialogs only respond to Esc.
    assert set(tui_keys.OVERLAY_KEYS) == set(EXPECTED_OVERLAY_KEYS)


def test_text_modes_and_main_screen_keys_match_the_old_chain() -> None:
    assert set(tui_keys.EDIT_KEYS.keys) == EXPECTED_EDIT_KEYS
    assert tui_keys.EDIT_KEYS.fallback is not None
    assert set(tui_keys.QUALITY_PATH_KEYS.keys) == EXPECTED_QUALITY_PATH_KEYS
    assert tui_keys.QUALITY_PATH_KEYS.fallback is not None
    assert set(tui_keys.GLOBAL_KEYS) == EXPECTED_GLOBAL_KEYS
    assert set(tui_keys.CONFIGURE_KEYS) == EXPECTED_CONFIGURE_KEYS
    assert set(tui_keys.RUNS_KEYS) == EXPECTED_RUNS_KEYS
    assert tui_keys.COMPACT_DIGITS == {
        "2": "DOWN",
        "8": "UP",
        "3": "PAGEDOWN",
        "9": "PAGEUP",
        "0": "ESC",
    }


def _mode(**state: object) -> str:
    defaults: dict[str, object] = {
        "overlay": None,
        "action_menu": None,
        "editing": None,
        "evaluation_panel": None,
    }
    return tui_keys.key_mode(SimpleNamespace(**{**defaults, **state}))


def test_mode_precedence_matches_the_old_chain() -> None:
    menu, panel = object(), object()
    assert _mode(overlay="planning", editing="x") == "planning"
    assert _mode(overlay="actions", action_menu=menu, editing="x") == "actions"
    assert _mode(overlay="actions", editing="x") == "editing"
    assert _mode(overlay="actions") == "overlay"
    assert _mode(overlay="quality-path", editing="x") == "quality-path"
    assert _mode(overlay="evaluation", editing="", evaluation_panel=panel) == (
        "editing"
    )
    assert _mode(overlay="evaluation", evaluation_panel=panel) == "evaluation"
    assert _mode(overlay="evaluation") == "overlay"
    assert _mode(overlay="logs") == "overlay"
    assert _mode(overlay="") == "main"
    assert _mode() == "main"
    assert set(tui_keys.MODE_HANDLERS) == {
        "planning",
        "actions",
        "quality-path",
        "editing",
        "evaluation",
        "overlay",
        "main",
    }


def _table_handlers() -> list[object]:
    tables = [
        tui_keys.GLOBAL_KEYS,
        tui_keys.CONFIGURE_KEYS,
        tui_keys.RUNS_KEYS,
        tui_keys.EDIT_KEYS.keys,
        tui_keys.QUALITY_PATH_KEYS.keys,
        *(keymap.keys for keymap in tui_keys.OVERLAY_KEYS.values()),
    ]
    fallbacks = [
        keymap.fallback
        for keymap in (
            tui_keys.EDIT_KEYS,
            tui_keys.QUALITY_PATH_KEYS,
            *tui_keys.OVERLAY_KEYS.values(),
        )
        if keymap.fallback is not None
    ]
    return [
        *(handler for table in tables for handler in table.values()),
        *fallbacks,
        *tui_keys.MODE_HANDLERS.values(),
        *tui_screens.OVERLAY_BODIES.values(),
    ]


def test_table_handlers_are_small_documented_module_functions() -> None:
    # Every handler lives in one of these modules, so checking the modules
    # with the complexity ratchet's own measure covers every handler.
    modules = {"stateset_agents.tui_keys", "stateset_agents.tui_screens"}
    for handler in _table_handlers():
        assert handler.__module__ in modules, handler
        assert handler.__doc__, handler
    oversized = {
        (entry.split(":", 1)[0], entry.split(" ", 2)[1])
        for entry in oversized_functions()
    }
    for path in ("stateset_agents/tui_keys.py", "stateset_agents/tui_screens.py"):
        assert not [name for file, name in oversized if file == path], path
    for key in (
        ("stateset_agents/tui.py", "handle"),
        ("stateset_agents/tui.py", "refresh"),
        ("stateset_agents/tui.py", "lines"),
        ("stateset_agents/tui_views.py", "run_details_lines"),
    ):
        assert key not in oversized, f"{key} is oversized again"


@pytest.fixture
def studio(tmp_path: Path) -> tui.Studio:
    return tui.Studio(tmp_path / "workspace")


def test_configure_tab_clamps_the_setting_for_every_non_global_key(
    studio: tui.Studio,
) -> None:
    last = len(studio.fields()) - 1
    studio.field = 999
    assert studio.handle("?") is True
    assert (studio.field, studio.overlay) == (999, "help")
    studio.overlay = None
    assert studio.handle("y") is True  # unbound, but still clamps
    assert studio.field == last


def test_q_exits_only_from_the_main_screen(studio: tui.Studio) -> None:
    studio.overlay = "help"
    assert studio.handle("q") is True
    studio.editing = "text"
    assert studio.handle("QUIT") is False
    studio.editing, studio.overlay = None, None
    assert studio.handle("q") is False


def test_compact_digits_navigate_except_in_text_editors(studio: tui.Studio) -> None:
    studio.compact_layout = True
    assert studio.handle("4") is True
    assert studio.tab == "runs"
    studio.handle("4")
    studio.handle("2")
    assert (studio.tab, studio.field) == ("configure", 1)
    studio.editing, studio.edit_cursor = "", 0
    studio.handle("2")
    assert (studio.editing, studio.field) == ("2", 1)


def test_runs_tab_conditional_keys_need_a_run(studio: tui.Studio) -> None:
    studio.tab = "runs"
    for key in ("ENTER", "d", "g", "l"):
        assert studio.handle(key) is True
        assert studio.overlay is None


def test_escape_resets_dialogs_without_their_own_bindings(studio: tui.Studio) -> None:
    studio.overlay, studio.comparison, studio.dialog_offset = "compare", {}, 7
    studio.data_panel.show_samples = True
    assert studio.handle("x") is True
    assert studio.overlay == "compare"
    studio.handle("ESC")
    assert (studio.overlay, studio.comparison, studio.dialog_offset) == (None, None, 0)
    assert studio.data_panel.show_samples is False
    studio.overlay, studio.action_menu = "actions", None
    studio.handle("ENTER")
    assert studio.overlay == "actions"
    studio.handle("ESC")
    assert studio.overlay is None


def test_action_menu_runs_the_chosen_command(studio: tui.Studio) -> None:
    studio.handle(":")
    assert studio.overlay == "actions"
    studio.handle("END")
    studio.handle("ENTER")
    assert studio.overlay == "help"
