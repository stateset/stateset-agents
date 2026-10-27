"""Keyboard bindings for the fine-tuning studio, as key -> handler tables.

:func:`handle_key` maps compact-layout digits, picks the input mode that owns
the key (see :func:`key_mode` for the precedence) and applies that mode's
binding. Handlers change studio state or call studio methods. Work that
resolves a patchable module-level name of :mod:`stateset_agents.tui` (launch,
stop, saving the draft, building the action menu, refreshing history) stays a
``Studio`` method there, so the handlers here only delegate to it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from stateset_agents.finetune_workspace import PROVIDERS

if TYPE_CHECKING:
    from stateset_agents.tui import Field, Studio

#: A binding: receives the studio and the (normalized) key that triggered it.
KeyHandler = Callable[["Studio", str], None]
#: An input mode: applies a key and returns false only when the user exits.
ModeHandler = Callable[["Studio", str], bool]

#: Compact layouts treat phone digits as navigation outside text editors.
COMPACT_DIGITS = {
    "2": "DOWN",
    "8": "UP",
    "3": "PAGEDOWN",
    "9": "PAGEUP",
    "0": "ESC",
}
#: Keys after which compact dialogs scroll the selected line back into view.
FOLLOW_SELECTION_KEYS = frozenset({"UP", "DOWN", "b", "w", "z"})
#: Upper bound for scroll offsets that renderers do not clamp themselves.
MAX_DIALOG_OFFSET = 1000000
#: Status filters cycled by ``t`` on the Runs tab.
RUN_FILTERS = ("all", "active", "failed", "succeeded")
#: Longest evidence path accepted by the quality import prompt.
MAX_EVIDENCE_PATH = 4096

ARROWS = ("UP", "DOWN")
PAGES = ("PAGEUP", "PAGEDOWN")
SCROLL_KEYS = ARROWS + PAGES


@dataclass(frozen=True)
class KeyMap:
    """Bindings for one input mode, plus an optional handler for other keys."""

    keys: Mapping[str, KeyHandler]
    fallback: KeyHandler | None = None

    def dispatch(self, studio: Studio, key: str) -> None:
        """Apply the key's binding, or the fallback when the key is unbound."""
        handler = self.keys.get(key, self.fallback)
        if handler is not None:
            handler(studio, key)


def _bind(*bindings: tuple[str | tuple[str, ...], KeyHandler]) -> dict[str, KeyHandler]:
    """Expand ``(key or keys, handler)`` pairs into a key -> handler table."""
    table: dict[str, KeyHandler] = {}
    for keys, handler in bindings:
        for key in (keys,) if isinstance(keys, str) else keys:
            table[key] = handler
    return table


def _direction(key: str) -> int:
    """+1 for keys that move forward (Down, PageDown, Right), otherwise -1."""
    return 1 if key in {"DOWN", "PAGEDOWN", "RIGHT"} else -1


def _scroll_delta(key: str) -> int:
    """Signed dialog movement: arrows move one line, pages move eight."""
    return _direction(key) * (8 if key in PAGES else 1)


def _printable(key: str) -> bool:
    """Whether a key is one literal character for a text field."""
    return len(key) == 1 and key.isprintable()


def _editor_text(studio: Studio) -> str:
    """The text being edited; the editing mode guarantees an open editor."""
    return studio.editing or ""


# Shared bindings -------------------------------------------------------------


def _scroll_dialog(studio: Studio, key: str) -> None:
    """Scroll the dialog body; renderers clamp the end."""
    studio.dialog_offset = max(0, studio.dialog_offset + _scroll_delta(key))


def _scroll_dialog_bounded(studio: Studio, key: str) -> None:
    """Scroll the dialog body within ``[0, MAX_DIALOG_OFFSET]``."""
    studio.dialog_offset = max(
        0, min(MAX_DIALOG_OFFSET, studio.dialog_offset + _scroll_delta(key))
    )


def _search_backspace(studio: Studio, key: str) -> None:
    """Delete the last character of the search or path prompt."""
    studio.search = studio.search[:-1]


def _search_clear(studio: Studio, key: str) -> None:
    """Clear the search or path prompt."""
    studio.search = ""


def _inspect(studio: Studio, key: str) -> None:
    """Open local readiness checks for the draft settings."""
    studio._inspect()


def _inspect_data(studio: Studio, key: str) -> None:
    """Open (or rescan) the dataset inspector."""
    studio._inspect_data()


def _check_settings(studio: Studio, key: str) -> None:
    """List configuration issues for the draft settings."""
    studio._check_settings()


def _open_workflow(studio: Studio, key: str) -> None:
    """Open (or recheck) the guided setup snapshot."""
    studio._open_workflow()


def _open_evaluation(studio: Studio, key: str) -> None:
    """Open the local evaluation editor for the selected run."""
    studio._open_evaluation()


def _open_quality(studio: Studio, key: str) -> None:
    """Open quality evidence for the selected run."""
    studio._open_quality()


def _open_recovery(studio: Studio, key: str) -> None:
    """Inspect recovery checkpoints for the selected run."""
    studio._open_recovery()


def _review_prepared(studio: Studio, key: str) -> None:
    """Review the selected prepared run."""
    studio._review_prepared()


def _review_stop(studio: Studio, key: str) -> None:
    """Review a stop request for the selected run."""
    studio._review_stop()


def _pin_reference(studio: Studio, key: str) -> None:
    """Pin or unpin the selected run as the comparison reference."""
    studio._pin_reference()


def _open_comparison(studio: Studio, key: str) -> None:
    """Compare the selected run with the pinned reference."""
    studio._open_comparison()


def _reuse_run(studio: Studio, key: str) -> None:
    """Copy the selected run's settings into a new draft."""
    studio._reuse_run()


# Modes that take precedence over ordinary dialogs ----------------------------


def _planning_mode(studio: Studio, key: str) -> bool:
    """While a preview runs only Esc acts: it cancels and discards the preview."""
    if key == "ESC":
        studio.planning.cancel()
        studio.overlay = None
        studio.message = (
            "Preview cancelled. Its result will be discarded; no run was created."
        )
    return True


def _actions_mode(studio: Studio, key: str) -> bool:
    """Esc closes the menu; a chosen command runs as if its key were pressed."""
    menu = studio.action_menu
    if key == "ESC" or menu is None:
        studio.overlay = None
        return True
    command = menu.handle(key)
    if command is None:
        return True
    studio.overlay = None
    if studio.action_run_id is not None and (
        not studio.runs or studio.runs[studio.run_index]["id"] != studio.action_run_id
    ):
        studio.message = "Selected run changed. Reopen : actions for the current run."
        return True
    return studio.handle(command)


def _quality_path_back(studio: Studio, key: str) -> None:
    """Return to the quality dialog without importing."""
    studio.overlay = "quality"


def _quality_path_verify(studio: Studio, key: str) -> None:
    """Return to the quality dialog and verify/import a non-blank path."""
    studio.overlay = "quality"
    if studio.search.strip():
        studio._scan_quality(Path(studio.search))


def _quality_path_type(studio: Studio, key: str) -> None:
    """Append a literal character to the evidence path, up to its limit."""
    if _printable(key) and len(studio.search) < MAX_EVIDENCE_PATH:
        studio.search += key


QUALITY_PATH_KEYS = KeyMap(
    _bind(
        ("ESC", _quality_path_back),
        ("ENTER", _quality_path_verify),
        ("BACKSPACE", _search_backspace),
        ("CLEAR", _search_clear),
    ),
    fallback=_quality_path_type,
)


def _quality_path_mode(studio: Studio, key: str) -> bool:
    """The evidence path prompt owns every key until Esc or Enter."""
    QUALITY_PATH_KEYS.dispatch(studio, key)
    return True


def _edit_discard(studio: Studio, key: str) -> None:
    """Close the editor without saving."""
    studio.editing = None


def _edit_save(studio: Studio, key: str) -> None:
    """Store the edited text in the selected setting and drop the old preview."""
    name = studio.fields()[studio.field].name
    if name == "model":
        studio._set_model(_editor_text(studio))
    else:
        studio.values[name] = studio.editing
    studio.editing = None
    studio.plan = None


def _edit_backspace(studio: Studio, key: str) -> None:
    """Delete the character before the cursor."""
    if studio.edit_cursor:
        text = _editor_text(studio)
        studio.editing = text[: studio.edit_cursor - 1] + text[studio.edit_cursor :]
        studio.edit_cursor -= 1


def _edit_delete(studio: Studio, key: str) -> None:
    """Delete the character under the cursor."""
    text = _editor_text(studio)
    studio.editing = text[: studio.edit_cursor] + text[studio.edit_cursor + 1 :]


def _edit_move(studio: Studio, key: str) -> None:
    """Move the cursor one character left or right, within the text."""
    studio.edit_cursor = max(
        0, min(len(_editor_text(studio)), studio.edit_cursor + _direction(key))
    )


def _edit_home(studio: Studio, key: str) -> None:
    """Move the cursor to the start of the text."""
    studio.edit_cursor = 0


def _edit_end(studio: Studio, key: str) -> None:
    """Move the cursor to the end of the text."""
    studio.edit_cursor = len(_editor_text(studio))


def _edit_clear(studio: Studio, key: str) -> None:
    """Clear the edited text."""
    studio.editing = ""
    studio.edit_cursor = 0


def _edit_type(studio: Studio, key: str) -> None:
    """Insert a literal character at the cursor."""
    if _printable(key):
        text = _editor_text(studio)
        studio.editing = text[: studio.edit_cursor] + key + text[studio.edit_cursor :]
        studio.edit_cursor += 1


EDIT_KEYS = KeyMap(
    _bind(
        ("ESC", _edit_discard),
        ("ENTER", _edit_save),
        ("BACKSPACE", _edit_backspace),
        ("DELETE", _edit_delete),
        (("LEFT", "RIGHT"), _edit_move),
        ("HOME", _edit_home),
        ("END", _edit_end),
        ("CLEAR", _edit_clear),
    ),
    fallback=_edit_type,
)


def _editing_mode(studio: Studio, key: str) -> bool:
    """The setting editor owns every key until Esc or Enter."""
    EDIT_KEYS.dispatch(studio, key)
    return True


def _evaluation_mode(studio: Studio, key: str) -> bool:
    """The evaluation editor owns its keys; f picks held-out cases in its form."""
    panel = studio.evaluation_panel
    if panel is None:  # key_mode only picks this mode while a panel is open
        return True
    if key == "f" and panel.mode == "form" and not panel.busy:
        studio._open_files(evaluation=panel)
    elif panel.handle(key):
        studio.overlay = None
    return True


# Dialog bindings ---------------------------------------------------------------


def _workflow_move(studio: Studio, key: str) -> None:
    """Select the previous or next guided setup step."""
    studio.workflow_panel.move(_direction(key))


def _workflow_open(studio: Studio, key: str) -> None:
    """Open the selected guided setup step."""
    studio._workflow_action()


def _recovery_recheck(studio: Studio, key: str) -> None:
    """Reinspect the captured run's checkpoints unless work is running."""
    panel = studio.recovery_panel
    if not panel.busy:
        studio._open_recovery(panel.run)


def _recovery_move(studio: Studio, key: str) -> None:
    """Select the previous or next checkpoint of a finished inspection."""
    panel = studio.recovery_panel
    if not panel.busy and panel.report:
        panel.move(_direction(key))


def _recovery_select(studio: Studio, key: str) -> None:
    """Review the selected available checkpoint of a finished inspection."""
    panel = studio.recovery_panel
    if not panel.busy and panel.report and panel.select():
        studio.overlay, studio.dialog_offset = "recovery-review", 0


def _recovery_prepare(studio: Studio, key: str) -> None:
    """Copy the reviewed checkpoint into a new prepared run."""
    studio._prepare_recovery()


def _stop_confirm(studio: Studio, key: str) -> None:
    """Submit the reviewed stop request."""
    studio._confirm_stop()


def _quality_import(studio: Studio, key: str) -> None:
    """Prompt for a paired evidence path unless inspection is running."""
    if not studio.quality_panel.busy:
        studio.overlay, studio.search = "quality-path", ""


def _quality_rescan(studio: Studio, key: str) -> None:
    """Reverify the captured run's recorded quality evidence."""
    studio._scan_quality()


def _files_pick(studio: Studio, key: str) -> None:
    """Let the file picker handle the key and return a chosen path."""
    chosen = studio.file_picker.handle(key)
    if chosen is not None:
        studio._close_files(chosen)


def _validation_fix(studio: Studio, key: str) -> None:
    """Jump to the setting behind the selected issue."""
    studio._fix_setting()


def _validation_move(studio: Studio, key: str) -> None:
    """Select another configuration issue."""
    studio.validation_index = max(
        0,
        min(
            len(studio.validation_issues) - 1,
            studio.validation_index + _scroll_delta(key),
        ),
    )


def _backend_move(studio: Studio, key: str) -> None:
    """Select the previous or next compute backend (wrapping)."""
    studio.backend_index = (studio.backend_index + _direction(key)) % len(PROVIDERS)
    studio.dialog_offset = 0


def _backend_choose(studio: Studio, key: str) -> None:
    """Apply the selected backend and close the dialog."""
    studio._set_backend(PROVIDERS[studio.backend_index])
    studio.overlay = None


def _data_toggle_samples(studio: Studio, key: str) -> None:
    """Show or hide the bounded private samples of a usable report."""
    if studio.data_panel.toggle_samples():
        studio.dialog_offset = 0


def _data_step_sample(studio: Studio, key: str) -> None:
    """Show the previous or next private sample of a report."""
    if studio.data_panel.report and studio.data_panel.step_sample(_direction(key)):
        studio.dialog_offset = 0


def _run_search_apply(studio: Studio, key: str) -> None:
    """Apply the typed run search and reselect from the first match."""
    studio.run_query = studio.search
    studio.overlay = None
    studio.run_index = 0
    studio.refresh()


def _run_search_type(studio: Studio, key: str) -> None:
    """Append a literal character to the run search."""
    if _printable(key):
        studio.search += key


def _model_move(studio: Studio, key: str) -> None:
    """Select the previous or next matching model."""
    studio.model_index = max(
        0,
        min(
            max(0, len(studio.filtered_models()) - 1),
            studio.model_index + _direction(key),
        ),
    )


def _model_choose(studio: Studio, key: str) -> None:
    """Use the selected matching model and close the catalog."""
    if studio.filtered_models():
        studio._set_model(studio.filtered_models()[studio.model_index][0])
        studio.overlay = None
        studio.plan = None


def _model_backspace(studio: Studio, key: str) -> None:
    """Delete the last search character and reselect the first match."""
    studio.search = studio.search[:-1]
    studio.model_index = 0


def _model_type(studio: Studio, key: str) -> None:
    """Append a literal search character and reselect the first match."""
    if _printable(key):
        studio.search += key
        studio.model_index = 0


def _prepared_launch(studio: Studio, key: str) -> None:
    """Launch the reviewed prepared run."""
    studio._launch_prepared()


def _review_launch(studio: Studio, key: str) -> None:
    """Launch the reviewed plan, when a preview produced one."""
    studio._launch_reviewed()


def _details_reuse(studio: Studio, key: str) -> None:
    """Copy the run's settings into a new draft and close the details."""
    studio._reuse_run()
    studio.overlay = None


def _details_logs(studio: Studio, key: str) -> None:
    """Open the run's logs."""
    studio._open_logs()


SCROLL_BINDINGS = _bind((SCROLL_KEYS, _scroll_dialog_bounded))

#: Bindings per dialog (``Studio.overlay``). Esc is handled before these for
#: every dialog except ``logs``; dialogs absent here only respond to Esc.
OVERLAY_KEYS: dict[str, KeyMap] = {
    "workflow": KeyMap(
        _bind(
            ("w", _open_workflow),
            (ARROWS, _workflow_move),
            ("ENTER", _workflow_open),
            (PAGES, _scroll_dialog),
        )
    ),
    "recovery": KeyMap(
        _bind(
            ("z", _recovery_recheck),
            (ARROWS, _recovery_move),
            ("ENTER", _recovery_select),
            (PAGES, _scroll_dialog),
        )
    ),
    "recovery-review": KeyMap({**SCROLL_BINDINGS, "ENTER": _recovery_prepare}),
    "stop": KeyMap({**SCROLL_BINDINGS, "ENTER": _stop_confirm}),
    "quality": KeyMap(
        _bind(
            ("e", _quality_rescan),
            ("j", _open_evaluation),
            ("f", _quality_import),
            (SCROLL_KEYS, _scroll_dialog),
        )
    ),
    "files": KeyMap({}, fallback=_files_pick),
    "validation": KeyMap(
        _bind(
            ("v", _check_settings),
            ("ENTER", _validation_fix),
            (SCROLL_KEYS, _validation_move),
        )
    ),
    "backends": KeyMap(
        _bind(
            (ARROWS, _backend_move),
            (PAGES, _scroll_dialog_bounded),
            ("ENTER", _backend_choose),
        )
    ),
    "data": KeyMap(
        _bind(
            ("u", _inspect_data),
            ("e", _data_toggle_samples),
            (("LEFT", "RIGHT"), _data_step_sample),
            (SCROLL_KEYS, _scroll_dialog_bounded),
        )
    ),
    "readiness": KeyMap({**SCROLL_BINDINGS, "i": _inspect}),
    "run-search": KeyMap(
        _bind(
            ("ENTER", _run_search_apply),
            ("BACKSPACE", _search_backspace),
            ("CLEAR", _search_clear),
        ),
        fallback=_run_search_type,
    ),
    "review": KeyMap({**SCROLL_BINDINGS, "ENTER": _review_launch}),
    "prepared-review": KeyMap({**SCROLL_BINDINGS, "ENTER": _prepared_launch}),
    "help": KeyMap(SCROLL_BINDINGS),
    "metrics": KeyMap(SCROLL_BINDINGS),
    "compare": KeyMap(SCROLL_BINDINGS),
    "details": KeyMap(
        {
            **SCROLL_BINDINGS,
            **_bind(
                ("c", _details_reuse),
                ("r", _review_prepared),
                ("l", _details_logs),
                ("a", _pin_reference),
                ("x", _open_comparison),
                ("e", _open_quality),
                ("j", _open_evaluation),
                ("k", _review_stop),
                ("z", _open_recovery),
            ),
        }
    ),
    "models": KeyMap(
        _bind(
            (ARROWS, _model_move),
            ("ENTER", _model_choose),
            ("BACKSPACE", _model_backspace),
        ),
        fallback=_model_type,
    ),
}


def _logs_key(studio: Studio, key: str) -> None:
    """The log viewer owns its keys; without one only Esc closes the dialog."""
    viewer = studio.log_viewer
    if viewer is None:
        if key == "ESC":
            studio.overlay = None
    elif viewer.handle(key):
        studio.overlay, studio.log_viewer = None, None


def _close_dialog(studio: Studio) -> None:
    """Esc: the file picker returns to its caller; other dialogs reset state."""
    if studio.overlay == "files":
        studio._close_files()
        return
    studio.overlay = None
    studio.prepared_review = None
    studio.comparison = None
    studio.data_panel.show_samples = False
    studio.dialog_offset = 0


def _overlay_mode(studio: Studio, key: str) -> bool:
    """Route a key to the open dialog; Esc closes every dialog except logs."""
    if studio.overlay == "logs":
        _logs_key(studio, key)
    elif key == "ESC":
        _close_dialog(studio)
    else:
        keymap = OVERLAY_KEYS.get(studio.overlay or "")
        if keymap is not None:
            keymap.dispatch(studio, key)
    return True


# Main screen -------------------------------------------------------------------


def _toggle_tab(studio: Studio, key: str) -> None:
    """Switch between the Configure and Runs tabs."""
    studio.tab = "runs" if studio.tab == "configure" else "configure"
    studio.log_offset = 0


def _open_actions(studio: Studio, key: str) -> None:
    """Open the action menu; Ctrl+P opens it searchable."""
    studio._open_actions(searchable=key == "COMMANDS")


def _open_help(studio: Studio, key: str) -> None:
    """Open the keyboard guide."""
    studio.overlay = "help"
    studio.dialog_offset = 0


#: Keys that act the same on both tabs of the main screen.
GLOBAL_KEYS = _bind(
    ("TAB", _toggle_tab),
    ((":", "COMMANDS"), _open_actions),
    ("?", _open_help),
    ("i", _inspect),
)


def _field_move(studio: Studio, key: str) -> None:
    """Select the previous or next visible setting (wrapping)."""
    studio.field = (studio.field + _direction(key)) % len(studio.fields())


def _field_jump(studio: Studio, key: str) -> None:
    """Select the first or last visible setting."""
    studio.field = 0 if key == "HOME" else len(studio.fields()) - 1


def _field_page(studio: Studio, key: str) -> None:
    """Move the setting selection by one page."""
    studio.field = max(
        0,
        min(
            len(studio.fields()) - 1,
            studio.field + studio.field_page_size * _direction(key),
        ),
    )


def _cycle_choice(studio: Studio, field: Field, step: int) -> None:
    """Select the previous or next choice of a setting with fixed options."""
    index = field.choices.index(str(studio.values[field.name]))
    value = field.choices[(index + step) % len(field.choices)]
    if field.name == "provider":
        studio._set_backend(value)
    else:
        studio.values[field.name] = value
    studio.field = min(studio.field, len(studio.fields()) - 1)
    studio.plan = None


def _field_change(studio: Studio, key: str) -> None:
    """Cycle a setting with choices; Enter opens the editor for free text."""
    field = studio._selected_field()
    if field.choices:
        _cycle_choice(studio, field, -1 if key == "LEFT" else 1)
    elif key == "ENTER":
        studio.editing = str(studio.values[field.name])
        studio.edit_cursor = len(studio.editing)


def _open_models(studio: Studio, key: str) -> None:
    """Open the model catalog with an empty search."""
    studio.overlay, studio.search, studio.model_index = "models", "", 0


def _open_backends(studio: Studio, key: str) -> None:
    """Open the backend chooser on the current provider."""
    studio.backend_index = PROVIDERS.index(str(studio.values["provider"]))
    studio.overlay = "backends"
    studio.dialog_offset = 0


def _open_dataset_picker(studio: Studio, key: str) -> None:
    """Browse for a dataset file; a listing failure stays on the main screen."""
    try:
        studio._open_files()
    except (OSError, ValueError, RuntimeError) as error:
        studio.overlay = None
        studio.message = f"Cannot open dataset picker ({type(error).__name__})."


def _preview(studio: Studio, key: str) -> None:
    """Preview resources (p) or preview and then review the launch (r)."""
    studio.preview(review=key == "r")


def _save_draft(studio: Studio, key: str) -> None:
    """Save the draft settings to the workspace."""
    studio._save_draft()


#: Configure-tab keys, applied after the selected setting is clamped.
CONFIGURE_KEYS = _bind(
    (ARROWS, _field_move),
    (("HOME", "END"), _field_jump),
    (PAGES, _field_page),
    (("ENTER", "LEFT", "RIGHT"), _field_change),
    ("w", _open_workflow),
    ("m", _open_models),
    ("u", _inspect_data),
    ("v", _check_settings),
    ("b", _open_backends),
    ("f", _open_dataset_picker),
    (("p", "r"), _preview),
    ("s", _save_draft),
)


def _run_move(studio: Studio, key: str) -> None:
    """Select the previous or next retained run and follow its log tail."""
    studio.run_index = max(
        0, min(max(0, len(studio.runs) - 1), studio.run_index + _direction(key))
    )
    studio.log_offset = 0


def _log_page(studio: Studio, key: str) -> None:
    """Page the recent log: PageUp goes back in time, PageDown towards the tail."""
    studio.log_offset = max(0, studio.log_offset + (12 if key == "PAGEUP" else -12))


def _run_first(studio: Studio, key: str) -> None:
    """Select the newest run and follow its log tail."""
    studio.run_index, studio.log_offset = 0, 0


def _log_follow(studio: Studio, key: str) -> None:
    """Follow the selected run's log tail."""
    studio.log_offset = 0


def _run_details(studio: Studio, key: str) -> None:
    """Open the selected run's details."""
    if studio.runs:
        studio.overlay = "details"
        studio.dialog_offset = 0


def _run_search_open(studio: Studio, key: str) -> None:
    """Edit the run search, starting from the applied query."""
    studio.overlay = "run-search"
    studio.search = studio.run_query


def _run_filter_cycle(studio: Studio, key: str) -> None:
    """Cycle the status filter and reload the run list."""
    studio.run_filter = RUN_FILTERS[
        (RUN_FILTERS.index(studio.run_filter) + 1) % len(RUN_FILTERS)
    ]
    studio.refresh()


def _run_filters_clear(studio: Studio, key: str) -> None:
    """Reset the search and status filter and reload the run list."""
    studio.run_query, studio.run_filter = "", "all"
    studio.refresh()


def _run_metrics(studio: Studio, key: str) -> None:
    """Open the selected run's metric history."""
    if studio.runs:
        studio.overlay = "metrics"
        studio.dialog_offset = 0


def _run_logs(studio: Studio, key: str) -> None:
    """Open the selected run's logs."""
    if studio.runs:
        studio._open_logs()


#: Runs-tab keys (every tab other than Configure).
RUNS_KEYS = _bind(
    (ARROWS, _run_move),
    (PAGES, _log_page),
    ("HOME", _run_first),
    ("END", _log_follow),
    (("ENTER", "d"), _run_details),
    ("c", _reuse_run),
    ("r", _review_prepared),
    ("/", _run_search_open),
    ("t", _run_filter_cycle),
    ("BACKSPACE", _run_filters_clear),
    ("g", _run_metrics),
    ("l", _run_logs),
    ("a", _pin_reference),
    ("x", _open_comparison),
    ("e", _open_quality),
    ("j", _open_evaluation),
    ("k", _review_stop),
    ("z", _open_recovery),
)


def _main_mode(studio: Studio, key: str) -> bool:
    """Main screen: q exits; global keys first, then the visible tab's keys."""
    if key in {"q", "QUIT"}:
        return False
    handler = GLOBAL_KEYS.get(key)
    if handler is None and studio.tab == "configure":
        studio._selected_field()
        handler = CONFIGURE_KEYS.get(key)
    elif handler is None:
        handler = RUNS_KEYS.get(key)
    if handler is not None:
        handler(studio, key)
    return True


# Dispatch ----------------------------------------------------------------------

#: Input modes by name; :func:`key_mode` decides which one owns a key.
MODE_HANDLERS: dict[str, ModeHandler] = {
    "planning": _planning_mode,
    "actions": _actions_mode,
    "quality-path": _quality_path_mode,
    "editing": _editing_mode,
    "evaluation": _evaluation_mode,
    "overlay": _overlay_mode,
    "main": _main_mode,
}


def key_mode(studio: Studio) -> str:
    """Name the input mode that owns the next key, in precedence order."""
    if studio.overlay == "planning":
        return "planning"
    if studio.overlay == "actions" and studio.action_menu is not None:
        return "actions"
    if studio.overlay == "quality-path":
        return "quality-path"
    if studio.editing is not None:
        return "editing"
    if studio.overlay == "evaluation" and studio.evaluation_panel is not None:
        return "evaluation"
    if studio.overlay:
        return "overlay"
    return "main"


def compact_key(studio: Studio, key: str) -> str:
    """Map phone digits to navigation in compact layouts, outside text editors."""
    if studio.compact_layout and not studio._text_input_active():
        if key == "4" and studio.overlay is None:
            key = "TAB"
        key = COMPACT_DIGITS.get(key, key)
    return key


def handle_key(studio: Studio, key: str) -> bool:
    """Apply a normalized key; return false only when the user exits."""
    if key == "QUIT":
        return False
    key = compact_key(studio, key)
    if key in FOLLOW_SELECTION_KEYS:
        studio._compact_follow_selection = True
    return MODE_HANDLERS[key_mode(studio)](studio, key)
