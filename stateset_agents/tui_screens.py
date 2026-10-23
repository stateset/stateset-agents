"""Frame pieces for the fine-tuning studio: header, dialog bodies and scrolling.

:func:`render_body` looks up the open dialog (``Studio.overlay``) in
:data:`OVERLAY_BODIES`; a renderer returns ``None`` when its dialog has nothing
to show (no action menu, evaluation panel or prepared review), and the visible
tab is rendered instead. :func:`scroll_body` then keeps compact selections in
view and applies the dialog scroll offset.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from rich.cells import cell_len, set_cell_size

from stateset_agents.tui_text import _wrap
from stateset_agents.tui_views import _run_config

if TYPE_CHECKING:
    from stateset_agents.tui import Studio

#: Renders one dialog body for ``(studio, width, available rows)``.
BodyRenderer = Callable[["Studio", int, int], "list[str] | None"]

HELP_LINES = (
    "KEYBOARD",
    "",
    "Ctrl+P opens searchable Commands; type words, arrows select and Enter opens a workflow",
    ": opens Actions on Configure or Runs; arrows select and Enter opens a workflow",
    "Phone terminals: 32 columns x 16 rows minimum. Compact screens use 2 down, 8 up, 3 page down, 9 page up, 0 back and 4 for the main setup/run views, outside text editors.",
    "Digits remain literal in values, paths and filters. Enter is still required to confirm a reviewed launch.",
    "Up/Down choose a setting or run",
    "Enter edits a setting, opens run details, or confirms a reviewed launch",
    "Left/Right, Home/End move the text cursor; Delete removes a character",
    "Ctrl+U clears text; Esc discards an edit or closes a dialog",
    "a pins/unpins a reference run; x compares the selected run to it",
    "Comparison shows saved settings and latest metrics; it never ranks quality",
    "m searches the model catalog",
    "b selects local CUDA, RunPod, Modal or River (remote SFT)",
    "v lists all configuration issues; Enter jumps to the selected setting",
    "f browses JSON/JSONL datasets without reading their contents",
    "i checks local readiness in the background; it never contacts a provider",
    "u inspects dataset statistics; e explicitly reveals a bounded private sample",
    "w guides settings, dataset, runtime, review and quality with explicit next steps",
    "On Runs: e verifies/imports paired held-out quality evidence; k reviews a local stop request",
    "z inspects local recovery checkpoints; copying and launch require separate reviews",
    "j configures a local base/adapter/judge evaluation; review submits through Buildr",
    "Inside j: b compares completed checkpoint validations; g reopens the queued comparison",
    "Inside a checkpoint comparison: n reviews a nomination for separate fresh test cases",
    "Inside j: h reopens saved checkpoint nominations without creating another receipt",
    "p validates data and previews resources without provisioning",
    "r reviews exact inputs; on Runs it reviews the same prepared run",
    "s saves draft.json (credentials are never saved)",
    "Tab switches configuration / retained runs",
    "Enter or d opens run details; c copies settings into a NEW run",
    "/ searches runs; t cycles all/active/failed/succeeded; Backspace resets filters",
    "g shows measured progress and reported loss/reward/other metric samples",
    "l opens full logs; s switches sources, Space pauses, / filters, End follows",
    "PageUp/Down scrolls dialogs or recent logs; End follows log tail",
    "q exits the main screen; Ctrl+C exits from any screen",
    "",
    "Closing the studio preserves detached jobs. Reopen the same workspace to reconnect.",
    "Local training requires CUDA. RunPod SFT is paid and requires RUNPOD_API_KEY in the launching environment.",
    "The studio starts no job when opening, previewing, viewing, or copying settings.",
)

PLANNING_LINES = (
    "PREPARING PREVIEW",
    "",
    "Reading and validating the dataset in the background.",
    "Esc cancels this preview. The current read may finish in the background.",
    "Training requires a separate review and Enter confirmation.",
)

#: Dialogs whose body scrolls with ``Studio.dialog_offset``.
SCROLLED_OVERLAYS = frozenset(
    {
        "backends",
        "review",
        "prepared-review",
        "help",
        "details",
        "readiness",
        "metrics",
        "compare",
        "data",
        "workflow",
        "quality",
        "stop",
        "recovery",
        "recovery-review",
    }
)
#: Dialogs that scroll their ``> `` selection into view on compact screens.
FOLLOWED_OVERLAYS = frozenset({"backends", "workflow", "recovery"})
#: Evaluation modes whose selected row is kept visible.
EVALUATION_SELECTION_MODES = frozenset(
    {"form", "edit", "targets", "comparison", "nominations"}
)


def header_lines(studio: Studio, width: int) -> list[str]:
    """Title and tab line; wide screens add workspace context, compact ones shrink."""
    configure = studio.tab == "configure"
    if studio.compact_layout:
        return [
            "STATESET / FINE-TUNING",
            (
                "[CONFIGURE] Runs | 4 views"
                if configure
                else "Configure [RUNS] | 4 views"
            ),
        ]
    header = [
        "STATESET / FINE-TUNING STUDIO",
        ("[CONFIGURE] Runs" if configure else "Configure [RUNS]")
        + " | Tab switches views",
        "=" * width,
    ]
    if width >= 90:
        active = sum(
            row.get("state", {}).get("status") in {"queued", "running", "starting"}
            for row in studio.all_runs
        )
        context = f"{studio.workspace.name}  /  {active} active"
        header[0] = (
            set_cell_size(header[0], max(1, width - cell_len(context) - 2))
            + "  "
            + context
        )
        commands = "Ctrl+P Commands"
        header[1] = (
            set_cell_size(header[1], width - len(commands) - 2) + "  " + commands
        )
    return header


def _actions_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """The open action menu."""
    menu = studio.action_menu
    return menu.lines(available) if menu is not None else None


def _planning_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """Progress note while a preview validates in the background."""
    return _wrap(list(PLANNING_LINES), width)


def _help_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """The keyboard guide."""
    return _wrap(list(HELP_LINES), width)


def _models_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """The searchable model catalog."""
    return studio._model_lines(width, available)


def _backends_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """The compute backend chooser."""
    return _wrap(studio._backend_lines(), width)


def _validation_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """Configuration issues for the draft."""
    return studio._validation_lines(width, available)


def _files_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """The dataset or held-out cases file picker."""
    return studio.file_picker.lines(width, available)


def _run_search_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """The run search prompt with up to eight live matches."""
    matches = studio._filter_runs(studio.search)
    return _wrap(
        [
            "FILTER RUNS / ID, model, method, provider or status",
            "",
            "Search: " + studio.search,
            f"{len(matches)} matches in the current {studio.run_filter} status filter",
            "",
            *[
                row["id"] + " " + str(_run_config(row).get("model", ""))
                for row in matches[:8]
            ],
            "",
            "Enter applies; Esc preserves the current filter.",
        ],
        width,
    )


def _data_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """The dataset inspector."""
    return _wrap(studio.data_panel.lines(studio._data_settings_signature()), width)


def _workflow_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """The guided setup snapshot."""
    return _wrap(studio.workflow_panel.lines(studio._settings_signature()), width)


def _evaluation_selected_line(body: list[str], mode: str) -> int:
    """Index of the selected row (or the edit line) in an evaluation body."""
    selected = next(
        (index for index, line in enumerate(body) if line.lstrip().startswith("> ")),
        0,
    )
    if mode == "edit":
        selected = next(
            (index for index, line in enumerate(body) if line.startswith("Edit:")),
            selected,
        )
    return selected


def _evaluation_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """The evaluation editor, scrolled to keep its selection visible."""
    panel = studio.evaluation_panel
    if panel is None:
        return None
    body = _wrap(panel.lines(), width)
    if panel.mode in EVALUATION_SELECTION_MODES:
        selected = _evaluation_selected_line(body, panel.mode)
        if selected < panel.offset or selected >= panel.offset + available:
            panel.offset = max(0, selected - available // 2)
    panel.offset = min(panel.offset, max(0, len(body) - available))
    return body[panel.offset :]


def _quality_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """Recorded quality evidence for the captured run."""
    return _wrap(studio.quality_panel.lines(), width)


def _recovery_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """Recovery checkpoints for the captured run."""
    return _wrap(studio.recovery_panel.lines(studio.message), width)


def _recovery_review_body(
    studio: Studio, width: int, available: int
) -> list[str] | None:
    """Review of the selected recovery checkpoint."""
    return _wrap(studio.recovery_panel.review_lines(), width)


def _quality_path_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """The evidence path prompt."""
    return _wrap(
        [
            "IMPORT QUALITY EVIDENCE / private paired reports",
            "",
            "Path: " + studio.search,
            "Enter hashes completed weights and validates paired test reports in the background.",
            "No models or providers are called. Successful import writes private quality.json.",
        ],
        width,
    )


def _stop_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """Review of a stop request."""
    return _wrap(studio._stop_lines(), width)


def _readiness_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """Local readiness checks for the draft."""
    return _wrap(
        studio.readiness_panel.lines(studio._settings_signature(), studio.values),
        width,
    )


def _metrics_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """Reported metric history for the selected run."""
    return _wrap(studio._metric_lines(width), width)


def _logs_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """The log viewer."""
    viewer = studio.log_viewer
    if viewer is None:
        return ["No log source selected. Esc returns to Runs."]
    return viewer.lines(width, available)


def _review_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """Launch review of the previewed plan."""
    return _wrap(studio._review_lines(), width)


def _prepared_review_body(
    studio: Studio, width: int, available: int
) -> list[str] | None:
    """Launch review of a retained prepared run."""
    review = studio.prepared_review
    if not review:
        return None
    return _wrap(
        [
            "REVIEW PREPARED RUN / same ID, retained snapshot",
            "Run: " + review.run.name,
            "Plan SHA256: " + review.plan_sha256,
            "Enter launches this run; Esc leaves it prepared.",
            "",
            *studio._review_lines(review.plan),
        ],
        width,
    )


def _compare_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """The saved snapshot comparison."""
    return _wrap(studio._comparison_lines(), width)


def _details_body(studio: Studio, width: int, available: int) -> list[str] | None:
    """Details of the selected run."""
    return _wrap(studio._run_details(), width)


#: Dialog body renderers by ``Studio.overlay``; other overlays show the tab.
OVERLAY_BODIES: dict[str, BodyRenderer] = {
    "actions": _actions_body,
    "planning": _planning_body,
    "help": _help_body,
    "models": _models_body,
    "backends": _backends_body,
    "validation": _validation_body,
    "files": _files_body,
    "run-search": _run_search_body,
    "data": _data_body,
    "workflow": _workflow_body,
    "evaluation": _evaluation_body,
    "quality": _quality_body,
    "recovery": _recovery_body,
    "recovery-review": _recovery_review_body,
    "quality-path": _quality_path_body,
    "stop": _stop_body,
    "readiness": _readiness_body,
    "metrics": _metrics_body,
    "logs": _logs_body,
    "review": _review_body,
    "prepared-review": _prepared_review_body,
    "compare": _compare_body,
    "details": _details_body,
}


def render_body(studio: Studio, width: int, available: int) -> list[str]:
    """The open dialog's body, or the visible tab when no dialog renders."""
    renderer = OVERLAY_BODIES.get(studio.overlay or "")
    body = renderer(studio, width, available) if renderer is not None else None
    if body is not None:
        return body
    if studio.tab == "configure":
        return studio._configure_lines(width, available)
    return studio._run_lines(width, available)


def _follow_selection(studio: Studio, body: list[str], available: int) -> None:
    """Scroll a compact dialog once so its ``> `` selection is visible."""
    selected_line = next(
        (index for index, line in enumerate(body) if line.lstrip().startswith("> ")),
        None,
    )
    if selected_line is None:
        return
    if (
        selected_line < studio.dialog_offset
        or selected_line >= studio.dialog_offset + available
    ):
        studio.dialog_offset = max(0, selected_line - available // 2)
    studio._compact_follow_selection = False


def scroll_body(studio: Studio, body: list[str], available: int) -> list[str]:
    """Keep compact selections visible, then apply the dialog scroll offset."""
    if (
        studio.compact_layout
        and studio._compact_follow_selection
        and studio.overlay in FOLLOWED_OVERLAYS
    ):
        _follow_selection(studio, body, available)
    if studio.overlay in SCROLLED_OVERLAYS:
        studio.dialog_offset = min(studio.dialog_offset, max(0, len(body) - available))
        body = body[studio.dialog_offset :]
    return body
