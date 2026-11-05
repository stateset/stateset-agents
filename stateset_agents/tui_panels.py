"""Self-contained studio overlays: each owns its state, keys and rendering.

Like :class:`~stateset_agents.finetune_evaluation_tui.EvaluationPanel`, a
panel exposes ``handle(key)`` and ``lines(width, available)``; the studio only
decides when a panel is open and what its result means.
"""

from __future__ import annotations

import os
from itertools import islice
from pathlib import Path

from stateset_agents.finetune_workspace import (
    LOG_SOURCES,
    LogSnapshot,
    read_log_snapshot,
)
from stateset_agents.tui_text import _clip, _wrap


class LogViewer:
    """A bounded, read-only tail of one run's private logs.

    Following re-reads the selected source on every frame; pausing freezes the
    snapshot so scrolling and filtering never chase new output.
    """

    def __init__(self, run: Path) -> None:
        self.run = run
        self.source = "auto"
        self.snapshot: LogSnapshot | None = None
        self.follow = True
        self.offset = 0
        self.query = ""
        #: The filter being typed, or ``None`` when not editing it.
        self.search_edit: str | None = None

    @property
    def editing(self) -> bool:
        """Whether keys are text for the filter editor."""
        return self.search_edit is not None

    def _pause(self) -> None:
        if self.snapshot is None:
            self.snapshot = read_log_snapshot(self.run, source=self.source)
        self.follow = False

    def handle(self, key: str) -> bool:
        """Apply one key; return true when the viewer should close."""
        if self.search_edit is not None:
            if key == "ESC":
                self.search_edit = None
            elif key == "ENTER":
                self.query = self.search_edit
                self.search_edit = None
                self.offset = 0
            elif key == "BACKSPACE":
                self.search_edit = self.search_edit[:-1]
            elif key == "CLEAR":
                self.search_edit = ""
            elif len(key) == 1 and key.isprintable() and len(self.search_edit) < 256:
                self.search_edit += key
            return False
        if key == "ESC":
            return True
        if key == "/":
            self._pause()
            self.search_edit = self.query
        elif key == "BACKSPACE":
            self.query = ""
            self.offset = 0
        elif key == "s":
            self.source = LOG_SOURCES[
                (LOG_SOURCES.index(self.source) + 1) % len(LOG_SOURCES)
            ]
            self.snapshot = read_log_snapshot(self.run, source=self.source)
            self.offset = 0
        elif key == "END":
            self.follow = True
            self.offset = 0
        elif key == " ":
            if self.follow:
                self._pause()
            else:
                self.follow = True
                self.offset = 0
        elif key == "HOME":
            self._pause()
            self.offset = 1000000
        elif key in {"UP", "DOWN", "PAGEUP", "PAGEDOWN"}:
            self._pause()
            delta = 12 if key in {"PAGEUP", "PAGEDOWN"} else 1
            self.offset = max(
                0,
                min(
                    1000000,
                    self.offset + (delta if key in {"UP", "PAGEUP"} else -delta),
                ),
            )
        return False

    BROWSE_KEYS = (
        "Space pause/follow | s source | / filter | Esc back",
        "Arrows/PgUp/Dn scroll | Home oldest | End follow | Backspace clear",
    )

    def keys(self, width: int) -> list[str]:
        """Footer hints for the current mode."""
        if self.search_edit is not None:
            return [
                "Filter: "
                + _clip(self.search_edit[-max(1, (width - 12) // 2) :], width - 8),
                "Enter apply | Esc discard | Backspace edit | Ctrl+U clear",
            ]
        return list(self.BROWSE_KEYS)

    def lines(self, width: int, available: int) -> list[str]:
        """Render the heading and the visible slice of matching lines."""
        if self.follow or self.snapshot is None:
            self.snapshot = read_log_snapshot(self.run, source=self.source)
        snapshot = self.snapshot
        retained = snapshot.lines[-2000:]
        query = self.query.casefold()
        selected = (
            [line for line in retained if query in line.casefold()]
            if query
            else list(retained)
        )
        heading = _wrap(
            [
                "RUN LOGS / long lines wrap",
                "Run: " + self.run.name,
                f"{'FOLLOW' if self.follow else 'PAUSED'} | {self.source} -> {snapshot.source} / {snapshot.filename}",
                f"Tail scope: 128 KiB / 2,000 lines | {len(selected)}/{len(retained)} match",
                "Filter: " + _clip(self.query or "(none)", max(1, width - 8)),
            ],
            width,
        )
        if snapshot.window_truncated:
            heading += _wrap(
                ["Earlier bytes omitted; the first retained line may be partial."],
                width,
            )
        if len(snapshot.lines) > 2000:
            heading += ["Earlier lines omitted from this view."]
        content = _wrap(selected, width)
        if not snapshot.available:
            content = [
                (
                    "Waiting for worker output..."
                    if snapshot.error == "missing"
                    else "Log unavailable; inspect permissions or select another source."
                )
            ]
        elif not content:
            content = _wrap(
                [
                    (
                        "No matching lines in this tail window; Backspace clears the filter."
                        if query
                        else "No log lines in the selected source yet."
                    )
                ],
                width,
            )
        count = max(1, available - len(heading) - 1)
        self.offset = min(self.offset, max(0, len(content) - count))
        end = len(content) - (0 if self.follow else self.offset)
        return [*heading, "", *content[max(0, end - count) : end]]


class FilePicker:
    """Browse directories for one dataset or case file without reading files.

    ``cases`` selects the held-out case picker: JSON only, and a file must not
    be reached through a symlink.
    """

    #: Directory entries read per listing; larger directories are truncated.
    MAX_ENTRIES = 2000

    def __init__(self, directory: Path, *, cases: bool = False) -> None:
        self.directory = directory
        self.cases = cases
        self.entries: list[tuple[str, Path, bool]] = []
        self.index = 0
        self.search = ""
        self.error = ""

    def read(self) -> None:
        """List the current directory, keeping ``..`` reachable on failure."""
        self.entries = []
        self.error = ""
        self.search = ""
        self.index = 0
        if self.directory.parent != self.directory:
            self.entries.append(("..", self.directory.parent, True))
        try:
            rows = []
            with os.scandir(self.directory) as entries:
                for entry in islice(entries, self.MAX_ENTRIES):
                    if entry.name.startswith("."):
                        continue
                    try:
                        directory = entry.is_dir()
                        if directory or (
                            entry.is_file(follow_symlinks=not self.cases)
                            and Path(entry.name).suffix.lower()
                            in ({".json"} if self.cases else {".json", ".jsonl"})
                        ):
                            rows.append((entry.name, Path(entry.path), directory))
                    except OSError:
                        continue
            self.entries += sorted(
                rows, key=lambda row: (not row[2], row[0].casefold())
            )
        except OSError as error:
            self.error = f"Cannot list this directory ({type(error).__name__}); select .. or Esc."

    def filtered(self) -> list[tuple[str, Path, bool]]:
        """Entries whose names contain the typed filter."""
        return [
            row for row in self.entries if self.search.casefold() in row[0].casefold()
        ]

    def handle(self, key: str) -> Path | None:
        """Apply one key; return the chosen file once one is selected."""
        entries = self.filtered()
        if key in {"UP", "DOWN", "PAGEUP", "PAGEDOWN"}:
            delta = 8 if key in {"PAGEUP", "PAGEDOWN"} else 1
            self.index = max(
                0,
                min(
                    len(entries) - 1,
                    self.index + (delta if key in {"DOWN", "PAGEDOWN"} else -delta),
                ),
            )
        elif key in {"HOME", "END"}:
            self.index = 0 if key == "HOME" else max(0, len(entries) - 1)
        elif key == "BACKSPACE":
            if self.search:
                self.search = self.search[:-1]
                self.index = 0
            else:
                self.directory = self.directory.parent
                self.read()
        elif key == "CLEAR":
            self.search = ""
            self.index = 0
        elif key == "ENTER" and entries:
            _, path, directory = entries[min(self.index, len(entries) - 1)]
            if not directory:
                return path
            self.directory = path.resolve()
            self.read()
        elif len(key) == 1 and key.isprintable():
            self.search += key
            self.index = 0
        return None

    def lines(self, width: int, available: int) -> list[str]:
        """Render the listing around the selection, with the selected path."""
        header = _wrap(
            [
                (
                    "HELD-OUT CASE PICKER / JSON files"
                    if self.cases
                    else "DATASET PICKER / JSON and JSONL files"
                ),
                str(self.directory),
                "Filter: " + self.search,
            ],
            width,
        )
        entries = self.filtered()
        self.index = max(0, min(self.index, len(entries) - 1))
        selected = (
            _wrap(["Selected: " + str(entries[self.index][1])], width)
            if entries
            else ["No matching files; Backspace goes to the parent directory."]
        )
        count = max(1, available - len(header) - len(selected) - 2)
        start = max(0, min(self.index - count // 2, len(entries) - count))
        rows = [
            (" > " if index == self.index else "   ")
            + _clip(name + ("/" if directory else ""), width - 4)
            for index, (name, _, directory) in enumerate(entries)
            if start <= index < start + count
        ]
        rows += [""] * max(0, count - len(rows))
        return [
            *header,
            *rows,
            "",
            *selected,
            self.error
            or "Lists at most 2,000 directory entries; selection does not read file contents.",
        ]
