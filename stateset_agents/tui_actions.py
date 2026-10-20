"""Discoverable terminal actions for desktop and compact layouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    """An existing keyboard command with a human-readable label."""

    key: str
    label: str


CONFIGURE_ACTIONS = (
    Action("w", "Guided setup"),
    Action("m", "Choose model"),
    Action("b", "Choose compute backend"),
    Action("f", "Browse dataset files"),
    Action("v", "Check settings"),
    Action("u", "Inspect dataset"),
    Action("i", "Check runtime readiness"),
    Action("p", "Preview resources"),
    Action("r", "Review training launch"),
    Action("s", "Save draft"),
    Action("TAB", "Open retained runs"),
    Action("?", "Keyboard guide"),
)

RUN_ACTIONS = (
    Action("d", "Run details"),
    Action("l", "View logs"),
    Action("g", "View metrics"),
    Action("e", "Inspect quality evidence"),
    Action("j", "Evaluate run"),
    Action("z", "Review checkpoint recovery"),
    Action("r", "Review prepared launch"),
    Action("k", "Review local stop"),
    Action("c", "Reuse settings for a new run"),
    Action("a", "Pin / unpin comparison run"),
    Action("x", "Compare runs"),
)


class ActionMenu:
    """A scrollable menu delegating execution to existing reviewed workflows."""

    def __init__(self, tab: str, *, has_run: bool, searchable: bool = False) -> None:
        self.index = 0
        self.searchable = searchable
        self.query = ""
        self.actions: tuple[Action, ...] = (
            CONFIGURE_ACTIONS
            if tab == "configure"
            else (
                *(RUN_ACTIONS if has_run else ()),
                Action("/", "Search all retained runs"),
                Action("t", "Cycle status filter"),
                Action("BACKSPACE", "Clear run filters"),
                Action("TAB", "Configure training"),
                Action("?", "Keyboard guide"),
            )
        )

    def matches(self) -> tuple[Action, ...]:
        """Filter labels and shortcuts without invoking the underlying action."""
        words = self.query.casefold().split()
        return tuple(
            action
            for action in self.actions
            if all(word in f"{action.label} {action.key}".casefold() for word in words)
        )

    def handle(self, key: str) -> str | None:
        """Move selection or return the selected command on Enter."""
        if self.searchable:
            if key == "BACKSPACE":
                self.query = self.query[:-1]
                self.index = 0
            elif key == "CLEAR":
                self.query, self.index = "", 0
            elif len(key) == 1 and key.isprintable() and len(self.query) < 128:
                self.query += key
                self.index = 0
        actions = self.matches()
        if key == "ENTER":
            return actions[self.index].key if actions else None
        if key in {"UP", "DOWN", "PAGEUP", "PAGEDOWN", "HOME", "END"}:
            delta = 6 if key.startswith("PAGE") else 1
            target = self.index + (delta if key in {"DOWN", "PAGEDOWN"} else -delta)
            if key == "HOME":
                target = 0
            elif key == "END":
                target = len(actions) - 1
            self.index = max(0, min(len(actions) - 1, target))
        return None

    def lines(self, available: int) -> list[str]:
        """Keep the selected action visible even in the smallest terminal."""
        actions = self.matches()
        count = max(1, available - (3 if self.searchable else 2))
        start = max(0, min(self.index - count // 2, len(actions) - count))
        return [
            (
                f"COMMANDS {self.index + 1 if actions else 0}/{len(actions)}"
                if self.searchable
                else f"ACTIONS {self.index + 1}/{len(actions)}"
            ),
            *(
                ["Search: " + (self.query or "type a command...")]
                if self.searchable
                else []
            ),
            "",
            *[
                (" > " if index == self.index else "   ")
                + action.label
                + (f"  [{action.key}]" if self.searchable else "")
                for index, action in enumerate(actions)
                if start <= index < start + count
            ],
            *(
                ["No matching commands. Backspace edits; Ctrl+U clears."]
                if not actions
                else []
            ),
        ]
