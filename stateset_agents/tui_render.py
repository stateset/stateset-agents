"""Incremental curses rendering with terminal-native colors and cell widths."""

from __future__ import annotations

from typing import Any

from rich.cells import set_cell_size


class FrameRenderer:
    """Paint changed rows only, preserving the terminal's background palette."""

    def __init__(self, curses: Any, *, colors: bool) -> None:
        self.curses = curses
        self.colors = colors
        self.size: tuple[int, int] | None = None
        self.previous: list[tuple[str, int] | None] = []

    def style(self, line: str, row: int) -> int:
        """Use both color and text/weight to distinguish meaningful states."""
        c = self.curses
        text = line.strip()
        if line.startswith(" > "):
            return int((c.color_pair(2) if self.colors else c.A_REVERSE) | c.A_BOLD)
        for prefix, pair in (("[PASS]", 3), ("[WARNING]", 4), ("[FAIL]", 5)):
            if text.startswith(prefix):
                return int((c.color_pair(pair) if self.colors else 0) | c.A_BOLD)
        if row == 0:
            return int((c.color_pair(1) if self.colors else 0) | c.A_BOLD)
        if row == 1:
            return int(c.A_BOLD)
        if text and all(char in "-=─│| " for char in text):
            return int(c.A_DIM)
        if text and len(text) <= 76 and text.isupper():
            return int((c.color_pair(1) if self.colors else 0) | c.A_BOLD)
        if text.startswith(("Search:", "Filter:", "RUNS ")):
            return int(c.color_pair(1) if self.colors else c.A_BOLD)
        return 0

    def draw(self, screen: Any, frame: list[str], width: int, height: int) -> None:
        """Clear on resize and pad changed rows to remove stale trailing text."""
        if self.size != (width, height):
            screen.erase()
            self.previous = [None] * height
            self.size = (width, height)
        for row in range(height):
            line = frame[row] if row < len(frame) else ""
            style = self.style(line, row)
            value = (line, style)
            if self.previous[row] == value:
                continue
            try:
                # addstr takes the complete Unicode string; limiting encoded
                # byte counts with addnstr can split wide/combining characters.
                screen.addstr(row, 0, set_cell_size(line, max(0, width - 1)), style)
            except self.curses.error:
                self.previous[row] = None  # Retry after a racing resize.
            else:
                self.previous[row] = value


def configure_terminal(screen: Any, curses: Any) -> None:
    """Keep standalone Escape responsive while honoring an explicit ESCDELAY."""
    import os

    screen.keypad(True)
    screen.timeout(100)
    if "ESCDELAY" not in os.environ and hasattr(curses, "set_escdelay"):
        curses.set_escdelay(35)
