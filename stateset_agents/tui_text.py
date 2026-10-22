"""Terminal-cell-safe text helpers shared by the studio and its panels."""

from __future__ import annotations

import textwrap
from typing import Any

from rich.cells import cell_len, chop_cells, set_cell_size


def _safe(value: Any) -> str:
    return "".join(char if char.isprintable() else " " for char in str(value))


def _clip(value: Any, width: int) -> str:
    text = _safe(value)
    if cell_len(text) <= width:
        return text
    if width < 4:
        return str(set_cell_size(text, max(0, width)))
    return str(set_cell_size(text, width - 3)) + "..."


def _wrap(lines: list[str], width: int) -> list[str]:
    result: list[str] = []
    for line in lines:
        parts = textwrap.wrap(
            _safe(line),
            width=max(1, width),
            break_long_words=False,
            break_on_hyphens=False,
        )
        for part in parts or [""]:
            result.extend(chop_cells(part, max(1, width)) or [""])
    return result
