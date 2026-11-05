"""JSON Lines splitting that matches how the repo writes JSONL.

Writers here use ``json.dumps(..., ensure_ascii=False)``, which leaves U+2028,
U+2029 and U+0085 unescaped inside strings — valid JSON. ``str.splitlines``
treats those (and ``\\v``, ``\\f``, ``\\x1c``-``\\x1e``) as line breaks, so
it cut such records in half and readers dropped or rejected valid rows.
JSON Lines separates records with ``"\\n"`` only (``"\\r\\n"`` tolerated);
every reader goes through :func:`jsonl_lines`.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["count_jsonl_records", "jsonl_lines", "read_jsonl_lines"]


def jsonl_lines(text: str) -> list[tuple[int, str]]:
    """``(line number, line)`` for each non-blank record line of ``text``.

    Splits on ``"\\n"`` only and drops one trailing ``"\\r"``; line numbers
    are 1-based physical lines, so diagnostics point at the right line.
    """
    lines = []
    for number, line in enumerate(text.split("\n"), 1):
        if line.endswith("\r"):
            line = line[:-1]
        if line.strip():
            lines.append((number, line))
    return lines


def read_jsonl_lines(path: str | Path) -> list[tuple[int, str]]:
    """:func:`jsonl_lines` of a UTF-8 file."""
    return jsonl_lines(Path(path).read_text(encoding="utf-8"))


def count_jsonl_records(text: str) -> int:
    """Number of non-blank record lines in JSONL ``text``."""
    return len(jsonl_lines(text))
