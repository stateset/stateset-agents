"""Compatibility entry point for :mod:`stateset_agents.data.grade_transcript`."""

from __future__ import annotations

from stateset_agents.data.grade_transcript import (  # noqa: F401
    get_reward,
    grade_transcript,
    load_contexts,
    load_transcript,
    main,
    render_markdown,
    write_curated_examples,
)

__all__ = [
    "get_reward",
    "grade_transcript",
    "load_contexts",
    "load_transcript",
    "main",
    "render_markdown",
    "write_curated_examples",
]


if __name__ == "__main__":
    raise SystemExit(main())
