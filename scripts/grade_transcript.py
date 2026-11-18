"""Compatibility entry point for :mod:`stateset_agents.data.grade_transcript`."""

from __future__ import annotations

from stateset_agents.data.grade_transcript import (  # noqa: F401
    assistant_turn_count,
    get_reward,
    grade_transcript,
    load_contexts,
    load_transcript,
    main,
    preceding_user_content,
    render_markdown,
    write_curated_examples,
)

__all__ = [
    "get_reward",
    "grade_transcript",
    "load_contexts",
    "load_transcript",
    "assistant_turn_count",
    "preceding_user_content",
    "main",
    "render_markdown",
    "write_curated_examples",
]


if __name__ == "__main__":
    raise SystemExit(main())
