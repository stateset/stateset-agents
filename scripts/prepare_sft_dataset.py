"""Compatibility entry point for :mod:`stateset_agents.data.prepare_sft_dataset`."""

from __future__ import annotations

from stateset_agents.data.prepare_sft_dataset import (  # noqa: F401
    FORMATTERS,
    filter_entries,
    load_curated,
    main,
    to_axolotl,
    to_chat,
    to_hf_trainer,
)

__all__ = [
    "FORMATTERS",
    "filter_entries",
    "load_curated",
    "main",
    "to_axolotl",
    "to_chat",
    "to_hf_trainer",
]


if __name__ == "__main__":
    raise SystemExit(main())
