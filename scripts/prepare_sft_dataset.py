"""
Convert a curated JSONL (from ``grade_transcript.py --output-curated``) into
the SFT dataset format your training tool of choice expects.

The curated file has one record per line::

    {"prompt": "...", "response": "...", "messages": [...],
     "score": 0.83, "source": "session1.jsonl"}

``messages`` (written by current ``grade_transcript.py``) is the full
conversation the response was graded in, ending with the response; older
files carry only ``prompt``/``response`` and are treated as single-turn.

This script reshapes those into one of three target formats:

* ``--format hf-trainer`` — Hugging Face ``Trainer`` text-style:
    ``{"text": "<prompt>\\n\\n<response>"}`` (single-turn: plain text cannot
    keep earlier turns out of the loss)

* ``--format chat`` — OpenAI / chat-template style:
    ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``
    (the format ``apply_chat_template`` consumes; also what TRL's SFTTrainer
    expects for chat models) — the full graded context, earlier assistant
    replies marked ``"weight": 0`` so only the graded reply is trained

* ``--format axolotl`` — axolotl/unsloth-friendly ``alpaca`` style:
    ``{"instruction": "...", "input": "<earlier turns>", "output": "..."}``

``--flatten`` reproduces the legacy single-turn ``(prompt, response)`` pairs.

The script also supports filtering by source/min-score and deduplicating by
context — useful when curating across many sessions. ``--min-score`` drops
entries that carry no score (they are counted, never assumed to pass).

Usage::

    python scripts/prepare_sft_dataset.py \\
        --input curated.jsonl \\
        --format chat \\
        --output sft_train.jsonl \\
        --min-score 0.7 \\
        --dedup
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stateset_agents.data.jsonl import read_jsonl_lines

logger = logging.getLogger("prepare_sft_dataset")


def entry_messages(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """The conversation an entry was graded in, ending with its response.

    Every earlier assistant reply carries ``"weight": 0``: it is context (it
    may have scored below the threshold, and each passing reply is its own
    entry), so only the graded response is a training target. Entries
    without ``messages`` (older curated files) are single-turn.
    """
    messages = entry.get("messages")
    if not (isinstance(messages, list) and messages):
        return [
            {"role": "user", "content": entry["prompt"]},
            {"role": "assistant", "content": entry["response"]},
        ]
    out: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        turn: dict[str, Any] = {"role": message["role"], "content": message["content"]}
        if turn["role"] == "assistant" and index < len(messages) - 1:
            turn["weight"] = 0
        out.append(turn)
    return out


def to_hf_trainer(entry: dict[str, Any]) -> dict[str, Any]:
    """HF Trainer text-style: ``prompt`` and ``response`` only.

    Plain text cannot mark which tokens are targets, so earlier turns (which
    may hold sub-threshold replies) are left out rather than trained on.
    """
    return {
        "text": f"{entry['prompt']}\n\n{entry['response']}",
    }


def to_chat(entry: dict[str, Any]) -> dict[str, Any]:
    """OpenAI / chat-template style — what ``apply_chat_template`` consumes.

    Emits the full graded context, so training sees what the grader saw.
    """
    return {"messages": entry_messages(entry)}


def to_axolotl(entry: dict[str, Any]) -> dict[str, Any]:
    """Axolotl / unsloth alpaca-style — ``instruction``, ``input``, ``output``.

    ``instruction`` is the last user turn; turns before it go in ``input``.
    """
    messages = entry_messages(entry)
    *context, response = messages
    last_user = max(
        (i for i, m in enumerate(context) if m["role"] == "user"), default=None
    )
    earlier = context[:last_user] if last_user is not None else context
    return {
        "instruction": context[last_user]["content"] if last_user is not None else "",
        "input": "\n\n".join(m["content"] for m in earlier),
        "output": response["content"],
    }


def flatten(entry: dict[str, Any]) -> dict[str, Any]:
    """Legacy single-turn view: drop ``messages`` and keep prompt/response."""
    return {k: v for k, v in entry.items() if k != "messages"}


def _valid_messages(messages: Any, response: Any) -> bool:
    return (
        isinstance(messages, list)
        and bool(messages)
        and all(
            isinstance(m, dict)
            and isinstance(m.get("role"), str)
            and isinstance(m.get("content"), str)
            for m in messages
        )
        and messages[-1]["role"] == "assistant"
        and messages[-1]["content"] == response
    )


FORMATTERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "hf-trainer": to_hf_trainer,
    "chat": to_chat,
    "axolotl": to_axolotl,
}


def load_curated(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate the curated JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"Curated file not found: {path}")
    entries: list[dict[str, Any]] = []
    for line_num, line in read_jsonl_lines(path):
        line = line.strip()
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("Skipping line %d: %s", line_num, e)
            continue
        if (
            not isinstance(entry, dict)
            or "prompt" not in entry
            or "response" not in entry
        ):
            logger.warning("Skipping line %d: missing prompt/response", line_num)
            continue
        if "messages" in entry and not _valid_messages(
            entry["messages"], entry["response"]
        ):
            logger.warning(
                "Skipping line %d: messages must be role/content turns ending "
                "with the response",
                line_num,
            )
            continue
        entries.append(entry)
    return entries


def entry_score(entry: dict[str, Any]) -> float | None:
    """The entry's numeric score, or ``None`` when it has none."""
    score = entry.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    return float(score)


def _dedup_key(entry: dict[str, Any]) -> str:
    context = entry_messages(entry)[:-1]
    return json.dumps([[m["role"], m["content"]] for m in context], ensure_ascii=False)


def filter_entries(
    entries: list[dict[str, Any]],
    min_score: float | None = None,
    sources: list[str] | None = None,
    dedup: bool = False,
) -> list[dict[str, Any]]:
    """Apply optional filters.

    ``min_score`` keeps only entries whose numeric score is at least
    ``min_score``; an entry without a score fails it (and is counted in a
    warning). ``dedup`` keeps the first entry per context (every turn before
    the response; the prompt for single-turn entries).
    """
    out = entries
    if min_score is not None:
        unscored = sum(1 for e in out if entry_score(e) is None)
        if unscored:
            logger.warning(
                "--min-score dropped %d entr%s with no numeric score",
                unscored,
                "y" if unscored == 1 else "ies",
            )
        out = [
            e
            for e in out
            if (score := entry_score(e)) is not None and score >= min_score
        ]
    if sources:
        source_set = set(sources)
        out = [e for e in out if e.get("source") in source_set]
    if dedup:
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for e in out:
            key = _dedup_key(e)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(e)
        out = deduped
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Curated JSONL from `grade_transcript.py --output-curated`.",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=sorted(FORMATTERS),
        required=True,
        help="Target SFT dataset format.",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output JSONL path."
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Drop entries with score below this threshold.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Keep only entries from these source files (repeatable).",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate by context, i.e. the turns before the response "
        "(keeps first occurrence).",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Emit legacy single-turn (prompt, response) examples, dropping "
        "earlier turns of the graded conversation.",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Print a summary of what got included."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    entries = load_curated(args.input)
    logger.info("Loaded %d entries from %s", len(entries), args.input)
    if args.flatten:
        entries = [flatten(entry) for entry in entries]

    filtered = filter_entries(
        entries,
        min_score=args.min_score,
        sources=args.source,
        dedup=args.dedup,
    )
    logger.info("After filtering: %d entries", len(filtered))

    formatter = FORMATTERS[args.format]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for entry in filtered:
            f.write(json.dumps(formatter(entry), ensure_ascii=False) + "\n")
    logger.info(
        "Wrote %d entries → %s (format=%s)", len(filtered), args.output, args.format
    )

    if args.stats:
        scores = [s for e in filtered if (s := entry_score(e)) is not None]
        sources = {e.get("source", "?") for e in filtered}
        print()
        print("Output stats:")
        print(f"  format:    {args.format}")
        print(f"  entries:   {len(filtered)}")
        print(f"  unscored:  {len(filtered) - len(scores)}")
        if scores:
            print(f"  min score: {min(scores):.3f}")
            print(f"  max score: {max(scores):.3f}")
            print(f"  mean:      {sum(scores) / len(scores):.3f}")
        print(f"  sources:   {len(sources)} distinct ({', '.join(sorted(sources))})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
