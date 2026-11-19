"""
Convert a curated JSONL (from ``grade_transcript.py --output-curated``) into
the SFT dataset format your training tool of choice expects.

Each record includes a prompt, response, score and source. Current exports also
include the messages through the graded assistant action and any tool schemas.
Earlier assistant replies have weight 0 and the selected action has weight 1.

This script reshapes those into one of three target formats:

* ``--format hf-trainer`` — Hugging Face ``Trainer`` text-style:
    ``{"text": "<prompt><response>"}`` (one per line)

* ``--format chat`` — OpenAI / chat-template style:
    ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``
    (the format ``apply_chat_template`` consumes; also what TRL's SFTTrainer
    expects for chat models)

* ``--format axolotl`` — axolotl/unsloth-friendly ``alpaca`` style:
    ``{"instruction": "...", "input": "", "output": "..."}``

The script supports filtering by source or minimum score. Deduplication compares
the full prompt history and tool schemas and retains the first matching row.

Usage::

    python -m stateset_agents.data.prepare_sft_dataset \\
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

logger = logging.getLogger("prepare_sft_dataset")


def _chat_messages(entry: dict[str, Any]) -> list[dict[str, Any]]:
    messages = entry.get("messages")
    if isinstance(messages, list):
        return messages
    return [
        {"role": "user", "content": entry["prompt"]},
        {"role": "assistant", "content": entry["response"]},
    ]


def example_key(entry: dict[str, Any], *, prompt_only: bool = False) -> str:
    """Identify a conversation by its messages and tool schemas, including legacy pairs."""
    messages = [
        {k: v for k, v in m.items() if k != "weight"} for m in _chat_messages(entry)
    ]
    return json.dumps(
        [entry.get("tools", []), messages[:-1] if prompt_only else messages],
        ensure_ascii=False,
        sort_keys=True,
    )


def _flat_pair(entry: dict[str, Any]) -> tuple[str, str]:
    """Keep structured context as JSON in formats that only accept strings."""
    messages = _chat_messages(entry)
    if (
        len(messages) == 2
        and messages[0]["role"] == "user"
        and set(messages[0]) == {"role", "content"}
        and not entry.get("tools")
    ):
        prompt = messages[0]["content"]
    else:
        prefix: dict[str, Any] = {
            "messages": [
                {k: v for k, v in message.items() if k != "weight"}
                for message in messages[:-1]
            ]
        }
        if "tools" in entry:
            prefix["tools"] = entry["tools"]
        prompt = json.dumps(prefix, ensure_ascii=False)
    target = messages[-1]
    response = (
        json.dumps(
            {k: v for k, v in target.items() if k != "weight"}, ensure_ascii=False
        )
        if target.get("tool_calls")
        else target["content"]
    )
    return prompt, response


def to_hf_trainer(entry: dict[str, Any]) -> dict[str, Any]:
    """HF Trainer text-style: one ``text`` field with prompt + response."""
    prompt, response = _flat_pair(entry)
    return {"text": f"{prompt}\n\n{response}"}


def to_chat(entry: dict[str, Any]) -> dict[str, Any]:
    """OpenAI / chat-template style — what ``apply_chat_template`` consumes."""
    row: dict[str, Any] = {"messages": _chat_messages(entry)}
    if "tools" in entry:
        row["tools"] = entry["tools"]
    return row


def to_axolotl(entry: dict[str, Any]) -> dict[str, Any]:
    """Axolotl / unsloth alpaca-style — ``instruction``, ``input``, ``output``."""
    prompt, response = _flat_pair(entry)
    return {"instruction": prompt, "input": "", "output": response}


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
    for line_num, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("Skipping line %d: %s", line_num, e)
            continue
        if "prompt" not in entry or "response" not in entry:
            logger.warning("Skipping line %d: missing prompt/response", line_num)
            continue
        entries.append(entry)
    return entries


def filter_entries(
    entries: list[dict[str, Any]],
    min_score: float | None = None,
    sources: list[str] | None = None,
    dedup: bool = False,
) -> list[dict[str, Any]]:
    """Apply optional filters."""
    out = entries
    if min_score is not None:
        out = [e for e in out if float(e.get("score", 1.0)) >= min_score]
    if sources:
        source_set = set(sources)
        out = [e for e in out if e.get("source") in source_set]
    if dedup:
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for e in out:
            key = example_key(e, prompt_only=True)
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
        help="Deduplicate by prompt (keeps first occurrence).",
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
        scores = [float(e.get("score", 0.0)) for e in filtered]
        sources = {e.get("source", "?") for e in filtered}
        print()
        print("Output stats:")
        print(f"  format:    {args.format}")
        print(f"  entries:   {len(filtered)}")
        if scores:
            print(f"  min score: {min(scores):.3f}")
            print(f"  max score: {max(scores):.3f}")
            print(f"  mean:      {sum(scores) / len(scores):.3f}")
        print(f"  sources:   {len(sources)} distinct ({', '.join(sorted(sources))})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
