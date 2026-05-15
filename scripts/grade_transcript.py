"""
Grade a saved conversation transcript with the framework's reward functions.

Reads a JSONL transcript (one ``{"role": "...", "content": "..."}`` per line)
written by ``stateset-agents chat --history <file>`` and scores each assistant
turn under the named reward function. Output is a markdown report with per-turn
scores + a summary, plus an optional JSON dump for downstream tooling.

This closes the human-in-the-loop:

* Chat with a fine-tune via ``stateset-agents chat --history conversations.jsonl``
* When you spot interesting outputs (good or bad), keep chatting — the JSONL
  grows.
* Run ``make grade-transcript HISTORY=conversations.jsonl REWARD=customer_support``
  to score what the same reward function used during training would have given.
* Compare reward scores to your intuition. Disagreements are bugs in the
  reward function (or surprising outputs from the model).

Supported reward names: ``gsm8k``, ``customer_support``, ``tool_calling``.

Usage::

    python scripts/grade_transcript.py \\
        --history conversations.jsonl \\
        --reward customer_support \\
        --context-file scenarios.jsonl \\
        --output graded.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("grade_transcript")


def load_transcript(path: Path) -> list[dict[str, str]]:
    """Load a JSONL transcript."""
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")
    turns: list[dict[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("Skipping malformed JSONL line: %s", e)
            continue
        if "role" in entry and "content" in entry:
            turns.append({"role": entry["role"], "content": entry["content"]})
    return turns


def load_contexts(path: Path | None) -> list[dict[str, Any]]:
    """Optional context JSONL (one context object per *assistant* turn).

    For verifiable-reward tasks, the reward function needs the ground truth
    (``expected_tool``, ``gold_answer``, ``must_acknowledge``, etc.). When
    grading a transcript from a benchmark dataset, pass the same scenarios
    JSONL the agent was prompted with.
    """
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def get_reward(reward_name: str) -> Any:
    """Map a reward name to a reward function instance."""
    if reward_name == "gsm8k":
        from stateset_agents.data.gsm8k import GSM8KReward
        return GSM8KReward()
    if reward_name == "customer_support":
        from stateset_agents.data.customer_support_bench import SupportRewardComposite
        return SupportRewardComposite()
    if reward_name == "tool_calling":
        from stateset_agents.data.tool_calling_bench import ToolCallReward
        return ToolCallReward()
    raise ValueError(
        f"Unknown reward: {reward_name!r}. Choose: gsm8k, customer_support, tool_calling."
    )


async def grade_transcript(
    turns: list[dict[str, str]],
    contexts: list[dict[str, Any]],
    reward: Any,
) -> list[dict[str, Any]]:
    """Score each assistant turn. Returns one dict per assistant turn."""
    from stateset_agents.core.trajectory import ConversationTurn

    rows: list[dict[str, Any]] = []
    assistant_idx = 0
    convo_so_far: list[ConversationTurn] = []

    for turn in turns:
        ct = ConversationTurn(role=turn["role"], content=turn["content"])
        convo_so_far.append(ct)

        if turn["role"] != "assistant":
            continue

        # Pick the matching context (1:1 with assistant turns).
        context = contexts[assistant_idx] if assistant_idx < len(contexts) else None
        result = await reward.compute_reward(convo_so_far, context=context)

        rows.append({
            "assistant_turn_idx": assistant_idx,
            "user_query": turns[max(0, assistant_idx * 2 - 1)]["content"][:80]
                if assistant_idx * 2 - 1 >= 0 and assistant_idx * 2 - 1 < len(turns)
                else "",
            "response_preview": turn["content"][:80],
            "score": float(result.score),
            "breakdown": dict(result.breakdown) if hasattr(result, "breakdown") else {},
            "explanation": getattr(result, "explanation", None),
        })
        assistant_idx += 1

    return rows


def render_markdown(rows: list[dict[str, Any]], reward_name: str) -> str:
    """Render the per-turn report."""
    lines: list[str] = []
    lines.append(f"# Transcript graded with reward = `{reward_name}`")
    lines.append("")
    if not rows:
        lines.append("_No assistant turns found in transcript._")
        return "\n".join(lines)

    scores = [r["score"] for r in rows]
    mean = sum(scores) / len(scores)
    n_full = sum(1 for s in scores if s >= 0.999)
    n_zero = sum(1 for s in scores if s < 0.001)

    lines.append(f"**Total assistant turns:** {len(rows)}")
    lines.append(f"**Mean score:** {mean:.3f}")
    lines.append(f"**Perfect (≥0.999):** {n_full}/{len(rows)}")
    lines.append(f"**Zero (<0.001):** {n_zero}/{len(rows)}")
    lines.append("")
    lines.append("| # | Score | Preview |")
    lines.append("|---|-------|---------|")
    for row in rows:
        preview = row["response_preview"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['assistant_turn_idx']} | {row['score']:.3f} | {preview} |")
    return "\n".join(lines)


def _existing_curated_keys(output_path: Path) -> set[tuple[str, str]]:
    """Read an existing curated file and return the set of (prompt, response) hashes.

    Used to make ``write_curated_examples`` idempotent across reruns. Returns
    an empty set if the file doesn't yet exist.
    """
    if not output_path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for line in output_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        prompt = entry.get("prompt", "")
        response = entry.get("response", "")
        keys.add((prompt, response))
    return keys


def write_curated_examples(
    transcript_path: Path,
    turns: list[dict[str, str]],
    rows: list[dict[str, Any]],
    threshold: float,
    output_path: Path,
) -> int:
    """Append high-scoring (user, assistant) pairs to a curated JSONL file.

    Returns the number of examples written. Each line is::

        {"prompt": "<user>", "response": "<assistant>",
         "score": 0.83, "source": "session1.jsonl"}

    Designed to be append-safe across many transcripts — multiple invocations
    of this script can write into the same curated file without clobbering.
    **Idempotent:** a (prompt, response) pair that already exists in the file
    is skipped, so re-running batch grading doesn't duplicate examples.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = _existing_curated_keys(output_path)

    # Build (user, assistant) pairs aligned with rows.
    pairs: list[tuple[str, str]] = []
    last_user: str | None = None
    for turn in turns:
        if turn["role"] == "user":
            last_user = turn["content"]
        elif turn["role"] == "assistant":
            pairs.append((last_user or "", turn["content"]))
            last_user = None

    written = 0
    with output_path.open("a", encoding="utf-8") as f:
        for pair, row in zip(pairs, rows):
            if row["score"] >= threshold and pair not in seen:
                f.write(json.dumps({
                    "prompt": pair[0],
                    "response": pair[1],
                    "score": row["score"],
                    "source": transcript_path.name,
                }, ensure_ascii=False) + "\n")
                seen.add(pair)
                written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True,
                        help="JSONL transcript from `chat --history`.")
    parser.add_argument("--reward", choices=["gsm8k", "customer_support", "tool_calling"],
                        required=True)
    parser.add_argument("--context-file", type=Path, default=None,
                        help="Optional JSONL of context dicts (one per assistant turn).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write the markdown report to this path (default: stdout).")
    parser.add_argument("--json", action="store_true",
                        help="Also write a raw JSON dump alongside the markdown report.")
    parser.add_argument("--output-curated", type=Path, default=None,
                        help="Append (prompt, response, score) tuples to this JSONL "
                             "for every assistant turn whose score >= --threshold. "
                             "Builds a curated training set across multiple runs.")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Minimum score for inclusion in --output-curated (default: 0.7).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    turns = load_transcript(args.history)
    contexts = load_contexts(args.context_file)
    reward = get_reward(args.reward)
    logger.info("Loaded %d turn(s), %d context(s), reward=%s",
                len(turns), len(contexts), args.reward)

    rows = asyncio.run(grade_transcript(turns, contexts, reward))
    md = render_markdown(rows, args.reward)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        logger.info("Wrote %s", args.output)
        if args.json:
            json_path = args.output.with_suffix(".json")
            json_path.write_text(json.dumps(rows, indent=2))
            logger.info("Wrote %s", json_path)
    else:
        print(md)

    if args.output_curated is not None:
        n_curated = write_curated_examples(
            args.history, turns, rows, args.threshold, args.output_curated
        )
        logger.info(
            "Curated %d example(s) (score >= %.2f) → %s",
            n_curated, args.threshold, args.output_curated,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
