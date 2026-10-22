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

from stateset_agents.data.jsonl import read_jsonl_lines

logger = logging.getLogger("grade_transcript")


def load_transcript(path: Path) -> list[dict[str, str]]:
    """Load a UTF-8 JSONL transcript, one ``{"role", "content"}`` per line.

    A malformed line (bad JSON, or no string ``role``/``content``) raises
    ``ValueError`` naming the line: skipping it would shift every later turn,
    pairing assistant replies with the wrong question and context.
    """
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")
    turns: list[dict[str, str]] = []
    for number, line in read_jsonl_lines(path):
        line = line.strip()
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{number}: malformed JSONL line: {e}") from e
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("role"), str)
            or not entry["role"].strip()
            or not isinstance(entry.get("content"), str)
        ):
            raise ValueError(
                f"{path}:{number}: each line needs a string role and content"
            )
        turns.append({"role": entry["role"], "content": entry["content"]})
    return turns


def load_contexts(path: Path | None) -> list[dict[str, Any]]:
    """Optional context JSONL (one context object per *assistant* turn).

    For verifiable-reward tasks, the reward function needs the ground truth
    (``expected_tool``, ``gold_answer``, ``must_acknowledge``, etc.). When
    grading a transcript from a benchmark dataset, pass the same scenarios
    JSONL the agent was prompted with. Every non-blank line must be a JSON
    object; the i-th object belongs to the i-th assistant turn.
    """
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")
    out: list[dict[str, Any]] = []
    for number, line in read_jsonl_lines(path):
        line = line.strip()
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{number}: malformed context line: {e}") from e
        if not isinstance(entry, dict):
            raise ValueError(f"{path}:{number}: each context must be a JSON object")
        out.append(entry)
    return out


def assistant_turn_count(turns: list[dict[str, str]]) -> int:
    """Number of assistant turns, i.e. graded rows and expected contexts."""
    return sum(1 for turn in turns if turn["role"] == "assistant")


def preceding_user_content(turns: list[dict[str, str]], index: int) -> str:
    """Content of the last user turn before ``turns[index]`` ("" if none)."""
    for turn in reversed(turns[:index]):
        if turn["role"] == "user":
            return turn["content"]
    return ""


def get_reward(reward_name: str, persona: dict[str, Any] | None = None) -> Any:
    """Map a reward name to a reward function instance.

    ``persona`` (optional, customer_support only) is a
    ``{"opener": [...], "signoff": [...]}`` config scored as a
    persona-fidelity component of ``SupportRewardComposite``.
    """
    if persona and reward_name != "customer_support":
        raise ValueError(
            f"persona config is only supported for the customer_support reward, "
            f"not {reward_name!r}."
        )
    if reward_name == "gsm8k":
        from stateset_agents.data.gsm8k import GSM8KReward

        return GSM8KReward()
    if reward_name == "customer_support":
        from stateset_agents.data.customer_support_bench import SupportRewardComposite

        return SupportRewardComposite(persona=persona)
    if reward_name == "tool_calling":
        from stateset_agents.data.tool_calling_bench import ToolCallReward

        return ToolCallReward()
    if reward_name == "nsr":
        # Deterministic proof-backed verifier (StateSet NSR). Fail-closed for
        # grading/curation: an unreachable verifier scores 0.0, never a
        # neutral pass. Context rows supply the decision question via
        # {"nsr_request": {...}} (see docs/NSR_INTEGRATION.md).
        from stateset_agents.rewards.nsr_verifier import (
            NSRVerifierConfig,
            NSRVerifierReward,
        )

        config = NSRVerifierConfig.from_env()
        config.error_score = 0.0
        return NSRVerifierReward(config=config)
    raise ValueError(
        f"Unknown reward: {reward_name!r}. Choose: gsm8k, customer_support, "
        "tool_calling, nsr."
    )


async def grade_transcript(
    turns: list[dict[str, str]],
    contexts: list[dict[str, Any]],
    reward: Any,
) -> list[dict[str, Any]]:
    """Score each assistant turn. Returns one dict per assistant turn.

    ``contexts`` is either empty (no ground truth) or holds exactly one
    context per assistant turn, in order; any other length raises
    ``ValueError`` instead of grading replies against another turn's gold.
    """
    from stateset_agents.core.trajectory import ConversationTurn

    expected = assistant_turn_count(turns)
    if contexts and len(contexts) != expected:
        raise ValueError(
            f"{len(contexts)} context(s) for {expected} assistant turn(s); "
            "the context file needs exactly one context per assistant turn"
        )
    rows: list[dict[str, Any]] = []
    assistant_idx = 0
    convo_so_far: list[ConversationTurn] = []

    for index, turn in enumerate(turns):
        ct = ConversationTurn(role=turn["role"], content=turn["content"])
        convo_so_far.append(ct)

        if turn["role"] != "assistant":
            continue

        # Contexts are 1:1 with assistant turns (length checked above).
        context = contexts[assistant_idx] if contexts else None
        result = await reward.compute_reward(convo_so_far, context=context)

        rows.append(
            {
                "assistant_turn_idx": assistant_idx,
                "user_query": preceding_user_content(turns, index)[:80],
                "response_preview": turn["content"][:80],
                "score": float(result.score),
                "breakdown": (
                    dict(result.breakdown) if hasattr(result, "breakdown") else {}
                ),
                "explanation": getattr(result, "explanation", None),
            }
        )
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
        lines.append(
            f"| {row['assistant_turn_idx']} | {row['score']:.3f} | {preview} |"
        )
    return "\n".join(lines)


def _messages_key(messages: list[dict[str, str]]) -> str:
    return json.dumps(
        [[m.get("role"), m.get("content")] for m in messages], ensure_ascii=False
    )


def _existing_curated_keys(output_path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    """Identities of the examples already in a curated file.

    Returns ``(message_keys, legacy_pairs)``: the full graded context of
    every entry carrying ``messages``, and the ``(prompt, response)`` pair of
    older entries written without it. Used to make ``write_curated_examples``
    idempotent across reruns; both sets are empty if the file doesn't exist.
    """
    keys: set[str] = set()
    legacy: set[tuple[str, str]] = set()
    if not output_path.exists():
        return keys, legacy
    for _, line in read_jsonl_lines(output_path):
        line = line.strip()
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("messages"), list):
            keys.add(_messages_key(entry["messages"]))
        else:
            legacy.add((entry.get("prompt", ""), entry.get("response", "")))
    return keys, legacy


def write_curated_examples(
    transcript_path: Path,
    turns: list[dict[str, str]],
    rows: list[dict[str, Any]],
    threshold: float,
    output_path: Path,
) -> int:
    """Append high-scoring assistant turns, with their context, to a curated JSONL.

    Returns the number of examples written. Each line is::

        {"prompt": "<preceding user turn>", "response": "<assistant>",
         "messages": [<every turn up to and including this reply>],
         "score": 0.83, "source": "session1.jsonl"}

    ``messages`` is the exact context the turn was graded in, so training
    sees what the grader saw (``prepare_sft_dataset.py --format chat`` emits
    it); ``prompt``/``response`` stay for single-turn consumers. Earlier
    assistant replies carry ``"weight": 0``: they are context, possibly below
    the threshold, and only the graded reply is a training target.

    Designed to be append-safe across many transcripts — multiple invocations
    of this script can write into the same curated file without clobbering.
    **Idempotent:** an example whose full context already exists in the file
    is skipped, so re-running batch grading doesn't duplicate examples.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen, legacy = _existing_curated_keys(output_path)

    # One example per assistant turn, aligned with rows.
    examples: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        # Earlier replies are context only (weight 0, the OpenAI fine-tuning
        # convention): they may have failed the threshold, and every passing
        # reply is curated once, in its own example.
        messages: list[dict[str, Any]] = [
            {"role": t["role"], "content": t["content"]} for t in turns[:index]
        ]
        for message in messages:
            if message["role"] == "assistant":
                message["weight"] = 0
        messages.append({"role": "assistant", "content": turn["content"]})
        examples.append(
            {
                "prompt": preceding_user_content(turns, index),
                "response": turn["content"],
                "messages": messages,
            }
        )

    written = 0
    with output_path.open("a", encoding="utf-8") as f:
        for example, row in zip(examples, rows, strict=True):
            key = _messages_key(example["messages"])
            pair = (example["prompt"], example["response"])
            if row["score"] < threshold or key in seen or pair in legacy:
                continue
            f.write(
                json.dumps(
                    {
                        **example,
                        "score": row["score"],
                        "source": transcript_path.name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            seen.add(key)
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        required=True,
        help="JSONL transcript from `chat --history`.",
    )
    parser.add_argument(
        "--reward",
        choices=["gsm8k", "customer_support", "tool_calling", "nsr"],
        required=True,
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        default=None,
        help="Optional JSONL of context dicts, exactly one per assistant turn.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the markdown report to this path (default: stdout).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write a raw JSON dump alongside the markdown report.",
    )
    parser.add_argument(
        "--output-curated",
        type=Path,
        default=None,
        help="Append (prompt, response, score) tuples to this JSONL "
        "for every assistant turn whose score >= --threshold. "
        "Builds a curated training set across multiple runs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum score for inclusion in --output-curated (default: 0.7).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    try:
        turns = load_transcript(args.history)
        contexts = load_contexts(args.context_file)
    except ValueError as exc:
        parser.error(str(exc))
    if args.context_file is not None and len(contexts) != assistant_turn_count(turns):
        parser.error(
            f"--context-file has {len(contexts)} context(s) but the transcript "
            f"has {assistant_turn_count(turns)} assistant turn(s); they must "
            "match 1:1"
        )
    reward = get_reward(args.reward)
    logger.info(
        "Loaded %d turn(s), %d context(s), reward=%s",
        len(turns),
        len(contexts),
        args.reward,
    )

    rows = asyncio.run(grade_transcript(turns, contexts, reward))
    md = render_markdown(rows, args.reward)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        logger.info("Wrote %s", args.output)
        if args.json:
            json_path = args.output.with_suffix(".json")
            json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            logger.info("Wrote %s", json_path)
    else:
        print(md)

    if args.output_curated is not None:
        n_curated = write_curated_examples(
            args.history, turns, rows, args.threshold, args.output_curated
        )
        logger.info(
            "Curated %d example(s) (score >= %.2f) → %s",
            n_curated,
            args.threshold,
            args.output_curated,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
