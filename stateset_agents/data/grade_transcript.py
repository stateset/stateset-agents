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

    python -m stateset_agents.data.grade_transcript \\
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
import math
import sys
from pathlib import Path
from typing import Any

from stateset_agents.data.prepare_sft_dataset import example_key
from stateset_agents.data.tool_calling_bench import (
    normalize_tool_call,
    tool_call_references,
)

logger = logging.getLogger("grade_transcript")


def load_transcript(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL transcript."""
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")
    turns: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("Skipping malformed JSONL line: %s", e)
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("role"), str):
            continue
        if (
            entry["role"] == "assistant"
            and entry.get("tool_calls")
            and entry.get("content") is None
        ):
            entry["content"] = ""
        if isinstance(entry.get("content"), str):
            turns.append(
                {
                    key: entry[key]
                    for key in (
                        "role",
                        "content",
                        "name",
                        "tool_calls",
                        "tool_call_id",
                        "tools",
                    )
                    if key in entry
                }
            )
    return turns


def load_contexts(
    path: Path | None, *, expected_count: int | None = None
) -> list[dict[str, Any]]:
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
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            context = json.loads(line)
            if not isinstance(context, dict):
                raise ValueError(f"{path}:{number}: expected a grading context object")
            out.append(context)
    if expected_count is not None and len(out) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} assistant-turn contexts, found {len(out)}"
        )
    return out


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

        return ToolCallReward(per_turn=True)
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


def _context_issue(reward_name: str | None, context: dict[str, Any]) -> str | None:
    if reward_name == "gsm8k":
        gold = context.get("gold_answer")
        try:
            if gold is None or isinstance(gold, bool) or not math.isfinite(float(gold)):
                return "Expected a finite numeric gold_answer"
        except (TypeError, ValueError, OverflowError):
            return "Expected a finite numeric gold_answer"
    if reward_name in ("tool_calling", "tool_call_composite"):
        try:
            tool_call_references(context)
        except ValueError as exc:
            return str(exc)
    return None


async def grade_transcript(
    turns: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    reward: Any,
    *,
    reward_name: str | None = None,
) -> list[dict[str, Any]]:
    """Score each assistant turn. Returns one dict per assistant turn."""
    from stateset_agents.core.trajectory import ConversationTurn

    rows: list[dict[str, Any]] = []
    assistant_idx = 0
    last_user_query = ""
    convo_so_far: list[ConversationTurn] = []

    reward_name = reward_name or getattr(reward, "name", None)
    if contexts and len(contexts) != sum(t["role"] == "assistant" for t in turns):
        raise ValueError("Expected one context object per assistant turn")
    for turn in turns:
        if turn["role"] == "user":
            last_user_query = turn["content"]
        ct = ConversationTurn(
            role=turn["role"],
            content=turn["content"],
            tool_calls=turn.get("tool_calls"),
            metadata={k: turn[k] for k in ("name", "tool_call_id") if k in turn},
        )
        convo_so_far.append(ct)

        if turn["role"] != "assistant":
            continue

        # Pick the matching context (1:1 with assistant turns).
        context = contexts[assistant_idx] if contexts else {}
        issue = _context_issue(reward_name, context)
        result = (
            None
            if issue
            else await reward.compute_reward(convo_so_far, context=context or None)
        )
        if result is not None:
            issue = getattr(result, "metadata", {}).get("ungradable_reason")

        rows.append(
            {
                "assistant_turn_idx": assistant_idx,
                "user_query": last_user_query[:80],
                "response_preview": turn["content"][:80],
                "score": (
                    float(result.score) if result is not None and not issue else None
                ),
                "status": "ungradable" if issue else "graded",
                "breakdown": dict(getattr(result, "breakdown", {})),
                "explanation": issue or getattr(result, "explanation", None),
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

    scores = [r["score"] for r in rows if r["score"] is not None]
    mean = f"{sum(scores) / len(scores):.3f}" if scores else "unavailable"
    n_full = sum(1 for s in scores if s >= 0.999)
    n_zero = sum(1 for s in scores if s < 0.001)

    lines.append(f"**Total assistant turns:** {len(rows)}")
    lines.append(f"**Ungradable turns:** {len(rows) - len(scores)}")
    lines.append(f"**Mean score:** {mean}")
    lines.append(f"**Perfect (≥0.999):** {n_full}/{len(scores)}")
    lines.append(f"**Zero (<0.001):** {n_zero}/{len(scores)}")
    lines.append("")
    lines.append("| # | Score | Preview |")
    lines.append("|---|-------|---------|")
    for row in rows:
        preview = row["response_preview"].replace("|", "\\|").replace("\n", " ")
        score = (
            f"{row['score']:.3f}"
            if row["score"] is not None
            else f"ungradable ({row['explanation']})"
        )
        lines.append(f"| {row['assistant_turn_idx']} | {score} | {preview} |")
    return "\n".join(lines)


def _existing_curated_keys(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    keys = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and (
            "messages" in entry or {"prompt", "response"} <= entry.keys()
        ):
            keys.add(example_key(entry))
    return keys


def write_curated_examples(
    transcript_path: Path,
    turns: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    threshold: float,
    output_path: Path,
) -> int:
    """Append each passing assistant action with its observed conversation prefix."""
    if len(rows) != sum(t["role"] == "assistant" for t in turns):
        raise ValueError("Expected one grade per assistant turn")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = _existing_curated_keys(output_path)
    history: list[dict[str, Any]] = []
    tools = turns[0].get("tools") if turns else None
    last_user = ""
    grades = iter(rows)
    written = 0
    with output_path.open("a", encoding="utf-8") as stream:
        for turn in turns:
            message: dict[str, Any] = {
                k: turn[k]
                for k in ("role", "content", "name", "tool_call_id")
                if k in turn
            }
            if turn.get("tool_calls"):
                message["tool_calls"] = [
                    normalize_tool_call(c) for c in turn["tool_calls"]
                ]
            history.append(message)
            if turn["role"] == "user":
                last_user = turn["content"]
            if turn["role"] != "assistant":
                continue
            message["weight"] = 1
            score = next(grades)["score"]
            if score is not None and score >= threshold:
                entry = {
                    "prompt": last_user,
                    "response": turn["content"],
                    "messages": history,
                    "score": score,
                    "source": transcript_path.name,
                }
                if tools is not None:
                    entry["tools"] = tools
                key = example_key(entry)
                if key not in seen:
                    stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    seen.add(key)
                    written += 1
            # Subsequent examples use this action solely as observed context.
            message["weight"] = 0
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
        help="Optional JSONL of context dicts (one per assistant turn).",
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
        help="Append each passing assistant turn and its conversation prefix to this JSONL.",
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

    turns = load_transcript(args.history)
    contexts = load_contexts(
        args.context_file, expected_count=sum(t["role"] == "assistant" for t in turns)
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
            n_curated,
            args.threshold,
            args.output_curated,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
