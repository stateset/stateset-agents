"""
Aggregate a directory of per-transcript grading JSONs into one umbrella report.

After ``make grade-batch DIR=transcripts/ REWARD=...`` produces a directory of
``*.json`` files (one per graded transcript), this script reads them all and
emits cross-transcript statistics: mean of means, total turns, perfect/zero
counts, plus a per-transcript table.

Designed to be the "one report I send to the team" at the end of a curation
session. Output is markdown so it pastes cleanly into PRs and Slack.

Usage::

    python scripts/summarize_graded_batch.py \\
        --graded-dir transcripts/graded \\
        --output transcripts/graded/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("summarize_graded_batch")


def load_graded_jsons(graded_dir: Path) -> list[tuple[Path, list[dict[str, Any]]]]:
    """Load every ``*.json`` (skipping summary.json) from ``graded_dir``."""
    if not graded_dir.exists():
        raise FileNotFoundError(f"Directory not found: {graded_dir}")
    out: list[tuple[Path, list[dict[str, Any]]]] = []
    for path in sorted(graded_dir.glob("*.json")):
        if path.name in {"summary.json", "passes_gates.json"}:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            logger.warning("Skipping %s: %s", path.name, e)
            continue
        if isinstance(data, list):
            out.append((path, data))
        else:
            logger.warning("Skipping %s: expected a list of turn rows", path.name)
    return out


def render_summary(
    transcripts: list[tuple[Path, list[dict[str, Any]]]],
) -> str:
    """Render the umbrella markdown report."""
    lines: list[str] = []
    lines.append("# Graded Transcripts — Cross-Session Summary")
    lines.append("")

    if not transcripts:
        lines.append("_No graded transcripts found._")
        return "\n".join(lines)

    all_scores: list[float] = []
    per_transcript: list[dict[str, Any]] = []

    for path, rows in transcripts:
        scores = [float(r["score"]) for r in rows if "score" in r]
        if not scores:
            continue
        all_scores.extend(scores)
        per_transcript.append(
            {
                "name": path.stem,
                "n_turns": len(scores),
                "mean": statistics.mean(scores),
                "perfect": sum(1 for s in scores if s >= 0.999),
                "zero": sum(1 for s in scores if s < 0.001),
                "min": min(scores),
                "max": max(scores),
            }
        )

    if not all_scores:
        lines.append("_No assistant turns found across the transcripts._")
        return "\n".join(lines)

    grand_mean = statistics.mean(all_scores)
    grand_std = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
    n_perfect = sum(1 for s in all_scores if s >= 0.999)
    n_zero = sum(1 for s in all_scores if s < 0.001)

    lines.append(f"**Transcripts:** {len(per_transcript)}")
    lines.append(f"**Total assistant turns:** {len(all_scores)}")
    lines.append(f"**Grand mean score:** {grand_mean:.3f} ± {grand_std:.3f}")
    lines.append(
        f"**Perfect (≥0.999):** {n_perfect}/{len(all_scores)} ({100 * n_perfect / len(all_scores):.1f}%)"
    )
    lines.append(
        f"**Zero (<0.001):** {n_zero}/{len(all_scores)} ({100 * n_zero / len(all_scores):.1f}%)"
    )
    lines.append("")

    # Per-transcript table — sorted by mean score descending.
    per_transcript.sort(key=lambda r: r["mean"], reverse=True)
    lines.append("## Per-transcript breakdown")
    lines.append("")
    lines.append("| Transcript | Turns | Mean | Min | Max | Perfect | Zero |")
    lines.append("|------------|-------|------|-----|-----|---------|------|")
    for row in per_transcript:
        lines.append(
            f"| `{row['name']}` | {row['n_turns']} | {row['mean']:.3f} "
            f"| {row['min']:.3f} | {row['max']:.3f} "
            f"| {row['perfect']} | {row['zero']} |"
        )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graded-dir",
        type=Path,
        required=True,
        help="Directory containing the *.json files from `grade-batch`.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the markdown report here (default: stdout).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    transcripts = load_graded_jsons(args.graded_dir)
    logger.info(
        "Loaded %d transcript file(s) from %s", len(transcripts), args.graded_dir
    )

    md = render_summary(transcripts)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        logger.info("Wrote %s", args.output)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
