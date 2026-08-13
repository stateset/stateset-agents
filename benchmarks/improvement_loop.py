#!/usr/bin/env python3
"""Measure the closed improvement loop: ingest -> grade -> curate, as a number.

The framework's differentiator is the closed improvement loop — third-party
conversation logs go in (``stateset-agents ingest``), get graded and curated
(``stateset-agents improve run``), and a training-ready dataset comes out.
This benchmark makes that loop's value measurable instead of a claim:

1. Generate a **deterministic, seeded synthetic corpus** of raw conversation
   logs in OpenAI chat-completions JSONL format, with a controlled mix of
   good and bad assistant behavior. The planted mix is the ground truth.
2. Run the **real** ingest -> grade -> curate pipeline via the library API
   (``stateset_agents.cli_improve.run_improve`` — the same function behind
   ``stateset-agents improve run``). Nothing is reimplemented here.
3. Report loop quality against the planted ground truth: precision / recall /
   F1 of curation (did it keep the good and drop the bad), dataset yield, and
   the grade distribution — as a human table plus JSON.
4. Exit non-zero when precision or recall falls below configurable floors.

Measured on the default corpus (60 conversations, 60% good, seed 42,
threshold 0.7, reward customer_support):

* precision = 1.0 — the ``deflection`` bad flavor (a 10+ word, safety-clean
  but unhelpful reply) used to slip past the rule-based reward at 0.75; the
  resolution/concreteness component of ``SupportRewardComposite`` now scores
  it ~0.625, below the 0.7 curation threshold. (Historical: precision was
  0.818 before that component existed.)
* recall = 1.0 — every planted good reply is curated.

Floors default slightly below measured (coverage-ratchet philosophy):
``--min-precision 0.95``, ``--min-recall 0.95``.

Usage::

    python benchmarks/improvement_loop.py
    python benchmarks/improvement_loop.py --conversations 200 --seed 7
    python benchmarks/improvement_loop.py --output results.json
    make benchmark-loop

No GPU, no network, no API keys — pure rule-based grading, runs in seconds.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_MIN_PRECISION = 0.95
DEFAULT_MIN_RECALL = 0.95

TOPICS = [
    ("a refund for order", "refund"),
    ("tracking for order", "tracking"),
    ("cancelling order", "cancellation"),
    ("an exchange for order", "exchange"),
    ("a missing item in order", "missing item"),
]

BAD_FLAVORS = ("rude", "curt", "deflection")


@dataclass
class PlantedConversation:
    """One synthetic conversation plus its planted ground-truth label."""

    index: int
    label: str  # "good" or "bad"
    flavor: str  # "good", "rude", "curt", "deflection"
    messages: list[dict[str, str]] = field(default_factory=list)


def _good_reply(topic_phrase: str, order_id: int) -> str:
    return (
        f"Thank you for reaching out — I'd be happy to help with {topic_phrase} "
        f"#{order_id}. I understand how important this is, and I've pulled up the "
        f"details for order #{order_id} right now. Please give me one moment and "
        "I'll walk you through the next steps."
    )


def _bad_reply(flavor: str, topic_phrase: str, order_id: int) -> str:
    if flavor == "rude":
        # Trips the safety gate ("you're stupid") -> score 0.0.
        return (
            f"Honestly, you're stupid for asking about {topic_phrase} "
            f"#{order_id}. Read the FAQ."
        )
    if flavor == "curt":
        # One word -> heavy brand-voice length penalty.
        return "ok"
    # "deflection": safety-clean, 10+ words, zero politeness, zero help.
    # Historically scored ~0.75 (a documented grader gap); the resolution/
    # concreteness component now penalizes the deflection phrasing and the
    # absence of any commitment/timeframe, scoring it ~0.625 (< 0.7).
    return (
        f"Regarding {topic_phrase} #{order_id}: that is not something this "
        "channel handles. Check the website yourself for more information."
    )


def generate_corpus(
    n_conversations: int, good_fraction: float, seed: int
) -> list[PlantedConversation]:
    """Build a deterministic corpus with a planted good/bad mix.

    Bad conversations cycle through the three bad flavors round-robin so the
    corpus exercises the safety gate, the length penalty, and the known
    deflection gap at every size. Order IDs are unique per conversation, so
    curation's exact (prompt, response) dedup never collapses two planted
    examples.
    """
    if not 0.0 <= good_fraction <= 1.0:
        raise ValueError(f"good_fraction must be in [0, 1], got {good_fraction}")
    n_good = round(n_conversations * good_fraction)
    labels = ["good"] * n_good + ["bad"] * (n_conversations - n_good)
    rng = random.Random(seed)
    rng.shuffle(labels)

    corpus: list[PlantedConversation] = []
    bad_seen = 0
    for i, label in enumerate(labels):
        topic_phrase, topic_word = TOPICS[i % len(TOPICS)]
        order_id = 10_000 + i
        user = f"Hi, I need help with {topic_phrase} #{order_id}, {topic_word} please."
        if label == "good":
            flavor = "good"
            assistant = _good_reply(topic_phrase, order_id)
        else:
            flavor = BAD_FLAVORS[bad_seen % len(BAD_FLAVORS)]
            bad_seen += 1
            assistant = _bad_reply(flavor, topic_phrase, order_id)
        corpus.append(
            PlantedConversation(
                index=i,
                label=label,
                flavor=flavor,
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
            )
        )
    return corpus


def write_corpus_jsonl(corpus: list[PlantedConversation], path: Path) -> None:
    """Write the corpus as OpenAI chat-completions JSONL (one line per convo)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for convo in corpus:
            f.write(json.dumps({"messages": convo.messages}) + "\n")


_SOURCE_RE = re.compile(r"conversation_(\d+)\.jsonl$")


def _curated_indices(curated_path: Path) -> list[int]:
    """Map curated.jsonl rows back to corpus indices via their source name.

    ``improve run`` ingests line N of the corpus into conversation_N.jsonl, so
    the ``source`` field of every curated example identifies the planted
    conversation it came from.
    """
    indices: list[int] = []
    if not curated_path.exists():
        return indices
    for line in curated_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        match = _SOURCE_RE.search(str(entry.get("source", "")))
        if match:
            indices.append(int(match.group(1)))
    return indices


def compute_metrics(
    corpus: list[PlantedConversation],
    curated_indices: list[int],
    improve_summary: dict[str, Any],
) -> dict[str, Any]:
    """Score curation against the planted ground truth."""
    good = {c.index for c in corpus if c.label == "good"}
    bad = {c.index for c in corpus if c.label == "bad"}
    kept = set(curated_indices)

    tp = len(kept & good)
    fp = len(kept & bad)
    fn = len(good - kept)
    tn = len(bad - kept)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    false_positive_flavors: dict[str, int] = {}
    by_index = {c.index: c for c in corpus}
    for idx in kept & bad:
        flavor = by_index[idx].flavor
        false_positive_flavors[flavor] = false_positive_flavors.get(flavor, 0) + 1

    total = len(corpus)
    return {
        "ground_truth": {
            "conversations": total,
            "planted_good": len(good),
            "planted_bad": len(bad),
        },
        "confusion": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yield": (len(kept) / total) if total else 0.0,
        "curated_count": len(kept),
        "false_positive_flavors": false_positive_flavors,
        "grade_distribution": {
            "mean_score": improve_summary.get("mean_score", 0.0),
            "assistant_turns": improve_summary.get("assistant_turn_count", 0),
            "above_threshold": improve_summary.get("above_threshold_count", 0),
            "threshold": improve_summary.get("threshold", 0.0),
        },
    }


def run_benchmark(
    *,
    conversations: int,
    good_fraction: float,
    seed: int,
    reward: str,
    threshold: float,
    workdir: Path,
) -> dict[str, Any]:
    """Generate the corpus, run the real pipeline, and score it."""
    from stateset_agents.cli_improve import CURATED_FILENAME, run_improve

    corpus = generate_corpus(conversations, good_fraction, seed)
    corpus_path = workdir / "raw_logs.jsonl"
    write_corpus_jsonl(corpus, corpus_path)

    loop_dir = workdir / "improve_output"
    improve_summary = run_improve(
        transcripts=str(corpus_path),
        reward=reward,
        output=str(loop_dir),
        threshold=threshold,
        format="openai",
    )

    metrics = compute_metrics(
        corpus, _curated_indices(loop_dir / CURATED_FILENAME), improve_summary
    )
    metrics["config"] = {
        "conversations": conversations,
        "good_fraction": good_fraction,
        "seed": seed,
        "reward": reward,
        "threshold": threshold,
        "corpus_path": str(corpus_path),
        "curated_path": improve_summary.get("curated_path", ""),
    }
    return metrics


def render_table(metrics: dict[str, Any]) -> str:
    gt = metrics["ground_truth"]
    cm = metrics["confusion"]
    gd = metrics["grade_distribution"]
    lines = [
        "Improvement-loop benchmark (ingest -> grade -> curate vs planted truth)",
        "=" * 72,
        f"{'Corpus':<28} {gt['conversations']} conversations "
        f"({gt['planted_good']} good / {gt['planted_bad']} bad)",
        f"{'Reward / threshold':<28} {metrics['config']['reward']} / "
        f"{gd['threshold']}",
        f"{'Mean grade':<28} {gd['mean_score']:.3f} "
        f"({gd['above_threshold']}/{gd['assistant_turns']} turns above threshold)",
        "-" * 72,
        f"{'Curated (yield)':<28} {metrics['curated_count']} "
        f"({metrics['yield']:.1%} of corpus)",
        f"{'Precision':<28} {metrics['precision']:.3f}   "
        f"(TP={cm['true_positive']}, FP={cm['false_positive']})",
        f"{'Recall':<28} {metrics['recall']:.3f}   "
        f"(FN={cm['false_negative']}, TN={cm['true_negative']})",
        f"{'F1':<28} {metrics['f1']:.3f}",
    ]
    if metrics["false_positive_flavors"]:
        flavors = ", ".join(
            f"{k}={v}" for k, v in sorted(metrics["false_positive_flavors"].items())
        )
        lines.append(f"{'False positives by flavor':<28} {flavors}")
    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--conversations",
        type=int,
        default=60,
        help="Synthetic conversations to generate (default: 60).",
    )
    parser.add_argument(
        "--good-fraction",
        type=float,
        default=0.6,
        help="Fraction of conversations with planted-good replies (default: 0.6).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Corpus RNG seed (default: 42)."
    )
    parser.add_argument(
        "--reward",
        default="customer_support",
        help="Rule-based reward for grading (default: customer_support).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Curation score threshold, same default as `improve run` (0.7).",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=DEFAULT_MIN_PRECISION,
        help=f"Fail if curation precision drops below this "
        f"(default: {DEFAULT_MIN_PRECISION}, measured 1.0 on defaults).",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=DEFAULT_MIN_RECALL,
        help=f"Fail if curation recall drops below this "
        f"(default: {DEFAULT_MIN_RECALL}, measured 1.0 on defaults).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Also write the metrics JSON to this file.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Directory for corpus + pipeline output (default: a temp dir).",
    )
    args = parser.parse_args(argv)

    if args.workdir is not None:
        args.workdir.mkdir(parents=True, exist_ok=True)
        workdir = args.workdir
        cleanup_ctx = None
    else:
        cleanup_ctx = tempfile.TemporaryDirectory(prefix="improvement_loop_bench_")
        workdir = Path(cleanup_ctx.name)

    try:
        metrics = run_benchmark(
            conversations=args.conversations,
            good_fraction=args.good_fraction,
            seed=args.seed,
            reward=args.reward,
            threshold=args.threshold,
            workdir=workdir,
        )
    finally:
        if cleanup_ctx is not None:
            cleanup_ctx.cleanup()

    floors = {"min_precision": args.min_precision, "min_recall": args.min_recall}
    passed = (
        metrics["precision"] >= args.min_precision
        and metrics["recall"] >= args.min_recall
    )
    metrics["floors"] = floors
    metrics["passed"] = passed

    print(render_table(metrics))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"Metrics written to {args.output}", file=sys.stderr)

    if not passed:
        print(
            f"FAIL: precision={metrics['precision']:.3f} "
            f"(floor {args.min_precision}) "
            f"recall={metrics['recall']:.3f} (floor {args.min_recall})",
            file=sys.stderr,
        )
        return 1
    print(
        f"PASS: precision={metrics['precision']:.3f} >= {args.min_precision}, "
        f"recall={metrics['recall']:.3f} >= {args.min_recall}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
