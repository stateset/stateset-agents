#!/usr/bin/env python3
"""Check the repo's own eval sets against the training data it ships or generates.

Each check pairs data the repo trains on with data it evaluates on, runs
:func:`stateset_agents.data.decontamination.find_contamination` (exact 13-gram
overlap plus row-level MinHash near duplicates at Jaccard >= 0.8), and records
the per-eval-set counts and the first contaminated rows as evidence. The
interpretation of each result lives in docs/DECONTAMINATION.md.

    python scripts/audit_decontamination.py --output audit.json

GSM8K needs the pinned dataset revision in the local Hugging Face cache (or
network access); pass ``--skip-gsm8k`` otherwise and it is recorded as skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stateset_agents.data.decontamination import (  # noqa: E402
    DecontaminationConfig,
    TextSource,
    builtin_eval_set,
    find_contamination,
    load_text_source,
    text_source_from_rows,
)

EVIDENCE_ROWS = 5
Check = Callable[[DecontaminationConfig], tuple[list[TextSource], list[TextSource]]]


def _rows(prefix: str, rows: list[Any], offset: int = 0) -> list[tuple[str, Any]]:
    return [(f"{prefix}[{offset + i}]", row) for i, row in enumerate(rows)]


def _support_bench(
    config: DecontaminationConfig,
) -> tuple[list[TextSource], list[TextSource]]:
    """scripts/run_phase0_benchmark.py and 03_first_finetune.py: 16 train / 8 eval."""
    from stateset_agents.data.customer_support_bench import load_support_scenarios

    queries = [{"user_query": s.user_query} for s in load_support_scenarios()]
    return (
        [
            text_source_from_rows(
                "support-bench[:16]", _rows("support", queries[:16]), config=config
            )
        ],
        [
            text_source_from_rows(
                "support-bench[16:]", _rows("support", queries[16:], 16), config=config
            )
        ],
    )


def _tool_bench(
    config: DecontaminationConfig,
) -> tuple[list[TextSource], list[TextSource]]:
    """The phase-0 tool adapter caps training at all but the last scenario."""
    from stateset_agents.data.tool_calling_bench import load_tool_call_scenarios

    queries = [{"user_query": s.user_query} for s in load_tool_call_scenarios()]
    return (
        [
            text_source_from_rows(
                "tool-bench[:-1]", _rows("tool", queries[:-1]), config=config
            )
        ],
        [
            text_source_from_rows(
                "tool-bench[-1:]",
                _rows("tool", queries[-1:], len(queries) - 1),
                config=config,
            )
        ],
    )


def _starter_scenarios(name: str) -> list[dict[str, Any]]:
    from stateset_agents.scaffolding import starter

    text = getattr(starter, name)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _cookbook_eval(
    config: DecontaminationConfig,
) -> tuple[list[TextSource], list[TextSource]]:
    """Cookbook Recipe 1 trains on the starter scenarios; Recipe 5 scores the
    checkpoint on examples/sample_eval_set.jsonl. The bundled support corpus is
    the other training set users are pointed at."""
    from stateset_agents.data.customer_support_bench import load_support_scenarios

    return (
        [
            text_source_from_rows(
                "starter-scenarios.jsonl",
                _rows("starter", _starter_scenarios("_CS_SCENARIOS")),
                config=config,
            ),
            text_source_from_rows(
                "support-bench",
                _rows(
                    "support",
                    [{"user_query": s.user_query} for s in load_support_scenarios()],
                ),
                config=config,
            ),
        ],
        [load_text_source(ROOT / "examples/sample_eval_set.jsonl", config=config)],
    )


def _starter_tools(
    config: DecontaminationConfig,
) -> tuple[list[TextSource], list[TextSource]]:
    """The tool-calling starter trains on its scenarios; the bundled tool bench
    is the eval the docs point at."""
    return (
        [
            text_source_from_rows(
                "starter-tool-scenarios.jsonl",
                _rows("starter-tool", _starter_scenarios("_TOOL_SCENARIOS")),
                config=config,
            )
        ],
        [builtin_eval_set("tool-calling-bench", config=config)],
    )


def _gsm8k(config: DecontaminationConfig) -> tuple[list[TextSource], list[TextSource]]:
    return (
        [builtin_eval_set("gsm8k-train", config=config)],
        [builtin_eval_set("gsm8k-test", config=config)],
    )


#: The domain the eval-ladder tests use (tests/unit/test_eval_ladder.py).
_LADDER_ISSUES = {
    "password": (
        "my password stopped working",
        "I've sent a secure reset link.",
        "reset link",
        "do not reset my password",
    ),
    "vpn": (
        "the VPN won't connect",
        "I've re-provisioned your VPN profile.",
        "vpn profile",
        "leave the VPN alone",
    ),
    "printer": (
        "the printer shows offline",
        "I've restarted the print spooler.",
        "print spooler",
        "don't touch the printer",
    ),
    "disk": (
        "my disk is full",
        "I've cleared the update cache.",
        "update cache",
        "do not delete anything",
    ),
    "email": (
        "Outlook stopped syncing",
        "I've rebuilt your sync profile.",
        "sync profile",
        "don't rebuild my mailbox",
    ),
}


def _ladder(depth: int, refusal: float) -> Check:
    def check(
        config: DecontaminationConfig,
    ) -> tuple[list[TextSource], list[TextSource]]:
        """Ladder eval prompts against its training rows and harvest prompts
        (the harvest is sampled and curated into the next generation's SFT)."""
        from stateset_agents.training.eval_ladder import DomainSpec, Issue, build_ladder

        spec = DomainSpec(
            persona="Byte @ TechNest",
            ref_label="Ticket #{ref}",
            issues={name: Issue(*fields) for name, fields in _LADDER_ISSUES.items()},
        )
        kit = build_ladder(spec, depth=depth, refusal_fraction=refusal, seed=0)
        return (
            [
                text_source_from_rows(
                    "ladder-train", _rows("train", kit["train"]), config=config
                ),
                text_source_from_rows(
                    "ladder-harvest", _rows("harvest", kit["harvest"]), config=config
                ),
            ],
            [
                text_source_from_rows(
                    "ladder-eval", _rows("eval", kit["eval"]), config=config
                )
            ],
        )

    return check


def _support_multiturn(
    config: DecontaminationConfig,
) -> tuple[list[TextSource], list[TextSource]]:
    """The published stateset/customer-support-multiturn-v1 generator (seed 0)."""
    from stateset_agents.data.support_multiturn import (
        build_support_dataset,
        export_dataset,
    )

    with tempfile.TemporaryDirectory() as directory:
        export_dataset(build_support_dataset(), Path(directory))
        return (
            [load_text_source(Path(directory) / "train.jsonl", config=config)],
            [load_text_source(Path(directory) / "eval.jsonl", config=config)],
        )


def _files(train: str, evals: str) -> Check:
    def check(
        config: DecontaminationConfig,
    ) -> tuple[list[TextSource], list[TextSource]]:
        return (
            [load_text_source(ROOT / train, config=config, name=train)],
            [load_text_source(ROOT / evals, config=config, name=evals)],
        )

    return check


def _industry_pilot_checks() -> dict[str, Check]:
    checks = {}
    for prepared in sorted(
        (ROOT / "benchmark_results/industry_pilot").glob("*/*/*/prepared")
    ):
        relative = prepared.relative_to(ROOT).as_posix()
        checks[f"industry-pilot:{relative}"] = _files(
            f"{relative}/train.jsonl", f"{relative}/validation.jsonl"
        )
    return checks


#: Checks re-run with stricter or narrower settings than the default.
VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    # Rows whose n-grams are mostly in training data, not shared template turns.
    "support-multiturn-v1:min-overlap-0.5": (
        "support-multiturn-v1",
        {"min_overlap": 0.5},
    ),
    # Questions only, numbers masked: GSM8K's templated problem families.
    "gsm8k-questions-masked-digits": (
        "gsm8k-train-vs-test",
        {"fields": ("question",), "mask_digits": True},
    ),
}


def all_checks(skip_gsm8k: bool) -> dict[str, Check]:
    checks: dict[str, Check] = {
        "support-bench-split": _support_bench,
        "tool-calling-bench-split": _tool_bench,
        "cookbook-sample-eval": _cookbook_eval,
        "starter-tools-vs-tool-calling-bench": _starter_tools,
        "support-multiturn-v1": _support_multiturn,
        "product-use-v0.3": _files(
            "benchmarks/product_use/examples.v0.3.jsonl",
            "benchmarks/product_use/tasks.v0.3.public.json",
        ),
    }
    for depth, refusal in ((2, 0.0), (2, 0.5), (3, 0.0), (3, 0.5)):
        checks[f"eval-ladder:depth{depth}-refusal{refusal}"] = _ladder(depth, refusal)
    checks.update(_industry_pilot_checks())
    if not skip_gsm8k:
        checks["gsm8k-train-vs-test"] = _gsm8k
    return checks


def _summarize(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "train": [
            {key: source[key] for key in ("name", "rows", "sha256")}
            for source in report["train"]["sources"]
        ],
        "summary": report["summary"],
        "eval_sets": {
            name: {
                **{
                    key: entry[key]
                    for key in (
                        "rows",
                        "sha256",
                        "checked_rows",
                        "unchecked_short_segments",
                        "contaminated_rows",
                        "contamination_rate",
                    )
                },
                "max_ngram_overlap": max(
                    (row["ngram_overlap"] for row in entry["contaminated"]), default=0.0
                ),
                "near_duplicate_rows": sum(
                    "near_duplicate" in row["match"] for row in entry["contaminated"]
                ),
                "evidence": entry["contaminated"][:EVIDENCE_ROWS],
            }
            for name, entry in report["eval_sets"].items()
        },
    }


def run_audit(skip_gsm8k: bool = False) -> dict[str, Any]:
    config = DecontaminationConfig(near_duplicates=True)
    checks = all_checks(skip_gsm8k)
    results: dict[str, Any] = {}
    for name, check in checks.items():
        train, evals = check(config)
        results[name] = _summarize(find_contamination(train, evals, config).report)
    for name, (base, overrides) in VARIANTS.items():
        if base not in checks:
            continue
        variant = replace(config, **overrides)
        train, evals = checks[base](variant)
        results[name] = {
            "settings": variant.to_dict(),
            **_summarize(find_contamination(train, evals, variant).report),
        }
    return {
        "schema_version": 1,
        "kind": "decontamination_audit",
        "generated_by": "scripts/audit_decontamination.py",
        "config": config.to_dict(),
        "gsm8k": "skipped" if skip_gsm8k else "checked",
        "checks": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-gsm8k", action="store_true")
    args = parser.parse_args(argv)
    audit = run_audit(skip_gsm8k=args.skip_gsm8k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    for name, result in audit["checks"].items():
        for eval_name, entry in result["eval_sets"].items():
            print(
                f"{name}: {eval_name}: {entry['contaminated_rows']}/{entry['rows']} "
                f"contaminated (near-dup {entry['near_duplicate_rows']}, "
                f"max overlap {entry['max_ngram_overlap']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
