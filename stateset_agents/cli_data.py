"""``stateset-agents data`` — inspect and clean training data.

``data decontaminate`` compares training files with eval sets by exact
n-gram overlap (and optionally MinHash near duplicates), writes the evidence
report, and can filter or flag the contaminated rows. The heavy lifting lives
in :mod:`stateset_agents.data.decontamination`; this module only parses
options, reads and writes files, and maps outcomes to exit codes (``0`` clean
or action applied, ``1`` contamination with ``--fail-on-contamination``,
``2`` invalid input).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

import typer

data_app = typer.Typer(help="Inspect and clean training data.")

_ACTIONS = ("report", "filter", "flag")
_TARGETS = ("train", "eval")


def _emit(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


def _fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=2)


def _write_new(path: Path, text: str) -> None:
    """Write a file that must not exist yet (outputs are never overwritten)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _render_rows(path: Path, rows: list[Any]) -> str:
    if path.suffix.lower() == ".json":
        return json.dumps(rows, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    return "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )


def _check_options(
    action: str,
    target: str,
    output: Path | None,
    train: list[str],
    eval_sets: list[str],
) -> None:
    if action not in _ACTIONS:
        _fail(f"--action must be one of {', '.join(_ACTIONS)}")
    if target not in _TARGETS:
        _fail(f"--target must be one of {', '.join(_TARGETS)}")
    if action == "report":
        if output is not None:
            _fail("--output needs --action filter or flag")
        return
    if output is None:
        _fail(f"--action {action} needs --output for the cleaned rows")
    if target == "train" and (len(train) != 1 or train[0].startswith("builtin:")):
        _fail("--action filter/flag on training rows needs exactly one --train file")
    if target == "eval" and (
        len(eval_sets) != 1 or eval_sets[0].startswith("builtin:")
    ):
        _fail("--target eval needs exactly one --eval file")


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": report["method"],
        "summary": report["summary"],
        "eval_sets": {
            name: {
                key: entry[key]
                for key in (
                    "rows",
                    "checked_rows",
                    "unchecked_short_segments",
                    "contaminated_rows",
                    "contamination_rate",
                )
            }
            for name, entry in report["eval_sets"].items()
        },
    }


@data_app.command("decontaminate")
def decontaminate(
    train: list[str] = typer.Option(
        ...,
        "--train",
        "-t",
        help="Training JSONL/JSON (chat, preference or prompt rows), NAME=PATH, "
        "or builtin:NAME; repeatable.",
    ),
    eval_sets: list[str] = typer.Option(
        ...,
        "--eval",
        "-e",
        help="Eval set: a JSONL/JSON path, NAME=PATH, or builtin:NAME "
        "(support-bench, tool-calling-bench, gsm8k-test, gsm8k-train); repeatable.",
    ),
    report: Path | None = typer.Option(
        None, "--report", help="Write the full JSON evidence report (new file)."
    ),
    action: str = typer.Option(
        "report", "--action", help="report (default), filter, or flag."
    ),
    target: str = typer.Option(
        "train",
        "--target",
        help="Rows the action changes: train (default) or eval (score a clean subset).",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Cleaned rows for --action filter/flag (new .jsonl or .json file).",
    ),
    ngram_size: int = typer.Option(13, "--ngram-size", min=1, max=64),
    min_tokens: int = typer.Option(
        8,
        "--min-tokens",
        min=1,
        max=64,
        help="Eval segments shorter than --ngram-size but at least this long "
        "are matched whole; shorter ones are reported as unchecked.",
    ),
    min_overlap: float = typer.Option(
        0.0,
        "--min-overlap",
        min=0.0,
        max=1.0,
        help="Fraction of an eval row's n-grams that must occur in training "
        "data (0 = any single n-gram).",
    ),
    mask_digits: bool = typer.Option(
        False, "--mask-digits", help="Treat every number as the same token."
    ),
    include_system: bool = typer.Option(
        False,
        "--include-system",
        help="Also compare system and developer messages (skipped by default).",
    ),
    fields: list[str] = typer.Option(
        [], "--field", help="Row keys to read instead of the default text keys."
    ),
    near_duplicates: bool = typer.Option(
        False, "--near-duplicates", help="Also find MinHash/LSH near duplicates."
    ),
    threshold: float = typer.Option(
        0.8,
        "--threshold",
        min=0.01,
        max=1.0,
        help="Minimum exact Jaccard similarity for a near duplicate.",
    ),
    fail_on_contamination: bool = typer.Option(
        False,
        "--fail-on-contamination",
        help="Exit 1 when any eval row is contaminated (after writing outputs).",
    ),
) -> None:
    """Find eval-set text in training data; report, filter or flag it."""
    from stateset_agents.data.decontamination import (
        DEFAULT_ROLES,
        DecontaminationConfig,
        apply_decontamination,
        find_contamination,
        read_rows,
        resolve_eval_source,
    )

    _check_options(action, target, output, train, eval_sets)
    try:
        config = DecontaminationConfig(
            ngram_size=ngram_size,
            min_tokens=min(min_tokens, ngram_size),
            mask_digits=mask_digits,
            roles=(
                (*DEFAULT_ROLES, "system", "developer")
                if include_system
                else DEFAULT_ROLES
            ),
            fields=tuple(fields) or None,
            min_overlap=min_overlap,
            near_duplicates=near_duplicates,
            near_duplicate_threshold=threshold,
        )
        train_sources = [resolve_eval_source(spec, config=config) for spec in train]
        evals = [resolve_eval_source(spec, config=config) for spec in eval_sets]
        if not any(record.segments for s in train_sources for record in s.records):
            raise ValueError("No training row has text; check --field/--include-system")
        result = find_contamination(train_sources, evals, config)
        summary = _summary(result.report)
        if report is not None:
            _write_new(
                report, json.dumps(result.report, indent=2, sort_keys=True) + "\n"
            )
            summary["report"] = str(report)
        if action != "report" and output is not None:
            source = Path((train_sources if target == "train" else evals)[0].path or "")
            rows, _digest = read_rows(source)
            kept, affected = apply_decontamination(
                rows,
                result,
                "filter" if action == "filter" else "flag",
                target="train" if target == "train" else "eval",
                eval_set=evals[0].name,
            )
            _write_new(output, _render_rows(output, kept))
            summary["action"] = {
                "mode": action,
                "target": target,
                "affected_rows": affected,
                "output_rows": len(kept),
                "output": str(output),
            }
    except FileExistsError as exc:
        _fail(f"Refusing to overwrite {exc.filename}; choose a new output path")
    except (OSError, ValueError, ImportError) as exc:
        _fail(f"Could not decontaminate: {exc}")
    _emit(summary)
    if fail_on_contamination and not result.report["summary"]["clean"]:
        raise typer.Exit(code=1)
