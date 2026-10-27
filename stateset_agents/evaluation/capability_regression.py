"""General-capability regression checks through lm-evaluation-harness.

Fine-tuning on a narrow domain can quietly cost general ability. This module
runs a named list of lm-evaluation-harness tasks (``hellaswag``,
``arc_easy``, ``gsm8k``...) against a local Hugging Face checkpoint, with or
without a LoRA adapter, records the scores in the repo's evaluation report
shape (``schema_version``, ``base_model``, ``seed``, ``settings``,
``selected_checkpoint``) and compares a candidate with a baseline under a
configurable allowed regression, in the gate shape the agent-run comparisons
use (``passed``, ``gates``, ``thresholds``, ``evidence_hashes``).

lm-evaluation-harness is optional (``pip install
'stateset-agents[eval-harness]'``). The module imports without it; asking for
a run raises :class:`CapabilityEvalUnavailable` naming the extra. Any object
with lm-eval's ``simple_evaluate`` can be passed as ``harness`` instead.

Command line::

    python -m stateset_agents.evaluation.capability_regression run \\
        --checkpoint Qwen/Qwen3.5-0.8B --adapter outputs/sft_v1 \\
        --tasks hellaswag,arc_easy --limit 200 --output candidate.json
    python -m stateset_agents.evaluation.capability_regression compare \\
        --baseline base.json --candidate candidate.json --max-regression 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stateset_agents.core.errors import StateSetError
from stateset_agents.evaluation.agent_runs import content_hash

__all__ = [
    "CapabilityEvalSettings",
    "CapabilityEvalUnavailable",
    "SCHEMA_VERSION",
    "capability_report",
    "compare_capability_reports",
    "load_harness",
    "main",
    "run_capability_eval",
    "validate_capability_report",
]

SCHEMA_VERSION = 1
INSTALL_HINT = "pip install 'stateset-agents[eval-harness]'"

#: Preferred headline metric per task when the caller names none: the first
#: one a task reports wins.
PRIMARY_METRICS = (
    "acc_norm",
    "acc",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match",
    "mc2",
    "f1",
    "em",
    "pass@1",
    "bleu",
    "word_perplexity",
)
#: Metrics where lower is better when the harness does not say.
LOWER_IS_BETTER = frozenset(
    {"perplexity", "word_perplexity", "byte_perplexity", "bits_per_byte", "ter"}
)


class CapabilityEvalUnavailable(StateSetError, ImportError):
    """lm-evaluation-harness is not installed (or failed to import)."""

    message_only_str = True


@dataclass(frozen=True)
class CapabilityEvalSettings:
    """Everything that must match for two capability reports to be compared."""

    tasks: tuple[str, ...]
    num_fewshot: int | None = None
    #: Examples per task (int) or fraction (float); ``None`` runs every example.
    limit: int | float | None = None
    batch_size: int | str = 1
    dtype: str = "bfloat16"
    seed: int = 1234
    apply_chat_template: bool = False

    def __post_init__(self) -> None:
        tasks = tuple(self.tasks)
        if not tasks or any(not isinstance(t, str) or not t.strip() for t in tasks):
            raise ValueError("tasks must be a nonempty list of task names")
        if len(set(tasks)) != len(tasks):
            raise ValueError("tasks must be unique")
        object.__setattr__(self, "tasks", tasks)
        if self.num_fewshot is not None and (
            type(self.num_fewshot) is not int or self.num_fewshot < 0
        ):
            raise ValueError("num_fewshot must be a nonnegative integer")
        if self.limit is not None:
            if isinstance(self.limit, bool) or not isinstance(self.limit, (int, float)):
                raise ValueError("limit must be a number")
            if isinstance(self.limit, float) and not 0 < self.limit <= 1:
                raise ValueError("a fractional limit must be in (0, 1]")
            if isinstance(self.limit, int) and self.limit < 1:
                raise ValueError("limit must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not (
            (type(self.batch_size) is int and self.batch_size > 0)
            or (isinstance(self.batch_size, str) and self.batch_size.startswith("auto"))
        ):
            raise ValueError("batch_size must be a positive integer or 'auto'")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tasks"] = list(self.tasks)
        return data


def load_harness() -> Any:
    """Import lm-evaluation-harness, or say how to install it."""
    try:
        import lm_eval
    except ImportError as exc:
        raise CapabilityEvalUnavailable(
            "General-capability evals need lm-evaluation-harness, which is not "
            f"installed: {INSTALL_HINT}"
        ) from exc
    return lm_eval


def _model_args(
    checkpoint: str,
    *,
    adapter: str | None,
    dtype: str,
    revision: str | None,
    trust_remote_code: bool,
) -> str:
    """lm-eval ``hf`` model arguments; ``peft`` loads a LoRA adapter on top."""
    for name, value in (("checkpoint", checkpoint), ("adapter", adapter)):
        if value is not None and ("," in value or "=" in value or not value.strip()):
            raise ValueError(f"{name} must be a nonempty path or id without ',' or '='")
    parts = [f"pretrained={checkpoint}", f"dtype={dtype}"]
    if adapter is not None:
        parts.append(f"peft={adapter}")
    if revision is not None:
        parts.append(f"revision={revision}")
    if trust_remote_code:
        parts.append("trust_remote_code=True")
    return ",".join(parts)


def run_capability_eval(
    checkpoint: str,
    settings: CapabilityEvalSettings,
    *,
    adapter: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    device: str | None = None,
    harness: Any = None,
    primary_metrics: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run ``settings.tasks`` on a local checkpoint (plus adapter) and report.

    ``checkpoint`` is the base model (a local directory or Hub id the local
    cache can serve); ``adapter`` is a PEFT adapter directory, such as the
    output of ``python -m stateset_agents.training.sft``.
    """
    harness = harness if harness is not None else load_harness()
    raw = harness.simple_evaluate(
        model="hf",
        model_args=_model_args(
            checkpoint,
            adapter=adapter,
            dtype=settings.dtype,
            revision=revision,
            trust_remote_code=trust_remote_code,
        ),
        tasks=list(settings.tasks),
        num_fewshot=settings.num_fewshot,
        limit=settings.limit,
        batch_size=settings.batch_size,
        device=device,
        apply_chat_template=settings.apply_chat_template,
        random_seed=settings.seed,
        numpy_random_seed=settings.seed,
        torch_random_seed=settings.seed,
        fewshot_random_seed=settings.seed,
        log_samples=False,
    )
    if not isinstance(raw, dict):
        raise ValueError("The harness returned no results (non-primary process?)")
    return capability_report(
        raw,
        checkpoint=checkpoint,
        adapter=adapter,
        revision=revision,
        settings=settings,
        harness_version=str(getattr(harness, "__version__", "unknown")),
        primary_metrics=primary_metrics,
    )


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _task_entry(
    task: str,
    scores: Mapping[str, Any],
    raw: Mapping[str, Any],
    primary: str | None,
) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    stderr: dict[str, float] = {}
    for key, value in scores.items():
        number = _finite(value)
        if number is None or key == "alias":
            continue
        name, _, filter_name = key.partition(",")
        if name.endswith("_stderr"):
            metric = name[: -len("_stderr")]
            label = metric if filter_name in ("", "none") else f"{metric},{filter_name}"
            stderr[label] = number
        else:
            label = name if filter_name in ("", "none") else f"{name},{filter_name}"
            metrics[label] = number
    if not metrics:
        raise ValueError(f"Task {task!r} reported no numeric metrics")
    if primary is None:
        primary = next((m for m in PRIMARY_METRICS if m in metrics), sorted(metrics)[0])
    if primary not in metrics:
        raise ValueError(f"Task {task!r} has no metric {primary!r}")
    declared = raw.get("higher_is_better", {}).get(task, {})
    higher = declared.get(primary.partition(",")[0])
    if not isinstance(higher, bool):
        higher = primary.partition(",")[0] not in LOWER_IS_BETTER
    samples = raw.get("n-samples", {}).get(task, {})
    return {
        "metrics": metrics,
        "stderr": stderr,
        "primary_metric": primary,
        "higher_is_better": higher,
        "score": metrics[primary],
        "samples": samples.get("effective") if isinstance(samples, dict) else None,
        "version": raw.get("versions", {}).get(task),
        "n_shot": raw.get("n-shot", {}).get(task),
    }


def capability_report(
    raw: Mapping[str, Any],
    *,
    checkpoint: str,
    settings: CapabilityEvalSettings,
    adapter: str | None = None,
    revision: str | None = None,
    harness_version: str = "unknown",
    primary_metrics: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Normalize an lm-eval ``simple_evaluate`` result into a capability report.

    Every requested task must have results. Scores keep lm-eval's metric names
    without the default ``,none`` filter suffix (``acc``, ``acc_norm``,
    ``exact_match,strict-match``); non-finite values (``N/A`` stderr) are
    dropped. ``primary_metrics`` picks each task's headline metric.
    """
    results = raw.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("Harness output has no 'results' mapping")
    missing = [task for task in settings.tasks if task not in results]
    if missing:
        raise ValueError(f"Harness output is missing tasks: {missing}")
    chosen = dict(primary_metrics or {})
    tasks = {
        task: _task_entry(task, results[task], raw, chosen.get(task))
        for task in settings.tasks
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "capability_eval",
        "harness": {"name": "lm-evaluation-harness", "version": harness_version},
        "base_model": checkpoint,
        "seed": settings.seed,
        "settings": settings.to_dict(),
        "selected_checkpoint": {
            "path": adapter if adapter is not None else checkpoint,
            "base_model": checkpoint,
            "adapter": adapter,
            "revision": revision,
        },
        "tasks": tasks,
        "metrics": {task: entry["score"] for task, entry in tasks.items()},
    }
    validate_capability_report(report)
    return report


def validate_capability_report(report: Any) -> None:
    """Reject malformed or incomplete capability reports (and NaN anywhere)."""
    if (
        not isinstance(report, Mapping)
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("kind") != "capability_eval"
    ):
        raise ValueError(
            f"Expected a schema_version {SCHEMA_VERSION} capability_eval report"
        )
    settings = report.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("Capability report settings are required")
    try:
        CapabilityEvalSettings(
            **{**settings, "tasks": tuple(settings.get("tasks") or ())}
        )
    except TypeError as exc:
        raise ValueError(f"Unknown capability eval settings: {exc}") from None
    checkpoint = report.get("selected_checkpoint")
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("path"), str
    ):
        raise ValueError("Selected checkpoint identity is required")
    tasks = report.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != set(settings["tasks"]):
        raise ValueError("Capability report must score exactly its settings' tasks")
    for task, entry in tasks.items():
        if (
            not isinstance(entry, Mapping)
            or entry.get("primary_metric") not in entry.get("metrics", {})
            or _finite(entry.get("score")) is None
            or not isinstance(entry.get("higher_is_better"), bool)
        ):
            raise ValueError(f"Invalid capability score for task {task!r}")
    content_hash(report)


def _comparable_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    # Batch size changes padding, not the measurement; everything else counts.
    return {key: value for key, value in settings.items() if key != "batch_size"}


def compare_capability_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    max_regression: float = 0.01,
    task_max_regression: Mapping[str, float] | None = None,
    relative: bool = False,
) -> dict[str, Any]:
    """Gate a candidate on its baseline's general-capability scores.

    A task regresses when its headline metric moves in the worse direction by
    more than the allowed amount: ``max_regression`` (or the task's entry in
    ``task_max_regression``) in the metric's own units, e.g. ``0.01`` is one
    accuracy point, or a fraction of the baseline score with ``relative``.
    The gate also requires both reports to use the same tasks and settings;
    a comparison across different few-shot counts or limits is refused.
    """
    validate_capability_report(baseline)
    validate_capability_report(candidate)
    overrides = dict(task_max_regression or {})
    for value in [max_regression, *overrides.values()]:
        if _finite(value) is None or value < 0:
            raise ValueError("Allowed regressions must be finite and nonnegative")
    unknown = sorted(set(overrides) - set(baseline["tasks"]))
    if unknown:
        raise ValueError(
            f"Per-task thresholds name tasks the baseline lacks: {unknown}"
        )
    same_settings = _comparable_settings(baseline["settings"]) == _comparable_settings(
        candidate["settings"]
    )
    tasks: dict[str, Any] = {}
    for task, before in baseline["tasks"].items():
        after = candidate["tasks"].get(task)
        allowed = float(overrides.get(task, max_regression))
        if relative:
            allowed *= abs(before["score"])
        entry: dict[str, Any] = {
            "metric": before["primary_metric"],
            "higher_is_better": before["higher_is_better"],
            "baseline": before["score"],
            "baseline_stderr": before["stderr"].get(before["primary_metric"]),
            "allowed_regression": allowed,
        }
        if after is None or after["primary_metric"] != before["primary_metric"]:
            entry.update(candidate=None, delta=None, regression=None, regressed=True)
        else:
            delta = after["score"] - before["score"]
            regression = -delta if before["higher_is_better"] else delta
            entry.update(
                candidate=after["score"],
                candidate_stderr=after["stderr"].get(before["primary_metric"]),
                delta=delta,
                regression=regression,
                regressed=regression > allowed + 1e-12,
            )
        tasks[task] = entry
    regressions = [task for task, entry in tasks.items() if entry["regressed"]]
    gates = {
        "same_settings": same_settings,
        "all_tasks_scored": all(
            entry["candidate"] is not None for entry in tasks.values()
        ),
        "no_task_regression": not regressions,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "capability_regression",
        "passed": all(gates.values()),
        "gates": gates,
        "thresholds": {
            "max_regression": max_regression,
            "task_max_regression": overrides,
            "relative": relative,
        },
        "regressions": regressions,
        "tasks": tasks,
        "checkpoints": {
            "baseline": dict(baseline["selected_checkpoint"]),
            "candidate": dict(candidate["selected_checkpoint"]),
        },
        "evidence_hashes": {
            "baseline": content_hash(baseline),
            "candidate": content_hash(candidate),
        },
        "interpretation": (
            "Scores come from lm-evaluation-harness on the stated tasks and "
            "settings; a pass bounds regression on those tasks only. Compare "
            "deltas with the reported stderr before reading small moves as real."
        ),
    }


def _task_thresholds(values: Sequence[str]) -> dict[str, float]:
    thresholds = {}
    for value in values:
        task, separator, amount = value.partition("=")
        if not separator or not task:
            raise ValueError(f"Expected TASK=AMOUNT, got {value!r}")
        thresholds[task] = float(amount)
    return thresholds


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Score a checkpoint (and adapter) on tasks.")
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--adapter")
    run.add_argument("--revision")
    run.add_argument(
        "--tasks", required=True, help="Comma-separated lm-eval task names."
    )
    run.add_argument("--num-fewshot", type=int)
    run.add_argument(
        "--limit", type=float, help="Examples per task (>= 1) or a fraction (< 1)."
    )
    run.add_argument("--batch-size", default="1")
    run.add_argument("--dtype", default="bfloat16")
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--device")
    run.add_argument("--apply-chat-template", action="store_true")
    run.add_argument("--trust-remote-code", action="store_true")
    run.add_argument("--output", type=Path, required=True, help="New JSON report file.")
    compare = commands.add_parser(
        "compare", help="Gate a candidate report on a baseline."
    )
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--max-regression", type=float, default=0.01)
    compare.add_argument(
        "--task-max-regression", action="append", default=[], metavar="TASK=AMOUNT"
    )
    compare.add_argument("--relative", action="store_true")
    compare.add_argument("--output", type=Path, help="New JSON gate report file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """``run`` exits 0 after writing a report; ``compare`` exits 1 on a failed gate.

    Invalid input or a missing harness exits 2.
    """
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            limit = args.limit
            if limit is not None and limit >= 1:
                limit = int(limit)
            batch = (
                args.batch_size
                if args.batch_size.startswith("auto")
                else int(args.batch_size)
            )
            settings = CapabilityEvalSettings(
                tasks=tuple(t.strip() for t in args.tasks.split(",") if t.strip()),
                num_fewshot=args.num_fewshot,
                limit=limit,
                batch_size=batch,
                dtype=args.dtype,
                seed=args.seed,
                apply_chat_template=args.apply_chat_template,
            )
            report = run_capability_eval(
                args.checkpoint,
                settings,
                adapter=args.adapter,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                device=args.device,
            )
            _write_new(args.output, report)
            print(json.dumps(report["metrics"], indent=2, sort_keys=True))
            return 0
        gate = compare_capability_reports(
            json.loads(args.baseline.read_text(encoding="utf-8")),
            json.loads(args.candidate.read_text(encoding="utf-8")),
            max_regression=args.max_regression,
            task_max_regression=_task_thresholds(args.task_max_regression),
            relative=args.relative,
        )
        if args.output is not None:
            _write_new(args.output, gate)
        print(
            json.dumps(
                {
                    "passed": gate["passed"],
                    "gates": gate["gates"],
                    "regressions": gate["regressions"],
                },
                indent=2,
            )
        )
        return 0 if gate["passed"] else 1
    except (OSError, ValueError, ImportError) as exc:
        print(f"capability eval failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
