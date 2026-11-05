"""Background snapshot overlays for the studio: readiness, setup, data, quality.

Each panel owns one :class:`~stateset_agents.tui_tasks.BackgroundTask`, the
report it produced, the inputs (signature) that report describes, and its
rendering. The studio starts work, polls on its refresh tick and decides what
the status line says; panels never touch the screen or the studio's state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Generic, TypeVar

from stateset_agents.tui_tasks import BackgroundTask

T = TypeVar("T")


class BackgroundReport(Generic[T]):
    """One background report bound to the inputs it was taken for."""

    #: Thread-name suffix, for diagnostics.
    name = "report"

    def __init__(self) -> None:
        self.task: BackgroundTask[tuple[str, T]] = BackgroundTask()
        self.report: T | None = None
        self.signature = ""

    @property
    def busy(self) -> bool:
        """Whether a report is still being produced."""
        return self.task.busy

    def start(self, signature: str, operation: Callable[[], T]) -> bool:
        """Produce a report for ``signature`` off the terminal thread."""
        return self.task.start(
            lambda: (signature, operation()), name=f"stateset-{self.name}"
        )

    def set(self, signature: str, report: T) -> None:
        """Record a report produced synchronously (for example a settings error)."""
        self.signature, self.report = signature, report

    def poll(self) -> bool:
        """Apply a finished report; return whether one finished."""
        result = self.task.poll()
        if result is None:
            return False
        if result.value is not None:
            self.signature, self.report = result.value
        return True


class ReadinessReport(BackgroundReport[dict[str, Any]]):
    """Local readiness checks for the draft settings."""

    name = "readiness"

    def lines(self, current: str, values: Mapping[str, Any]) -> list[str]:
        rows = [
            "LOCAL READINESS / i refreshes this snapshot",
            "",
            "Checks do not load models, authenticate providers or provision compute.",
        ]
        if self.busy:
            return [
                *rows,
                "",
                "Checking local files, packages, hardware and queue in the background...",
            ]
        report = self.report
        if report is None:
            return [*rows, "", "Press i to inspect."]
        if self.signature != current:
            rows += [
                "",
                "[WARNING] Settings changed; press i to refresh before using this report.",
            ]
        rows += [
            "",
            "Draft model: " + str(report.get("model", values["model"])),
            "Compute: " + str(report.get("provider", values["provider"])),
        ]
        rows += [
            "",
            f"{report['failed']} failed checks | {report['warnings']} warnings",
        ]
        for check in report["checks"]:
            rows += [
                "",
                f"[{check['status'].upper()}] {check['name']}",
                check["detail"],
            ]
        rows += [
            "",
            "A passing local check is not GPU/model conformance or a provider quote.",
            "Press p separately to validate dataset contents and prepare a launch review.",
        ]
        return rows


class WorkflowReport(BackgroundReport[dict[str, Any]]):
    """The guided setup snapshot and its selected step."""

    name = "setup-inspection"

    def __init__(self) -> None:
        super().__init__()
        self.index = 0

    def move(self, step: int) -> None:
        """Select the next or previous of the five setup steps."""
        self.index = (self.index + step) % 5

    def action(self, current: str) -> str | None:
        """The selected step's action, if a current snapshot is available."""
        report = self.report
        if self.busy or not report or "error" in report or self.signature != current:
            return None
        return str(report["steps"][self.index]["action"])

    def lines(self, current: str) -> list[str]:
        rows = ["GUIDED SETUP / check, repair, review, evaluate", ""]
        if self.busy:
            return rows + ["Checking in the background. Esc returns; no job starts."]
        report = self.report
        if report is None:
            return rows + ["w checks the current settings."]
        if self.signature != current:
            return rows + [
                "Settings changed; press w to recheck before choosing a step."
            ]
        if "error" in report:
            return rows + [report["error"]]
        steps = report["steps"]
        rows += [
            (" > " if index == self.index else "   ")
            + f"[{step['status'].upper()}] {step['name']}"
            for index, step in enumerate(steps)
        ]
        return rows + [
            "",
            steps[self.index]["detail"],
            "",
            *report["notes"],
        ]


class DataReport(BackgroundReport[dict[str, Any]]):
    """Dataset statistics and up to three private samples, hidden by default."""

    name = "data-inspection"

    def __init__(self) -> None:
        super().__init__()
        self.show_samples = False
        self.sample_index = 0

    def toggle_samples(self) -> bool:
        """Show or hide private samples; return whether a usable report exists."""
        if not self.report or "error" in self.report:
            return False
        self.show_samples = not self.show_samples
        return True

    def step_sample(self, step: int) -> bool:
        """Select the next or previous sample; return whether there was one."""
        count = len((self.report or {}).get("samples", []))
        if not count:
            return False
        self.sample_index = (self.sample_index + step) % count
        return True

    def lines(self, current: str) -> list[str]:
        rows = ["DATASET INSPECTOR / statistics from one captured file", ""]
        if self.busy:
            return rows + [
                "Reading and validating dataset... Esc returns while the scan finishes."
            ]
        report = self.report
        if report is None:
            return rows + ["Press u to scan the current dataset."]
        if self.signature != current:
            return rows + [
                "Dataset path or method changed. Press u to scan the current settings."
            ]
        if "error" in report:
            diagnostic = report.get("diagnostic", {})
            location = diagnostic.get("location", {})
            return rows + [
                " | ".join(f"{key}: {value}" for key, value in location.items()),
                report["error"],
                "u retries after you fix the path or format.",
            ]
        rows += [
            f"Method: {report['method']} | Examples: {report['examples']} | Bytes: {report['bytes']:,}",
            "SHA256: " + report["dataset_sha256"],
            "Characters per example: "
            + " | ".join(
                f"{key}={value:g}"
                for key, value in report["characters_per_example"].items()
            ),
            "Turns ("
            + report["turn_definition"]
            + "): "
            + " | ".join(
                f"{key}={value:g}" for key, value in report["turns_per_example"].items()
            ),
            "Duplicate examples: " + str(report["duplicate_examples"]),
            "Definition: " + report["duplicate_definition"],
        ]
        if report["role_counts"]:
            rows.append(
                "Messages: "
                + " | ".join(
                    f"{role}={count}" for role, count in report["role_counts"].items()
                )
            )
        rows += ["", *report["warnings"], *report["notes"]]
        samples = report.get("samples", [])
        if self.show_samples and samples:
            sample = samples[self.sample_index % len(samples)]
            # Put explicit private output first so even a narrow pane can inspect it.
            return [
                rows[0],
                "",
                f"PRIVATE SAMPLE {sample['example']}/{len(samples)} / first three only",
                "Left/Right selects; e hides private text.",
                (
                    "Excerpt clipped to 4,096 characters."
                    if sample["truncated"]
                    else "Complete example text."
                ),
                *sample["text"].splitlines(),
                "",
                "DATASET STATISTICS",
                *rows[2:],
            ]
        return rows + ["", "Private samples hidden. Press e to show a bounded excerpt."]


class QualityReport(BackgroundReport[dict[str, Any]]):
    """Recorded base-versus-trained outcomes for one captured run."""

    name = "quality-inspection"

    def __init__(self) -> None:
        super().__init__()
        self.run: Path | None = None

    def lines(self) -> list[str]:
        rows = [
            "QUALITY / recorded base-versus-trained outcomes",
            "Run: " + (self.run.name if self.run else "none"),
            "",
        ]
        if self.busy:
            return rows + [
                "Verifying reports and hashing weights... Esc returns while inspection finishes."
            ]
        report = self.report
        if not report:
            return rows + ["e rechecks; f imports paired evidence."]
        if "error" in report:
            return rows + [
                report["error"],
                "",
                "Evidence format: schema_version: 1, baselines: [...], candidates: [...].",
                "Reports use the held-out evaluation schema documented in FINETUNING_AGENTS.md.",
            ]
        rows += [
            f"Paired seeds: {report['seeds']} | Recorded training seed: {report.get('training_seed') if report.get('training_seed') is not None else 'not recorded'}",
            f"Mean success gain: {100 * report['mean_success_gain']:+.1f} percentage points",
            f"Recorded study gate: {'PASS' if report['recorded_study_passed'] else 'NOT PASSED'}",
            f"Success regressions: {report['regressions_total']}",
            "Artifact SHA256: " + report["artifact_sha256"],
            "",
            f"Required mean gain: {100 * report['thresholds']['min_gain']:.1f} points | min seeds: {report['thresholds']['min_seeds']}",
            f"Seed sign-test p: {report['seed_inference']['bonferroni_adjusted_p_value']:.4g} | alpha: {report['thresholds']['alpha']:g}",
        ]
        for pair in report["pairs"]:
            rows.append(
                f"Seed {pair['seed']}: base {pair['baseline']['success_rate']:.1%} -> trained {pair['candidate']['success_rate']:.1%} | wins {pair['wins']} / regressions {pair['regressions']}"
            )
            rows.append(
                f"Recorded reward: {pair['baseline']['mean_reward']:.4g} -> {pair['candidate']['mean_reward']:.4g} | violation rate: {pair['baseline']['violation_rate']:.1%} -> {pair['candidate']['violation_rate']:.1%}"
            )
        rows += [
            "",
            "STUDY GATES",
            *[
                f"[{'PASS' if passed else 'FAIL'}] {key}"
                for key, passed in report["study_gates"].items()
            ],
            "",
            "REGRESSION HANDLES (up to 50)",
        ]
        rows += [
            f"Seed {row['seed']} | case {row['case_handle']} | reward {row['baseline_reward']:.4g} -> {row['candidate_reward']:.4g}"
            for row in report["regressions"]
        ]
        return rows + ["", *report["notes"]]


class RecoveryPanel:
    """Checkpoint recovery: its parent run, choices and one owned background step.

    Inspection and reviews may be abandoned on exit; preparing a recovery run
    and launching it are authorized steps that run on non-daemon threads, so
    closing the studio waits for them.
    """

    def __init__(self) -> None:
        self.task: BackgroundTask[dict[str, Any]] = BackgroundTask()
        self.run: Path | None = None
        self.report: dict[str, Any] | None = None
        self.choice: dict[str, Any] | None = None
        self.index = 0

    @property
    def busy(self) -> bool:
        """Whether a recovery step is still running."""
        return self.task.busy

    def start(
        self, operation: Callable[[], dict[str, Any]], *, step: str, authorized: bool
    ) -> bool:
        """Run one recovery step off the terminal thread."""
        return self.task.start(
            operation, daemon=not authorized, name=f"stateset-recovery-{step}"
        )

    def poll(self) -> dict[str, Any] | None:
        """Take a finished step's result, which also becomes the report."""
        result = self.task.poll()
        if result is None:
            return None
        self.report = (
            result.value
            if result.value is not None
            else {
                "error": "Recovery work failed unexpectedly; reinspect retained state."
            }
        )
        return self.report

    def choices(self) -> list[dict[str, Any]]:
        """Checkpoints offered by the latest inspection report."""
        return list((self.report or {}).get("checkpoints", []))

    def move(self, step: int) -> None:
        """Select the next or previous checkpoint, staying in range."""
        choices = self.choices()
        if choices:
            self.index = max(0, min(len(choices) - 1, self.index + step))

    def select(self) -> bool:
        """Capture the selected checkpoint for review if it is available."""
        choices = self.choices()
        if not choices or not choices[self.index]["available"]:
            return False
        self.choice = dict(choices[self.index])
        return True

    def lines(self, message: str) -> list[str]:
        rows = [
            "RECOVERY / exact local checkpoint",
            "Parent run: " + (self.run.name if self.run else "none"),
            "",
        ]
        if self.busy:
            return rows + [
                message,
                "Esc returns while owned recovery work finishes. Launch requires its own confirmation.",
                "Exiting waits for an authorized snapshot or launch submission to finish.",
            ]
        report = self.report or {}
        if "error" in report:
            return rows + [
                report["error"],
                "",
                "z reinspects; Esc returns. Older unbound artifacts are not resumable here.",
            ]
        choices = report.get("checkpoints", [])
        if not choices:
            return rows + [
                "No published recovery checkpoints for this run.",
                "New local workers publish receipts after complete saves. A model-only export cannot restore optimizer progress.",
            ]
        rows += [
            f"{' >' if index == self.index else '  '} {item['checkpoint']} | "
            + (
                f"step {item['step']} | {item['remaining']:g} {item['budget_unit']} remain"
                if item["available"]
                else "UNAVAILABLE"
            )
            for index, item in enumerate(choices)
        ]
        selected = choices[min(self.index, len(choices) - 1)]
        rows += [""]
        if selected["available"]:
            rows += [
                "Checkpoint SHA256: " + selected["checkpoint_sha256"],
                "Recovery SHA256: " + selected["recovery_sha256"],
                f"Private copy: {selected['bytes']:,} bytes / {selected['file_count']} files",
                "Enter reviews this checkpoint; it does not launch training.",
            ]
        else:
            rows.append(selected["reason"])
        return rows + ["", *report.get("notes", [])]

    def review_lines(self) -> list[str]:
        choice = self.choice or {}
        return [
            "REVIEW CHECKPOINT RECOVERY / prepare only",
            "",
            "Parent: " + (self.run.name if self.run else "none"),
            "Checkpoint: " + str(choice.get("checkpoint", "none")),
            "Checkpoint SHA256: " + str(choice.get("checkpoint_sha256", "")),
            "Recovery SHA256: " + str(choice.get("recovery_sha256", "")),
            f"Copy size: {choice.get('bytes', 0):,} bytes",
            "",
            "Enter copies the reviewed checkpoint and retained dataset into a NEW private run.",
            "Training settings and the original total budget stay fixed. Source artifacts are preserved.",
            "A separate prepared-run launch review follows. Esc prepares nothing.",
        ]
