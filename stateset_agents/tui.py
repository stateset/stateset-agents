"""Keyboard-driven fine-tuning studio for Herdr and other POSIX terminals.

The UI imports no training dependencies. Workers own training lifetimes;
closing or reconnecting the terminal never cancels a run.
"""

from __future__ import annotations

import hashlib
import math
import shlex
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any, TypeGuard

from rich.cells import cell_len, set_cell_size

from stateset_agents.data.text_sft import DatasetFormatError
from stateset_agents.finetune_evaluation_tui import EvaluationPanel
from stateset_agents.finetune_inspection import (
    compare_runs,
    inspect_readiness,
    inspect_training_data,
    inspect_workflow,
    metric_history,
    metric_sparkline,
)
from stateset_agents.finetune_provider_submission import inspect_submission
from stateset_agents.finetune_quality import inspect_quality, record_quality
from stateset_agents.finetune_recovery import inspect_checkpoints, prepare_resume
from stateset_agents.finetune_workspace import (
    BACKENDS,
    MAX_DATA_BYTES,
    METHODS,
    PROVIDERS,
    ConfigIssue,
    FinetuneConfig,
    PreparedRunReview,
    RunPlan,
    check_config_values,
    create_run,
    failure_info,
    launch_run,
    list_runs,
    log_tail,
    plan_run,
    queue_executable,
    read_json,
    read_run_plan,
    redact_finetune_secrets,
    request_stop,
    require_backend_credentials,
    review_prepared_run,
    write_json,
)
from stateset_agents.tui_actions import ActionMenu
from stateset_agents.tui_panels import FilePicker, LogViewer
from stateset_agents.tui_render import FrameRenderer, configure_terminal
from stateset_agents.tui_reports import (
    DataReport,
    QualityReport,
    ReadinessReport,
    WorkflowReport,
)
from stateset_agents.tui_tasks import BackgroundTask, TaskResult
from stateset_agents.tui_text import _clip, _safe, _wrap

MIN_TERMINAL_COLUMNS = 32
MIN_TERMINAL_ROWS = 16
COMPACT_COLUMNS = 72
COMPACT_ROWS = 22


@dataclass(frozen=True)
class Field:
    """One setting and its terminal editing guidance."""

    name: str
    label: str
    help: str
    choices: tuple[str, ...] = ()


FIELDS = (
    Field(
        "method",
        "Training",
        "SFT learns replies; RL optimizes the support conversation reward.",
        METHODS,
    ),
    Field(
        "provider",
        "Compute",
        "b selects local CUDA, RunPod, Modal or River. Remote studio paths support SFT.",
        PROVIDERS,
    ),
    Field(
        "cost_policy",
        "Cost policy",
        "Modal/River require provider_billed: no enforced dollar ceiling.",
        ("require_cap", "provider_billed"),
    ),
    Field(
        "gpu",
        "Remote GPU",
        "Blank uses the backend recommendation. Modal names: A10, A100, H100, etc.",
    ),
    Field(
        "gpu_count",
        "Remote GPUs",
        "0 uses the backend recommendation; hardware fit still needs review.",
    ),
    Field(
        "model",
        "Model",
        "Press m to search the model catalog, or Enter to type an HF model ID.",
    ),
    Field(
        "revision",
        "Model commit",
        "Full 40-character model commit. Required for local checkpoint recovery and paired evaluation.",
    ),
    Field(
        "dataset",
        "Dataset",
        "f browses datasets. SFT: chat JSONL. RL: support scenarios JSON.",
    ),
    Field("epochs", "Epochs", "Complete passes through the SFT dataset."),
    Field(
        "seed",
        "Training seed",
        "0–4294967295. Seeds initialization and training; a seed alone does not guarantee identical GPU results.",
    ),
    Field(
        "steps",
        "RL updates",
        "Requested committed optimizer updates, not generated episodes.",
    ),
    Field(
        "learning_rate",
        "Learning rate",
        "Adapter learning rate; scientific notation is accepted.",
    ),
    Field(
        "kl_coefficient",
        "KL penalty",
        "Positive penalty against a frozen reference for native RL.",
    ),
    Field(
        "generations",
        "RL generations",
        "At least two sampled replies per scenario for relative advantages.",
    ),
    Field(
        "lora_rank",
        "LoRA rank",
        "Adapter rank. Alpha is twice this value; dropout is zero for RL.",
    ),
    Field(
        "context_length",
        "Context tokens",
        "SFT sequence limit or native RL prompt token limit.",
    ),
    Field(
        "max_new_tokens",
        "Reply tokens",
        "Maximum generated tokens per native RL reply.",
    ),
    Field("max_turns", "Conversation turns", "Maximum native RL conversation steps."),
    Field(
        "checkpoint_every",
        "Checkpoint interval",
        "Local native RL updates between checkpoints; 0 keeps only the final save. Retain at most 100 checkpoints for reviewed validation.",
    ),
    Field(
        "max_cost_usd",
        "RunPod ceiling $",
        "Provider pod-lifetime ceiling; ancillary costs need separate review.",
    ),
    Field(
        "timeout_seconds",
        "Timeout seconds",
        "Training timeout; queue wait is additional.",
    ),
    Field(
        "cpu_slots", "Queue CPUs", "Local CPU reservation and OMP/MKL worker budgets."
    ),
    Field(
        "memory_gb",
        "Queue RAM GiB",
        "Local RAM reservation. This does not reserve GPU memory.",
    ),
    Field(
        "scratch_gb",
        "Queue scratch GiB",
        "Additional local free-disk reservation, excluding model cache already present.",
    ),
)


def model_choices() -> list[tuple[str, str]]:
    """Show existing catalog maturity; listing a model does not certify a backend."""
    from stateset_agents.core.model_presets import PRESETS
    from stateset_agents.remote.model_catalog import MODEL_CATALOG

    models = {preset.model_id: "configured" for preset in PRESETS.values()}
    models.update({row.model: row.certification for row in MODEL_CATALOG})
    return sorted(models.items())


def _field_text(value: Any) -> str:
    """Keep valid numeric zeros visible while labelling absent values."""
    if value is None or (isinstance(value, str) and not value):
        return "(not set)"
    return str(value)


def _teardown_note(state: dict[str, Any]) -> str:
    """Describe a failure observed after the worker committed its success."""
    code = state.get("teardown_exit_code")
    return (
        "Note: the worker exited after committing a verified result"
        + (f" (exit {code})" if isinstance(code, int) else "")
        + f"; teardown: {_safe(str(state.get('teardown_error')))}. The result stands."
    )


def _run_config(row: dict[str, Any]) -> dict[str, Any]:
    config = row.get("plan", {}).get("config", {})
    return config if isinstance(config, dict) else {}


def _finite(value: Any) -> TypeGuard[int | float]:
    """Reject malformed retained counters, including integers too large for floats."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class Studio:
    """Testable screen state; curses only paints and supplies keyboard events."""

    def __init__(
        self,
        workspace: Path,
        config: FinetuneConfig | None = None,
        *,
        background_history: bool = False,
    ) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.values = asdict(config or FinetuneConfig())
        self.tab = "configure"
        self.field = 0
        self.field_page_size = 8
        self.run_index = 0
        self.pending_run_id: str | None = None
        self.log_offset = 0
        self.editing: str | None = None
        self.edit_cursor = 0
        self.editor_column = 6
        self.terminal_columns = 110
        self.compact_layout = False
        self._compact_follow_selection = True
        self.overlay: str | None = None
        self.search = ""
        self.model_index = 0
        self.backend_index = PROVIDERS.index(str(self.values["provider"]))
        self.dialog_offset = 0
        self.models = model_choices()
        self.file_picker = FilePicker(Path.cwd())
        self.file_evaluation: EvaluationPanel | None = None
        self.plan: RunPlan | None = None
        self.planning: BackgroundTask[RunPlan] = BackgroundTask()
        self.planning_signature = ""
        self.planning_review = False
        self.action_menu: ActionMenu | None = None
        self.action_run_id: str | None = None
        self.prepared_review: PreparedRunReview | None = None
        self.comparison_reference: Path | None = None
        self.comparison: dict[str, Any] | None = None
        self.data_panel = DataReport()
        self.workflow_panel = WorkflowReport()
        self.quality_panel = QualityReport()
        self.evaluation_panel: EvaluationPanel | None = None
        self.stop_run: Path | None = None
        self.stop_digest = ""
        self.recovery_run: Path | None = None
        self.recovery_report: dict[str, Any] | None = None
        self.recovery_choice: dict[str, Any] | None = None
        self.recovery_index = 0
        self.recovery_busy = False
        self._recovery_results: SimpleQueue[dict[str, Any]] = SimpleQueue()
        self.validation_issues: list[ConfigIssue] = []
        self.validation_index = 0
        self.validation_signature = ""
        self.validation_field: str | None = None
        self.log_viewer: LogViewer | None = None
        self.message = "Start here: w guided setup | : actions. Preview is free."
        self.runs: list[dict[str, Any]] = []
        self.all_runs: list[dict[str, Any]] = []
        self.background_history = background_history
        self.history_task: BackgroundTask[list[dict[str, Any]]] = BackgroundTask()
        # Launch, stop and prepared-run review verify retained bytes (dataset
        # hashes, package identity) and must not freeze the terminal thread.
        self.submission: BackgroundTask[Any] = BackgroundTask()
        self.submission_kind = ""
        self.submission_overlay: str | None = None
        self.history_loaded = False
        self.history_error = ""
        self.run_query = ""
        self.run_filter = "all"
        self.readiness_panel = ReadinessReport()
        self.refresh()

    def fields(self) -> list[Field]:
        """Show only settings consumed by the chosen execution path."""
        hidden = (
            {
                "steps",
                "kl_coefficient",
                "generations",
                "max_new_tokens",
                "max_turns",
                "checkpoint_every",
            }
            if self.values["method"] == "sft"
            else {"epochs"}
        )
        if self.values["provider"] == "local":
            hidden.add("max_cost_usd")
        else:
            if not self.values["revision"]:
                hidden.add("revision")
            hidden.update({"cpu_slots", "memory_gb", "scratch_gb"})
        if self.values["provider"] not in {"runpod", "modal"}:
            hidden.update({"gpu", "gpu_count"})
        if self.values["provider"] not in {"modal", "river"}:
            hidden.add("cost_policy")
        if self.values["provider"] != "runpod":
            hidden.add("max_cost_usd")
        if self.values["provider"] == "river":
            hidden.add("timeout_seconds")
        return [
            field
            for field in FIELDS
            if field.name not in hidden or field.name == self.validation_field
        ]

    def config(self) -> FinetuneConfig:
        """Parse editable values without silently truncating invalid integers."""
        values, issues = check_config_values(self.values, parse_text=True)
        parse_errors = [
            issue
            for issue in issues
            if issue.code in {"invalid_type", "unknown_fields"}
        ]
        if parse_errors:
            raise ValueError(parse_errors[0].message)
        return FinetuneConfig(**values)

    def refresh(self, *, history: bool = True) -> None:
        """Reconnect to retained runs without polling external services."""
        submitted = self.submission.poll()
        if submitted is not None:
            self._finish_submission(submitted)
        selected = (
            self.runs[self.run_index]["id"]
            if self.runs and self.run_index < len(self.runs)
            else None
        )
        if self.background_history:
            history_result = self.history_task.poll()
            if history_result is not None:
                if history_result.error is not None:
                    self.history_error = "Cannot refresh retained runs; check workspace access. Retrying."
                elif history_result.value is not None:
                    self.all_runs = history_result.value
                    self.history_loaded = True
                    self.history_error = ""
            if history and not self.history_task.busy:
                self.history_task.start(lambda: list_runs(self.workspace, limit=None))
        elif history:
            self.all_runs = list_runs(self.workspace, limit=None)
            self.history_loaded = True
        self.runs = self._filter_runs(self.run_query)
        if self.pending_run_id is not None and any(
            row["id"] == self.pending_run_id for row in self.runs
        ):
            selected, self.pending_run_id = self.pending_run_id, None
        watched = (
            self.log_viewer.run.name
            if self.overlay == "logs" and self.log_viewer
            else selected
        )
        if (
            watched is not None
            and self.overlay in {"metrics", "details", "logs"}
            and not any(row["id"] == watched for row in self.runs)
        ):
            self.overlay = None
            self.dialog_offset = 0
            self.log_viewer = None
            self.message = (
                "Selected run left this filter. Backspace shows all retained runs."
            )
        self.run_index = next(
            (index for index, row in enumerate(self.runs) if row["id"] == selected),
            min(self.run_index, max(0, len(self.runs) - 1)),
        )
        preview_result = self.planning.poll()
        if preview_result is not None and not preview_result.cancelled:
            if self.planning_signature != self._settings_signature():
                self.message = "Settings changed; press p or r for a fresh preview."
                if self.overlay == "planning":
                    self.overlay = None
            elif preview_result.error is not None:
                self.message = f"Fix settings: {redact_finetune_secrets(str(preview_result.error))}"
                if self.overlay == "planning":
                    self.overlay = None
            else:
                self.plan = preview_result.value
                if self.plan is not None:
                    self.message = f"Valid: {self.plan.examples} examples. Press r to review launch."
                    if self.overlay == "planning":
                        self.overlay = "review" if self.planning_review else None
                        self.dialog_offset = 0
        self.readiness_panel.poll()

        try:
            result = self._recovery_results.get_nowait()
            self.recovery_busy = False
            self.recovery_report = result
            if "prepared_review" in result:
                review = result["prepared_review"]
                if self.overlay == "recovery":
                    self.prepared_review = review
                    self.overlay, self.dialog_offset = "prepared-review", 0
                self.message = f"Prepared recovery run {review.run.name}; separate r/Enter launch review required."
            elif result.get("launched"):
                self.message = f"Launched reviewed recovery run {result['run_id']}."
                if self.overlay == "recovery":
                    self.overlay = None
            elif self.overlay == "recovery":
                self.message = (
                    "Recovery inspection failed; check eligibility and retained state."
                    if "error" in result
                    else "Recovery choices verified; Enter reviews a selected checkpoint."
                )
        except Empty:
            pass
        if self.data_panel.poll():
            self.data_panel.show_samples = False
            if self.message in {
                "Reading one dataset snapshot; no run is prepared or started.",
                "Dataset scan is already running. Esc leaves it in the background.",
            }:
                if self.data_panel.signature != self._data_settings_signature():
                    self.message = "Dataset settings changed; press u to scan again."
                elif (
                    self.data_panel.report is None or "error" in self.data_panel.report
                ):
                    self.message = "Dataset scan failed; press u to inspect and retry."
                else:
                    self.message = (
                        "Dataset scan complete; no run was prepared or started."
                    )

        if (
            self.workflow_panel.poll()
            and self.message == "Checking guided setup in the background."
        ):
            self.message = (
                "Setup snapshot ready; Enter opens the selected step."
                if self.workflow_panel.signature == self._settings_signature()
                else "Settings changed; press w for a current setup snapshot."
            )
        if self.evaluation_panel is not None:
            self.evaluation_panel.refresh()
        if self.quality_panel.poll() and self.message in {
            "Verifying recorded outcomes and artifact bytes in the background.",
            "Previous artifact inspection is still running; Esc returns.",
        }:
            self.message = (
                "Quality inspection complete; review gates and evidence limits."
            )

    def _submit(
        self,
        kind: str,
        operation: Callable[[], Any],
        message: str,
        *,
        authorized: bool,
    ) -> bool:
        """Run one launch, stop or review step off the terminal thread.

        Authorized submissions use a non-daemon thread, so exiting the studio
        waits for them instead of abandoning a half-submitted launch.
        """
        if self.submission.busy:
            self.message = "A launch, stop or review is still being processed; try again when it finishes."
            return False
        self.submission_kind, self.submission_overlay = kind, self.overlay
        self.submission.start(operation, daemon=not authorized, name=f"stateset-{kind}")
        self.message = message
        return True

    def _show_submitted_run(self, run: Path) -> None:
        """Select a newly launched run unless the user moved on meanwhile."""
        self.pending_run_id = run.name
        if self.overlay is None and self.editing is None:
            self.tab = "runs"
            self.run_index = 0
            self.run_query, self.run_filter = "", "all"

    def _finish_submission(self, result: TaskResult[Any]) -> None:
        """Apply a completed submission on the terminal thread."""
        kind, error, value = self.submission_kind, result.error, result.value
        if kind == "stop":
            if error is not None or not isinstance(value, dict):
                self.message = (
                    "Stop request rejected; inspect current run and reviewed plan."
                )
            elif value.get("stop_requested"):
                self.message = (
                    "Local stop receipt saved; supervisor lost, so worker exit remains unconfirmed."
                    if value.get("execution_unconfirmed")
                    else "Local stop requested; workers check at safe boundaries. Queue entries remain until admitted."
                )
            else:
                self.message = "Run already terminal; no stop was requested."
        elif kind in {"launch", "launch-prepared"}:
            if error is not None or not isinstance(value, Path):
                self.message = f"Launch failed: {error}"
            else:
                self._show_submitted_run(value)
                self.message = (
                    f"Launched {value.name}. Logs and artifacts stay in {value}."
                    if kind == "launch"
                    else f"Launched prepared run {value.name}."
                )
        elif kind == "review":
            if error is not None or not isinstance(value, PreparedRunReview):
                self.prepared_review = None
                self.message = f"Cannot review prepared run: {error}"
            elif self.overlay != self.submission_overlay:
                self.message = "Prepared run review is ready; press r to open it."
            else:
                self.prepared_review = value
                self.overlay = "prepared-review"
                self.dialog_offset = 0
                self.message = (
                    "Review the retained plan. Enter launches this same run ID."
                )

    def _open_workflow(self) -> None:
        self.overlay, self.dialog_offset = "workflow", 0
        if self.workflow_panel.busy:
            return
        values = dict(self.values)
        self.message = "Checking guided setup in the background."

        def inspect() -> dict[str, Any]:
            try:
                return inspect_workflow(values, self.workspace, parse_text=True)
            except Exception:
                return {
                    "error": "Setup inspection failed. Use v, u and i to inspect individual steps."
                }

        self.workflow_panel.start(self._settings_signature(), inspect)

    def _workflow_action(self) -> None:
        action = self.workflow_panel.action(self._settings_signature())
        if action is None:
            self.message = "Press w to obtain a current setup snapshot first."
            return
        self.overlay = None
        self.dialog_offset = 0
        if action == "settings":
            self._check_settings()
        elif action == "dataset":
            if self.values["dataset"]:
                self._inspect_data()
            else:
                self.handle("f")
        elif action == "runtime":
            self._inspect()
        elif action == "review":
            self.preview(review=True)
        elif action == "quality":
            self.tab = "runs"
            self._open_quality()

    def _open_quality(self) -> None:
        if self.quality_panel.busy:
            self.overlay = "quality"
            self.message = "Previous artifact inspection is still running; Esc returns."
            return
        if not self.runs:
            self.message = "Choose a completed run on Runs to inspect quality evidence."
            return
        self.quality_panel.run = Path(self.runs[self.run_index]["path"])
        self.overlay, self.dialog_offset = "quality", 0
        self._scan_quality()

    def _open_evaluation(self) -> None:
        run = (
            self.quality_panel.run
            if self.overlay == "quality"
            else (Path(self.runs[self.run_index]["path"]) if self.runs else None)
        )
        if run is None:
            self.message = (
                "Select a local run with final weights or a published checkpoint."
            )
            return
        if self.evaluation_panel is not None and self.evaluation_panel.busy:
            self.message = "Evaluation work remains bound to its captured run."
        elif self.evaluation_panel is None or self.evaluation_panel.run != run:
            self.evaluation_panel = EvaluationPanel(run)
        self.overlay = "evaluation"

    def _scan_quality(self, source: Path | None = None) -> None:
        run = self.quality_panel.run
        if self.quality_panel.busy or run is None:
            return
        self.quality_panel.report = None
        self.message = (
            "Verifying recorded outcomes and artifact bytes in the background."
        )

        def inspect() -> dict[str, Any]:
            try:
                return record_quality(run, source) if source else inspect_quality(run)
            except Exception:
                return {
                    "error": "No verifiable quality evidence. Import a paired report file with f; local completed weights and matching checkpoint hashes are required."
                }

        self.quality_panel.start("", inspect)

    def _review_stop(self) -> None:
        if not self.runs:
            return
        run = Path(self.runs[self.run_index]["path"])
        try:
            plan, digest = read_run_plan(run)
            if plan["config"].get("provider") != "local":
                self.message = "Remote stop needs provider-specific cleanup; no stop was requested."
                return
            self.stop_run, self.stop_digest = run, digest
            self.overlay, self.dialog_offset = "stop", 0
        except (OSError, ValueError, TypeError, KeyError):
            self.message = "Cannot verify the selected run for a stop request."

    def _filter_runs(self, query: str) -> list[dict[str, Any]]:
        rows = []
        for row in self.all_runs:
            status = row["state"].get("status")
            # Interrupted supervision may leave detached work alive.
            active = status in {
                "prepared",
                "starting",
                "queued",
                "running",
                "interrupted",
            }
            if self.run_filter == "active" and not active:
                continue
            if self.run_filter == "failed" and status not in {"failed", "interrupted"}:
                continue
            if self.run_filter == "succeeded" and status != "succeeded":
                continue
            config = _run_config(row)
            text = " ".join(
                str(value)
                for value in (
                    row["id"],
                    status,
                    config.get("model"),
                    config.get("method"),
                    config.get("provider"),
                )
            )
            if all(word in text.casefold() for word in query.casefold().split()):
                rows.append(row)
        return rows

    def _settings_signature(self) -> str:
        return repr(sorted(self.values.items()))

    def _inspect(self) -> None:
        self.overlay = "readiness"
        self.dialog_offset = 0
        if self.readiness_panel.busy:
            return
        signature = self._settings_signature()
        try:
            config = self.config()
        except (ValueError, TypeError) as error:
            self.readiness_panel.set(
                signature,
                {
                    "failed": 1,
                    "warnings": 0,
                    "checks": [
                        {"name": "Settings", "status": "fail", "detail": str(error)}
                    ],
                },
            )
            return

        def collect() -> dict[str, Any]:
            try:
                return inspect_readiness(config, self.workspace)
            except Exception as error:
                return {
                    "model": config.model,
                    "provider": config.provider,
                    "failed": 1,
                    "warnings": 0,
                    "checks": [
                        {
                            "name": "Inspection",
                            "status": "fail",
                            "detail": f"Local inspection unavailable ({type(error).__name__}).",
                        }
                    ],
                }

        self.readiness_panel.start(signature, collect)

    def _set_model(self, model: str) -> None:
        if model != self.values["model"] and self.values["revision"]:
            self.values["revision"] = ""
            self.message = "Model changed; cleared the previous model's commit."
        self.values["model"] = model
        self.plan = None

    def _open_files(self, *, evaluation: EvaluationPanel | None = None) -> None:
        self.file_evaluation = evaluation
        value = evaluation.values["cases"] if evaluation else self.values["dataset"]
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        while not path.is_dir() and path != path.parent:
            path = path.parent
        self.file_picker = FilePicker(path.resolve(), cases=evaluation is not None)
        self.overlay = "files"
        self.file_picker.read()

    def _close_files(self, selected: Path | None = None) -> None:
        """Return a path only to the captured editor; never read or prepare it."""
        target, self.file_evaluation = self.file_evaluation, None
        if target is not None:
            self.overlay = "evaluation" if self.evaluation_panel is not None else None
            if selected is None:
                return
            if (
                self.evaluation_panel is None
                or target is not self.evaluation_panel
                or target.mode != "form"
                or target.busy
            ):
                if self.evaluation_panel:
                    self.evaluation_panel.message = "Cases selection discarded because the evaluation editor changed."
                return
            target.values["cases"] = str(selected)
            target.review, target.prepared_path, target.result = None, None, None
            target.field = 0
            target.message = (
                "Held-out cases selected. Press r to validate and capture a review."
            )
            return
        self.overlay = None
        if selected is None:
            self.prepared_review = None
            self.comparison = None
            self.data_panel.show_samples = False
            self.dialog_offset = 0
        if selected is not None:
            self.values["dataset"] = str(selected)
            self.plan = None
            self.message = "Dataset selected. Press p to validate its contents."

    def _reuse_run(self) -> None:
        if not self.runs:
            return
        row = self.runs[self.run_index]
        try:
            snapshot = Path(row["path"]) / "dataset"
            if snapshot.stat().st_size > MAX_DATA_BYTES or hashlib.sha256(
                snapshot.read_bytes()
            ).hexdigest() != row["plan"].get("dataset_sha256"):
                raise ValueError(
                    "Retained dataset snapshot differs from its reviewed hash"
                )
            config = FinetuneConfig(**{**_run_config(row), "dataset": str(snapshot)})
            config.validate()
            self.values = asdict(config)
            self.plan = None
            self.field = 0
            self.tab = "configure"
            self.message = f"New run from {row['id']}; review before launching."
        except (OSError, ValueError, TypeError) as error:
            self.message = f"Cannot reuse settings: {error}"

    def preview(self, *, review: bool = False) -> None:
        """Capture the dataset in the background, bound to the current settings."""
        if self.planning.busy:
            self.message = (
                "Previous preview is still finishing. Try again when it completes."
            )
            return
        self.plan = None
        try:
            config = self.config()
            config.validate()
            self.planning_signature = self._settings_signature()
            self.planning_review = review
            self.planning.start(lambda: plan_run(config))
            self.overlay, self.dialog_offset = "planning", 0
            self.message = "Validating dataset and resources in the background. Esc cancels this preview."
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.message = f"Fix settings: {error}"

    def _data_settings_signature(self) -> str:
        return repr((self.values["dataset"], self.values["method"]))

    def _inspect_data(self) -> None:
        self.overlay = "data"
        self.dialog_offset = 0
        self.data_panel.show_samples = False
        self.data_panel.sample_index = 0
        if self.data_panel.busy:
            self.message = (
                "Dataset scan is already running. Esc leaves it in the background."
            )
            return
        dataset, method = self.values["dataset"], self.values["method"]
        self.data_panel.report = None
        self.message = "Reading one dataset snapshot; no run is prepared or started."

        def inspect() -> dict[str, Any]:
            try:
                if (
                    not isinstance(dataset, str)
                    or not dataset.strip()
                    or not isinstance(method, str)
                ):
                    raise ValueError("Choose a dataset and method")
                return inspect_training_data(
                    Path(dataset), method, include_samples=True
                )
            except DatasetFormatError as error:
                return {"error": str(error), "diagnostic": error.diagnostic}
            except Exception:
                return {
                    "error": "Cannot inspect dataset. Choose a regular file within 64 MiB matching the documented SFT/RL format."
                }

        self.data_panel.start(self._data_settings_signature(), inspect)

    def _check_settings(self) -> None:
        _, self.validation_issues = check_config_values(self.values, parse_text=True)
        self.validation_index = 0
        self.validation_signature = self._settings_signature()
        self.validation_field = None
        self.overlay = "validation"
        self.message = (
            "Settings only; p validates dataset contents. No job starts here."
        )

    def _fix_setting(self) -> None:
        if not self.validation_issues:
            return
        issue = self.validation_issues[self.validation_index]
        if issue.field not in {field.name for field in FIELDS}:
            self.message = "Edit the configuration file to remove unknown fields."
            return
        self.validation_field = issue.field
        self.tab = "configure"
        self.field = next(
            index
            for index, field in enumerate(self.fields())
            if field.name == issue.field
        )
        self.overlay = None
        self.message = issue.message + " Edit this setting, then press v to recheck."

    def filtered_models(self) -> list[tuple[str, str]]:
        """Filter model IDs and maturity labels using an ordinary substring."""
        query = self.search.casefold()
        return [row for row in self.models if query in " ".join(row).casefold()]

    def _open_recovery(self, captured_run: Path | None = None) -> None:
        if self.recovery_busy:
            self.overlay = "recovery"
            self.message = (
                "Recovery work is still running; Esc leaves it in the background."
            )
            return
        if not self.runs and captured_run is None:
            self.message = (
                "Select a retained local run to inspect recovery checkpoints."
            )
            return
        self.recovery_run = captured_run or Path(self.runs[self.run_index]["path"])
        self.recovery_report, self.recovery_choice = None, None
        self.recovery_index, self.dialog_offset = 0, 0
        self.overlay, self.recovery_busy = "recovery", True
        self.message = "Verifying recovery checkpoint bytes in the background."
        run = self.recovery_run

        def inspect() -> None:
            try:
                result = inspect_checkpoints(run)
            except Exception:
                result = {
                    "error": "Recovery requires a terminal local run, stopped workers, matching code, pinned model commit and complete published checkpoint state."
                }
            self._recovery_results.put(result)

        threading.Thread(
            target=inspect, daemon=True, name="stateset-recovery-inspection"
        ).start()

    def _prepare_recovery(self) -> None:
        run, choice = self.recovery_run, self.recovery_choice
        if self.recovery_busy or run is None or choice is None:
            return
        self.recovery_busy, self.overlay = True, "recovery"
        self.message = "Copying the reviewed checkpoint into a new private run; no training is launched."

        def prepare() -> None:
            try:
                child = prepare_resume(
                    run,
                    choice["checkpoint"],
                    expected_recovery_sha256=choice["recovery_sha256"],
                )
                result: dict[str, Any] = {"prepared_review": review_prepared_run(child)}
            except Exception:
                result = {
                    "error": "Recovery preparation failed. Reinspect checkpoint state, current code, worker ownership and free disk space; no launch was attempted."
                }
            self._recovery_results.put(result)

        threading.Thread(
            target=prepare, daemon=False, name="stateset-recovery-preparation"
        ).start()

    def _recovery_lines(self) -> list[str]:
        rows = [
            "RECOVERY / exact local checkpoint",
            "Parent run: " + (self.recovery_run.name if self.recovery_run else "none"),
            "",
        ]
        if self.recovery_busy:
            return rows + [
                self.message,
                "Esc returns while owned recovery work finishes. Launch requires its own confirmation.",
                "Exiting waits for an authorized snapshot or launch submission to finish.",
            ]
        report = self.recovery_report or {}
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
            f"{' >' if index == self.recovery_index else '  '} {item['checkpoint']} | "
            + (
                f"step {item['step']} | {item['remaining']:g} {item['budget_unit']} remain"
                if item["available"]
                else "UNAVAILABLE"
            )
            for index, item in enumerate(choices)
        ]
        selected = choices[min(self.recovery_index, len(choices) - 1)]
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

    def handle(self, key: str) -> bool:
        """Apply a normalized key; return false only when the user exits."""
        if key == "QUIT":
            return False
        if self.compact_layout and not self._text_input_active():
            if key == "4" and self.overlay is None:
                key = "TAB"
            key = {
                "2": "DOWN",
                "8": "UP",
                "3": "PAGEDOWN",
                "9": "PAGEUP",
                "0": "ESC",
            }.get(key, key)
        if key in {"UP", "DOWN", "b", "w", "z"}:
            self._compact_follow_selection = True
        if self.overlay == "planning":
            if key == "ESC":
                self.planning.cancel()
                self.overlay = None
                self.message = "Preview cancelled. Its result will be discarded; no run was created."
            return True
        if self.overlay == "actions" and self.action_menu is not None:
            if key == "ESC":
                self.overlay = None
            else:
                command = self.action_menu.handle(key)
                if command is not None:
                    self.overlay = None
                    if self.action_run_id is not None and (
                        not self.runs
                        or self.runs[self.run_index]["id"] != self.action_run_id
                    ):
                        self.message = "Selected run changed. Reopen : actions for the current run."
                    else:
                        return self.handle(command)
            return True
        if self.overlay == "quality-path":
            if key == "ESC":
                self.overlay = "quality"
            elif key == "ENTER":
                self.overlay = "quality"
                if self.search.strip():
                    self._scan_quality(Path(self.search))
            elif key == "BACKSPACE":
                self.search = self.search[:-1]
            elif key == "CLEAR":
                self.search = ""
            elif len(key) == 1 and key.isprintable() and len(self.search) < 4096:
                self.search += key
            return True
        if self.editing is not None:
            if key == "ESC":
                self.editing = None
            elif key == "ENTER":
                name = self.fields()[self.field].name
                if name == "model":
                    self._set_model(self.editing)
                else:
                    self.values[name] = self.editing
                self.editing = None
                self.plan = None
            elif key == "BACKSPACE":
                if self.edit_cursor:
                    self.editing = (
                        self.editing[: self.edit_cursor - 1]
                        + self.editing[self.edit_cursor :]
                    )
                    self.edit_cursor -= 1
            elif key == "DELETE":
                self.editing = (
                    self.editing[: self.edit_cursor]
                    + self.editing[self.edit_cursor + 1 :]
                )
            elif key in {"LEFT", "RIGHT"}:
                self.edit_cursor = max(
                    0,
                    min(
                        len(self.editing),
                        self.edit_cursor + (1 if key == "RIGHT" else -1),
                    ),
                )
            elif key == "HOME":
                self.edit_cursor = 0
            elif key == "END":
                self.edit_cursor = len(self.editing)
            elif key == "CLEAR":
                self.editing = ""
                self.edit_cursor = 0
            elif len(key) == 1 and key.isprintable():
                self.editing = (
                    self.editing[: self.edit_cursor]
                    + key
                    + self.editing[self.edit_cursor :]
                )
                self.edit_cursor += 1
            return True
        if self.overlay == "evaluation" and self.evaluation_panel is not None:
            if (
                key == "f"
                and self.evaluation_panel.mode == "form"
                and not self.evaluation_panel.busy
            ):
                self._open_files(evaluation=self.evaluation_panel)
            elif self.evaluation_panel.handle(key):
                self.overlay = None
            return True
        if self.overlay:
            if self.overlay == "logs":
                if self.log_viewer is None:
                    if key == "ESC":
                        self.overlay = None
                elif self.log_viewer.handle(key):
                    self.overlay, self.log_viewer = None, None
                return True
            if key == "ESC":
                if self.overlay == "files":
                    self._close_files()
                    return True
                self.overlay = None
                self.prepared_review = None
                self.comparison = None
                self.data_panel.show_samples = False
                self.dialog_offset = 0
            elif self.overlay == "workflow":
                if key == "w":
                    self._open_workflow()
                elif key in {"UP", "DOWN"}:
                    self.workflow_panel.move(1 if key == "DOWN" else -1)
                elif key == "ENTER":
                    self._workflow_action()
                elif key in {"PAGEUP", "PAGEDOWN"}:
                    self.dialog_offset = max(
                        0, self.dialog_offset + (8 if key == "PAGEDOWN" else -8)
                    )
            elif self.overlay == "recovery":
                if key == "z" and not self.recovery_busy:
                    self._open_recovery(self.recovery_run)
                elif not self.recovery_busy and self.recovery_report:
                    choices = self.recovery_report.get("checkpoints", [])
                    if key in {"UP", "DOWN"} and choices:
                        self.recovery_index = max(
                            0,
                            min(
                                len(choices) - 1,
                                self.recovery_index + (1 if key == "DOWN" else -1),
                            ),
                        )
                    elif (
                        key == "ENTER"
                        and choices
                        and choices[self.recovery_index]["available"]
                    ):
                        self.recovery_choice = dict(choices[self.recovery_index])
                        self.overlay, self.dialog_offset = "recovery-review", 0
                if key in {"PAGEUP", "PAGEDOWN"}:
                    self.dialog_offset = max(
                        0, self.dialog_offset + (8 if key == "PAGEDOWN" else -8)
                    )
            elif self.overlay == "recovery-review" and key == "ENTER":
                self._prepare_recovery()
            elif self.overlay == "stop" and key == "ENTER" and self.stop_run:
                run, digest = self.stop_run, self.stop_digest
                if self._submit(
                    "stop",
                    lambda: request_stop(run, expected_plan_sha256=digest),
                    "Saving the local stop request in the background.",
                    authorized=True,
                ):
                    self.overlay = None
            elif self.overlay == "quality":
                if key == "e":
                    self._scan_quality()
                elif key == "j":
                    self._open_evaluation()
                elif key == "f" and not self.quality_panel.busy:
                    self.overlay, self.search = "quality-path", ""
                elif key in {"UP", "DOWN", "PAGEUP", "PAGEDOWN"}:
                    delta = 8 if key in {"PAGEUP", "PAGEDOWN"} else 1
                    self.dialog_offset = max(
                        0,
                        self.dialog_offset
                        + (delta if key in {"DOWN", "PAGEDOWN"} else -delta),
                    )
            elif self.overlay == "files":
                chosen = self.file_picker.handle(key)
                if chosen is not None:
                    self._close_files(chosen)
            elif self.overlay == "validation":
                if key == "v":
                    self._check_settings()
                elif key == "ENTER":
                    self._fix_setting()
                elif key in {"UP", "DOWN", "PAGEUP", "PAGEDOWN"}:
                    delta = 8 if key in {"PAGEUP", "PAGEDOWN"} else 1
                    self.validation_index = max(
                        0,
                        min(
                            len(self.validation_issues) - 1,
                            self.validation_index
                            + (delta if key in {"DOWN", "PAGEDOWN"} else -delta),
                        ),
                    )
            elif self.overlay == "backends":
                if key in {"UP", "DOWN"}:
                    self.backend_index = (
                        self.backend_index + (1 if key == "DOWN" else -1)
                    ) % len(PROVIDERS)
                    self.dialog_offset = 0
                elif key in {"PAGEUP", "PAGEDOWN"}:
                    self.dialog_offset = max(
                        0,
                        min(
                            1000000,
                            self.dialog_offset + (8 if key == "PAGEDOWN" else -8),
                        ),
                    )
                elif key == "ENTER":
                    self._set_backend(PROVIDERS[self.backend_index])
                    self.overlay = None
            elif self.overlay == "data":
                if key == "u":
                    self._inspect_data()
                elif key == "e" and self.data_panel.toggle_samples():
                    self.dialog_offset = 0
                elif key in {"LEFT", "RIGHT"} and self.data_panel.report:
                    if self.data_panel.step_sample(1 if key == "RIGHT" else -1):
                        self.dialog_offset = 0
                elif key in {"UP", "DOWN", "PAGEUP", "PAGEDOWN"}:
                    delta = 8 if key in {"PAGEUP", "PAGEDOWN"} else 1
                    self.dialog_offset = max(
                        0,
                        min(
                            1000000,
                            self.dialog_offset
                            + (delta if key in {"DOWN", "PAGEDOWN"} else -delta),
                        ),
                    )
            elif self.overlay == "readiness" and key == "i":
                self._inspect()
            elif self.overlay == "run-search":
                if key == "ENTER":
                    self.run_query = self.search
                    self.overlay = None
                    self.run_index = 0
                    self.refresh()
                elif key == "BACKSPACE":
                    self.search = self.search[:-1]
                elif key == "CLEAR":
                    self.search = ""
                elif len(key) == 1 and key.isprintable():
                    self.search += key
            elif self.overlay in {
                "review",
                "prepared-review",
                "help",
                "details",
                "readiness",
                "metrics",
                "compare",
                "stop",
                "recovery-review",
            } and key in {
                "UP",
                "DOWN",
                "PAGEUP",
                "PAGEDOWN",
            }:
                delta = 1 if key in {"UP", "DOWN"} else 8
                self.dialog_offset = max(
                    0,
                    min(
                        1000000,
                        self.dialog_offset
                        + (delta if key in {"DOWN", "PAGEDOWN"} else -delta),
                    ),
                )
            elif self.overlay == "models":
                if key in {"UP", "DOWN"}:
                    self.model_index = max(
                        0,
                        min(
                            max(0, len(self.filtered_models()) - 1),
                            self.model_index + (1 if key == "DOWN" else -1),
                        ),
                    )
                elif key == "ENTER" and self.filtered_models():
                    self._set_model(self.filtered_models()[self.model_index][0])
                    self.overlay = None
                    self.plan = None
                elif key == "BACKSPACE":
                    self.search = self.search[:-1]
                    self.model_index = 0
                elif len(key) == 1 and key.isprintable():
                    self.search += key
                    self.model_index = 0
            elif self.overlay == "prepared-review" and key == "ENTER":
                self._launch_prepared()
            elif self.overlay == "review" and key == "ENTER" and self.plan:
                plan, workspace = self.plan, self.workspace

                def launch_reviewed() -> Path:
                    require_backend_credentials(plan.config.provider)
                    run = create_run(workspace, plan)
                    launch_run(run)
                    return run

                if self._submit(
                    "launch",
                    launch_reviewed,
                    "Submitting the reviewed launch in the background; closing the studio waits for it.",
                    authorized=True,
                ):
                    self.overlay = None
            elif self.overlay == "details" and key == "c":
                self._reuse_run()
                self.overlay = None
            elif self.overlay == "details" and key == "r":
                self._review_prepared()
            elif self.overlay == "details" and key == "l":
                self._open_logs()
            elif self.overlay == "details" and key == "a":
                self._pin_reference()
            elif self.overlay == "details" and key == "x":
                self._open_comparison()
            elif self.overlay == "details" and key == "e":
                self._open_quality()
            elif self.overlay == "details" and key == "j":
                self._open_evaluation()
            elif self.overlay == "details" and key == "k":
                self._review_stop()
            elif self.overlay == "details" and key == "z":
                self._open_recovery()
            return True
        if key in {"q", "QUIT"}:
            return False
        if key == "TAB":
            self.tab = "runs" if self.tab == "configure" else "configure"
            self.log_offset = 0
        elif key in {":", "COMMANDS"}:
            self.action_menu = ActionMenu(
                self.tab, has_run=bool(self.runs), searchable=key == "COMMANDS"
            )
            self.action_run_id = (
                self.runs[self.run_index]["id"]
                if self.tab == "runs" and self.runs
                else None
            )
            self.overlay = "actions"
        elif key == "?":
            self.overlay = "help"
            self.dialog_offset = 0
        elif key == "i":
            self._inspect()
        elif self.tab == "configure":
            fields = self.fields()
            self.field = min(self.field, len(fields) - 1)
            field = fields[self.field]
            if key in {"UP", "DOWN"}:
                self.field = (self.field + (1 if key == "DOWN" else -1)) % len(fields)
            elif key in {"HOME", "END", "PAGEUP", "PAGEDOWN"}:
                if key in {"HOME", "END"}:
                    self.field = 0 if key == "HOME" else len(fields) - 1
                else:
                    self.field = max(
                        0,
                        min(
                            len(fields) - 1,
                            self.field
                            + self.field_page_size * (1 if key == "PAGEDOWN" else -1),
                        ),
                    )
            elif key in {"ENTER", "LEFT", "RIGHT"}:
                if field.choices:
                    index = field.choices.index(str(self.values[field.name]))
                    value = field.choices[
                        (index + (-1 if key == "LEFT" else 1)) % len(field.choices)
                    ]
                    if field.name == "provider":
                        self._set_backend(value)
                    else:
                        self.values[field.name] = value
                    self.field = min(self.field, len(self.fields()) - 1)
                    self.plan = None
                elif key == "ENTER":
                    self.editing = str(self.values[field.name])
                    self.edit_cursor = len(self.editing)
            elif key == "w":
                self._open_workflow()
            elif key == "m":
                self.overlay, self.search, self.model_index = "models", "", 0
            elif key == "u":
                self._inspect_data()
            elif key == "v":
                self._check_settings()
            elif key == "b":
                self.backend_index = PROVIDERS.index(str(self.values["provider"]))
                self.overlay = "backends"
                self.dialog_offset = 0
            elif key == "f":
                try:
                    self._open_files()
                except (OSError, ValueError, RuntimeError) as error:
                    self.overlay = None
                    self.message = (
                        f"Cannot open dataset picker ({type(error).__name__})."
                    )
            elif key in {"p", "r"}:
                self.preview(review=key == "r")
            elif key == "s":
                try:
                    config = self.config()
                    self.workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
                    write_json(self.workspace / "draft.json", asdict(config))
                    self.message = f"Saved {self.workspace / 'draft.json'}"
                except (OSError, ValueError, TypeError) as error:
                    self.message = f"Cannot save: {error}"
        elif key in {"UP", "DOWN"}:
            self.run_index = max(
                0,
                min(
                    max(0, len(self.runs) - 1),
                    self.run_index + (1 if key == "DOWN" else -1),
                ),
            )
            self.log_offset = 0
        elif key in {"PAGEUP", "PAGEDOWN"}:
            self.log_offset = max(0, self.log_offset + (12 if key == "PAGEUP" else -12))
        elif key == "HOME":
            self.run_index, self.log_offset = 0, 0
        elif key == "END":
            self.log_offset = 0
        elif key in {"ENTER", "d"} and self.runs:
            self.overlay = "details"
            self.dialog_offset = 0
        elif key == "c":
            self._reuse_run()
        elif key == "r":
            self._review_prepared()
        elif key == "/":
            self.overlay = "run-search"
            self.search = self.run_query
        elif key == "t":
            filters = ("all", "active", "failed", "succeeded")
            self.run_filter = filters[
                (filters.index(self.run_filter) + 1) % len(filters)
            ]
            self.refresh()
        elif key == "BACKSPACE":
            self.run_query, self.run_filter = "", "all"
            self.refresh()
        elif key == "g" and self.runs:
            self.overlay = "metrics"
            self.dialog_offset = 0
        elif key == "l" and self.runs:
            self._open_logs()
        elif key == "a":
            self._pin_reference()
        elif key == "x":
            self._open_comparison()
        elif key == "e":
            self._open_quality()
        elif key == "j":
            self._open_evaluation()
        elif key == "k":
            self._review_stop()
        elif key == "z":
            self._open_recovery()
        return True

    def _editor_line(self, width: int) -> str:
        text = self.editing or ""
        self.edit_cursor = min(self.edit_cursor, len(text))
        start = self.edit_cursor
        space = max(1, width - 8)
        while start > 0 and cell_len(text[start - 1 : self.edit_cursor]) < space:
            start -= 1
        prefix = "EDIT< " if start else "EDIT: "
        self.editor_column = cell_len(prefix + text[start : self.edit_cursor])
        return prefix + str(set_cell_size(_safe(text[start:]), max(0, width - 6)))

    def _footer(self, width: int) -> list[str]:
        if self.editing is not None:
            return [
                "-" * width,
                self._editor_line(width),
                (
                    "Enter save | Esc discard"
                    if self.compact_layout or width < COMPACT_COLUMNS
                    else "Arrows/Home/End move | Enter save | Esc discard | Ctrl+U clear"
                ),
            ]
        status_message = (
            self.evaluation_panel.message.split("\n", 1)[0]
            if self.overlay == "evaluation" and self.evaluation_panel is not None
            else self.message
        )
        message = _wrap([status_message], width)[:2]
        if self.compact_layout or width < COMPACT_COLUMNS:
            return [
                "-" * width,
                *message,
                *[_clip(key, width) for key in self._compact_keys()],
            ]
        if self.overlay == "planning":
            keys = ["Esc cancel preview | Ctrl+C exit"]
        elif self.overlay == "actions":
            keys = [
                (
                    "Type to search | Arrows select | Enter open | Esc back"
                    if self.action_menu and self.action_menu.searchable
                    else "Arrows/PgUp/PgDn select | Enter open | Esc back"
                )
            ]
        elif self.overlay in {"review", "prepared-review"}:
            keys = ["PgUp/Dn scroll | Enter launch | Esc back"]
        elif self.overlay == "details":
            keys = [
                "r review prepared | c reuse | l logs | PgUp/Dn scroll | Esc back",
                "a pin/unpin reference | x compare",
            ]
        elif self.overlay == "models":
            keys = ["Type to search | Up/Down select | Enter choose | Esc back"]
        elif self.overlay == "backends":
            keys = [
                "Up/Down select | Enter choose backend | PgUp/Dn details | Esc cancel"
            ]
        elif self.overlay == "validation":
            keys = ["Up/Down select | Enter fix field | v recheck | Esc back"]
        elif self.overlay == "files":
            keys = [
                "Type filter | Arrows/PgUp/Dn | Enter select | Backspace up | Esc back"
            ]
        elif self.overlay == "run-search":
            keys = ["Type keywords | Enter apply | Ctrl+U clear | Esc cancel"]
        elif self.overlay == "data":
            keys = [
                "u rescan | e show/hide private sample | Left/Right sample",
                "PgUp/Dn scroll | Esc back",
            ]
        elif self.overlay == "readiness":
            keys = ["i recheck | PgUp/Dn scroll | Esc back"]
        elif self.overlay == "workflow":
            keys = [
                "Up/Down select | Enter open | w recheck | PgUp/Dn scroll | Esc back"
            ]
        elif self.overlay == "evaluation" and self.evaluation_panel is not None:
            keys = self.evaluation_panel.keys()
        elif self.overlay == "quality":
            keys = ["e reverify | f import | j run evaluation | PgUp/Dn | Esc back"]
        elif self.overlay == "quality-path":
            keys = [
                "Type evidence path | Enter verify/import | Ctrl+U clear | Esc back"
            ]
        elif self.overlay == "stop":
            keys = ["Enter requests local stop | Esc keeps training"]
        elif self.overlay == "recovery":
            keys = [
                "Up/Down select | Enter review | z recheck | PgUp/Dn scroll | Esc back"
            ]
        elif self.overlay == "recovery-review":
            keys = [
                "Enter copies checkpoint / prepares NEW run | PgUp/Dn scroll | Esc back"
            ]
        elif self.overlay == "compare":
            keys = ["PgUp/Dn scroll | Esc back | Snapshot; reopen x to refresh"]
        elif self.overlay == "metrics":
            keys = ["PgUp/Dn scroll | Esc back | Charts use reported samples"]
        elif self.overlay == "logs":
            keys = (
                self.log_viewer.keys(width)
                if self.log_viewer is not None
                else list(LogViewer.BROWSE_KEYS)
            )
        elif self.overlay == "help":
            keys = ["PgUp/Dn scroll | Esc back | Ctrl+C exit"]
        elif self.tab == "runs":
            keys = [
                "Enter details | l logs | r review prepared | c reuse | g metrics",
                "/ search | t status | : actions | Arrows select | PgUp/Dn logs | Tab | q quit",
                "a pin | x compare | e quality | j evaluate | k stop | z recover",
            ]
            if self.run_query or self.run_filter != "all":
                keys.append("Backspace clears filters | End follows the log tail")
        else:
            keys = [
                "Arrows/Enter | b backends | m models | v check | p preview | r review",
                "f files | u data | i readiness | s save | Tab runs | ? help | q quit",
                "w guided setup | Ctrl+P commands | : actions | ? help",
            ]
        return ["-" * width, *message, *_wrap(keys, width)]

    def _text_input_active(self) -> bool:
        """Keep phone navigation digits literal in every text/filter editor."""
        return (
            self.editing is not None
            or (
                self.overlay == "actions"
                and self.action_menu is not None
                and self.action_menu.searchable
            )
            or self.overlay in {"quality-path", "run-search", "models", "files"}
            or (
                self.overlay == "logs"
                and self.log_viewer is not None
                and self.log_viewer.editing
            )
            or (
                self.overlay == "evaluation"
                and self.evaluation_panel is not None
                and self.evaluation_panel.mode == "edit"
            )
        )

    def _compact_keys(self) -> list[str]:
        """Keep phone controls visible while preserving explicit launch confirmation."""
        if self.overlay == "planning":
            return ["0/Esc cancel preview", "Ctrl+C exits"]
        if self.overlay == "actions":
            if self.action_menu and self.action_menu.searchable:
                return ["Type to search | Enter open", "Arrows move | Esc back"]
            return ["2/8 move 3/9 page", "Enter open 0 back"]
        if self._text_input_active():
            if self.overlay in {"models", "files"}:
                return ["Type filter | Enter choose", "Backspace edits | Esc back"]
            if self.overlay == "quality-path":
                return ["Type path | Enter verify", "Backspace edits | Esc back"]
            return ["Type text | Enter save/apply", "Backspace edits | Esc back"]
        if self.overlay == "evaluation" and self.evaluation_panel is not None:
            mode = self.evaluation_panel.mode
            keys = {
                "list": ["2/8 move Enter details 0 back", "n new h saved g compare"],
                "form": ["2/8 field Enter edit 0 back", "f cases h saved r review"],
                "comparison": [
                    "2/8 rows Enter details 0 back",
                    "n nominate h saved r status",
                ],
                "nominations": [
                    "2/8 rows Enter restore 0 back",
                    "r refresh; selection metadata",
                ],
                "targets": ["2/8 rows Enter choose 0 back", "c refresh"],
                "nomination": [
                    "2/8 scroll Enter retains 0 back",
                    "Selection receipt only",
                ],
                "review": ["2/8 scroll 3/9 page 0 back", "Enter launches inference"],
                "stop": ["Enter confirms local stop", "2/8 scroll | 0 back"],
                "detail": ["2/8 scroll | 0 back", "r review v verify l logs"],
            }
            return keys.get(
                mode, ["2/8 scroll | 0 back", "Enter confirms reviewed action"]
            )
        if self.overlay in {"review", "prepared-review"}:
            return ["2/8 scroll 3/9 page 0 back", "Enter launches training"]
        if self.overlay in {"stop", "recovery-review"}:
            return ["2/8 scroll 3/9 page 0 back", "Enter confirms reviewed action"]
        if self.overlay == "logs":
            return ["2/8 scroll / filter 0 back", "Space follow s source"]
        actions = {
            "details": "r review c reuse l logs",
            "backends": "Enter choose backend",
            "validation": "Enter fix v recheck",
            "workflow": "Enter open w recheck",
            "recovery": "Enter review z refresh",
            "data": "u scan e private sample",
            "quality": "f import j eval e verify",
        }
        if self.overlay:
            return [
                "2/8 move 3/9 page 0 back",
                actions.get(self.overlay, "PgUp/PgDn also scroll"),
            ]
        if self.tab == "runs":
            return ["2/8 move Enter details 4 setup", ": actions ? help q exit"]
        return ["2/8 move Enter edit 4 runs", "m model : actions ? help q quit"]

    def _configure_lines(self, width: int, available: int) -> list[str]:
        fields = self.fields()
        self.field = min(self.field, len(fields) - 1)
        selected = fields[self.field]
        detail = _wrap(
            [
                f"{selected.label}: {_field_text(self.values[selected.name])}",
                selected.help,
            ],
            width,
        )
        if len(detail) > 4:
            detail = detail[:3] + [
                "Enter edits the full value; arrow keys move the cursor."
            ]
        heading = _wrap(["Model: " + str(self.values["model"])], width)[:4]
        heading += [
            f"{self.values['method'].upper()} / {self.values['provider']} | "
            + (
                f"{self.plan.examples} examples validated"
                if self.plan
                else "Inputs not validated"
            )
        ]
        row_budget = max(1, available - len(heading) - len(detail) - 2)
        self.field_page_size = row_budget
        if not self.compact_layout:
            heading[-1] += f" | Setting {self.field + 1}/{len(fields)}"
        start = max(0, self.field - row_budget + 1)
        if width < 112:
            body = [
                (" > " if index == self.field else "   ")
                + f"{field.label:19} "
                + _clip(_field_text(self.values[field.name]), width - 23)
                for index, field in enumerate(fields)
                if start <= index < start + row_budget
            ]
        else:
            split = width // 2
            left = [
                (" > " if index == self.field else "   ")
                + f"{field.label:19} "
                + _field_text(self.values[field.name])
                for index, field in enumerate(fields)
                if start <= index < start + row_budget
            ]
            preview = ["RUN PREVIEW", "", *self._resource_lines(), ""]
            if self.plan:
                preview += [
                    "Input SHA256: " + self.plan.describe()["dataset_sha256"],
                    *self.plan.notes,
                ]
            else:
                preview += [
                    "p: validate data and resources",
                    "r: review before launching",
                    "",
                    "SFT: text chat JSONL",
                    "RL: support scenarios JSON",
                    "No allocation during preview.",
                ]
            right = _wrap(preview, width - split - 2)
            body = [
                set_cell_size(
                    _clip(left[index] if index < len(left) else "", split - 2), split
                )
                + "| "
                + (right[index] if index < len(right) else "")
                for index in range(min(row_budget, max(len(left), len(right))))
            ]
        body += [""] * max(0, row_budget - len(body))
        return [*heading, "", *body, "", *detail]

    def _model_lines(self, width: int, available: int) -> list[str]:
        choices = self.filtered_models()
        header = [
            "MODEL CATALOG / type to search",
            "Search: " + self.search,
            f"{len(choices)} matches | Maturity is not workflow certification.",
        ]
        if not choices:
            return _wrap(
                [
                    *header,
                    "",
                    "No matches. Esc returns; Enter on Model lets you type a custom ID.",
                ],
                width,
            )
        self.model_index = min(self.model_index, len(choices) - 1)
        name, stage = choices[self.model_index]
        detail = _wrap(["Selected: " + name, "Catalog stage: " + stage], width)
        count = max(1, available - len(header) - len(detail) - 1)
        start = max(0, min(self.model_index - count // 2, len(choices) - count))
        rows = []
        for index in range(start, min(len(choices), start + count)):
            model, maturity = choices[index]
            badge = " [" + maturity + "]"
            cells = max(1, width - cell_len(badge) - 4)
            rows.append(
                (" > " if index == self.model_index else "   ")
                + set_cell_size(_clip(model, cells), cells)
                + badge
            )
        rows += [""] * max(0, count - len(rows))
        return [*header, *rows, "", *detail]

    def _set_backend(self, provider: str) -> None:
        self.values["provider"] = provider
        self.validation_field = None
        self.plan = None
        self.message = (
            f"Backend: {BACKENDS[provider]['label']}. Preview and review before launch."
        )
        if provider == "river" and (
            self.values["gpu"] or self.values["gpu_count"] != 0
        ):
            self.values.update(gpu="", gpu_count=0)
            self.message += " Cleared GPU overrides; River manages hardware."
        if provider in {"modal", "river"}:
            self.message += " Choose provider_billed cost policy to acknowledge no enforced dollar ceiling."

    def _backend_lines(self) -> list[str]:
        provider = PROVIDERS[self.backend_index]
        info = BACKENDS[provider]
        credentials = {
            "local": "Local CUDA and stateset-agents[training].",
            "runpod": "RUNPOD_API_KEY from the launching environment.",
            "modal": "stateset-agents[modal]; configured Modal profile or environment credentials.",
            "river": "Python 3.12+, stateset-agents[river], tokenizer dependencies and RIVER_API_KEY.",
        }
        return [
            "TRAINING BACKENDS / no connection or allocation during selection",
            "",
            *[
                (" > " if index == self.backend_index else "   ")
                + str(BACKENDS[name]["label"])
                + " / "
                + name
                for index, name in enumerate(PROVIDERS)
            ],
            "",
            "Selected: " + str(info["label"]),
            "Studio methods: " + ", ".join(info["methods"]),
            "Result: " + str(info["result_kind"]),
            "Billing: " + str(info["cost"]),
            "Setup: " + credentials[provider],
            "",
            "Remote native RL is not remapped to SFT. River RL uses the dedicated River workflow.",
            "Selection preserves training settings; incompatible settings must be fixed before launch.",
        ]

    def _validation_lines(self, width: int, available: int) -> list[str]:
        heading = _wrap(
            [
                "SETTINGS CHECK / configuration only",
                f"Configuration issues: {len(self.validation_issues)} | Enter jumps to a setting",
                "No dataset contents, hardware or provider access checked.",
            ],
            width,
        )
        if self.validation_signature != self._settings_signature():
            heading += _wrap(
                ["[WARNING] Settings changed; v refreshes this report."], width
            )
        if not self.validation_issues:
            return [
                *heading,
                "",
                "Configuration settings passed.",
                *_wrap(
                    [
                        "Press p to validate the dataset, i for local readiness, then r for a separate launch review. Nothing has been prepared or launched."
                    ],
                    width,
                ),
            ]
        self.validation_index = min(
            self.validation_index, len(self.validation_issues) - 1
        )
        labels = {field.name: field.label for field in FIELDS}
        issue = self.validation_issues[self.validation_index]
        detail = _wrap(
            [
                "",
                "Selected: " + labels.get(issue.field, "Config file"),
                issue.message,
                "Enter opens this field, including settings hidden by the backend.",
            ],
            width,
        )
        count = max(1, available - len(heading) - len(detail) - 1)
        start = max(
            0,
            min(
                self.validation_index - count // 2, len(self.validation_issues) - count
            ),
        )
        rows = [
            (" > " if index == self.validation_index else "   ")
            + labels.get(item.field, "Config file")
            + " / "
            + item.code
            for index, item in enumerate(self.validation_issues)
            if start <= index < start + count
        ]
        return [*heading, "", *rows, *detail]

    def _review_prepared(self) -> None:
        """Review the selected retained plan without changing the draft or cloning."""
        if not self.runs:
            self.message = "Select a prepared run first."
            return
        try:
            run = self.workspace / self.runs[self.run_index]["id"]
            if "resume" in read_run_plan(run)[0]:
                if self.recovery_busy:
                    self.message = "Recovery inspection is already running; no new review was opened."
                    return
                self.recovery_run = run
                self.recovery_busy, self.overlay = True, "recovery"
                self.message = (
                    "Verifying the prepared recovery snapshot in the background."
                )

                def inspect() -> None:
                    try:
                        result: dict[str, Any] = {
                            "prepared_review": review_prepared_run(run)
                        }
                    except Exception:
                        result = {
                            "error": "Cannot verify the prepared recovery run; inspect state, code and checkpoint bytes."
                        }
                    self._recovery_results.put(result)

                threading.Thread(
                    target=inspect, daemon=True, name="stateset-recovery-review"
                ).start()
                return
            self._submit(
                "review",
                lambda: review_prepared_run(run),
                "Verifying the retained plan and dataset bytes in the background.",
                authorized=False,
            )
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
            self.prepared_review = None
            self.message = f"Cannot review prepared run: {error}"

    def _launch_prepared(self) -> None:
        """Submit only the run and bytes captured when its review opened."""
        review = self.prepared_review
        if review is not None and review.plan.resume is not None:
            if self.recovery_busy:
                self.message = "Recovery work is already running; no duplicate launch was submitted."
                return
            self.recovery_busy = True
            self.overlay = "recovery"
            self.prepared_review = None
            self.message = "Submitting the reviewed recovery run in the background; closing this view does not cancel an authorized launch."

            def launch() -> None:
                try:
                    launch_run(review.run, expected_plan_sha256=review.plan_sha256)
                    result = {"launched": True, "run_id": review.run.name}
                except Exception:
                    result = {
                        "error": "Recovery launch failed; inspect the prepared run and retained logs before any retry."
                    }
                self._recovery_results.put(result)

            threading.Thread(
                target=launch, daemon=False, name="stateset-recovery-launch"
            ).start()
            return
        if review is None:
            self.prepared_review, self.overlay = None, None
            self.message = "Launch failed: Open a prepared run review first"
            return

        def launch_prepared() -> Path:
            require_backend_credentials(review.plan.config.provider)
            launch_run(review.run, expected_plan_sha256=review.plan_sha256)
            return review.run

        if self._submit(
            "launch-prepared",
            launch_prepared,
            "Submitting the reviewed run in the background; closing the studio waits for it.",
            authorized=True,
        ):
            self.prepared_review, self.overlay = None, None

    def _review_lines(self, plan: RunPlan | None = None) -> list[str]:
        plan = plan or self.plan
        if plan is None:
            return ["No validated launch plan. Esc returns to configuration."]
        cfg = plan.config
        return [
            "REVIEW LAUNCH / Enter launches, Esc returns",
            "",
            f"{cfg.method.upper()} | {cfg.provider}",
            "Model: " + cfg.model,
            "Model commit: " + (cfg.revision or "mutable / unpinned"),
            f"Input snapshot: {plan.examples} examples",
            "SHA256: " + plan.describe()["dataset_sha256"],
            f"Learning rate: {cfg.learning_rate:g} | LoRA rank: {cfg.lora_rank}",
            f"Training seed: {cfg.seed}",
            (
                f"Budget: {cfg.epochs} epochs"
                if cfg.method == "sft"
                else f"Budget: {cfg.steps} updates | KL: {cfg.kl_coefficient:g}"
            ),
            (
                "Timeout: provider request settings; no whole-run deadline"
                if cfg.provider == "river"
                else f"Timeout: {cfg.timeout_seconds}s ({'function execution only' if cfg.provider == 'modal' else 'queue wait is additional'})"
            ),
            (
                f"RunPod pod ceiling: ${cfg.max_cost_usd:.2f}"
                if cfg.provider == "runpod"
                else (
                    "Compute: local CUDA"
                    if cfg.provider == "local"
                    else "Provider billed: NO enforced dollar ceiling"
                )
            ),
            "Result: " + str(BACKENDS[cfg.provider]["result_kind"]),
            "",
            *self._resource_lines(plan),
            "",
            *plan.notes,
            *(
                [
                    "",
                    "RECOVERY LINEAGE",
                    "Parent: " + str(plan.resume["parent_run_id"]),
                    "Checkpoint: " + str(plan.resume["checkpoint"]),
                    "Checkpoint SHA256: " + str(plan.resume["checkpoint_sha256"]),
                    "Original total training budget retained; no fresh-training fallback.",
                ]
                if plan.resume is not None
                else []
            ),
            "",
            "Closing this terminal will not stop a launched job.",
        ]

    def lines(self, width: int = 110, height: int = 32) -> list[str]:
        """Render by terminal cell width, with compact layouts and scrollable details."""
        width, height = max(20, width), max(10, height)
        self.terminal_columns = width
        self.compact_layout = width < COMPACT_COLUMNS or height < COMPACT_ROWS
        header = [
            "STATESET / FINE-TUNING STUDIO",
            ("[CONFIGURE] Runs" if self.tab == "configure" else "Configure [RUNS]")
            + " | Tab switches views",
            "=" * width,
        ]
        if width >= 90 and not self.compact_layout:
            active = sum(
                row.get("state", {}).get("status") in {"queued", "running", "starting"}
                for row in self.all_runs
            )
            context = f"{self.workspace.name}  /  {active} active"
            header[0] = (
                set_cell_size(header[0], max(1, width - cell_len(context) - 2))
                + "  "
                + context
            )
            commands = "Ctrl+P Commands"
            header[1] = (
                set_cell_size(header[1], width - len(commands) - 2) + "  " + commands
            )
        if self.compact_layout:
            header = [
                "STATESET / FINE-TUNING",
                (
                    "[CONFIGURE] Runs | 4 views"
                    if self.tab == "configure"
                    else "Configure [RUNS] | 4 views"
                ),
            ]
        footer = self._footer(width)
        available = max(1, height - len(header) - len(footer))
        if self.overlay == "actions" and self.action_menu is not None:
            body = self.action_menu.lines(available)
        elif self.overlay == "planning":
            body = _wrap(
                [
                    "PREPARING PREVIEW",
                    "",
                    "Reading and validating the dataset in the background.",
                    "Esc cancels this preview. The current read may finish in the background.",
                    "Training requires a separate review and Enter confirmation.",
                ],
                width,
            )
        elif self.overlay == "help":
            body = _wrap(
                [
                    "KEYBOARD",
                    "",
                    "Ctrl+P opens searchable Commands; type words, arrows select and Enter opens a workflow",
                    ": opens Actions on Configure or Runs; arrows select and Enter opens a workflow",
                    "Phone terminals: 32 columns x 16 rows minimum. Compact screens use 2 down, 8 up, 3 page down, 9 page up, 0 back and 4 for the main setup/run views, outside text editors.",
                    "Digits remain literal in values, paths and filters. Enter is still required to confirm a reviewed launch.",
                    "Up/Down choose a setting or run",
                    "Enter edits a setting, opens run details, or confirms a reviewed launch",
                    "Left/Right, Home/End move the text cursor; Delete removes a character",
                    "Ctrl+U clears text; Esc discards an edit or closes a dialog",
                    "a pins/unpins a reference run; x compares the selected run to it",
                    "Comparison shows saved settings and latest metrics; it never ranks quality",
                    "m searches the model catalog",
                    "b selects local CUDA, RunPod, Modal or River (remote SFT)",
                    "v lists all configuration issues; Enter jumps to the selected setting",
                    "f browses JSON/JSONL datasets without reading their contents",
                    "i checks local readiness in the background; it never contacts a provider",
                    "u inspects dataset statistics; e explicitly reveals a bounded private sample",
                    "w guides settings, dataset, runtime, review and quality with explicit next steps",
                    "On Runs: e verifies/imports paired held-out quality evidence; k reviews a local stop request",
                    "z inspects local recovery checkpoints; copying and launch require separate reviews",
                    "j configures a local base/adapter/judge evaluation; review submits through Buildr",
                    "Inside j: b compares completed checkpoint validations; g reopens the queued comparison",
                    "Inside a checkpoint comparison: n reviews a nomination for separate fresh test cases",
                    "Inside j: h reopens saved checkpoint nominations without creating another receipt",
                    "p validates data and previews resources without provisioning",
                    "r reviews exact inputs; on Runs it reviews the same prepared run",
                    "s saves draft.json (credentials are never saved)",
                    "Tab switches configuration / retained runs",
                    "Enter or d opens run details; c copies settings into a NEW run",
                    "/ searches runs; t cycles all/active/failed/succeeded; Backspace resets filters",
                    "g shows measured progress and reported loss/reward/other metric samples",
                    "l opens full logs; s switches sources, Space pauses, / filters, End follows",
                    "PageUp/Down scrolls dialogs or recent logs; End follows log tail",
                    "q exits the main screen; Ctrl+C exits from any screen",
                    "",
                    "Closing the studio preserves detached jobs. Reopen the same workspace to reconnect.",
                    "Local training requires CUDA. RunPod SFT is paid and requires RUNPOD_API_KEY in the launching environment.",
                    "The studio starts no job when opening, previewing, viewing, or copying settings.",
                ],
                width,
            )
        elif self.overlay == "models":
            body = self._model_lines(width, available)
        elif self.overlay == "backends":
            body = _wrap(self._backend_lines(), width)
        elif self.overlay == "validation":
            body = self._validation_lines(width, available)
        elif self.overlay == "files":
            body = self.file_picker.lines(width, available)
        elif self.overlay == "run-search":
            matches = self._filter_runs(self.search)
            body = _wrap(
                [
                    "FILTER RUNS / ID, model, method, provider or status",
                    "",
                    "Search: " + self.search,
                    f"{len(matches)} matches in the current {self.run_filter} status filter",
                    "",
                    *[
                        row["id"] + " " + str(_run_config(row).get("model", ""))
                        for row in matches[:8]
                    ],
                    "",
                    "Enter applies; Esc preserves the current filter.",
                ],
                width,
            )
        elif self.overlay == "data":
            body = _wrap(self.data_panel.lines(self._data_settings_signature()), width)
        elif self.overlay == "workflow":
            body = _wrap(self.workflow_panel.lines(self._settings_signature()), width)
        elif self.overlay == "evaluation" and self.evaluation_panel is not None:
            panel = self.evaluation_panel
            body = _wrap(panel.lines(), width)
            if panel.mode in {"form", "edit", "targets", "comparison", "nominations"}:
                selected = next(
                    (
                        index
                        for index, line in enumerate(body)
                        if line.lstrip().startswith("> ")
                    ),
                    0,
                )
                if panel.mode == "edit":
                    selected = next(
                        (
                            index
                            for index, line in enumerate(body)
                            if line.startswith("Edit:")
                        ),
                        selected,
                    )
                if selected < panel.offset or selected >= panel.offset + available:
                    panel.offset = max(0, selected - available // 2)
            panel.offset = min(panel.offset, max(0, len(body) - available))
            body = body[panel.offset :]
        elif self.overlay == "quality":
            body = _wrap(self.quality_panel.lines(), width)
        elif self.overlay == "recovery":
            body = _wrap(self._recovery_lines(), width)
        elif self.overlay == "recovery-review":
            choice = self.recovery_choice or {}
            body = _wrap(
                [
                    "REVIEW CHECKPOINT RECOVERY / prepare only",
                    "",
                    "Parent: "
                    + (self.recovery_run.name if self.recovery_run else "none"),
                    "Checkpoint: " + str(choice.get("checkpoint", "none")),
                    "Checkpoint SHA256: " + str(choice.get("checkpoint_sha256", "")),
                    "Recovery SHA256: " + str(choice.get("recovery_sha256", "")),
                    f"Copy size: {choice.get('bytes', 0):,} bytes",
                    "",
                    "Enter copies the reviewed checkpoint and retained dataset into a NEW private run.",
                    "Training settings and the original total budget stay fixed. Source artifacts are preserved.",
                    "A separate prepared-run launch review follows. Esc prepares nothing.",
                ],
                width,
            )
        elif self.overlay == "quality-path":
            body = _wrap(
                [
                    "IMPORT QUALITY EVIDENCE / private paired reports",
                    "",
                    "Path: " + self.search,
                    "Enter hashes completed weights and validates paired test reports in the background.",
                    "No models or providers are called. Successful import writes private quality.json.",
                ],
                width,
            )
        elif self.overlay == "stop":
            body = _wrap(
                [
                    "REVIEW LOCAL STOP REQUEST",
                    "Run: " + (self.stop_run.name if self.stop_run else "none"),
                    "Plan SHA256: " + self.stop_digest,
                    "",
                    "Enter requests a cooperative stop; Esc leaves the run unchanged.",
                    "An active update finishes first. Native RL saves a checkpoint; SFT requests save/stop at its next step boundary.",
                    "Queued jobs remain in Buildr and exit before loading a model when admitted.",
                    "This does not cancel queue jobs, kill processes or control remote resources.",
                    "After the worker exits, z inspects published local recovery checkpoints.",
                ],
                width,
            )
        elif self.overlay == "readiness":
            body = _wrap(
                self.readiness_panel.lines(self._settings_signature(), self.values),
                width,
            )
        elif self.overlay == "metrics":
            body = _wrap(self._metric_lines(width), width)
        elif self.overlay == "logs":
            body = (
                self.log_viewer.lines(width, available)
                if self.log_viewer is not None
                else ["No log source selected. Esc returns to Runs."]
            )
        elif self.overlay == "review":
            body = _wrap(self._review_lines(), width)
        elif self.overlay == "prepared-review" and self.prepared_review:
            review = self.prepared_review
            body = _wrap(
                [
                    "REVIEW PREPARED RUN / same ID, retained snapshot",
                    "Run: " + review.run.name,
                    "Plan SHA256: " + review.plan_sha256,
                    "Enter launches this run; Esc leaves it prepared.",
                    "",
                    *self._review_lines(review.plan),
                ],
                width,
            )
        elif self.overlay == "compare":
            body = _wrap(self._comparison_lines(), width)
        elif self.overlay == "details":
            body = _wrap(self._run_details(), width)
        elif self.tab == "configure":
            body = self._configure_lines(width, available)
        else:
            body = self._run_lines(width, available)
        if (
            self.compact_layout
            and self._compact_follow_selection
            and self.overlay in {"backends", "workflow", "recovery"}
        ):
            selected_line = next(
                (
                    index
                    for index, line in enumerate(body)
                    if line.lstrip().startswith("> ")
                ),
                None,
            )
            if selected_line is not None:
                if (
                    selected_line < self.dialog_offset
                    or selected_line >= self.dialog_offset + available
                ):
                    self.dialog_offset = max(0, selected_line - available // 2)
                self._compact_follow_selection = False
        if self.overlay in {
            "backends",
            "review",
            "prepared-review",
            "help",
            "details",
            "readiness",
            "metrics",
            "compare",
            "data",
            "workflow",
            "quality",
            "stop",
            "recovery",
            "recovery-review",
        }:
            self.dialog_offset = min(self.dialog_offset, max(0, len(body) - available))
            body = body[self.dialog_offset :]
        result = header + body[:available]
        result += [""] * max(0, height - len(footer) - len(result))
        return [_clip(line, width) for line in (result + footer)[:height]]

    def _progress_lines(
        self, row: dict[str, Any], history: list[dict[str, Any]], width: int
    ) -> list[str]:
        config = _run_config(row)
        current: Any = row["state"].get("committed_updates")
        total: Any = config.get("steps")
        label = "Committed updates"
        if config.get("method") == "sft":
            current = next(
                (
                    record["metrics"]["epoch"]
                    for record in reversed(history)
                    if "epoch" in record["metrics"]
                ),
                None,
            )
            total = config.get("epochs")
            label = "Reported epoch"
        rows = []
        if _finite(current) and current >= 0 and _finite(total) and total > 0:
            fraction = current / total
            cells = max(8, min(24, width - 45))
            fill = int(min(1.0, max(0.0, fraction)) * cells)
            percent = f"{fraction:.0%}" if fraction < 1000 else "exceeds budget"
            rows.append(
                f"{label}: {current:g}/{total:g} ["
                + "#" * fill
                + "." * (cells - fill)
                + "] "
                + percent
            )
        else:
            rows.append(label + ": not reported yet")
        if not history:
            return [
                *rows,
                "No trainer metrics yet; initial loading or queue wait may precede logs.",
            ]
        last = history[-1]
        timestamp = last.get("updated_at")
        if _finite(timestamp) and timestamp >= 0 and timestamp <= time.time():
            rows.append(
                f"Trainer step {last['step']} | Metrics last reported {int(time.time() - timestamp)}s ago"
            )
        else:
            rows.append(f"Trainer step {last['step']} | Metric freshness unavailable")
        keys = (
            "loss",
            "total_loss",
            "policy_loss",
            "average_reward",
            "mean_reward",
            "reward",
            "kl",
            "entropy",
        )
        values = last["metrics"]
        visible = [key for key in keys if key in values][:4]
        if not visible:
            visible = list(values)[:4]
        rows.append(" | ".join(f"{key}={values[key]:.4g}" for key in visible))
        return rows

    def _pin_reference(self) -> None:
        if not self.runs:
            self.message = "Select a retained run to pin as a reference."
            return
        selected = Path(self.runs[self.run_index]["path"])
        self.comparison_reference = (
            None if selected == self.comparison_reference else selected
        )
        self.message = (
            "Reference cleared."
            if self.comparison_reference is None
            else "Reference pinned. Select another run, then x compares saved snapshots."
        )

    def _open_comparison(self) -> None:
        reference = self.comparison_reference
        if reference is None:
            self.message = (
                "Pin a reference run with a, select another run, then press x."
            )
            return
        if not self.runs:
            self.message = (
                "Select a candidate run to compare with the pinned reference."
            )
            return
        candidate = Path(self.runs[self.run_index]["path"])
        if reference == candidate:
            self.message = "Select a different run; the current run is the reference."
            return
        if reference.is_symlink() or not reference.is_dir():
            self.message = (
                "Pinned reference is unavailable. Pin another retained run with a."
            )
            return
        self.comparison = compare_runs(reference, candidate)
        self.overlay = "compare"
        self.dialog_offset = 0
        self.message = "Saved snapshot comparison; no run is changed or launched."

    def _comparison_lines(self) -> list[str]:
        report = self.comparison
        if report is None:
            return ["No comparison snapshot. Esc returns to Runs."]
        left, right = report["reference"], report["candidate"]
        rows = [
            "RUN COMPARISON / candidate minus reference",
            "Reference: " + left["run_id"],
            "Candidate: " + right["run_id"],
            f"Saved status: {left['status']} -> {right['status']}",
            "Dataset hashes: "
            + report["dataset_identity"]
            + " (recorded, not reverified)",
            "Training snapshots do not establish quality or a winner.",
            "",
            "CHANGED SETTINGS",
        ]
        for change in report["config_differences"]:
            a = change["reference"] if change["reference_present"] else "(not recorded)"
            b = change["candidate"] if change["candidate_present"] else "(not recorded)"
            rows += [change["field"] + ":", f"  reference: {a}", f"  candidate: {b}"]
        if not report["config_differences"]:
            rows.append("No differences among recorded configuration fields.")
        rows += [
            "",
            "LATEST TRAINING METRICS / metrics.json",
            f"Reported step: {left['latest_metrics']['step']} -> {right['latest_metrics']['step']}",
            f"Reported time: {left['latest_metrics']['updated_at']} -> {right['latest_metrics']['updated_at']}",
            "Snapshots may differ in step, time, data and objective.",
        ]
        for metric in report["metrics"]:
            a, b, delta = metric["reference"], metric["candidate"], metric["delta"]
            rows += [
                metric["metric"] + ":",
                "  reference: " + (f"{a:.6g}" if a is not None else "not reported"),
                "  candidate: " + (f"{b:.6g}" if b is not None else "not reported"),
                "  delta: " + (f"{delta:+.6g}" if delta is not None else "unavailable"),
            ]
        if not report["metrics"]:
            rows.append("No finite metrics reported in these snapshots.")
        for label, snapshot in (("REFERENCE", left), ("CANDIDATE", right)):
            rows += [
                "",
                label + " PROVENANCE",
                "Plan SHA256: " + str(snapshot["plan_sha256"]),
                "Input SHA256: " + str(snapshot["dataset_sha256"]),
                "Examples: " + str(snapshot["examples"]),
                "Framework: " + str(snapshot["framework_version"]),
                "Package SHA256: " + str(snapshot["package_implementation_sha256"]),
            ]
            if snapshot["unreadable_records"]:
                rows.append(
                    "Unavailable records: " + ", ".join(snapshot["unreadable_records"])
                )
            if snapshot["latest_metrics"]["omitted_fields"]:
                rows.append(
                    "Metric fields omitted (invalid or over limit): "
                    + str(snapshot["latest_metrics"]["omitted_fields"])
                )
        rows += ["", "INTERPRETATION", *report["notes"]]
        return rows

    def _open_logs(self) -> None:
        if not self.runs:
            return
        self.log_viewer = LogViewer(Path(self.runs[self.run_index]["path"]))
        self.overlay = "logs"
        self.message = (
            "Private output; source selection and filtering never change a run."
        )

    def _metric_lines(self, width: int) -> list[str]:
        if not self.runs:
            return ["No retained run selected."]
        row = self.runs[self.run_index]
        history = metric_history(Path(row["path"]), limit=120)
        rows = [
            "TRAINING METRICS / reported samples only",
            "Run: " + row["id"],
            "",
            *self._progress_lines(row, history, width),
            "",
        ]
        if not history:
            return [
                *rows,
                "History starts when the trainer reports metrics.",
                "Older runs can show their latest snapshot without a trend.",
            ]
        rows.append(
            f"{len(history)} retained events | reported steps {history[0]['step']}..{history[-1]['step']}"
        )
        keys = [
            key
            for key in (
                "loss",
                "total_loss",
                "policy_loss",
                "average_reward",
                "mean_reward",
                "reward",
                "kl",
                "entropy",
                "grad_norm",
                "learning_rate",
            )
            if any(key in record["metrics"] for record in history)
        ]
        if not keys:
            keys = list(history[-1]["metrics"])[:8]
        for key in keys:
            values = [
                record["metrics"][key] for record in history if key in record["metrics"]
            ]
            rows += [
                "",
                f"{key}: last {values[-1]:.4g} | range {min(values):.4g}..{max(values):.4g} | {len(values)} samples",
            ]
            rows.append(
                metric_sparkline(values, width=max(12, min(72, width - 4)))
                if len(values) > 1
                else "One sample; more logs are needed for a trend."
            )
        rows += [
            "",
            "Charts sample reported values; intervals are not assumed uniform.",
            "No ETA is inferred. Training reward is not independent quality evidence.",
        ]
        return rows

    def _resource_lines(self, plan: RunPlan | None = None) -> list[str]:
        plan = plan or self.plan
        values = asdict(plan.config) if plan else self.values
        if values["provider"] != "local":
            lines = ["Local queue: not held during remote training"]
            if plan and values["provider"] == "runpod":
                resources = plan.resources
                lines += [
                    f"GPU: {resources['gpu_count']} x {resources['gpu']}",
                    f"Disk: {resources['container_disk_gb']} GB",
                    "Resource evidence: " + resources["recommendation_evidence"],
                ]
            elif values["provider"] == "modal":
                resources = plan.resources if plan else {}
                lines += [
                    f"GPU: {resources.get('gpu_count', values['gpu_count'] or 1)} x {resources.get('gpu', values['gpu'] or 'A10')}",
                    "Artifacts: downloaded weights",
                    "Billing: no enforced dollar ceiling",
                ]
            elif values["provider"] == "river":
                lines += [
                    "Hardware: managed by River",
                    "Artifacts: hosted river:// checkpoint pointer",
                    "Billing: no enforced dollar ceiling",
                ]
            return lines
        try:
            queue = queue_executable()
        except ValueError as error:
            return [str(error)]
        lines = [
            "Queue: "
            + (Path(queue).name if queue else "not installed (direct worker)"),
            f"Local reservation: {values['cpu_slots']} CPUs / {values['memory_gb']} GiB RAM",
            f"Scratch reservation: {values['scratch_gb']} GiB",
        ]
        return lines

    def _run_lines(self, width: int, available: int) -> list[str]:
        if self.history_error:
            return _wrap(
                [
                    self.history_error,
                    "Last loaded runs remain available through search and Actions.",
                ],
                width,
            )
        if self.background_history and not self.history_loaded:
            return _wrap(
                [
                    "Loading retained runs in the background...",
                    "You can keep configuring training while history loads.",
                ],
                width,
            )
        if not self.runs:
            if self.run_query or self.run_filter != "all":
                return _wrap(
                    [
                        "No runs match the current filters.",
                        f"Status: {self.run_filter} | Search: {self.run_query or '(none)'}",
                        "Use / to edit search, t to change status, or Backspace to reset.",
                        f"{len(self.all_runs)} retained runs in this workspace.",
                    ],
                    width,
                )
            return _wrap(
                [
                    "No retained runs yet.",
                    "Tab to configure, then p to preview or r to review a launch.",
                    "Workspace: " + str(self.workspace),
                ],
                width,
            )
        selected = self.runs[self.run_index]
        state = selected["state"]
        run = Path(selected["path"])
        config = _run_config(selected)
        rows = [
            f"RUNS {len(self.runs)}/{len(self.all_runs)} | {self.run_filter} | Search: {self.run_query or '(none)'}",
            "Reference: "
            + (
                self.comparison_reference.name
                if self.comparison_reference
                else "(none; a pins selected run)"
            ),
            "",
        ]
        count = min(8, max(1, available // 3))
        start = max(0, min(self.run_index - count // 2, len(self.runs) - count))
        for index, row in enumerate(self.runs[start : start + count], start):
            rows.append(
                (" > " if index == self.run_index else "   ")
                + _clip(
                    f"{row['id']} [{row['state'].get('status', 'unknown')}] {_run_config(row).get('model', '')}",
                    width - 4,
                )
            )
        rows += [
            "",
            f"{str(config.get('method', '?')).upper()} | {state.get('status', 'unknown')} | {config.get('provider', '?')}",
        ]
        if state.get("execution_unconfirmed"):
            rows.append(
                "Supervisor lost; detached work may still be queued or running."
            )
        if state.get("teardown_error"):
            rows.append(_teardown_note(state))
        if state.get("stop_requested") and state.get("status") not in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            rows.append("Stop requested; awaiting a safe worker boundary.")
        if state.get("error"):
            rows += _wrap([str(state["error"])], width)[:2]
        diagnosis = failure_info(run, state)
        if diagnosis:
            rows += _wrap(["Next: " + diagnosis["action"]], width)[:2]
        history = metric_history(run, limit=120)
        rows += self._progress_lines(selected, history, width)[:3]
        rows += ["-" * width]
        rows += log_tail(
            run, lines=max(1, available - len(rows)), offset=self.log_offset
        )
        return rows

    def _run_details(self) -> list[str]:
        if not self.runs:
            return ["No retained run is selected."]
        row = self.runs[self.run_index]
        config = _run_config(row)
        state = row["state"]
        run = Path(row["path"])
        result = [
            "RUN DETAILS / c reuses settings for a NEW run",
            "",
            "Run: " + row["id"],
            f"Status: {state.get('status', 'unknown')}",
            "Model: " + str(config.get("model", "unknown")),
            "Model commit: " + str(config.get("revision") or "mutable / unpinned"),
            f"Method: {str(config.get('method', '?')).upper()} | Provider: {config.get('provider', '?')}",
        ]
        if state.get("teardown_error"):
            result.append(_teardown_note(state))
        if state.get("recovery_unavailable"):
            result.append(
                "Recovery checkpoints: unavailable ("
                + _safe(str(state["recovery_unavailable"]))
                + "); training continued without recovery receipts."
            )
        if state.get("execution_unconfirmed"):
            result += [
                "Execution: unconfirmed; detached work may still be queued or running.",
                (
                    "Local stop receipt is saved; worker exit is not yet confirmed."
                    if state.get("stop_requested")
                    else (
                        "k reviews a cooperative stop for a local run."
                        if config.get("provider") == "local"
                        else "Check the retained provider handle and account before retrying."
                    )
                ),
            ]
        start = state.get("started_at", state.get("created_at"))
        end = state.get("finished_at", time.time())
        if (
            isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and 0 <= end - start < 10**9
        ):
            seconds = int(end - start)
            result.append(
                f"Elapsed (including queue wait): {seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"
            )
        if state.get("error"):
            result += ["", "Error: " + str(state["error"])]
        diagnosis = failure_info(run, state)
        if diagnosis:
            result += [
                "",
                "FAILURE AND NEXT STEPS",
                diagnosis["title"],
                f"Code: {diagnosis['code']} | Stage: {diagnosis['stage']}",
                diagnosis["action"],
            ]
            if not diagnosis["recorded"]:
                result.append(
                    "No valid structured diagnosis is retained; the cause has not been inferred from logs."
                )
            if config.get("provider") != "local":
                result.append(
                    "Check the retained provider handle and your provider account for active resources or charges before a new paid launch."
                )
            result.append(
                "c reuses settings for a NEW reviewed run; it does not resume a checkpoint or retry a provider job."
            )
        if config.get("provider") != "local":
            result += ["", "PROVIDER SUBMISSION / RETAINED OBSERVATIONS"]
            try:
                provider = redact_finetune_secrets(inspect_submission(run))
                if not provider["available"]:
                    result.append(
                        "No submission receipt retained; provider identity and cleanup are unknown."
                    )
                else:
                    result += [
                        _safe(
                            f"Logical job: {provider.get('job_id') or 'not yet recorded'}"
                        ),
                        _safe(
                            f"Submission: {provider['submission_state']} | Observed result: {provider.get('observed_status') or 'not reported'}"
                        ),
                    ]
                    for resource in provider["resources"]:
                        result.append(
                            _safe(
                                f"{resource['kind']}: {resource['id']} | {resource['state'].replace('_', ' ')}"
                            )
                        )
                        if resource.get("name"):
                            result.append(_safe("Owned name: " + resource["name"]))
                    for intent in provider["unresolved_intents"]:
                        result.append(
                            _safe(
                                f"Unresolved allocation: {intent['resource_kind']} | {intent.get('resource_name') or 'identity not returned'}"
                            )
                        )
                    result.append(
                        "Cleanup required or unconfirmed; reconcile provider records before retrying."
                        if provider["cleanup_required_or_unconfirmed"]
                        else "No unresolved cleanup in this receipt; live provider state is unchecked."
                    )
                result += [
                    "Provider status is not checked live; this view makes no provider request.",
                    "Cleanup acknowledgement does not prove billing stopped. No automatic restart.",
                ]
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                RecursionError,
                OverflowError,
            ):
                result.append(
                    "Provider receipt unavailable; retained identity and cleanup could not be verified."
                )
        result += [
            "",
            "INPUT AND SETTINGS",
            f"Examples: {row['plan'].get('examples', 'unknown')}",
            "Input SHA256: " + str(row["plan"].get("dataset_sha256", "unavailable")),
            f"Learning rate: {config.get('learning_rate', '?')} | LoRA rank: {config.get('lora_rank', '?')}",
            f"Training seed: {config.get('seed', 'not recorded')}",
            (
                f"Budget: {config.get('epochs', '?')} epochs"
                if config.get("method") == "sft"
                else f"Budget: {config.get('steps', '?')} updates | Recorded: {state.get('committed_updates', 'not reported')}"
            ),
            "Dataset snapshot: " + str(run / "dataset"),
        ]
        resume = row["plan"].get("resume")
        if isinstance(resume, dict):
            result += [
                "",
                "RECOVERY LINEAGE",
                "Parent: " + str(resume.get("parent_run_id", "unknown")),
                "Checkpoint: " + str(resume.get("checkpoint", "unknown")),
                "Checkpoint SHA256: " + str(resume.get("checkpoint_sha256", "unknown")),
                "Original total training budget retained.",
            ]
        for filename, label, keys in (
            (
                "completion.json",
                "COMPLETION",
                (
                    "output_dir",
                    "committed_updates",
                    "quality_gain_verified",
                    "result_kind",
                    "checkpoint",
                ),
            ),
            ("provider-job.json", "PROVIDER HANDLE", ("provider", "job_id")),
            (
                "provider-result.json",
                "PROVIDER-REPORTED COST",
                ("cost_usd", "duration_s"),
            ),
        ):
            if not (run / filename).exists():
                continue
            result += ["", label]
            try:
                record = read_json(run / filename)
                result += [f"{key}: {record[key]}" for key in keys if key in record]
            except (OSError, ValueError, TypeError):
                result.append(filename + " is unreadable; inspect the retained record.")
        try:
            metrics = read_json(run / "metrics.json")
            if isinstance(metrics.get("metrics"), dict):
                result += [
                    "",
                    "LATEST METRICS",
                    f"Reported step: {metrics.get('step', '?')}",
                ]
                result += [
                    f"{key}: {value}" for key, value in metrics["metrics"].items()
                ]
        except (OSError, ValueError, TypeError):
            pass
        result += [
            "",
            "FILES",
            "Workspace: " + str(run),
            "Training log: " + str(run / "run.log"),
            "Supervisor log: " + str(run / "supervisor.log"),
            "",
            "AI AGENT ACCESS / JSON",
            shlex.join(
                [
                    sys.executable,
                    "-m",
                    "stateset_agents.cli",
                    "finetune",
                    "status",
                    row["id"],
                    "--workspace",
                    str(self.workspace),
                ]
            ),
            "Use finetune schema for agent commands and config fields.",
            "",
            "Reusing settings creates a new configuration from this private input snapshot.",
            "No job starts until you validate and confirm a new launch review.",
            "Completion alone does not establish a quality gain.",
        ]
        return result


def run_studio(workspace: Path, config: FinetuneConfig | None = None) -> None:
    """Open the curses studio on the caller's terminal, including a Herdr pane."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError(
            "The studio needs an interactive terminal; use --snapshot for a static view"
        )
    try:
        import curses
    except ImportError as error:
        raise ValueError(
            "This terminal UI requires POSIX curses (use WSL on Windows)"
        ) from error

    studio = Studio(workspace, config, background_history=True)

    def main(screen: Any) -> None:
        configure_terminal(screen, curses)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        colors = False
        if curses.has_colors():
            try:
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_CYAN, -1)
                curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)
                curses.init_pair(3, curses.COLOR_GREEN, -1)
                curses.init_pair(4, curses.COLOR_YELLOW, -1)
                curses.init_pair(5, curses.COLOR_RED, -1)
                colors = True
            except curses.error:
                pass
        mapping = {
            curses.KEY_UP: "UP",
            curses.KEY_DOWN: "DOWN",
            curses.KEY_LEFT: "LEFT",
            curses.KEY_RIGHT: "RIGHT",
            curses.KEY_NPAGE: "PAGEDOWN",
            curses.KEY_PPAGE: "PAGEUP",
            curses.KEY_END: "END",
            curses.KEY_HOME: "HOME",
            curses.KEY_DC: "DELETE",
            curses.KEY_BACKSPACE: "BACKSPACE",
            curses.KEY_ENTER: "ENTER",
        }
        renderer = FrameRenderer(curses, colors=colors)
        last_history_refresh = 0.0
        while True:
            now = time.monotonic()
            history_due = now - last_history_refresh >= 1.0
            studio.refresh(history=history_due)
            if history_due:
                last_history_refresh = now
            height, width = screen.getmaxyx()
            usable = width >= MIN_TERMINAL_COLUMNS and height >= MIN_TERMINAL_ROWS
            if not usable:
                frame = _wrap(
                    [
                        "StateSet Fine-tuning Studio",
                        "Paused: terminal too small.",
                        "Need 32 columns x 16 rows.",
                        "Resize to resume editing.",
                        "q exits; existing jobs keep running.",
                    ],
                    max(1, width - 1),
                )
            else:
                frame = studio.lines(width - 1, height)
            renderer.draw(screen, frame[:height], width, height)
            screen.refresh()
            try:
                editing = studio.editing is not None and usable
                curses.curs_set(1 if editing else 0)
                if editing:
                    screen.move(height - 2, min(studio.editor_column, width - 2))
                    screen.refresh()
            except curses.error:
                pass
            try:
                key = screen.get_wch()
            except curses.error:
                continue
            if isinstance(key, int):
                normalized = mapping.get(key, "RESIZE")
            else:
                normalized = {
                    "\n": "ENTER",
                    "\r": "ENTER",
                    "\t": "TAB",
                    "\x1b": "ESC",
                    "\x7f": "BACKSPACE",
                    "\x08": "BACKSPACE",
                    "\x15": "CLEAR",
                    "\x03": "QUIT",
                    "\x10": "COMMANDS",
                }.get(key, key)
            # A phone may rotate after drawing but before delivering a key.
            # Repaint before accepting actions against a changed viewport.
            current_height, current_width = screen.getmaxyx()
            if (
                not usable
                or current_width < MIN_TERMINAL_COLUMNS
                or current_height < MIN_TERMINAL_ROWS
            ):
                if normalized in {"q", "QUIT"}:
                    break
                continue
            if (current_height, current_width) != (height, width):
                if normalized == "QUIT":
                    break
                continue
            if not studio.handle(normalized):
                break

    try:
        curses.wrapper(main)
    except curses.error as error:
        raise ValueError(
            "Terminal setup failed; check TERM or use --snapshot"
        ) from error
