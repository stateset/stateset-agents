"""Keyboard-driven fine-tuning studio for Herdr and other POSIX terminals.

The UI imports no training dependencies. Workers own training lifetimes;
closing or reconnecting the terminal never cancels a run.
"""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.cells import cell_len, set_cell_size

from stateset_agents.data.text_sft import DatasetFormatError
from stateset_agents.finetune_evaluation_tui import EvaluationPanel
from stateset_agents.finetune_inspection import (
    compare_runs,
    inspect_readiness,
    inspect_training_data,
    inspect_workflow,
    metric_history,
)
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
    read_run_plan,
    redact_finetune_secrets,
    remote_stop_supported,
    request_stop,
    require_backend_credentials,
    review_prepared_run,
    write_json,
)
from stateset_agents.tui_actions import ActionMenu
from stateset_agents.tui_keys import handle_key
from stateset_agents.tui_panels import FilePicker, LogViewer
from stateset_agents.tui_render import FrameRenderer, configure_terminal
from stateset_agents.tui_reports import (
    DataReport,
    QualityReport,
    ReadinessReport,
    RecoveryPanel,
    WorkflowReport,
)
from stateset_agents.tui_screens import header_lines, render_body, scroll_body
from stateset_agents.tui_tasks import BackgroundTask, TaskResult
from stateset_agents.tui_text import _clip, _safe, _wrap
from stateset_agents.tui_views import (
    _run_config,
    _teardown_note,
    backend_lines,
    comparison_lines,
    metric_lines,
    model_lines,
    progress_lines,
    resource_lines,
    review_lines,
    run_details_lines,
    stop_dialog_lines,
    validation_lines,
)

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
        self.recovery_panel = RecoveryPanel()
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
        self.lease_sweep: BackgroundTask[tuple[list[str], dict[str, str]]] = (
            BackgroundTask()
        )
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
        self._poll_lease_sweep()
        submitted = self.submission.poll()
        if submitted is not None:
            self._finish_submission(submitted)
        selected = self._selected_run_id()
        self._load_history(history)
        self._reconcile_selection(selected)
        self._poll_preview()
        self.readiness_panel.poll()
        self._poll_recovery()
        self._poll_data()
        self._poll_workflow()
        if self.evaluation_panel is not None:
            self.evaluation_panel.refresh()
        self._poll_quality()

    def _poll_lease_sweep(self) -> None:
        """Report RunPod pods the background lease sweep ended or left unconfirmed."""
        swept = self.lease_sweep.poll()
        if swept is None or swept.value is None:
            return
        reclaimed, failed = swept.value
        if reclaimed or failed:
            self.message = (
                f"Ended {len(reclaimed)} RunPod pod(s) left by exited jobs."
                + (
                    f" {len(failed)} could not be confirmed; run "
                    "`stateset-agents runpod-orphans`."
                    if failed
                    else ""
                )
            )

    def _selected_run_id(self) -> str | None:
        """The selected run's ID, captured before the run list is reloaded."""
        selected: str | None = (
            self.runs[self.run_index]["id"]
            if self.runs and self.run_index < len(self.runs)
            else None
        )
        return selected

    def _load_history(self, history: bool) -> None:
        """Collect retained runs, from the background task or synchronously."""
        if self.background_history:
            self._poll_history(history)
        elif history:
            self.all_runs = list_runs(self.workspace, limit=None)
            self.history_loaded = True

    def _poll_history(self, history: bool) -> None:
        """Apply a finished background history read and start the next one."""
        history_result = self.history_task.poll()
        if history_result is not None:
            if history_result.error is not None:
                self.history_error = (
                    "Cannot refresh retained runs; check workspace access. Retrying."
                )
            elif history_result.value is not None:
                self.all_runs = history_result.value
                self.history_loaded = True
                self.history_error = ""
        if history and not self.history_task.busy:
            self.history_task.start(lambda: list_runs(self.workspace, limit=None))

    def _reconcile_selection(self, selected: str | None) -> None:
        """Refilter runs, keep the selection and close views of a hidden run."""
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

    def _poll_preview(self) -> None:
        """Apply a finished preview unless it was cancelled or settings changed."""
        preview_result = self.planning.poll()
        if preview_result is None or preview_result.cancelled:
            return
        if self.planning_signature != self._settings_signature():
            self.message = "Settings changed; press p or r for a fresh preview."
            if self.overlay == "planning":
                self.overlay = None
        elif preview_result.error is not None:
            self.message = (
                f"Fix settings: {redact_finetune_secrets(str(preview_result.error))}"
            )
            if self.overlay == "planning":
                self.overlay = None
        else:
            self.plan = preview_result.value
            if self.plan is not None:
                self.message = (
                    f"Valid: {self.plan.examples} examples. Press r to review launch."
                )
                if self.overlay == "planning":
                    self.overlay = "review" if self.planning_review else None
                    self.dialog_offset = 0

    def _poll_recovery(self) -> None:
        """Apply a finished recovery inspection, preparation or launch."""
        result = self.recovery_panel.poll()
        if result is None:
            return
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

    def _poll_data(self) -> None:
        """Apply a finished dataset scan, hiding samples and updating its status."""
        if not self.data_panel.poll():
            return
        self.data_panel.show_samples = False
        if self.message in {
            "Reading one dataset snapshot; no run is prepared or started.",
            "Dataset scan is already running. Esc leaves it in the background.",
        }:
            if self.data_panel.signature != self._data_settings_signature():
                self.message = "Dataset settings changed; press u to scan again."
            elif self.data_panel.report is None or "error" in self.data_panel.report:
                self.message = "Dataset scan failed; press u to inspect and retry."
            else:
                self.message = "Dataset scan complete; no run was prepared or started."

    def _poll_workflow(self) -> None:
        """Apply a finished guided setup snapshot."""
        if (
            self.workflow_panel.poll()
            and self.message == "Checking guided setup in the background."
        ):
            self.message = (
                "Setup snapshot ready; Enter opens the selected step."
                if self.workflow_panel.signature == self._settings_signature()
                else "Settings changed; press w for a current setup snapshot."
            )

    def _poll_quality(self) -> None:
        """Apply a finished quality inspection."""
        if self.quality_panel.poll() and self.message in {
            "Verifying recorded outcomes and artifact bytes in the background.",
            "Previous artifact inspection is still running; Esc returns.",
        }:
            self.message = (
                "Quality inspection complete; review gates and evidence limits."
            )

    def start_lease_sweep(self) -> bool:
        """End RunPod pods whose submitting process provably exited, in the background.

        Only with ``RUNPOD_API_KEY`` set; live and unverifiable owners are kept.
        """
        import os

        if not os.environ.get("RUNPOD_API_KEY", "").strip():
            return False

        def sweep() -> tuple[list[str], dict[str, str]]:
            from stateset_agents.remote.runpod import RunPodExecutor

            return RunPodExecutor(allow_environment_wheel=False).sweep_abandoned_pods()

        return self.lease_sweep.start(sweep, name="stateset-lease-sweep")

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
            provider = plan["config"].get("provider")
            if provider != "local" and not remote_stop_supported(provider):
                self.message = "Remote stop needs provider-specific cleanup; no stop was requested."
                return
            self.stop_run, self.stop_digest = run, digest
            self.overlay, self.dialog_offset = "stop", 0
        except (OSError, ValueError, TypeError, KeyError):
            self.message = "Cannot verify the selected run for a stop request."

    def _stop_is_remote(self) -> bool:
        """Whether the stop under review targets a remote (RunPod) run."""
        try:
            plan, _ = read_run_plan(self.stop_run) if self.stop_run else ({}, "")
            return plan.get("config", {}).get("provider") not in {None, "local"}
        except (OSError, ValueError, TypeError, KeyError):
            return False

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
        panel = self.recovery_panel
        if panel.busy:
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
        run = captured_run or Path(self.runs[self.run_index]["path"])
        panel.run, panel.report, panel.choice, panel.index = run, None, None, 0
        self.dialog_offset = 0
        self.overlay = "recovery"
        self.message = "Verifying recovery checkpoint bytes in the background."

        def inspect() -> dict[str, Any]:
            try:
                return inspect_checkpoints(run)
            except Exception:
                return {
                    "error": "Recovery requires a terminal local run, stopped workers, matching code, pinned model commit and complete published checkpoint state."
                }

        panel.start(inspect, step="inspection", authorized=False)

    def _prepare_recovery(self) -> None:
        panel = self.recovery_panel
        run, choice = panel.run, panel.choice
        if panel.busy or run is None or choice is None:
            return
        self.overlay = "recovery"
        self.message = "Copying the reviewed checkpoint into a new private run; no training is launched."

        def prepare() -> dict[str, Any]:
            try:
                child = prepare_resume(
                    run,
                    choice["checkpoint"],
                    expected_recovery_sha256=choice["recovery_sha256"],
                )
                return {"prepared_review": review_prepared_run(child)}
            except Exception:
                return {
                    "error": "Recovery preparation failed. Reinspect checkpoint state, current code, worker ownership and free disk space; no launch was attempted."
                }

        panel.start(prepare, step="preparation", authorized=True)

    def handle(self, key: str) -> bool:
        """Apply a normalized key; return false only when the user exits.

        Bindings are tables in :mod:`stateset_agents.tui_keys`.
        """
        return handle_key(self, key)

    def _selected_field(self) -> Field:
        """Clamp the setting cursor to the visible settings and return its field."""
        fields = self.fields()
        self.field = min(self.field, len(fields) - 1)
        return fields[self.field]

    def _open_actions(self, *, searchable: bool) -> None:
        """Open the action menu for the visible tab (searchable for Ctrl+P)."""
        self.action_menu = ActionMenu(
            self.tab, has_run=bool(self.runs), searchable=searchable
        )
        self.action_run_id = (
            self.runs[self.run_index]["id"]
            if self.tab == "runs" and self.runs
            else None
        )
        self.overlay = "actions"

    def _confirm_stop(self) -> None:
        """Submit the reviewed stop request for the captured run and plan digest."""
        run, digest = self.stop_run, self.stop_digest
        if not run:
            return
        if self._submit(
            "stop",
            lambda: request_stop(run, expected_plan_sha256=digest),
            "Saving the local stop request in the background.",
            authorized=True,
        ):
            self.overlay = None

    def _launch_reviewed(self) -> None:
        """Create and launch the reviewed plan off the terminal thread."""
        plan, workspace = self.plan, self.workspace
        if not plan:
            return

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

    def _save_draft(self) -> None:
        """Save the parsed draft settings (never credentials) to the workspace."""
        try:
            config = self.config()
            self.workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
            write_json(self.workspace / "draft.json", asdict(config))
            self.message = f"Saved {self.workspace / 'draft.json'}"
        except (OSError, ValueError, TypeError) as error:
            self.message = f"Cannot save: {error}"

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
            keys = [
                (
                    "Enter signals the RunPod worker to stop | Esc keeps training"
                    if self._stop_is_remote()
                    else "Enter requests local stop | Esc keeps training"
                )
            ]
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
        rows, self.model_index = model_lines(
            self.filtered_models(), self.model_index, self.search, width, available
        )
        return rows

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
        return backend_lines(self.backend_index)

    def _validation_lines(self, width: int, available: int) -> list[str]:
        rows, self.validation_index = validation_lines(
            self.validation_issues,
            self.validation_index,
            self.validation_signature != self._settings_signature(),
            width,
            available,
            {field.name: field.label for field in FIELDS},
        )
        return rows

    def _review_prepared(self) -> None:
        """Review the selected retained plan without changing the draft or cloning."""
        if not self.runs:
            self.message = "Select a prepared run first."
            return
        try:
            run = self.workspace / self.runs[self.run_index]["id"]
            if "resume" in read_run_plan(run)[0]:
                if self.recovery_panel.busy:
                    self.message = "Recovery inspection is already running; no new review was opened."
                    return
                self.recovery_panel.run = run
                self.overlay = "recovery"
                self.message = (
                    "Verifying the prepared recovery snapshot in the background."
                )

                def inspect() -> dict[str, Any]:
                    try:
                        return {"prepared_review": review_prepared_run(run)}
                    except Exception:
                        return {
                            "error": "Cannot verify the prepared recovery run; inspect state, code and checkpoint bytes."
                        }

                self.recovery_panel.start(inspect, step="review", authorized=False)
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
            if self.recovery_panel.busy:
                self.message = "Recovery work is already running; no duplicate launch was submitted."
                return
            self.overlay = "recovery"
            self.prepared_review = None
            self.message = "Submitting the reviewed recovery run in the background; closing this view does not cancel an authorized launch."

            def launch() -> dict[str, Any]:
                try:
                    launch_run(review.run, expected_plan_sha256=review.plan_sha256)
                    return {"launched": True, "run_id": review.run.name}
                except Exception:
                    return {
                        "error": "Recovery launch failed; inspect the prepared run and retained logs before any retry."
                    }

            self.recovery_panel.start(launch, step="launch", authorized=True)
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
        return review_lines(plan or self.plan, self.values)

    def _stop_lines(self) -> list[str]:
        return stop_dialog_lines(
            self.stop_run, self.stop_digest, remote=self._stop_is_remote()
        )

    def lines(self, width: int = 110, height: int = 32) -> list[str]:
        """Render by terminal cell width, with compact layouts and scrollable details."""
        width, height = max(20, width), max(10, height)
        self.terminal_columns = width
        self.compact_layout = width < COMPACT_COLUMNS or height < COMPACT_ROWS
        header = header_lines(self, width)
        footer = self._footer(width)
        available = max(1, height - len(header) - len(footer))
        body = scroll_body(self, render_body(self, width, available), available)
        result = header + body[:available]
        result += [""] * max(0, height - len(footer) - len(result))
        return [_clip(line, width) for line in (result + footer)[:height]]

    def _progress_lines(
        self, row: dict[str, Any], history: list[dict[str, Any]], width: int
    ) -> list[str]:
        return progress_lines(row, history, width)

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
        return comparison_lines(self.comparison)

    def _open_logs(self) -> None:
        if not self.runs:
            return
        self.log_viewer = LogViewer(Path(self.runs[self.run_index]["path"]))
        self.overlay = "logs"
        self.message = (
            "Private output; source selection and filtering never change a run."
        )

    def _metric_lines(self, width: int) -> list[str]:
        return metric_lines(self._selected_row(), width)

    def _resource_lines(self, plan: RunPlan | None = None) -> list[str]:
        return resource_lines(plan or self.plan, self.values)

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
        return run_details_lines(self._selected_row(), self.workspace)

    def _selected_row(self) -> dict[str, Any] | None:
        return self.runs[self.run_index] if self.runs else None


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
    studio.start_lease_sweep()

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
