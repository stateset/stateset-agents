"""Responsive local evaluation editor and monitor for the fine-tuning studio."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.finetune_evaluation import (
    EVALUATION_FIELDS,
    EvaluationConfig,
    EvaluationProtocolError,
    EvaluationReview,
    evaluation_status,
    evaluation_targets,
    inspect_evaluation,
    launch_evaluation,
    list_evaluations,
    plan_evaluation,
    prepare_evaluation,
    read_evaluation,
    stop_evaluation,
)
from stateset_agents.finetune_nomination import (
    list_nominations,
    nomination_config,
    nomination_plan,
    prepare_nomination,
)
from stateset_agents.finetune_validation import (
    ValidationComparisonError,
    comparison_status,
    latest_comparison,
    submit_comparison,
)
from stateset_agents.finetune_workspace import read_json
from stateset_agents.tui_tasks import BackgroundTask


class EvaluationPanel:
    """Capture one run; keep model work outside the UI and behind explicit review."""

    def __init__(self, run: Path) -> None:
        self.run = run
        self.values: dict[str, Any] = asdict(EvaluationConfig())
        self.mode = "list"
        self.field = 0
        self.index = 0
        self.offset = 0
        self.edit = ""
        self.cursor = 0
        self.busy = False
        self.message = (
            "n configures a local paired evaluation. No inference starts on inspection."
        )
        self.review: EvaluationReview | None = None
        self.prepared_path: Path | None = None
        self.rows: list[dict[str, Any]] = []
        self.selected: dict[str, Any] | None = None
        self.result: dict[str, Any] | None = None
        self.targets: list[dict[str, Any]] = []
        self.target_index = 0
        self.comparison_path: Path | None = None
        self.comparison: dict[str, Any] | None = None
        self.comparison_index = 0
        self.nomination_review: dict[str, Any] | None = None
        self.nominations: list[dict[str, Any]] = []
        self.nomination_index = 0
        self.nomination_origin = "list"
        self._task: BackgroundTask[tuple[str, str, Any]] = BackgroundTask()
        self._last_poll = 0.0
        try:
            self.comparison_path = latest_comparison(run)
        except (OSError, ValueError):
            self.message = "Comparison history is unreadable; inspect retained records."
        self.refresh()

    def refresh(self) -> None:
        """Poll small metadata and collect owned background work without blocking."""
        finished = self._task.poll()
        if finished is not None and finished.value is not None:
            operation, origin, result = finished.value
            self.busy = False
            if isinstance(result, dict) and "error" in result:
                self.message = result["error"]
            elif operation == "review":
                self.review = result
                self.prepared_path = None
                if self.mode == origin:
                    self.mode, self.offset = "review", 0
                self.message = (
                    "Inputs captured. Review the complete workload before submitting."
                )
            elif operation == "restore":
                self.review, self.prepared_path = result
                if self.mode == origin:
                    self.mode, self.offset = "review", 0
                self.message = (
                    "Retained evaluation verified; review to submit this same ID."
                )
            elif operation == "launch":
                self.selected = result
                if self.mode == origin:
                    self.mode, self.offset = "detail", 0
                self.message = (
                    "Submitted evaluation "
                    + result["evaluation_id"]
                    + "; training is unchanged."
                )
            elif operation == "verify":
                self.result = result
                self.message = (
                    "Judge recording replay and current artifact binding verified."
                )
            elif operation == "targets":
                self.targets = result["targets"]
                self.target_index = 0
                if self.mode == origin:
                    self.mode, self.offset = "targets", 0
                self.message = "Receipt metadata loaded. Review rehashes the selected checkpoint; validation is used for checkpoint selection."
            elif operation == "comparison":
                self.comparison_path, self.comparison_index = result, 0
                if self.mode == origin:
                    self.mode, self.offset = "comparison", 0
                self.message = "Verification request retained. Buildr owns checkpoint hashing and judge replay; no inference or training starts."
            elif operation == "nomination-review":
                self.nomination_review = result
                if self.mode == origin:
                    self.mode, self.offset = "nomination", 0
                self.message = "Review the selected checkpoint before retaining its fresh-test nomination."
            elif operation == "nominate":
                if self.mode == origin:
                    self._configure_nominated_test(result)
                else:
                    self.message = (
                        "Nomination retained: "
                        + result["nomination_id"]
                        + "; no evaluation started."
                    )
            elif operation == "nominations":
                self.nominations = result["nominations"]
                self.nomination_index = min(
                    self.nomination_index, max(0, len(self.nominations) - 1)
                )
                if self.mode == origin:
                    if origin != "nominations":
                        self.nomination_origin = origin
                        self.nomination_index = 0
                    self.mode, self.offset = "nominations", 0
                self.message = "Saved selection metadata only. Enter restores a fresh-test form; r review checks current checkpoint bytes."
            elif operation == "reopen-nomination":
                if self.mode == origin:
                    self._apply_nomination_config(result)
                else:
                    self.message = (
                        "Saved nomination inspected; the current editor was preserved."
                    )
            elif operation == "stop":
                self.message = "Evaluation stop requested; queue admission and current generation may still finish."
                if self.mode == origin:
                    self.mode = "detail"
            self._last_poll = 0
        if time.monotonic() - self._last_poll >= 2:
            try:
                self.rows = list_evaluations(self.run)
                if self.selected:
                    self.selected = next(
                        (
                            row
                            for row in self.rows
                            if row["evaluation_id"] == self.selected["evaluation_id"]
                        ),
                        self.selected,
                    )
                self.index = min(self.index, max(0, len(self.rows) - 1))
            except (OSError, ValueError):
                self.message = "Cannot read evaluation records; check workspace access."
            if self.comparison_path:
                try:
                    self.comparison = comparison_status(self.comparison_path)
                    result = self.comparison.get("result") or {}
                    self.comparison_index = min(
                        self.comparison_index, max(0, len(result.get("rows", [])) - 1)
                    )
                except (OSError, ValueError, TypeError, KeyError):
                    self.comparison = {
                        "state": {"status": "unreadable"},
                        "result": None,
                    }
            self._last_poll = time.monotonic()

    def _background(
        self, operation: str, action: Callable[[], Any], *, owned: bool = False
    ) -> None:
        if self.busy:
            return
        origin = self.mode
        self.busy = True
        self.message = "Working in the background. Esc leaves the view; authorized submission/stop finishes before exit."

        def work() -> tuple[str, str, Any]:
            result: Any
            try:
                result = action()
            except (EvaluationProtocolError, ValidationComparisonError) as error:
                result = {"error": str(error)}
            except (ValueError, TypeError) as error:
                # Only our static validation errors belong in product output;
                # model/library exceptions may contain private input values.
                result = {
                    "error": "Evaluation could not be verified or submitted. Check cases, pins, adapter, resources and queue availability. "
                    + (
                        str(error)
                        if type(error) is ValueError
                        and str(error).startswith(
                            (
                                "Choose ",
                                "Pin ",
                                "Declare ",
                                "Evaluation requires ",
                                "The trained model ",
                                "Context tokens ",
                            )
                        )
                        else "See the documented evaluation input format."
                    )
                }
            except Exception:
                result = {
                    "error": "Evaluation operation failed; inspect retained state and logs before retrying."
                }
            return operation, origin, result

        # Owned (authorized) work runs on a non-daemon thread: exiting waits.
        self._task.start(
            work, daemon=not owned, name="stateset-evaluation-" + operation
        )

    def _config(self) -> EvaluationConfig:
        values = dict(self.values)
        defaults = asdict(EvaluationConfig())
        for name, default in defaults.items():
            if type(default) is int:
                text = str(values[name])
                if not text.isdecimal():
                    raise ValueError(
                        "Integer evaluation settings require decimal digits"
                    )
                values[name] = int(text)
            elif type(default) is float:
                values[name] = float(values[name])
        config = EvaluationConfig(**values)
        config.validate()
        return config

    def _launch(self) -> None:
        review = self.review
        if review is None or self.busy:
            return
        checksum = review.sha256

        def launch() -> dict[str, Any]:
            from stateset_agents.finetune_evaluation import evaluation_status

            path = self.prepared_path or prepare_evaluation(
                review, expected_plan_sha256=checksum
            )
            launch_evaluation(path, expected_plan_sha256=checksum)
            return evaluation_status(path)

        self._background("launch", launch, owned=True)

    def _compare(self) -> None:
        if self.busy:
            return
        resubmit = False
        if (
            self.mode == "comparison"
            and self.comparison
            and self.comparison.get("anchor_id")
        ):
            anchor = self.comparison["anchor_id"]
            # b on an uncertain submission is the explicit request to retry.
            resubmit = (
                self.comparison.get("state", {}).get("status") == "submission_uncertain"
            )
        else:
            selected = (
                self.selected
                if self.mode == "detail"
                else self.rows[self.index] if self.rows else None
            )
            if (
                not selected
                or selected["state"].get("status") != "succeeded"
                or selected.get("split") != "validation"
                or not selected.get("checkpoint")
            ):
                self.message = "Select a completed published-checkpoint validation, then b compares matching evidence."
                return
            anchor = selected["evaluation_id"]
        self._background(
            "comparison",
            lambda: submit_comparison(self.run, anchor, resubmit=resubmit),
            owned=True,
        )

    def _use_recommendation(self) -> None:
        result = (self.comparison or {}).get("result")
        if not result:
            self.message = "A complete valid comparison is required; inspect its status and Buildr handle before choosing."
            return
        choice = result["recommendation"]
        if choice["kind"] != "checkpoint":
            self.message = "No checkpoint recommendation is available; the base is retained or evidence is incomplete. No adapter was selected."
            return
        try:
            if self.comparison_path is None:
                raise ValueError("Missing comparison record.")
            captured = read_json(self.comparison_path / "plan.json")
            if content_hash(captured) != result["comparison_plan_sha256"]:
                raise ValueError("Comparison protocol changed.")
            anchor_sha = next(
                row["plan_sha256"]
                for row in captured["evaluations"]
                if row["evaluation_id"] == result["anchor_id"]
            )
            anchor = read_json(
                self.run / "evaluations" / result["anchor_id"] / "plan.json"
            )
            if content_hash(anchor) != anchor_sha:
                raise ValueError("Anchor protocol changed.")
            values = asdict(EvaluationConfig(**anchor["config"]))
            values.update(
                checkpoint=choice["checkpoint"],
                split="validation",
                cases=str(
                    self.run / "evaluations" / result["anchor_id"] / "cases.json"
                ),
            )
            EvaluationConfig(**values).validate()
        except (OSError, ValueError, TypeError, KeyError, StopIteration):
            self.message = (
                "Cannot restore the compared protocol; inspect its retained evaluation."
            )
            return
        self.values = values
        self.review, self.prepared_path, self.result = None, None, None
        self.mode, self.offset, self.field = "form", 0, 0
        self.message = "Recommended checkpoint selected for validation. r captures a fresh byte-bound review; training and completed artifacts are unchanged."

    def _configure_nominated_test(self, receipt: dict[str, Any]) -> None:
        try:
            restored = nomination_config(
                self.run,
                receipt["nomination_id"],
                expected_plan_sha256=receipt["plan_sha256"],
            )
        except (OSError, ValueError, TypeError, KeyError):
            self.message = "Nomination retained, but its protocol changed. Inspect the receipt before configuring a test."
            return
        self._apply_nomination_config(restored)

    def _apply_nomination_config(self, restored: dict[str, Any]) -> None:
        self.values = restored["config"]
        self.review, self.prepared_path, self.result = None, None, None
        self.mode, self.offset, self.field = "form", 0, 0
        self.message = (
            "Nomination "
            + restored["nomination"]["nomination_id"]
            + " ready. Choose separate fresh test cases; r rehashes selected bytes before the launch review. No inference started."
        )

    def handle(self, key: str) -> bool:
        """Return true only when Esc should close this entire panel."""
        if self.mode == "edit":
            if key == "ESC":
                self.mode = "form"
            elif key == "ENTER":
                self.values[EVALUATION_FIELDS[self.field][0]] = self.edit
                self.review = None
                self.mode = "form"
            elif key == "CLEAR":
                self.edit, self.cursor = "", 0
            elif key == "BACKSPACE" and self.cursor:
                self.edit = self.edit[: self.cursor - 1] + self.edit[self.cursor :]
                self.cursor -= 1
            elif key == "DELETE":
                self.edit = self.edit[: self.cursor] + self.edit[self.cursor + 1 :]
            elif key in {"LEFT", "RIGHT"}:
                self.cursor = max(
                    0, min(len(self.edit), self.cursor + (1 if key == "RIGHT" else -1))
                )
            elif key in {"HOME", "END"}:
                self.cursor = 0 if key == "HOME" else len(self.edit)
            elif len(key) == 1 and key.isprintable() and len(self.edit) < 4096:
                self.edit = self.edit[: self.cursor] + key + self.edit[self.cursor :]
                self.cursor += 1
            return False
        if key == "ESC":
            if self.mode == "nominations":
                self.mode, self.offset = self.nomination_origin, 0
                return False
            if self.mode == "nomination":
                self.mode, self.offset = "comparison", 0
                return False
            if self.mode == "targets":
                self.mode, self.offset = "form", 0
                return False
            if self.mode == "list":
                return True
            self.mode, self.offset = "list", 0
            return False
        if (
            key == "h"
            and self.mode in {"list", "detail", "comparison", "form"}
            and not self.busy
        ):
            self._background("nominations", lambda: list_nominations(self.run))
            return False
        if key == "b" and self.mode in {"list", "detail", "comparison"}:
            self._compare()
            return False
        if key == "g" and self.mode == "list":
            if self.comparison_path:
                self.mode, self.offset, self._last_poll = "comparison", 0, 0
                self.refresh()
            else:
                self.message = "No retained comparison. Select a completed checkpoint validation, then b starts queued verification."
            return False
        if self.mode == "nominations":
            if key in {"UP", "DOWN", "PAGEUP", "PAGEDOWN", "HOME", "END"}:
                delta = 8 if key.startswith("PAGE") else 1
                self.nomination_index = max(
                    0,
                    min(
                        len(self.nominations) - 1,
                        (
                            0
                            if key == "HOME"
                            else (
                                len(self.nominations) - 1
                                if key == "END"
                                else self.nomination_index
                                + (delta if key in {"DOWN", "PAGEDOWN"} else -delta)
                            )
                        ),
                    ),
                )
            elif key == "r" and not self.busy:
                self._background("nominations", lambda: list_nominations(self.run))
            elif key == "ENTER" and self.nominations and not self.busy:
                selected = self.nominations[self.nomination_index]
                if selected["state"] != "available":
                    self.message = "This saved nomination cannot be restored. Inspect its receipt and selection evidence."
                else:
                    self._background(
                        "reopen-nomination",
                        lambda: nomination_config(
                            self.run,
                            selected["nomination_id"],
                            expected_plan_sha256=selected["plan_sha256"],
                        ),
                    )
            return False
        if self.mode == "nomination":
            if key == "ENTER" and self.nomination_review and not self.busy:
                review = self.nomination_review
                self._background(
                    "nominate",
                    lambda: prepare_nomination(
                        self.run,
                        review["comparison_id"],
                        expected_plan_sha256=content_hash(review),
                    ),
                    owned=True,
                )
            return False
        if self.mode == "comparison":
            rows = ((self.comparison or {}).get("result") or {}).get("rows", [])
            if key in {"UP", "DOWN", "PAGEUP", "PAGEDOWN", "HOME", "END"}:
                delta = 8 if key.startswith("PAGE") else 1
                self.comparison_index = max(
                    0,
                    min(
                        len(rows) - 1,
                        (
                            0
                            if key == "HOME"
                            else (
                                len(rows) - 1
                                if key == "END"
                                else self.comparison_index
                                + (delta if key in {"DOWN", "PAGEDOWN"} else -delta)
                            )
                        ),
                    ),
                )
            elif key == "c":
                self._use_recommendation()
            elif key == "n" and self.comparison_path and not self.busy:
                path = self.comparison_path
                self._background(
                    "nomination-review", lambda: nomination_plan(self.run, path.name)
                )
            elif key == "r":
                self._last_poll = 0
                self.refresh()
            elif key == "ENTER" and rows:
                try:
                    self.selected = evaluation_status(
                        self.run
                        / "evaluations"
                        / rows[self.comparison_index]["evaluation_id"]
                    )
                    self.result, self.mode, self.offset = None, "detail", 0
                except (OSError, ValueError, TypeError, KeyError):
                    self.message = "The original evaluation is unavailable; recheck retained records."
            return False
        if self.mode == "form":
            if key in {"UP", "DOWN", "TAB"}:
                self.field = (self.field + (1 if key in {"DOWN", "TAB"} else -1)) % len(
                    EVALUATION_FIELDS
                )
            elif key == "ENTER":
                self.edit = str(self.values[EVALUATION_FIELDS[self.field][0]])
                self.cursor, self.mode = len(self.edit), "edit"
            elif key == "c" and not self.busy:
                self._background("targets", lambda: evaluation_targets(self.run))
            elif key == "r" and not self.busy:
                try:
                    config = self._config()
                except EvaluationProtocolError as error:
                    self.message = str(error)
                except (ValueError, TypeError, OverflowError) as error:
                    del error
                    self.message = "Invalid evaluation settings; check paths, model pins and numeric limits."
                else:
                    self._background(
                        "review", lambda: plan_evaluation(self.run, config)
                    )
        elif self.mode == "targets":
            if key == "c" and not self.busy:
                self._background("targets", lambda: evaluation_targets(self.run))
            elif key in {"UP", "DOWN", "PAGEUP", "PAGEDOWN", "HOME", "END"}:
                delta = 8 if key in {"PAGEUP", "PAGEDOWN"} else 1
                self.target_index = max(
                    0,
                    min(
                        len(self.targets) - 1,
                        (
                            0
                            if key == "HOME"
                            else (
                                len(self.targets) - 1
                                if key == "END"
                                else self.target_index
                                + (delta if key in {"DOWN", "PAGEDOWN"} else -delta)
                            )
                        ),
                    ),
                )
            elif key == "ENTER" and self.targets:
                target = self.targets[self.target_index]
                if not target.get("available"):
                    self.message = "This receipt does not expose a usable adapter. Inspect the original checkpoint before selection."
                else:
                    self.values["checkpoint"] = target["checkpoint"]
                    self.values["nomination_id"] = ""
                    if target["checkpoint"]:
                        self.values["split"] = "validation"
                    self.review, self.prepared_path, self.result = None, None, None
                    self.mode, self.offset = "form", 0
                    self.message = "Evaluation target selected. Use reviewed cases for the displayed partition, then r captures a fresh review."
        elif self.mode == "review" and key == "ENTER":
            self._launch()
        elif self.mode == "list":
            if key == "n" and not self.busy:
                self.mode, self.offset = "form", 0
                self.message = "Edit fields, then r captures a read-only review."
            elif key in {"UP", "DOWN"}:
                self.index = max(
                    0,
                    min(len(self.rows) - 1, self.index + (1 if key == "DOWN" else -1)),
                )
            elif key == "ENTER" and self.rows:
                self.selected, self.result = self.rows[self.index], None
                self.mode, self.offset = "detail", 0
        elif self.mode == "detail" and self.selected:
            path = self.run / "evaluations" / self.selected["evaluation_id"]
            if key == "v" and not self.busy:
                self._background("verify", lambda: inspect_evaluation(path))
            elif (
                key == "r"
                and not self.busy
                and self.selected["state"].get("status") == "prepared"
            ):
                self._background("restore", lambda: (read_evaluation(path), path))
            elif (
                key == "s"
                and not self.busy
                and self.selected["state"].get("status")
                in {"prepared", "queued", "running", "interrupted"}
            ):
                self.mode = "stop"
            elif key == "l":
                from stateset_agents.finetune_workspace import log_tail

                # Reuse bounded, redacted private-log handling; no shell pager.
                self.message = "\n".join(log_tail(path, lines=12))
        elif self.mode == "stop" and key == "ENTER" and self.selected:
            path = self.run / "evaluations" / self.selected["evaluation_id"]
            checksum = self.selected["plan_sha256"]
            self._background(
                "stop",
                lambda: stop_evaluation(path, expected_plan_sha256=checksum),
                owned=True,
            )
        if (
            key in {"PAGEUP", "PAGEDOWN"}
            or self.mode in {"detail", "review"}
            and key in {"UP", "DOWN"}
        ):
            delta = 8 if key.startswith("PAGE") else 1
            self.offset = max(
                0, self.offset + (delta if key in {"DOWN", "PAGEDOWN"} else -delta)
            )
        return False

    def lines(self) -> list[str]:
        """Render private-data-free configuration, workload and retained outcomes."""
        rows = [
            "LOCAL PAIRED EVALUATION / base vs trained",
            "Run: " + self.run.name,
            "",
        ]
        if self.mode == "nominations":
            rows += [
                "SAVED CHECKPOINT NOMINATIONS / metadata only",
                "Enter restores a fresh-test form; no receipt or evaluation is created.",
                "Checkpoint bytes and fresh cases are checked during the later r review.",
                "",
            ]
            rows += [
                (" > " if index == self.nomination_index else "   ")
                + row["nomination_id"]
                + " | "
                + (
                    row["checkpoint"] + " | metadata ready"
                    if row["state"] == "available"
                    else "unavailable; inspect retained evidence"
                )
                for index, row in enumerate(self.nominations)
            ] or [
                "No saved nominations. In a completed comparison, n reviews a new nomination."
            ]
        elif self.mode == "nomination" and self.nomination_review:
            plan = self.nomination_review
            rows += [
                "REVIEW CHECKPOINT NOMINATION / separate fresh test",
                "Checkpoint: " + plan["checkpoint"],
                "Compared artifact SHA256: " + plan["artifact_sha256"],
                "Comparison: " + plan["comparison_id"],
                "Selection evaluation: " + plan["selection_evaluation_id"],
                "Nomination SHA256: " + content_hash(plan),
                "",
                "Enter retains this selection and opens a test form with no cases selected.",
                "No weight copy, inference, training or promotion occurs.",
                "The test review rehashes weights and rejects validation case overlap.",
                "Reviewer freshness remains an attestation; this does not certify quality.",
                "Esc returns without retaining a nomination.",
            ]
        elif self.mode == "comparison":
            status = self.comparison or {"state": {"status": "loading"}}
            state = status["state"]
            rows += [
                "CHECKPOINT VALIDATION COMPARISON",
                "State: " + str(state["status"]),
                "Comparison: " + str(status.get("comparison_id", "pending")),
                "Buildr job: " + str(status.get("job_id") or "awaiting receipt"),
                f"Verified records: {state.get('completed', 0)} / {state.get('total', '?')}",
            ]
            report = status.get("result")
            if report:
                base = report["baseline"]
                choice = report["recommendation"]
                rows += [
                    f"Base ({report['case_count']} cases): success {base['success_rate']:.1%} | judge {base['judge_score']:.4f}",
                    "Recommendation: " + (choice.get("checkpoint") or choice["kind"]),
                    "Recorded verification; another review checks current bytes.",
                    "",
                ]
                rows += [
                    (" > " if index == self.comparison_index else "   ")
                    + f"{row['checkpoint']} | success {row['success_rate']:.1%} | judge {row['judge_score']:.4f} | "
                    + ("eligible" if row["eligible"] else "not eligible")
                    for index, row in enumerate(report["rows"])
                ]
                rows += [
                    "",
                    f"Excluded records: {len(report['excluded'])}; complete: {report['comparison_complete']}",
                    *report["notes"],
                ]
            else:
                rows += [
                    "",
                    "Verification survives leaving this view or exiting the TUI.",
                    "Use the Buildr handle to inspect waits/logs. Uncertain progress does not trigger automatic retry.",
                ]
        elif self.mode in {"form", "edit"}:
            rows += [
                (" > " if index == self.field else "   ")
                + label
                + ": "
                + str(self.values[name])
                for index, (name, label) in enumerate(EVALUATION_FIELDS)
            ]
            if self.mode == "edit":
                rows += [
                    "",
                    "Edit: "
                    + self.edit[: self.cursor]
                    + "|"
                    + self.edit[self.cursor :],
                ]
            rows += [
                "",
                "Cache-only local models; pinned base and independent judge required.",
                "Cases: schema 1, the displayed partition, reviewer provenance and scripted user turns.",
                "No endpoint calls, downloads or paid resources are provisioned.",
            ]
        elif self.mode == "review" and self.review:
            plan = self.review.plan
            rows += [
                (
                    "REVIEW / Enter submits this retained evaluation"
                    if self.prepared_path
                    else "REVIEW / Enter prepares and submits this evaluation"
                ),
                "Evaluation SHA256: " + self.review.sha256,
                "Base: "
                + plan["binding"]["model"]
                + "@"
                + plan["binding"]["model_revision"],
                "Adapter SHA256: " + plan["binding"]["artifact_sha256"],
                "Judge: "
                + plan["config"]["judge_model"]
                + "@"
                + plan["config"]["judge_revision"],
                "Cases SHA256: " + plan["cases_sha256"],
                f"Training seed: {plan['binding'].get('training_seed') if plan['binding'].get('training_seed') is not None else 'not recorded'} | Evaluation seed: {plan['config']['seed']}",
                "Partition: "
                + plan["config"].get("split", "test")
                + " | target: "
                + (plan["config"].get("checkpoint") or "final adapter"),
                "Published receipt SHA256: "
                + str(
                    plan["binding"].get("checkpoint_receipt_sha256")
                    or "final completion"
                ),
                f"{plan['cases']} cases | {plan['reply_generations']} policy replies | at most {plan['max_judge_generations']} judge replies (2 passes + 1 retry)",
                "Queue CPU/RAM/scratch: "
                + " / ".join(
                    str(plan["config"][k])
                    for k in ("cpu_slots", "memory_gb", "scratch_gb")
                ),
                f"Timeout: {plan['config']['timeout_seconds']}s | context: {plan['config']['context_length']} tokens",
                "Policy/judge reply limits: "
                + str(plan["config"]["max_new_tokens"])
                + " / "
                + str(plan["config"]["judge_max_new_tokens"]),
                "Three separate processes free policy memory before loading the judge.",
                "Greedy scripted turns; overlong prompts fail, never silently truncate.",
                "Success needs >=0.8 and every resolution, accuracy, policy and non-repetition check in BOTH judge passes.",
                "Private transcripts and replayable judge recordings are retained.",
                "Existing training and quality.json are preserved. This diagnostic does not rank checkpoints automatically or approve deployment.",
                "Esc returns without submission. Authorized submission finishes before exit.",
            ]
        elif self.mode == "targets":
            rows += [
                "CHOOSE EVALUATION TARGET / receipt metadata",
                "Checkpoint bytes are verified during r review; checkpoint selection uses validation cases.",
                "",
            ]
            rows += [
                (" > " if index == self.target_index else "   ")
                + (target["checkpoint"] or "Final adapter")
                + " | "
                + ("available for review" if target.get("available") else "unavailable")
                for index, target in enumerate(self.targets)
            ] or ["No published safetensors adapter checkpoints are available."]
        elif self.mode in {"detail", "stop"} and self.selected:
            selected = self.selected
            rows += [
                "Evaluation: " + selected["evaluation_id"],
                "Plan SHA256: " + selected.get("plan_sha256", "unavailable"),
                "State: " + str(selected["state"].get("status")),
                "Phase: " + str(selected["state"].get("phase", "not started")),
                "Completed cases in phase: "
                + str(selected["state"].get("completed_cases", 0)),
                "Stop requested: " + str(selected.get("stop_requested", False)),
                "Partition: "
                + selected.get("split", "test")
                + " | target: "
                + (selected.get("checkpoint") or "final adapter"),
            ]
            if selected["state"].get("error"):
                rows += ["Error: " + selected["state"]["error"]]
            if selected["state"].get("execution_unconfirmed"):
                rows += [
                    "Supervisor lost; detached queue work may still be running. s requests a cooperative stop."
                ]
            queue = selected.get("queue", {})
            if queue:
                rows += [
                    "Buildr job: " + queue["job_id"] + " | runner " + queue["state"],
                    f"Sampled queue wait: {queue.get('wait_seconds', 0):.0f}s | sample age: {queue['sample_age_seconds']:.0f}s",
                ]
                if queue.get("waiting_for"):
                    rows += ["Waiting for: " + queue["waiting_for"]]
                    resources = queue.get("resources", {})
                    for label, free, required in (
                        ("Disk", "disk_free_gb", "required_free_gb"),
                        ("RAM", "available_memory_gb", "required_memory_gb"),
                    ):
                        if free in resources and required in resources:
                            rows += [
                                f"{label}: {resources[free]:.1f} / {resources[required]:.1f} GiB free / required"
                            ]
                rows += [
                    "Read-only queue inspection: buildr show --json " + queue["job_id"]
                ]
            elif selected["state"].get("status") in {"queued", "running"}:
                rows += [
                    "Waiting for a complete queue lifecycle event; open l for worker output."
                ]
            if self.mode == "stop":
                rows += [
                    "",
                    "Enter requests cooperative stop of this evaluation only.",
                    "Current generation and queue waits may finish; no training or other queue jobs are cancelled.",
                ]
            elif self.result:
                summary = self.result["summary"]
                scores = self.result["mean_judge_scores"]
                pair = summary["pairs"][0]
                rows += [
                    "",
                    (
                        "VERIFIED VALIDATION / fresh test assessment still required"
                        if self.result.get("split") == "validation"
                        else "VERIFIED DIAGNOSTIC / judge recordings replayed"
                    ),
                    f"Judge mean: base {scores['baseline']:.4f} -> trained {scores['candidate']:.4f}",
                    f"Success: {pair['baseline']['success_rate']:.1%} -> {pair['candidate']['success_rate']:.1%}",
                    f"Wins: {pair['wins']} | regressions: {pair['regressions']}",
                    (
                        "Checkpoint validation; fresh-test quality assessment remains separate."
                        if self.result.get("split") == "validation"
                        else f"Study gate: {'PASS' if summary['recorded_study_passed'] else 'NOT PASSED'} | 6 independent training seeds required"
                    ),
                    *self.result["notes"],
                ]
            else:
                rows += [
                    "",
                    "v rehashes the artifact and replays judge recordings after success.",
                    "paired-reports.json can be explicitly imported via e/f or the agent CLI.",
                ]
        else:
            rows += [
                (" > " if index == self.index else "   ")
                + row["evaluation_id"]
                + " | "
                + str(row["state"].get("status"))
                + " | "
                + str(row["state"].get("phase", ""))
                + " | "
                + (row.get("checkpoint") or "final")
                + " / "
                + row.get("split", "test")
                for index, row in enumerate(self.rows)
            ] or ["No retained evaluations. Press n to configure one."]
        return rows + [
            "",
            "Working..." if self.busy else "",
            *self.message.splitlines(),
        ]

    def keys(self) -> list[str]:
        """Return controls for the current screen, including narrow panes."""
        if self.mode == "edit":
            return ["Type | Arrows/Home/End | Enter save | Ctrl+U clear | Esc discard"]
        if self.mode == "form":
            return [
                "Arrows/Tab field | Enter edit | f cases | c checkpoints | h saved | r review | Esc"
            ]
        if self.mode == "targets":
            return ["Arrows/PgUp/Dn | Enter choose | c refresh | Esc form"]
        if self.mode == "comparison":
            return [
                "Arrows/Home/End | Enter details | c use recommendation | r status",
                "n nominate | h saved nominations | b new/reconnect verification | Esc",
            ]
        if self.mode == "nomination":
            return ["Enter retains nomination only | PgUp/Dn scroll | Esc comparison"]
        if self.mode == "nominations":
            return ["Arrows/PgUp/Dn/Home/End | Enter restore | r refresh | Esc back"]
        if self.mode in {"review", "stop"}:
            return ["Enter confirms reviewed action | PgUp/Dn scroll | Esc back"]
        if self.mode == "detail":
            return [
                "r review prepared | v verify | b compare validations | s stop | l logs",
                "PgUp/Dn scroll | Esc back",
            ]
        return [
            "n new evaluation | b compare | g reopen comparison | h saved nominations",
            "Up/Down select | Enter details | Esc back",
        ]
