"""Replayable validation ranking, safe recommendations and durable queue handoff."""

import copy
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as cli
from stateset_agents import finetune_evaluation as evaluation
from stateset_agents import finetune_validation as comparison
from stateset_agents import finetune_workspace as ws
from stateset_agents.evaluation.support_judge import ALL_CHECKS
from stateset_agents.finetune_evaluation_tui import EvaluationPanel
from stateset_agents.tui import Studio
from tests.unit import test_finetune_evaluation as evaluation_tests
from tests.unit.test_finetune_checkpoint_evaluation import published
from tests.unit.test_finetune_evaluation import wait


class Generator:
    """Synthetic policy/judge transport; real rubric and recording/replay remain."""

    def __init__(self, model, revision, *, adapter=None):
        self.judge = model == "fixture/judge"
        self.adapter = adapter

    def generate(self, messages, *, tokens, context):
        if self.judge:
            checks = dict.fromkeys(ALL_CHECKS, True)
            if "UNSUPPORTED_MARKER" in messages[-1]["content"]:
                checks["resolution_correct_outcome"] = False
            return json.dumps({"checks": checks}), 20, False
        case_one = "parcel 1" in messages[-1]["content"]
        bad = (
            case_one
            if self.adapter is None
            else not case_one if self.adapter.name == "checkpoint-2" else False
        )
        return "UNSUPPORTED_MARKER" if bad else "Offer a free replacement.", 8, False


def cohort(tmp_path, monkeypatch, *, checkpoints=(1, 2, 3)):
    run, _, config = published(tmp_path)
    original = run / "artifacts/checkpoint-1"
    for step in checkpoints:
        path = run / "artifacts" / f"checkpoint-{step}"
        if step != 1:
            shutil.copytree(original, path)
            ws.write_json(
                path / "trainer_state.json", {"global_step": step, "epoch": float(step)}
            )
            from stateset_agents.finetune_recovery import publish_checkpoint

            publish_checkpoint(run, path, step=step)
    monkeypatch.setattr(evaluation_tests, "Generator", Generator)
    paths = []
    for step in checkpoints:
        review = evaluation.plan_evaluation(
            run, replace(config, checkpoint=f"checkpoint-{step}")
        )
        path = evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256)
        evaluation_tests.run_pipeline(path, monkeypatch)
        paths.append(path)
    return run, config, paths


def queued(monkeypatch, *, execute=False, response=None):
    calls = []
    monkeypatch.setattr(comparison, "queue_executable", lambda: "/queue/buildr")

    def submit(argv, **kwargs):
        calls.append((argv, kwargs))
        if execute:
            path = Path(argv[-1])
            code = comparison.execute_comparison(path)
            assert code == 0, (
                (path / "worker-error.json").read_text()
                if (path / "worker-error.json").exists()
                else "No error receipt"
            )
        return response or SimpleNamespace(
            returncode=0,
            stdout="submitted abcdef123456 detached | log /private/queue.log\n",
            stderr="",
        )

    monkeypatch.setattr(comparison.subprocess, "run", submit)
    return calls


def test_rank_replays_same_protocol_and_prefers_earliest_safe_improvement(
    tmp_path, monkeypatch
):
    run, _, paths = cohort(tmp_path, monkeypatch)
    before = {
        path.relative_to(run): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file()
    }
    plan = comparison.comparison_plan(run, paths[-1].name)
    report = comparison.compare_validations(run, plan)
    assert report["comparison_complete"]
    assert report["baseline"]["success_rate"] == 0.5
    assert report["case_count"] == 2
    assert report["recommendation"] == {
        "kind": "checkpoint",
        "checkpoint": "checkpoint-1",
        "evaluation_id": paths[0].name,
    }
    assert [row["checkpoint"] for row in report["rows"]] == [
        "checkpoint-1",
        "checkpoint-3",
        "checkpoint-2",
    ]
    assert [row["eligible"] for row in report["rows"]] == [True, True, False]
    assert not report["rows"][-1]["guards"]["no_success_regressions"]
    assert not report["rows"][-1]["guards"]["no_new_critical_check_failures"]
    assert not report["quality_gain_verified"] and "PRIVATE" not in json.dumps(report)
    assert before == {
        path.relative_to(run): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file()
    }


def test_base_is_retained_when_high_judge_score_hides_new_failure(
    tmp_path, monkeypatch
):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(2,))
    report = comparison.compare_validations(
        run, comparison.comparison_plan(run, paths[0].name)
    )
    assert report["recommendation"] == {"kind": "base", "checkpoint": ""}
    assert report["rows"][0]["judge_score"] > 0.8
    assert not report["rows"][0]["eligible"]


@pytest.mark.parametrize("mutation", ["weights", "recording", "result", "plan"])
def test_tampered_same_cohort_record_blocks_recommendation(
    tmp_path, monkeypatch, mutation
):
    run, _, paths = cohort(tmp_path, monkeypatch)
    plan = comparison.comparison_plan(run, paths[0].name)
    if mutation == "weights":
        (run / "artifacts/checkpoint-2/adapter_model.safetensors").write_bytes(
            b"changed"
        )
    elif mutation == "recording":
        (paths[1] / "judge-recordings/batch-000001/manifest.json").write_text("{}")
    elif mutation == "result":
        ws.write_json(paths[1] / "result.json", {"PRIVATE": True})
    else:
        changed = ws.read_json(paths[1] / "plan.json")
        changed["cases_sha256"] = "0" * 64
        ws.write_json(paths[1] / "plan.json", changed)
    report = comparison.compare_validations(run, plan)
    assert (
        not report["comparison_complete"] and report["recommendation"]["kind"] == "none"
    )
    assert {"evaluation_id": paths[1].name, "reason": "unverifiable"} in report[
        "excluded"
    ]
    assert "PRIVATE" not in json.dumps(report)


def test_different_protocol_is_explicitly_excluded_without_model_byte_reads(
    tmp_path, monkeypatch
):
    run, _, paths = cohort(tmp_path, monkeypatch)
    changed = ws.read_json(paths[1] / "plan.json")
    changed["config"]["seed"] += 1
    ws.write_json(paths[1] / "plan.json", changed)
    inspect = comparison.inspect_evaluation
    calls = []

    def verify(path):
        calls.append(path)
        assert path != paths[1]
        return inspect(path)

    monkeypatch.setattr(comparison, "inspect_evaluation", verify)
    report = comparison.compare_validations(
        run, comparison.comparison_plan(run, paths[0].name)
    )
    assert report["comparison_complete"] and len(calls) == 2
    assert report["excluded"] == [
        {"evaluation_id": paths[1].name, "reason": "different_protocol"}
    ]


def test_duplicate_evaluation_is_one_checkpoint_not_independent_evidence(
    tmp_path, monkeypatch
):
    run, config, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    review = evaluation.plan_evaluation(run, config)
    duplicate = evaluation.prepare_evaluation(
        review, expected_plan_sha256=review.sha256
    )
    evaluation_tests.run_pipeline(duplicate, monkeypatch)
    report = comparison.compare_validations(
        run, comparison.comparison_plan(run, paths[0].name)
    )
    assert len(report["rows"]) == 1 and report["rows"][0]["replicates"] == 2


def test_conflicting_replicate_cannot_silently_cherry_pick_higher_score(
    tmp_path, monkeypatch
):
    run, config, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    generate = Generator.generate

    def altered(self, messages, **kwargs):
        if not self.judge and self.adapter is not None:
            return "UNSUPPORTED_MARKER", 8, False
        return generate(self, messages, **kwargs)

    monkeypatch.setattr(Generator, "generate", altered)
    review = evaluation.plan_evaluation(run, config)
    duplicate = evaluation.prepare_evaluation(
        review, expected_plan_sha256=review.sha256
    )
    evaluation_tests.run_pipeline(duplicate, monkeypatch)
    report = comparison.compare_validations(
        run, comparison.comparison_plan(run, paths[0].name)
    )
    assert (
        not report["comparison_complete"] and report["recommendation"]["kind"] == "none"
    )
    assert any(row["reason"] == "conflicting_replicate" for row in report["excluded"])


def test_submission_queues_once_and_worker_can_finish_before_acknowledgement(
    tmp_path, monkeypatch
):
    run, _, paths = cohort(tmp_path, monkeypatch)
    parent = (run / "state.json").read_bytes()
    calls = queued(monkeypatch, execute=True)
    request = comparison.submit_comparison(run, paths[0].name)
    status = comparison.comparison_status(request)
    assert (
        status["state"]["status"] == "succeeded" and status["job_id"] == "abcdef123456"
    )
    assert status["result"]["recommendation"]["checkpoint"] == "checkpoint-1"
    assert calls[0][0][:5] == ["/queue/buildr", "run", "--detach", "--backend", "local"]
    assert calls[0][0][-3:] == [
        "stateset_agents.finetune_validation",
        "--execute",
        str(request),
    ]
    assert request.stat().st_mode & 0o077 == 0
    assert all(
        path.stat().st_mode & 0o077 == 0 for path in request.iterdir() if path.is_file()
    )
    assert (
        parent == (run / "state.json").read_bytes() and not (run / "stop.json").exists()
    )
    assert comparison.execute_comparison(request) == 1


@pytest.mark.parametrize("uncertain", [False, True])
def test_pending_or_uncertain_request_reconnects_without_duplicate_submission(
    tmp_path, monkeypatch, uncertain
):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    response = (
        SimpleNamespace(
            returncode=0,
            stdout="unexpected queue response",
            stderr="PRIVATE_QUEUE_VALUE",
        )
        if uncertain
        else None
    )
    calls = queued(monkeypatch, response=response)
    first = comparison.submit_comparison(run, paths[0].name)
    second = comparison.submit_comparison(run, paths[0].name)
    assert first == second and len(calls) == 1
    state = comparison.comparison_status(first)
    assert state["state"]["status"] == (
        "submission_uncertain" if uncertain else "queued"
    )
    assert "PRIVATE" not in json.dumps(state)


def test_status_and_reopening_only_read_metadata(tmp_path, monkeypatch):
    run, _, paths = cohort(tmp_path, monkeypatch)
    queued(monkeypatch, execute=True)
    request = comparison.submit_comparison(run, paths[0].name)

    def forbidden(*args, **kwargs):
        pytest.fail("Reopening must not replay judges, hash model bytes or resubmit")

    monkeypatch.setattr(comparison, "compare_validations", forbidden)
    monkeypatch.setattr(comparison, "inspect_evaluation", forbidden)
    monkeypatch.setattr(comparison.subprocess, "run", forbidden)
    assert comparison.latest_comparison(run) == request
    panel = EvaluationPanel(run)
    panel.handle("g")
    assert (
        panel.mode == "comparison"
        and panel.comparison["state"]["status"] == "succeeded"
    )
    assert "checkpoint-1" in "\n".join(panel.lines())


@pytest.mark.parametrize("size", [(72, 22), (74, 44), (110, 32)])
def test_tui_compares_and_uses_recommendation_without_changing_training_draft(
    tmp_path, monkeypatch, size
):
    run, _, paths = cohort(tmp_path, monkeypatch)
    queued(monkeypatch, execute=True)
    studio = Studio(run.parent)
    studio.handle("s")
    draft = (run.parent / "draft.json").read_bytes()
    values = copy.deepcopy(studio.values)
    panel = EvaluationPanel(run)
    studio.evaluation_panel, studio.overlay = panel, "evaluation"
    panel.index = next(
        i for i, row in enumerate(panel.rows) if row["evaluation_id"] == paths[-1].name
    )
    panel.handle("b")
    wait(panel)
    assert (
        panel.mode == "comparison" and panel.comparison["result"]["comparison_complete"]
    )
    screen = "\n".join(studio.lines(*size))
    assert "CHECKPOINT VALIDATION COMPARISON" in screen
    panel.handle("END")
    assert "checkpoint-2" in "\n".join(studio.lines(*size))
    panel.handle("c")
    assert panel.mode == "form" and panel.values["checkpoint"] == "checkpoint-1"
    assert panel.values["split"] == "validation" and panel.review is None
    assert Path(panel.values["cases"]).name == "cases.json"
    assert studio.values == values and (run.parent / "draft.json").read_bytes() == draft
    assert not (run / "stop.json").exists()


def test_agent_cli_submits_and_reconnects_without_private_values(tmp_path, monkeypatch):
    run, _, paths = cohort(tmp_path, monkeypatch)
    calls = queued(monkeypatch, execute=True)
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["validation-compare", run.name, paths[0].name, "--workspace", str(run.parent)],
    )
    assert result.exit_code == 0, result.stdout
    status = json.loads(result.stdout)["result"]
    result = runner.invoke(
        cli.app,
        [
            "validation-comparison-status",
            run.name,
            status["comparison_id"],
            "--workspace",
            str(run.parent),
        ],
    )
    assert result.exit_code == 0 and len(calls) == 1 and "PRIVATE" not in result.stdout


@pytest.mark.parametrize("anchor", ["../private", "checkpoint-1", "x" * 100])
def test_invalid_anchor_fails_before_queue_or_snapshot_write(
    tmp_path, monkeypatch, anchor
):
    run, _, _ = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    calls = queued(monkeypatch)
    with pytest.raises(ValueError):
        comparison.submit_comparison(run, anchor)
    assert not calls and not (run / comparison.DIRECTORY).exists()


def test_oversize_history_fails_closed_but_explicit_subset_can_be_compared(
    tmp_path, monkeypatch
):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    for index in range(101):
        (run / "evaluations" / f"ignored-{index}").mkdir()
    with pytest.raises(ValueError, match="100"):
        comparison.comparison_plan(run, paths[0].name)
    plan = comparison.comparison_plan(run, paths[0].name, [paths[0].name])
    assert len(plan["evaluations"]) == 1


@pytest.mark.parametrize("extra", ["../private", "PRIVATE_VALUE", "", "checkpoint-1"])
def test_bad_explicit_subset_fails_before_queue_or_request_creation(
    tmp_path, monkeypatch, extra
):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    calls = queued(monkeypatch)
    result = CliRunner().invoke(
        cli.app,
        [
            "validation-compare",
            run.name,
            paths[0].name,
            "--workspace",
            str(run.parent),
            "--evaluation-id",
            extra,
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"]["code"] == "invalid_validation_comparison"
    assert not calls and not (run / comparison.DIRECTORY).exists()
    assert "PRIVATE" not in result.stdout


def test_changed_implementation_refuses_execution_and_retains_safe_diagnostics(
    tmp_path, monkeypatch
):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    queued(monkeypatch)
    request = comparison.submit_comparison(run, paths[0].name)
    plan = ws.read_json(request / "plan.json")
    plan["implementation_sha256"] = "0" * 64
    ws.write_json(request / "plan.json", plan)
    assert comparison.execute_comparison(request) == 1
    assert comparison.comparison_status(request)["state"]["status"] == "failed"
    assert "PRIVATE" not in (request / "worker-error.json").read_text()


def test_running_worker_disappearance_is_not_a_completed_comparison(
    tmp_path, monkeypatch
):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    queued(monkeypatch)
    request = comparison.submit_comparison(run, paths[0].name)
    ws.write_json(request / "state.json", {"status": "running"})
    ws.write_json(
        request / "execution.json", {"pid": os.getpid(), "process_token": "old-token"}
    )
    assert comparison.comparison_status(request)["state"]["status"] == "interrupted"


def test_queue_timeout_does_not_retry_or_turn_into_training(tmp_path, monkeypatch):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    monkeypatch.setattr(comparison, "queue_executable", lambda: "/queue/buildr")
    calls = []

    def timeout(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired("queue", 20)

    monkeypatch.setattr(comparison.subprocess, "run", timeout)
    path = comparison.submit_comparison(run, paths[0].name)
    assert (
        comparison.comparison_status(path)["state"]["status"] == "submission_uncertain"
    )
    assert comparison.submit_comparison(run, paths[0].name) == path and len(calls) == 1


def test_changed_baseline_prevents_comparing_nominally_identical_protocols(
    tmp_path, monkeypatch
):
    run, config, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    generate = Generator.generate

    def altered(self, messages, **kwargs):
        if not self.judge and self.adapter is None:
            return "Offer a free replacement.", 8, False
        return generate(self, messages, **kwargs)

    monkeypatch.setattr(Generator, "generate", altered)
    review = evaluation.plan_evaluation(run, config)
    changed = evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256)
    evaluation_tests.run_pipeline(changed, monkeypatch)
    report = comparison.compare_validations(
        run, comparison.comparison_plan(run, paths[0].name)
    )
    assert (
        report["recommendation"]["kind"] == "none" and not report["comparison_complete"]
    )
    assert report["excluded"] == [
        {"evaluation_id": changed.name, "reason": "different_baseline"}
    ]


@pytest.mark.parametrize(
    "field,value", [("anchor_id", "PRIVATE"), ("anchor_id", {}), ("evaluations", [])]
)
def test_malformed_comparison_metadata_is_unreadable_without_private_values(
    tmp_path, monkeypatch, field, value
):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    queued(monkeypatch)
    request = comparison.submit_comparison(run, paths[0].name)
    plan = ws.read_json(request / "plan.json")
    plan[field] = value
    ws.write_json(request / "plan.json", plan)
    panel = EvaluationPanel(run)
    panel.handle("g")
    assert panel.comparison["state"]["status"] == "unreadable"
    assert "PRIVATE" not in "\n".join(panel.lines())


def test_public_status_does_not_export_unrecognized_private_result_fields(
    tmp_path, monkeypatch
):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    queued(monkeypatch, execute=True)
    request = comparison.submit_comparison(run, paths[0].name)
    report = ws.read_json(request / "result.json")
    report["private"] = report["rows"][0]["private"] = "PRIVATE_VALUE"
    report["notes"] = ["PRIVATE_VALUE"]
    report["quality_gain_verified"] = True
    ws.write_json(request / "result.json", report)
    state = ws.read_json(request / "state.json")
    state["result_sha256"] = comparison.content_hash(report)
    ws.write_json(request / "state.json", state)
    status = comparison.comparison_status(request)
    assert (
        "PRIVATE" not in json.dumps(status)
        and not status["result"]["quality_gain_verified"]
    )


@pytest.mark.parametrize("bad", ["row", "metric", "recommendation"])
def test_corrupt_result_shape_stays_controlled_when_receipt_is_rewritten(
    tmp_path, monkeypatch, bad
):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    queued(monkeypatch, execute=True)
    request = comparison.submit_comparison(run, paths[0].name)
    report = ws.read_json(request / "result.json")
    if bad == "row":
        report["rows"] = ["PRIVATE"]
    elif bad == "metric":
        report["baseline"]["judge_score"] = 10**400
    else:
        report["recommendation"] = {"kind": "base", "checkpoint": "PRIVATE"}
    ws.write_json(request / "result.json", report)
    state = ws.read_json(request / "state.json")
    state["result_sha256"] = comparison.content_hash(report)
    ws.write_json(request / "state.json", state)
    with pytest.raises(ValueError) as error:
        comparison.comparison_status(request)
    assert "PRIVATE" not in str(error.value)


def test_schema_discovery_remains_model_free():
    source = """import sys
from stateset_agents.finetune_workspace import finetune_schema
schema=finetune_schema()
assert schema['checkpoint_validation_comparison']['queue_required']
assert not any(key in sys.modules for key in ['torch','transformers','peft'])
"""
    subprocess.run([sys.executable, "-c", source], check=True)


def test_changed_anchor_cannot_replace_the_compared_protocol(tmp_path, monkeypatch):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    queued(monkeypatch, execute=True)
    comparison.submit_comparison(run, paths[0].name)
    panel = EvaluationPanel(run)
    panel.handle("g")
    before = copy.deepcopy(panel.values)
    plan = ws.read_json(paths[0] / "plan.json")
    plan["config"]["judge_model"] = "changed/judge"
    ws.write_json(paths[0] / "plan.json", plan)
    panel.handle("c")
    assert panel.values == before
    assert panel.mode == "comparison"
    assert "Cannot restore" in panel.message


def test_base_recommendation_never_selects_the_final_adapter(tmp_path, monkeypatch):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(2,))
    queued(monkeypatch, execute=True)
    comparison.submit_comparison(run, paths[0].name)
    panel = EvaluationPanel(run)
    before = copy.deepcopy(panel.values)
    panel.handle("g")
    panel.handle("c")
    assert panel.mode == "comparison" and panel.values == before
    assert "No adapter was selected" in panel.message and panel.review is None


def test_missing_history_has_actionable_reopen_guidance(tmp_path):
    panel = EvaluationPanel(tmp_path / "no-run")
    panel.handle("g")
    assert "No retained comparison" in panel.message and panel.mode == "list"
    assert not (tmp_path / "no-run").exists()


def test_late_submission_cannot_overwrite_a_different_evaluation_editor(
    tmp_path, monkeypatch
):
    started, release = threading.Event(), threading.Event()
    panel = EvaluationPanel(tmp_path / "no-run")
    panel.rows = [
        {
            "evaluation_id": "20261017-000000-aaaaaaaa",
            "state": {"status": "succeeded"},
            "checkpoint": "checkpoint-1",
            "split": "validation",
        }
    ]

    def submit(*args, **kwargs):
        started.set()
        release.wait(timeout=10)
        return tmp_path / "comparison"

    monkeypatch.setattr(
        "stateset_agents.finetune_evaluation_tui.submit_comparison", submit
    )
    panel.handle("b")
    try:
        assert started.wait(timeout=5)
        panel.mode = "form"
        panel.values["judge_model"] = "user/changed-model"
    finally:
        release.set()
        wait(panel)
    assert panel.mode == "form" and panel.values["judge_model"] == "user/changed-model"


def test_timeout_preserves_a_partial_durable_handle_without_echoing_diagnostics(
    tmp_path, monkeypatch
):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    monkeypatch.setattr(comparison, "queue_executable", lambda: "/queue/buildr")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            "queue",
            20,
            output=b"submitted abcdef123456 detached | log /private/log\n",
            stderr=b"PRIVATE_DIAGNOSTIC",
        )

    monkeypatch.setattr(comparison.subprocess, "run", timeout)
    request = comparison.submit_comparison(run, paths[0].name)
    status = comparison.comparison_status(request)
    assert (
        status["state"]["status"] == "submission_uncertain"
        and status["job_id"] == "abcdef123456"
    )
    assert "PRIVATE" not in json.dumps(status)
    assert "PRIVATE_DIAGNOSTIC" in (request / "submission-diagnostic.json").read_text()


@pytest.mark.skipif(
    sys.platform != "linux", reason="Curses comparison uses a Linux PTY"
)
@pytest.mark.parametrize("size", [(74, 44), (110, 32)])
def test_real_terminal_reconnects_uses_winner_and_reviews_without_submission(
    tmp_path, monkeypatch, size
):
    import fcntl
    import pty
    import select
    import struct
    import termios
    import time

    run, _, paths = cohort(tmp_path, monkeypatch)
    calls = queued(monkeypatch, execute=True)
    request = comparison.submit_comparison(run, paths[0].name)
    studio = Studio(run.parent)
    studio.handle("s")
    draft = (run.parent / "draft.json").read_bytes()
    evaluations = len(list((run / "evaluations").glob("*/plan.json")))
    terminal, child = pty.openpty()
    fcntl.ioctl(
        terminal, termios.TIOCSWINSZ, struct.pack("HHHH", size[1], size[0], 0, 0)
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "stateset_agents.cli",
            "tui",
            "--workspace",
            str(run.parent),
        ],
        stdin=child,
        stdout=child,
        stderr=child,
        env=dict(
            os.environ, TERM="xterm-256color", COLUMNS=str(size[0]), LINES=str(size[1])
        ),
    )
    os.close(child)
    captured, full = bytearray(), bytearray()

    def wait_for(marker):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if marker.encode() in captured:
                return
            ready, _, _ = select.select([terminal], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(terminal, 65536)
                except OSError:
                    break
                captured.extend(chunk)
                full.extend(chunk)
        pytest.fail(f"Missing terminal marker {marker!r}: {bytes(captured)[-1500:]!r}")

    def send(keys, marker):
        captured.clear()
        os.write(terminal, keys)
        wait_for(marker)

    try:
        wait_for("FINE-TUNING STUDIO")
        send(b"\t", "RUNS 1/1")
        send(b"j", "LOCAL PAIRED EVALUATION")
        send(b"g", "CHECKPOINT VALIDATION COMPARISON")
        send(b"c", "Recommended checkpoint selected")
        send(b"r", "Evaluation SHA256")
        send(b"\x1b", "n new evaluation")
        send(b"\x1b", "j evaluate")
        os.write(terminal, b"q")
        assert process.wait(timeout=15) == 0
        assert (run.parent / "draft.json").read_bytes() == draft
        assert len(list((run / "evaluations").glob("*/plan.json"))) == evaluations
        assert comparison.latest_comparison(run) == request and len(calls) == 1
        assert b"PRIVATE" not in full
    finally:
        os.close(terminal)
        if process.poll() is None:
            process.kill()
            process.wait()


def test_queue_submission_is_serialized_for_concurrent_callers(tmp_path, monkeypatch):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    calls = queued(monkeypatch)
    requests, errors = [], []

    def submit():
        try:
            requests.append(comparison.submit_comparison(run, paths[0].name))
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert (
        not errors
        and len(requests) == 2
        and requests[0] == requests[1]
        and len(calls) == 1
    )


def test_verification_state_changes_go_through_the_locked_transition(tmp_path):
    """A recorded outcome is final: late progress or failure writes cannot
    reopen a verification whose result was retained."""
    from stateset_agents import finetune_validation as validation
    from stateset_agents.finetune_workspace import read_json, write_json

    record = tmp_path / "record"
    record.mkdir()
    write_json(record / "state.json", {"status": "queued"})

    assert validation._state(record, status="running", phase="verifying")
    write_json(record / "result.json", {"ok": True})
    assert validation._state(record, status="succeeded", phase="complete")

    assert not validation._state(record, status="failed", error="late failure")
    assert not validation._state(record, status="running", completed=1, total=2)
    state = read_json(record / "state.json")
    assert state["status"] == "succeeded" and "error" not in state


def test_worker_that_cannot_start_fails_the_request_instead_of_queuing_forever(
    tmp_path, monkeypatch
):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    calls = queued(monkeypatch)
    request = comparison.submit_comparison(run, paths[0].name)
    (run / "plan.json").write_text("{not json")  # e.g. swapped at admission
    assert comparison.execute_comparison(request) == 1
    assert ws.read_json(request / "state.json")["status"] == "failed"
    assert not (request / "execution.json").exists()
    (run / "plan.json").unlink()
    assert len(calls) == 1


def test_a_second_worker_never_changes_the_owners_state(tmp_path, monkeypatch):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    queued(monkeypatch)
    request = comparison.submit_comparison(run, paths[0].name)
    ws.write_json(
        request / "execution.json", {"pid": os.getpid(), "process_token": "x"}
    )
    before = (request / "state.json").read_bytes()
    assert comparison.execute_comparison(request) == 1
    assert (request / "state.json").read_bytes() == before


def test_stale_queued_request_no_longer_blocks_its_anchor(tmp_path, monkeypatch):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    calls = queued(monkeypatch)
    first = comparison.submit_comparison(run, paths[0].name)
    state = ws.read_json(first / "state.json")
    ws.write_json(
        first / "state.json",
        {
            **state,
            "created_at": state["created_at"]
            - comparison.QUEUED_STALE_AFTER_SECONDS
            - 1,
        },
    )
    assert comparison.comparison_status(first)["state"]["status"] == "stale"
    second = comparison.submit_comparison(run, paths[0].name)
    assert second != first and len(calls) == 2


def test_claimed_request_whose_worker_died_is_interrupted(tmp_path, monkeypatch):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    queued(monkeypatch)
    request = comparison.submit_comparison(run, paths[0].name)
    ws.write_json(request / "execution.json", {"pid": 4242, "process_token": "dead"})
    monkeypatch.setattr(ws, "process_token", lambda _pid: None)
    monkeypatch.setattr(
        ws, "probe_process", lambda _pid: (_ for _ in ()).throw(ProcessLookupError())
    )
    assert comparison.comparison_status(request)["state"]["status"] == "interrupted"


def test_resubmit_queues_a_fresh_request_after_an_uncertain_submission(
    tmp_path, monkeypatch
):
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    calls = queued(
        monkeypatch,
        response=SimpleNamespace(returncode=1, stdout="", stderr="queue down"),
    )
    first = comparison.submit_comparison(run, paths[0].name)
    assert comparison.comparison_status(first)["state"]["status"] == (
        "submission_uncertain"
    )
    assert comparison.submit_comparison(run, paths[0].name) == first
    second = comparison.submit_comparison(run, paths[0].name, resubmit=True)
    assert second != first and len(calls) == 2
    result = CliRunner().invoke(
        cli.app,
        [
            "validation-compare",
            run.name,
            paths[0].name,
            "--workspace",
            str(run.parent),
            "--resubmit",
        ],
    )
    assert json.loads(result.stdout)["ok"] and len(calls) == 3
