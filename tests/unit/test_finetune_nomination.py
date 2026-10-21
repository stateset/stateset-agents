"""A nominated intermediate gets a separate test review, never final-weight fallback."""

import json
import os
import stat
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as cli
from stateset_agents import finetune_evaluation as evaluation
from stateset_agents import finetune_nomination as nomination
from stateset_agents import finetune_validation as comparison
from stateset_agents import finetune_workspace as ws
from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.finetune_evaluation_tui import EvaluationPanel
from stateset_agents.tui import Studio
from tests.unit import test_finetune_evaluation as evaluation_tests
from tests.unit import test_finetune_validation_comparison as comparison_tests
from tests.unit.test_finetune_validation_comparison import cohort, queued


def candidate(tmp_path, monkeypatch, *, checkpoints=(1,)):
    run, config, paths = cohort(tmp_path, monkeypatch, checkpoints=checkpoints)
    queued(monkeypatch, execute=True)
    compared = comparison.submit_comparison(run, paths[-1].name)
    plan = nomination.nomination_plan(run, compared.name)
    receipt = nomination.prepare_nomination(
        run, compared.name, expected_plan_sha256=content_hash(plan)
    )
    document = ws.read_json(paths[0] / "cases.json")
    document["split"] = "test"
    document["provenance"] = {
        "source": "reviewed",
        "reviewer": "synthetic fixture",
        "test_is_fresh": True,
    }
    for case in document["cases"]:
        case["id"] = "fresh-" + case["id"]
        case["user_turns"] = ["Fresh test: " + turn for turn in case["user_turns"]]
    fresh = tmp_path / "fresh.json"
    ws.write_json(fresh, document)
    config = replace(
        config,
        cases=str(fresh),
        checkpoint=receipt["checkpoint"],
        split="test",
        nomination_id=receipt["nomination_id"],
    )
    return run, config, paths, compared, receipt


def test_nominated_earlier_checkpoint_runs_actual_record_replay_on_fresh_test(
    tmp_path, monkeypatch
):
    original = comparison_tests.published

    def completed(root):
        run, training, config = original(root, state="succeeded")
        ws.write_json(
            run / "completion.json", {"output_dir": str(run / "artifacts/checkpoint-3")}
        )
        return run, training, config

    monkeypatch.setattr(comparison_tests, "published", completed)
    run, config, _, _, receipt = candidate(tmp_path, monkeypatch, checkpoints=(1, 2, 3))
    before = {
        name: (run / name).read_bytes()
        for name in ("plan.json", "state.json", "completion.json")
    }
    review = evaluation.plan_evaluation(run, config)
    assert review.plan["binding"]["checkpoint"] == "checkpoint-1"
    assert (
        review.plan["binding"]["nomination"]["nomination_id"]
        == receipt["nomination_id"]
    )
    path = evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256)
    evaluation_tests.run_pipeline(path, monkeypatch)
    summary = evaluation.inspect_evaluation(path)
    assert summary["split"] == "test" and summary["checkpoint"] == "checkpoint-1"
    assert not summary["quality_gain_verified"]
    assert ws.read_json(run / "completion.json")["output_dir"].endswith("checkpoint-3")
    assert {name: (run / name).read_bytes() for name in before} == before
    assert not (run / "quality.json").exists()
    assert ws.read_json(path / "paired-reports.json")["candidates"][0][
        "selected_checkpoint"
    ]["path"].endswith("checkpoint-1")


@pytest.mark.parametrize("kind", ["id", "turn", "normalized_turn"])
def test_validation_overlap_is_rejected_even_when_renamed(tmp_path, monkeypatch, kind):
    run, config, paths, _, _ = candidate(tmp_path, monkeypatch)
    document = ws.read_json(Path(config.cases))
    old = ws.read_json(paths[0] / "cases.json")["cases"][0]
    if kind == "id":
        document["cases"][0]["id"] = old["id"]
    else:
        turn = old["user_turns"][0]
        document["cases"][0]["user_turns"][0] = (
            "  " + turn.upper() + "  " if kind == "normalized_turn" else turn
        )
    ws.write_json(Path(config.cases), document)
    with pytest.raises(ValueError, match="overlap retained validation"):
        evaluation.plan_evaluation(run, config)


@pytest.mark.parametrize(
    "change",
    ["receipt", "selected_plan", "comparison_result", "weights", "validation_cases"],
)
def test_nomination_cannot_authorize_changed_evidence_or_weights(
    tmp_path, monkeypatch, change
):
    run, config, paths, compared, receipt = candidate(tmp_path, monkeypatch)
    if change == "receipt":
        path = run / "nominations" / receipt["nomination_id"] / "receipt.json"
        value = ws.read_json(path)
        value["plan"]["checkpoint"] = "checkpoint-9"
        ws.write_json(path, value)
    elif change == "selected_plan":
        path = paths[0] / "plan.json"
        value = ws.read_json(path)
        value["config"]["seed"] += 1
        ws.write_json(path, value)
    elif change == "comparison_result":
        path = compared / "result.json"
        value = ws.read_json(path)
        value["verified_at"] += 1
        ws.write_json(path, value)
    elif change == "weights":
        (run / "artifacts/checkpoint-1/adapter_model.safetensors").write_bytes(
            b"changed"
        )
    else:
        path = paths[0] / "cases.json"
        value = ws.read_json(path)
        value["cases"][0]["user_turns"][0] = "changed validation text"
        ws.write_json(path, value)
    with pytest.raises(ValueError):
        evaluation.plan_evaluation(run, config)
    assert len(list((run / "evaluations").iterdir())) == 1


def test_receipt_is_private_and_metadata_only_but_review_rehashes(
    tmp_path, monkeypatch
):
    run, config, _, compared, _ = candidate(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("Nomination must not hash checkpoint bytes")

    monkeypatch.setattr(evaluation, "_inventory", forbidden)
    monkeypatch.setattr(evaluation, "inspect_evaluation_checkpoint", forbidden)
    plan = nomination.nomination_plan(run, compared.name)
    receipt = nomination.prepare_nomination(
        run, compared.name, expected_plan_sha256=content_hash(plan)
    )
    path = run / "nominations" / receipt["nomination_id"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert stat.S_IMODE((path / "receipt.json").stat().st_mode) == 0o600
    with pytest.raises(AssertionError, match="must not hash"):
        evaluation.plan_evaluation(run, config)


def test_stale_nomination_review_creates_no_new_receipt(tmp_path, monkeypatch):
    run, _, paths, compared, _ = candidate(tmp_path, monkeypatch)
    plan = nomination.nomination_plan(run, compared.name)
    before = set((run / "nominations").iterdir())
    path = paths[0] / "plan.json"
    changed = ws.read_json(path)
    changed["config"]["judge_model"] = "changed/judge"
    ws.write_json(path, changed)
    with pytest.raises(ValueError):
        nomination.prepare_nomination(
            run, compared.name, expected_plan_sha256=content_hash(plan)
        )
    assert set((run / "nominations").iterdir()) == before


@pytest.mark.parametrize(
    "change", ["no_nomination", "wrong_checkpoint", "validation", "final", "path_id"]
)
def test_test_target_requires_its_own_well_formed_nomination(
    tmp_path, monkeypatch, change
):
    run, config, _, _, _ = candidate(tmp_path, monkeypatch)
    values = {
        "no_nomination": {"nomination_id": ""},
        "wrong_checkpoint": {"checkpoint": "checkpoint-2"},
        "validation": {"split": "validation"},
        "final": {"checkpoint": ""},
        "path_id": {"nomination_id": "../receipt"},
    }[change]
    with pytest.raises(ValueError):
        evaluation.plan_evaluation(run, replace(config, **values))


def test_form_nomination_review_requires_enter_and_clears_validation_cases(
    tmp_path, monkeypatch
):
    run, _, _, _, _ = candidate(tmp_path, monkeypatch)
    count = len(list((run / "nominations").iterdir()))
    panel = EvaluationPanel(run)
    panel.handle("g")
    panel.handle("n")
    evaluation_tests.wait(panel)
    assert panel.mode == "nomination" and "REVIEW CHECKPOINT NOMINATION" in "\n".join(
        panel.lines()
    )
    assert len(list((run / "nominations").iterdir())) == count
    panel.handle("ESC")
    assert panel.mode == "comparison"
    panel.handle("n")
    evaluation_tests.wait(panel)
    panel.handle("ENTER")
    evaluation_tests.wait(panel)
    assert panel.mode == "form" and panel.values["cases"] == ""
    assert (
        panel.values["split"] == "test" and panel.values["checkpoint"] == "checkpoint-1"
    )
    assert panel.values["nomination_id"]
    assert panel.review is None and panel.prepared_path is None
    assert len(list((run / "nominations").iterdir())) == count + 1
    panel.handle("c")
    evaluation_tests.wait(panel)
    panel.handle("ENTER")
    assert panel.values["nomination_id"] == ""


def test_agent_protocol_reviews_then_retains_without_inference(tmp_path, monkeypatch):
    run, config, _, compared, _ = candidate(tmp_path, monkeypatch)
    runner = CliRunner()
    common = ["--workspace", str(run.parent)]
    result = runner.invoke(
        cli.app, ["nomination-plan", run.name, compared.name, *common]
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)["result"]
    assert not plan["prepared"] and "PRIVATE" not in result.output
    rejected = runner.invoke(
        cli.app,
        [
            "nomination-prepare",
            run.name,
            compared.name,
            "--expected-plan-sha256",
            "0" * 64,
            *common,
        ],
    )
    assert rejected.exit_code == 1
    result = runner.invoke(
        cli.app,
        [
            "nomination-prepare",
            run.name,
            compared.name,
            "--expected-plan-sha256",
            plan["plan_sha256"],
            *common,
        ],
    )
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)["result"]
    assert not receipt["inference_started"] and not receipt["training_started"]
    review = evaluation.plan_evaluation(
        run, replace(config, nomination_id=receipt["nomination_id"])
    )
    assert review.plan["config"]["split"] == "test"


def test_validation_history_scan_fails_closed_above_limit(tmp_path, monkeypatch):
    run, config, _, _, _ = candidate(tmp_path, monkeypatch)
    for index in range(100):
        (run / "evaluations" / f"extra-{index}").mkdir()
    with pytest.raises(ValueError, match="above 100"):
        evaluation.plan_evaluation(run, config)


@pytest.mark.parametrize("state", ["prepared", "failed"])
def test_test_cases_cannot_reuse_other_unscored_validation_history(
    tmp_path, monkeypatch, state
):
    run, config, _, _, _ = candidate(tmp_path, monkeypatch)
    document = ws.read_json(Path(config.cases))
    document["split"] = "validation"
    document["provenance"] = {
        "source": "reviewed",
        "reviewer": "fixture",
        "validation_is_fresh": True,
    }
    path = tmp_path / "other-validation.json"
    ws.write_json(path, document)
    review = evaluation.plan_evaluation(
        run, replace(config, cases=str(path), split="validation", nomination_id="")
    )
    retained = evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256)
    ws.write_json(retained / "state.json", {"status": state})
    with pytest.raises(ValueError, match="overlap retained validation"):
        evaluation.plan_evaluation(run, config)


def test_base_recommendation_cannot_be_nominated(tmp_path, monkeypatch):
    original = comparison_tests.Generator.generate

    def generate(self, messages, **kwargs):
        return (
            original(self, messages, **kwargs)
            if self.judge
            else ("Offer a free replacement.", 8, False)
        )

    monkeypatch.setattr(comparison_tests.Generator, "generate", generate)
    run, _, paths = cohort(tmp_path, monkeypatch, checkpoints=(1,))
    queued(monkeypatch, execute=True)
    compared = comparison.submit_comparison(run, paths[0].name)
    assert (
        comparison.comparison_status(compared)["result"]["recommendation"]["kind"]
        == "base"
    )
    with pytest.raises(ValueError, match="does not recommend"):
        nomination.nomination_plan(run, compared.name)
    assert not (run / "nominations").exists()


def test_escaping_owned_nomination_submission_preserves_editor(tmp_path, monkeypatch):
    run, _, _, _, _ = candidate(tmp_path, monkeypatch)
    panel = EvaluationPanel(run)
    panel.handle("g")
    panel.handle("n")
    evaluation_tests.wait(panel)
    before = dict(panel.values)
    original = nomination.prepare_nomination
    started, release = threading.Event(), threading.Event()

    def delayed(*args, **kwargs):
        started.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "stateset_agents.finetune_evaluation_tui.prepare_nomination", delayed
    )
    panel.handle("ENTER")
    assert started.wait(10)
    panel.handle("ESC")
    assert panel.mode == "comparison"
    release.set()
    evaluation_tests.wait(panel)
    assert panel.mode == "comparison" and panel.values == before
    assert "Nomination retained:" in panel.message


def test_agent_schema_accepts_only_explicit_nominated_test_targets():
    from jsonschema import Draft202012Validator

    schema = evaluation.evaluation_schema()["config_schema"]
    validator = Draft202012Validator(schema)
    base = {
        "cases": "held-out.json",
        "judge_model": "fixture/judge",
        "judge_revision": "b" * 40,
        "checkpoint": "checkpoint-1",
    }
    assert not list(validator.iter_errors(dict(base, split="validation")))
    assert list(validator.iter_errors(dict(base, split="test")))
    nominated = dict(base, split="test", nomination_id="20261017-000000-abcdef12")
    assert not list(validator.iter_errors(nominated))
    assert list(validator.iter_errors(dict(nominated, split="validation")))
    assert list(validator.iter_errors(dict(nominated, checkpoint="")))


@pytest.mark.skipif(
    sys.platform != "linux", reason="Curses nomination uses a Linux PTY"
)
@pytest.mark.parametrize("size", [(74, 44), (110, 32)])
def test_real_terminal_nominates_and_reviews_fresh_cases_without_inference(
    tmp_path, monkeypatch, size
):
    import fcntl
    import pty
    import select
    import struct
    import termios
    import time

    run, config, _, _, _ = candidate(tmp_path, monkeypatch)
    Studio(run.parent).handle("s")
    draft = (run.parent / "draft.json").read_bytes()
    nominations = len(list((run / "nominations").iterdir()))
    evaluations = len(list((run / "evaluations").iterdir()))
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
        send(b"n", "REVIEW CHECKPOINT NOMINATION")
        send(b"\r", "Arrows/Tab field")
        send(b"\r", "Edit:")
        send(str(config.cases).encode() + b"\r", "Arrows/Tab field")
        send(b"r", "Evaluation SHA256")
        send(b"\x1b", "n new evaluation")
        send(b"\x1b", "j evaluate")
        os.write(terminal, b"q")
        assert process.wait(timeout=15) == 0
        assert (run.parent / "draft.json").read_bytes() == draft
        assert len(list((run / "nominations").iterdir())) == nominations + 1
        assert len(list((run / "evaluations").iterdir())) == evaluations
        assert b"PRIVATE" not in full
    finally:
        os.close(terminal)
        if process.poll() is None:
            process.kill()
            process.wait()
