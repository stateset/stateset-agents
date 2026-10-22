"""Artifact-bound quality, guided setup and cooperative stop contracts."""

import copy
import hashlib
import json
import os
import threading
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as api
from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as ws
from stateset_agents.evaluation.agent_runs import evaluation_report
from stateset_agents.finetune_inspection import inspect_workflow
from stateset_agents.finetune_quality import inspect_quality, record_quality
from stateset_agents.training.promotion import artifact_digest
from stateset_agents.tui import Studio


def make_completed(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "training question"},
                    {"role": "assistant", "content": "training answer"},
                ]
            }
        )
        + "\n"
    )
    config = ws.FinetuneConfig(dataset=str(data), revision="a" * 40, seed=0)
    run = ws.create_run(tmp_path / "workspace", ws.plan_run(config))
    output = run / "artifacts"
    output.mkdir()
    (output / "adapter_model.bin").write_bytes(b"synthetic-weights")
    ws.write_json(
        run / "completion.json",
        {"output_dir": str(output), "result_kind": "local_weights"},
    )
    ws.write_json(run / "state.json", {"status": "succeeded"})
    return run, config


@pytest.fixture
def completed(tmp_path):
    return make_completed(tmp_path)


def evidence_for(run, config, *, seeds=1, regression=False):
    digest = artifact_digest(run / "artifacts")

    def report(seed, candidate):
        success = (not candidate) if regression else candidate
        return evaluation_report(
            environment="synthetic-fixture",
            base_model=config.model,
            seed=seed,
            cases={f"PRIVATE-ID-{i}": {"unique": i} for i in range(32)},
            settings={
                "model_revision": config.revision,
                "judge_model": "fixture-judge",
                "judge_revision": "b" * 40,
                "rubric_revision": "c" * 64,
            },
            checkpoint={
                "path": (
                    str(run / "artifacts")
                    if candidate and seed == 0
                    else f"{'trained' if candidate else 'base'}-{seed}"
                ),
                "inventory_sha256": (
                    digest if candidate and seed == 0 else f"{seed + 101:064x}"
                ),
            },
            outcomes=[
                {
                    "case_id": f"PRIVATE-ID-{i}",
                    "success": success,
                    "reward": 0.99 if candidate else 0.5,
                    "policy_violations": 0,
                    "truncated": None,
                    "generated_tokens": 4,
                    "tool_calls": 0,
                    "elapsed_seconds": 0.1,
                }
                for i in range(32)
            ],
        )

    return {
        "schema_version": 1,
        "review": {"source": "reviewed", "reviewer": "synthetic fixture"},
        "baselines": [report(seed, False) for seed in range(seeds)],
        "candidates": [report(seed, True) for seed in range(seeds)],
    }


def write_evidence(tmp_path, value):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(value))
    return path


def wait_for(studio, attribute):
    from operator import attrgetter

    busy = attrgetter(attribute)
    until = time.monotonic() + 10
    while busy(studio) and time.monotonic() < until:
        studio.refresh()
        time.sleep(0.005)
    assert not busy(studio)


@pytest.mark.parametrize("seeds", [1, 6])
def test_quality_recomputes_paired_evidence_and_keeps_stronger_gate_separate(
    completed, tmp_path, seeds
):
    run, config = completed
    before = {
        name: (run / name).read_bytes()
        for name in ["plan.json", "state.json", "completion.json", "dataset"]
    }
    source = write_evidence(tmp_path, evidence_for(run, config, seeds=seeds))
    summary = record_quality(run, source)
    assert summary["recorded_study_passed"] is (seeds == 6)
    assert summary["seeds"] == seeds and summary["bound_seed"] == 0
    assert summary["mean_success_gain"] == 1
    assert inspect_quality(run) == summary
    assert "PRIVATE-ID" not in json.dumps(summary)
    if os.name != "nt":  # Windows st_mode carries no POSIX group/other bits.
        assert (run / "quality.json").stat().st_mode & 0o077 == 0
    assert all((run / name).read_bytes() == raw for name, raw in before.items())


def test_reward_increase_does_not_hide_outcome_regressions(completed, tmp_path):
    run, config = completed
    source = write_evidence(tmp_path, evidence_for(run, config, regression=True))
    summary = record_quality(run, source)
    assert summary["mean_success_gain"] == -1 and not summary["recorded_study_passed"]
    assert summary["regressions_total"] == 32 and len(summary["regressions"]) == 32
    assert (
        summary["pairs"][0]["candidate"]["mean_reward"]
        > summary["pairs"][0]["baseline"]["mean_reward"]
    )
    assert (
        summary["regressions"][0]["case_handle"]
        == hashlib.sha256(b"PRIVATE-ID-0").hexdigest()[:16]
    )
    assert "PRIVATE-ID" not in json.dumps(summary)


@pytest.mark.parametrize(
    "problem",
    [
        "hash",
        "model",
        "cases",
        "split",
        "duplicate_seed",
        "nan",
        "path",
        "hosted",
        "rowtype",
    ],
)
def test_invalid_evidence_cannot_overwrite_a_good_receipt(completed, tmp_path, problem):
    run, config = completed
    evidence = evidence_for(run, config)
    record_quality(run, write_evidence(tmp_path, evidence))
    retained = (run / "quality.json").read_bytes()
    bad = copy.deepcopy(evidence)
    candidate = bad["candidates"][0]
    if problem == "hash":
        candidate["selected_checkpoint"]["inventory_sha256"] = "0" * 64
    elif problem == "model":
        candidate["base_model"] = "PRIVATE-MODEL"
    elif problem == "cases":
        candidate["case_hashes"]["PRIVATE-ID-0"] = "0" * 64
    elif problem == "split":
        candidate["split"] = "validation"
    elif problem == "duplicate_seed":
        bad["candidates"].append(candidate)
        bad["baselines"].append(bad["baselines"][0])
    elif problem == "nan":
        candidate["outcomes"][0]["reward"] = float("nan")
    elif problem == "path":
        candidate["selected_checkpoint"]["path"] = str(tmp_path / "PRIVATE")
    elif problem == "hosted":
        ws.write_json(run / "completion.json", {"result_kind": "hosted_pointer"})
    elif problem == "rowtype":
        bad["candidates"] = [None]
    with pytest.raises(ValueError) as error:
        record_quality(run, write_evidence(tmp_path, bad))
    assert "PRIVATE" not in str(error.value)
    assert (run / "quality.json").read_bytes() == retained


@pytest.mark.parametrize("changed", ["weights", "completion", "plan", "receipt"])
def test_retained_quality_is_rejected_after_binding_changes(
    completed, tmp_path, changed
):
    run, config = completed
    record_quality(run, write_evidence(tmp_path, evidence_for(run, config)))
    path = {
        "weights": run / "artifacts/adapter_model.bin",
        "completion": run / "completion.json",
        "plan": run / "plan.json",
        "receipt": run / "quality.json",
    }[changed]
    if changed == "weights":
        path.write_bytes(b"different")
    else:
        record = json.loads(path.read_text())
        record["changed"] = True
        if changed == "receipt":
            record["binding"]["artifact_sha256"] = "0" * 64
        ws.write_json(path, record)
    with pytest.raises(ValueError, match="No verifiable"):
        inspect_quality(run)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "oversize"])
def test_quality_sources_are_bounded_regular_files(
    completed, tmp_path, kind, monkeypatch
):
    from stateset_agents import finetune_quality as quality

    run, _ = completed
    path = tmp_path / "unsafe"
    if kind == "symlink":
        path.symlink_to(run / "dataset")
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO unavailable")
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    else:
        path.write_bytes(b"x" * 100)
        monkeypatch.setattr(quality, "MAX_EVIDENCE_BYTES", 32)
    with pytest.raises(ValueError):
        record_quality(run, path)
    assert not (run / "quality.json").exists()


def test_cli_quality_and_stop_keep_versioned_controlled_responses(completed, tmp_path):
    run, config = completed
    evidence = write_evidence(tmp_path, evidence_for(run, config, regression=True))
    runner = CliRunner()
    result = runner.invoke(
        api.app,
        [
            "quality",
            run.name,
            "--workspace",
            str(run.parent),
            "--evidence",
            str(evidence),
        ],
    )
    assert (
        result.exit_code == 0
        and json.loads(result.stdout)["result"]["regressions_total"] == 32
    )
    assert "PRIVATE-ID" not in result.stdout
    ws.write_json(run / "state.json", {"status": "prepared"})
    _, digest = ws.read_run_plan(run)
    bad = runner.invoke(
        api.app,
        [
            "stop",
            run.name,
            "--workspace",
            str(run.parent),
            "--expected-plan-sha256",
            "0" * 64,
        ],
    )
    assert bad.exit_code == 1 and not (run / "stop.json").exists()
    good = runner.invoke(
        api.app,
        [
            "stop",
            run.name,
            "--workspace",
            str(run.parent),
            "--expected-plan-sha256",
            digest,
        ],
    )
    assert (
        good.exit_code == 0
        and json.loads(good.stdout)["result"]["status"] == "cancelled"
    )
    status = runner.invoke(
        api.app, ["status", run.name, "--workspace", str(run.parent)]
    )
    assert json.loads(status.stdout)["result"]["run"]["state"]["stop_requested"] is True


@pytest.mark.parametrize("state", ["prepared", "queued", "running"])
def test_stop_request_is_idempotent_and_prevents_model_loading(
    completed, monkeypatch, state
):
    run, _ = completed
    ws.write_json(run / "state.json", {"status": state})
    _, digest = ws.read_run_plan(run)
    result = ws.request_stop(run, expected_plan_sha256=digest)
    assert result["stop_requested"] and ws.stop_requested(run)
    raw = (run / "stop.json").read_bytes()
    ws.request_stop(run, expected_plan_sha256=digest)
    assert (run / "stop.json").read_bytes() == raw
    monkeypatch.setattr(worker.os, "umask", lambda _: 0)
    monkeypatch.setattr(
        worker, "_train_sft", lambda *_: pytest.fail("Must not load/train")
    )
    (run / "completion.json").unlink()
    assert worker.execute(run) == 1
    reported = ws.run_status(run)
    assert (
        reported["status"]
        == {
            "prepared": "cancelled",
            "queued": "queued",
            "running": "interrupted",
        }[state]
    )
    assert reported["execution_unconfirmed"] is (state == "running")
    assert (
        ws.run_terminal_status(reported)
        is {
            "prepared": True,
            "queued": False,
            "running": None,
        }[state]
    )
    assert not (run / "completion.json").exists()
    # Queued/running metadata and an ambient stop do not authorize a direct
    # worker to take over that run. The real supervised child consumes its stop.
    assert not (run / "execution.json").exists()
    assert not (run / "admission.json").exists()
    assert ws.stop_requested(run) and (run / "stop.json").read_bytes() == raw
    with pytest.raises(ws.FinetuneFailure) as stopped:
        worker._check_stop(run)
    assert stopped.value.code == "cancelled"


def test_remote_stop_and_stale_plan_are_rejected_without_side_effects(completed):
    run, _ = completed
    with pytest.raises(ValueError):
        ws.request_stop(run, expected_plan_sha256="0" * 64)
    plan, _ = ws.read_run_plan(run)
    plan["config"]["provider"] = "river"
    ws.write_json(run / "plan.json", plan)
    with pytest.raises(ValueError, match="Remote stop"):
        ws.request_stop(run, expected_plan_sha256=ws.read_run_plan(run)[1])
    assert not (run / "stop.json").exists()


@pytest.mark.parametrize(
    "missing", ["review", "judge_revision", "model_revision", "cases"]
)
def test_six_seed_gate_requires_pins_review_and_distinct_cases(
    completed, tmp_path, missing
):
    run, config = completed
    evidence = evidence_for(run, config, seeds=6)
    if missing == "review":
        evidence.pop("review")
    elif missing in {"judge_revision", "model_revision"}:
        for report in evidence["baselines"] + evidence["candidates"]:
            report["settings"].pop(missing)
    else:
        for report in evidence["baselines"] + evidence["candidates"]:
            report["case_hashes"] = dict.fromkeys(report["case_hashes"], "1" * 64)
    summary = record_quality(run, write_evidence(tmp_path, evidence))
    assert not summary["recorded_study_passed"]


def test_completion_cannot_redirect_quality_hashing_outside_run(
    completed, tmp_path, monkeypatch
):
    run, config = completed
    source = write_evidence(tmp_path, evidence_for(run, config))
    ws.write_json(
        run / "completion.json",
        {"output_dir": str(run / "artifacts" / ".." / ".." / "PRIVATE")},
    )
    monkeypatch.setattr(
        "stateset_agents.training.promotion.artifact_digest",
        lambda *_: pytest.fail("Outside artifact must not be read"),
    )
    with pytest.raises(ValueError):
        record_quality(run, source)


def test_stop_during_launch_review_cannot_start_process(completed, monkeypatch):
    run, _ = completed
    ws.write_json(run / "state.json", {"status": "prepared"})
    original = ws.verified_run_settings

    def cancel_during_review(path):
        ws.request_stop(path, expected_plan_sha256=ws.read_run_plan(path)[1])
        return original(path)

    monkeypatch.setattr(ws, "verified_run_settings", cancel_during_review)
    monkeypatch.setattr(
        ws.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("Stopped run cannot launch"),
    )
    with pytest.raises(ValueError, match="stop request"):
        ws.launch_run(run)
    assert ws.run_status(run)["status"] == "cancelled"


@pytest.mark.parametrize("bad", ["checksum", "boolean_version", "symlink", "fifo"])
def test_stop_receipts_are_bounded_and_bound_to_plan(completed, bad):
    run, _ = completed
    path = run / "stop.json"
    if bad == "symlink":
        path.symlink_to(run / "dataset")
    elif bad == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO unavailable")
        os.mkfifo(path)
    else:
        ws.write_json(
            path,
            {
                "schema_version": True if bad == "boolean_version" else 1,
                "plan_sha256": "0" * 64,
            },
        )
    with pytest.raises((OSError, ValueError)):
        ws.stop_requested(run)


def test_queued_stop_preserves_child_cancellation_through_supervisor(
    completed, monkeypatch
):
    run, _ = completed
    (run / "completion.json").unlink()
    ws.write_json(run / "state.json", {"status": "prepared"})
    monkeypatch.setattr(worker.os, "umask", lambda _: 0)
    monkeypatch.setattr(worker, "queue_executable", lambda: "/fixture/buildr")
    commands = []

    def start(command, **kwargs):
        commands.append(command)

        def wait(**kwargs):
            ws.request_stop(run, expected_plan_sha256=ws.read_run_plan(run)[1])
            return worker.execute(
                run,
                supervisor_token=captured_environment[worker.SUPERVISOR_TOKEN_ENV],
            )

        captured_environment = kwargs["env"]
        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(worker.subprocess, "Popen", start)
    assert worker.supervise(run) == 1
    assert commands[0][0] == "/fixture/buildr"
    assert ws.run_status(run)["status"] == "cancelled"
    assert ws.failure_info(run, ws.run_status(run))["code"] == "cancelled"


@pytest.mark.parametrize("size", [(72, 22), (74, 44), (110, 32)])
def test_tui_quality_import_and_local_stop_are_explicit_and_preserve_draft(
    completed, tmp_path, size, settle_preview
):
    run, config = completed
    source = write_evidence(tmp_path, evidence_for(run, config, regression=True))
    studio = Studio(run.parent, config)
    before = dict(studio.values)
    studio.handle("TAB")
    studio.handle("e")
    wait_for(studio, "quality_panel.busy")
    assert "No verifiable" in "\n".join(studio.lines(*size))
    studio.handle("f")
    for character in str(source):
        studio.handle(character)
    studio.handle("ENTER")
    wait_for(studio, "quality_panel.busy")
    assert studio.quality_panel.report["regressions_total"] == 32
    assert "PRIVATE-ID" not in "\n".join(studio.lines(*size))
    studio.handle("ESC")
    ws.write_json(run / "state.json", {"status": "prepared"})
    studio.refresh()
    studio.handle("k")
    assert studio.overlay == "stop" and not (run / "stop.json").exists()
    studio.handle("ESC")
    assert not (run / "stop.json").exists()
    studio.handle("k")
    studio.handle("ENTER")
    settle_preview(studio)
    assert ws.stop_requested(run) and studio.values == before


def test_workflow_gives_safe_next_actions_for_incomplete_configuration(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "stateset_agents.finetune_inspection.inspect_readiness",
        lambda *_: pytest.fail("Settings invalid"),
    )
    report = inspect_workflow({"dataset": "", "epochs": 0}, tmp_path)
    assert [step["action"] for step in report["steps"]] == [
        "settings",
        "dataset",
        "runtime",
        "review",
        "quality",
    ]
    assert [step["status"] for step in report["steps"][:4]] == [
        "blocked",
        "blocked",
        "unchecked",
        "blocked",
    ]
    assert (
        not report["training_started"]
        and not report["prepared"]
        and not list(tmp_path.iterdir())
    )


def test_guided_setup_runs_in_background_and_requires_explicit_launch_review(
    completed, monkeypatch, settle_preview
):
    run, config = completed
    monkeypatch.setattr(
        "stateset_agents.finetune_inspection.inspect_readiness",
        lambda *_: {"failed": 0, "warnings": 1},
    )
    studio = Studio(run.parent, config)
    before = dict(studio.values)
    studio.handle("w")
    wait_for(studio, "workflow_panel.busy")
    assert studio.workflow_panel.report["steps"][1]["status"] == "ready"
    assert "GUIDED SETUP" in "\n".join(studio.lines(72, 22))
    studio.workflow_panel.index = 3
    studio.handle("ENTER")
    settle_preview(studio)
    assert studio.overlay == "review" and not (run / "launch.lock").exists()
    studio.handle("ESC")
    assert studio.values == before
    studio.handle("w")
    wait_for(studio, "workflow_panel.busy")
    studio.values["epochs"] += 1
    studio.handle("ENTER")
    assert studio.overlay == "workflow" and "current setup snapshot" in studio.message


def test_quality_hashing_does_not_block_escaping_or_change_target_run(
    completed, monkeypatch
):
    run, config = completed
    entered, release = threading.Event(), threading.Event()

    def block(_run):
        assert _run == run
        entered.set()
        assert release.wait(5)
        raise ValueError("PRIVATE")

    monkeypatch.setattr("stateset_agents.tui.inspect_quality", block)
    studio = Studio(run.parent, config)
    studio.handle("TAB")
    studio.handle("e")
    try:
        assert entered.wait(2)
        studio.handle("ESC")
        assert studio.overlay is None and studio.quality_panel.busy
        studio.run_query = "no matches"
        studio.refresh()
    finally:
        release.set()
    wait_for(studio, "quality_panel.busy")
    assert "PRIVATE" not in json.dumps(studio.quality_panel.report)
    assert studio.quality_panel.run == run


def test_workflow_agent_command_accepts_incomplete_drafts_without_preparing(tmp_path):
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(asdict(ws.FinetuneConfig())))
    result = CliRunner().invoke(
        api.app,
        ["workflow", "--config", str(path), "--workspace", str(tmp_path / "unused")],
    )
    assert result.exit_code == 0
    report = json.loads(result.stdout)["result"]
    assert (
        report["steps"][0]["status"] == "blocked" and not (tmp_path / "unused").exists()
    )


def test_sft_stop_requests_checkpoint_save_at_step_boundary(completed, monkeypatch):
    run, config = completed
    ws.write_json(run / "state.json", {"status": "running"})
    _, digest = ws.read_run_plan(run)

    def fake_sft(**kwargs):
        ws.request_stop(run, expected_plan_sha256=digest)
        control = SimpleNamespace(should_training_stop=False, should_save=False)
        returned = kwargs["callbacks"][0].on_step_end(None, None, control)
        assert (
            returned is control and control.should_training_stop and control.should_save
        )
        return run / "artifacts"

    monkeypatch.setattr("stateset_agents.training.sft.run_sft", fake_sft)
    assert worker._train_sft(run, config) == run / "artifacts"


@pytest.mark.asyncio
async def test_native_rl_stop_saves_checkpoint_at_committed_update(
    completed, tmp_path, monkeypatch
):
    from pathlib import Path

    run, config = completed
    data = tmp_path / "rl.json"
    data.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "context": "question",
                    "expected_resolution": "answer",
                    "policy_facts": ["fact"],
                    "user_responses": ["reply"],
                }
            ]
        )
    )
    config = ws.FinetuneConfig(
        **{**asdict(config), "method": "grpo", "dataset": str(data)}
    )
    run = ws.create_run(tmp_path / "rl-workspace", ws.plan_run(config))
    ws.write_json(run / "state.json", {"status": "running"})
    captured = {}

    class Agent:
        def __init__(self, config):
            pass

        async def initialize(self):
            pass

    class Trainer:
        def __init__(self, *args, **kwargs):
            self.callback = kwargs["callbacks"][0]
            self.global_step = 1

        async def initialize(self):
            pass

        async def train(self):
            ws.request_stop(run, expected_plan_sha256=ws.read_run_plan(run)[1])
            await self.callback.on_optimizer_step(1, {})

        async def save_checkpoint(self, *, checkpoint_name):
            captured["name"] = checkpoint_name

    monkeypatch.setattr("stateset_agents.core.agent.MultiTurnAgent", Agent)
    monkeypatch.setattr(
        "stateset_agents.training.multi_turn_trainer.MultiTurnGRPOTrainer", Trainer
    )
    with pytest.raises(ws.FinetuneFailure) as error:
        await worker._train_rl(run, config)
    assert error.value.code == "cancelled" and captured["name"] == "stopped-step-1"
    state = ws.read_json(run / "state.json")
    assert (
        state["committed_updates"] == 1
        and Path(state["recovery_checkpoint"]).name == "stopped-step-1"
    )
