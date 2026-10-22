"""Reviewed, queued policy/judge evaluations and responsive terminal controls."""

import copy
import json
import os
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as cli
from stateset_agents import finetune_evaluation as evaluation
from stateset_agents import finetune_evaluation_worker as worker
from stateset_agents import finetune_workspace as ws
from stateset_agents.evaluation.support_judge import ALL_CHECKS
from stateset_agents.finetune_evaluation_tui import EvaluationPanel
from stateset_agents.tui import Studio
from tests.unit.test_finetune_recovery import make_parent


def make_evaluation(tmp_path, kind="sft"):
    run, _ = make_parent(tmp_path, kind)
    artifact = run / "artifacts/checkpoint-1"
    ws.write_json(
        artifact / "adapter_config.json",
        {"base_model_name_or_path": "Qwen/Qwen3.5-0.8B"},
    )
    ws.write_json(run / "state.json", {"status": "succeeded"})
    ws.write_json(run / "completion.json", {"output_dir": str(artifact)})
    cases = {
        "schema_version": 1,
        "split": "test",
        "provenance": {
            "source": "reviewed",
            "reviewer": "Synthetic fixture reviewer",
            "test_is_fresh": True,
        },
        "cases": [
            {
                "id": f"case-{index}",
                "context": "You help with deliveries.",
                "user_turns": [f"PRIVATE package {index} was damaged."],
                "policy_facts": ["Replace damaged deliveries."],
                "expected_resolution": "Offer replacement.",
                "must_avoid": [],
            }
            for index in range(2)
        ],
    }
    source = tmp_path / "held-out.json"
    ws.write_json(source, cases)
    config = evaluation.EvaluationConfig(
        cases=str(source), judge_model="fixture/judge", judge_revision="b" * 40
    )
    return run, config


def prepared(tmp_path):
    run, config = make_evaluation(tmp_path)
    review = evaluation.plan_evaluation(run, config)
    return (
        evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256),
        review,
    )


def wait(panel):
    deadline = time.monotonic() + 15
    while panel.busy and time.monotonic() < deadline:
        panel.refresh()
        time.sleep(0.005)
    assert not panel.busy


def test_review_and_snapshot_are_private_source_independent_and_not_training(tmp_path):
    run, config = make_evaluation(tmp_path)
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    review = evaluation.plan_evaluation(run, config)
    assert "PRIVATE" not in json.dumps(review.plan)
    assert review.plan["reply_generations"] == 4
    assert review.plan["max_judge_generations"] == 16
    assert before == {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="checksum"):
        evaluation.prepare_evaluation(review, expected_plan_sha256="0" * 64)
    assert not (run / "evaluations").exists()
    path = evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256)
    Path(config.cases).unlink()
    assert evaluation.read_evaluation(path).sha256 == review.sha256
    assert evaluation.evaluation_status(path)["state"]["status"] == "prepared"
    assert not (path / "launch.lock").exists()
    assert not (run / "launch.lock").exists() and not (run / "quality.json").exists()
    assert all(p.stat().st_mode & 0o077 == 0 for p in [path, *path.rglob("*")])


@pytest.mark.parametrize(
    "mutation",
    ["cases", "weights", "plan", "completion", "dataset", "protocol", "code"],
)
def test_changed_inputs_fail_before_inference(tmp_path, monkeypatch, mutation):
    path, review = prepared(tmp_path)
    if mutation == "weights":
        Path(
            review.plan["binding"]["artifact_path"], "adapter_model.safetensors"
        ).write_bytes(b"changed")
    elif mutation == "dataset":
        (review.run / "dataset").write_bytes(b"changed")
    elif mutation == "code":
        monkeypatch.setattr(
            evaluation, "package_implementation", lambda: {"changed": True}
        )
    else:
        record = (
            path / "cases.json"
            if mutation == "cases"
            else (
                path / "plan.json"
                if mutation in {"protocol"}
                else review.run / (mutation + ".json")
            )
        )
        value = json.loads(record.read_text())
        if mutation == "protocol":
            value["thresholds"]["success_score"] = 0.1
        else:
            value["changed"] = True
        ws.write_json(record, value)
    calls = []
    monkeypatch.setattr(
        worker.subprocess, "call", lambda *a, **kw: calls.append(a) or 0
    )
    assert worker.execute(path) == 1
    assert calls == [] and not (path / "result.json").exists()


@pytest.mark.parametrize(
    "problem",
    [
        "unreviewed",
        "validation",
        "duplicate",
        "overlap",
        "no_turns",
        "many_turns",
        "nonfinite",
        "missing",
        "symlink",
        "fifo",
    ],
)
def test_cases_fail_closed_without_writes_or_private_errors(tmp_path, problem):
    run, config = make_evaluation(tmp_path)
    source = Path(config.cases)
    document = json.loads(source.read_text())
    if problem == "unreviewed":
        document["provenance"]["source"] = "generated"
    elif problem == "validation":
        document["split"] = "validation"
    elif problem == "duplicate":
        document["cases"].append(copy.deepcopy(document["cases"][0]))
    elif problem == "overlap":
        document["cases"][0]["user_turns"] = ["PRIVATE CUSTOMER"]
    elif problem == "no_turns":
        document["cases"][0]["user_turns"] = []
    elif problem == "many_turns":
        document["cases"][0]["user_turns"] *= 9
    elif problem == "nonfinite":
        document["private"] = float("nan")
    elif problem == "missing":
        document["cases"][0].pop("expected_resolution")
    if problem in {"symlink", "fifo"}:
        source.unlink()
        if problem == "symlink":
            source.symlink_to(run / "dataset")
        else:
            os.mkfifo(source)
    else:
        source.write_text(json.dumps(document))
    with pytest.raises((ValueError, OSError)) as error:
        evaluation.plan_evaluation(run, config)
    assert "PRIVATE" not in str(error.value)
    assert not (run / "evaluations").exists()


@pytest.mark.parametrize(
    "turn",
    [
        "private\u00a0\u00a0customer",  # case + non-breaking, repeated spaces
        "\uff30\uff32\uff29\uff36\uff21\uff34\uff25 CUSTOMER",  # full-width
        "  PRIVATE\n\tCUSTOMER ",
    ],
)
def test_normalized_overlap_with_training_text_is_rejected(tmp_path, turn):
    """Only byte-identical-after-strip().casefold() repeats were caught."""
    run, config = make_evaluation(tmp_path)
    source = Path(config.cases)
    document = json.loads(source.read_text())
    document["cases"][0]["user_turns"] = [turn]
    source.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="overlap retained training text"):
        evaluation.plan_evaluation(run, config)


def test_normalized_duplicate_initial_questions_are_rejected(tmp_path):
    run, config = make_evaluation(tmp_path)
    source = Path(config.cases)
    document = json.loads(source.read_text())
    first = document["cases"][0]["user_turns"][0]
    document["cases"][1]["user_turns"] = [
        "\u00a0" + first.upper().replace(" ", "  ") + " "
    ]
    source.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="must be distinct"):
        evaluation.plan_evaluation(run, config)


@pytest.mark.parametrize(
    "changes",
    [
        {"judge_revision": "main"},
        {"judge_revision": "0" * 40},
        {"max_new_tokens": True},
        {"memory_gb": float("nan")},
        {"seed": -1},
        {"timeout_seconds": 1},
        {"context_length": 512},
        {"judge_model": "../../secret"},
    ],
)
def test_invalid_configuration_has_no_model_or_workspace_effects(tmp_path, changes):
    run, config = make_evaluation(tmp_path)
    with pytest.raises(ValueError):
        evaluation.plan_evaluation(run, replace(config, **changes))
    assert not (run / "evaluations").exists()


def test_launch_requires_queue_and_exact_once_receipt(tmp_path, monkeypatch):
    path, review = prepared(tmp_path)
    calls = []
    monkeypatch.setattr(evaluation, "queue_executable", lambda: None)
    with pytest.raises(ValueError, match="Buildr"):
        evaluation.launch_evaluation(path, expected_plan_sha256=review.sha256)
    assert not (path / "launch.lock").exists()
    monkeypatch.setattr(evaluation, "queue_executable", lambda: "/queue")
    monkeypatch.setattr(
        evaluation.subprocess,
        "Popen",
        lambda argv, **kwargs: calls.append((argv, kwargs))
        or type("Process", (), {"pid": 123})(),
    )
    assert evaluation.launch_evaluation(path, expected_plan_sha256=review.sha256) == 123
    with pytest.raises(FileExistsError):
        evaluation.launch_evaluation(path, expected_plan_sha256=review.sha256)
    assert len(calls) == 1
    assert calls[0][0][-2:] == ["--supervise", str(path)]
    assert calls[0][1]["start_new_session"]
    assert not (review.run / "stop.json").exists()


def test_supervisor_preserves_queue_argv_and_bounds_entire_pipeline(
    tmp_path, monkeypatch
):
    path, review = prepared(tmp_path)
    ws.write_json(path / "launch.lock", {"plan_sha256": review.sha256})
    calls = []
    monkeypatch.setattr(worker, "queue_executable", lambda: "/queue")

    def call(argv, **kwargs):
        calls.append((argv, kwargs))
        kwargs["stderr"].write(
            (
                json.dumps(
                    {
                        "job_id": "abcdef123456",
                        "state": "queued",
                        "timestamp": time.time(),
                    }
                )
                + "\n"
            ).encode()
        )
        ws.write_json(path / "state.json", {"status": "succeeded"})
        return 0

    monkeypatch.setattr(worker.subprocess, "call", call)
    assert worker.supervise(path) == 0
    argv, kwargs = calls[0]
    assert argv[:5] == ["/queue", "run", "--events-json", "--lane", "heavy"]
    assert argv[argv.index("--timeout") + 1] == str(
        review.plan["config"]["timeout_seconds"]
    )
    assert argv[-4:] == [
        "-m",
        "stateset_agents.finetune_evaluation_worker",
        "--execute",
        str(path),
    ]
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert evaluation.evaluation_status(path)["queue"]["job_id"] == "abcdef123456"
    assert (path / "queue-events.jsonl").stat().st_mode & 0o777 == 0o600


def _owner(pid, token):
    return {"pid": pid, "process_token": token}


@pytest.fixture
def controlled_owners(monkeypatch):
    """Decide owner liveness deterministically; never inspect a real PID."""
    alive = {}
    monkeypatch.setattr(ws, "sys", type("Sys", (), {"platform": "linux"})())
    monkeypatch.setattr(ws, "process_token", lambda pid: alive.get(pid))
    return alive


@pytest.mark.parametrize(
    "status,record", [("queued", "supervisor.json"), ("running", "execution.json")]
)
def test_lost_evaluation_owner_is_interrupted_not_active_forever(
    tmp_path, controlled_owners, status, record
):
    """After a reboot or OOM kill nobody will ever finish or fail the job."""
    path, review = prepared(tmp_path)
    ws.write_json(path / "launch.lock", {"plan_sha256": review.sha256})
    ws.write_json(path / "state.json", {"status": status})
    ws.write_json(path / record, _owner(4242, "owner-start"))
    controlled_owners[4242] = "owner-start"
    assert evaluation.evaluation_status(path)["state"]["status"] == status
    assert evaluation.evaluation_status(path)["state"]["execution_unconfirmed"] is False

    controlled_owners.pop(4242)
    state = evaluation.evaluation_status(path)["state"]
    assert state["status"] == "interrupted"
    assert state["execution_unconfirmed"] is True
    assert "Evaluation incomplete" in state["error"]
    assert ws.read_json(path / "state.json")["status"] == status  # derived only
    panel = EvaluationPanel(review.run)
    panel.selected, panel.mode = evaluation.evaluation_status(path), "detail"
    assert "Supervisor lost" in "\n".join(panel.lines())
    panel.handle("s")
    assert panel.mode == "stop"


def test_unclaimed_evaluation_launch_is_interrupted_after_grace_and_stoppable(
    tmp_path, controlled_owners
):
    path, review = prepared(tmp_path)
    ws.write_json(path / "launch.lock", {"plan_sha256": review.sha256})
    assert evaluation.evaluation_status(path)["state"]["status"] == "prepared"
    submitted = os.stat(path / "launch.lock").st_mtime - 600
    os.utime(path / "launch.lock", (submitted, submitted))
    assert evaluation.evaluation_status(path)["state"]["status"] == "interrupted"
    evaluation.stop_evaluation(path, expected_plan_sha256=review.sha256)
    assert ws.read_json(path / "state.json")["status"] == "cancelled"


def test_second_supervisor_cannot_claim_or_change_a_supervised_evaluation(
    tmp_path, monkeypatch
):
    path, review = prepared(tmp_path)
    ws.write_json(path / "launch.lock", {"plan_sha256": review.sha256})
    ws.write_json(path / "state.json", {"status": "queued"})
    original = _owner(4242, "first-supervisor")
    ws.write_json(path / "supervisor.json", original)
    monkeypatch.setattr(worker, "queue_executable", lambda: "/queue")
    monkeypatch.setattr(
        worker.subprocess, "call", lambda *_a, **_k: pytest.fail("No second queue job")
    )
    assert worker.supervise(path) == 1
    assert ws.read_json(path / "supervisor.json") == original
    assert ws.read_json(path / "state.json") == {"status": "queued"}
    assert not (path / "command.json").exists()


def test_queue_progress_is_bounded_private_and_survives_partial_events(tmp_path):
    path, review = prepared(tmp_path)
    event = {
        "job_id": "abcdef123456",
        "state": "waiting",
        "timestamp": time.time() - 12,
        "wait_seconds": 123.5,
        "reason": "disk below PRIVATE requirement",
        "argv": ["SECRET_TOKEN"],
        "resources": {
            "disk_free_gb": 2.5,
            "required_free_gb": 4.0,
            "available_memory_gb": float("nan"),
            "required_memory_gb": True,
            "blocking_job": {"command_display": "PRIVATE CUSTOMER"},
        },
    }
    events = path / "queue-events.jsonl"
    events.write_bytes(
        b"x" * 70000 + b"\n" + json.dumps(event).encode() + b'\n{"partial":'
    )
    status = evaluation.evaluation_status(path)
    queue = status["queue"]
    assert queue["job_id"] == "abcdef123456" and queue["wait_seconds"] == 123.5
    assert queue["waiting_for"] == "free disk headroom"
    assert queue["sample_age_seconds"] >= 12
    assert queue["resources"] == {"disk_free_gb": 2.5, "required_free_gb": 4.0}
    assert "PRIVATE" not in json.dumps(status) and "SECRET" not in json.dumps(status)
    assert not (review.run / "stop.json").exists()
    panel = EvaluationPanel(review.run)
    panel.selected, panel.mode = status, "detail"
    screen = "\n".join(panel.lines())
    assert "Buildr job: abcdef123456" in screen and "free disk headroom" in screen
    assert "2.5 / 4.0 GiB" in screen and "sample age" in screen
    event.update(state="running", timestamp=time.time(), wait_seconds=140)
    events.write_text(json.dumps(event) + "\n")
    queue = evaluation.evaluation_status(path)["queue"]
    assert queue["state"] == "running" and "waiting_for" not in queue
    assert "resources" not in queue


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "symlink",
        "fifo",
        "bad-json",
        "bad-id",
        "bad-state",
        "bad-time",
        "huge-time",
        "bad-wait",
        "foreign-id",
    ],
)
def test_queue_metadata_never_blocks_or_exposes_untrusted_values(tmp_path, kind):
    path, _ = prepared(tmp_path)
    record = path / "queue-events.jsonl"
    event = {
        "job_id": "abcdef123456",
        "state": "waiting",
        "timestamp": time.time(),
        "reason": "PRIVATE_SECRET",
    }
    if kind == "symlink":
        record.symlink_to(path / "cases.json")
    elif kind == "fifo":
        os.mkfifo(record)
    elif kind == "bad-json":
        record.write_bytes(b"bad\n\xff\n[]\n")
    elif kind != "missing":
        if kind == "bad-id":
            event["job_id"] = "PRIVATE_SECRET"
        elif kind == "bad-state":
            event["state"] = []
        elif kind == "bad-time":
            event["timestamp"] = float("inf")
        elif kind == "huge-time":
            event["timestamp"] = 10**400
        elif kind == "bad-wait":
            event["wait_seconds"] = -1
        record.write_text(json.dumps(event) + "\n")
        if kind == "foreign-id":
            event.update(job_id="111111111111", state="succeeded")
            with record.open("a") as stream:
                stream.write(json.dumps(event) + "\n")
    status = evaluation.evaluation_status(path)
    assert "PRIVATE_SECRET" not in json.dumps(status)
    assert status["state"]["status"] == "prepared"
    if kind in {"bad-wait", "foreign-id"}:
        assert status["queue"]["waiting_for"] == "another admission constraint"
        assert status["queue"]["job_id"] == "abcdef123456"
        assert "wait_seconds" not in status["queue"]
    else:
        assert status["queue"] == {}


def test_evaluation_stop_is_bound_and_separate_from_training(tmp_path, monkeypatch):
    path, review = prepared(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        evaluation.stop_evaluation(path, expected_plan_sha256="0" * 64)
    evaluation.stop_evaluation(path, expected_plan_sha256=review.sha256)
    calls = []
    monkeypatch.setattr(
        worker.subprocess, "call", lambda *a, **kw: calls.append(a) or 0
    )
    assert worker.execute(path) == 1
    assert ws.read_json(path / "state.json")["status"] == "cancelled"
    assert calls == [] and not (review.run / "stop.json").exists()


class Generator:
    def __init__(self, model, revision, *, adapter=None):
        self.judge = model == "fixture/judge"
        self.adapter = adapter

    def generate(self, messages, *, tokens, context):
        if self.judge:
            checks = dict.fromkeys(ALL_CHECKS, True)
            if "Bad unsupported answer" in messages[-1]["content"]:
                checks["resolution_correct_outcome"] = False
            return json.dumps({"checks": checks}), 20, False
        return (
            (
                "Bad unsupported answer"
                if self.adapter
                else "I can replace the damaged delivery."
            ),
            8,
            False,
        )


def run_pipeline(path, monkeypatch):
    ws.write_json(
        path / "launch.lock", {"plan_sha256": evaluation.read_evaluation(path).sha256}
    )
    monkeypatch.setattr(worker, "LocalGenerator", Generator)
    calls = []

    def phase(argv, **kwargs):
        name = argv[-2]
        calls.append(name)
        review = evaluation.read_evaluation(path)
        if name == "judge":
            worker.judge_phase(path, review)
        else:
            worker.policy_phase(path, review, name)
        return 0

    monkeypatch.setattr(worker.subprocess, "call", phase)
    assert worker.execute(path) == 0
    assert calls == ["baseline", "candidate", "judge"]


def test_complete_pipeline_replays_real_rubric_without_overwriting_study(
    tmp_path, monkeypatch
):
    path, review = prepared(tmp_path)
    ws.write_json(review.run / "quality.json", {"existing_multiseed": True})
    retained = (review.run / "quality.json").read_bytes()
    run_pipeline(path, monkeypatch)
    result = evaluation.inspect_evaluation(path)
    assert result["judge_replay_verified"] and not result["quality_gain_verified"]
    assert (
        result["mean_judge_scores"]["baseline"]
        > result["mean_judge_scores"]["candidate"]
    )
    assert not result["summary"]["recorded_study_passed"]
    assert result["summary"]["regressions_total"] == 2
    assert (review.run / "quality.json").read_bytes() == retained
    assert "PRIVATE" not in json.dumps(result)
    assert evaluation.list_evaluations(review.run)[0]["state"]["status"] == "succeeded"
    assert all(p.stat().st_mode & 0o077 == 0 for p in path.rglob("*") if p.is_file())
    (path / "judge-recordings/batch-000000/response-000.json").write_text("tampered")
    with pytest.raises(ValueError):
        evaluation.inspect_evaluation(path)


def test_failed_judge_never_accepts_partial_quality(tmp_path, monkeypatch):
    path, _ = prepared(tmp_path)

    class InvalidJudge(Generator):
        def generate(self, messages, **kwargs):
            if self.judge:
                return "not valid JSON PRIVATE", 5, False
            return super().generate(messages, **kwargs)

    monkeypatch.setattr(worker, "LocalGenerator", InvalidJudge)
    review = evaluation.read_evaluation(path)
    worker.policy_phase(path, review, "baseline")
    worker.policy_phase(path, review, "candidate")
    with pytest.raises(ValueError, match="parsing"):
        worker.judge_phase(path, review)
    assert not (path / "result.json").exists()
    assert not (review.run / "quality.json").exists()


def test_cli_review_prepare_status_and_stale_submission(tmp_path, monkeypatch):
    run, config = make_evaluation(tmp_path)
    config_path = tmp_path / "eval-config.json"
    ws.write_json(config_path, asdict(config))
    runner = CliRunner()
    args = [run.name, "--workspace", str(run.parent)]
    result = runner.invoke(
        cli.app, ["evaluation-plan", *args, "--config", str(config_path)]
    )
    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    digest = response["result"]["plan_sha256"]
    assert not (run / "evaluations").exists() and "PRIVATE" not in result.stdout
    result = runner.invoke(
        cli.app,
        [
            "evaluation-prepare",
            *args,
            "--config",
            str(config_path),
            "--expected-plan-sha256",
            "0" * 64,
        ],
    )
    assert result.exit_code == 1 and not (run / "evaluations").exists()
    result = runner.invoke(
        cli.app,
        [
            "evaluation-prepare",
            *args,
            "--config",
            str(config_path),
            "--expected-plan-sha256",
            digest,
        ],
    )
    assert result.exit_code == 0, result.output
    evaluation_id = json.loads(result.stdout)["result"]["evaluation_id"]
    result = runner.invoke(cli.app, ["evaluation-status", *args, evaluation_id])
    assert (
        result.exit_code == 0
        and json.loads(result.stdout)["result"]["state"]["status"] == "prepared"
    )
    result = runner.invoke(cli.app, ["evaluation-status", *args, "../secret"])
    assert result.exit_code == 1 and "invalid_evaluation_id" in result.stdout
    assert not (run / "evaluations" / evaluation_id / "launch.lock").exists()


@pytest.mark.parametrize("size", [(72, 22), (74, 44), (110, 32)])
def test_tui_config_review_escape_preserves_draft_and_never_submits(tmp_path, size):
    run, config = make_evaluation(tmp_path)
    studio = Studio(run.parent)
    draft = dict(studio.values)
    studio.tab = "runs"
    studio.handle("j")
    panel = studio.evaluation_panel
    assert panel is not None and panel.run == run
    panel.values = asdict(config)
    studio.handle("n")
    studio.handle("r")
    wait(panel)
    assert panel.mode == "review"
    screen = "\n".join(studio.lines(*size))
    assert "LOCAL PAIRED EVALUATION" in screen
    assert "Inputs captured" in screen
    assert "Enter confirms" in screen and "PRIVATE" not in screen
    studio.handle("ESC")
    studio.handle("ESC")
    assert studio.overlay is None
    assert not (run / "evaluations").exists()
    assert studio.values == draft


def test_tui_launch_is_owned_background_exact_run_and_duplicate_safe(
    tmp_path, monkeypatch
):
    run, config = make_evaluation(tmp_path)
    panel = EvaluationPanel(run)
    panel.values = asdict(config)
    panel.handle("n")
    panel.handle("r")
    wait(panel)
    gate = threading.Event()
    entered = threading.Event()
    calls = []

    def launch(path, *, expected_plan_sha256):
        calls.append((path, expected_plan_sha256, threading.current_thread().daemon))
        entered.set()
        gate.wait(10)
        return 123

    monkeypatch.setattr(
        "stateset_agents.finetune_evaluation_tui.launch_evaluation", launch
    )
    panel.handle("ENTER")
    assert entered.wait(10)
    panel.handle("ENTER")
    panel.handle("ESC")
    assert panel.mode == "list" and panel.busy
    gate.set()
    wait(panel)
    assert len(calls) == 1 and not calls[0][2]
    assert panel.mode == "list" and panel.selected["evaluation_id"] == calls[0][0].name
    assert calls[0][1] == panel.review.sha256


def test_real_tiny_cpu_policy_generation_uses_pinned_base_and_safe_adapter(
    tmp_path, monkeypatch
):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tests.sft_fixtures import write_tiny_text_policy

    cache = tmp_path / "cache"
    base = cache / "models--fixture--base" / "snapshots" / ("a" * 40)
    base.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.chdir(tmp_path)
    collision = tmp_path / "fixture/base"
    collision.mkdir(parents=True)
    (collision / "config.json").write_text("relative directory must never load")
    model = write_tiny_text_policy(base)
    adapter = tmp_path / "adapter"
    get_peft_model(
        model,
        LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM"),
    ).save_pretrained(adapter)
    original_model = AutoModelForCausalLM.from_pretrained
    original_tokenizer = AutoTokenizer.from_pretrained
    calls = []

    def load(model, **kwargs):
        calls.append(kwargs)
        assert Path(model) == base
        assert kwargs["local_files_only"] and kwargs["trust_remote_code"] is False
        return original_model(
            base, **{k: v for k, v in kwargs.items() if k != "revision"}
        )

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda model, **kwargs: original_tokenizer(
            base, **{k: v for k, v in kwargs.items() if k != "revision"}
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    first = worker.LocalGenerator("fixture/base", "a" * 40)
    second = worker.LocalGenerator("fixture/base", "a" * 40, adapter=adapter)
    messages = [{"role": "user", "content": "reply"}]
    output = first.generate(messages, tokens=4, context=64)
    assert output[0].split() == ["reply"] * 4 and output[1:] == (4, True)
    assert output == second.generate(messages, tokens=4, context=64)
    assert all(call["revision"] == "a" * 40 for call in calls)
    with pytest.raises(ValueError, match="context"):
        first.generate(messages, tokens=128, context=128)


@pytest.mark.parametrize("reply_token_id", [0, 1, 2], ids=["unknown", "eos", "padding"])
def test_real_tiny_generation_rejects_special_only_completions(
    tmp_path, monkeypatch, reply_token_id
):
    import torch

    from tests.sft_fixtures import write_tiny_text_policy

    cache = tmp_path / "cache"
    base = cache / "models--fixture--base" / "snapshots" / ("a" * 40)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    write_tiny_text_policy(base, reply_token_id=reply_token_id)
    generator = worker.LocalGenerator("fixture/base", "a" * 40)
    with pytest.raises(ValueError, match="empty completion"):
        generator.generate([{"role": "user", "content": "reply"}], tokens=4, context=64)


@pytest.mark.parametrize("name", ["../..", "fixture/..", "fixture/a--b", "fixture/a."])
def test_local_paths_are_not_pinned_hub_model_ids(tmp_path, name):
    run, config = make_evaluation(tmp_path)
    with pytest.raises(ValueError, match="repository ID"):
        evaluation.plan_evaluation(run, replace(config, judge_model=name))


@pytest.mark.parametrize("snapshot", [None, "wrong-commit"])
def test_missing_or_mismatched_cache_never_initializes_a_model(
    tmp_path, monkeypatch, snapshot
):
    import huggingface_hub
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def resolve(**kwargs):
        assert kwargs["repo_id"] == "fixture/base"
        assert kwargs["revision"] == "a" * 40 and kwargs["local_files_only"]
        if snapshot is None:
            raise FileNotFoundError("Missing pinned cache")
        return str(tmp_path / "snapshots" / snapshot)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "No model/tokenizer may initialize before resolving its pinned cache"
        )

    monkeypatch.setattr(huggingface_hub, "snapshot_download", resolve)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        worker.LocalGenerator("fixture/base", "a" * 40)


def test_tui_can_review_and_submit_an_existing_prepared_id(tmp_path, monkeypatch):
    path, review = prepared(tmp_path)
    panel = EvaluationPanel(review.run)
    panel.handle("ENTER")
    panel.handle("r")
    wait(panel)
    assert panel.prepared_path == path and panel.mode == "review"
    calls = []
    monkeypatch.setattr(
        "stateset_agents.finetune_evaluation_tui.launch_evaluation",
        lambda path, **kwargs: calls.append((path, kwargs)) or 123,
    )
    panel.handle("ENTER")
    wait(panel)
    assert calls == [(path, {"expected_plan_sha256": review.sha256})]
    assert len(evaluation.list_evaluations(review.run)) == 1


def test_empty_evaluation_log_does_not_break_the_terminal_footer(tmp_path):
    path, review = prepared(tmp_path)
    (path / "run.log").write_bytes(b"")
    studio = Studio(review.run.parent)
    studio.tab = "runs"
    studio.handle("j")
    panel = studio.evaluation_panel
    panel.handle("ENTER")
    panel.handle("l")
    assert panel.message == ""
    assert len(studio.lines(74, 44)) == 44


@pytest.mark.parametrize(
    "failed",
    [
        "resolution_correct_outcome",
        "resolution_no_redundant_request",
        "accuracy_no_invented_details",
        "conciseness_no_repetition",
    ],
)
def test_high_mean_judge_score_cannot_hide_a_critical_failure(failed):
    first = dict.fromkeys(ALL_CHECKS, True)
    second = dict(first)
    assert evaluation.judge_success(1.0, [first, second], None)
    first[failed] = False
    assert not evaluation.judge_success(0.96, [first, second], None)
    assert failed in evaluation.evaluation_thresholds()["required_checks"]


def test_snapshot_write_failure_exposes_no_partial_evaluation(tmp_path, monkeypatch):
    run, config = make_evaluation(tmp_path)
    review = evaluation.plan_evaluation(run, config)
    original = evaluation.write_json

    def fail(path, value):
        if path.name == "cases.json":
            raise OSError("fixture disk full")
        original(path, value)

    monkeypatch.setattr(evaluation, "write_json", fail)
    with pytest.raises(OSError):
        evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256)
    assert list((run / "evaluations").iterdir()) == []


def test_snapshot_serialization_must_fit_before_preparation(tmp_path, monkeypatch):
    run, config = make_evaluation(tmp_path)
    path = Path(config.cases)
    raw = json.dumps(json.loads(path.read_text()), separators=(",", ":"))
    path.write_text(raw)
    monkeypatch.setattr(evaluation, "MAX_CASE_BYTES", len(raw.encode()) + 1)
    with pytest.raises(ValueError, match="snapshot exceeds"):
        evaluation.plan_evaluation(run, config)
    assert not (run / "evaluations").exists()


def test_execution_requires_launch_receipt_and_never_restarts_a_claimed_pipeline(
    tmp_path, monkeypatch
):
    path, review = prepared(tmp_path)
    calls = []
    monkeypatch.setattr(
        worker.subprocess, "call", lambda *a, **kw: calls.append(a) or 0
    )
    assert worker.execute(path) == 1 and calls == []
    assert not (path / "execution.lock").exists()
    ws.write_json(path / "launch.lock", {"plan_sha256": review.sha256})
    (path / "execution.lock").write_bytes(b"")
    retained = (path / "state.json").read_bytes()
    assert worker.execute(path) == 1 and calls == []
    assert (path / "state.json").read_bytes() == retained


@pytest.mark.parametrize("part", ["mean", "report", "transcript", "recording"])
def test_replay_rejects_modified_derived_evidence(tmp_path, monkeypatch, part):
    path, _ = prepared(tmp_path)
    run_pipeline(path, monkeypatch)
    record = (
        path
        / {
            "mean": "result.json",
            "report": "paired-reports.json",
            "transcript": "baseline.json",
            "recording": "result.json",
        }[part]
    )
    data = json.loads(record.read_text())
    if part == "mean":
        data["mean_judge_scores"]["candidate"] = 1.0
    elif part == "report":
        data["baselines"][0]["settings"]["context_length"] = 1
    elif part == "transcript":
        data["rows"][0]["transcript"][0][1] = "different customer input"
    else:
        data["judge_recordings"]["baseline"]["path"] = str(tmp_path)
    ws.write_json(record, data)
    with pytest.raises(ValueError):
        evaluation.inspect_evaluation(path)


def test_schema_and_cli_discovery_load_no_training_runtime(tmp_path):
    import subprocess
    import sys

    code = "import sys,json;from stateset_agents.finetune_workspace import finetune_schema;s=finetune_schema();assert not any(n in sys.modules for n in ['torch','transformers','peft']);print(json.dumps(s['local_evaluation']))"
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    schema = json.loads(result.stdout)
    assert schema["config_defaults"] == asdict(evaluation.EvaluationConfig())
    assert schema["config_schema"]["properties"]["seed"]["minimum"] == 0
    assert not schema["selection_or_promotion"]


@pytest.mark.skipif(
    sys.platform != "linux", reason="Curses evaluation uses a Linux PTY"
)
@pytest.mark.parametrize("size", [(74, 44), (110, 32)])
def test_real_terminal_edits_and_reviews_evaluation_without_submitting(tmp_path, size):
    import fcntl
    import pty
    import select
    import struct
    import subprocess
    import sys
    import termios

    run, config = make_evaluation(tmp_path)
    studio = Studio(run.parent)
    studio.handle("s")
    draft = (run.parent / "draft.json").read_bytes()
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
    captured = bytearray()

    def wait_for(text):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if text.encode() in captured:
                return
            ready, _, _ = select.select([terminal], [], [], 0.1)
            if ready:
                try:
                    captured.extend(os.read(terminal, 65536))
                except OSError:
                    break
        pytest.fail(f"Missing terminal marker {text!r}: {bytes(captured)[-1600:]!r}")

    try:
        wait_for("FINE-TUNING STUDIO")
        os.write(terminal, b"\t")
        wait_for("RUNS 1/1")
        os.write(terminal, b"j")
        wait_for("LOCAL PAIRED EVALUATION")
        os.write(terminal, b"n")
        wait_for("Held-out cases JSON")
        for value in asdict(config).values():
            os.write(terminal, b"\r\x15" + str(value).encode() + b"\r\t")
        captured.clear()
        os.write(terminal, b"r")
        wait_for("Evaluation SHA256")
        assert not (run / "evaluations").exists()
        os.write(terminal, b"\x1b")
        captured.clear()
        wait_for("n new evaluation")
        os.write(terminal, b"\x1b")
        captured.clear()
        wait_for("j evaluate")
        os.write(terminal, b"q")
        assert process.wait(timeout=15) == 0
        assert (run.parent / "draft.json").read_bytes() == draft
        assert not (run / "evaluations").exists()
        assert b"PRIVATE package" not in captured
    finally:
        os.close(terminal)
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("receipt", ["bound", "missing", "other-plan"])
def test_queue_exit_after_a_committed_evaluation_keeps_its_success(
    tmp_path, monkeypatch, receipt
):
    """A teardown or queue error after the executor committed is not a failure."""
    path, review = prepared(tmp_path)
    ws.write_json(path / "launch.lock", {"plan_sha256": review.sha256})
    monkeypatch.setattr(worker, "queue_executable", lambda: "/queue")

    def call(argv, **kwargs):
        if receipt != "missing":
            plan_sha = (
                review.sha256 if receipt == "bound" else "0" * 64
            )  # result.json binds the reviewed plan by hash
            ws.write_json(
                path / "result.json",
                {"schema_version": 1, "evaluation_plan_sha256": plan_sha},
            )
        ws.write_json(path / "state.json", {"status": "succeeded"})
        return 3

    monkeypatch.setattr(worker.subprocess, "call", call)
    assert worker.supervise(path) == 3
    state = evaluation.evaluation_status(path)["state"]
    if receipt == "bound":
        assert state["status"] == "succeeded"
        assert state["teardown_error"] == "queue_exit"
        assert state["teardown_exit_code"] == 3
    else:
        assert state["status"] == "failed" and state["exit_code"] == 3
