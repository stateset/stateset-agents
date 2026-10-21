"""Published checkpoint validation is separate from recovery and final test evidence."""

import copy
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as cli
from stateset_agents import finetune_evaluation as evaluation
from stateset_agents import finetune_quality as quality
from stateset_agents import finetune_recovery as recovery
from stateset_agents import finetune_worker as training_worker
from stateset_agents import finetune_workspace as ws
from stateset_agents.evaluation.agent_runs import compare_runs, content_hash
from stateset_agents.finetune_evaluation_tui import EvaluationPanel
from stateset_agents.tui import Studio
from tests.unit.test_finetune_evaluation import run_pipeline, wait
from tests.unit.test_finetune_recovery import make_parent


def published(tmp_path, kind="sft", state="cancelled"):
    run, config = make_parent(tmp_path, kind)
    path = run / "artifacts/checkpoint-1"
    if kind != "sft":
        ws.write_json(
            path / "adapter_config.json", {"base_model_name_or_path": config.model}
        )
        files = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.iterdir()
            if p.name != ".stateset-checkpoint.json"
        }
        ws.write_json(
            path / ".stateset-checkpoint.json", {"schema_version": 1, "files": files}
        )
        recovery.publish_checkpoint(run, path, step=1)
    ws.write_json(run / "state.json", {"status": state})
    if state == "succeeded":
        ws.write_json(run / "completion.json", {"output_dir": str(path)})
    source = tmp_path / "validation.json"
    ws.write_json(
        source,
        {
            "schema_version": 1,
            "split": "validation",
            "provenance": {
                "source": "reviewed",
                "reviewer": "Synthetic fixture",
                "validation_is_fresh": True,
            },
            "cases": [
                {
                    "id": f"validation-{index}",
                    "context": "Help with deliveries.",
                    "user_turns": [f"PRIVATE damaged parcel {index}"],
                    "policy_facts": ["Replace damaged deliveries."],
                    "expected_resolution": "Offer replacement.",
                    "must_avoid": [],
                }
                for index in range(2)
            ],
        },
    )
    settings = evaluation.EvaluationConfig(
        cases=str(source),
        judge_model="fixture/judge",
        judge_revision="b" * 40,
        checkpoint="checkpoint-1",
        split="validation",
    )
    return run, config, settings


@pytest.mark.parametrize("kind", ["sft", "native"])
def test_target_discovery_reads_only_receipt_metadata_and_no_model_bytes(
    tmp_path, monkeypatch, kind
):
    run, _, _ = published(tmp_path, kind)

    def forbidden(*args, **kwargs):
        pytest.fail("Discovery must not hash checkpoint bytes")

    monkeypatch.setattr(recovery, "_inventory", forbidden)
    monkeypatch.setattr(evaluation, "_inventory", forbidden)
    result = evaluation.evaluation_targets(run)
    assert result["bytes_verified"] is False
    assert result["targets"] == [
        {
            "checkpoint": "checkpoint-1",
            "step": 1,
            "kind": kind,
            "available": True,
            "bytes_verified": False,
        }
    ]
    assert "PRIVATE" not in json.dumps(result)
    cli_result = CliRunner().invoke(
        cli.app, ["evaluation-targets", run.name, "--workspace", str(run.parent)]
    )
    assert cli_result.exit_code == 0, cli_result.stdout
    assert json.loads(cli_result.stdout)["result"] == result
    assert not (run / "evaluations").exists()


@pytest.mark.parametrize("kind", ["sft", "native"])
def test_review_binds_selected_weights_receipt_and_stopped_parent_without_writes(
    tmp_path, kind
):
    run, _, config = published(tmp_path, kind)
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    review = evaluation.plan_evaluation(run, config)
    binding = review.plan["binding"]
    assert binding["checkpoint"] == "checkpoint-1" and binding["checkpoint_step"] == 1
    assert binding["artifact_path"] == str(run / "artifacts/checkpoint-1")
    assert binding["checkpoint_receipt_sha256"] == content_hash(
        ws.read_json(run / "recovery/checkpoint-1.json")
    )
    assert before == {
        p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()
    }
    assert not (run / "completion.json").exists()
    path = evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256)
    assert evaluation.read_evaluation(path).sha256 == review.sha256
    assert evaluation.evaluation_status(path)["split"] == "validation"
    assert evaluation.evaluation_status(path)["checkpoint"] == "checkpoint-1"


@pytest.mark.parametrize(
    "mutation",
    ["weights", "optimizer", "receipt", "progress", "missing_adapter", "parent_state"],
)
def test_selected_checkpoint_changes_fail_before_inference(tmp_path, mutation):
    run, _, config = published(tmp_path)
    review = evaluation.plan_evaluation(run, config)
    path = run / "artifacts/checkpoint-1"
    if mutation == "weights":
        (path / "adapter_model.safetensors").write_bytes(b"changed")
    elif mutation == "optimizer":
        (path / "optimizer.pt").write_bytes(b"changed")
    elif mutation == "receipt":
        record = ws.read_json(run / "recovery/checkpoint-1.json")
        record["dataset_sha256"] = "0" * 64
        ws.write_json(run / "recovery/checkpoint-1.json", record)
    elif mutation == "progress":
        ws.write_json(path / "trainer_state.json", {"global_step": 2, "epoch": 2.0})
    elif mutation == "parent_state":
        ws.write_json(run / "state.json", {"status": "running"})
    else:
        (path / "adapter_model.safetensors").unlink()
    with pytest.raises(ValueError):
        evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256)
    assert not (run / "evaluations").exists()


@pytest.mark.parametrize("kind", ["sft", "native"])
def test_exhausted_checkpoint_can_be_evaluated_but_not_resumed(tmp_path, kind):
    run, _, config = published(tmp_path, kind)
    path = run / "artifacts/checkpoint-1"
    target = path.with_name("checkpoint-3")
    path.rename(target)
    if kind == "sft":
        ws.write_json(target / "trainer_state.json", {"global_step": 3, "epoch": 3.0})
    recovery.publish_checkpoint(run, target, step=3)
    config = replace(config, checkpoint="checkpoint-3")
    assert (
        evaluation.plan_evaluation(run, config).plan["binding"]["checkpoint_step"] == 3
    )
    rows = recovery.inspect_checkpoints(run)["checkpoints"]
    assert all(not row["available"] for row in rows)


def test_older_trainer_code_can_be_inferred_without_restoring_its_state(tmp_path):
    run, _, config = published(tmp_path)
    plan = ws.read_json(run / "plan.json")
    plan["package_implementation_sha256"] = "0" * 64
    ws.write_json(run / "plan.json", plan)
    recovery.publish_checkpoint(run, run / "artifacts/checkpoint-1", step=1)
    assert (
        evaluation.plan_evaluation(run, config).plan["binding"]["checkpoint"]
        == "checkpoint-1"
    )
    with pytest.raises(ValueError, match="code changed"):
        recovery.inspect_checkpoints(run)


def test_live_parent_owner_prevents_checkpoint_evaluation(tmp_path):
    run, _, config = published(tmp_path)
    ws.write_json(
        run / "execution.json",
        {"pid": os.getpid(), "process_token": ws.process_token(os.getpid())},
    )
    with pytest.raises(ValueError, match="Wait for"):
        evaluation.plan_evaluation(run, config)
    assert not (run / "evaluations").exists()


@pytest.mark.parametrize(
    "checkpoint",
    ["../checkpoint-1", "checkpoint-0", "checkpoint-1/..", "/tmp/weights", False],
)
def test_checkpoint_names_and_validation_partition_fail_closed(checkpoint):
    with pytest.raises(ValueError):
        evaluation.EvaluationConfig(
            cases="unused", checkpoint=checkpoint, split="validation"
        ).validate()


def test_selection_rejects_fresh_test_cases_and_mismatched_partition_provenance(
    tmp_path,
):
    run, _, config = published(tmp_path)
    with pytest.raises(ValueError, match="validation"):
        evaluation.plan_evaluation(run, replace(config, split="test"))
    source = Path(config.cases)
    document = ws.read_json(source)
    document["split"] = "test"
    ws.write_json(source, document)
    with pytest.raises(ValueError):
        evaluation.plan_evaluation(run, config)
    document["split"] = "validation"
    document["provenance"] = {
        "source": "reviewed",
        "reviewer": "Synthetic",
        "test_is_fresh": True,
    }
    ws.write_json(source, document)
    with pytest.raises(ValueError):
        evaluation.plan_evaluation(run, config)
    assert not (run / "evaluations").exists()


@pytest.mark.parametrize("problem", ["selection", "partition", "provenance"])
def test_agent_protocol_errors_are_actionable_and_do_not_echo_case_values(
    tmp_path, problem
):
    run, _, config = published(tmp_path)
    if problem == "selection":
        config = replace(config, split="test")
        expected = "evaluation_checkpoint_partition"
    else:
        source = Path(config.cases)
        value = ws.read_json(source)
        if problem == "partition":
            value["split"] = "PRIVATE_CASE_VALUE"
            expected = "evaluation_case_partition"
        else:
            value["provenance"] = {"reviewer": "PRIVATE_CASE_VALUE"}
            expected = "evaluation_provenance"
        ws.write_json(source, value)
    config_path = tmp_path / "evaluation.json"
    ws.write_json(config_path, asdict(config))
    result = CliRunner().invoke(
        cli.app,
        [
            "evaluation-plan",
            run.name,
            "--workspace",
            str(run.parent),
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 1
    error = json.loads(result.stdout)["error"]
    assert error["code"] == expected
    assert error["diagnostic"]["expected_partition"] == "validation"
    assert "PRIVATE" not in result.stdout
    assert not (run / "evaluations").exists()


def test_checkpoint_schema_requires_validation_and_preserves_portable_test_default():
    jsonschema = pytest.importorskip("jsonschema")
    schema = evaluation.evaluation_schema()["config_schema"]
    values = asdict(
        evaluation.EvaluationConfig(
            cases="not-read.json", judge_model="fixture/judge", judge_revision="b" * 40
        )
    )
    jsonschema.validate(values, schema)
    values["checkpoint"] = "checkpoint-1"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(values, schema)
    values["split"] = "validation"
    jsonschema.validate(values, schema)


def test_corrupt_receipt_is_not_listed_as_reviewable_without_hashing(
    tmp_path, monkeypatch
):
    run, _, _ = published(tmp_path)
    path = run / "recovery/checkpoint-1.json"
    value = ws.read_json(path)
    value["checkpoint_sha256"] = ["PRIVATE"]
    ws.write_json(path, value)
    monkeypatch.setattr(
        recovery,
        "_inventory",
        lambda *args, **kwargs: pytest.fail("Discovery cannot hash"),
    )
    rows = evaluation.evaluation_targets(run)["targets"]
    assert rows == [
        {"checkpoint": "checkpoint-1", "available": False, "bytes_verified": False}
    ]


@pytest.mark.parametrize("state", ["cancelled", "succeeded"])
def test_validation_pipeline_replays_selected_artifact_and_cannot_be_quality_imported(
    tmp_path, monkeypatch, state
):
    run, _, config = published(tmp_path, state=state)
    review = evaluation.plan_evaluation(run, config)
    path = evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256)
    run_pipeline(path, monkeypatch)
    result = evaluation.inspect_evaluation(path)
    assert result["split"] == "validation" and result["checkpoint"] == "checkpoint-1"
    assert result["judge_replay_verified"] and not result["quality_gain_verified"]
    assert not result["summary"]["recorded_study_passed"]
    assert not result["summary"]["study_gates"]["held_out_test_partition"]
    assert (
        result["summary"]["artifact_sha256"]
        == review.plan["binding"]["artifact_sha256"]
    )
    assert result["summary"]["mean_success_gain"] == -1
    with pytest.raises(ValueError):
        quality.record_quality(run, path / "paired-reports.json")
    assert not (run / "quality.json").exists()
    reports = ws.read_json(path / "paired-reports.json")
    with pytest.raises(ValueError, match="test report"):
        compare_runs(reports["baselines"], reports["candidates"])


def test_replay_rejects_extra_unjudged_seed_reports(tmp_path, monkeypatch):
    run, _, config = published(tmp_path)
    review = evaluation.plan_evaluation(run, config)
    path = evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256)
    run_pipeline(path, monkeypatch)
    reports = ws.read_json(path / "paired-reports.json")
    reports["baselines"].append(copy.deepcopy(reports["baselines"][0]))
    reports["candidates"].append(copy.deepcopy(reports["candidates"][0]))
    ws.write_json(path / "paired-reports.json", reports)
    with pytest.raises(ValueError, match="exactly one"):
        evaluation.inspect_evaluation(path)


@pytest.mark.parametrize("size", [(72, 22), (74, 44), (110, 32)])
def test_target_picker_preserves_training_and_switches_selection_to_validation(
    tmp_path,
    size,
):
    run, _, _ = published(tmp_path, state="succeeded")
    studio = Studio(run.parent)
    studio.handle("s")
    draft = (run.parent / "draft.json").read_bytes()
    values = dict(studio.values)
    panel = EvaluationPanel(run)
    panel.mode = "form"
    studio.evaluation_panel, studio.overlay = panel, "evaluation"
    studio.handle("c")
    wait(panel)
    assert panel.mode == "targets"
    assert "receipt metadata" in "\n".join(studio.lines(*size))
    studio.handle("DOWN")
    studio.handle("ENTER")
    assert panel.mode == "form"
    assert panel.values["checkpoint"] == "checkpoint-1"
    assert panel.values["split"] == "validation"
    assert studio.values == values and (run.parent / "draft.json").read_bytes() == draft
    assert not (run / "evaluations").exists()
    studio.handle("c")
    wait(panel)
    studio.handle("ESC")
    assert panel.mode == "form" and panel.values["checkpoint"] == "checkpoint-1"


def test_target_list_keeps_last_selection_visible_in_a_small_pane(
    tmp_path, monkeypatch
):
    run, _, _ = published(tmp_path)
    studio = Studio(run.parent)
    panel = EvaluationPanel(run)
    panel.mode = "form"
    studio.evaluation_panel, studio.overlay = panel, "evaluation"
    monkeypatch.setattr(
        "stateset_agents.finetune_evaluation_tui.evaluation_targets",
        lambda _: {
            "targets": [
                {
                    "checkpoint": f"checkpoint-{step}",
                    "available": True,
                    "bytes_verified": False,
                }
                for step in range(1, 101)
            ]
        },
    )
    studio.handle("c")
    wait(panel)
    studio.handle("END")
    assert "checkpoint-100" in "\n".join(studio.lines(72, 22))
    studio.handle("ESC")
    assert panel.mode == "form" and panel.values["checkpoint"] == ""


def test_selected_checkpoint_loads_real_cpu_safetensors_instead_of_final_weights(
    tmp_path, monkeypatch
):
    import torch
    from peft import LoraConfig, get_peft_model

    from stateset_agents.finetune_evaluation_worker import LocalGenerator
    from tests.sft_fixtures import write_tiny_text_policy

    run, config, settings = published(tmp_path)
    cache = tmp_path / "cache"
    snapshot = (
        cache
        / ("models--" + config.model.replace("/", "--"))
        / "snapshots"
        / config.revision
    )
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    base = write_tiny_text_policy(snapshot)
    model = get_peft_model(
        base,
        LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM"),
    )
    for settings_config in model.peft_config.values():
        settings_config.base_model_name_or_path = config.model
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_A" in name:
                value.fill_(0.125)
            if "lora_B" in name:
                value.zero_()
    checkpoint = run / "artifacts/checkpoint-1"
    model.save_pretrained(checkpoint)
    recovery.publish_checkpoint(run, checkpoint, step=1)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_A" in name:
                value.fill_(0.75)
    final = run / "artifacts/final"
    model.save_pretrained(final)
    ws.write_json(run / "state.json", {"status": "succeeded"})
    ws.write_json(run / "completion.json", {"output_dir": str(final)})
    review = evaluation.plan_evaluation(run, settings)
    assert review.plan["binding"]["artifact_path"] == str(checkpoint)

    def forbidden(*args, **kwargs):
        pytest.fail("Inference may not deserialize training state or pickle weights")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    selected = LocalGenerator(
        config.model,
        config.revision,
        adapter=Path(review.plan["binding"]["artifact_path"]),
    )
    last = LocalGenerator(config.model, config.revision, adapter=final)
    left = [
        value for name, value in selected.model.named_parameters() if "lora_A" in name
    ]
    right = [value for name, value in last.model.named_parameters() if "lora_A" in name]
    assert left and all(torch.all(value == 0.125) for value in left)
    assert right and all(torch.all(value == 0.75) for value in right)
    baseline = LocalGenerator(config.model, config.revision)
    messages = [{"role": "user", "content": "reply"}]
    assert selected.generate(messages, tokens=4, context=64) == baseline.generate(
        messages, tokens=4, context=64
    )


def test_stopping_unlaunched_evaluation_is_terminal_and_never_submitted(
    tmp_path, monkeypatch
):
    run, _, config = published(tmp_path)
    review = evaluation.plan_evaluation(run, config)
    path = evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256)
    evaluation.stop_evaluation(path, expected_plan_sha256=review.sha256)
    assert evaluation.evaluation_status(path)["state"]["status"] == "cancelled"
    monkeypatch.setattr(evaluation, "queue_executable", lambda: "/queue")
    with pytest.raises(ValueError, match="Only a prepared"):
        evaluation.launch_evaluation(path, expected_plan_sha256=review.sha256)
    assert not (path / "launch.lock").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint", {"PRIVATE": True}),
        ("checkpoint", "../PRIVATE"),
        ("split", ["PRIVATE"]),
    ],
)
def test_malformed_target_metadata_is_unreadable_without_crashing_the_ui(
    tmp_path, field, value
):
    run, _, config = published(tmp_path)
    review = evaluation.plan_evaluation(run, config)
    path = evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256)
    plan = ws.read_json(path / "plan.json")
    plan["config"][field] = value
    ws.write_json(path / "plan.json", plan)
    rows = evaluation.list_evaluations(run)
    assert rows == [{"evaluation_id": path.name, "state": {"status": "unreadable"}}]
    studio = Studio(run.parent)
    studio.evaluation_panel = panel = EvaluationPanel(run)
    studio.overlay = "evaluation"
    panel.handle("ENTER")
    screen = "\n".join(studio.lines(74, 44))
    assert "unreadable" in screen and "PRIVATE" not in screen


@pytest.mark.skipif(sys.platform != "linux", reason="Curses picker uses a Linux PTY")
@pytest.mark.parametrize("size", [(74, 44), (110, 32)])
def test_real_terminal_selects_checkpoint_and_reviews_without_submitting(
    tmp_path, size
):
    import fcntl
    import pty
    import select
    import struct
    import subprocess
    import termios

    run, _, config = published(tmp_path)
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
    full = bytearray()

    def wait_for(text):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if text.encode() in captured:
                return
            ready, _, _ = select.select([terminal], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(terminal, 65536)
                except OSError:
                    break
                captured.extend(chunk)
                full.extend(chunk)
        pytest.fail(f"Missing terminal marker {text!r}: {bytes(captured)[-1600:]!r}")

    def send(keys, marker):
        captured.clear()
        os.write(terminal, keys)
        wait_for(marker)

    try:
        wait_for("FINE-TUNING STUDIO")
        send(b"\t", "RUNS 1/1")
        send(b"j", "LOCAL PAIRED EVALUATION")
        send(b"n", "Held-out cases JSON")
        send(b"c", "CHOOSE EVALUATION TARGET")
        send(b"\x1b", "Held-out cases JSON")
        send(b"c", "CHOOSE EVALUATION TARGET")
        send(b"\r", "Evaluation target selected")
        for value in asdict(config).values():
            os.write(terminal, b"\r\x15" + str(value).encode() + b"\r\t")
        send(b"r", "Evaluation SHA256")
        assert not (run / "evaluations").exists()
        send(b"\x1b", "n new evaluation")
        send(b"\x1b", "j evaluate")
        os.write(terminal, b"q")
        assert process.wait(timeout=15) == 0
        assert (run.parent / "draft.json").read_bytes() == draft
        assert not (run / "evaluations").exists()
        assert b"PRIVATE damaged" not in full
    finally:
        os.close(terminal)
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize(
    "values,valid",
    [
        ({"method": "grpo", "checkpoint_every": 0}, True),
        ({"method": "grpo", "checkpoint_every": 2, "steps": 5}, True),
        ({"method": "grpo", "checkpoint_every": -1}, False),
        ({"method": "grpo", "checkpoint_every": True}, False),
        ({"method": "sft", "checkpoint_every": 1}, False),
        ({"method": "grpo", "checkpoint_every": 1, "steps": 101}, False),
        ({"method": "sft", "provider": "runpod", "checkpoint_every": 1}, False),
    ],
)
def test_periodic_checkpoint_settings_validate_backend_and_bounded_snapshot_count(
    values, valid
):
    config = ws.FinetuneConfig(dataset="not-read", **values)
    _, issues = ws.check_config_values(asdict(config))
    assert (not issues) is valid
    assert all(issue.field == "checkpoint_every" for issue in issues)
    schema = ws.finetune_schema()["config_schema"]["properties"]["checkpoint_every"]
    assert schema["minimum"] == 0 and "exclusiveMinimum" not in schema


@pytest.mark.asyncio
@pytest.mark.parametrize("cadence,expected", [(0, [3]), (1, [1, 2, 3]), (2, [2, 3])])
async def test_worker_periodic_cadence_uses_real_cpu_commits_and_published_state(
    tmp_path, monkeypatch, cadence, expected
):
    import torch

    from tests.unit.test_multi_turn_checkpoint_accumulation import accumulate, trainer

    if not sys.platform.startswith("linux"):
        pytest.skip("Native worker publication requires Linux ownership")
    data = tmp_path / "train.json"
    data.write_text(
        json.dumps(
            [
                {
                    "id": "train",
                    "context": "Help.",
                    "policy_facts": ["Replace broken items."],
                    "expected_resolution": "Replacement",
                    "user_responses": ["Yes."],
                }
            ]
        )
    )
    config = ws.FinetuneConfig(
        dataset=str(data),
        revision="a" * 40,
        method="grpo",
        steps=3,
        checkpoint_every=cadence,
    )
    run = ws.create_run(tmp_path / "workspace", ws.plan_run(config))
    ws.write_json(run / "state.json", {"status": "running"})
    captured = []

    class Agent:
        def __init__(self, config):
            self.config = config

        async def initialize(self):
            pass

    def build(*args, config, callbacks, **kwargs):
        value = trainer(run / "artifacts", steps=1)
        value.config.save_steps = config.save_steps
        value.config.eval_steps = config.eval_steps
        value.config.logging_steps = config.logging_steps
        value.callbacks = callbacks
        initial = value.agent.model.weight.detach().clone()

        async def initialize():
            pass

        async def train():
            for index in range(3):
                previous = value.global_step
                accumulate(value, [index + 1, 2])
                value._next_episode = index + 1
                await value._report_optimizer_update(
                    value._optimizer_update_receipt(previous, "batch"), [], []
                )
            await value.save_checkpoint()
            assert not torch.equal(initial, value.agent.model.weight)

        value.initialize, value.train = initialize, train
        captured.append(value)
        return value

    monkeypatch.setattr("stateset_agents.core.agent.MultiTurnAgent", Agent)
    monkeypatch.setattr(
        "stateset_agents.training.multi_turn_trainer.MultiTurnGRPOTrainer", build
    )
    path, updates = await training_worker._train_rl(run, config)
    assert updates == 3 and path.name == "checkpoint-3"
    receipts = list((run / "recovery").glob("*.json"))
    assert sorted(int(p.stem.rsplit("-", 1)[1]) for p in receipts) == expected
    assert ws.read_json(run / "state.json")["committed_updates"] == 3
    ws.write_json(run / "state.json", {"status": "succeeded"})
    for step in expected:
        selected = recovery.inspect_evaluation_checkpoint(run, f"checkpoint-{step}")
        assert (
            selected["step"] == step
            and selected["checkpoint_sha256"]
            == recovery._inventory(run / "artifacts" / f"checkpoint-{step}")["sha256"]
        )
