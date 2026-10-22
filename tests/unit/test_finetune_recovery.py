"""Exact checkpoint recovery across the studio, CLI and native trainer."""

import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as cli
from stateset_agents import finetune_recovery as recovery
from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as ws
from stateset_agents.tui import Studio


def make_parent(tmp_path, kind="sft"):
    if not sys.platform.startswith("linux"):
        pytest.skip("Published recovery fixtures require Linux worker ownership")
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = tmp_path / "source.json"
    rows = {
        "messages": [
            {"role": "user", "content": "PRIVATE CUSTOMER"},
            {"role": "assistant", "content": "PRIVATE ANSWER"},
        ]
    }
    if kind != "sft":
        rows = [
            {
                "id": "case",
                "context": "PRIVATE CUSTOMER",
                "policy_facts": ["Replace damaged items"],
                "expected_resolution": "Replacement",
                "user_responses": ["Please replace it"],
                "must_avoid": [],
            }
        ]
    data.write_text(json.dumps(rows) + "\n")
    config = ws.FinetuneConfig(
        dataset=str(data),
        method="sft" if kind == "sft" else "grpo",
        revision="a" * 40,
        steps=3,
        epochs=3,
    )
    run = ws.create_run(tmp_path / "workspace", ws.plan_run(config))
    ws.write_json(run / "state.json", {"status": "cancelled"})
    path = run / "artifacts" / "checkpoint-1"
    path.mkdir(parents=True)
    (path / "adapter_model.safetensors").write_bytes(b"synthetic checkpoint weights")
    if kind == "sft":
        ws.write_json(path / "trainer_state.json", {"global_step": 1, "epoch": 1.0})
        ws.write_json(
            path / "adapter_config.json", {"base_model_name_or_path": config.model}
        )
        for name in ["optimizer.pt", "scheduler.pt", "rng_state.pth"]:
            (path / name).write_bytes(b"synthetic optimizer fixture")
    else:
        (path / "training_state.pt").write_bytes(b"synthetic native fixture")
        ws.write_json(path / "resolved_config.json", {})
        files = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()
        }
        ws.write_json(
            path / ".stateset-checkpoint.json", {"schema_version": 1, "files": files}
        )
    recovery.publish_checkpoint(run, path, step=1)
    return run, config


def choice(run):
    return recovery.inspect_checkpoints(run)["checkpoints"][0]


def prepare(run):
    selected = choice(run)
    return recovery.prepare_resume(
        run,
        selected["checkpoint"],
        expected_recovery_sha256=selected["recovery_sha256"],
    )


def wait(studio):
    deadline = time.monotonic() + 12
    while studio.recovery_busy and time.monotonic() < deadline:
        studio.refresh()
        time.sleep(0.005)
    assert not studio.recovery_busy


@pytest.mark.parametrize("kind", ["sft", "native"])
def test_exact_snapshot_is_private_independent_and_separately_reviewed(tmp_path, kind):
    run, config = make_parent(tmp_path, kind)
    before = {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    report = recovery.inspect_checkpoints(run)
    assert not report["prepared"] and not report["training_started"]
    assert "PRIVATE" not in json.dumps(report)
    child = prepare(run)
    assert child != run and child.parent == run.parent
    assert before == {
        p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()
    }
    review = ws.review_prepared_run(child)
    assert review.plan.config == config
    assert review.plan.resume["parent_run_id"] == run.name
    assert review.plan.resume["step"] == 1
    assert ws.run_status(child)["status"] == "prepared"
    assert not (child / "launch.lock").exists()
    assert not list(run.parent.glob(".resume-*"))
    assert child.stat().st_mode & 0o077 == 0
    assert all(p.stat().st_mode & 0o077 == 0 for p in child.rglob("*") if p.is_file())
    for p in (run / "artifacts/checkpoint-1").iterdir():
        if p.is_file():
            original = child / "resume-checkpoint" / p.name
            assert p.stat().st_ino != original.stat().st_ino
    # Parent data and artifacts can disappear after preparation. The child owns
    # its exact copied input and launch review; it never reads the mutable source.
    (run / "artifacts/checkpoint-1/adapter_model.safetensors").write_bytes(
        b"parent changed"
    )
    Path(config.dataset).unlink()
    assert ws.verified_run_settings(child)[1] == config
    (child / "resume-checkpoint/adapter_model.safetensors").write_bytes(
        b"child changed"
    )
    with pytest.raises(ValueError, match="snapshot differs|publication"):
        ws.verified_run_settings(child)


@pytest.mark.parametrize(
    "mutation",
    [
        "weights",
        "receipt",
        "plan",
        "missing_optimizer",
        "symlink",
        "fifo",
        "empty_weights",
        "publication",
    ],
)
def test_invalid_recovery_stays_unavailable_and_never_creates_a_run(tmp_path, mutation):
    run, _ = make_parent(tmp_path, "native" if mutation == "publication" else "sft")
    row = choice(run)
    path = run / "artifacts/checkpoint-1"
    if mutation == "receipt":
        ws.write_json(run / "recovery/checkpoint-1.json", {"schema_version": True})
    elif mutation == "plan":
        plan = ws.read_json(run / "plan.json")
        plan["package_implementation_sha256"] = "0" * 64
        ws.write_json(run / "plan.json", plan)
    elif mutation == "missing_optimizer":
        (path / "optimizer.pt").unlink()
    elif mutation == "publication":
        (path / ".stateset-checkpoint.json").unlink()
    else:
        weights = path / "adapter_model.safetensors"
        weights.unlink()
        if mutation == "symlink":
            weights.symlink_to(run / "dataset")
        elif mutation == "fifo":
            os.mkfifo(weights)
        else:
            weights.write_bytes(b"" if mutation == "empty_weights" else b"changed")
    with pytest.raises((ValueError, OSError, KeyError)):
        recovery.prepare_resume(
            run, row["checkpoint"], expected_recovery_sha256=row["recovery_sha256"]
        )
    assert len(ws.list_runs(run.parent)) == 1
    assert not list(run.parent.glob(".resume-*"))
    if mutation != "plan":
        assert not recovery.inspect_checkpoints(run)["checkpoints"][0]["available"]


@pytest.mark.parametrize("kind", ["sft", "native"])
def test_stale_review_cannot_copy_or_fall_back_to_fresh_training(tmp_path, kind):
    run, _ = make_parent(tmp_path, kind)
    with pytest.raises(ValueError, match="checksum differs"):
        recovery.prepare_resume(run, "checkpoint-1", expected_recovery_sha256="0" * 64)
    assert len(ws.list_runs(run.parent)) == 1


@pytest.mark.parametrize(
    "condition",
    ["running", "supervisor", "execution", "unversioned", "remote", "multi_process"],
)
def test_recovery_refuses_unverifiable_parent_or_worker_lifecycle(
    tmp_path, monkeypatch, condition
):
    run, _ = make_parent(tmp_path)
    if condition == "running":
        ws.write_json(run / "state.json", {"status": "running"})
    elif condition in {"supervisor", "execution"}:
        ws.write_json(
            run / (condition + ".json"),
            {"pid": os.getpid(), "process_token": ws.process_token(os.getpid())},
        )
    elif condition == "multi_process":
        monkeypatch.setenv("WORLD_SIZE", "2")
    else:
        plan = ws.read_json(run / "plan.json")
        plan["config"]["revision"] = (
            "" if condition == "unversioned" else plan["config"]["revision"]
        )
        if condition == "remote":
            plan["config"]["provider"] = "runpod"
            plan["config"]["revision"] = ""
        ws.write_json(run / "plan.json", plan)
    with pytest.raises(ValueError):
        recovery.inspect_checkpoints(run)
    assert len(ws.list_runs(run.parent)) == 1


def test_copy_failure_is_not_published_as_a_prepared_run(tmp_path, monkeypatch):
    run, _ = make_parent(tmp_path)
    selected = choice(run)
    original = recovery._inventory

    def fail(path, *, copy_to=None):
        if copy_to is not None:
            copy_to.mkdir()
            raise OSError("synthetic disk-full copy failure")
        return original(path)

    monkeypatch.setattr(recovery, "_inventory", fail)
    with pytest.raises(OSError):
        recovery.prepare_resume(
            run, "checkpoint-1", expected_recovery_sha256=selected["recovery_sha256"]
        )
    assert len(ws.list_runs(run.parent)) == 1 and not list(run.parent.glob(".resume-*"))


@pytest.mark.parametrize("kind", ["sft", "native"])
def test_exhausted_checkpoint_does_not_extend_training_budget(tmp_path, kind):
    run, _ = make_parent(tmp_path, kind)
    if kind == "sft":
        ws.write_json(
            run / "artifacts/checkpoint-1/trainer_state.json",
            {"global_step": 1, "epoch": 3.0},
        )
        recovery.publish_checkpoint(run, run / "artifacts/checkpoint-1", step=1)
    else:
        (run / "artifacts/checkpoint-1").rename(run / "artifacts/checkpoint-3")
        (run / "recovery/checkpoint-1.json").unlink()
        recovery.publish_checkpoint(run, run / "artifacts/checkpoint-3", step=3)
    assert not choice(run)["available"]
    with pytest.raises(ValueError, match="exhausted"):
        recovery.prepare_resume(
            run,
            "checkpoint-1" if kind == "sft" else "checkpoint-3",
            expected_recovery_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "field,value", [("epochs", 4), ("learning_rate", 1e-4), ("context_length", 2048)]
)
def test_changed_child_training_settings_fail_before_model_loading(
    tmp_path, monkeypatch, field, value
):
    run, _ = make_parent(tmp_path)
    child = prepare(run)
    plan = ws.read_json(child / "plan.json")
    plan["config"][field] = value
    ws.write_json(child / "plan.json", plan)
    monkeypatch.setattr(
        worker,
        "_train_sft",
        lambda *a, **kw: pytest.fail("Changed settings cannot load a model"),
    )
    assert worker.execute(child) == 1
    assert ws.failure_info(child, ws.run_status(child))["code"] == "review_invalid"
    assert not (child / "completion.json").exists()


def test_insufficient_copy_headroom_keeps_parent_and_workspace_intact(
    tmp_path, monkeypatch
):
    run, _ = make_parent(tmp_path)
    row = choice(run)
    monkeypatch.setattr(
        recovery.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 0})()
    )
    with pytest.raises(ValueError, match="headroom"):
        recovery.prepare_resume(
            run, row["checkpoint"], expected_recovery_sha256=row["recovery_sha256"]
        )
    assert len(ws.list_runs(run.parent)) == 1
    assert not list(run.parent.glob(".resume-*"))


def test_resumed_launch_is_background_owned_exclusive_and_bound_to_review(
    tmp_path, monkeypatch
):
    run, _ = make_parent(tmp_path)
    child = prepare(run)
    studio = Studio(run.parent)
    studio.tab = "runs"
    studio.run_index = next(
        i for i, row in enumerate(studio.runs) if row["id"] == child.name
    )
    studio.handle("r")
    wait(studio)
    assert studio.overlay == "prepared-review" and studio.prepared_review.run == child
    digest = studio.prepared_review.plan_sha256
    entered, release = threading.Event(), threading.Event()
    called = []

    def launch(target, *, expected_plan_sha256):
        called.append((target, expected_plan_sha256, threading.current_thread().daemon))
        entered.set()
        assert release.wait(8)
        return 123

    monkeypatch.setattr("stateset_agents.tui.launch_run", launch)
    studio.handle("ENTER")
    try:
        assert entered.wait(3)
        studio.handle("ENTER")
        studio.handle("ESC")
        assert studio.overlay is None and studio.recovery_busy
        studio.run_index = next(
            i for i, row in enumerate(studio.runs) if row["id"] == run.name
        )
    finally:
        release.set()
        wait(studio)
    assert called == [(child, digest, False)]
    assert not (run / "launch.lock").exists()


def test_recovery_cli_returns_exact_prepare_review_without_launch(tmp_path):
    run, _ = make_parent(tmp_path)
    runner = CliRunner()
    args = ["--workspace", str(run.parent)]
    report = runner.invoke(cli.app, ["checkpoints", run.name, *args])
    assert report.exit_code == 0
    selected = json.loads(report.stdout)["result"]["checkpoints"][0]
    failed = runner.invoke(
        cli.app,
        [
            "prepare-resume",
            run.name,
            *args,
            "--checkpoint",
            "checkpoint-1",
            "--expected-recovery-sha256",
            "0" * 64,
        ],
    )
    assert (
        failed.exit_code == 1
        and json.loads(failed.stdout)["error"]["code"] == "recovery_unavailable"
    )
    prepared = runner.invoke(
        cli.app,
        [
            "prepare-resume",
            run.name,
            *args,
            "--checkpoint",
            "checkpoint-1",
            "--expected-recovery-sha256",
            selected["recovery_sha256"],
        ],
    )
    assert prepared.exit_code == 0
    result = json.loads(prepared.stdout)["result"]
    assert result["prepared"] and not result["launched"]
    assert result["plan_sha256"] == ws.read_run_plan(Path(result["run"]["path"]))[1]
    assert "PRIVATE" not in report.stdout + prepared.stdout


@pytest.mark.parametrize("size", [(72, 22), (74, 44), (110, 32)])
def test_tui_recovery_needs_prepare_and_launch_reviews_and_preserves_draft(
    tmp_path, size, monkeypatch
):
    run, _ = make_parent(tmp_path)
    studio = Studio(run.parent)
    values = dict(studio.values)
    monkeypatch.setattr(
        "stateset_agents.tui.launch_run",
        lambda *args, **kw: pytest.fail("No launch before second review"),
    )
    studio.handle("TAB")
    studio.handle("z")
    wait(studio)
    assert "checkpoint-1" in "\n".join(studio.lines(*size))
    studio.handle("ENTER")
    assert "REVIEW CHECKPOINT RECOVERY" in "\n".join(studio.lines(*size))
    studio.handle("ESC")
    assert len(ws.list_runs(run.parent)) == 1
    studio.handle("z")
    wait(studio)
    studio.handle("ENTER")
    studio.handle("ENTER")
    wait(studio)
    assert studio.overlay == "prepared-review"
    assert "RECOVERY LINEAGE" in "\n".join(
        studio._review_lines(studio.prepared_review.plan)
    )
    child = studio.prepared_review.run
    assert not (child / "launch.lock").exists()
    studio.handle("ESC")
    assert studio.values == values and len(ws.list_runs(run.parent)) == 2


def test_recovery_hashing_stays_bound_to_run_and_escape_remains_responsive(
    tmp_path, monkeypatch
):
    run, _ = make_parent(tmp_path)
    studio = Studio(run.parent)
    entered, release = threading.Event(), threading.Event()
    original = recovery.inspect_checkpoints

    def inspect(parent):
        assert parent == run
        entered.set()
        assert release.wait(8)
        return original(parent)

    monkeypatch.setattr("stateset_agents.tui.inspect_checkpoints", inspect)
    studio.handle("TAB")
    studio.handle("z")
    try:
        assert entered.wait(3)
        studio.handle("ESC")
        assert studio.overlay is None
        assert studio.recovery_busy and studio.recovery_run == run
    finally:
        release.set()
        wait(studio)
    assert not (run / "launch.lock").exists()


@pytest.mark.asyncio
async def test_native_snapshot_restores_real_optimizer_rng_and_cursor(tmp_path):
    import torch

    from tests.unit.test_multi_turn_checkpoint_accumulation import (
        accumulate,
        assert_equal,
        trainer,
    )

    run, _ = make_parent(tmp_path, "native")
    # Replace the synthetic fixture with a real CPU trainer checkpoint.
    original = trainer(run / "artifacts")
    accumulate(original, [1, 2])
    accumulate(original, [2, 3])
    original._next_episode = 2
    await original.save_checkpoint(checkpoint_name="checkpoint-1")
    recovery.publish_checkpoint(run, run / "artifacts/checkpoint-1", step=1)
    child = prepare(run)
    rng = torch.get_rng_state().clone()
    restored = trainer(child / "artifacts")
    assert restored.load_checkpoint(child / "resume-checkpoint")
    assert torch.equal(torch.get_rng_state(), rng)
    assert restored._next_episode == 2 and restored.global_step == original.global_step
    assert_equal(original.optimizer.state_dict(), restored.optimizer.state_dict())
    for vector in ([2, 4], [1, 5], [3, 2]):
        accumulate(original, vector)
        accumulate(restored, vector)
    assert_equal(original.agent.model.state_dict(), restored.agent.model.state_dict())
    assert_equal(original.optimizer.state_dict(), restored.optimizer.state_dict())
    assert_equal(original.lr_scheduler.state_dict(), restored.lr_scheduler.state_dict())
    assert torch.isfinite(restored.agent.model.weight).all()


@pytest.mark.parametrize("missing,legacy", [(True, False), (False, True)])
def test_explicit_sft_resume_never_silently_starts_fresh(tmp_path, missing, legacy):
    from stateset_agents.training.sft import run_sft

    path = tmp_path / "checkpoint-1"
    if not missing:
        path.mkdir()
        ws.write_json(path / "trainer_state.json", {"global_step": 1})
    with pytest.raises(ValueError, match="fresh fallback is disabled"):
        run_sft(
            rows=[],
            base_model="never-loaded",
            output_dir=tmp_path / "out",
            num_epochs=3,
            lora_r=4,
            lora_alpha=8,
            learning_rate=1e-5,
            max_length=32,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            resume=legacy,
            resume_from_checkpoint=path,
        )
    assert not (tmp_path / "out").exists()


def test_real_cpu_sft_stop_snapshot_and_resume_matches_uninterrupted(
    tmp_path, monkeypatch
):
    """Exercise HF/PEFT optimizer, scheduler, cursor and RNG recovery offline."""
    import copy
    import shutil

    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from stateset_agents.data.text_sft import read_text_conversations
    from stateset_agents.training import sft
    from stateset_agents.training.lineage import read_manifest

    run, config = make_parent(tmp_path)
    shutil.rmtree(run / "artifacts")
    backend = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[EOS]": 2,
                "user": 3,
                "assistant": 4,
                "PRIVATE": 5,
                "CUSTOMER": 6,
                "ANSWER": 7,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}"
    torch.manual_seed(17)
    template = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=8,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=1,
        )
    )

    def load(*args, **kwargs):
        torch.manual_seed(42)
        return copy.deepcopy(template)

    build = sft.build_training_arguments

    def arguments(cls, **kwargs):
        return build(
            cls,
            **{
                **kwargs,
                "bf16": False,
                "use_cpu": True,
                "disable_tqdm": True,
                "logging_steps": 1,
            },
        )

    monkeypatch.setattr(sft, "load_base_model_for_sft", load)
    monkeypatch.setattr(sft, "build_training_arguments", arguments)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: copy.deepcopy(tokenizer),
    )
    ws.write_json(run / "state.json", {"status": "running"})
    ws.request_stop(run, expected_plan_sha256=ws.read_run_plan(run)[1])
    worker._train_sft(run, config)
    assert (
        ws.read_json(run / "artifacts/checkpoint-1/trainer_state.json")["global_step"]
        == 1
    )
    ws.write_json(run / "state.json", {"status": "cancelled"})
    child = prepare(run)
    worker._train_sft(child, config)
    final = ws.read_json(child / "artifacts/checkpoint-3/trainer_state.json")
    assert final["global_step"] == 3 and final["epoch"] == 3
    assert not (child / "artifacts/checkpoint-1").exists()
    assert read_manifest(child / "artifacts")["hyperparameters"]["resumed"]
    uninterrupted = tmp_path / "uninterrupted"
    sft.run_sft(
        rows=read_text_conversations(str(run / "dataset")),
        base_model=config.model,
        model_revision=config.revision,
        output_dir=uninterrupted,
        num_epochs=3,
        lora_r=config.lora_rank,
        lora_alpha=2 * config.lora_rank,
        learning_rate=config.learning_rate,
        max_length=config.context_length,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
    )
    before = load_file(uninterrupted / "adapter_model.safetensors")
    after = load_file(child / "artifacts/adapter_model.safetensors")
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key], after[key]) for key in before)


@pytest.mark.skipif(
    sys.platform != "linux", reason="Real curses recovery uses Linux PTY"
)
@pytest.mark.parametrize("width,height", [(74, 44), (110, 32)])
def test_real_terminal_recovery_prepares_then_escapes_launch_review(
    tmp_path, width, height
):
    import fcntl
    import pty
    import select
    import struct
    import subprocess
    import termios

    run, config = make_parent(tmp_path)
    ws.write_json(run.parent / "draft.json", asdict(config))
    draft = (run.parent / "draft.json").read_bytes()
    terminal, slave = pty.openpty()
    fcntl.ioctl(terminal, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "stateset_agents.cli",
            "tui",
            "--workspace",
            str(run.parent),
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
        env=dict(os.environ, TERM="xterm-256color"),
    )
    os.close(slave)
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
        pytest.fail(
            f"Missing recovery terminal output {text!r}: {bytes(captured)[-1200:]!r}"
        )

    try:
        wait_for("FINE-TUNING STUDIO")
        os.write(terminal, b"\t")
        wait_for("RUNS 1/1")
        os.write(terminal, b"z")
        wait_for("exact local checkpoint")
        wait_for("choices verified")
        os.write(terminal, b"\r")
        wait_for("prepare only")
        captured.clear()
        os.write(terminal, b"\x1b")
        wait_for("r review prepared")
        assert len(ws.list_runs(run.parent)) == 1
        captured.clear()
        os.write(terminal, b"z")
        wait_for("choices verified")
        os.write(terminal, b"\r")
        wait_for("prepare only")
        captured.clear()
        os.write(terminal, b"\r")
        wait_for("same ID, retained snapshot")
        children = [
            Path(row["path"])
            for row in ws.list_runs(run.parent)
            if row["id"] != run.name
        ]
        assert len(children) == 1 and ws.run_status(children[0])["status"] == "prepared"
        captured.clear()
        os.write(terminal, b"\x1b")
        wait_for("r review prepared")
        os.write(terminal, b"q")
        assert process.wait(timeout=15) == 0
        assert not any((path / "launch.lock").exists() for path in [run, *children])
        assert not (children[0] / "stop.json").exists()
        assert (run.parent / "draft.json").read_bytes() == draft
        assert b"PRIVATE CUSTOMER" not in captured
    finally:
        os.close(terminal)
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.asyncio
async def test_native_worker_uses_the_exact_checkpoint_and_original_budget(
    tmp_path, monkeypatch
):
    run, config = make_parent(tmp_path, "native")
    child = prepare(run)
    calls = []

    class Agent:
        def __init__(self, config):
            self.config = config

        async def initialize(self):
            pass

    class Trainer:
        def __init__(self, *args, **kwargs):
            self.config = kwargs["config"]
            self.global_step = 0
            calls.append(self.config)

        async def initialize(self):
            pass

        async def train(self):
            assert self.config.resume_from_checkpoint == str(
                child / "resume-checkpoint"
            )
            assert self.config.num_episodes == config.steps
            assert self.config.num_generations == config.generations
            self.global_step = config.steps

    monkeypatch.setattr("stateset_agents.core.agent.MultiTurnAgent", Agent)
    monkeypatch.setattr(
        "stateset_agents.training.multi_turn_trainer.MultiTurnGRPOTrainer", Trainer
    )
    path, updates = await worker._train_rl(child, config)
    assert updates == config.steps and path == child / "artifacts/checkpoint-3"
    assert len(calls) == 1


def test_live_but_uninspectable_owner_is_not_treated_as_a_dead_worker(
    tmp_path, monkeypatch
):
    run, _ = make_parent(tmp_path)
    ws.write_json(
        run / "execution.json", {"pid": os.getpid(), "process_token": "unverifiable"}
    )
    monkeypatch.setattr(ws, "process_token", lambda _: None)
    with pytest.raises(ValueError, match="Cannot verify"):
        recovery.inspect_checkpoints(run)


def test_checkpoint_byte_and_file_limits_fail_before_copy(tmp_path, monkeypatch):
    run, _ = make_parent(tmp_path)
    selected = choice(run)
    monkeypatch.setattr(recovery, "MAX_CHECKPOINT_BYTES", 1)
    with pytest.raises(ValueError, match="bytes exceed"):
        recovery.prepare_resume(
            run,
            selected["checkpoint"],
            expected_recovery_sha256=selected["recovery_sha256"],
        )
    monkeypatch.setattr(recovery, "MAX_CHECKPOINT_BYTES", 1024**4)
    monkeypatch.setattr(recovery, "MAX_CHECKPOINT_FILES", 1)
    with pytest.raises(ValueError, match="file count"):
        recovery.prepare_resume(
            run,
            selected["checkpoint"],
            expected_recovery_sha256=selected["recovery_sha256"],
        )
    assert len(ws.list_runs(run.parent)) == 1
