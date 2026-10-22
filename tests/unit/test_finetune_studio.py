"""Terminal studio lifecycle, input binding, and failure semantics without GPUs."""

import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace
from stateset_agents.finetune_workspace import (
    FinetuneConfig,
    create_run,
    list_runs,
    log_tail,
    plan_run,
    read_json,
    run_status,
    write_json,
)
from stateset_agents.tui import Studio


@pytest.fixture
def settings(tmp_path):
    dataset = tmp_path / "chat with spaces.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Return policy?"},
                    {
                        "role": "assistant",
                        "content": "Returns are allowed for 30 days.",
                    },
                ]
            }
        )
        + "\n"
    )
    return FinetuneConfig(dataset=str(dataset))


@pytest.mark.parametrize(
    "field,value",
    [
        ("method", "imaginary"),
        ("provider", "unknown"),
        ("steps", 1.5),
        ("steps", True),
        ("learning_rate", float("nan")),
        ("memory_gb", float("inf")),
        ("kl_coefficient", 0),
        ("max_cost_usd", -1),
        ("model", "stub://demo"),
        ("revision", "main"),
        ("lora_rank", 0),
    ],
)
def test_invalid_settings_cannot_reach_provisioning(settings, field, value):
    with pytest.raises(ValueError):
        plan_run(replace(settings, **{field: value}))


def test_runpod_rl_and_unenforceable_revision_are_rejected(settings):
    with pytest.raises(ValueError, match="RunPod SFT"):
        plan_run(replace(settings, method="gspo", provider="runpod"))
    with pytest.raises(ValueError, match="cannot enforce a revision"):
        plan_run(replace(settings, revision="a" * 40, provider="runpod"))


def test_review_freezes_exact_data_and_private_run(settings, tmp_path):
    plan = plan_run(settings)
    original = Path(settings.dataset).read_bytes()
    description = plan.describe()
    Path(settings.dataset).write_text("changed after review")
    run = create_run(tmp_path / "workspace", plan)
    assert (run / "dataset").read_bytes() == original
    assert read_json(run / "plan.json") == description
    assert read_json(run / "state.json")["status"] == "prepared"
    if os.name != "nt":
        assert run.stat().st_mode & 0o077 == 0
        assert (run / "dataset").stat().st_mode & 0o077 == 0
        assert (run / "plan.json").stat().st_mode & 0o077 == 0
    assert worker._settings(run)[1] == settings


def test_malformed_row_is_not_silently_skipped(settings):
    with Path(settings.dataset).open("a") as stream:
        stream.write("not json\n")
    with pytest.raises(ValueError, match=":2:"):
        plan_run(settings)


def test_worker_rejects_changed_snapshot_or_code(settings, tmp_path):
    run = create_run(tmp_path / "workspace", plan_run(settings))
    (run / "dataset").write_text("replaced")
    with pytest.raises(ValueError, match="dataset bytes changed"):
        worker._settings(run)
    (run / "dataset").write_bytes(Path(settings.dataset).read_bytes())
    plan = read_json(run / "plan.json")
    plan["package_implementation_sha256"] = "0" * 64
    write_json(run / "plan.json", plan)
    with pytest.raises(ValueError, match="Package code changed"):
        worker._settings(run)


def test_support_scenarios_require_facts_and_customer_replies(settings):
    row = {
        "id": "case",
        "context": "My parcel arrived damaged.",
        "policy_facts": ["Replace damaged items free."],
        "expected_resolution": "Offer a replacement.",
        "user_responses": ["Please replace it."],
        "must_avoid": [],
    }
    path = Path(settings.dataset)
    path.write_text(json.dumps([row]))
    assert plan_run(replace(settings, method="gspo")).examples == 1
    row.pop("policy_facts")
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="policy_facts"):
        plan_run(replace(settings, method="gspo"))


def test_launch_is_detached_shell_free_and_exclusive(settings, tmp_path, monkeypatch):
    run = create_run(tmp_path / "workspace", plan_run(settings))
    calls = []

    def start(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(workspace.subprocess, "Popen", start)
    assert workspace.launch_run(run) == 123
    argv, kwargs = calls[0]
    assert argv[-2:] == ["--supervise", str(run)]
    assert kwargs["start_new_session"] and "shell" not in kwargs
    with pytest.raises(FileExistsError):
        workspace.launch_run(run)
    assert len(calls) == 1


def _process_state(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return None  # reaped: no process table entry remains


@pytest.mark.skipif(sys.platform != "linux", reason="Reads the Linux process table")
def test_finished_supervisor_is_reaped_on_the_next_status_refresh(
    settings, tmp_path, monkeypatch
):
    """The long-lived studio must not accumulate zombie supervisors."""
    import subprocess

    run = create_run(tmp_path / "workspace", plan_run(settings))
    real_popen = subprocess.Popen

    def short_lived(argv, **kwargs):
        return real_popen([sys.executable, "-c", "pass"], **kwargs)

    monkeypatch.setattr(workspace.subprocess, "Popen", short_lived)
    pid = workspace.launch_run(run)
    deadline = time.monotonic() + 10
    while _process_state(pid) not in {"Z", None} and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _process_state(pid) in {"Z", None}
    workspace.run_status(run)
    assert _process_state(pid) is None


def test_reaping_skips_handles_that_cannot_be_polled(monkeypatch):
    finished = SimpleNamespace(poll=lambda: 0)
    running = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(workspace, "_DETACHED_CHILDREN", [])
    for handle in (finished, running, SimpleNamespace(pid=1)):
        workspace.retain_child(handle)
    workspace.reap_children()
    assert workspace._DETACHED_CHILDREN == [running]


@pytest.mark.parametrize(
    "exit_code,receipt,expected",
    [(0, True, "succeeded"), (0, False, "failed"), (2, True, "failed")],
)
def test_supervisor_uses_queue_preserves_args_and_requires_receipt(
    settings, tmp_path, monkeypatch, exit_code, receipt, expected
):
    run = create_run(tmp_path / "workspace", plan_run(settings))
    monkeypatch.setattr(worker, "queue_executable", lambda: "/queue/buildr")
    monkeypatch.setattr(worker.os, "umask", lambda _: 0)
    captured = {}

    def start(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        if receipt:
            write_json(run / "completion.json", {"output_dir": "artifacts"})
        return SimpleNamespace(wait=lambda **_: exit_code)

    monkeypatch.setattr(worker.subprocess, "Popen", start)
    code = worker.supervise(run)
    assert read_json(run / "state.json")["status"] == expected
    assert (code == 0) == (expected == "succeeded")
    assert captured["argv"][:3] == ["/queue/buildr", "run", "--lane"]
    assert captured["argv"].count("--") == 1
    assert captured["argv"][-2:] == ["--execute", str(run)]
    assert "shell" not in captured
    assert captured["env"]["OMP_NUM_THREADS"] == str(settings.cpu_slots)
    assert "resource-key" in " ".join(captured["argv"])


def test_local_training_refuses_cpu_dry_run(settings, tmp_path, monkeypatch):
    from stateset_agents.training import sft

    run = create_run(tmp_path / "workspace", plan_run(settings))
    monkeypatch.setattr(worker.os, "umask", lambda _: 0)
    monkeypatch.setattr(sft, "gpu_available", lambda: False)
    monkeypatch.setattr(worker, "_train_sft", lambda *_: pytest.fail("must not start"))
    assert worker.execute(run) == 1
    assert not (run / "completion.json").exists()


@pytest.mark.parametrize("with_weights", [False, True])
def test_execution_does_not_treat_zero_exit_without_weights_as_training(
    settings, tmp_path, monkeypatch, with_weights
):
    from stateset_agents.training import sft

    run = create_run(tmp_path / "workspace", plan_run(settings))
    output = run / "artifacts"
    output.mkdir()
    if with_weights:
        (output / "adapter_model.safetensors").write_bytes(b"synthetic weights")
    monkeypatch.setattr(worker.os, "umask", lambda _: 0)
    monkeypatch.setattr(sft, "gpu_available", lambda: True)
    monkeypatch.setattr(worker, "_train_sft", lambda *_: output)
    assert worker.execute(run) == (0 if with_weights else 1)
    assert (run / "completion.json").exists() is with_weights


def test_reconnect_marks_dead_or_reused_owner_without_signalling(
    settings, tmp_path, monkeypatch
):
    run = create_run(tmp_path / "workspace", plan_run(settings))
    write_json(run / "state.json", {"status": "queued"})
    write_json(run / "supervisor.json", {"pid": 123, "process_token": "original"})
    monkeypatch.setattr(workspace, "process_token", lambda _: "reused")
    assert run_status(run)["status"] == "interrupted"
    write_json(run / "state.json", {"status": "succeeded"})
    assert run_status(run)["status"] == "succeeded"


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_log_tail_is_bounded_scrollable_and_terminal_safe(tmp_path, newline):
    (tmp_path / "run.log").write_bytes(
        (
            newline.join([f"row-{i}" for i in range(20)])
            + newline
            + "\x1b[31merror\x1b[0m\x00"
        ).encode()
    )
    assert log_tail(tmp_path, lines=2) == ["row-19", "error"]
    assert log_tail(tmp_path, lines=2, offset=2) == ["row-17", "row-18"]


@pytest.mark.parametrize("state,expected", [("S", "1234"), ("Z", None), ("X", None)])
def test_dead_or_zombie_supervisor_is_not_reported_as_running(
    monkeypatch, state, expected
):
    raw = "42 (worker (training)) " + state + " " + " ".join(["0"] * 18 + ["1234"])
    monkeypatch.setattr(
        workspace, "Path", lambda _: SimpleNamespace(read_text=lambda: raw)
    )
    assert workspace.process_token(42) == expected


def test_corrupt_history_stays_visible(settings, tmp_path):
    run = create_run(tmp_path / "workspace", plan_run(settings))
    (run / "state.json").write_text("broken")
    assert list_runs(tmp_path / "workspace")[0]["state"]["status"] == "failed"


def test_ui_plan_model_search_edit_cancel_and_launch_review(
    settings, tmp_path, monkeypatch, settle_preview
):
    studio = Studio(tmp_path / "workspace", settings)
    calls = []
    monkeypatch.setattr("stateset_agents.tui.launch_run", lambda run: calls.append(run))
    studio.handle("p")
    settle_preview(studio)
    assert studio.plan and not calls
    studio.handle("m")
    for key in "Qwen3.5-0.8B":
        studio.handle(key)
    assert studio.filtered_models()
    studio.handle("ENTER")
    assert studio.overlay is None
    studio.field = next(i for i, f in enumerate(studio.fields()) if f.name == "dataset")
    studio.handle("ENTER")
    studio.handle("CLEAR")
    studio.handle("x")
    studio.handle("ESC")
    assert studio.values["dataset"] == settings.dataset
    studio.handle("r")
    settle_preview(studio)
    assert studio.overlay == "review" and not calls
    reviewed_data = studio.plan.data
    Path(settings.dataset).write_text("changed after launch review")
    studio.handle("ENTER")
    settle_preview(studio)
    assert len(calls) == 1 and studio.tab == "runs"
    assert (calls[0] / "dataset").read_bytes() == reviewed_data
    assert studio.handle("q") is False
    assert len(calls) == 1  # Exit never signals a worker.


def test_missing_runpod_key_blocks_launch_and_never_persists_secrets(
    settings, tmp_path, monkeypatch, settle_preview
):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    studio = Studio(tmp_path / "workspace", replace(settings, provider="runpod"))
    studio.handle("r")
    settle_preview(studio)
    assert studio.overlay == "review"
    studio.handle("ENTER")
    settle_preview(studio)
    assert "RUNPOD_API_KEY is missing" in studio.message
    assert not studio.workspace.exists()
    monkeypatch.setenv("RUNPOD_API_KEY", "synthetic-secret-never-save")
    studio.handle("s")
    assert (
        "synthetic-secret-never-save"
        not in (studio.workspace / "draft.json").read_text()
    )


def test_frames_fit_terminal_and_cli_snapshot_does_not_need_tty(settings, tmp_path):
    from stateset_agents.cli import app

    studio = Studio(tmp_path / "workspace", settings)
    for overlay in (None, "models", "help"):
        studio.overlay = overlay
        for width, height in ((72, 22), (110, 32), (160, 45)):
            frame = studio.lines(width, height)
            assert len(frame) == height
            assert all(len(line) <= width for line in frame)
    config = tmp_path / "draft.json"
    write_json(config, asdict(settings))
    result = CliRunner().invoke(
        app,
        [
            "tui",
            "--workspace",
            str(studio.workspace),
            "--config",
            str(config),
            "--snapshot",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "FINE-TUNING STUDIO" in result.output
    assert not studio.workspace.exists()
    result = CliRunner().invoke(app, ["tui", "--workspace", str(studio.workspace)])
    assert result.exit_code == 2
    assert "interactive terminal" in result.output


def test_sft_progress_callback_and_revision_are_forwarded(
    settings, tmp_path, monkeypatch
):
    from stateset_agents.training import sft

    run = create_run(tmp_path / "workspace", plan_run(settings))
    captured = {}

    def train(**kwargs):
        captured.update(kwargs)
        kwargs["callbacks"][0].on_log(
            None, SimpleNamespace(global_step=7), None, logs={"loss": 0.3}
        )
        return kwargs["output_dir"]

    monkeypatch.setattr(sft, "run_sft", train)
    worker._train_sft(run, replace(settings, revision="a" * 40))
    assert captured["model_revision"] == "a" * 40
    assert read_json(run / "metrics.json")["step"] == 7
    assert read_json(run / "metrics.json")["metrics"] == {"loss": 0.3}


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [False, True])
async def test_native_rl_applies_objective_reference_and_update_budget(
    settings, tmp_path, monkeypatch, complete
):
    from stateset_agents.core import agent as agent_module
    from stateset_agents.training import multi_turn_trainer

    row = {
        "id": "support",
        "context": "My item broke.",
        "policy_facts": ["Broken items can be replaced."],
        "expected_resolution": "Offer a replacement.",
        "user_responses": ["Yes please."],
    }
    Path(settings.dataset).write_text(json.dumps([row]))
    config = replace(settings, method="gspo", steps=2, revision="a" * 40)
    run = create_run(tmp_path / "workspace", plan_run(config))
    captured = {}

    class Agent:
        def __init__(self, configuration):
            captured["agent"] = configuration

        async def initialize(self):
            pass

    class Trainer:
        def __init__(self, agent, environment, **kwargs):
            captured.update(environment=environment, **kwargs)
            self.global_step = 0

        async def initialize(self):
            pass

        async def train(self):
            self.global_step = config.steps if complete else 1
            callback = captured["callbacks"][0]
            callback.on_step_end(self.global_step, {"loss": 0.2})
            await callback.on_optimizer_step(self.global_step, {})

    monkeypatch.setattr(agent_module, "MultiTurnAgent", Agent)
    monkeypatch.setattr(multi_turn_trainer, "MultiTurnGRPOTrainer", Trainer)
    if complete:
        output, updates = await worker._train_rl(run, config)
        assert updates == 2 and output.name == "checkpoint-2"
        assert read_json(run / "state.json")["committed_updates"] == 2
    else:
        with pytest.raises(RuntimeError, match="optimizer updates"):
            await worker._train_rl(run, config)
    assert captured["config"].objective == "gspo"
    assert captured["config"].use_reference_model is True
    assert captured["config"].beta == config.kl_coefficient
    assert captured["config"].num_episodes == 2
    assert captured["agent"].peft_config["lora_dropout"] == 0
    assert captured["agent"].model_revision == "a" * 40
    assert (
        "Broken items can be replaced."
        in captured["environment"].scenarios[0]["context"]
    )
    assert captured["reward_fn"].name == "support_conversation_v3"


def test_runpod_preserves_reviewed_resources_budget_and_handle(
    settings, tmp_path, monkeypatch
):
    from stateset_agents.remote import registry
    from stateset_agents.remote.job import JobHandle, JobStatus

    config = replace(settings, provider="runpod", max_cost_usd=7.5)
    plan = plan_run(config)
    run = create_run(tmp_path / "workspace", plan)
    captured = {}

    class Executor:
        def bind_reviewed_wheel(self, wheel, *, sha256, size_bytes):
            assert wheel is None and sha256 is None and size_bytes is None

        def submit(self, spec):
            captured["spec"] = spec
            return JobHandle(provider="runpod", job_id="synthetic-unit-test-handle")

        def wait(self, handle):
            return SimpleNamespace(
                status=JobStatus.SUCCEEDED,
                succeeded=True,
                output_dir=run / "artifacts",
                logs=["synthetic provider fixture"],
                cost_usd=0.1,
                duration_s=3,
            )

    monkeypatch.setattr(registry, "get_executor", lambda provider: Executor())
    assert worker._train_runpod(run, config) == run / "artifacts"
    spec = captured["spec"]
    assert spec.max_cost_usd == 7.5 and spec.timeout_s == config.timeout_seconds
    assert spec.gpu_count == plan.resources["gpu_count"]
    assert spec.container_disk_gb == plan.resources["container_disk_gb"]
    assert (
        read_json(run / "provider-job.json")["job_id"] == "synthetic-unit-test-handle"
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Herdr PTY contract uses Linux")
@pytest.mark.parametrize("width,height,edit", [(74, 44, True), (110, 32, False)])
def test_real_terminal_model_picker_navigation_and_clean_exit(
    tmp_path, width, height, edit
):
    import fcntl
    import pty
    import select
    import struct
    import subprocess
    import termios

    terminal, slave = pty.openpty()
    fcntl.ioctl(terminal, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
    captured = bytearray()
    reaped = False
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "stateset_agents.cli",
            "tui",
            "--workspace",
            str(tmp_path / "studio"),
        ],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
        env=dict(os.environ, TERM="xterm-256color"),
    )
    os.close(slave)

    def wait_for(text):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if text.encode() in captured:
                return
            ready, _, _ = select.select([terminal], [], [], 0.1)
            if ready:
                try:
                    captured.extend(os.read(terminal, 65536))
                except OSError:
                    break
        pytest.fail(f"Missing terminal output {text!r}: {bytes(captured)[-2000:]!r}")

    try:
        wait_for("FINE-TUNING STUDIO")
        os.write(terminal, b"v")
        wait_for("SETTINGS CHECK")
        captured.clear()
        os.write(terminal, b"\n")
        wait_for("Edit this setting")
        if edit:
            os.write(terminal, b"\n")
            wait_for("EDIT:")
            os.write(terminal, b"chat.jsonl\x1bOH/tmp/\x1bOF\ns")
            wait_for("Saved")
            assert (
                read_json(tmp_path / "studio" / "draft.json")["dataset"]
                == "/tmp/chat.jsonl"
            )
        os.write(terminal, b"b")
        wait_for("TRAINING BACKENDS")
        os.write(terminal, b"\x1bOB\x1bOB\n")
        wait_for("Backend: Modal.")
        os.write(terminal, b":")
        wait_for("ACTIONS")
        os.write(terminal, b"\x1bOB\n")
        wait_for("CATALOG / type to search")
        os.write(terminal, b"Qwen3.5")
        wait_for("Qwen3.5")
        os.write(terminal, b"\n\t")
        wait_for("No retained runs")
        os.write(terminal, b"/")
        wait_for("FILTER RUNS / ID")
        os.write(terminal, b"no-match\n")
        wait_for("No runs match")
        os.write(terminal, b"\x7f")
        # Reconnection and filtering never create or launch a run.
        os.write(terminal, b"q")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = process.poll()
            if status is not None:
                reaped = True
                assert status == 0
                break
            select.select([], [], [], 0.05)
        assert reaped, "Studio did not exit after q"
        assert (tmp_path / "studio").exists() is edit
        assert not list((tmp_path / "studio").glob("*/plan.json"))
    finally:
        os.close(terminal)
        if not reaped:
            process.kill()
            process.wait()


@pytest.mark.parametrize("width,height", [(72, 22), (74, 44), (110, 32), (140, 40)])
def test_responsive_layout_keeps_model_and_all_shortcuts_visible(
    settings, tmp_path, width, height
):
    studio = Studio(tmp_path / "studio", settings)
    frame = studio.lines(width, height)
    text = "\n".join(frame)
    assert settings.model in text
    assert "q quit" in text and "? help" in text
    assert "r review" in text and "Tab runs" in text
    if width < 112:
        assert "RUN PREVIEW" not in text
    else:
        assert "RUN PREVIEW" in text
    assert len(frame) == height


def test_wide_character_models_and_paths_fit_terminal_cells(settings, tmp_path):
    from rich.cells import cell_len

    model = "组织/支持模型-" + "非常长" * 20
    studio = Studio(tmp_path / "studio", replace(settings, model=model))
    studio.field = next(
        i for i, field in enumerate(studio.fields()) if field.name == "model"
    )
    for width in (72, 74, 120):
        for overlay in (None, "help", "models"):
            studio.overlay = overlay
            frame = studio.lines(width, 40)
            assert all(cell_len(line) <= width for line in frame)
    studio.overlay = None
    studio.handle("ENTER")
    studio.handle("END")
    frame = studio.lines(74, 40)
    assert 6 <= studio.editor_column < 73
    assert all(cell_len(line) <= 74 for line in frame)


def test_editing_inserts_and_deletes_at_cursor_without_losing_other_text(
    settings, tmp_path
):
    studio = Studio(tmp_path / "studio", settings)
    studio.field = next(
        i for i, field in enumerate(studio.fields()) if field.name == "dataset"
    )
    studio.handle("ENTER")
    studio.handle("CLEAR")
    for char in "abde":
        studio.handle(char)
    studio.handle("LEFT")
    studio.handle("LEFT")
    studio.handle("c")
    assert studio.editing == "abcde"
    studio.handle("BACKSPACE")
    assert studio.editing == "abde"
    studio.handle("DELETE")
    assert studio.editing == "abe"
    studio.handle("HOME")
    studio.handle("x")
    studio.handle("END")
    studio.handle("z")
    assert studio.editing == "xabez"
    studio.handle("ENTER")
    assert studio.values["dataset"] == "xabez"
    assert studio.plan is None


@pytest.mark.parametrize("overlay", [None, "help", "models", "review", "details"])
def test_control_c_exits_every_screen_without_starting_a_job(
    settings, tmp_path, monkeypatch, overlay
):
    studio = Studio(tmp_path / "studio", settings)
    studio.overlay = overlay
    monkeypatch.setattr(
        "stateset_agents.tui.launch_run", lambda _: pytest.fail("must not launch")
    )
    assert studio.handle("QUIT") is False
    studio.editing = "a draft in progress"
    assert studio.handle("QUIT") is False


def test_model_change_clears_stale_revision_but_same_model_keeps_it(settings, tmp_path):
    studio = Studio(tmp_path / "studio", replace(settings, revision="a" * 40))
    studio._set_model(settings.model)
    assert studio.values["revision"] == "a" * 40
    studio._set_model("different/model")
    assert studio.values["revision"] == ""
    assert "cleared" in studio.message
    studio.values.update(provider="runpod", revision="b" * 40)
    assert "revision" in {field.name for field in studio.fields()}


def test_run_selection_survives_newer_runs(settings, tmp_path, monkeypatch):
    root = tmp_path / "studio"
    original = create_run(root, plan_run(settings))
    studio = Studio(root, settings)
    original_id = studio.runs[0]["id"]
    newer = create_run(root, plan_run(settings))
    # Ordering is deliberately supplied; refresh must retain identity, not index.
    rows = list_runs(root)
    rows.sort(key=lambda row: row["id"] != newer.name)
    monkeypatch.setattr("stateset_agents.tui.list_runs", lambda _, **kwargs: rows)
    studio.refresh()
    assert studio.runs[studio.run_index]["id"] == original_id == original.name


def test_details_and_reuse_reference_private_snapshot_without_launch(
    settings, tmp_path, monkeypatch, settle_preview
):
    root = tmp_path / "studio"
    run = create_run(root, plan_run(settings))
    write_json(
        run / "state.json",
        {"status": "succeeded", "started_at": 1000, "finished_at": 1065},
    )
    write_json(
        run / "completion.json",
        {
            "output_dir": str(run / "artifacts"),
            "committed_updates": 7,
            "quality_gain_verified": False,
        },
    )
    write_json(
        run / "provider-job.json",
        {
            "provider": "runpod",
            "job_id": "fixture-handle",
            "token": "fixture-secret-do-not-display",
        },
    )
    write_json(run / "provider-result.json", {"cost_usd": 1.25, "duration_s": 65})
    studio = Studio(root, settings)
    studio.handle("TAB")
    studio.handle("ENTER")
    assert studio.overlay == "details"
    detail = "\n".join(studio._run_details())
    assert "00:01:05" in detail and "fixture-handle" in detail and "1.25" in detail
    assert str(run / "artifacts") in detail
    assert "fixture-secret-do-not-display" not in detail
    Path(settings.dataset).unlink()  # Source need not exist once retained.
    monkeypatch.setattr(
        "stateset_agents.tui.launch_run",
        lambda _: pytest.fail("copying must not launch"),
    )
    studio.handle("c")
    assert studio.tab == "configure" and studio.overlay is None
    assert studio.values["dataset"] == str(run / "dataset")
    assert studio.values["model"] == settings.model
    assert studio.plan is None
    studio.handle("p")
    settle_preview(studio)
    assert studio.plan is not None
    assert len(list_runs(root)) == 1


def test_reuse_rejects_tampered_retained_data(settings, tmp_path):
    run = create_run(tmp_path / "studio", plan_run(settings))
    (run / "dataset").write_text("tampered")
    studio = Studio(tmp_path / "studio", settings)
    studio.handle("TAB")
    studio.handle("c")
    assert studio.tab == "runs"
    assert "differs from its reviewed hash" in studio.message


def test_dialog_scroll_reaches_last_line_after_wrapping(
    settings, tmp_path, settle_preview
):
    studio = Studio(tmp_path / "studio", settings)
    studio.handle("r")
    settle_preview(studio)
    assert studio.overlay == "review"
    studio.plan = replace(
        studio.plan,
        notes=tuple(
            f"Long review item {index}: " + "detail " * 30 for index in range(30)
        ),
    )
    for _ in range(100):
        studio.handle("PAGEDOWN")
        studio.lines(72, 22)
    frame = "\n".join(studio.lines(72, 22))
    assert "Closing this terminal" in frame
    assert studio.dialog_offset > 18
    for _ in range(100):
        studio.handle("PAGEUP")
    assert "REVIEW LAUNCH" in "\n".join(studio.lines(72, 22))


@pytest.mark.parametrize("target,value", [("config", []), ("config", None)])
def test_structurally_damaged_history_does_not_crash_rendering(
    settings, tmp_path, target, value
):
    run = create_run(tmp_path / "studio", plan_run(settings))
    plan = read_json(run / "plan.json")
    plan[target] = value
    write_json(run / "plan.json", plan)
    studio = Studio(tmp_path / "studio", settings)
    studio.handle("TAB")
    assert "failed" in "\n".join(studio.lines(74, 44))
    studio.handle("ENTER")
    assert "RUN DETAILS" in "\n".join(studio.lines(74, 44))


def test_incomplete_draft_can_be_saved_and_reopened(tmp_path):
    from stateset_agents.cli import app

    studio = Studio(tmp_path / "studio")
    studio.handle("s")
    draft = studio.workspace / "draft.json"
    assert draft.exists()
    assert read_json(draft)["dataset"] == ""
    result = CliRunner().invoke(
        app, ["tui", "--workspace", str(studio.workspace), "--snapshot"]
    )
    assert result.exit_code == 0
    studio.handle("r")
    assert studio.overlay is None and "dataset path" in studio.message


def test_file_picker_filters_selects_without_reading_contents(
    settings, tmp_path, monkeypatch
):
    (tmp_path / "unrelated.txt").write_text("ignored")
    (tmp_path / ".hidden.json").write_text("hidden")
    chosen = tmp_path / "unvalidated data.jsonl"
    chosen.write_text("not validated until p")
    studio = Studio(tmp_path / "studio", settings)
    studio.handle("f")
    assert studio.overlay == "files"
    assert "unrelated.txt" not in {row[0] for row in studio.file_picker.entries}
    assert ".hidden.json" not in {row[0] for row in studio.file_picker.entries}
    for char in chosen.name:
        studio.handle(char)
    assert len(studio.file_picker.filtered()) == 1
    monkeypatch.setattr(
        Path, "read_bytes", lambda _: pytest.fail("picker must not read file content")
    )
    studio.handle("ENTER")
    assert studio.overlay is None
    assert studio.values["dataset"] == str(chosen)
    assert studio.plan is None
    assert not studio.workspace.exists()


def test_file_picker_directory_navigation_and_cancel_preserve_draft(settings, tmp_path):
    directory = tmp_path / "cases"
    directory.mkdir()
    (directory / "train.json").write_text("[]")
    studio = Studio(tmp_path / "studio", settings)
    studio.handle("f")
    for char in "cases":
        studio.handle(char)
    studio.handle("ENTER")
    assert studio.file_picker.directory == directory
    assert studio.file_picker.search == ""
    studio.handle("BACKSPACE")
    assert studio.file_picker.directory == tmp_path
    studio.handle("ESC")
    assert studio.values["dataset"] == settings.dataset
    assert studio.overlay is None


def test_file_picker_permission_failure_remains_navigable(
    settings, tmp_path, monkeypatch
):
    studio = Studio(tmp_path / "studio", settings)
    monkeypatch.setattr(
        "stateset_agents.tui_panels.os.scandir",
        lambda _: (_ for _ in ()).throw(PermissionError("denied")),
    )
    studio.handle("f")
    assert studio.overlay == "files" and "PermissionError" in studio.file_picker.error
    assert studio.file_picker.entries[0][0] == ".."
    assert "DATASET PICKER" in "\n".join(studio.lines(74, 44))
    studio.handle("ESC")
    assert studio.overlay is None
