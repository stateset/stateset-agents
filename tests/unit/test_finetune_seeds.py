"""Training seeds reach initialization, provider payloads and paired evidence."""

import copy
import json
import shlex
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents import cli as root_cli
from stateset_agents import cli_finetune as cli
from stateset_agents import finetune_evaluation as evaluation
from stateset_agents import finetune_quality as quality
from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as ws
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec
from stateset_agents.remote.registry import get_executor
from stateset_agents.training import sft
from stateset_agents.tui import Studio


def data(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Welcome"},
                ]
            }
        )
        + "\n"
    )
    return path


@pytest.mark.parametrize("seed", [0, 17, 2**32 - 1])
def test_seed_survives_draft_plan_cli_and_schema(tmp_path, seed):
    config = ws.FinetuneConfig(dataset=str(data(tmp_path)), seed=seed)
    studio = Studio(tmp_path / "workspace", config)
    studio.handle("s")
    restored = Studio(
        studio.workspace,
        ws.FinetuneConfig(**ws.read_json(studio.workspace / "draft.json")),
    )
    snapshot = CliRunner().invoke(
        root_cli.app, ["tui", "--workspace", str(studio.workspace), "--snapshot"]
    )
    assert snapshot.exit_code == 0 and "Training seed" in snapshot.stdout
    assert restored.config().seed == seed
    assert "seed" in {field.name for field in restored.fields()}
    plan = ws.plan_run(restored.config())
    assert plan.describe()["config"]["seed"] == seed
    assert f"Training seed: {seed}" in "\n".join(plan.notes)
    file = tmp_path / "config.json"
    ws.write_json(file, asdict(config))
    result = CliRunner().invoke(cli.app, ["validate", "--config", str(file)])
    assert result.exit_code == 0 and json.loads(result.stdout)["result"]["valid"]
    schema = ws.finetune_schema()["config_schema"]["properties"]["seed"]
    assert schema == {
        "type": "integer",
        "default": 42,
        "minimum": 0,
        "maximum": 2**32 - 1,
    }


@pytest.mark.parametrize("seed", [True, False, -1, 2**32, 1.5, None, "PRIVATE"])
def test_invalid_seed_is_field_addressable_without_echoing_values(seed):
    _, issues = ws.check_config_values({"dataset": "not-read", "seed": seed})
    assert len(issues) == 1 and issues[0].field == "seed"
    assert "PRIVATE" not in issues[0].message


def test_old_draft_defaults_seed_without_rewriting_it(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ws.write_json(workspace / "draft.json", {"dataset": "old.jsonl"})
    before = (workspace / "draft.json").read_bytes()
    studio = Studio(workspace)
    assert studio.config().seed == 42
    snapshot = CliRunner().invoke(
        root_cli.app, ["tui", "--workspace", str(workspace), "--snapshot"]
    )
    assert snapshot.exit_code == 0 and "Training seed" in snapshot.stdout
    assert (workspace / "draft.json").read_bytes() == before


@pytest.mark.parametrize("size", [(72, 22), (74, 44), (110, 32), (140, 40)])
@pytest.mark.parametrize("name", ["seed", "checkpoint_every", "gpu_count"])
def test_configure_displays_valid_zero_in_rows_and_selected_help(tmp_path, size, name):
    config = ws.FinetuneConfig(
        seed=0,
        method="grpo" if name == "checkpoint_every" else "sft",
        provider="runpod" if name == "gpu_count" else "local",
    )
    studio = Studio(tmp_path / "workspace", config)
    studio.field = next(
        i for i, field in enumerate(studio.fields()) if field.name == name
    )
    label = studio.fields()[studio.field].label
    lines = studio.lines(*size)
    assert f"{label}: 0" in "\n".join(lines)
    row = next(line for line in lines if line.lstrip().startswith("> " + label))
    left = row[: size[0] // 2] if size[0] >= 112 else row
    assert left.rstrip().endswith(" 0")
    assert studio.values[name] == 0 and not studio.workspace.exists()


@pytest.mark.skipif(
    sys.platform != "linux", reason="Curses seed editor uses a Linux PTY"
)
@pytest.mark.parametrize("size", [(74, 44), (110, 32)])
def test_real_terminal_edits_seed_reviews_and_saves_without_launch(tmp_path, size):
    import fcntl
    import os
    import pty
    import select
    import struct
    import subprocess
    import sys
    import termios
    import time

    config = ws.FinetuneConfig(dataset=str(data(tmp_path)), seed=17)
    workspace = tmp_path / "workspace"
    studio = Studio(workspace, config)
    studio.handle("s")
    index = next(i for i, field in enumerate(studio.fields()) if field.name == "seed")
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
            str(workspace),
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

    def send(keys, marker):
        captured.clear()
        os.write(terminal, keys)
        wait_for(marker)

    try:
        wait_for("FINE-TUNING STUDIO")
        send(b"\x1bOB" * index + b"\n", "EDIT:")
        send(b"\x150\n", "s save")
        send(b"r", "REVIEW LAUNCH")
        assert not list(workspace.glob("*/plan.json"))
        send(b"\x1b", "s save")
        send(b"s", "Saved")
        os.write(terminal, b"q")
        assert process.wait(timeout=15) == 0
        assert ws.read_json(workspace / "draft.json")["seed"] == 0
        assert not list(workspace.glob("*/plan.json"))
        result = CliRunner().invoke(
            root_cli.app, ["tui", "--workspace", str(workspace), "--snapshot"]
        )
        assert result.exit_code == 0 and "Training seed" in result.stdout
    finally:
        os.close(terminal)
        if process.poll() is None:
            process.kill()
            process.wait()


def test_local_sft_worker_forwards_recorded_seed(tmp_path, monkeypatch):
    config = ws.FinetuneConfig(dataset=str(data(tmp_path)), seed=17)
    run = ws.create_run(tmp_path / "workspace", ws.plan_run(config))
    captured = {}
    monkeypatch.setattr(sft, "run_sft", lambda **kwargs: captured.update(kwargs))
    worker._train_sft(run, config)
    assert captured["seed"] == 17


@pytest.mark.parametrize("kind", ["sft", "native"])
def test_exact_recovery_preserves_seed_and_rejects_changes(tmp_path, kind):
    from stateset_agents import finetune_recovery as recovery
    from tests.unit.test_finetune_recovery import make_parent

    run, _ = make_parent(tmp_path, kind)
    plan = ws.read_json(run / "plan.json")
    plan["config"]["seed"] = 17
    ws.write_json(run / "plan.json", plan)
    recovery.publish_checkpoint(run, run / "artifacts/checkpoint-1", step=1)
    choice = recovery.inspect_checkpoints(run)["checkpoints"][0]
    child = recovery.prepare_resume(
        run, "checkpoint-1", expected_recovery_sha256=choice["recovery_sha256"]
    )
    assert ws.review_prepared_run(child).plan.config.seed == 17
    changed = ws.read_json(child / "plan.json")
    changed["config"]["seed"] = 18
    ws.write_json(child / "plan.json", changed)
    with pytest.raises(ValueError):
        ws.verified_run_settings(child)


@pytest.mark.parametrize("seed", [0, 17, 2**32 - 1])
def test_transport_seed_round_trips_and_reaches_runpod_command_parser(tmp_path, seed):
    spec = RemoteJobSpec(dataset=data(tmp_path), base_model="fixture/model", seed=seed)
    restored = RemoteJobSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored.seed == seed
    assert sft.build_parser().parse_args(restored.to_cli_args()).seed == seed
    from stateset_agents.remote.runpod import RunPodExecutor

    command = RunPodExecutor()._remote_commands(restored, "/workspace/data.jsonl")[-1]
    assert sft.build_parser().parse_args(shlex.split(command)[3:]).seed == seed


@pytest.mark.parametrize("seed", [True, -1, 2**32, "17", 1.5])
def test_transport_and_dry_run_reject_invalid_seed_before_data_read(
    tmp_path, monkeypatch, seed
):
    source = data(tmp_path)
    with pytest.raises(ValueError, match="seed"):
        RemoteJobSpec(dataset=source, base_model="fixture/model", seed=seed)
    monkeypatch.setattr(
        sft,
        "load_chat_dataset",
        lambda *args: pytest.fail("No data read for invalid seed"),
    )
    result = sft.run_sft_job(
        {"dataset": str(source), "output_dir": str(tmp_path / "out"), "seed": seed}
    )
    assert result["returncode"] == 1 and not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["rl", "harvest"])
def test_shared_seed_cannot_silently_override_other_job_seed_protocols(tmp_path, kind):
    with pytest.raises(ValueError, match="SFT jobs only"):
        RemoteJobSpec(
            dataset=data(tmp_path), base_model="fixture/model", job_kind=kind, seed=17
        )


@pytest.mark.parametrize(
    "provider",
    ["local", "runpod", "modal", "river", "huggingface", "coreweave", "nebius"],
)
def test_seed_preserving_transports_declare_capability(tmp_path, provider):
    executor = get_executor(provider)
    spec = RemoteJobSpec(dataset=data(tmp_path), base_model="fixture/model", seed=17)
    executor.validate_spec(spec)
    assert executor.capabilities()["sft_seed"] is True


@pytest.mark.parametrize("provider", ["fireworks", "together", "tinker", "prime"])
def test_other_transports_reject_explicit_seed_before_provider_calls(
    tmp_path, provider
):
    executor = get_executor(provider)
    spec = RemoteJobSpec(dataset=data(tmp_path), base_model="fixture/model", seed=17)
    with pytest.raises(RemoteExecutionError, match="explicit SFT seed"):
        executor.submit(spec)
    assert executor.capabilities()["sft_seed"] is False


@pytest.mark.parametrize("provider", ["runpod", "modal", "river"])
def test_studio_passes_reviewed_seed_to_provider(tmp_path, monkeypatch, provider):
    config = ws.FinetuneConfig(
        dataset=str(data(tmp_path)),
        provider=provider,
        seed=17,
        cost_policy="provider_billed",
    )
    run = ws.create_run(tmp_path / "workspace", ws.plan_run(config))
    captured = []

    class Executor:
        def bind_reviewed_wheel(self, wheel, *, sha256, size_bytes):
            assert wheel is None and sha256 is None and size_bytes is None

        def submit(self, spec):
            captured.append(spec)
            return JobHandle(provider, "synthetic")

        def wait(self, handle):
            return SimpleNamespace(
                status=JobStatus.SUCCEEDED,
                logs=[],
                succeeded=True,
                output_dir=run / "artifacts",
                cost_usd=None,
                duration_s=1,
            )

    monkeypatch.setattr(
        "stateset_agents.remote.registry.get_executor", lambda name: Executor()
    )
    worker._train_remote(run, config)
    assert captured[0].seed == 17


def test_river_seed_reaches_adapter_and_retained_pointer(tmp_path, monkeypatch):
    from stateset_agents.remote.river import RiverExecutor, _sft_options
    from tests.unit.test_remote_river_executor import FakeRiverClient
    from tests.unit.test_river_batches import FakeTokenizer

    def no_renderer(*args, **kwargs):
        raise ValueError("Synthetic transport uses raw tokenization")

    monkeypatch.setattr(
        "stateset_agents.remote.river_renderers.get_river_renderer", no_renderer
    )
    client = FakeRiverClient()
    executor = RiverExecutor(
        client=client, tokenizer=FakeTokenizer(), ledger_path=tmp_path / "ledger"
    )
    spec = RemoteJobSpec(
        dataset=data(tmp_path),
        base_model="Qwen/Qwen3.5-9B",
        output_dir=tmp_path / "out",
        seed=17,
    )
    result = executor.wait(executor.submit(spec), poll_interval_s=0)
    assert result.succeeded and client.sessions[0].models[0].lora.seed == 17
    pointer = ws.read_json(result.output_dir / "river_checkpoint.json")
    assert pointer["sft_options"]["seed"] == 17
    assert _sft_options(replace(spec, seed=None))["seed"] == 0
    assert _sft_options(replace(spec, harvest={"seed": 17}))["seed"] == 17
    with pytest.raises(RemoteExecutionError, match="Conflicting"):
        executor.submit(replace(spec, harvest={"seed": 18}))
    assert len(client.sessions) == 1


@pytest.mark.parametrize("seed", [None, 0, 17])
def test_sft_worker_payload_preserves_legacy_default_and_explicit_seed(
    tmp_path, monkeypatch, seed
):
    captured = {}
    monkeypatch.setattr(sft, "gpu_available", lambda: True)

    def train(**kwargs):
        from tests.sft_fixtures import write_sft_export

        captured.update(kwargs)
        write_sft_export(
            kwargs["output_dir"],
            **{key: value for key, value in kwargs.items() if key != "output_dir"},
        )

    monkeypatch.setattr(sft, "run_sft", train)
    spec = RemoteJobSpec(
        dataset=data(tmp_path),
        base_model="fixture/model",
        output_dir=tmp_path / "out",
        seed=seed,
    )
    assert sft.run_sft_job(spec.to_dict())["returncode"] == 0
    assert captured["seed"] == (42 if seed is None else seed)


@pytest.mark.parametrize("seed", [0, 17])
def test_remote_cli_forwards_explicit_seed_without_running_provider(
    tmp_path, monkeypatch, seed
):
    captured = []

    class Executor:
        def submit(self, spec):
            captured.append(spec.seed)
            return JobHandle("local", "synthetic")

        def wait(self, handle):
            return SimpleNamespace(
                logs=[], succeeded=True, output_dir=tmp_path / "out", cost_usd=None
            )

    monkeypatch.setattr(
        "stateset_agents.cli_remote.get_executor", lambda provider: Executor()
    )
    result = CliRunner().invoke(
        root_cli.app,
        [
            "train-remote",
            "--provider",
            "local",
            "--dataset",
            str(data(tmp_path)),
            "--base-model",
            "fixture/model",
            "--seed",
            str(seed),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert captured == [seed]


@pytest.fixture
def rng_state():
    torch = pytest.importorskip("torch")
    from stateset_agents.training.runtime_state import capture_rng, restore_rng

    before = capture_rng(torch)
    try:
        yield
    finally:
        restore_rng(before, torch)


@pytest.mark.asyncio
async def test_native_seed_is_applied_before_agent_initialization_and_trainer(
    tmp_path, monkeypatch, rng_state
):
    import random

    import numpy as np
    import torch

    source = tmp_path / "rl.json"
    ws.write_json(
        source,
        [
            {
                "id": "train",
                "context": "Help",
                "policy_facts": ["Replace damaged items"],
                "expected_resolution": "Replacement",
                "user_responses": ["Yes"],
            }
        ],
    )
    initialized, training_seeds = [], []

    class Agent:
        def __init__(self, config):
            initialized.append(
                (random.random(), float(np.random.random()), torch.rand(4))
            )

        async def initialize(self):
            pass

    class Trainer:
        def __init__(self, *args, config, **kwargs):
            training_seeds.append(config.seed)
            self.global_step = config.num_episodes

        async def initialize(self):
            pass

        async def train(self):
            pass

    monkeypatch.setattr("stateset_agents.core.agent.MultiTurnAgent", Agent)
    monkeypatch.setattr(
        "stateset_agents.training.multi_turn_trainer.MultiTurnGRPOTrainer", Trainer
    )
    for seed in [17, 17, 29]:
        config = ws.FinetuneConfig(dataset=str(source), method="grpo", seed=seed)
        run = ws.create_run(tmp_path / "workspace", ws.plan_run(config))
        await worker._train_rl(run, config)
    assert initialized[0][:2] == initialized[1][:2] != initialized[2][:2]
    assert torch.equal(initialized[0][2], initialized[1][2])
    assert not torch.equal(initialized[0][2], initialized[2][2])
    assert training_seeds == [17, 17, 29]


def test_real_cpu_sft_seed_controls_initial_adapter_training_and_manifest(
    tmp_path, monkeypatch, rng_state
):
    import torch
    from safetensors.torch import load_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from stateset_agents.training.lineage import read_manifest

    backend = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[EOS]": 2,
                "user": 3,
                "assistant": 4,
                "Hello": 5,
                "Welcome": 6,
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
    template = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=7,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=1,
            pad_token_id=1,
            bos_token_id=2,
            eos_token_id=2,
        )
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: copy.deepcopy(tokenizer),
    )
    monkeypatch.setattr(
        sft, "load_base_model_for_sft", lambda *args, **kwargs: copy.deepcopy(template)
    )
    build = sft.build_training_arguments
    argument_seeds = []

    def arguments(cls, **kwargs):
        argument_seeds.append((kwargs["seed"], kwargs["data_seed"]))
        return build(
            cls,
            **{
                **kwargs,
                "use_cpu": True,
                "bf16": False,
                "disable_tqdm": True,
                "dataloader_num_workers": 0,
            },
        )

    monkeypatch.setattr(sft, "build_training_arguments", arguments)
    source = data(tmp_path)
    rows = [json.loads(source.read_text())] * 4
    results = []
    for index, seed in enumerate([17, 17, 29]):
        output = sft.run_sft(
            rows,
            "fixture/model",
            tmp_path / f"adapter-{index}",
            num_epochs=1,
            lora_r=4,
            lora_alpha=8,
            learning_rate=1e-3,
            max_length=32,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            seed=seed,
        )
        tensors = load_file(output / "adapter_model.safetensors")
        assert any(
            torch.count_nonzero(value)
            for key, value in tensors.items()
            if "lora_B" in key
        )
        results.append(tensors)
        manifest = read_manifest(output)
        assert (
            manifest["hyperparameters"]["seed"]
            == manifest["hyperparameters"]["data_seed"]
            == seed
        )
    assert results[0].keys() == results[1].keys() == results[2].keys()
    assert all(torch.equal(results[0][key], results[1][key]) for key in results[0])
    assert any(not torch.equal(results[0][key], results[2][key]) for key in results[0])
    assert argument_seeds == [(17, 17), (17, 17), (29, 29)]


def test_evaluation_seed_cannot_relabel_training_independence(tmp_path, monkeypatch):
    from tests.unit.test_finetune_evaluation import make_evaluation, run_pipeline

    run, config = make_evaluation(tmp_path)
    plan = ws.read_json(run / "plan.json")
    plan["config"]["seed"] = 17
    ws.write_json(run / "plan.json", plan)
    config = replace(config, seed=99)
    review = evaluation.plan_evaluation(run, config)
    assert review.plan["binding"]["training_seed"] == 17
    from stateset_agents.finetune_evaluation_tui import EvaluationPanel

    panel = EvaluationPanel(run)
    panel.review, panel.mode = review, "review"
    assert "Training seed: 17 | Evaluation seed: 99" in "\n".join(panel.lines())
    path = evaluation.prepare_evaluation(review, expected_plan_sha256=review.sha256)
    run_pipeline(path, monkeypatch)
    reports = ws.read_json(path / "paired-reports.json")
    assert reports["baselines"][0]["seed"] == reports["candidates"][0]["seed"] == 17
    assert reports["baselines"][0]["settings"]["evaluation_seed"] == 99
    result = evaluation.inspect_evaluation(path)
    assert result["summary"]["training_seed"] == 17 and result["judge_replay_verified"]
    reports["baselines"][0]["seed"] = reports["candidates"][0]["seed"] = 99
    with pytest.raises(ValueError, match="protocol"):
        evaluation.summarize_evaluation(review, reports)


@pytest.mark.parametrize("legacy", [False, True])
def test_quality_binds_training_seed_and_marks_unknown_history(tmp_path, legacy):
    from tests.unit.test_finetune_quality import evidence_for, make_completed

    run, config = make_completed(tmp_path)
    reports = evidence_for(run, config, seeds=6)
    plan = ws.read_json(run / "plan.json")
    if legacy:
        del plan["config"]["seed"]
        ws.write_json(run / "plan.json", plan)
        summary = quality._summarize(run, reports)
        assert summary["training_seed"] is None and not summary["recorded_study_passed"]
        assert not summary["study_gates"]["bound_training_seed_recorded"]
    else:
        assert quality._summarize(run, reports)["training_seed"] == 0
        reports["baselines"][0]["seed"] = reports["candidates"][0]["seed"] = 17
        with pytest.raises(ValueError, match="recorded training seed"):
            quality._summarize(run, reports)
