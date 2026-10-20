"""Studio backend selection, paid controls and result contracts without cloud calls."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as api
from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace
from stateset_agents.finetune_inspection import inspect_readiness
from stateset_agents.finetune_workspace import FinetuneConfig, read_json, write_json
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobResult
from stateset_agents.tui import Studio


@pytest.fixture
def settings(tmp_path):
    data = tmp_path / "chat.jsonl"
    data.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        )
        + "\n"
    )
    return FinetuneConfig(dataset=str(data))


@pytest.mark.parametrize("provider", workspace.PROVIDERS)
def test_picker_selects_each_backend_without_network_or_changing_objective(
    settings, tmp_path, provider
):
    studio = Studio(tmp_path / "studio", settings)
    studio.plan = workspace.plan_run(settings)
    draft = dict(studio.values)
    studio.handle("b")
    assert studio.overlay == "backends"
    studio.backend_index = workspace.PROVIDERS.index(provider)
    for width, height in [(74, 44), (110, 32)]:
        text = "\n".join(studio.lines(width, height))
        assert "TRAINING BACKENDS" in text and "Enter choose backend" in text
        assert str(workspace.BACKENDS[provider]["label"]) in text
    studio.handle("ESC")
    assert studio.values == draft and studio.plan is not None
    studio.handle("b")
    studio.backend_index = workspace.PROVIDERS.index(provider)
    studio.handle("ENTER")
    assert (
        studio.values["provider"] == provider
        and studio.values["method"] == settings.method
    )
    assert studio.plan is None and not (tmp_path / "studio").exists()


def test_compute_field_cycles_all_backends_and_river_clears_hidden_gpu_overrides(
    settings, tmp_path
):
    studio = Studio(tmp_path / "studio", settings)
    studio.field = next(
        index for index, field in enumerate(studio.fields()) if field.name == "provider"
    )
    seen = []
    for _ in workspace.PROVIDERS:
        seen.append(studio.values["provider"])
        studio.handle("RIGHT")
    assert set(seen) == set(workspace.PROVIDERS)
    studio.values.update(gpu="H100", gpu_count=2, method="gspo")
    studio._set_backend("river")
    assert studio.values["gpu"] == "" and studio.values["gpu_count"] == 0
    assert studio.values["method"] == "gspo"
    assert "Cleared GPU overrides" in studio.message
    with pytest.raises(ValueError, match="dedicated River RL"):
        studio.config().validate()


@pytest.mark.parametrize("provider", ["modal", "river"])
def test_uncapped_remote_billing_requires_explicit_policy(settings, provider):
    with pytest.raises(ValueError, match="provider_billed"):
        workspace.plan_run(replace(settings, provider=provider))
    plan = workspace.plan_run(
        replace(settings, provider=provider, cost_policy="provider_billed")
    )
    assert plan.resources["provider"] == provider
    assert any("no enforced dollar ceiling" in note for note in plan.notes)
    for method in ["grpo", "gspo", "dapo"]:
        with pytest.raises(ValueError, match="SFT"):
            workspace.plan_run(replace(plan.config, method=method))
    with pytest.raises(ValueError, match="cannot enforce a revision"):
        workspace.plan_run(replace(plan.config, revision="a" * 40))


@pytest.mark.parametrize("provider", ["runpod", "modal"])
def test_reviewed_gpu_selection_reaches_remote_spec(
    settings, tmp_path, monkeypatch, provider
):
    config = replace(
        settings,
        provider=provider,
        cost_policy="provider_billed",
        gpu="H100",
        gpu_count=2,
    )
    run = workspace.create_run(tmp_path / "studio", workspace.plan_run(config))
    calls = []

    class Executor:
        def bind_reviewed_wheel(self, wheel, *, sha256, size_bytes):
            assert wheel is None and sha256 is None and size_bytes is None

        def submit(self, spec):
            calls.append(spec)
            return JobHandle(provider, "synthetic-provider-job")

        def wait(self, handle):
            return RemoteJobResult(handle, JobStatus.SUCCEEDED, run / "artifacts", [])

    from stateset_agents.remote import registry

    monkeypatch.setattr(
        registry,
        "get_executor",
        lambda name: Executor() if name == provider else pytest.fail("wrong provider"),
    )
    assert worker._train_remote(run, config) == run / "artifacts"
    spec = calls[0]
    assert spec.package_version == read_json(run / "plan.json")["framework_version"]
    assert spec.gpu == "H100" and spec.gpu_count == 2
    assert spec.dataset == run / "dataset"
    assert spec.max_cost_usd == (config.max_cost_usd if provider == "runpod" else None)
    assert (
        spec.learning_rate == config.learning_rate and spec.num_epochs == config.epochs
    )
    assert read_json(run / "provider-job.json")["provider"] == provider


@pytest.mark.parametrize("provider", ["modal", "river"])
def test_cli_paid_opt_in_and_tui_handoff_apply_to_every_remote_backend(
    settings, tmp_path, monkeypatch, provider
):
    config = replace(settings, provider=provider, cost_policy="provider_billed")
    source = tmp_path / "config.json"
    root = tmp_path / "studio"
    write_json(source, asdict(config))
    runner = CliRunner()
    result = runner.invoke(
        api.app, ["prepare", "--config", str(source), "--workspace", str(root)]
    )
    assert result.exit_code == 0, result.stdout
    prepared = json.loads(result.stdout)["result"]
    assert prepared["requires_paid_opt_in"]
    row = prepared["run"]
    argv = [
        "launch",
        row["run_id"],
        "--workspace",
        str(root),
        "--expected-plan-sha256",
        row["plan_sha256"],
    ]
    result = runner.invoke(api.app, argv)
    assert json.loads(result.stdout)["error"]["code"] == "paid_opt_in_required"
    calls = []
    monkeypatch.setattr(
        workspace.subprocess,
        "Popen",
        lambda *a, **k: calls.append((a, k)) or SimpleNamespace(pid=123),
    )
    if provider == "river":
        monkeypatch.delenv("RIVER_API_KEY", raising=False)
        result = runner.invoke(api.app, argv + ["--allow-paid"])
        assert json.loads(result.stdout)["error"]["code"] == "credential_missing"
        monkeypatch.setenv("RIVER_API_KEY", "synthetic-key-for-unit-test")
    studio = Studio(root)
    studio.tab = "runs"
    studio.handle("r")
    text = "\n".join(studio.lines(110, 70))
    assert "NO enforced dollar ceiling" in text
    assert ("hosted_pointer" if provider == "river" else "local_weights") in text
    studio.handle("ENTER")
    assert len(calls) == 1
    replay = runner.invoke(api.app, argv + ["--allow-paid"])
    assert json.loads(replay.stdout)["result"]["already_submitted"] and len(calls) == 1


@pytest.mark.parametrize("provider", ["modal", "river"])
def test_remote_readiness_never_requires_local_cuda_or_contacts_a_provider(
    settings, tmp_path, monkeypatch, provider
):
    config = replace(settings, provider=provider, cost_policy="provider_billed")
    monkeypatch.setattr(
        "stateset_agents.finetune_inspection.subprocess.run",
        lambda *a, **k: pytest.fail(
            "no hardware or queue subprocess for remote readiness"
        ),
    )
    monkeypatch.setenv("RIVER_API_KEY", "synthetic-hidden-key")
    report = inspect_readiness(config, tmp_path)
    text = json.dumps(report)
    assert not report["provisions_hardware"] and "synthetic-hidden-key" not in text
    assert "NVIDIA hardware" not in text and "Training packages" not in text
    assert "Provider billing" in text and "no dollar ceiling" in text


def test_schema_exposes_backend_modes_cost_policy_and_constraints():
    result = CliRunner().invoke(api.app, ["schema"])
    schema = json.loads(result.stdout)["result"]
    assert set(schema["backends"]) == set(workspace.PROVIDERS)
    assert schema["backends"]["river"]["result_kind"] == "hosted_pointer"
    assert schema["backends"]["modal"]["methods"] == ["sft"]
    assert schema["config_schema"]["properties"]["cost_policy"]["enum"] == [
        "require_cap",
        "provider_billed",
    ]


def test_backend_setup_details_remain_readable_in_a_small_pane(settings, tmp_path):
    studio = Studio(tmp_path / "studio", settings)
    studio.handle("b")
    studio.backend_index = workspace.PROVIDERS.index("river")
    seen = []
    for _ in range(8):
        seen.extend(studio.lines(72, 22))
        studio.handle("PAGEDOWN")
    text = "\n".join(seen)
    assert "Setup:" in text and "Selection preserves training settings" in text
    assert "Enter choose backend" in text
    studio.handle("UP")
    assert studio.dialog_offset == 0
    studio.handle("ESC")
    assert studio.values["provider"] == "local" and not (tmp_path / "studio").exists()


@pytest.mark.parametrize(
    "change", [None, "uri", "model", "rank", "steps", "empty", "oversized"]
)
def test_river_hosted_result_is_verified_without_inventing_local_weights(
    settings, tmp_path, monkeypatch, change
):
    config = replace(settings, provider="river", cost_policy="provider_billed")
    run = workspace.create_run(tmp_path / "studio", workspace.plan_run(config))
    output = run / "artifacts"
    output.mkdir()
    pointer = {
        "provider": "river",
        "checkpoint": "river://synthetic-trained-checkpoint",
        "base_model": config.model,
        "lora": {"rank": config.lora_rank},
        "steps": 2,
        "num_epochs": config.epochs,
        "learning_rate": config.learning_rate,
    }
    if change == "uri":
        pointer["checkpoint"] = "not-a-checkpoint"
    elif change == "model":
        pointer["base_model"] = "other/model"
    elif change == "rank":
        pointer["lora"]["rank"] = 32
    elif change == "steps":
        pointer["steps"] = 0
    if change != "empty":
        write_json(output / "river_checkpoint.json", pointer)
    if change == "oversized":
        (output / "river_checkpoint.json").write_text(" " * (64 * 1024 + 1))
    monkeypatch.setattr(worker, "_train_remote", lambda *a: output)
    assert worker.execute(run) == (0 if change is None else 1)
    if change is None:
        receipt = read_json(run / "completion.json")
        assert receipt["result_kind"] == "hosted_pointer"
        assert (
            receipt["weights"] == [] and receipt["checkpoint"] == pointer["checkpoint"]
        )
        assert not receipt["quality_gain_verified"]
    else:
        assert not (run / "completion.json").exists()
        assert read_json(run / "failure.json")["code"] == "missing_weights"


def test_modal_does_not_accept_pointer_or_empty_output_as_weights(
    settings, tmp_path, monkeypatch
):
    config = replace(settings, provider="modal", cost_policy="provider_billed")
    run = workspace.create_run(tmp_path / "studio", workspace.plan_run(config))
    output = run / "artifacts"
    output.mkdir()
    write_json(output / "river_checkpoint.json", {"checkpoint": "river://synthetic"})
    monkeypatch.setattr(worker, "_train_remote", lambda *a: output)
    assert worker.execute(run) == 1
    assert read_json(run / "failure.json")["code"] == "missing_weights"


@pytest.mark.parametrize("count", [2, True])
def test_prepared_modal_review_uses_saved_hardware_and_rejects_malformed_counts(
    settings, tmp_path, count
):
    config = replace(settings, provider="modal", cost_policy="provider_billed")
    plan = workspace.plan_run(config)
    plan = replace(
        plan, resources={**plan.resources, "gpu": "H100", "gpu_count": count}
    )
    root = tmp_path / "studio"
    workspace.create_run(root, plan)
    studio = Studio(root)
    studio.tab = "runs"
    studio.handle("r")
    if type(count) is bool:
        assert studio.prepared_review is None
        assert "Malformed retained resource metadata" in studio.message
    else:
        assert studio.overlay == "prepared-review"
        text = "\n".join(studio.lines(110, 70))
        assert "GPU: 2 x H100" in text and "GPU: 1 x A10" not in text
        studio.handle("ESC")


def test_remote_supervisor_does_not_hold_buildr_slots(settings, tmp_path, monkeypatch):
    config = replace(settings, provider="modal", cost_policy="provider_billed")
    run = workspace.create_run(tmp_path / "studio", workspace.plan_run(config))
    monkeypatch.setattr(
        worker,
        "queue_executable",
        lambda: pytest.fail("no local queue reservation for remote training"),
    )

    def wait(**kwargs):
        assert kwargs["timeout"] is None
        return 1

    monkeypatch.setattr(
        worker.subprocess, "Popen", lambda *a, **k: SimpleNamespace(pid=123, wait=wait)
    )
    assert worker.supervise(run) == 1
    assert read_json(run / "command.json")["queue"] is None
