"""Known native-loader blockers fail before a Studio plan can spend compute."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from stateset_agents import finetune_inspection as inspection
from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace
from stateset_agents.remote import registry

NATIVE_BACKENDS = ("local", "runpod", "modal")
DEEPSEEK_MODELS = (
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "deepseek-v4.1-flash",
    "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "deepseek-v4.1-flash-1017",
)


def settings(model, provider):
    return workspace.FinetuneConfig(
        model=model,
        provider=provider,
        dataset="not-read.jsonl",
        cost_policy=(
            "provider_billed" if provider in {"modal", "river"} else "require_cap"
        ),
    )


@pytest.mark.parametrize("provider", NATIVE_BACKENDS)
@pytest.mark.parametrize("model", DEEPSEEK_MODELS)
def test_old_and_dated_model_ids_and_keys_report_controlled_native_blocker(
    model, provider
):
    config = settings(model, provider)
    _, issues = workspace.check_config_values(asdict(config))
    assert len(issues) == 1
    issue = issues[0]
    assert issue.field == "model" and issue.code == "blocked_training_layout"
    assert "unsupported training layout" in issue.message
    assert "frontier-models" in issue.message and model not in issue.message
    with pytest.raises(ValueError, match="unsupported training layout"):
        config.validate()


@pytest.mark.parametrize("provider", NATIVE_BACKENDS)
@pytest.mark.parametrize(
    "model", ["qwen3.8-flash-next", "Qwen/Qwen3.8-Flash-Next", "acme/unlisted-model"]
)
def test_native_catalog_and_unknown_models_keep_existing_validation(model, provider):
    config = settings(model, provider)
    assert workspace.check_config_values(asdict(config))[1] == []
    config.validate()


@pytest.mark.parametrize("model", DEEPSEEK_MODELS)
def test_river_access_remains_independent_of_native_checkpoint_layout(model):
    config = settings(model, "river")
    assert workspace.check_config_values(asdict(config))[1] == []
    config.validate()


@pytest.mark.parametrize("provider", NATIVE_BACKENDS)
def test_blocked_plan_cannot_read_dataset_create_run_or_submit_compute(
    tmp_path, monkeypatch, provider
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Blocked plan reached dataset, process or provider work")

    monkeypatch.setattr(workspace, "read_training_data", forbidden)
    monkeypatch.setattr(workspace.subprocess, "Popen", forbidden)
    monkeypatch.setattr(registry, "get_executor", forbidden)
    config = settings(DEEPSEEK_MODELS[-2], provider)
    target = tmp_path / "workspace"
    with pytest.raises(ValueError, match="unsupported training layout"):
        workspace.create_run(target, workspace.plan_run(config))
    assert not target.exists()


@pytest.mark.parametrize("provider", NATIVE_BACKENDS)
def test_readiness_reports_actionable_blocker_without_provider_or_gpu_queries(
    tmp_path, monkeypatch, provider
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Readiness attempted provider or hardware work")

    monkeypatch.setattr(inspection.shutil, "which", lambda _name: None)
    monkeypatch.setattr(inspection, "queue_executable", lambda: None)
    monkeypatch.setattr(inspection.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        inspection.shutil, "disk_usage", lambda _path: SimpleNamespace(free=2**40)
    )
    monkeypatch.setattr(inspection.subprocess, "run", forbidden)
    monkeypatch.setattr(registry, "get_executor", forbidden)
    report = inspection.inspect_readiness(
        settings(DEEPSEEK_MODELS[-2], provider), tmp_path
    )
    check = next(row for row in report["checks"] if row["name"] == "Settings")
    assert check["status"] == "fail" and "frontier-models" in check["detail"]
    assert report["failed"] >= 1 and not report["provisions_hardware"]


@pytest.mark.parametrize("provider", NATIVE_BACKENDS)
@pytest.mark.parametrize("entrypoint", ["launch", "execute"])
def test_saved_blocked_plan_is_revalidated_before_worker_or_provider_start(
    tmp_path, monkeypatch, provider, entrypoint
):
    source = tmp_path / "data.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "synthetic question"},
                    {"role": "assistant", "content": "synthetic response"},
                ]
            }
        )
        + "\n"
    )
    config = workspace.FinetuneConfig(
        dataset=str(source),
        provider=provider,
        cost_policy="provider_billed" if provider == "modal" else "require_cap",
    )
    run = workspace.create_run(tmp_path / "workspace", workspace.plan_run(config))
    plan = workspace.read_json(run / "plan.json")
    plan["config"]["model"] = DEEPSEEK_MODELS[-2]
    workspace.write_json(run / "plan.json", plan)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Saved blocked plan reached worker or provider work")

    monkeypatch.setattr(worker.os, "umask", lambda _mask: 0)
    monkeypatch.setattr(workspace.subprocess, "Popen", forbidden)
    monkeypatch.setattr(registry, "get_executor", forbidden)
    monkeypatch.setattr(worker, "_train_sft", forbidden)
    monkeypatch.setattr(worker, "_train_remote", forbidden)
    if entrypoint == "launch":
        with pytest.raises(ValueError, match="unsupported training layout"):
            workspace.launch_run(run)
        assert not (run / "admission.json").exists()
        assert not (run / "execution.json").exists()
    else:
        assert worker.execute(run) == 1
        assert workspace.read_json(run / "failure.json")["code"] == "review_invalid"
    assert workspace.read_json(run / "state.json")["status"] == "failed"
    assert not (run / "provider-job.json").exists()
    assert not (run / "completion.json").exists()


@pytest.mark.parametrize("provider", NATIVE_BACKENDS)
@pytest.mark.parametrize("model", ["glm5.3", "zai-org/GLM-5.3-Flash"])
def test_native_fp8_models_need_the_dedicated_recipe_before_studio_preparation(
    model, provider
):
    config = settings(model, provider)
    _, issues = workspace.check_config_values(asdict(config))
    assert len(issues) == 1
    assert issues[0].field == "model"
    assert issues[0].code == "requires_frontier_recipe"
    assert "frontier-sft" in issues[0].message
    with pytest.raises(ValueError, match="frontier-sft"):
        workspace.plan_run(config)
