"""Training continues where studio recovery is unsupported, and says so."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from stateset_agents import finetune_recovery as recovery
from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace


@pytest.fixture
def run(tmp_path, monkeypatch):
    source = tmp_path / "chat.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        )
        + "\n"
    )
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    config = workspace.FinetuneConfig(dataset=str(source))
    return workspace.create_run(tmp_path / "workspace", workspace.plan_run(config))


def _unsupported(monkeypatch, why):
    if why == "windows":
        monkeypatch.setattr(recovery, "sys", SimpleNamespace(platform="win32"))
    else:
        monkeypatch.setenv("WORLD_SIZE", "2")


@pytest.mark.parametrize("why", ["windows", "world_size"])
def test_unsupported_recovery_is_recorded_instead_of_failing_the_save(
    run, monkeypatch, why
):
    _unsupported(monkeypatch, why)
    checkpoint = run / "artifacts" / "checkpoint-1"
    worker._publish_recovery(run, checkpoint, step=1)  # used to raise ValueError
    state = workspace.read_json(run / "state.json")
    assert state["recovery_unavailable"] == recovery.recovery_unavailable_reason()
    assert "recovery_checkpoint" not in state
    assert not (run / "recovery").exists()
    with pytest.raises(ValueError):  # recovery itself still refuses
        recovery.publish_checkpoint(run, checkpoint, step=1)


@pytest.mark.parametrize("why", ["windows", "world_size"])
def test_local_sft_survives_its_first_epoch_save(run, monkeypatch, why):
    pytest.importorskip("transformers")
    from stateset_agents.training import sft

    _unsupported(monkeypatch, why)
    saves = []

    def fake_run_sft(*, callbacks, output_dir, **_kwargs):
        state = SimpleNamespace(global_step=4)
        (output_dir / "checkpoint-4").mkdir(parents=True)
        for callback in callbacks:
            callback.on_save(None, state, SimpleNamespace())
            saves.append(state.global_step)
        return output_dir

    monkeypatch.setattr(sft, "run_sft", fake_run_sft)
    monkeypatch.setattr(
        sft, "load_sft_dataset", lambda *_args, **_kwargs: ([], {"sha256": "x"})
    )
    config = workspace.FinetuneConfig(**workspace.read_run_plan(run)[0]["config"])
    assert worker._train_sft(run, config) == run / "artifacts"
    assert saves == [4]
    state = workspace.read_json(run / "state.json")
    assert "unavailable" not in str(state.get("recovery_checkpoint", ""))
    assert state["recovery_unavailable"]


def test_cli_status_reports_unavailable_recovery(run, monkeypatch):
    from typer.testing import CliRunner

    from stateset_agents import cli_finetune as cli

    _unsupported(monkeypatch, "world_size")
    worker._publish_recovery(run, run / "artifacts" / "checkpoint-1", step=1)
    response = CliRunner().invoke(
        cli.app, ["status", run.name, "--workspace", str(run.parent)]
    )
    state = json.loads(response.stdout)["result"]["run"]["state"]
    assert "single training process" in state["recovery_unavailable"]
