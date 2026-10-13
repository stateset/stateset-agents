"""Recovery must restore a coherent commit before trusting progress or weights."""

import json
from dataclasses import replace

import pytest

from stateset_agents.remote.executor import RemoteExecutionError
from tests.unit.test_river_rl_contract import _executor
from tests.unit.test_river_submit_golden import stub_river_renderers  # noqa: F401
from tests.unit.test_river_submit_golden import (
    RecordingClient,
    RecordingModel,
    RiverConnectionError,
    scenario_rl,
)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_checkpoint",
        "empty_checkpoint",
        "invalid_checkpoint",
        "schema",
        "boolean_schema",
        "future_round",
        "boolean_round",
        "fractional_round",
        "steps",
        "best_round",
        "best_steps",
        "loss",
        "best_loss",
        "best_eval",
        "missing_round",
        "duplicate_round",
        "reordered_rounds",
        "missing_metrics",
        "nonfinite_metrics",
        "skipped_update",
        "negative_datums",
        "empty_results",
        "false_result",
        "boolean_pass_count",
        "wrong_shape",
        "malformed",
    ],
)
def test_corrupt_resume_fails_before_provider_calls_and_preserves_artifacts(
    tmp_path, corruption
):
    spec, client = scenario_rl(tmp_path)
    executor = _executor(tmp_path, client)
    executor.submit(spec)
    path = spec.output_dir / "rl_state.json"
    state = json.loads(path.read_text())
    replacements = {
        "missing_checkpoint": ("training_checkpoint", None),
        "empty_checkpoint": ("best_checkpoint", ""),
        "invalid_checkpoint": ("training_checkpoint", "file:///wrong-model"),
        "schema": ("schema_version", 3),
        "boolean_schema": ("schema_version", True),
        "future_round": ("completed_round", 3),
        "boolean_round": ("completed_round", True),
        "fractional_round": ("completed_round", 2.5),
        "steps": ("steps", 1),
        "best_round": ("best_round", 2),  # Tied validation must keep the baseline.
        "best_steps": ("best_steps", 1),
        "loss": ("loss", None),
        "best_loss": ("best_loss", 0.0),
        "best_eval": ("best_eval", None),
    }
    if corruption in replacements:
        key, value = replacements[corruption]
        state[key] = value
    elif corruption == "missing_round":
        state["rounds"].pop()
    elif corruption == "duplicate_round":
        state["rounds"].append(state["rounds"][-1])
    elif corruption == "reordered_rounds":
        state["rounds"].reverse()
    elif corruption == "missing_metrics":
        state["rounds"][1].pop("backward_metrics")
    elif corruption == "nonfinite_metrics":
        state["rounds"][1]["backward_metrics"][0]["loss_sum"] = float("nan")
    elif corruption == "skipped_update":
        state["rounds"][1]["skipped"] = True
    elif corruption == "negative_datums":
        state["rounds"][1]["datums"] = -1
    elif corruption == "empty_results":
        state["best_eval"]["results"] = []
    elif corruption == "false_result":
        state["best_eval"]["results"][0]["checks"]["passed"] = False
    elif corruption == "boolean_pass_count":
        state["best_eval"]["passed"] = True
    elif corruption == "wrong_shape":
        state = []
    path.write_text("{" if corruption == "malformed" else json.dumps(state))
    before = {p.name: p.read_bytes() for p in spec.output_dir.glob("*.json")}
    calls = list(client.calls)
    with pytest.raises(ValueError):
        executor.submit(replace(spec, resume=True))
    assert client.calls == calls
    assert {p.name: p.read_bytes() for p in spec.output_dir.glob("*.json")} == before


def test_missing_committed_state_cannot_restart_from_initial_model(tmp_path):
    spec, client = scenario_rl(tmp_path)
    executor = _executor(tmp_path, client)
    executor.submit(spec)
    (spec.output_dir / "rl_state.json").unlink()
    calls = list(client.calls)
    usage = (spec.output_dir / "rl_usage.json").read_bytes()
    with pytest.raises(ValueError, match="committed state is missing"):
        executor.submit(replace(spec, resume=True))
    assert client.calls == calls
    assert (spec.output_dir / "rl_usage.json").read_bytes() == usage
    assert not (spec.output_dir / "rl_state.json").exists()


@pytest.mark.parametrize("remove", [True, False])
def test_transient_retry_revalidates_state_without_reopening_provider(tmp_path, remove):
    spec, _ = scenario_rl(tmp_path)
    path = spec.output_dir / "rl_state.json"

    class Interrupted(RecordingModel):
        def forward_backward(self, *args, **kwargs):
            if remove:
                path.unlink()
            else:
                state = json.loads(path.read_text())
                state["training_checkpoint"] = None
                path.write_text(json.dumps(state))
            raise RiverConnectionError("interrupted update")

    client = RecordingClient(model_cls=Interrupted)
    with pytest.raises(RemoteExecutionError, match="Invalid River RL state"):
        _executor(tmp_path, client).submit(spec)
    assert sum(c["call"] == "create_model" for c in client.calls) == 1
    assert not any(c["call"] == "optim_step" for c in client.calls)
    report = json.loads((spec.output_dir / "rl_report.json").read_text())
    assert report["status"] == "failed"
    assert "best_checkpoint" not in report and "completed_round" not in report


def test_invalid_saved_checkpoint_cannot_replace_previous_commit(tmp_path):
    class BrokenCheckpoint(RecordingModel):
        def save_weights(self, name, **kwargs):
            saved = super().save_weights(name, **kwargs)
            return "invalid-checkpoint" if name == "rl-state-1" else saved

    spec, _ = scenario_rl(tmp_path)
    client = RecordingClient(model_cls=BrokenCheckpoint)
    with pytest.raises(RemoteExecutionError, match="training_checkpoint"):
        _executor(tmp_path, client).submit(spec)
    state = json.loads((spec.output_dir / "rl_state.json").read_text())
    assert state["completed_round"] == 0 and state["steps"] == 0
    assert state["training_checkpoint"].endswith("rl-state-0")


def test_no_validation_uses_latest_update_and_resumes_without_provider_calls(tmp_path):
    spec, client = scenario_rl(tmp_path)
    spec = replace(spec, eval_prompts=[])
    executor = _executor(tmp_path, client)
    executor.submit(spec)
    state = json.loads((spec.output_dir / "rl_state.json").read_text())
    assert state["best_round"] == state["completed_round"] == 2
    assert state["best_eval"] is None
    calls = list(client.calls)
    executor.submit(replace(spec, resume=True))
    assert client.calls == calls


def test_failure_before_first_commit_can_resume_without_resetting_usage(tmp_path):
    class FailFirstSave(RecordingModel):
        failed = False

        def save_weights(self, *args, **kwargs):
            saved = super().save_weights(*args, **kwargs)
            if not type(self).failed:
                type(self).failed = True
                raise RuntimeError("lost initial checkpoint response")
            return saved

    spec, _ = scenario_rl(tmp_path)
    client = RecordingClient(model_cls=FailFirstSave)
    executor = _executor(tmp_path, client)
    with pytest.raises(RemoteExecutionError, match="initial checkpoint"):
        executor.submit(spec)
    assert not (spec.output_dir / "rl_state.json").exists()
    usage_path = spec.output_dir / "rl_usage.json"
    prior_usage = json.loads(usage_path.read_text())["generated_tokens_or_reserved"]
    assert prior_usage > 0
    executor.submit(replace(spec, resume=True))
    state = json.loads((spec.output_dir / "rl_state.json").read_text())
    assert state["completed_round"] == 2
    assert (
        json.loads(usage_path.read_text())["generated_tokens_or_reserved"] > prior_usage
    )
