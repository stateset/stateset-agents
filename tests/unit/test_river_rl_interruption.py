"""Interruptions must retain committed progress and accurate terminal evidence."""

import json
from asyncio import CancelledError
from dataclasses import replace

import pytest

from stateset_agents.remote import river_rl_runner as runner
from stateset_agents.remote.job import JobHandle, JobStatus
from stateset_agents.remote.river import RiverExecutor
from tests.unit.test_river_submit_golden import stub_river_renderers  # noqa: F401
from tests.unit.test_river_submit_golden import (
    RecordingClient,
    RecordingModel,
    scenario_rl,
)


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, CancelledError])
@pytest.mark.parametrize("stage", ["initial_sample", "update", "save"])
def test_interrupt_publishes_only_committed_progress_and_resumes(
    tmp_path, interrupt_type, stage
):
    spec, _ = scenario_rl(tmp_path)
    interrupt = interrupt_type("private provider metadata")

    class InterruptedModel(RecordingModel):
        updates = 0

        def sample(self, *args, **kwargs):
            if stage == "initial_sample":
                raise interrupt
            return super().sample(*args, **kwargs)

        def optim_step(self, **kwargs):
            type(self).updates += 1
            result = super().optim_step(**kwargs)
            if stage == "update" and self.updates == 2:
                raise interrupt  # Remote update may have completed.
            return result

        def save_weights(self, name, **kwargs):
            result = super().save_weights(name, **kwargs)
            if stage == "save" and name == "rl-state-2":
                raise interrupt  # Save reply lost before the local commit.
            return result

    client = RecordingClient(model_cls=InterruptedModel)
    ledger = tmp_path / "costs.jsonl"
    executor = RiverExecutor(client=client, ledger_path=ledger)
    with pytest.raises(interrupt_type) as caught:
        executor.submit(spec)
    assert caught.value is interrupt
    handle = JobHandle("river", "river-1")
    assert executor.status(handle) is JobStatus.CANCELLED
    assert client.calls[-1] == {"call": "session_close"}
    report = json.loads((spec.output_dir / "rl_report.json").read_text())
    assert report["status"] == "cancelled"
    assert "private provider metadata" not in json.dumps(report)
    usage_before_resume = json.loads((spec.output_dir / "rl_usage.json").read_text())
    assert report["usage"] == usage_before_resume
    assert usage_before_resume["generated_tokens_or_reserved"] > 0
    assert json.loads(ledger.read_text())["status"] == "cancelled"
    state_path = spec.output_dir / "rl_state.json"
    if stage == "initial_sample":
        assert not state_path.exists()
        assert "completed_round" not in report
        assert not (spec.output_dir / "river_checkpoint.json").exists()
    else:
        state = json.loads(state_path.read_text())
        assert state["completed_round"] == report["completed_round"] == 1
        assert report["steps"] == state["steps"] == 1
        assert report["training_checkpoint"].endswith("rl-state-1")
        assert report["rounds"] == state["rounds"]

    # Fresh provider session restores the last commit, never a partial update.
    restored_client = RecordingClient(model_cls=RecordingModel)
    resumed = RiverExecutor(client=restored_client, ledger_path=ledger)
    resumed.submit(replace(spec, resume=True))
    final = json.loads(state_path.read_text())
    assert final["completed_round"] == final["steps"] == 2
    updates = [c for c in restored_client.calls if c["call"] == "optim_step"]
    assert len(updates) == (2 if stage == "initial_sample" else 1)
    if stage != "initial_sample":
        created = next(c for c in restored_client.calls if c["call"] == "create_model")
        assert "rl-state-1" in created["checkpoint"]
    usage_after_resume = json.loads((spec.output_dir / "rl_usage.json").read_text())
    assert (
        usage_after_resume["generated_tokens_or_reserved"]
        >= usage_before_resume["generated_tokens_or_reserved"]
    )
    assert (
        json.loads((spec.output_dir / "rl_report.json").read_text())["status"]
        == "succeeded"
    )


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, CancelledError])
@pytest.mark.parametrize("failure", ["state", "report", "ledger", "report_and_ledger"])
def test_diagnostic_failures_preserve_interrupt_and_never_modify_commit(
    tmp_path, monkeypatch, interrupt_type, failure
):
    spec, _ = scenario_rl(tmp_path)
    state_path = spec.output_dir / "rl_state.json"
    usage_path = spec.output_dir / "rl_usage.json"
    interrupt = interrupt_type("private interruption context")
    captured = {}

    class InterruptedModel(RecordingModel):
        updates = 0

        def optim_step(self, **kwargs):
            type(self).updates += 1
            result = super().optim_step(**kwargs)
            if self.updates == 2:
                if failure == "state":
                    state_path.write_text("{malformed")
                captured["state"] = state_path.read_bytes()
                captured["usage"] = usage_path.read_bytes()
                raise interrupt
            return result

    original_write = runner.atomic_json

    def write(path, payload):
        if (
            failure in ("report", "report_and_ledger")
            and path.name == "rl_report.json"
            and payload.get("status") == "cancelled"
        ):
            raise OSError("private filesystem context")
        return original_write(path, payload)

    monkeypatch.setattr(runner, "atomic_json", write)
    client = RecordingClient(model_cls=InterruptedModel)
    ledger = tmp_path / "costs.jsonl"
    executor = RiverExecutor(client=client, ledger_path=ledger)

    def fail_ledger(*args):
        raise RuntimeError("private ledger context")

    if failure in ("ledger", "report_and_ledger"):
        monkeypatch.setattr(executor, "_record_cost", fail_ledger)
    with pytest.raises(interrupt_type) as caught:
        executor.submit(spec)
    assert caught.value is interrupt
    handle = JobHandle("river", "river-1")
    assert executor.status(handle) is JobStatus.CANCELLED
    assert state_path.read_bytes() == captured["state"]
    assert usage_path.read_bytes() == captured["usage"]
    logs = "\n".join(executor.logs(handle))
    assert "private" not in logs
    if failure in ("report", "report_and_ledger"):
        assert "durable status may be stale" in logs
    else:
        report = json.loads((spec.output_dir / "rl_report.json").read_text())
        assert report["status"] == "cancelled"
        if failure == "state":
            assert "completed_round" not in report
            assert "best_checkpoint" not in report
            before = list(client.calls)
            with pytest.raises(ValueError):
                executor.submit(replace(spec, resume=True))
            assert client.calls == before
    if failure in ("ledger", "report_and_ledger"):
        assert "cost record could not be written" in logs
    else:
        assert json.loads(ledger.read_text())["status"] == "cancelled"


def test_interrupt_after_final_commit_can_finish_offline(tmp_path, monkeypatch):
    spec, client = scenario_rl(tmp_path)
    interrupt = KeyboardInterrupt()
    original_write = runner.atomic_json
    interrupted = False

    def write(path, payload):
        nonlocal interrupted
        if (
            path.name == "rl_report.json"
            and payload.get("status") == "succeeded"
            and not interrupted
        ):
            interrupted = True
            raise interrupt
        return original_write(path, payload)

    monkeypatch.setattr(runner, "atomic_json", write)
    executor = RiverExecutor(client=client, ledger_path=tmp_path / "costs.jsonl")
    with pytest.raises(KeyboardInterrupt):
        executor.submit(spec)
    report = json.loads((spec.output_dir / "rl_report.json").read_text())
    assert report["status"] == "cancelled"
    assert report["completed_round"] == 2
    calls = list(client.calls)
    handle = executor.submit(replace(spec, resume=True))
    assert executor.status(handle) is JobStatus.SUCCEEDED
    assert client.calls == calls
    assert (
        json.loads((spec.output_dir / "rl_report.json").read_text())["status"]
        == "succeeded"
    )
