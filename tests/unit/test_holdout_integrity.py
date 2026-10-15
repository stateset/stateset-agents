"""Incomplete training and uncertain tests must never silently reopen holdouts."""

import asyncio
import json
import subprocess
import sys
import textwrap
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.checkpoint_selection import (
    require_training_progress,
    select_validation_checkpoint,
)
from stateset_agents.training.river_refund import campaign, prepare_run
from tests.unit.river_fakes import traced_refund_trajectory

VALIDATION_CASES = {
    "validation-A": {"order_id": "validation-A", "amount_cents": 100, "eligible": True}
}


def evaluation(step, reward=0):
    return {
        "step": step,
        "checkpoint": {"path": f"river://step-{step}"},
        "metrics": {"reward_mean": reward},
        "case_hashes": {
            key: content_hash(row) for key, row in VALIDATION_CASES.items()
        },
        "outcomes": [
            {
                "case_id": "validation-A",
                "case_hash": content_hash(VALIDATION_CASES["validation-A"]),
                "reward": reward,
            }
        ],
    }


def test_validation_selection_requires_complete_unambiguous_evidence():
    values = [evaluation(2, 0.8), evaluation(0, 0.5), evaluation(1, 0.8)]
    assert (
        select_validation_checkpoint(values, steps=2, cases=VALIDATION_CASES)["step"]
        == 1
    )
    for invalid in [
        values[:2],
        [*values, values[0]],
        None,
        [evaluation(0), evaluation(1), evaluation(2, float("nan"))],
        [evaluation(0), evaluation(1), evaluation(2, True)],
    ]:
        with pytest.raises(ValueError):
            select_validation_checkpoint(invalid, steps=2, cases=VALIDATION_CASES)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "foreign", "hash", "mean", "bool", "nan", "huge", "uri"],
)
def test_validation_selection_rejects_invalid_case_evidence(mutation):
    cases = {**VALIDATION_CASES, "validation-B": {"order_id": "validation-B"}}
    values = [evaluation(step) for step in range(3)]
    for item in values:
        item["case_hashes"] = {key: content_hash(row) for key, row in cases.items()}
        item["outcomes"].append(
            {
                "case_id": "validation-B",
                "case_hash": content_hash(cases["validation-B"]),
                "reward": 0,
            }
        )
    target = values[1]
    if mutation == "missing":
        target["outcomes"].pop()
    elif mutation == "duplicate":
        target["outcomes"][1]["case_id"] = "validation-A"
    elif mutation == "foreign":
        target["outcomes"][1]["case_id"] = "test-A"
    elif mutation == "hash":
        target["case_hashes"]["validation-A"] = "0" * 64
    elif mutation == "mean":
        target["metrics"]["reward_mean"] = 1
    elif mutation == "uri":
        target["checkpoint"]["path"] = "river://"
    else:
        target["outcomes"][0]["reward"] = {
            "bool": True,
            "nan": float("nan"),
            "huge": 10**1000,
        }[mutation]
    with pytest.raises(ValueError):
        select_validation_checkpoint(values, steps=2, cases=cases)


def test_validation_legacy_aggregates_do_not_prove_case_coverage():
    values = [evaluation(step) for step in range(3)]
    for item in values:
        del item["outcomes"]
        del item["case_hashes"]
    with pytest.raises(ValueError, match="Validation cases"):
        select_validation_checkpoint(values, steps=2, cases=VALIDATION_CASES)


@pytest.mark.parametrize(
    "progress",
    [
        None,
        [],
        [{"step": True}, {"step": 2}],
        [{"step": 2}, {"step": 1}],
        [{"step": 1}],
        [{"step": 1}, {"step": 1}],
    ],
)
def test_incomplete_training_does_not_unlock_test(progress):
    with pytest.raises(ValueError, match="training progress"):
        require_training_progress(progress, steps=2)


@pytest.fixture
def native_run(tmp_path, monkeypatch):
    state = SimpleNamespace(mode="complete", test_engines=0, sampled=0)

    @dataclass
    class Checkpoint:
        path: str

    class Engine:
        def __init__(self, model, **kwargs):
            if hasattr(model, "checkpoint"):
                assert (tmp_path / "test_attempt.json").exists()
                state.test_engines += 1

        async def rollout(self, rows, **kwargs):
            state.sampled += 1
            if state.mode == "test_failure":
                raise ConnectionError("uncertain test request")
            if state.mode == "cancelled":
                raise asyncio.CancelledError
            trajectory = await traced_refund_trajectory(
                rows[0],
                successful=False,
                environment="refund-v1",
                termination="environment_timeout" if state.mode == "timeout" else None,
            )
            trajectory.generated_tokens, trajectory.elapsed = 4, 0.1
            if state.mode == "test_wrong_input":
                trajectory.stateset_case_hash = "0" * 64
            if state.mode == "test_wrong_metrics":
                trajectory.metrics["tool_calls"] = 0
            if state.mode == "test_wrong_action":
                trajectory.stateset_environment_trace["steps"][0]["action"][
                    "content"
                ] = "not an action"
            yield [trajectory]

    async def validation_trajectories(reward):
        result = [
            await traced_refund_trajectory(
                VALIDATION_CASES["validation-A"],
                successful=reward > 0,
                environment="refund-v1",
            )
        ]
        # Deliberately corrupt reward evidence in this SDK fixture when requested.
        result[0].reward = reward
        if state.mode == "partial_cases":
            return []
        if state.mode == "duplicate_cases":
            return result * 2
        if state.mode == "unknown_case":
            result[0].stateset_episode_id = "test-A"
        if state.mode == "wrong_input":
            result[0].stateset_case_hash = "0" * 64
        if state.mode == "missing_input":
            del result[0].stateset_case_hash
        if state.mode == "missing_trace":
            del result[0].stateset_environment_trace
        if state.mode == "tampered_validation_action":
            result[0].stateset_environment_trace["steps"][0]["action"][
                "content"
            ] = "not an action"
        if state.mode == "inflated_mean":
            result[0].reward = reward - 1
        return result

    class Trainer:
        def __init__(self, **kwargs):
            self.sink = kwargs["evaluator"].sink

        async def run(self, rows, steps, after_recovery=None):
            if state.mode == "recovered_complete":
                await after_recovery(steps)
            elif state.mode == "recovered_partial":
                await after_recovery(1)
            for step in range(3):
                if state.mode == "partial" and step == 2:
                    return
                if not (state.mode == "missing_validation" and step == 1):
                    reward = (
                        float("nan")
                        if state.mode == "nan_validation"
                        else (1.0 if step == 2 else -1.0)
                    )
                    await self.sink(
                        SimpleNamespace(
                            step=step,
                            checkpoint=Checkpoint(f"river://step-{step}"),
                            metrics={"reward_mean": reward},
                            trajectories=await validation_trajectories(reward),
                        )
                    )
                if state.mode == "replay" and step == 1:
                    await self.sink(
                        SimpleNamespace(
                            step=1,
                            checkpoint=Checkpoint("river://stale"),
                            metrics={"reward_mean": 1},
                            trajectories=await validation_trajectories(1),
                        )
                    )
                    await self.sink(
                        SimpleNamespace(
                            step=1,
                            checkpoint=Checkpoint("river://replayed"),
                            metrics={"reward_mean": -1.0},
                            trajectories=await validation_trajectories(-1.0),
                        )
                    )
                if (
                    step
                    and state.mode != "recovered_complete"
                    and not (state.mode == "recovered_partial" and step == 1)
                ):
                    yield SimpleNamespace(
                        n=step,
                        metrics=(
                            {"loss": state.invalid_training_metric}
                            if state.mode == "invalid_training_metric"
                            else {"loss": 0.1}
                        ),
                    )

    rl = SimpleNamespace(
        Env=object,
        Budget=lambda **kw: kw,
        Schedule=lambda **kw: kw,
        GroupCompletion=lambda **kw: kw,
        RolloutEngine=Engine,
        CheckpointSampler=lambda *a, **kw: SimpleNamespace(**kw),
        AsyncTrainer=Trainer,
        Evaluator=lambda *a, **kw: SimpleNamespace(**kw),
        Adam=lambda **kw: kw,
        GroupCentered=lambda: None,
        Truncation=lambda **kw: kw,
        Checkpointing=lambda *a, **kw: kw,
    )
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=rl, Checkpoint=Checkpoint)
    )
    args = SimpleNamespace(
        output=tmp_path,
        base_model="base",
        seed=42,
        steps=2,
        concurrency=8,
        max_staleness=0,
        learning_rate=1e-5,
        checkpoint=None,
        evaluate_only=False,
        dry_run=False,
    )
    splits = {
        "train": [{"order_id": "train-A", "amount_cents": 100, "eligible": True}],
        "validation": list(VALIDATION_CASES.values()),
        "test": [{"order_id": "test-A", "amount_cents": 100, "eligible": True}],
    }
    prepare_run(args, splits)
    model = SimpleNamespace(save_weights=lambda *a, **kw: Checkpoint("river://base"))

    async def run():
        await campaign(
            model, object(), SimpleNamespace(tokenizer=object()), args, splits
        )

    return state, args, splits, run


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "recovered,observed,expected_steps",
    [(None, 0, [1, 2]), (1, 1, [2]), (2, 2, []), (0, 2, [1, 2]), (1, 0, [2, 3])],
)
async def test_zero_update_stop_uses_reconciled_records_and_closes_trainer(
    native_run, monkeypatch, recovered, observed, expected_steps
):
    from stateset_agents.training.river_progress import RiverTrainingProgress

    _, args, splits, run = native_run
    args.output = args.output / "early-stop"
    args.output.mkdir()
    args.steps = 3
    args.zero_update_patience = 2
    prepare_run(args, splits)
    manifest_hash = content_hash(
        json.loads((args.output / "run_manifest.json").read_text())
    )
    ledger = RiverTrainingProgress(
        args.output, steps=3, run_manifest_hash=manifest_hash
    )
    for step in range(1, observed + 1):
        ledger.observe(step, {"train/updated": 0, "train/datums": 0})
    emitted, closed = [], []

    async def train(self, rows, steps, after_recovery):
        try:
            if recovered is not None:
                await after_recovery(recovered)
            for step in range((recovered or 0) + 1, steps + 1):
                emitted.append(step)
                yield SimpleNamespace(n=step, metrics={"train/updated": 0})
        finally:
            closed.append(True)

    monkeypatch.setattr(sys.modules["river_client"].rl.AsyncTrainer, "run", train)
    with pytest.raises(ValueError, match="2 consecutive skipped optimizer updates"):
        await run()
    assert emitted == expected_steps and closed == [True]
    records = json.loads((args.output / "training_metrics.json").read_text())
    stop = json.loads((args.output / "training_stop.json").read_text())
    assert stop["step"] == len(records)
    assert stop["patience"] == 2
    assert stop["run_manifest_hash"] == manifest_hash
    assert stop["training_progress_hash"] == content_hash(records)
    assert not (args.output / "test_attempt.json").exists()
    assert not (args.output / "test_results.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
async def test_validation_writes_serialize_with_callbacks_and_recovery(
    native_run, monkeypatch, recover
):
    import stateset_agents.training.river_refund as module

    _, args, splits, run = native_run
    trainer = sys.modules["river_client"].rl.AsyncTrainer
    checkpoint = sys.modules["river_client"].Checkpoint
    original_write = module.atomic_json
    entered, second_started = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    writes = []

    def write(path, value):
        if path.name == "validation_results.json":
            writes.append([item["step"] for item in value])
            if len(writes) == 1:
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(5), "validation write was not released"
        return original_write(path, value)

    async def exercise(self, rows, steps, after_recovery=None):
        trajectory = await traced_refund_trajectory(
            splits["validation"][0], successful=False, environment="refund-v1"
        )

        def result(step):
            return SimpleNamespace(
                step=step,
                checkpoint=checkpoint(f"river://step-{step}"),
                metrics={"reward_mean": trajectory.reward},
                trajectories=[trajectory],
            )

        first = asyncio.create_task(self.sink(result(0)))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), 2)

            async def another():
                second_started.set()
                if recover:
                    await after_recovery(1)
                else:
                    await self.sink(result(1))

            second = asyncio.create_task(another())
            await second_started.wait()
            # The first write remains blocked while the second callback reaches
            # its lock. Neither the shared list nor the file may change yet.
            assert writes == [[0]]
            assert not second.done()
        finally:
            release.set()
            await asyncio.gather(first, *([second] if second is not None else []))
        raise RuntimeError("finished write exercise")
        yield  # Keep the trainer's async-generator contract.

    monkeypatch.setattr(module, "atomic_json", write)
    monkeypatch.setattr(trainer, "run", exercise)
    with pytest.raises(RuntimeError, match="finished write exercise"):
        await run()
    expected = [0] if recover else [0, 1]
    assert writes == [[0], expected]
    assert [
        entry["step"]
        for entry in json.loads((args.output / "validation_results.json").read_text())
    ] == expected
    assert not (args.output / "test_attempt.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_cleanup", [False, True])
async def test_training_record_failure_drains_iterator_before_campaign_returns(
    native_run, monkeypatch, cancel_cleanup
):
    import stateset_agents.training.river_refund as module

    state, args, _, run = native_run
    trainer = sys.modules["river_client"].rl.AsyncTrainer
    original = trainer.run
    entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    providers_closed = []

    async def tracked(self, *a, **kw):
        iterator = original(self, *a, **kw)
        try:
            async for step in iterator:
                yield step
        finally:
            entered.set()
            await release.wait()
            await iterator.aclose()
            closed.set()

    def fail_record(*a, **kw):
        raise OSError("progress storage failed")

    monkeypatch.setattr(trainer, "run", tracked)
    monkeypatch.setattr(module.RiverTrainingProgress, "observe", fail_record)

    async def caller():
        try:
            await run()
        finally:
            providers_closed.append(closed.is_set())

    task = asyncio.create_task(caller())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not task.done() and providers_closed == []
        if cancel_cleanup:
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done() and not closed.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError if cancel_cleanup else OSError):
            await asyncio.wait_for(task, 3)
    assert providers_closed == [True]
    assert state.test_engines == 0
    assert not (args.output / "test_attempt.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.1"])
async def test_invalid_training_metrics_never_unlock_holdout(native_run, value):
    state, args, _, run = native_run
    state.mode = "invalid_training_metric"
    state.invalid_training_metric = value
    with pytest.raises(ValueError, match="Training metrics"):
        await run()
    assert state.test_engines == 0
    assert not (args.output / "test_attempt.json").exists()
    assert not (args.output / "test_results.json").exists()
    assert not (args.output / "training_metrics.json").exists()


@pytest.mark.asyncio
async def test_test_report_counts_environment_timeout_as_truncation(native_run):
    state, args, _, run = native_run
    state.mode = "timeout"
    await run()
    report = json.loads((args.output / "test_results.json").read_text())
    assert report["truncation_rate"] == 1
    assert report["outcomes"][0]["truncated"] == "environment_timeout"


@pytest.mark.asyncio
async def test_source_drift_stops_before_sdk_import_or_sampling(
    native_run, monkeypatch
):
    import sys

    from stateset_agents.training.river_refund import execute_run

    state, args, splits, run = native_run
    path = args.output / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["implementation"]["files"]["__init__.py"] = "0" * 64
    path.write_text(json.dumps(manifest))
    monkeypatch.setitem(sys.modules, "river_client", None)
    with pytest.raises(ValueError, match="implementation differs"):
        execute_run(args, splits)
    with pytest.raises(ValueError, match="implementation differs"):
        await run()
    with pytest.raises(ValueError):
        prepare_run(args, splits)
    assert state.test_engines == state.sampled == 0
    assert not (args.output / "test_attempt.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 43),
        ("base_model", "other"),
        ("steps", 3),
        ("concurrency", 9),
        ("max_staleness", 1),
        ("learning_rate", 2e-5),
        ("checkpoint", "river://other"),
        ("evaluate_only", True),
        ("rollout_token_budget", 1024),
        ("benchmark", "refund-policy-v2"),
    ],
)
async def test_execution_rejects_configuration_drift_before_sdk(
    native_run, monkeypatch, field, value
):
    from stateset_agents.training.river_refund import execute_run

    state, args, splits, run = native_run
    setattr(args, field, value)
    before = {p: p.read_bytes() for p in args.output.iterdir()}
    monkeypatch.setitem(sys.modules, "river_client", None)
    with pytest.raises(ValueError, match="configuration or cases differ"):
        execute_run(args, splits)
    with pytest.raises(ValueError, match="configuration or cases differ"):
        await run()
    assert {p: p.read_bytes() for p in args.output.iterdir()} == before
    assert state.test_engines == state.sampled == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("split", ["train", "validation", "test"])
@pytest.mark.parametrize("source", ["caller", "saved", "missing", "unreadable"])
async def test_execution_requires_unchanged_prepared_cases(
    native_run, monkeypatch, split, source
):
    from stateset_agents.training.river_refund import execute_run

    state, args, splits, run = native_run
    path = args.output / f"{split}.json"
    if source == "caller":
        splits[split] = [{**row, "extra": "changed"} for row in splits[split]]
    elif source == "saved":
        path.write_text("[]")
    elif source == "missing":
        path.unlink()
    else:
        path.write_text("{")
    before = {p: p.read_bytes() for p in args.output.iterdir()}
    monkeypatch.setitem(sys.modules, "river_client", None)
    with pytest.raises(ValueError, match="cases"):
        execute_run(args, splits)
    with pytest.raises(ValueError, match="cases"):
        await run()
    assert {p: p.read_bytes() for p in args.output.iterdir()} == before
    assert state.test_engines == state.sampled == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "artifact,reason",
    [
        ("test_attempt.json", "sealed"),
        ("test_results.json", "completed"),
        ("training_candidates.json", "Collection output already exists"),
    ],
)
async def test_direct_execution_honors_existing_evidence_seals(
    native_run, monkeypatch, artifact, reason
):
    from stateset_agents.training.river_refund import execute_run

    state, args, splits, run = native_run
    (args.output / artifact).write_text("{}")
    before = {p: p.read_bytes() for p in args.output.iterdir()}
    monkeypatch.setitem(sys.modules, "river_client", None)
    with pytest.raises(ValueError, match=reason):
        execute_run(args, splits)
    with pytest.raises(ValueError, match=reason):
        await run()
    assert {p: p.read_bytes() for p in args.output.iterdir()} == before
    assert state.test_engines == state.sampled == 0


@pytest.mark.asyncio
async def test_running_campaign_owns_configuration_and_nested_cases(
    native_run, monkeypatch
):
    state, args, splits, run = native_run
    output = args.output
    original = sys.modules["river_client"].rl.AsyncTrainer.run
    started, release = asyncio.Event(), asyncio.Event()

    async def paused(self, rows, steps, after_recovery=None):
        started.set()
        await release.wait()
        async for step in original(self, rows, steps, after_recovery):
            yield step

    monkeypatch.setattr(sys.modules["river_client"].rl.AsyncTrainer, "run", paused)
    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        args.seed = 999
        args.steps = 100
        args.output = output / "wrong-output"
        splits["test"][0]["amount_cents"] = 999
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=5)
    report = json.loads((output / "test_results.json").read_text())
    assert report["seed"] == 42
    assert report["case_hashes"]["test-A"] == content_hash(
        {"order_id": "test-A", "amount_cents": 100, "eligible": True}
    )
    assert not args.output.exists()
    assert state.sampled == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "partial",
        "missing_validation",
        "nan_validation",
        "partial_cases",
        "duplicate_cases",
        "unknown_case",
        "inflated_mean",
        "wrong_input",
        "missing_input",
        "missing_trace",
        "tampered_validation_action",
    ],
)
async def test_runner_refuses_test_before_complete_training_and_validation(
    native_run, mode
):
    state, args, splits, run = native_run
    state.mode = mode
    with pytest.raises(ValueError):
        await run()
    assert state.test_engines == state.sampled == 0
    assert not (args.output / "test_attempt.json").exists()
    # Training can resume; the test split has not been consumed.
    prepare_run(args, splits)
    state.mode = "complete"
    await run()
    assert state.sampled == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,error",
    [("test_failure", ConnectionError), ("cancelled", asyncio.CancelledError)],
)
async def test_uncertain_test_is_durably_sealed(native_run, mode, error):
    state, args, splits, run = native_run
    state.mode = mode
    with pytest.raises(error):
        await run()
    assert (args.output / "test_attempt.json").exists()
    assert not (args.output / "test_results.json").exists()
    with pytest.raises(ValueError, match="sealed"):
        prepare_run(args, splits)
    with pytest.raises(ValueError, match="sealed"):
        await run()
    assert state.sampled == 1


@pytest.mark.asyncio
async def test_replayed_validation_replaces_stale_selection(native_run):
    state, args, _, run = native_run
    state.mode = "replay"
    await run()
    report = json.loads((args.output / "test_results.json").read_text())
    evaluations = json.loads((args.output / "validation_results.json").read_text())
    assert report["selected_validation_step"] == 2
    assert len(evaluations) == 3
    for evaluation in evaluations:
        outcome = evaluation["outcomes"][0]
        assert outcome["environment_trace"]["terminal"]["reward"] == outcome["reward"]
    assert report["outcomes"][0]["environment_trace"]["case_id"] == "test-A"
    assert not any(
        item["checkpoint"]["path"] == "river://stale" for item in evaluations
    )


@pytest.mark.asyncio
async def test_test_rollout_with_wrong_reset_input_stays_sealed(native_run):
    state, args, _, run = native_run
    state.mode = "test_wrong_input"
    with pytest.raises(ValueError, match="reset input"):
        await run()
    assert (args.output / "test_attempt.json").exists()
    assert not (args.output / "test_results.json").exists()
    with pytest.raises(ValueError, match="sealed"):
        await run()
    assert state.sampled == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["test_wrong_metrics", "test_wrong_action"])
async def test_replay_mismatch_preserves_rejected_evidence_without_publishing(
    native_run, mode
):
    state, args, _, run = native_run
    state.mode = mode
    with pytest.raises(ValueError, match="Test trace replay failed"):
        await run()
    assert not (args.output / "test_results.json").exists()
    failure = json.loads((args.output / "test_replay_failure.json").read_text())
    assert failure["status"] == "rejected"
    assert not failure["replay_audit"]["passed"]
    assert failure["replay_audit"]["report_hash"] == content_hash(
        failure["candidate_report"]
    )
    attempt = json.loads((args.output / "test_attempt.json").read_text())
    assert failure["candidate_report"]["test_attempt_hash"] == content_hash(attempt)
    with pytest.raises(ValueError, match="sealed"):
        await run()
    assert state.sampled == 1


@pytest.mark.asyncio
async def test_fully_committed_resume_can_finish_without_invented_metrics(native_run):
    state, args, _, run = native_run
    state.mode = "recovered_complete"
    await run()
    progress = json.loads((args.output / "training_metrics.json").read_text())
    assert [entry["step"] for entry in progress] == [1, 2]
    assert all(entry["metrics"] is None for entry in progress)
    assert all(entry["source"] == "river_recovery" for entry in progress)
    assert (args.output / "recovery_receipts.json").exists()
    assert state.sampled == 1


@pytest.mark.asyncio
async def test_recovery_discards_validation_and_metric_suffix(native_run):
    state, args, _, run = native_run
    state.mode = "recovered_partial"
    (args.output / "training_metrics.json").write_text(
        json.dumps(
            [
                {"step": 1, "metrics": {"loss": 0.7}},
                {"step": 2, "metrics": {"loss": 99}},
            ]
        )
    )
    stale = evaluation(2, 1)
    stale["checkpoint"]["path"] = "river://uncommitted"
    (args.output / "validation_results.json").write_text(json.dumps([stale]))
    await run()
    report = json.loads((args.output / "test_results.json").read_text())
    assert report["selected_checkpoint"]["path"] == "river://step-2"
    progress = json.loads((args.output / "training_metrics.json").read_text())
    assert progress == [
        {"step": 1, "metrics": {"loss": 0.7}},
        {"step": 2, "metrics": {"loss": 0.1}},
    ]


@pytest.mark.asyncio
async def test_failed_seal_write_prevents_test_engine(native_run, monkeypatch):
    state, args, _, run = native_run
    from stateset_agents.training import river_refund

    original = river_refund.atomic_json

    def write(path, value):
        if path.name == "test_attempt.json":
            raise OSError("disk full")
        original(path, value)

    monkeypatch.setattr(river_refund, "atomic_json", write)
    with pytest.raises(OSError, match="disk full"):
        await run()
    assert state.test_engines == state.sampled == 0
    assert not (args.output / "test_attempt.json").exists()


def test_hard_process_exit_cannot_reopen_test(tmp_path):
    code = textwrap.dedent("""
        import asyncio, os, sys
        from dataclasses import dataclass
        from pathlib import Path
        from types import SimpleNamespace
        from stateset_agents.training.river_refund import campaign, prepare_run
        @dataclass
        class Checkpoint:
            path: str
        class Engine:
            def __init__(self, *args, **kwargs):
                assert (Path(sys.argv[1]) / "test_attempt.json").exists()
                os._exit(73)
        rl = SimpleNamespace(Env=object, Budget=lambda **kw: kw,
            Schedule=lambda **kw: kw, RolloutEngine=Engine,
            CheckpointSampler=lambda *a, **kw: SimpleNamespace(**kw))
        sys.modules["river_client"] = SimpleNamespace(rl=rl, Checkpoint=Checkpoint)
        args = SimpleNamespace(output=Path(sys.argv[1]), base_model="base", seed=42,
            steps=2, concurrency=8, max_staleness=0, learning_rate=1e-5,
            checkpoint=None, evaluate_only=True, dry_run=False)
        splits = {"test": [{"order_id": "test-A", "amount_cents": 100, "eligible": True}]}
        prepare_run(args, splits)
        model = SimpleNamespace(save_weights=lambda *a, **kw: Checkpoint("river://base"))
        asyncio.run(campaign(model, object(), SimpleNamespace(tokenizer=object()), args, splits))
    """)
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 73, result.stderr
    assert (tmp_path / "test_attempt.json").exists()
    assert not (tmp_path / "test_results.json").exists()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    args = SimpleNamespace(**manifest, output=tmp_path, dry_run=False)
    with pytest.raises(ValueError, match="sealed"):
        prepare_run(
            args,
            {"test": [{"order_id": "test-A", "amount_cents": 100, "eligible": True}]},
        )
